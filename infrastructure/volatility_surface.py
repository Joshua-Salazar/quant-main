from datetime import datetime, timedelta, time
import glob
import pandas as pd

from ..analytics import constants
from ..analytics.utils import float_equal
from ..analytics.symbology import option_root_from_ticker
from ..data.market import extract_expiration_date_isoformat, get_vola_surface, get_intraday_vola_surface
from ..dates.utils import datetime_diff, datetime_equal, datetime_to_tenor, datetime_to_vola_datetime, \
    vola_datetime_to_datetime, coerce_timezone, count_business_days
from ..constants.strike_type import StrikeType
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..infrastructure.shock import DatetimeShiftType
from ..interface.ishock import IShock, ShockType
from ..interface.market_item import MarketItem
from ..interface.market_items.ifutureprice import IFuturePrice
from ..interface.market_items.ispot import ISpot
from ..interface.market_items.ivolatilitysurface import IVolatilitySurface
from ..infrastructure.option_data_container import CassandraDSPreLoadFromPickle, OptionDataRequest
from ..data.datalake_cassandra import DatalakeCassandra, ExpiryFilterByDateOffset
from ..analytics.utils import find_bracket_bounds, interpolate_curve

import math
import numpy as np
import os
import bisect
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola


def vix_v2x_imagine_bump(base_date, expiration, spot, CrashB):
    term = datetime_diff(expiration, base_date).days
    return max(11.0, min(CrashB[3], spot + CrashB[2] * math.sqrt(CrashB[0] / max(term, CrashB[1]))))


def vix_v2x_sc2024_bump(base_date, expiration, spot, sc_params):
    tenor_days = datetime_diff(expiration, base_date).days

    available_tenors = list(sorted(list(sc_params['vix_linear'].keys())))
    lb, ub = find_bracket_bounds(available_tenors, tenor_days)
    shift_curve = {}
    if lb != -float('inf'):
        shift_curve[lb] = sc_params['vix_linear'][lb][0] * sc_params['crash_size'] + sc_params['vix_linear'][lb][1]
    if ub != float('inf'):
        shift_curve[ub] = sc_params['vix_linear'][ub][0] * sc_params['crash_size'] + sc_params['vix_linear'][ub][1]
    shift_full = interpolate_curve(shift_curve, tenor_days, flat_extrapolate_lower=True, flat_extrapolate_upper=True)

    realized_vix_change = interpolate_curve(sc_params['realized_vix_change'], tenor_days, flat_extrapolate_lower=True, flat_extrapolate_upper=True)

    remaining_shift = shift_full - realized_vix_change
    pivot = 10
    a = 3.678794411714422
    b = 0.1
    if remaining_shift >= pivot:
        shift = remaining_shift
    else:
        shift = a * np.exp(b * remaining_shift)

    return spot + shift


def imagine_bump(A, tenor_days):
    shift = (A[3] + A[0] / (math.exp(A[4] * tenor_days / constants.YEAR_CALENDAR_DAY_COUNT)
                            * math.sqrt(A[2] + A[1] * tenor_days / constants.YEAR_CALENDAR_DAY_COUNT))) / 100
    return shift


def sc2024_bump(sc_params, tenor_days):
    available_tenors = list(sorted(list(sc_params['linear'].keys())))
    lb, ub = find_bracket_bounds(available_tenors, tenor_days)
    shift_curve = {}
    if lb != -float('inf'):
        shift_curve[lb] = sc_params['linear'][lb][0] * sc_params['crash_size'] + sc_params['linear'][lb][1]
    if ub != float('inf'):
        shift_curve[ub] = sc_params['linear'][ub][0] * sc_params['crash_size'] + sc_params['linear'][ub][1]
    shift_full = interpolate_curve(shift_curve, tenor_days, flat_extrapolate_lower=True, flat_extrapolate_upper=True)

    realized_vol_change = interpolate_curve(sc_params['realized_vol_change'], tenor_days, flat_extrapolate_lower=True, flat_extrapolate_upper=True)

    remaining_shift = shift_full - realized_vol_change * 100
    pivot = 10
    a = 3.678794411714422
    b = 0.1
    if remaining_shift >= pivot:
        shift = remaining_shift / 100
    else:
        shift = a * np.exp(b * remaining_shift) / 100

    return shift


class VolatilitySurface(MarketItem, ISpot, IFuturePrice, IVolatilitySurface):

    def __init__(self, underlying_type: UnderlyingType, vola_surface: vola.VolSurface if ENABLE_PYVOLAR else None, underlying: str,
                 overrided_base_datetime: datetime = None,
                 future_prices: dict = {}, underlying_symbols=[]):
        self.underlying_type = underlying_type
        self.vola_surface = vola_surface
        self.underlying = underlying
        self.market_key = market_utils.create_vol_surface_key(underlying)
        self.base_datetime = vola_datetime_to_datetime(vola_surface.asOfTime)
        self.overrided_base_datetime = overrided_base_datetime
        self.sticky_date = False
        self.future_prices = {}
        self.setup_future_prices(self.get_base_datetime(), future_prices)

        self.underlying_symbols = underlying_symbols

        self.use_cache = True
        self.cached_discount_curve = {}
        self.cached_forward_discount_curve = {}
        self.cached_borrow_curve = {}
        self.cached_forward_borrow_curve = {}
        self.cached_vol_surface = {}
        self.cached_vola_datetime = {}

    def get_vol_surface(self, base_datetime):
        assert base_datetime.date() == self.get_base_datetime().date()
        return self

    def get_market_key(self):
        return self.market_key

    def setup_future_prices(self, base_date: datetime, future_prices: dict):
        if self.underlying_type == UnderlyingType.EQUITY:
            if len(future_prices) > 1 \
                    or (len(future_prices) == 1 and (self.get_base_datetime() not in future_prices) and (vola_datetime_to_datetime(self.vola_surface.asOfTime) not in future_prices)):
                raise Exception("not support future price for equity")
            base_date_used = self.get_base_datetime() if self.get_base_datetime() in future_prices else vola_datetime_to_datetime(self.vola_surface.asOfTime)
            self.future_prices[base_date_used] = self.get_spot(base_date)
        else:
            self.future_prices.update(future_prices)

    @classmethod
    def load(cls, underlying_id, ref_time, underlying_type: UnderlyingType, underlying_full_name: str,
             future_prices: dict = {}, load_shared_file: bool = False, load_file: bool = False, save_file: bool = False,
             file_dir: str = "", file_name: str = "", universe=None, exclude_vixw=True, reset_close_spot=False,
             override_shared_folder=None):

        underlying_shorthand = underlying_full_name.split(" ")[0]
        if load_shared_file:
            if load_file:
                raise Exception("Cannot load surface from both shared and local file")
            if save_file:
                raise Exception("Cannot save surface when load from shared file")
            data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
            vola_file_dir = f"{data_file_folder}Solutions/VOLA_Surfaces/" if override_shared_folder is None else override_shared_folder
            file_path_pattern = os.path.join(vola_file_dir, underlying_full_name,
                                             f"{underlying_full_name}_{ref_time.split('|')[-1] }"
                                             f"-[0-9][0-9][0-9][0-9][0-9][0-9]-EDT_vs-eq.yml")
            vol_surface_files_list = glob.glob(file_path_pattern)

            if len(vol_surface_files_list) == 1:
                file_path = vol_surface_files_list[0]
            elif len(vol_surface_files_list) == 0:
                raise Exception(f"No Vola surface exists for {ref_time} | {underlying_full_name}")
            else:
                raise Exception(f"Found multiple Vola surface ({len(vol_surface_files_list)}) for {ref_time} | {underlying_full_name}")
        else:
            if file_name is None:
                file_name = f"{underlying_shorthand}.txt"
            file_path = os.path.join(file_dir, file_name)

        if load_file or load_shared_file:
            with open(file_path, "r") as f:
                vola_surface_str = f.read()

                fa = vola.makeFactoryAnalytics()
                if underlying_type == UnderlyingType.EQUITY:
                    vola_surface = fa.makeVolSurfaceEquity(serializedVolSurface=vola_surface_str,
                                                           format=vola.FormatIO.DEFAULT)
                else:
                    assert underlying_type == UnderlyingType.FUTURES
                    vola_surface = fa.makeVolSurfaceFutures(serializedVolSurface=vola_surface_str,
                                                            format=vola.FormatIO.DEFAULT)
        else:
            if isinstance(ref_time, str):
                vola_surface = get_vola_surface(underlying_id, ref_time, underlying_type.value)
            else:
                vola_surface = get_intraday_vola_surface(underlying_id, ref_time, underlying_type.value)

        if vola_surface is None:
            return None

        non_us_unds = ["SX5E Index", "SX7E Index", "SMI Index", "UKX Index", "DAX Index", "CAC Index", "AS51 Index",
                       "HSI Index", "KOSPI2 Index", "NKY Index", "FTSEMIB Index"]
        if reset_close_spot and ref_time != "LIVE" and underlying_full_name in non_us_unds:
            cassandra = DatalakeCassandra()
            base_datetime = vola_datetime_to_datetime(vola_surface.asOfTime)
            data_df = cassandra.get_stock_data(option_root_from_ticker(underlying_full_name), base_datetime, base_datetime)
            close_spot = list(set(data_df["price_close_opt"].values))
            if len(close_spot) > 1:
                raise Exception(f"Found multiple spot {close_spot}")
            vola_surface.adjustToNewSpot(close_spot[0])

        if save_file:
            if load_file:
                raise Exception("Only save file if not load file")
            folder = os.path.split(file_path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            with open(file_path, "w") as f:
                f.write(vola_surface.toString())

        underlying_symbols = []
        # tenor_days set 2 months approximately due to fact that up to 6 consecutive weekly vix option may be listed
        tenor_days = 60
        if underlying_type == UnderlyingType.FUTURES and universe is not None and "VIX" in underlying_full_name:
            # request option data
            calendar = "CBOE"
            base_datetime = vola_datetime_to_datetime(vola_surface.asOfTime)
            # for backtest, we already pass in ivol universe, for fallback method, we query
            # table vol.optionvalue_1545 instead of default table ivol.optionvalue
            is_input_ivol_universe = "expiration_rule" in universe.columns
            option_data_request = OptionDataRequest(
                base_datetime, base_datetime, calendar=calendar, root="VIX",
                expiry_filter=ExpiryFilterByDateOffset(tenor_days), frequency="daily",
                use_1545table=is_input_ivol_universe)

            if isinstance(ref_time, str) and ref_time.lower() == 'live':
                ivol_universe = pd.DataFrame()
            else:
                option_data_container = CassandraDSPreLoadFromPickle(force_reload=True, update_cache=False).initialize(option_data_request)
                ivol_universe = option_data_container.get_option_universe(datetime.combine(base_datetime.date(), datetime.min.time()))
            err_msg = []
            for idx, expiryTime in enumerate(vola_surface.expiryTimes):
                expiration_datetime = vola_datetime_to_datetime(expiryTime)
                if expiration_datetime.date() <= base_datetime.date():
                    roots = ["VIX"]
                else:
                    is_input_ivol_universe = "expiration_rule" in universe.columns
                    expiration_datetime_str = expiration_datetime.isoformat()
                    expiration_date_str = datetime(expiration_datetime.year, expiration_datetime.month,
                                                   expiration_datetime.day).isoformat()
                    input_universe_expiration = universe[universe.expiration == expiration_date_str] \
                        if is_input_ivol_universe else universe[universe.expiration == expiration_datetime_str]
                    if input_universe_expiration.empty:
                        if ivol_universe.empty:
                            raise Exception("empty ivol universe")
                        ivol_universe_expiration = ivol_universe[ivol_universe.expiration_date == expiration_date_str]
                        roots = ["VIXW" if "week" in x.lower() else "VIX" for x in
                                 ivol_universe_expiration["expiration_rule"].unique()]
                    else:
                        roots = ["VIXW" if "week" in x.lower() else "VIX" for x in
                                 input_universe_expiration["expiration_rule"].unique()] if is_input_ivol_universe \
                            else input_universe_expiration["symbol"].str.split(" ", expand=True)[0].unique()

                if len(roots) != 1:
                    msg = f"Missing surface expiry datetime {expiration_datetime_str} from option chain on " \
                          f"{base_datetime.strftime('%Y-%m-%d')} for {underlying_full_name}"
                    if idx != len(vola_surface.expiryTimes)-1:
                        msg += f" next expiry {vola_datetime_to_datetime(vola_surface.expiryTimes[idx+1]).strftime('%Y-%m-%d')}"
                    else:
                        msg += " max expiry date."
                    err_msg.append(msg)
                else:
                    underlying_symbols.append(roots[0])

            if len(err_msg) > 0:
                raise Exception(",".join(err_msg))

            assert len(underlying_symbols) == len(vola_surface.expiryTimes)
            if exclude_vixw:
                und_symbols_updated = []
                for symbol, expiry in zip(underlying_symbols, vola_surface.expiryTimes):
                    if symbol == "VIXW":
                        vola_surface.deleteVolSlice(expiry)
                    else:
                        und_symbols_updated.append(symbol)
                underlying_symbols = und_symbols_updated

        return cls(underlying_type, vola_surface, underlying_full_name, future_prices=future_prices,
                   underlying_symbols=underlying_symbols)

    @classmethod
    def create_from_vola_surface(cls, vola_surface: vola.VolSurface if ENABLE_PYVOLAR else None, underlying: str):

        if isinstance(vola_surface, vola.VolSurfaceEquity):
            return cls(UnderlyingType.EQUITY, vola_surface, underlying)
        elif isinstance(vola_surface, vola.VolSurfaceFutures):
            return cls(UnderlyingType.FUTURES, vola_surface, underlying)
        else:
            raise Exception("unsupported vola surface")

    def clone(self):
        return VolatilitySurface(self.underlying_type, self.vola_surface.clone(),  self.underlying,
                                 self.overrided_base_datetime, self.future_prices, self.underlying_symbols)

    def get_base_datetime(self) -> datetime:
        return self.overrided_base_datetime if self.overrided_base_datetime else self.base_datetime

    def get_underlying_type(self) -> UnderlyingType:
        return self.underlying_type

    def get_vola_datetime(self, dt: datetime):
        if self.use_cache and dt in self.cached_vola_datetime:
            return self.cached_vola_datetime[dt]
        else:
            vola_datetime = datetime_to_vola_datetime(dt)
            if self.use_cache:
                self.cached_vola_datetime[dt] = vola_datetime
            return vola_datetime

    def get_discount_rate(self, expiry_dt: datetime) -> float:
        key = (self.base_datetime, expiry_dt)
        if self.use_cache and key in self.cached_discount_curve:
            rate = self.cached_discount_curve[key]
        else:
            expiry_vola_dt = self.get_vola_datetime(expiry_dt)
            rate = self.vola_surface.discountCurve.rate(expiry_vola_dt)
            if self.use_cache:
                self.cached_discount_curve[key] = rate
        return rate

    def get_forward_discount_rate(self, st_dt: datetime, ed_dt: datetime) -> float:
        key = (st_dt, ed_dt)
        if self.use_cache and key in self.cached_forward_discount_curve:
            rate = self.cached_forward_discount_curve[key]
        else:
            st_vola_dt = self.get_vola_datetime(st_dt)
            ed_vola_dt = self.get_vola_datetime(ed_dt)
            rate = self.vola_surface.discountCurve.rate(st_vola_dt, ed_vola_dt)
            if self.use_cache:
                self.cached_forward_discount_curve[key] = rate
        return rate

    def get_borrow_rate(self, expiry_dt: datetime) -> float:
        if self.underlying_type == UnderlyingType.FUTURES:
            raise Exception("Unsupported borrow rate for Futures Volatility Surface")

        key = (self.base_datetime, expiry_dt)
        if self.use_cache and key in self.cached_borrow_curve:
            rate = self.cached_borrow_curve[key]
        else:
            expiry_vola_dt = self.get_vola_datetime(expiry_dt)
            rate = self.vola_surface.borrowCurve.rate(expiry_vola_dt)
            if self.use_cache:
                self.cached_borrow_curve[key] = rate
        return rate

    def get_forward_borrow_rate(self, st_dt: datetime, ed_dt: datetime) -> float:
        if self.underlying_type == UnderlyingType.FUTURES:
            raise Exception("Unsupported borrow rate for Futures Volatility Surface")
        key = (st_dt, ed_dt)
        if self.use_cache and key in self.cached_forward_borrow_curve:
            rate = self.cached_forward_borrow_curve[key]
        else:
            st_vola_dt = self.get_vola_datetime(st_dt)
            ed_vola_dt = self.get_vola_datetime(ed_dt)
            rate = self.vola_surface.borrowCurve.rate(st_vola_dt, ed_vola_dt)
            if self.use_cache:
                self.cached_forward_borrow_curve[key] = rate
        return rate

    def get_spot(self, base_date: datetime) -> float:

        vol_surface_base_date = self.get_base_datetime()
        if not datetime_equal(*coerce_timezone(base_date, vol_surface_base_date), und=self.underlying):
            raise Exception(f"Found inconsistent base date in vol surface {self.underlying}: "
                            f"{vol_surface_base_date.isoformat()} and target date: {base_date.isoformat()}")

        assert self.underlying_type == UnderlyingType.EQUITY
        return self.vola_surface.spot

    def get_future_price(self, base_date: datetime, expiry_dt: datetime) -> float:
        """
        Get future price from vol surface. Different from get forward, there is NO Interpolation used here.
        It could be either Equity Vol Surface or Future Vol Surface:
        a user case is hedging instrument for equity option is a Future instrument.
        """
        vol_surface_base_date = self.get_base_datetime()
        if not datetime_equal(*coerce_timezone(base_date, vol_surface_base_date)):
            msg = f"Found inconsistent base date in vol surface {self.underlying}: "\
                  f"{vol_surface_base_date.isoformat()} and target date: {base_date.isoformat()}"
            if self.underlying == "HSCEI Index":
                print(msg)
            else:
                raise Exception(msg)

        assert self.underlying_type == UnderlyingType.FUTURES or self.underlying_type == UnderlyingType.EQUITY
        future_date_prices = {dt.date(): px for dt, px in self.future_prices.items()}
        if expiry_dt in self.future_prices:
            return self.future_prices[expiry_dt]
        elif expiry_dt.date() in future_date_prices:
            return future_date_prices[expiry_dt.date()]
        else:
            return self.get_undelrying_price_from_vola_surface(base_date, expiry_dt)

    def get_undelrying_price_from_vola_surface(self, base_date: datetime, expiration: datetime, return_on_the_run_future_if_missing=False):
        if self.underlying_type == UnderlyingType.EQUITY:
            return self.get_spot(base_date)
        elif self.underlying_type == UnderlyingType.FUTURES:
            if expiration:
                # TODO: for future options we now only support option expiries that are on the vol surface
                expiry_indices = {vola_datetime_to_datetime(expiry_time).date(): i for i, expiry_time in
                                  enumerate(self.vola_surface.expiryTimes)}
                if expiration.date() in expiry_indices:
                    slice_index = expiry_indices[expiration.date()]
                elif return_on_the_run_future_if_missing:
                    slice_index = 0
                else:
                    # add fallback for HSCEI where we cannot find hedge future expiry in Future surface
                    # but exists in Equity surface. It happens in instantaneous crash calculation for existing pfo
                    # so that we relax it to return interpolated price
                    if self.underlying == "HSCEI Index":
                        return self.get_forward(expiration)
                    elif self.underlying in ["CRUDE", "SOYBEAN"]:
                        nearest_expiration = self.find_nearest_expiration(expiration)
                        slice_index = expiry_indices[nearest_expiration.date()]
                    else:
                        raise Exception(f"missing date {expiration.date()} in vol surface {self.underlying} on "
                                        f"{self.base_datetime.date()}")
            else:
                slice_index = 0
            future_price = self.vola_surface.underlierPrices[slice_index]
            return future_price
        else:
            raise RuntimeError('Unknown vola surface type')

    def get_vol(self, expiry_dt: datetime, strike: float, strike_type: StrikeType = StrikeType.K) -> float:
        # adjust expiry if override base date
        if self.overrided_base_datetime and not self.sticky_date:
            days = timedelta(days=(expiry_dt - self.overrided_base_datetime.replace(tzinfo=expiry_dt.tzinfo)).days)
            expiry_dt = self.base_datetime + days

        key = (expiry_dt, f"{strike:.8f}", strike_type)
        if self.use_cache and key in self.cached_vol_surface:
            vol = self.cached_vol_surface[key]
        else:
            expiry_vola_dt = self.get_vola_datetime(expiry_dt)
            vol = self.vola_surface.volAtT(expiry_vola_dt, strike, strike_type.value)
            if self.use_cache:
                self.cached_vol_surface[key] = vol
        return vol

    def find_nearest_expiration(self, expiration: datetime) -> datetime:
        """
        find nearest expiration date by strip off time infomation
        :param expiration:
        :return:
        """
        expiration_dates = list(
                map(lambda x: extract_expiration_date_isoformat(vola_datetime_to_datetime(x).isoformat()),
                    self.vola_surface.expiryTimes))
        expiration_date = datetime.fromisoformat(extract_expiration_date_isoformat(expiration.isoformat()))
        nearest_expiration_date = min(expiration_dates,
                                      key=lambda x: abs(datetime_diff(datetime.fromisoformat(x), expiration_date)))
        return datetime.fromisoformat(nearest_expiration_date)

    def find_nearest_am_expiration(self, expiration: datetime) -> datetime:
        """
        find nearest am expiration date after strip off time infomation
        :param expiration:
        :return:
        """
        expiration_datetimes = list(map(lambda x: vola_datetime_to_datetime(x), self.vola_surface.expiryTimes))
        expiration_datetimes = [exp_dt for exp_dt in expiration_datetimes if exp_dt.time() < time(12, 0, 0)]
        nearest_expiration_datetime = min(expiration_datetimes, key=lambda x: abs(datetime_diff(x, expiration)))
        return nearest_expiration_datetime.replace(tzinfo=None)

    def get_forward(self, expiry_dt: datetime) -> float:
        """
        return forward with special handle index spot
        """
        first_expiry = vola_datetime_to_datetime(self.vola_surface.expiryTimes[0])
        base_date = vola_datetime_to_datetime(self.vola_surface.asOfTime)
        if first_expiry < base_date:
            missing_spot_date = True
            first_valid_expiry = vola_datetime_to_datetime(self.vola_surface.expiryTimes[1])
            first_valid_fwd = self.vola_surface.forwards[1]
        elif first_expiry == base_date:
            missing_spot_date = False
            first_valid_expiry = vola_datetime_to_datetime(self.vola_surface.expiryTimes[0])
            first_valid_fwd = self.vola_surface.forwards[0]
        else:
            missing_spot_date = True
            first_valid_expiry = vola_datetime_to_datetime(self.vola_surface.expiryTimes[0])
            first_valid_fwd = self.vola_surface.forwards[0]

        # check first expiry date is before base date
        if missing_spot_date:
            index_date = None
            index_price = None
            ratio = None
            for date, price in self.future_prices.items():
                if date.date() == base_date.date():
                    index_date = date
                    index_price = price
                    ratio = (first_valid_fwd - index_price) / datetime_to_tenor(first_valid_expiry, index_date)
                    break
            # we might applied VOLA time shift so simulated spot not added. relax the check
            if index_date is None and expiry_dt <= first_valid_expiry:
                raise Exception(f"Missing index spot for {self.vola_surface.symbol}")

        # build fwd curve for VIX where we exclude weekly options
        if len(self.underlying_symbols) > 0:
            # build variance interpolation base on varinace(vol_yrs) = vol_fwd^2 * vol_yrs = cal_fwd^2 * cal_yrs
            # then cal_fwd = sqrt(variance / cal_yrs)
            vol_yrs = []
            variance = []
            min_yrs = 0.001
            if missing_spot_date:
                cal_yrs = max(min_yrs, self.vola_surface.timeConverterC.timeInYears(self.vola_surface.asOfTime, datetime_to_vola_datetime(index_date)))
                vol_yrs.append(
                    max(min_yrs, self.vola_surface.timeConverterV.timeInYears(self.vola_surface.asOfTime, datetime_to_vola_datetime(index_date))))
                variance.append(index_price**2 * cal_yrs)

            for idx, expiry_date in enumerate(self.vola_surface.expiryTimes):
                if self.underlying_symbols[idx] == "VIX":
                    this_expiry = vola_datetime_to_datetime(expiry_date)
                    if this_expiry >= first_valid_expiry:
                        cal_yrs = max(min_yrs, self.vola_surface.timeConverterC.timeInYears(self.vola_surface.asOfTime, expiry_date))
                        vol_yrs.append(
                            max(min_yrs, self.vola_surface.timeConverterV.timeInYears(self.vola_surface.asOfTime, expiry_date)))
                        variance.append(self.vola_surface.forwards[idx]**2 * cal_yrs)

            target_vol_yrs = max(min_yrs, self.vola_surface.timeConverterV.timeInYears(self.vola_surface.asOfTime, datetime_to_vola_datetime(expiry_dt)))

            target_variance = np.interp(target_vol_yrs, vol_yrs, variance)
            target_cal_yrs = max(min_yrs, self.vola_surface.timeConverterC.timeInYears(self.vola_surface.asOfTime, datetime_to_vola_datetime(expiry_dt)))
            fwd = np.sqrt(target_variance / target_cal_yrs)
            return fwd
        else:
            expiry_dt, first_valid_expiry = coerce_timezone(expiry_dt, first_valid_expiry)
            if missing_spot_date and expiry_dt <= first_valid_expiry:
                # linear interpolation between base date and first expiry date
                # and flat extrapolation before base date
                fwd = index_price if expiry_dt <= index_date \
                    else index_price + ratio * datetime_to_tenor(expiry_dt, index_date)
            else:
                # no need check rolled date here
                # since if query date < base date, forwardAtT gives flat rate as spot value
                fwd = self.vola_surface.forwardAtT(datetime_to_vola_datetime(expiry_dt))

        return fwd

    @staticmethod
    def get_shifted_spot(shocks, vol_surface):
        need_spot_shift = False
        new_spot = None
        sticky_strike = False
        pvol = None
        for shock in shocks:
            if shock and shock.type == ShockType.SPOTSHOCK and vol_surface.underlying_type == UnderlyingType.EQUITY:
                need_spot_shift = True
                original_spot = vol_surface.vola_surface.spot
                spot_beta = 1. if vol_surface.underlying == shock.benchmark_underlying or shock.spot_beta is None else shock.spot_beta
                sticky_strike = shock.sticky_strike
                pvol = shock.pvol
                vola_surface = vola.VolUtils.convertToStickyStrike(
                    vol_surface.vola_surface) if sticky_strike else vol_surface.vola_surface
                if shock.method == 'percentage':
                    new_spot = max(sum(vola_surface.dividendData.divCashs),
                                   original_spot * (1 + shock.size * spot_beta))
                    if not float_equal(new_spot, original_spot * (1 + shock.size * spot_beta)):
                        print(
                            f'Bumped spot floored at sum of dividends {sum(vola_surface.dividendData.divCashs)}')
                    minimum_spot = 1e-4
                    if new_spot < minimum_spot:
                        print(f'New spot ({new_spot}) after bump ({shock.size * spot_beta}) floored at 1bps')
                        new_spot = minimum_spot
                elif shock.method == 'level':
                    new_spot = shock.size * spot_beta
                else:
                    raise RuntimeError('Unknown spot shock method ' + shock.method)
        return need_spot_shift, new_spot, sticky_strike, pvol

    # helping functions to get special spot/forward/vol from vola
    @staticmethod
    def get_spot_benchmark(vol_surface):
        # avoid circular import
        from ..infrastructure.fx_sabr_vol_data_container import FXSABRVolSurface
        is_vola = isinstance(vol_surface, VolatilitySurface)
        is_sabr = isinstance(vol_surface, FXSABRVolSurface)
        if not is_vola and not is_sabr:
            raise Exception("Unexpected vol surface")
        if is_vola:
            spot = vol_surface.vola_surface.spot
        else:
            spot = vol_surface.get_spot(vol_surface.base_date)
        return spot

    @staticmethod
    def get_forward_benchmark(expiry, vol_surface):
        # avoid circular import
        from ..infrastructure.fx_sabr_vol_data_container import FXSABRVolSurface
        is_vola = isinstance(vol_surface, VolatilitySurface)
        is_sabr = isinstance(vol_surface, FXSABRVolSurface)
        if not is_vola and not is_sabr:
            raise Exception("Unexpected vol surface")
        if is_vola:
            fwd = vol_surface.vola_surface.forwardAtT(expiry)
        else:
            fwd = vol_surface.get_forward(vola_datetime_to_datetime(expiry))
        return fwd

    @staticmethod
    def get_vol_benchmark(expiry, strike, vol_surface):
        # avoid circular import
        from ..infrastructure.fx_sabr_vol_data_container import FXSABRVolSurface
        is_vola = isinstance(vol_surface, VolatilitySurface)
        is_sabr = isinstance(vol_surface, FXSABRVolSurface)
        if not is_vola and not is_sabr:
            raise Exception("Unexpected vol surface")
        if is_vola:
            vol = vol_surface.vola_surface.volAtT(expiry, strike, StrikeType.K.value)
        else:
            vol = vol_surface.get_vol(vola_datetime_to_datetime(expiry), strike, StrikeType.K)
        return vol

    @staticmethod
    def get_shifted_forwards(shocks, vol_surface, as_of_date):
        need_forwards_shift = False
        new_forwards = []
        sticky_strike = False
        pvol = None
        for shock in shocks:
            if shock and shock.type == ShockType.SPOTSHOCK and vol_surface.underlying_type == UnderlyingType.FUTURES:
                need_forwards_shift = True
                spot_beta = 1. if shock.spot_beta is None else shock.spot_beta
                sticky_strike = shock.sticky_strike
                pvol = shock.pvol
                vola_surface = vola.VolUtils.convertToStickyStrike(
                    vol_surface.vola_surface) if sticky_strike else vola_surface.vola_surface
                if shock.method == 'percentage':
                    for expiry, original_forward in zip(vola_surface.expiryTimes, vola_surface.forwards):
                        if expiry >= as_of_date:
                            new_forward = original_forward * (1 + shock.size * spot_beta)
                            minimum_forward = 1e-4
                            if new_forward < minimum_forward:
                                print(f'New forward ({new_forward}) after bump ({shock.size * spot_beta}) floored at 1bps')
                                new_forward = minimum_forward
                            new_forwards.append(new_forward)
                        else:
                            new_forwards.append(original_forward)
                elif shock.method == 'level':
                    for expiry, original_forward in zip(vola_surface.expiryTimes, vola_surface.forwards):
                        if expiry >= as_of_date:
                            new_forwards.append(shock.size * spot_beta)
                        else:
                            new_forwards.append(original_forward)
                else:
                    raise RuntimeError('Unknown spot shock method ' + shock.method)
        return need_forwards_shift, new_forwards, sticky_strike, pvol

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        # shock apply in the following order
        # 1. time shift
        # 2. spot shock if shift vol before underlying else vol shock
        # 3. vol shock if shift vol before underlying else spot shock

        new_vol_surface = self.clone()
        # time shift
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                if shock.shift_type == DatetimeShiftType.STICKY_TENOR:
                    new_vol_surface.overrided_base_datetime = shock.shifted_datetime
                    if shock.roll_future_price:
                        if new_vol_surface.underlying_type == UnderlyingType.FUTURES:
                            for date, price in new_vol_surface.future_prices.items():
                                if shock.shift_days is None:
                                    shift_days = datetime_diff(shock.shifted_datetime, original_market.get_base_datetime()).days
                                else:
                                    shift_days = shock.shift_days
                                rolled_date = date - timedelta(days=shift_days)
                                new_vol_surface.future_prices[date] = new_vol_surface.get_forward(rolled_date)
                    new_vol_surface.sticky_date = False
                    continue
                elif shock.shift_type == DatetimeShiftType.STICKY_DATE:
                    new_vol_surface.overrided_base_datetime = shock.shifted_datetime
                    new_vol_surface.sticky_date = True
                    # do nothing
                    continue
                    # new_vol_surface.overrided_base_datetime = shock.shifted_datetime
                    # new_vol_surface.vola_surface.shiftAsOfTime(datetime_to_vola_datetime(shock.shifted_datetime))
                    # as_of_date = shock.shifted_datetime
                    # raise Exception("stick date shift is not implement yet")
                elif shock.shift_type == DatetimeShiftType.VOLA_STICKY_TENOR:
                    new_vol_model = new_vol_surface.vola_surface.volModel
                    new_vol_model.timeShiftTypeVol0 = vola.TimeShiftTypeVol0.FIXED_BY_TENOR
                    new_vol_model.timeShiftTypeVolShape = vola.TimeShiftTypeVolShape.FIXED_BY_TENOR
                    new_vol_surface.vola_surface.updateVolModel(new_vol_model)
                    new_vol_surface.vola_surface.shiftAsOfTime(datetime_to_vola_datetime(shock.shifted_datetime))
                    new_vol_surface.overrided_base_datetime = shock.shifted_datetime
                    # update future prices in new surface
                    if new_vol_surface.underlying_type == UnderlyingType.EQUITY:
                        if len(new_vol_surface.future_prices) > 0:
                            assert len(new_vol_surface.future_prices) == 1
                            spot_date = list(new_vol_surface.future_prices.keys())[0]
                            new_vol_surface.future_prices[shock.shifted_datetime] = new_vol_surface.future_prices.pop(spot_date)
                    break
                elif shock.shift_type == DatetimeShiftType.VOLA_STICKY_DATE:
                    new_vol_model = new_vol_surface.vola_surface.volModel
                    new_vol_model.timeShiftTypeVol0 = vola.TimeShiftTypeVol0.FIXED_BY_DATE
                    new_vol_model.timeShiftTypeVolShape = vola.TimeShiftTypeVolShape.FIXED_BY_DATE
                    new_vol_surface.vola_surface.updateVolModel(new_vol_model)
                    new_vol_surface.vola_surface.shiftAsOfTime(datetime_to_vola_datetime(shock.shifted_datetime))
                    new_vol_surface.overrided_base_datetime = shock.shifted_datetime
                    # update future prices in new surface
                    if new_vol_surface.underlying_type == UnderlyingType.EQUITY:
                        if len(new_vol_surface.future_prices) > 0:
                            assert len(new_vol_surface.future_prices) == 1
                            spot_date = list(new_vol_surface.future_prices.keys())[0]
                            new_vol_surface.future_prices[shock.shifted_datetime] = new_vol_surface.future_prices.pop(spot_date)
                    break

        as_of_date = datetime_to_vola_datetime(new_vol_surface.get_base_datetime())

        def shift_vol_surface(_input_surface, _vol_shock, _as_of_date, _underlying,  original_market,
                              _need_underlying_spot_shift, _new_underlying_spot,
                              _need_underlying_forwards_shift, _new_underlying_forwards):

            if _vol_shock is not None:
                atm_vol_shifts = []
                base_dt = vola_datetime_to_datetime(as_of_date)
                # work out vol beta shift
                benchmark_underlying = _vol_shock.benchmark_underlying
                benchmark_vol_surface = None
                if _vol_shock.vol_beta and _underlying != benchmark_underlying:
                    key = market_utils.create_vol_surface_key(benchmark_underlying)
                    if original_market.has_item(key):
                        benchmark_vol_surface = original_market.get_item(key)
                    else:
                        key = market_utils.create_fx_vol_surface_key(benchmark_underlying)
                        benchmark_vol_surface = original_market.get_item(key)
                    # avoid circular import
                    from ..infrastructure.fx_sabr_vol_data_container import FXSABRVolSurface
                    is_vola_benchmark = isinstance(benchmark_vol_surface, VolatilitySurface)
                    is_sabr_benchmark = isinstance(benchmark_vol_surface, FXSABRVolSurface)

                if _vol_shock.method == "bucket":
                    original_vol = original_market.get_vol_surface(_underlying)

                for i, expiry in zip(range(len(_input_surface.expiryTimes)), _input_surface.expiryTimes):
                    if expiry >= _as_of_date:
                        if _vol_shock.method == 'imagine':
                            A = _vol_shock.parameters
                            tenor_days = (expiry.nanos - _as_of_date.nanos) / vola.DateTime.nanosPerDay
                            vol_shift = imagine_bump(A, tenor_days)
                        elif _vol_shock.method == 'imagine_interpolated':
                            A_all = _vol_shock.parameters['crash_A_all']
                            tenor_days = (expiry.nanos - _as_of_date.nanos) / vola.DateTime.nanosPerDay
                            available_crash_sizes = list(sorted(list(A_all.keys())))
                            lb, ub = find_bracket_bounds(available_crash_sizes, _vol_shock.parameters['crash_size'])
                            imagine_shock_curve = {}
                            if lb != -float('inf'):
                                imagine_shock_curve[lb] = imagine_bump(A_all[lb], tenor_days)
                            if ub != float('inf'):
                                imagine_shock_curve[ub] = imagine_bump(A_all[ub], tenor_days)
                            vol_shift = interpolate_curve(imagine_shock_curve, _vol_shock.parameters['crash_size'], flat_extrapolate_lower=True, flat_extrapolate_upper=True)
                        elif _vol_shock.method == 'sc2024':
                            sc_params = _vol_shock.parameters
                            tenor_days = (expiry.nanos - _as_of_date.nanos) / vola.DateTime.nanosPerDay
                            vol_shift = sc2024_bump(sc_params, tenor_days)
                        elif _vol_shock.method == 'time_weighted_level':
                            bds = 21
                            expiry_bds = count_business_days(base_dt, vola_datetime_to_datetime(expiry))
                            vol_shift = _vol_shock.parameters * bds / max(bds, expiry_bds)
                        elif _vol_shock.method == 'time_weighted_percentage':
                            curr_vol0 = _input_surface.vol0s[i]
                            bds = 21
                            expiry_bds = count_business_days(base_dt, vola_datetime_to_datetime(expiry))
                            vol_shift = curr_vol0 * _vol_shock.parameters * np.sqrt(bds / max(bds, expiry_bds))
                        elif _vol_shock.method == "bucket":
                            curr_vol0 = _input_surface.vol0s[i]
                            expiry_dt = vola_datetime_to_datetime(expiry)
                            expiry_dt, expiry_lb = coerce_timezone(expiry_dt, _vol_shock.parameters["expiry_lb"])
                            expiry_dt, expiry_ub = coerce_timezone(expiry_dt, _vol_shock.parameters["expiry_ub"])
                            if expiry_ub < expiry_lb:
                                raise Exception(f"lb {expiry_lb.strftime('%Y-%m-%d')} ub {expiry_ub.strftime('%Y-%m-%d')}")
                            elif expiry_lb == expiry_ub:
                                # remains the same vol
                                vol_shift = 0
                            elif expiry_dt <= expiry_lb:
                                # remains the same vol
                                vol_shift = 0
                            else:
                                expiry_min = min(expiry_dt, expiry_ub)
                                # forward starting vol between expiry_lb and expiry_min
                                vol_min = original_vol.get_vol(expiry_min, original_vol.get_forward(expiry_min))
                                vol_lb = original_vol.get_vol(expiry_lb, original_vol.get_forward(expiry_lb))
                                expiry_bds_min = count_business_days(base_dt, expiry_min)
                                expiry_bds_lb = count_business_days(base_dt, expiry_lb)
                                vol_0 = np.sqrt((vol_min**2 * expiry_bds_min - vol_lb**2 * expiry_bds_lb) /(expiry_bds_min - expiry_bds_lb))
                                # forward starting vol between expiry_min and expiry
                                vol_dt = original_vol.get_vol(expiry_dt, original_vol.get_forward(expiry_dt))
                                expiry_bds_dt = count_business_days(base_dt, expiry_dt)
                                if expiry_bds_dt == expiry_bds_min:
                                    vol_1 = 0
                                else:
                                    vol_1 = np.sqrt((vol_dt**2 * expiry_bds_dt - vol_min**2 * expiry_bds_min) /(expiry_bds_dt - expiry_bds_min))
                                var_dt = vol_lb**2 * expiry_bds_lb + (vol_0 * (1 + _vol_shock.parameters["shock_size"]))**2 * (expiry_bds_min - expiry_bds_lb) + vol_1 **2 * (expiry_bds_dt - expiry_bds_min)
                                vol_dt = np.sqrt(var_dt/expiry_bds_dt)
                                vol_shift = vol_dt - curr_vol0
                        elif _vol_shock.method == 'level':
                            vol_shift = _vol_shock.parameters
                        elif _vol_shock.method == 'percentage':
                            curr_vol0 = _input_surface.vol0s[i]
                            if benchmark_vol_surface is None:
                                vol_shift = curr_vol0 * _vol_shock.parameters
                            else:
                                if is_vola_benchmark or is_sabr_benchmark:
                                    benchmark_fwd = VolatilitySurface.get_forward_benchmark(expiry, benchmark_vol_surface)
                                    benchmark_atmvol = VolatilitySurface.get_vol_benchmark(expiry, benchmark_fwd, benchmark_vol_surface)
                                else:
                                    # flat bs vol so no need strike
                                    benchmark_atmvol = benchmark_vol_surface.get_vol(vola_datetime_to_datetime(expiry),
                                                                                     strike=0)
                                vol_shift = benchmark_atmvol * _vol_shock.parameters
                        else:
                            raise RuntimeError('Unknown vol shock method ' + _vol_shock.method)

                        if benchmark_vol_surface is not None:
                            if is_vola_benchmark or is_sabr_benchmark:
                                benchmark_fwd = VolatilitySurface.get_forward_benchmark(expiry, benchmark_vol_surface)
                                benchmark_spot = VolatilitySurface.get_spot_benchmark(benchmark_vol_surface)
                            underlying_vola_surface = original_market.get_item(
                                market_utils.create_vol_surface_key(_underlying)).vola_surface
                            underlying_fwd = underlying_vola_surface.forwardAtT(expiry)
                            if _need_underlying_spot_shift:
                                if is_vola_benchmark or is_sabr_benchmark:
                                    if is_vola_benchmark:
                                        need_benchmark_spot_shift, new_benchmark_spot, _, _ = VolatilitySurface.get_shifted_spot(shocks, benchmark_vol_surface)
                                    else:
                                        assert is_sabr_benchmark
                                        need_benchmark_spot_shift, new_benchmark_spot = FXSABRVolSurface.get_shifted_spot(shocks, benchmark_vol_surface)
                                    assert need_benchmark_spot_shift
                                    benchmark_fwd *= new_benchmark_spot / benchmark_spot
                                underlying_fwd *= _new_underlying_spot / underlying_vola_surface.spot
                            if _need_underlying_forwards_shift:
                                if is_vola_benchmark:
                                    need_benchmark_spot_shift, new_benchmark_spot, _, _ = VolatilitySurface.get_shifted_spot(shocks, benchmark_vol_surface)
                                    assert need_benchmark_spot_shift
                                    benchmark_fwd *= new_benchmark_spot / benchmark_spot
                                underlying_fwd *= _new_underlying_forwards[i] / underlying_vola_surface.forwards[i]

                            if is_vola_benchmark or is_sabr_benchmark:
                                benchmark_atmvol = VolatilitySurface.get_vol_benchmark(expiry, benchmark_fwd, benchmark_vol_surface)
                            else:
                                # flat bs vol so no need strike
                                benchmark_atmvol = benchmark_vol_surface.get_vol(vola_datetime_to_datetime(expiry), strike=0)
                            underlying_atmvol = underlying_vola_surface.volAtT(expiry, underlying_fwd,
                                                                               StrikeType.K.value)
                            vol_shift = _vol_shock.vol_beta * (benchmark_atmvol + vol_shift) - underlying_atmvol

                        # cap vol
                        curr_vol0 = _input_surface.vol0s[i]
                        if (curr_vol0 + vol_shift) < _vol_shock.min_vol0:
                            vol_shift = _vol_shock.min_vol0 - curr_vol0

                        atm_vol_shifts.append(vol_shift)
                    else:
                        atm_vol_shifts.append(0.0)

                _input_surface.shiftVol(atm_vol_shifts)

        # first workout spot move first
        need_spot_shift, new_spot, sticky_strike, pvol = VolatilitySurface.get_shifted_spot(shocks, new_vol_surface)
        need_forwards_shift, new_forwards, sticky_strike_forwards, pvol_forward = VolatilitySurface.get_shifted_forwards(shocks, new_vol_surface, as_of_date)
        shift_vol_before_underlying = kwargs["shift_vol_before_underlying"] if "shift_vol_before_underlying" in kwargs else True
        if shift_vol_before_underlying:
            for shock in shocks:
                if shock and shock.type == ShockType.VOLSHOCK:
                    if shock.sticky_strike:
                        new_vol_surface.vola_surface = vola.VolUtils.convertToStickyStrike(new_vol_surface.vola_surface)
                    shift_vol_surface(new_vol_surface.vola_surface, shock, as_of_date, self.underlying, original_market,
                                      _need_underlying_spot_shift=need_spot_shift, _new_underlying_spot=new_spot,
                                      _need_underlying_forwards_shift=need_forwards_shift, _new_underlying_forwards=new_forwards,
                                      )

        if need_spot_shift:
            if sticky_strike:
                new_vol_surface.vola_surface = vola.VolUtils.convertToStickyStrike(new_vol_surface.vola_surface)
            if pvol is not None:
                new_vol_surface.vola_surface.updateVolModel(pVol0=pvol)
            new_vol_surface.vola_surface.adjustToNewSpot(new_spot)

        if need_forwards_shift:
            if sticky_strike_forwards:
                new_vol_surface.vola_surface = vola.VolUtils.convertToStickyStrike(new_vol_surface.vola_surface)
            if pvol_forward is not None:
                new_vol_surface.vola_surface.updateVolModel(pVol0=pvol_forward)
            new_vol_surface.vola_surface.adjustToNewForwards(new_forwards)

        # shift future underlying
        for shock in shocks:
            # TODO: test this also works for other future based options
            if shock.type == ShockType.VOLFUTURESHOCK:
                assert (new_vol_surface.underlying_type == UnderlyingType.FUTURES)
                if shock.sticky_strike:
                    new_vol_surface.vola_surface = vola.VolUtils.convertToStickyStrike(new_vol_surface.vola_surface)
                if shock.method == 'imagine':
                    vol_future_forward_shocked = []
                    if shock.use_forward_vol_level and self.underlying == "VIX Index" and shock.benchmark_parameters:
                        # work out vol beta shift
                        benchmark_und = shock.benchmark_underlying
                        benchmark_vol_surface = original_market.get_item(
                            market_utils.create_vol_surface_key(benchmark_und))
                        one_month_days = 30
                        A = shock.benchmark_parameters
                        # print("tenor", "vol", "new shift", "old shift")
                        for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes,
                                                                       new_vol_surface.vola_surface.forwards):
                            if expiry >= as_of_date:
                                tenor_days = (expiry.nanos - as_of_date.nanos) / vola.DateTime.nanosPerDay
                                vol_shift = imagine_bump(A, tenor_days)
                                benchmark_fwd = benchmark_vol_surface.vola_surface.forwardAtT(expiry)
                                benchmark_atmvol = benchmark_vol_surface.vola_surface.volAtT(expiry, benchmark_fwd,
                                                                                             StrikeType.K.value)
                                if shock.use_shifted_spot:
                                    benchmark_fwd_shifted = max(
                                        sum(benchmark_vol_surface.vola_surface.dividendData.divCashs)*benchmark_fwd/benchmark_vol_surface.vola_surface.spot,
                                        benchmark_fwd * (1 + shock.spot_shift_size))
                                    benchmark_atmvol_shifted = benchmark_vol_surface.vola_surface.volAtT(
                                        expiry, benchmark_fwd_shifted, StrikeType.K.value) + vol_shift
                                else:
                                    benchmark_atmvol_shifted = benchmark_atmvol + vol_shift

                                expiry_1m = datetime_to_vola_datetime(
                                    vola_datetime_to_datetime(expiry) + timedelta(days=one_month_days))
                                tenor_days_1m = (expiry_1m.nanos - as_of_date.nanos) / vola.DateTime.nanosPerDay
                                vol_shift_1m = imagine_bump(A, tenor_days_1m)
                                benchmark_fwd_1m = benchmark_vol_surface.vola_surface.forwardAtT(expiry_1m)
                                benchmark_atmvol_1m = benchmark_vol_surface.vola_surface.volAtT(expiry_1m,
                                                                                                benchmark_fwd_1m,
                                                                                                StrikeType.K.value)

                                if shock.use_shifted_spot:
                                    benchmark_fwd_1m_shifted = max(
                                        sum(benchmark_vol_surface.vola_surface.dividendData.divCashs)*benchmark_fwd_1m/benchmark_vol_surface.vola_surface.spot,
                                        benchmark_fwd_1m * (1 + shock.spot_shift_size))
                                    benchmark_atmvol_1m_shifted = benchmark_vol_surface.vola_surface.volAtT(
                                        expiry_1m, benchmark_fwd_1m_shifted, StrikeType.K.value) + vol_shift_1m
                                else:
                                    benchmark_atmvol_1m_shifted = benchmark_atmvol_1m + vol_shift_1m

                                benchmark_1m_forward_variance = benchmark_atmvol_1m ** 2 * tenor_days_1m - benchmark_atmvol ** 2 * tenor_days
                                benchmark_1m_forward_vol = np.sqrt(
                                    benchmark_1m_forward_variance / (tenor_days_1m - tenor_days))

                                benchmark_1m_forward_variance_shifted = benchmark_atmvol_1m_shifted ** 2 * tenor_days_1m - benchmark_atmvol_shifted ** 2 * tenor_days
                                benchmark_1m_forward_vol_shifted = np.sqrt(
                                    benchmark_1m_forward_variance_shifted / (tenor_days_1m - tenor_days))

                                shift_size = (benchmark_1m_forward_vol_shifted - benchmark_1m_forward_vol) * 100
                                shifted_vol_future_forward = original_vol_future_forward + shift_size
                                # imagine_vol_future_forward = vix_v2x_imagine_bump(vola_datetime_to_datetime(as_of_date), vola_datetime_to_datetime(expiry), original_vol_future_forward, shock.parameters)
                                # print(tenor_days, original_vol_future_forward, shifted_vol_future_forward, imagine_vol_future_forward)
                                vol_future_forward_shocked.append(shifted_vol_future_forward)
                            else:
                                vol_future_forward_shocked.append(original_vol_future_forward)
                        new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)
                    else:
                        for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes, new_vol_surface.vola_surface.forwards):
                            if expiry >= as_of_date:
                                vol_future_forward_shocked.append(vix_v2x_imagine_bump(
                                    vola_datetime_to_datetime(as_of_date), vola_datetime_to_datetime(expiry),
                                    original_vol_future_forward, shock.parameters))
                            else:
                                vol_future_forward_shocked.append(original_vol_future_forward)
                        new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)

                elif shock.method == "imagine_interpolated":
                    vol_future_forward_shocked = []
                    for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes, new_vol_surface.vola_surface.forwards):
                        if expiry >= as_of_date:
                            B_all = shock.parameters['crash_B_all']
                            available_crash_sizes = list(sorted(list(B_all.keys())))
                            lb, ub = find_bracket_bounds(available_crash_sizes, shock.parameters['crash_size'])
                            imagine_shock_curve = {}
                            if lb != -float('inf'):
                                imagine_shock_curve[lb] = vix_v2x_imagine_bump(
                                    vola_datetime_to_datetime(as_of_date), vola_datetime_to_datetime(expiry),
                                    original_vol_future_forward, B_all[lb])
                            if ub != float('inf'):
                                imagine_shock_curve[ub] = vix_v2x_imagine_bump(
                                    vola_datetime_to_datetime(as_of_date), vola_datetime_to_datetime(expiry),
                                    original_vol_future_forward, B_all[ub])
                            vol_future_forward_shocked.append(interpolate_curve(imagine_shock_curve, shock.parameters['crash_size'], flat_extrapolate_lower=True, flat_extrapolate_upper=True))
                        else:
                            vol_future_forward_shocked.append(original_vol_future_forward)
                    new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)

                elif shock.method == "sc2024":
                    vol_future_forward_shocked = []
                    for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes, new_vol_surface.vola_surface.forwards):
                        if expiry >= as_of_date:
                            vol_future_forward_shocked.append(vix_v2x_sc2024_bump(
                                vola_datetime_to_datetime(as_of_date), vola_datetime_to_datetime(expiry),
                                original_vol_future_forward, shock.parameters))
                        else:
                            vol_future_forward_shocked.append(original_vol_future_forward)
                    new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)

                elif shock.method == 'percentage':
                    vol_future_forward_shocked = []
                    for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes, new_vol_surface.vola_surface.forwards):
                        if expiry >= as_of_date:
                            vol_future_forward_shocked.append(
                                original_vol_future_forward * (1 + shock.parameters))
                        else:
                            vol_future_forward_shocked.append(original_vol_future_forward)
                    new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)
                elif shock.method == 'level':
                    vol_future_forward_shocked = []
                    for expiry, original_vol_future_forward in zip(new_vol_surface.vola_surface.expiryTimes, new_vol_surface.vola_surface.forwards):
                        if expiry >= as_of_date:
                            vol_future_forward_shocked.append(shock.parameters)
                        else:
                            vol_future_forward_shocked.append(original_vol_future_forward)
                    new_vol_surface.vola_surface.adjustToNewForwards(vol_future_forward_shocked)
                else:
                    raise RuntimeError('Unknown vol future shock method ' + shock.method)

                # update future prices if exists
                for date, price in new_vol_surface.future_prices.items():
                    # skip if base date is not on surface expiry date.
                    if (date.date() == new_vol_surface.base_datetime.date()) \
                            and (date not in new_vol_surface.vola_surface.expiryTimes):
                        continue
                    new_vol_surface.future_prices[date] = new_vol_surface.get_undelrying_price_from_vola_surface(
                        new_vol_surface.base_datetime, date)

        if not shift_vol_before_underlying:
            for shock in shocks:
                if shock.type == ShockType.VOLSHOCK:
                    # no need shift spot since new vol surface has already been shifted
                    shift_vol_surface(new_vol_surface.vola_surface, shock, as_of_date, self.underlying, original_market,
                                      _need_underlying_spot_shift=False, _new_underlying_spot=None,
                                      _need_underlying_forwards_shift=False, _new_underlying_forwards=None)

        # shock div
        for shock in shocks:
            if shock.type == ShockType.DIVSHOCK:
                assert shock.method == "percentage"
                div_times = []
                div_cashes = []
                div_props = []
                div_cashErrors = []
                div_propErrors = []
                index = bisect.bisect_left(new_vol_surface.vola_surface.dividendData.divTimes, new_vol_surface.vola_surface.asOfTime)
                t_year = (new_vol_surface.vola_surface.dividendData.divTimes[-1].nanos - new_vol_surface.vola_surface.asOfTime.nanos) / new_vol_surface.vola_surface.dividendData.divTimes[0].nanosPerYear
                spot = new_vol_surface.vola_surface.spot
                divSum = sum(new_vol_surface.vola_surface.dividendData.divCashs[index:])
                bumpYield = spot * shock.size_bps * 1e-4 * t_year/ divSum

                for i in range(index, len(new_vol_surface.vola_surface.dividendData.divTimes)):
                    div_times.append(new_vol_surface.vola_surface.dividendData.divTimes[i])
                    div_cashes.append(max(0, new_vol_surface.vola_surface.dividendData.divCashs[i] * (1 + bumpYield)))
                    div_props.append(new_vol_surface.vola_surface.dividendData.divProps[i])
                    div_cashErrors.append(new_vol_surface.vola_surface.dividendData.divCashsErrors[i])
                    div_propErrors.append(new_vol_surface.vola_surface.dividendData.divPropsErrors[i])
                new_vol_surface.vola_surface.updateDividendData(vola.makeDividendData(div_times, div_cashes, div_props, div_cashErrors, div_propErrors))

        # shock repo
        for shock in shocks:
            if shock.type == ShockType.REPOSHOCK:
                assert shock.method == "level"
                fa = vola.makeFactoryAnalytics()
                rate_times = []
                rate_values = []
                rate_errors = []

                for i in range(len(new_vol_surface.vola_surface.borrowCurve.rateData.dateTimes)):
                    rate_time = new_vol_surface.vola_surface.borrowCurve.rateData.dateTimes[i]
                    if rate_time >= new_vol_surface.vola_surface.asOfTime:
                        rate_times.append(rate_time)
                        rate_value = new_vol_surface.vola_surface.borrowCurve.rateData.rates[i] + shock.size_bps * 1e-4
                        rate_values.append(rate_value)
                        rate_error = new_vol_surface.vola_surface.borrowCurve.rateData.ratesErrors[i]
                        rate_errors.append(rate_error)
                new_vol_surface.vola_surface.updateBorrowCurve(
                    fa.makeBorrowCurve(new_vol_surface.vola_surface.asOfTime, rate_times, rate_values, rate_errors,
                                       new_vol_surface.vola_surface.borrowCurve.timeConverter,
                                       new_vol_surface.vola_surface.borrowCurve.rateModel))

        # shock rate
        for shock in shocks:
            if shock.type == ShockType.RATESHOCK:
                assert shock.method == "level"
                fa = vola.makeFactoryAnalytics()
                rate_times = []
                rate_values = []
                for i in range(len(new_vol_surface.vola_surface.discountCurve.rateData.dateTimes)):
                    rate_time = new_vol_surface.vola_surface.discountCurve.rateData.dateTimes[i]
                    if rate_time >= new_vol_surface.vola_surface.asOfTime:
                        rate_times.append(rate_time)
                        rate_value = new_vol_surface.vola_surface.discountCurve.rateData.rates[i] + shock.size_bps * 1e-4
                        rate_values.append(rate_value)
                new_vol_surface.vola_surface.updateDiscountCurve(fa.makeRateCurve(
                    new_vol_surface.vola_surface.asOfTime, rate_times, rate_values,
                    new_vol_surface.vola_surface.discountCurve.timeConverter,
                    new_vol_surface.vola_surface.discountCurve.rateModel), True)

        return new_vol_surface

    def get_num_regular(self):
        tc_vb = vola.FactoryTime.toTimeConverterBUS(self.vola_surface.modelData.timeConverterV)
        if tc_vb is None:
            return None
        else:
            return tc_vb.numRegular

    def override_num_regular(self, num_regular):
        md = self.vola_surface.modelData.clone()
        tc_vb = vola.FactoryTime.toTimeConverterBUS(md.timeConverterV)
        if tc_vb is not None:
            md.timeConverterV = vola.makeTimeConverterBUS(tc_vb.weightOvernight, tc_vb.weightIntradayHalfDay, tc_vb.weightNonTrading, tc_vb.calendar, num_regular, tc_vb.numIntradayHalfDay, tc_vb.numNonTrading, tc_vb.startDate, tc_vb.endDate)
            new_vol_surface = self.clone()
            new_vol_surface.vola_surface.updateModelData(md)
            return new_vol_surface
        else:
            return self







