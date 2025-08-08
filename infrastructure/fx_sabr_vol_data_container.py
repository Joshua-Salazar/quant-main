import copy
import math
from datetime import datetime, time
from datalake.datalakeapi import DataLakeAPI
from ..analytics.options import get_strike_from_delta
from ..analytics.utils import interpolate_curve
from ..dates.utils import coerce_timezone, count_business_days, date_to_datetime, datetime_equal, datetime_to_vola_datetime, get_fx_spot_date
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_item import MarketItem
from ..data.refdata import get_underlyings_map
from ..interface.ishock import IShock, ShockType
from ..interface.market_items.ispot import ISpot
from ..interface.market_items.ivolatilitysurface import IVolatilitySurface
from ..analytics import constants
from ..constants.strike_type import StrikeType
from ..constants.underlying_type import UnderlyingType
from ..infrastructure.volatility_surface import VolatilitySurface, imagine_bump, sc2024_bump
import numpy as np
import pandas as pd
import copy

from ctp.models.vol import Alpha, Beta, Rho, Nu
from ctp.models.vol import SABRModel, SABRSmile
from ctp.specifications.defs import TheoreticalForwardPrice, Strike
from ctp.utils.time import datetime_to_timepoint
from ctp.specifications.daycount import Actual365


def sabr_surface(as_of_time, spot, expirys, forwards, alphas, betas, rhos, nus):
    dcc = Actual365()
    smiles = []
    for expiry, forward, alpha, beta, rho, nu in zip(expirys, forwards, alphas, betas, rhos, nus):
        time_to_expiry = dcc.yearFractionTime(datetime_to_timepoint(as_of_time), datetime_to_timepoint(expiry))
        sabr_smile = SABRSmile(
            time_to_expiry=time_to_expiry
            , expiry=datetime_to_timepoint(expiry)
            , forward=TheoreticalForwardPrice(forward)
            , alpha=Alpha(alpha)
            , beta=Beta(beta)
            , rho=Rho(rho)
            , nu=Nu(nu)
        )
        smiles.append(sabr_smile)

    surface = SABRModel(
        day_counter=dcc
        , ref_time=as_of_time
        , spot_ref=spot
        , smiles=smiles
    )

    return surface


def interpolate_sabr_surface(sabr_surface, expiry, strike):
    strike = Strike(strike)
    exp_timepoint = datetime_to_timepoint(expiry)
    vol = sabr_surface.volatility(exp_timepoint, strike)
    return vol


def get_atm_vol(alpha, forward, beta):
    vol = alpha
    if beta != 1:
        vol /= math.pow(forward, 1-beta)
    return vol


def get_alpha(atm_vol, forward, beta):
    alpha = atm_vol
    if beta != 1:
        alpha *= math.pow(forward, 1-beta)
    return alpha


class FXSABRVolDataContainer(DataContainer):
    def __init__(self, pair: str):
        self.market_key = market_utils.create_fx_vol_surface_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_sabr_surface(self, dt=None):
        return self._get_fx_sabr_surface(dt)

    def get_market_item(self, dt):
        return self.get_fx_sabr_surface(dt)


class FXSABRVolDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, base_currency, term_currency):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.base_currency = base_currency
        self.term_currency = term_currency


# we make distinction between DataContainers and MarketItem as for some case data container has too much data
# and it is less efficient to have it in market whereas MarketItem can contain only one day worth of data
# each data container should have a get_market_item function, the return of which will be contained in the market object
class FXSABRVolSurface(MarketItem, ISpot, IVolatilitySurface):
    def __init__(self, market_key, base_date, quoted_delta_strikes, spot, forwards, vols, inversed, underlying_id):
        self.market_key = market_key
        self.base_date = base_date
        self.quoted_delta_strikes = quoted_delta_strikes
        self.spot = spot
        self.forwards = forwards
        self.vols = vols
        self.is_inversed = inversed
        self.underlying_id = underlying_id
        self.underlying = market_key.split(".")[-1]

    def __eq__(self, other):
        if not isinstance(other, FXSABRVolSurface):
            return False
        if self.base_date != other.base_date:
            return False
        if self.spot != other.spot:
            return False
        return self.vols == other.vols

    def __hash__(self):
        alpha = set()
        # only hash alpha, spot, date since we only atm vol bump, spot bump and time bump
        for k, v in self.vols.items():
            alpha.add(v["alpha"])
        return hash((self.base_date, self.spot, frozenset(alpha)))

    def clone(self):
        return copy.deepcopy(self)

    def get_base_datetime(self) -> datetime:
        return self.base_date

    def get_market_key(self):
        return self.market_key

    def get_underlying_type(self) -> UnderlyingType:
        return UnderlyingType.FX

    def get_vol(self, expiry_dt: datetime, strike: float, strike_type: StrikeType = StrikeType.K) -> float:
        assert strike_type == StrikeType.K
        expiries = [datetime.fromisoformat(x['term']) for x in self.vols.values()]
        forwards = [x['forward'] for x in self.vols.values()]
        alphas = [x['alpha'] for x in self.vols.values()]
        betas = [x['beta'] for x in self.vols.values()]
        rhos = [x['rho'] for x in self.vols.values()]
        nus = [x['nu'] for x in self.vols.values()]
        s = sabr_surface(self.base_date, self.spot, expiries, forwards, alphas, betas, rhos, nus)
        vol = interpolate_sabr_surface(s, expiry_dt, strike)
        return vol.val

    def get_delta_strike(self, delta: float, expiry_dt: datetime):
        dcc = Actual365()
        ttm = dcc.yearFractionTime(datetime_to_timepoint(self.base_date), datetime_to_timepoint(expiry_dt)).val
        fwd = self.get_forward(expiry_dt)
        vol = self.get_vol(expiry_dt, fwd)
        is_call = delta > 0
        strike = get_strike_from_delta(delta, fwd, vol, ttm, is_call)
        return strike

    def get_spot(self, base_date: datetime) -> float:

        if not datetime_equal(*coerce_timezone(base_date, self.base_date)):
            raise Exception(f"Found inconsistent base date in vol surface {self.market_key}: "
                            f"{self.base_date.isoformat()} and target date: {base_date.isoformat()}")
        return self.spot

    def get_forward(self, expiry_dt: datetime) -> float:
        dcc = Actual365()
        ttm = dcc.yearFractionTime(datetime_to_timepoint(self.base_date), datetime_to_timepoint(expiry_dt)).val
        fwd = interpolate_curve(self.forwards, ttm, flat_extrapolate_lower=True, flat_extrapolate_upper=True)
        return fwd

    @staticmethod
    def calc_atm_vol_shift(_input_surface, _vol_shock, _as_of_date, original_market,
                           _shift_benchmark_spot_before_taking_atm_vol, _shocks):

        if _vol_shock is not None:
            atm_vol_shifts = {}
            for k, v in _input_surface.vols.items():
                expiry = datetime.fromisoformat(v['term'])
                if expiry >= _as_of_date:
                    # benchmark vol
                    benchmark_underlying = _vol_shock.benchmark_underlying
                    benchmark_vol_surface = None
                    if _vol_shock.vol_beta and _input_surface.underlying != benchmark_underlying:
                        key = market_utils.create_vol_surface_key(benchmark_underlying)
                        if original_market.has_item(key):
                            benchmark_vol_surface = original_market.get_item(key)
                        else:
                            key = market_utils.create_fx_vol_surface_key(benchmark_underlying)
                            benchmark_vol_surface = original_market.get_item(key)
                        # avoid circular import
                        is_vola_benchmark = isinstance(benchmark_vol_surface, VolatilitySurface)
                        is_sabr_benchmark = isinstance(benchmark_vol_surface, FXSABRVolSurface)

                        if is_vola_benchmark or is_sabr_benchmark:
                            benchmark_fwd = VolatilitySurface.get_forward_benchmark(datetime_to_vola_datetime(expiry), benchmark_vol_surface)
                            benchmark_spot = VolatilitySurface.get_spot_benchmark(benchmark_vol_surface)
                            if _shift_benchmark_spot_before_taking_atm_vol:
                                if is_vola_benchmark:
                                    need_benchmark_spot_shift, new_benchmark_spot, _ = VolatilitySurface.get_shifted_spot(_shocks, benchmark_vol_surface)
                                else:
                                    assert is_sabr_benchmark
                                    need_benchmark_spot_shift, new_benchmark_spot = FXSABRVolSurface.get_shifted_spot(_shocks, benchmark_vol_surface)
                                if need_benchmark_spot_shift:
                                    benchmark_fwd *= new_benchmark_spot / benchmark_spot
                            benchmark_atmvol = VolatilitySurface.get_vol_benchmark(datetime_to_vola_datetime(expiry), benchmark_fwd, benchmark_vol_surface)
                        else:
                            # flat bs vol so no need strike
                            benchmark_atmvol = benchmark_vol_surface.get_vol(expiry, strike=0)

                    # vol shift on benchmark
                    if _vol_shock.method == 'imagine':
                        A = _vol_shock.parameters
                        tenor_days = (expiry - _as_of_date).days
                        vol_shift = imagine_bump(A, tenor_days)
                    elif _vol_shock.method == 'sc2024':
                        sc_params = _vol_shock.parameters
                        tenor_days = (expiry - _as_of_date).days
                        vol_shift = sc2024_bump(sc_params, tenor_days)
                    elif _vol_shock.method == '1y_time_weighted_level':
                        # risk team measure: vol shift = sqrt(252/bus days to expiry) * 1y shock
                        expiry_bds = count_business_days(_as_of_date, expiry)
                        # floor expiry bd to 1 month, i.e. 21 to avoid vega brew out
                        bds = 21
                        vol_shift = _vol_shock.parameters * np.sqrt(constants.YEAR_BUSINESS_DAY_COUNT / max(bds, expiry_bds))
                    elif _vol_shock.method == 'time_weighted_level':
                        bds = 21
                        expiry_bds = count_business_days(_as_of_date, expiry)
                        vol_shift = _vol_shock.parameters * bds / max(bds, expiry_bds)
                    elif _vol_shock.method == 'level':
                        vol_shift = _vol_shock.parameters
                    elif _vol_shock.method == 'percentage':
                        curr_vol0 = get_atm_vol(v["alpha"], v["forward"], v["beta"])
                        if benchmark_vol_surface is None:
                            vol_shift = curr_vol0 * _vol_shock.parameters
                        else:
                            vol_shift = benchmark_atmvol * _vol_shock.parameters
                    else:
                        raise RuntimeError('Unknown vol shock method ' + _vol_shock.method)

                    curr_vol0 = get_atm_vol(v["alpha"], v["forward"], v["beta"])
                    # work out vol beta adjusted shift
                    if benchmark_vol_surface is not None:
                        vol_shift = _vol_shock.vol_beta * (benchmark_atmvol + vol_shift) - curr_vol0

                    # cap vol
                    if (curr_vol0 + vol_shift) < _vol_shock.min_vol0:
                        vol_shift = _vol_shock.min_vol0 - curr_vol0

                    # workout alpha shift
                    shifted_vol = curr_vol0 + vol_shift
                    shifted_alpha = get_alpha(shifted_vol, v["forward"], v["beta"])
                    alpha_shift = shifted_alpha - v["alpha"]
                    atm_vol_shifts[k] = alpha_shift
                else:
                    atm_vol_shifts[k] = 0.0

            return atm_vol_shifts
        else:
            return None

    @staticmethod
    def get_shifted_spot(shocks, vol_surface):
        assert isinstance(vol_surface, FXSABRVolSurface)
        need_spot_shift = False
        new_spot = vol_surface.spot
        for shock in shocks:
            if shock and shock.type == ShockType.SPOTSHOCK:
                need_spot_shift = True
                if shock.method == 'percentage':
                    if vol_surface.is_inversed:
                        multiplier = 1 - shock.size * shock.spot_beta
                    else:
                        multiplier = 1 + shock.size * shock.spot_beta
                else:
                    raise RuntimeError('Unknown spot shock method ' + shock.method)
                new_spot *= multiplier
        return need_spot_shift, new_spot

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        new_vol_surface = self.clone()
        dcc = Actual365()
        # time shift
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                new_vol_surface.base_date = shock.shifted_datetime
                new_vols = {}
                new_forwards = {}
                pair = self.underlying
                shifted_spot_date = get_fx_spot_date(shock.shifted_datetime, pair, market=original_market)
                for k, v in new_vol_surface.vols.items():
                    if shifted_spot_date <= datetime.fromisoformat(v['term']):
                        new_t2e = dcc.yearFractionTime(datetime_to_timepoint(shock.shifted_datetime), datetime_to_timepoint(datetime.fromisoformat(v['term']))).val
                        new_forward = interpolate_curve(new_vol_surface.forwards, new_t2e, flat_extrapolate_lower=True, flat_extrapolate_upper=True)
                        new_vols[new_t2e] = {
                            'term': v['term'],
                            'forward': new_forward,
                            'alpha': v['alpha'],
                            'beta': v['beta'],
                            'rho': v['rho'],
                            'nu': v['nu'],
                        }
                        new_forwards[new_t2e] = new_forward
                new_vol_surface.vols = new_vols
                new_vol_surface.forwards = new_forwards

        for shock in shocks:
            if shock and shock.type == ShockType.SPOTSHOCK:
                if shock.method == 'percentage':
                    if self.is_inversed:
                        multiplier = 1 - shock.size if shock.spot_beta is None else 1 - shock.size * shock.spot_beta
                    else:
                        multiplier = 1 + shock.size if shock.spot_beta is None else 1 + shock.size * shock.spot_beta
                else:
                    raise RuntimeError('Unknown spot shock method ' + shock.method)
                new_vols = {}
                new_forwards = {}
                for k, v in new_vol_surface.vols.items():
                    new_vols[k] = {
                        'term': v['term'],
                        'forward': v['forward'] * multiplier,
                        'alpha': v['alpha'],
                        'beta': v['beta'],
                        'rho': v['rho'],
                        'nu': v['nu'],
                    }
                    new_forwards[k] = v['forward'] * multiplier
                new_vol_surface.spot *= multiplier
                new_vol_surface.vols = new_vols
                new_vol_surface.forwards = new_forwards

        for shock in shocks:
            if shock and shock.type == ShockType.VOLSHOCK:
                atm_vol_shifts = FXSABRVolSurface.calc_atm_vol_shift(new_vol_surface, shock, new_vol_surface.base_date, original_market, False, shocks)
                new_vols = {}
                for k, v in new_vol_surface.vols.items():
                    new_vols[k] = {
                        'term': v['term'],
                        'forward': v['forward'],
                        'alpha': v['alpha'] + atm_vol_shifts[k],
                        'beta': v['beta'],
                        'rho': v['rho'],
                        'nu': v['nu'],
                    }
                new_vol_surface.vols = new_vols

        return new_vol_surface


class DatalakeFXSABRVolDataSource(IDataSource):
    def __init__(self, credentials, live=False):
        self.data_container = pd.DataFrame()
        self.credentials = credentials
        self.live = live

    def initialize(self, data_request):
        underlying = f"{data_request.base_currency}{data_request.term_currency}"
        underlying_mapping = get_underlyings_map(return_df=True)
        underlying_mapping = dict(zip(underlying_mapping['m_symbol'].values, [str(x) for x in underlying_mapping['m_id'].values]))

        if underlying in underlying_mapping:
            inversed = False
            #data_pair_ticker = data_request.base_currency + '.' + data_request.term_currency
            CTP_id = underlying_mapping[underlying]
        elif f"{data_request.term_currency}{data_request.base_currency}" in underlying_mapping:
            inversed = True
            #data_pair_ticker = data_request.term_currency + '.' + data_request.base_currency
            CTP_id = underlying_mapping[f"{data_request.term_currency}{data_request.base_currency}"]
        else:
            raise RuntimeError(f"Cannot find underlying id for {underlying}")

        # see https://gitea.capstoneco.com/dcirmirakis/ctp_py_examples/src/branch/master/data_provider.py
        fields = 'under_id,under_pricing_id,dimension,term,fit_source,spot,forward,time_to_expiry,alpha,beta,rho,nu'
        fields = fields.split(",")
        if self.live:
            assert data_request.start_date == data_request.end_date
            source = "CTP_VOL_SABR"
            fields += ["fit_method", "ingrp", "isterm", "latest", "modified", "modifier", "term_trun", "termdate"]
            kwargs = {'dimension': 'SURFACE'}
            st_used = date_to_datetime(data_request.start_date.date())
            et_used = datetime.today()
        else:
            source = "CTP_DAILY_VOL_SABR"
            fields += ["capture_date", "actual_date", "source", "location", "type", ]
            kwargs = {'source': 'CLOSE', 'location': 'NYC', 'dimension': 'SURFACE'}
            st_used = data_request.start_date
            et_used = data_request.end_date
        mktdata = DataLakeAPI(username=self.credentials["username"], token=self.credentials["token"])
        all_data = mktdata.getData(source, CTP_id, fields, st_used, et_used, 0, None, kwargs)
        if self.live:
            all_data["tstamp"] = st_used.date()
            all_data.reset_index(inplace=True)
        else:
            all_data.index.name = "tstamp"
            all_data.reset_index(inplace=True)

        container = FXSABRVolDataContainer(underlying)

        data = []
        data_dict = {}
        all_dates = all_data.tstamp.unique()
        for tstamp_dt in all_dates:
            data_dt = all_data[all_data.tstamp == tstamp_dt]
            if len(np.unique(all_data.capture_date.dt.time)) > 1:
                final_capture = data_dt.capture_date.values[-1]
                print(f"final capture timestamp: {final_capture}")
                data_dt = data_dt[data_dt.capture_date == final_capture]
            forwards = dict(zip(data_dt['time_to_expiry'].values, data_dt['forward'].values))
            vols = dict(zip(data_dt['time_to_expiry'].values,
                            data_dt[['term', 'forward', 'alpha', 'beta', 'rho', 'nu']].to_dict('records')))
            dt = date_to_datetime(tstamp_dt)
            surface = FXSABRVolSurface(container.get_market_key(), dt, None, data_dt['spot'].values[0], forwards, vols, inversed, CTP_id)
            data_dict[dt] = {underlying: surface}
            data.append(data_dt)
        self.vol_data_dict = data_dict

        def _get_fx_sabr_surface(dt):
            return self.vol_data_dict[dt][underlying]

        container._get_fx_sabr_surface = _get_fx_sabr_surface
        return container
