from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.indices import Fixing, Fixings
    from ctp.models.vol import NxFXVolatilityEngine, FxVolatilityGrid, Slice, VolSliceMap
    from ctp.specifications.currency import CurrencyPair, getCurrency
    from ctp.specifications.daycount import Actual365, DayCountConvention
    from ctp.specifications.defs import AnnualRate, Strike, TheoreticalForwardPrice
    from ctp.termstructures.common import InputFxForwardTS, InterestRateTermStructure
    from ctp.utils.time import DayDuration, datetime_to_timepoint, date_to_gregorian_date
from datetime import datetime
from scipy.optimize import fsolve
from ..dates import utils as date_utils
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..interface.ivaluer import IValuer
from ..reporting.trade_reporter import TradeReporter
from ..tradable.varianceswap import VarianceSwap
from ..tradable.volswap import VolSwap
from ..tradable.voloption import VolOption
from ..valuation.fx_nx_model_parameters import FXNxModelParams
from ..valuation.utils import find_fx_for_tradable
import numpy as np
import typing


class FXNxValuer(IValuer):
    def __init__(self, override_model_params=None):
        self.model_params = FXNxModelParams(override_model_params)

    @staticmethod
    def build_yield_ts(market, ccy):
        dcc = DayCountConvention.A365
        curve_data = market.get_spot_rates(currency=ccy, curve="BBG_ZERO_RATES")
        yield_data = [
            (DayDuration(days), AnnualRate(curve_data.data_dict[days])) for days in sorted(curve_data.data_dict.keys())
        ]
        yield_ts = InterestRateTermStructure.make(
            refdate=date_to_gregorian_date(market.get_base_datetime()),
            daycounterconvention=dcc,
            currency=getCurrency(ccy),
            values=yield_data
        )
        return yield_ts

    @staticmethod
    def build_fx_fwd_ts(vol_surface, base_date, pair):
        dcc = DayCountConvention.A365
        fx_fwd_data = [
            (date_to_gregorian_date(datetime.fromisoformat(x["term"])), TheoreticalForwardPrice(x["forward"]))
            for x in vol_surface.vols.values()
        ]
        fx_fwd_ts = InputFxForwardTS(
            CurrencyPair=CurrencyPair(getCurrency(pair[:3]), getCurrency(pair[3:])),
            refdate=date_to_gregorian_date(base_date),
            daycounterconvention=dcc,
            values=fx_fwd_data
        )
        return fx_fwd_ts

    @staticmethod
    def build_fx_vol_grid(vol_surface, base_date, pair):
        vols = VolSliceMap()
        num_stdev = 5
        num_strike = 21
        max_expiry = max(vol_surface.vols.keys())
        vol = vol_surface.vols[max_expiry]["alpha"]
        fwd = vol_surface.vols[max_expiry]["forward"]
        min_strike = fwd * np.exp(-num_stdev * vol * np.sqrt(max_expiry))
        max_strike = fwd * np.exp(num_stdev * vol * np.sqrt(max_expiry))
        strikes = np.linspace(min_strike, max_strike, num_strike)
        for expiry, param in vol_surface.vols.items():
            slice = Slice()
            expiry = datetime.fromisoformat(param["term"])
            for strike in strikes:
                vol = vol_surface.get_vol(expiry, strike)
                slice[Strike(strike)] = vol

            vols[date_to_gregorian_date(expiry)] = slice

        vol_grid = FxVolatilityGrid.make(
            identifier=vol_surface.underlying_id,
            dc=Actual365(),
            refDate=datetime_to_timepoint(base_date),
            vols=vols,
            ccyPair=CurrencyPair(getCurrency(pair[:3]), getCurrency(pair[3:]))
        )
        return vol_grid

    def price(self, swap: typing.Union[VolSwap, VarianceSwap, VolOption], market: Market, calc_types='price', **kwargs):

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        pair = swap.underlying
        if swap.is_expired(market) or swap.expiration.date() <= market.base_datetime.date():
            values = []
            for calc_type in calc_types_list:
                if calc_type == "price":
                    value = swap.intrinsic_value(market)
                    values.append(value)
                else:
                    values.append(0.)
        else:
            vol_surface = market.get_fx_sabr_vol_surface(pair)

            price_datastore = kwargs.get("price_datastore", None)
            base_date = market.get_base_datetime()
            calc_types_list_to_calc = []
            calc_key_map = {}
            if price_datastore is None:
                calc_types_list_to_calc = calc_types_list
            else:
                fixing_table_key = market_utils.create_fixing_table_key()
                if market.has_item(fixing_table_key):
                    fixing_table = market.get_item(fixing_table_key)
                else:
                    fixing_table = None
                for calc_type in calc_types_list:
                    key = (base_date, swap, vol_surface, fixing_table, calc_type, self.model_params)
                    calc_key_map[calc_type] = key
                    if key not in price_datastore:
                        # print("storage:")
                        # [print(key[0].date(), "VOL" if isinstance(key[1], VolSwap) else "VAR", key[2].spot,
                        #        list(key[2].vols.values())[3]["alpha"], key[3].fixing_table.iloc[-1]["fixing"],
                        #        key[4]) for key in price_datastore.keys()]
                        # print("key:")
                        # print(base_date.date(), "VOL" if isinstance(swap, VolSwap) else "VAR",
                        #       vol_surface.spot, list(vol_surface.vols.values())[3]["alpha"],
                        #       None if fixing_table is None else fixing_table.fixing_table.iloc[-1]["fixing"],
                        #       calc_type)
                        calc_types_list_to_calc.append(calc_type)

            if len(calc_types_list_to_calc) > 0:
                fx_fwd_ts = self.build_fx_fwd_ts(vol_surface, base_date, pair)
                yield_curve_ccy = "USD" if pair[3:] == "USD" else pair[:3]
                yield_curve = self.build_yield_ts(market, yield_curve_ccy)
                engine = NxFXVolatilityEngine.make(
                    valuationTimePoint=datetime_to_timepoint(base_date),
                    quote=vol_surface.get_spot(base_date),
                    fxforward=fx_fwd_ts,
                    discountCurve=yield_curve,
                    simPath=self.model_params.sim_path,
                    timeSteps=self.model_params.calib_time_steps,
                )
                vol_grid = self.build_fx_vol_grid(vol_surface, base_date, pair)

                engine.setVolatilityModel(self.model_params.model_name)

                vol_grid.setVolatilityEngine(engine)

                fit_results = vol_grid.fit()
                holidays = TradeReporter(swap).get_holidays(market, self)
                fixing_data = []
                dt = swap.inception
                while dt <= market.get_base_datetime():
                    if market.has_fixing(swap.fixing_src, dt.date()):
                        fixing = market.get_fixing_from_fixing_table(swap.fixing_src, dt.date())
                        fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                    else:
                        # add spot if we ask for intraday fixing, i.e. inception datetime was 1 pm but current market on 3 pm
                        if dt.date() == market.get_base_datetime().date():
                            fixing = market.get_spot(swap.underlying)
                            fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                        else:
                            raise Exception(f"Not found {dt.strftime('%Y-%m-%d')} {swap.underlying} fixings on "
                                            f"{market.get_base_datetime().date().strftime('%Y-%m-%d')}")
                    dt = date_utils.add_business_days(dt, 1, holidays)
                fixings = Fixings(fixing_data)
                pricing_engine = self.build_pricing_engine(
                    base_date=base_date,
                    fit_result_map={vol_surface.underlying_id: fit_results},
                    fixings=fixings)

                swap_ctp = self.build_trade(swap, vol_surface.underlying_id)
                swap_ctp.setPricingEngine(pricing_engine)
                sensitivities = [calc_type for calc_type in calc_types_list_to_calc if calc_type in ["delta", "vega"]]
                r = swap_ctp.calculateAllResults(sensitivities) if len(sensitivities) > 0 else swap_ctp.calculateAllResults()

            values = tuple()
            for calc_type in calc_types_list:
                if calc_type in calc_types_list_to_calc:
                    if calc_type == "price":
                        try:
                            values = values + (r.getResult("PV"),)
                        except:
                            raise
                    elif calc_type in ["delta", "gamma", "vega"]:
                        values = values + (r.getResult(calc_type),)
                    elif calc_type == "volga":
                        values = values + (r.getResult("vomma"),)
                    elif calc_type in ["theta", "vanna", "rho"]:
                        values = values + (0.,)
                    else:
                        raise Exception(f"Unsupported greek: {calc_type}")
                    if price_datastore is not None:
                        price_datastore[calc_key_map[calc_type]] = values[-1]
                else:
                    values = values + (price_datastore[calc_key_map[calc_type]],)

        conversion_to_trade_ccy = 1.
        if swap.currency != pair[3:]:
            conversion_to_trade_ccy = market.get_fx_spot(f"{pair[3:]}{swap.currency}")
        currency = kwargs.get('currency', None)
        fx = find_fx_for_tradable(market, swap, currency) * conversion_to_trade_ccy
        if isinstance(calc_types, list):
            return [x * swap.contract_size * fx for x in list(values)]
        else:
            return values[0] * swap.contract_size * fx

    def create_fair_swap(self, vol_swap: VolSwap, market: Market, price_datastore=None):
        if not isinstance(vol_swap, VolSwap) and not isinstance(vol_swap, VarianceSwap):
            raise Exception(f"Only support vol or var swap")

        assert not vol_swap.is_expired(market)
        pair = vol_swap.underlying
        vol_surface = market.get_fx_sabr_vol_surface(pair)
        base_date = market.get_base_datetime()

        fixing_table_key = market_utils.create_fixing_table_key()
        if market.has_item(fixing_table_key):
            fixing_table = market.get_item(fixing_table_key)
        else:
            fixing_table = None

        fair_strike = None
        if price_datastore is not None:
            key = (base_date, vol_swap, vol_surface, fixing_table, "fair_strike")
            if key in price_datastore:
                fair_strike = price_datastore[key]

        if fair_strike is None:
            fx_fwd_ts = self.build_fx_fwd_ts(vol_surface, base_date, pair)
            yield_curve_ccy = "USD" if pair[3:] == "USD" else pair[:3]
            yield_curve = self.build_yield_ts(market, yield_curve_ccy)
            engine = NxFXVolatilityEngine.make(
                valuationTimePoint=datetime_to_timepoint(base_date),
                quote=vol_surface.get_spot(base_date),
                fxforward=fx_fwd_ts,
                discountCurve=yield_curve,
            )
            vol_grid = self.build_fx_vol_grid(vol_surface, base_date, pair)
            engine.setVolatilityModel(self.model_params.model_name)
            vol_grid.setVolatilityEngine(engine)
            try:
                fit_results = vol_grid.fit()
            except Exception as e:
                msg = f"Failed to fit Underling {pair}: {str(e)}"
                raise Exception(msg)
            holidays = TradeReporter(vol_swap).get_holidays(market, self)
            fixing_data = []
            dt = vol_swap.inception
            while dt <= market.get_base_datetime():
                if market.has_fixing(vol_swap.fixing_src, dt.date()):
                    fixing = market.get_fixing_from_fixing_table(vol_swap.fixing_src, dt.date())
                    fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                else:
                    # add spot if we ask for intraday fixing, i.e. inception datetime was 1 pm but current market on 3 pm
                    if dt.date() == market.get_base_datetime().date():
                        fixing = market.get_spot(vol_swap.underlying)
                        fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                    else:
                        raise Exception(f"Not found {dt.strftime('%Y-%m-%d')} {vol_swap.underlying} fixings on "
                                        f"{market.get_base_datetime().date().strftime('%Y-%m-%d')}")

                dt = date_utils.add_business_days(dt, 1, holidays)
            fixings = Fixings(fixing_data)

            pricing_engine = self.build_pricing_engine(
                base_date=base_date,
                fit_result_map={vol_surface.underlying_id: fit_results},
                fixings=fixings)

            def func(strike):
                vol_swap_cloned = vol_swap.clone()
                vol_swap_cloned.strike_in_vol = abs(strike)
                swap = self.build_trade(vol_swap_cloned, vol_surface.underlying_id)
                swap.setPricingEngine(pricing_engine)
                r = swap.calculateAllResults()
                pv = r.getResult("PV")
                return pv

            sol = fsolve(lambda x: func(x[0]), vol_swap.strike_in_vol)
            fair_strike = abs(sol[0])
            if price_datastore is not None:
                price_datastore[key] = fair_strike

        fair_strike_swap = vol_swap.override_strike(fair_strike)

        return fair_strike_swap
