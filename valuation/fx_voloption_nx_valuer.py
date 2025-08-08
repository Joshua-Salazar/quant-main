from ctp.instruments.indices import Fixing, Fixings
from ctp.instruments.swaps import FxVolOption
from ctp.instruments.specs import BuySell
from ctp.models.vol import NxFXVolatilityEngine
from ctp.pricing.engines import NxFxVolOptionEngine
from ctp.specifications.currency import CurrencyPair, getCurrency
from ctp.specifications.defs import Notional, Strike
from ctp.utils.time import Calendar, date_to_gregorian_date, datetime_to_timepoint
from scipy.optimize import fsolve
from ..dates import utils as date_utils
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..reporting.trade_reporter import TradeReporter
from ..tradable.voloption import VolOption
from ..valuation.fx_nx_valuer import FXNxValuer


class FXVolOptionNxValuer(FXNxValuer):
    def __init__(self):
        super(FXVolOptionNxValuer, self).__init__()

    @staticmethod
    def build_trade(vol_option: VolOption, underlying_id: str):
        hc = Calendar(name="te", start_date=date_to_gregorian_date(vol_option.inception),
                      end_date=date_to_gregorian_date(vol_option.expiration), holidays=[])
        option = FxVolOption.make(
            instrumentID="DummyID",
            underIDs=[underlying_id],
            currencyPair=CurrencyPair(getCurrency(vol_option.underlying[:3]), getCurrency(vol_option.underlying[3:])),
            effectiveTimepoint=datetime_to_timepoint(vol_option.inception),
            maturityTimepoint=datetime_to_timepoint(vol_option.expiration),
            strike=Strike(float(vol_option.vol_strike)),
            notional=Notional(vol_option.notional),
            buySell=BuySell.BUY,
            lag=int(vol_option.lag),
            isCap=bool(vol_option.is_cap),
            cap=Strike(float(vol_option.cap)),
            settlementCalendar=hc
        )
        return option

    def build_pricing_engine(self, base_date, fit_result_map, fixings):
        engine = NxFxVolOptionEngine.make(
            valuationTimePoint=datetime_to_timepoint(base_date),
            fitResultsMap=fit_result_map,
            fixings=fixings,
            simPath=self.model_params.sim_path,
            timeSteps=self.model_params.pricing_time_steps,
            direction="Forward"
        )
        return engine

    def create_option_with_delta_strike(self, tradable: VolOption, delta_strike: float, market: Market, price_datastore=None):
        if not isinstance(tradable, VolOption):
            raise Exception(f"Only support vol option")

        pair = tradable.underlying
        vol_surface = market.get_fx_sabr_vol_surface(pair)
        base_date = market.get_base_datetime()

        fixing_table_key = market_utils.create_fixing_table_key()
        if market.has_item(fixing_table_key):
            fixing_table = market.get_item(fixing_table_key)
        else:
            fixing_table = None

        fair_strike = None
        if price_datastore is not None:
            key = (base_date, tradable, vol_surface, fixing_table, f"delta_strike:{delta_strike}")
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
            engine.setVolatilityModel("DUPIRE")
            vol_grid.setVolatilityEngine(engine)
            fit_results = vol_grid.fit()
            holidays = TradeReporter(tradable).get_holidays(market)
            fixing_data = []
            dt = tradable.inception
            while dt <= market.get_base_datetime():
                if market.has_fixing(tradable.fixing_src, dt.date()):
                    fixing = market.get_fixing_from_fixing_table(tradable.fixing_src, dt.date())
                    fixing_data.append(tradable(date_to_gregorian_date(dt), fixing))
                else:
                    # add spot if we ask for intraday fixing, i.e. inception datetime was 1 pm but current market on 3 pm
                    if dt.date() == market.get_base_datetime().date():
                        fixing = market.get_spot(tradable.underlying)
                        fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                    else:
                        raise Exception(f"Not found {dt.strftime('%Y-%m-%d')} {tradable.underlying} fixings on "
                                        f"{market.get_base_datetime().date().strftime('%Y-%m-%d')}")

                dt = date_utils.add_business_days(dt, 1, holidays)
            fixings = Fixings(fixing_data)

            pricing_engine = self.build_pricing_engine(
                base_date=base_date,
                fit_result_map={vol_surface.underlying_id: fit_results},
                fixings=fixings)

            def func(strike):
                tradable_clone = tradable.override_strike(strike)
                swap = self.build_trade(tradable_clone, vol_surface.underlying_id)
                swap.setPricingEngine(pricing_engine)
                r = swap.calculateAllResults(["vega"])
                delta = r.getResult("vega")
                return (delta - delta_strike)**2

            sol = fsolve(lambda x: func(x[0]), 8)
            fair_strike = sol[0]
            if price_datastore is not None:
                price_datastore[key] = fair_strike

        delta_strike_tradable = tradable.override_strike(fair_strike)

        return delta_strike_tradable

    def ask_keys(self, vol_option: VolOption, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = vol_option.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_fx_vol_surface_key(und))
            yield_ccy = "USD" if und[3:] == "USD" else und[:3]
            if yield_ccy not in keys:
                keys.append(market_utils.create_spot_rates_key(yield_ccy, "BBG_ZERO_RATES"))
        # leave holiday and fixing requirement outside
        return keys