from ctp.instruments.swaps import FxVarSwap
from ctp.instruments.specs import BuySell
from ctp.pricing.engines import NxFxLaggedSwapEngine
from ctp.specifications.currency import CurrencyPair, getCurrency
from ctp.specifications.defs import Notional, Strike
from ctp.utils.time import Calendar, date_to_gregorian_date, datetime_to_timepoint
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..valuation.fx_nx_valuer import FXNxValuer
from ..tradable.varianceswap import VarianceSwap


class FXVarSwapNxValuer(FXNxValuer):
    def __init__(self, *args, **kwargs):
        super(FXVarSwapNxValuer, self).__init__(*args, **kwargs)

    @staticmethod
    def build_trade(var_swap: VarianceSwap, underlying_id: str):
        hc = Calendar(name="te", start_date=date_to_gregorian_date(var_swap.inception),
                      end_date=date_to_gregorian_date(var_swap.expiration), holidays=[])
        swap = FxVarSwap.make(
            instrumentID="DummyID",
            underIDs=[underlying_id],
            currencyPair=CurrencyPair(getCurrency(var_swap.underlying[:3]), getCurrency(var_swap.underlying[3:])),
            effectiveTimepoint=datetime_to_timepoint(var_swap.inception),
            maturityTimepoint=datetime_to_timepoint(var_swap.expiration),
            strike=Strike(var_swap.strike_in_vol),
            notional=Notional(var_swap.notional),
            buySell=BuySell.BUY,
            lag=int(var_swap.lag),
            cap=Strike(var_swap.cap),
            settlementCalendar=hc
        )
        return swap

    def build_pricing_engine(self, base_date, fit_result_map, fixings):
        engine = NxFxLaggedSwapEngine.make(
            valuationTimePoint=datetime_to_timepoint(base_date),
            fitResultsMap=fit_result_map,
            fixings=fixings,
            simPath=self.model_params.sim_path,
            timeSteps=self.model_params.pricing_time_steps,
            direction="Forward"
        )
        return engine

    def ask_keys(self, var_swap: VarianceSwap, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = var_swap.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_fx_vol_surface_key(und))
            yield_ccy = "USD" if und[3:] == "USD" else und[:3]
            if yield_ccy not in keys:
                keys.append(market_utils.create_spot_rates_key(yield_ccy, "BBG_ZERO_RATES"))
        # leave holiday and fixing requirement outside
        return keys