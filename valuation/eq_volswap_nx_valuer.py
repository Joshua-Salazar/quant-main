from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.specs import BuySell
    from ctp.instruments.swaps import VolSwap
    from ctp.pricing.engines import NxEqVolSwapEngine
    from ctp.specifications.currency import getCurrency
    from ctp.specifications.defs import Notional, Strike
    from ctp.utils.time import Calendar, date_to_gregorian_date, datetime_to_timepoint
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..valuation.eq_nx_valuer import EQNxValuer
from ..tradable.volswap import VolSwap as SolVolSwap


class EQVolSwapNxValuer(EQNxValuer):
    def __init__(self, *args, **kwargs):
        super(EQVolSwapNxValuer, self).__init__(*args, **kwargs)

    @staticmethod
    def build_trade(vol_swap: SolVolSwap, underlying_id: str, trade_suffix=None):
        if vol_swap.notional != 1:
            raise Exception(f"Notional {vol_swap.notional} must be 1.")
        hc = Calendar(name="te", start_date=date_to_gregorian_date(vol_swap.inception),
                      end_date=date_to_gregorian_date(vol_swap.expiration), holidays=[])
        swap = VolSwap.make(
            instrumentID="DummyID",
            underIDs=[underlying_id],
            buySell=BuySell.BUY,
            notional=Notional(vol_swap.notional),
            strike=Strike(float(vol_swap.strike_in_vol)),
            currency=getCurrency(vol_swap.currency),
            effectiveTimepoint=datetime_to_timepoint(vol_swap.inception),
            maturityTimepoint=datetime_to_timepoint(vol_swap.expiration),
            lag=int(vol_swap.lag),
            cap=float(vol_swap.cap),
            settlementCalendars=hc,
        )
        return swap

    def build_pricing_engine(self, base_date, fit_result_map, fixings, engine):
        pricing_engine = NxEqVolSwapEngine.make(
            valuationTimePoint=datetime_to_timepoint(base_date),
            fitResultsMap=fit_result_map,
            fixings=fixings,
            simPath=self.model_params.sim_path,
            timeSteps=self.model_params.pricing_time_steps,
            direction="Forward"
        )
        return pricing_engine

    def ask_keys(self, vol_swap: SolVolSwap, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = vol_swap.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_vol_surface_key(und))
        # leave holiday and fixing requirement outside
        return keys
