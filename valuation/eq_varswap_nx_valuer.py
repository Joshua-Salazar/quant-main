from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.specs import BuySell
    from ctp.instruments.swaps import VarSwap
    from ctp.pricing.engines import NxEqVarSwapEngine
    from ctp.specifications.currency import getCurrency
    from ctp.specifications.defs import Notional, Strike
    from ctp.utils.time import Calendar, date_to_gregorian_date, datetime_to_timepoint
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..valuation.eq_nx_valuer import EQNxValuer
from ..tradable.varianceswap import VarianceSwap


class EQVarSwapNxValuer(EQNxValuer):
    def __init__(self, *args, **kwargs):
        super(EQVarSwapNxValuer, self).__init__(*args, **kwargs)

    @staticmethod
    def build_trade(var_swap: VarianceSwap, underlying_id: str, trade_suffix=None):
        if var_swap.notional != 1:
            raise Exception(f"Notional {var_swap.notional} must be 1.")
        hc = Calendar(name="te", start_date=date_to_gregorian_date(var_swap.inception),
                      end_date=date_to_gregorian_date(var_swap.expiration), holidays=[])
        inst_id = var_swap.name() if var_swap.inst_id is None else str(var_swap.inst_id)
        if trade_suffix is not None:
            inst_id = inst_id + trade_suffix
        swap = VarSwap.make(
            instrumentID=inst_id,
            underIDs=[underlying_id],
            buySell=BuySell.BUY,
            notional=Notional(var_swap.notional),
            strike=Strike(float(var_swap.strike_in_vol)),
            currency=getCurrency(var_swap.currency),
            effectiveTimepoint=datetime_to_timepoint(var_swap.inception),
            maturityTimepoint=datetime_to_timepoint(var_swap.expiration),
            lag=int(var_swap.lag),
            cap=float(var_swap.cap),
            settlementCalendars=hc,
        )
        return swap

    def build_pricing_engine(self, base_date, fit_result_map, fixings, engine):
        engine = NxEqVarSwapEngine.make(
            valuationTimePoint=datetime_to_timepoint(base_date),
            fitResultsMap=fit_result_map,
            fixings=fixings,
            simPath=self.model_params.sim_path,
            timeSteps=self.model_params.pricing_time_steps,
            direction="Forward",
            marketID="MARKET.BASE",
            solverType="default",
            targetType="target price" if self.model_params.use_ctp_config else "target bs vol",
            randomNumber=self.model_params.random_number,
            mcType=self.model_params.mc_type,
            mcDiscretizationScheme=self.model_params.mc_discretization_scheme,
            antithetic=self.model_params.antithetic,
            useModelDate=self.model_params.use_model_date,
            processingUnit=self.model_params.processing_unit,
            seed=self.model_params.seed,
            volEngine=engine
        )
        return engine

    def get_additional_calc_types(self):
        calc_types = dict(
            vanilla_var="VANILLAVAR",
            vanilla_vol="VANILLAVOL",
            total_days="TOTALDAYS",
            total_days_accrued="TOTALDAYSACCRUED",
            accrued_vol="ACCRUEDVOl"
        )
        return calc_types

    def ask_keys(self, var_swap: VarianceSwap, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = var_swap.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_vol_surface_key(und))
        # leave holiday and fixing requirement outside
        return keys
