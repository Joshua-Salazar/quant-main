from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.specs import BuySell
    from ctp.instruments.swaps import CondVarSwap
    from ctp.pricing.engines import NxEqCondVarSwapEngine
    from ctp.specifications.currency import getCurrency
    from ctp.specifications.defs import Notional, Strike
    from ctp.utils.time import Calendar, date_to_gregorian_date, datetime_to_timepoint
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..valuation.eq_nx_valuer import EQNxValuer
from ..tradable.condvarianceswap import CondVarianceSwap


class EQCondVarSwapNxValuer(EQNxValuer):
    def __init__(self, *args, **kwargs):
        super(EQCondVarSwapNxValuer, self).__init__(*args, **kwargs)

    @staticmethod
    def build_trade(cond_var_swap: CondVarianceSwap, underlying_id: str, trade_suffix=None):
        hc = Calendar(name="te", start_date=date_to_gregorian_date(cond_var_swap.inception),
                      end_date=date_to_gregorian_date(cond_var_swap.expiration), holidays=[])
        if cond_var_swap.notional != 1:
            raise Exception(f"Notional {cond_var_swap.notional} must be 1.")
        inst_id = cond_var_swap.name() if cond_var_swap.inst_id is None else str(cond_var_swap.inst_id)
        if trade_suffix is not None:
            inst_id = inst_id + trade_suffix
        swap = CondVarSwap.make(
            instrumentID=inst_id,
            underIDs=[underlying_id],
            buySell=BuySell.BUY,
            notional=Notional(cond_var_swap.notional),
            strike=Strike(float(cond_var_swap.strike_in_vol)),
            currency=getCurrency(cond_var_swap.currency),
            effectiveTimepoint=datetime_to_timepoint(cond_var_swap.inception),
            maturityTimepoint=datetime_to_timepoint(cond_var_swap.expiration),
            barrierCondition=cond_var_swap.barrier_condition,
            barrierType=cond_var_swap.barrier_type,
            downVarBarrier=1000000000 if cond_var_swap.down_var_barrier is None else cond_var_swap.down_var_barrier,
            upVarBarrier=cond_var_swap.up_var_barrier,
            lag=1,  # Todo: hard-code 1 for now
            cap=float(cond_var_swap.cap),
            settlementCalendars=hc,
        )
        return swap

    def build_pricing_engine(self, base_date, fit_result_map, fixings, engine):
        engine = NxEqCondVarSwapEngine.make(
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
            conditional_var="CONDITIONALVAR",
            in_days="INDAYS",
            total_days="TOTALDAYS",
            in_days_accrued="INDAYSACCRUED",
            accrued_vol="ACCRUEDVOl",
            corridor_prob="CORRIDORPROB"
        )
        return calc_types

    def ask_keys(self, cond_var_swap: CondVarianceSwap, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = cond_var_swap.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_vol_surface_key(und))
        # leave holiday and fixing requirement outside
        return keys