from dateutil.relativedelta import relativedelta
from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.options import AutoCallable
    from ctp.instruments.specs import BuySell, OptionType, ExerciseType
    from ctp.models.vol import NxEQVolatilityEngine
    from ctp.pricing.engines import NxEqAutoCallableEngine
    from ctp.specifications.currency import getCurrency
    from ctp.specifications.daycount import DayCountConvention
    from ctp.specifications.defs import AnnualRate, Notional, Strike
    from ctp.termstructures.common import InterestRateTermStructure
    from ctp.utils.time import DayDuration, date_to_gregorian_date, datetime_to_timepoint
from scipy.optimize import fsolve
from ..analytics.symbology import currency_from_ticker
from ..constants.ccy import Ccy
from ..dates.utils import is_aware
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..tradable.autocallable import AutoCallable as SolAutoCallable
from ..valuation.eq_nx_valuer import EQNxValuer
from ..valuation.nx_valuer_utils import get_table_res
from ..valuation.utils import find_fx_for_tradable, return_valuer_res
from ..risk.greeks import calculate_numerical_greeks
import pandas as pd


class EQAutoCallableNXValuer(EQNxValuer):
    def __init__(self, *args, **kwargs):
        if "override_model_params" not in kwargs:
            # kwargs["override_model_params"] = dict(sim_path=30000, calib_time_steps=500, pricing_time_steps=500, x_steps=200, exercise_probability_threshold=0.01, use_ctp_config=False)
            kwargs["override_model_params"] = dict(sim_path=60000, calib_time_steps=500, pricing_time_steps=0, x_steps=200, exercise_probability_threshold=0.01, use_ctp_config=False,
                                                   random_number="QUASI-RANDOM", mc_type="STANDARD", mc_discretization_scheme="2ND ORDER", antithetic=False,
                                                   use_model_date=True, processing_unit="", seed=0, use_abs_strike=True)
            # turn on use_abs_strike as we force surface is also sticky strike
        super(EQAutoCallableNXValuer, self).__init__(*args, **kwargs)

    @staticmethod
    def build_trade(auto_callable: SolAutoCallable, trade_suffix=None):
        inst_id = auto_callable.name() if auto_callable.inst_id is None else str(auto_callable.inst_id)
        if trade_suffix is not None:
            inst_id = inst_id + trade_suffix
        trd = AutoCallable.make(
            instrumentID=inst_id,
            underIDsIn=auto_callable.und_list,
            type=auto_callable.type,
            startSpotsIn=auto_callable.start_spots,
            startTimePoint=datetime_to_timepoint(auto_callable.start_date),
            maturityTimePoint=datetime_to_timepoint(auto_callable.expiration),
            notionalCurrency=getCurrency(auto_callable.currency),
            # sell put
            buySell=BuySell.BUY,
            notional=Notional(auto_callable.notional),
            callPut=OptionType.PUT if auto_callable.call_put == "PUT" else OptionType.CALL,
            strike=Strike(auto_callable.strike), # 0.55
            lowerStrike=Strike(auto_callable.lower_strike), # 0.35
            exerciseType=ExerciseType.EUROPEAN if auto_callable.exercise_type == "EUROPEAN" else ExerciseType.AMERICAN,
            knockInBarrier=auto_callable.knock_in_barrier, # 0.55
            KnockInBarrierObs=auto_callable.knock_in_barrier_obs,
            knockInPutStrike=auto_callable.knock_in_put_strike, # 0.55
            putGearing=auto_callable.put_gearing, # 0.2
            #  coupon
            autoCallDatesIn=[date_to_gregorian_date(dt) for dt in auto_callable.autocall_dates],
            autoCallBarriersIn=auto_callable.autocall_barriers,
            couponBarrier=auto_callable.coupon_barrier,
            couponDatesIn=[date_to_gregorian_date(dt) for dt in auto_callable.coupon_dates],
            couponDown=auto_callable.coupon_down,
            couponUp=auto_callable.coupon_up,
            couponIsMemory=auto_callable.coupon_is_memory,
            couponIsSnowBall=auto_callable.coupon_is_snowball,
            # float rate
            floatDatesIn=[date_to_gregorian_date(dt) for dt in auto_callable.float_dates],
            floatFixedDatesIn=[date_to_gregorian_date(dt) for dt in auto_callable.float_fixed_dates], # float rate fixed dates
            floatStartDate=datetime_to_timepoint(auto_callable.float_start_date),
            fundingSpread=auto_callable.funding_spread,
            rateIndexId=auto_callable.rate_index,
            # glider even
            gliderEvent=auto_callable.glider_event,
            guaranteedCoupon=auto_callable.guaranteed_coupon,   # 0.0103, # 0.0103
            lowerStrikeGearing=0,
            # alive
            width=0,
            lowerMultiplier=0,
            upperMultiplier=0,
            # barrier shift
            autoCallBarrierShiftSize=auto_callable.autocall_barrier_shift_size,
            autoCallBarrierShiftSide=auto_callable.autocall_barrier_shift_side, # side -1: shift left side, 1: shift right side, 0: shift both sides
            couponBarrierShiftSize=auto_callable.coupon_barrier_shift_size,
            couponBarrierShiftSide=auto_callable.coupon_barrier_shift_side, # side -1: shift left side, 1: shift right side, 0: shift both sides
            knockInBarrierShiftSize=auto_callable.knock_in_barrier_shift_size,
            knockInBarrierShiftSide=auto_callable.knock_in_barrier_shift_side # side -1: shift left side, 1: shift right side, 0: shift both sides
        )
        return trd

    def build_pricing_engine(self, base_date, fit_result_map, fixings_map,  auto_callable: SolAutoCallable, market: Market, engine):

        und_ccy_map = {}
        for und in auto_callable.get_underlyings():
            und_ccy_map[currency_from_ticker(und)] = und
        if auto_callable.currency not in und_ccy_map:
            raise Exception(f"Not found volar surface for trade ccy curve {auto_callable.currency}")

        vol_surface = market.get_vol_surface(und_ccy_map[auto_callable.currency])
        trade_ccy_curve = self.build_yield_ts(vol_surface, base_date, auto_callable.currency)

        float_index_curve = self.build_index_ts(market, auto_callable.rate_index_ccy, auto_callable.rate_index)
        corr_row_header = auto_callable.und_list
        corr_matrix = []
        if len(corr_row_header) == 1:
            corr_matrix.append([1])
        else:
            for i in range(len(corr_row_header)):
                corr_matrix.append([0] * len(corr_row_header))

            corr_df = market.get_correlation_matrix(auto_callable.get_underlyings()).corr_matrix
            for i in range(len(corr_row_header)):
                for j in range(len(corr_row_header)):
                    if j >= i:
                        corr = corr_df.loc[corr_row_header[i], corr_row_header[j]]
                        corr_matrix[i][j] = corr
                        corr_matrix[j][i] = corr

        engine = NxEqAutoCallableEngine.make(
            valuationTimePoint=datetime_to_timepoint(base_date),
            fitResultsMap=fit_result_map,
            fixingsMap=fixings_map,
            floatIndexID=auto_callable.rate_index,
            floatIndexCurve=float_index_curve,
            corrRowHeader=corr_row_header,
            corrColHeader=corr_row_header,
            corrMatrix=corr_matrix,
            tradeCcyCurve=trade_ccy_curve,
            fxSpot={},
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

    @staticmethod
    def build_index_ts(market, ccy, curve):
        dcc = DayCountConvention.A365
        use_citi = curve == "SWAP_SOFR"
        curve_data = market.get_spot_rates(currency=ccy, curve=curve)
        days_scaling_factor = 365 if use_citi else 1
        rate_scaling_factor = 100 if use_citi else 1
        yield_data = [
            (DayDuration(int(days*days_scaling_factor)), AnnualRate(curve_data.data_dict[days]/rate_scaling_factor)) for days in sorted(curve_data.data_dict.keys())
        ]
        yield_ts = InterestRateTermStructure.make(
            refdate=date_to_gregorian_date(market.get_base_datetime()),
            daycounterconvention=dcc,
            currency=getCurrency(ccy),
            values=yield_data
        )
        return yield_ts

    def ask_keys(self, ac: SolAutoCallable, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = ac.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_vol_surface_key(und))
        # add curve data
        keys.append(market_utils.create_spot_rates_key(ac.rate_index_ccy, ac.rate_index))
        # add correction
        keys.append(market_utils.create_correlation_matrix(und_list))
        # level holiday and fixing requirement outside
        return keys

    def price(self, auto_callable: SolAutoCallable, market: Market, calc_types='price', **kwargs):

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        numerical_greeks_calc_types_list = [x for x in calc_types_list if "numerical#" in x]
        non_numerical_greeks_calc_types_list = [x for x in calc_types_list if "numerical#" not in x]
        if len(non_numerical_greeks_calc_types_list) > 0:
            und_list = auto_callable.get_underlyings()
            engine = None
            fit_result_map = {}
            for und in und_list:
                vol_surface = market.get_vol_surface(und)
                base_date = market.get_base_datetime()
                ccy = currency_from_ticker(und)
                borrow_curve = self.build_borrow_ts(vol_surface, base_date, ccy)
                yield_curve = self.build_yield_ts(vol_surface, base_date, ccy)
                dividend_curve = self.build_dividend_data(vol_surface, base_date)
                vol_basis = self.get_vol_basis(vol_surface)
                holidays = self.get_holidays(auto_callable.cdr_code,auto_callable.start_date.date(), market)
                engine = NxEQVolatilityEngine.make(
                    valuationTimePoint=datetime_to_timepoint(base_date),
                    quote=vol_surface.get_spot(base_date),
                    borrowCurve=borrow_curve,
                    yieldCurve=yield_curve,
                    dividendCurve=dividend_curve,
                    holidays=holidays,
                    simPath=self.model_params.sim_path,
                    timeSteps=self.model_params.calib_time_steps,
                    direction="Forward",
                    marketID="MARKET.BASE",
                    solverType="fast",
                    targetType="target price" if self.model_params.use_ctp_config else "target bs vol",
                    calibrationMethod="" if self.model_params.use_ctp_config else "BACKWARD FIN DIFF",
                    xSteps=self.model_params.x_steps,
                    exerciseProbabilityThreshold=self.model_params.exercise_probability_threshold,
                    volBasis=vol_basis,
                    volEngine=engine
                )
                surface_id = und
                vol_grid = self.build_eq_vol_grid(vol_surface, surface_id, base_date, und, ccy, self.model_params.use_ctp_config, self.model_params.use_abs_strike, **kwargs)
                engine.setVolatilityModel(self.model_params.model_name)
                vol_grid.setVolatilityEngine(engine)
                fit_result_map[und] = vol_grid.fit()

            fixings_map = {}
            holidays = market.get_holidays(auto_callable.cdr_code, auto_callable.start_date.date(), auto_callable.expiration.date())
            st = auto_callable.start_date.astimezone(auto_callable.expiration.tzinfo) if is_aware(market.get_base_datetime()) else auto_callable.start_date
            for und in und_list:
                fixings_map[und] = self.build_fixings(und, und, market, st, holidays)
            index_holidays = market.get_holidays(auto_callable.index_cdr_code, auto_callable.start_date.date(), auto_callable.expiration.date())
            index_holidays = list(set(holidays + index_holidays))
            # skip today's fixing for sofr
            fixings_map[auto_callable.rate_index] = self.build_fixings(auto_callable.rate_index, auto_callable.rate_index, market, st, index_holidays, skip_today=True)
            pricing_engine = self.build_pricing_engine(
                base_date=base_date,
                fit_result_map=fit_result_map,
                fixings_map=fixings_map,
                auto_callable=auto_callable,
                market=market,
                engine=engine,
            )

            auto_callable_ctp = self.build_trade(auto_callable, trade_suffix=kwargs.get("trade_suffix", None))
            auto_callable_ctp.setPricingEngine(pricing_engine)
            sensitivities = [calc_type for calc_type in calc_types_list if calc_type in ["delta", "vega"]]
            r = auto_callable_ctp.calculateAllResults(sensitivities) if len(sensitivities) > 0 else auto_callable_ctp.calculateAllResults()

        values = tuple()
        for calc_type in non_numerical_greeks_calc_types_list:
            if calc_type == "price":
                values = values + (r.getResult("PV"),)
            elif calc_type in ["ac_log", "cc_log", "log_eq", "log_fund", "acs", "ccs", "kis"]:
                df_res = get_table_res(r, calc_type.upper())
                date_cols = [col for col in df_res.columns if "date" in col.lower()]
                for col in date_cols:
                    df_res[col] = pd.to_datetime(df_res[col], origin="1899-12-30", unit="D")
                values = values + (df_res,)
            elif calc_type in ["funding_leg", "eq_leg", "ac_last", "ki", "expected_life"]:
                values = values + (r.getResult(calc_type.upper()),)
            elif calc_type in ["delta", "gamma", "vega"]:
                values = values + (r.getResult(calc_type),)
            elif calc_type == "volga":
                values = values + (r.getResult("vomma"),)
            elif calc_type in ["theta", "vanna", "rho"]:
                values = values + (0.,)
            elif calc_type == "global_fit":
                fit_res = {}
                for und in und_list:
                    fit_res[und] = r.getResult(calc_type.upper() + "#" + und)
                values = values + (fit_res,)
            elif calc_type == "mc_error":
                values = values + (r.getResult(calc_type.upper()),)
            else:
                raise Exception(f"Unsupported greek: {calc_type}")
        if len(numerical_greeks_calc_types_list) > 0:
            one_side = kwargs.get("one_side", False)
            same_vol_grid_in_delta = kwargs.get("same_vol_grid_in_delta", False)
            sticky_strike_in_vega = kwargs.get("sticky_strike_in_vega", False)
            greeks_kwargs = {k: v for k, v in kwargs.items() if k not in ["one_side", "same_vol_grid_in_delta", "sticky_strike_in_vega"]}
            if len(non_numerical_greeks_calc_types_list) > 0:
                greeks_kwargs["origin_price"] = r.getResult("PV")
            numerical_greeks = calculate_numerical_greeks(
                tradable=auto_callable, market=market, greeks_list=numerical_greeks_calc_types_list, valuer=self,
                one_side=one_side, same_vol_grid_in_delta=same_vol_grid_in_delta, sticky_strike_in_vega=sticky_strike_in_vega, **greeks_kwargs)
            non_numerical_greeks = dict(zip(non_numerical_greeks_calc_types_list, values))
            values = tuple()
            for x in calc_types_list:
                if x in numerical_greeks_calc_types_list:
                    values = values + (numerical_greeks[x],)
                else:
                    values = values + (non_numerical_greeks[x],)
        currency = kwargs.get('currency', None)
        fx = find_fx_for_tradable(market, auto_callable, currency)
        cols = ["ac_log", "cc_log", "log_eq", "log_fund", "acs", "ccs", "kis", "ac_last", "ki", "expected_life", "global_fit", "mc_error"]
        return return_valuer_res(calc_types, values, exclude_scaling_cols=cols, contract_size=auto_callable.contract_size, fx=fx)

    def solve_coupon_up(self, auto_callable: SolAutoCallable, market: Market):
        und_list = auto_callable.get_underlyings()
        engine = None
        fit_result_map = {}
        for und in und_list:
            vol_surface = market.get_vol_surface(und)
            base_date = market.get_base_datetime()
            ccy = currency_from_ticker(und)
            borrow_curve = self.build_borrow_ts(vol_surface, base_date, ccy)
            yield_curve = self.build_yield_ts(vol_surface, base_date, ccy)
            dividend_curve = self.build_dividend_data(vol_surface, base_date)
            vol_basis = self.get_vol_basis(vol_surface)
            # TODO: revisit holidays with diffferent underlyings
            holidays = self.get_holidays(auto_callable.cdr_code,auto_callable.start_date.date(), market)
            engine = NxEQVolatilityEngine.make(
                valuationTimePoint=datetime_to_timepoint(base_date),
                quote=vol_surface.get_spot(base_date),
                borrowCurve=borrow_curve,
                yieldCurve=yield_curve,
                dividendCurve=dividend_curve,
                holidays=holidays,
                simPath=self.model_params.sim_path,
                timeSteps=self.model_params.calib_time_steps,
                direction="Forward",
                marketID="MARKET.BASE",
                solverType="fast",
                targetType="target price" if self.model_params.use_ctp_config else "target bs vol",
                calibrationMethod="" if self.model_params.use_ctp_config else "BACKWARD FIN DIFF",
                xSteps=self.model_params.x_steps,
                exerciseProbabilityThreshold=self.model_params.exercise_probability_threshold,
                volBasis=vol_basis,
                volEngine=engine
            )
            surface_id = und
            # calculate pv so no need vol_grid_spot in kwargs
            vol_grid = self.build_eq_vol_grid(vol_surface, surface_id, base_date, und, ccy, self.model_params.use_ctp_config, self.model_params.use_abs_strike)
            engine.setVolatilityModel(self.model_params.model_name)
            vol_grid.setVolatilityEngine(engine)
            fit_result_map[und] = vol_grid.fit()

        fixings_map = {}
        holidays = market.get_holidays(auto_callable.cdr_code, auto_callable.start_date.date(),
                                       auto_callable.expiration.date())
        for und in und_list:
            fixings_map[und] = self.build_fixings(und, und, market, auto_callable.start_date, holidays)
        index_holidays = market.get_holidays(auto_callable.index_cdr_code, auto_callable.start_date.date(),
                                             auto_callable.expiration.date())
        fixings_map[auto_callable.rate_index] = self.build_fixings(auto_callable.rate_index, auto_callable.rate_index,
                                                                   market, auto_callable.start_date, index_holidays, skip_today=True)
        pricing_engine = self.build_pricing_engine(
            base_date=base_date,
            fit_result_map=fit_result_map,
            fixings_map=fixings_map,
            auto_callable=auto_callable,
            market=market,
            engine=engine,
        )

        pv_cache = {}

        def price(coupon_up):
            tradable_clone = auto_callable.override_coupon_up(coupon_up)
            ac = self.build_trade(tradable_clone)
            ac.setPricingEngine(pricing_engine)
            r = ac.calculateAllResults()
            pv = r.getResult("PV")
            return pv

        def func(coupon_up):
            if coupon_up in pv_cache:
                pv = pv_cache[coupon_up]
            else:
                pv = price(coupon_up)
                pv_cache[coupon_up] = pv
            print(coupon_up, pv)
            return pv**2

        rate_curve = market.get_spot_rate_curve(Ccy(auto_callable.rate_index_ccy), auto_callable.rate_index)
        x0 = rate_curve.get_rate(auto_callable.expiration)

        # stop pv < 5bps. for 1mm notional, pv is below 500
        ytol = 5e-4
        y = abs(ytol * auto_callable.notional) * 100
        maxfev = 5
        cnt = 0
        while abs(y) > abs(ytol * auto_callable.notional) and cnt <= 20:
            print(f"solving {int(cnt/maxfev)}-th iteration with x0={x0}, maxfev={maxfev}")
            res = fsolve(lambda x: func(x[0]), x0, maxfev=maxfev, full_output=1)
            # keep coupon in 4bps
            cpn = round(res[0][0], 4)
            if cpn in pv_cache:
                y = pv_cache[cpn]
            else:
                y = price(cpn)
            x0 = cpn
            cnt += maxfev
        print(f"finished in {cnt} iteration")
        return cpn, y
