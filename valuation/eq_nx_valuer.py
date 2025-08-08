from .. import ENABLE_NXP
if ENABLE_NXP:
    from ctp.instruments.indices import Fixing, Fixings
    from ctp.models.vol import NxEQVolatilityEngine, EqVolatilityGrid, Slice, StrikeType, VolSliceMap
    from ctp.specifications.currency import getCurrency
    from ctp.specifications.daycount import Actual365, DayCountConvention
    from ctp.specifications.defs import AnnualRate, Strike
    from ctp.termstructures.common import InterestRateTermStructure
    from ctp.utils.time import DayDuration, datetime_to_timepoint, date_to_gregorian_date
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve

from ..dates import utils as date_utils
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..interface.ivaluer import IValuer
from ..tradable.condvarianceswap import CondVarianceSwap
from ..tradable.varianceswap import VarianceSwap
from ..tradable.volswap import VolSwap
from ..tradable.voloption import VolOption
from ..risk.greeks import calculate_numerical_greeks
from ..valuation.eq_nx_model_parameters import EQNxModelParams
from ..valuation.utils import find_fx_for_tradable, return_valuer_res
import typing


class EQNxValuer(IValuer):
    MIN_VOL_DAYS = 3
    MAX_VOL_DAYS = 3650

    def __init__(self, override_model_params=None):
        self.model_params = EQNxModelParams(override_model_params)

    def get_vol_basis(self, vol_surface):
        num_regular = vol_surface.get_num_regular()
        if self.model_params.use_ctp_config or num_regular is None:
            vol_basis = "ACT/ACT"
        elif num_regular == 252:
            # numerix only support bus 252
            vol_basis = "BUS/252"
        else:
            raise Exception(f"Found unexpected num regular: {num_regular}, must be 252")
        return vol_basis

    def get_holidays(self, cdr_code, trade_start_date, market):
        if self.model_params.use_ctp_config:
            holidays = []
        else:
            base_date = market.get_base_datetime()
            min_date = min(trade_start_date, base_date.date())
            max_dt = base_date + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
            holidays = market.get_holidays(cdr_code, min_date, max_dt.date())
        return holidays

    @staticmethod
    def build_ir_ts_from_volar_data(vola_ts, vola_as_of_date, base_date, ccy):
        dcc = DayCountConvention.A365
        borrow_data = []
        as_of_date = date_utils.vola_datetime_to_datetime(vola_as_of_date)
        if as_of_date.date() != base_date.date():
            raise Exception("Not support base date and vola surface date are different.")
        max_expiry = base_date + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
        for i, vola_exp in enumerate(vola_ts.dateTimes):
            exp = date_utils.vola_datetime_to_datetime(vola_exp)
            if as_of_date.date() < exp.date() < max_expiry.date() and base_date.date() < exp.date():
                days = (exp.date() - base_date.date()).days
                rate = vola_ts.rates[i]
                borrow_data.append((DayDuration(days), AnnualRate(rate)))
        ir_ts = InterestRateTermStructure.make(
            refdate=date_to_gregorian_date(base_date),
            daycounterconvention=dcc,
            currency=getCurrency(ccy),
            values=borrow_data
        )
        return ir_ts

    @staticmethod
    def build_borrow_ts(vol_surface, base_date, ccy):
        vola_ts = vol_surface.vola_surface.borrowData
        vola_ref_time = vol_surface.vola_surface.asOfTime
        ts = EQNxValuer.build_ir_ts_from_volar_data(vola_ts, vola_ref_time, base_date, ccy)
        return ts

    @staticmethod
    def build_yield_ts(vol_surface, base_date, ccy):
        vola_ts = vol_surface.vola_surface.discountData
        vola_ref_time = vol_surface.vola_surface.asOfTime
        ts = EQNxValuer.build_ir_ts_from_volar_data(vola_ts, vola_ref_time, base_date, ccy)
        return ts

    @staticmethod
    def build_dividend_data(vol_surface, base_date):
        max_expiry = base_date + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
        vola_ts = vol_surface.vola_surface.dividendData
        as_of_date = date_utils.vola_datetime_to_datetime(vol_surface.vola_surface.asOfTime)
        div_data = []
        for i, vola_exp in enumerate(vola_ts.divTimes):
            exp = date_utils.vola_datetime_to_datetime(vola_exp)
            if as_of_date.date() < exp.date() < max_expiry.date():
                div_data.append((date_to_gregorian_date(exp.date()), vola_ts.divCashs[i]))
        return div_data

    @staticmethod
    def build_eq_vol_grid(vol_surface, vol_surface_id, base_date, und, ccy, use_ctp_config, use_abs_strike, **kwargs):
        exch_code = "US"
        moneyness_list = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 100,
                          102.5, 105, 107.5, 110, 115, 120, 125, 130, 140, 150, 160, 170, 180, 190, 200,
                          225, 250, 300]
        if use_ctp_config:
            tenor_list = ['5D', '10D', '20D', '1M', '2M', '3M', '6M', '9M', '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y']
        else:
            tenor_list = ['2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y']
        min_dt = base_date + relativedelta(days=EQNxValuer.MIN_VOL_DAYS)
        max_dt = base_date + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
        vola_surface = vol_surface.vola_surface
        vols = VolSliceMap()
        vol_grid_test = False
        strike_spot = kwargs["vol_grid_spot"][und] if "vol_grid_spot" in kwargs else vola_surface.spot
        for tenor in tenor_list:
            exp_dt = date_utils.add_tenor(base_date, tenor)
            if "W" in tenor:
                tenor_bds = int(tenor[:-1]) * 5
                vola_exp_dt = vola_surface.timeConverterV.toDateTime(vola_surface.asOfTime, tenor_bds / 252)
            else:
                vola_exp_dt = date_utils.datetime_to_vola_datetime(exp_dt)
            if vol_grid_test:
                vola_exp_dt = min(vola_surface.expiryTimes, key=lambda x: abs((x - vola_exp_dt).seconds()))

            if min_dt < exp_dt < max_dt:
                slice = Slice()
                for moneyness_strike in moneyness_list:
                    moneyness_strike /= 100
                    strike = strike_spot * moneyness_strike
                    vol = vola_surface.volAtT(vola_exp_dt, strike)
                    strike_used = strike if use_abs_strike else moneyness_strike
                    slice[Strike(strike_used)] = vol
            if vol_grid_test:
                vols[date_to_gregorian_date(date_utils.vola_datetime_to_datetime(vola_exp_dt))] = slice
            else:
                vols[date_to_gregorian_date(exp_dt)] = slice

        add_expiry = False
        if add_expiry:
            # add expiry date
            from datetime import datetime
            vola_exp_dt = date_utils.datetime_to_vola_datetime(datetime(2026, 1, 16))
            vola_exp_dt = min(vola_surface.expiryTimes, key=lambda x: abs((x - vola_exp_dt).seconds()))
            slice = Slice()
            for moneyness_strike in moneyness_list:
                moneyness_strike /= 100
                strike = strike_spot * moneyness_strike
                vol = vola_surface.volAtT(vola_exp_dt, strike)
                strike_used = strike if use_abs_strike else moneyness_strike
                slice[Strike(strike_used)] = vol
            vols[date_to_gregorian_date(date_utils.vola_datetime_to_datetime(vola_exp_dt))] = slice

        vol_grid = EqVolatilityGrid.make(
            identifier=vol_surface_id,
            dc=Actual365(),
            refDate=datetime_to_timepoint(base_date),
            vols=vols,
            underlying=und,
            ccy=getCurrency(ccy),
            exchCode=exch_code,
            strikeType=StrikeType.K if use_abs_strike else StrikeType.KS
        )
        return vol_grid

    @staticmethod
    def build_fixings(fixing_src, underlying, market, start_date, holidays, skip_today=False):
        fixing_data = []
        dt = start_date
        while dt <= market.get_base_datetime():
            is_today = dt.date() == market.get_base_datetime().date()
            if not skip_today or not is_today:
                if market.has_fixing(fixing_src, dt.date()):
                    fixing = market.get_fixing_from_fixing_table(fixing_src, dt.date())
                    fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                else:
                    # add spot if we ask for intraday fixing, i.e. inception datetime was 1 pm but current market on 3 pm
                    if dt.date() == market.get_base_datetime().date():
                        fixing = market.get_spot(underlying)
                        fixing_data.append(Fixing(date_to_gregorian_date(dt), fixing))
                    else:
                        raise Exception(f"Not found {dt.strftime('%Y-%m-%d')} {underlying} fixings on "
                                        f"{market.get_base_datetime().date().strftime('%Y-%m-%d')}")
            dt = date_utils.add_business_days(dt, 1, holidays)
        fixings = Fixings(fixing_data)
        return fixings

    def get_additional_calc_types(self):
        return {}

    def price(self, swap: typing.Union[VolSwap, VarianceSwap, VolOption], market: Market, calc_types='price', **kwargs):
        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        und = swap.underlying
        additional_calc_types = self.get_additional_calc_types()
        if swap.is_expired(market) or swap.expiration.date() <= market.base_datetime.date():
            values = []
            for calc_type in calc_types_list:
                if calc_type == "price":
                    value = swap.intrinsic_value(market)
                    values.append(value)
                else:
                    values.append(0.)
            additional_calc_types = self.get_additional_calc_types()
        else:
            vol_surface = market.get_vol_surface(und)

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

            numerical_greeks_calc_types_list_to_calc = [x for x in calc_types_list_to_calc if "numerical#" in x]
            non_numerical_greeks_calc_types_list_to_calc = [x for x in calc_types_list_to_calc if "numerical#" not in x]
            if len(non_numerical_greeks_calc_types_list_to_calc) > 0:
                ccy = swap.currency
                borrow_curve = self.build_borrow_ts(vol_surface, base_date, ccy)
                yield_curve = self.build_yield_ts(vol_surface, base_date, ccy)
                dividend_curve = self.build_dividend_data(vol_surface, base_date)
                holidays = self.get_holidays(swap.cdr_code,swap.inception.date(), market)
                vol_basis = self.get_vol_basis(vol_surface)
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
                )

                dummy_surface_id = "DummyID"
                vol_grid = self.build_eq_vol_grid(vol_surface, dummy_surface_id, base_date, und, ccy, self.model_params.use_ctp_config, self.model_params.use_abs_strike, **kwargs)

                engine.setVolatilityModel(self.model_params.model_name)

                vol_grid.setVolatilityEngine(engine)

                fit_results = vol_grid.fit()
                from ..reporting.trade_reporter import TradeReporter
                holidays = TradeReporter(swap).get_holidays(market, self)
                fixings = self.build_fixings(swap.fixing_src, swap.underlying, market, swap.inception, holidays)
                pricing_engine = self.build_pricing_engine(
                    base_date=base_date,
                    fit_result_map={dummy_surface_id: fit_results},
                    fixings=fixings,
                    engine=engine)

                swap_ctp = self.build_trade(swap, dummy_surface_id, kwargs.get("trade_suffix", None))
                swap_ctp.setPricingEngine(pricing_engine)
                sensitivities = [calc_type for calc_type in calc_types_list_to_calc if calc_type in ["delta", "vega"]]
                r = swap_ctp.calculateAllResults(sensitivities) if len(sensitivities) > 0 else swap_ctp.calculateAllResults()

            values = tuple()

            for calc_type in calc_types_list:
                if calc_type in non_numerical_greeks_calc_types_list_to_calc:
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
                    elif calc_type == "global_fit":
                        fit_res = r.getResult(calc_type.upper() + "#" + swap.underlying)
                        values = values + (fit_res,)
                    elif calc_type in additional_calc_types:
                        val = r.getResult(additional_calc_types[calc_type])
                        values = values + (val,)
                    else:
                        raise Exception(f"Unsupported greek: {calc_type}")
                    if price_datastore is not None:
                        price_datastore[calc_key_map[calc_type]] = values[-1]
                elif calc_type not in numerical_greeks_calc_types_list_to_calc:
                    values = values + (price_datastore[calc_key_map[calc_type]],)

            if len(numerical_greeks_calc_types_list_to_calc) > 0:
                one_side = kwargs.get("one_side", False)
                greeks_kwargs = {k: v for k, v in kwargs.items() if k not in ["one_side"]}
                if len(non_numerical_greeks_calc_types_list_to_calc) > 0:
                    greeks_kwargs["origin_price"] = r.getResult("PV")
                numerical_greeks = calculate_numerical_greeks(tradable=swap, market=market, greeks_list=numerical_greeks_calc_types_list_to_calc, valuer=self, one_side=one_side, **greeks_kwargs)
                non_numerical_greeks = dict(zip(non_numerical_greeks_calc_types_list_to_calc, values))
                values = tuple()
                for x in calc_types_list:
                    if x in numerical_greeks_calc_types_list_to_calc:
                        values = values + (numerical_greeks[x],)
                    else:
                        values = values + (non_numerical_greeks[x],)

        currency = kwargs.get('currency', None)
        fx = find_fx_for_tradable(market, swap, currency)
        cols = ["global_fit"] + list(additional_calc_types.keys())
        return return_valuer_res(calc_types, values, exclude_scaling_cols=cols, contract_size=swap.contract_size, fx=fx)

    def create_fair_swap(self, vol_swap: typing.Union[VolSwap, VarianceSwap, VolOption], market: Market, price_datastore=None):
        if not isinstance(vol_swap, VolSwap) and not isinstance(vol_swap, VarianceSwap) and not isinstance(vol_swap, CondVarianceSwap):
            raise Exception(f"Only support vol or var swap")

        assert not vol_swap.is_expired(market)
        und = vol_swap.underlying
        vol_surface = market.get_vol_surface(und)
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
            ccy = vol_swap.currency
            borrow_curve = self.build_borrow_ts(vol_surface, base_date, ccy)
            yield_curve = self.build_yield_ts(vol_surface, base_date, ccy)
            dividend_curve = self.build_dividend_data(vol_surface, base_date)
            holidays = self.get_holidays(vol_swap.cdr_code,vol_swap.inception.date(), market)
            vol_basis = self.get_vol_basis(vol_surface)
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
            )

            dummy_surface_id = "DummyID"
            # calculate pv so no need vol_grid_spot in kwargs
            vol_grid = self.build_eq_vol_grid(vol_surface, dummy_surface_id, base_date, und, ccy,
                                              self.model_params.use_ctp_config, self.model_params.use_abs_strike)

            engine.setVolatilityModel(self.model_params.model_name)

            vol_grid.setVolatilityEngine(engine)
            try:
                fit_results = vol_grid.fit()
            except Exception as e:
                msg = f"Failed to fit Underling {und}: {str(e)}"
                raise Exception(msg)
            from ..reporting.trade_reporter import TradeReporter
            holidays = TradeReporter(vol_swap).get_holidays(market)
            fixings = self.build_fixings(vol_swap.fixing_src, vol_swap.underlying, market, vol_swap.inception, holidays)

            pricing_engine = self.build_pricing_engine(
                base_date=base_date,
                fit_result_map={dummy_surface_id: fit_results},
                fixings=fixings,
                engine=engine)

            def func(strike):
                vol_swap_cloned = vol_swap.clone()
                vol_swap_cloned.strike_in_vol = abs(strike)
                swap = self.build_trade(vol_swap_cloned, dummy_surface_id)
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

