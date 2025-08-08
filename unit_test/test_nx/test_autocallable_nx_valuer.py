import unittest
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ...tools import test_utils
from ...data import datalake
from ...tradable.autocallable import AutoCallable
from ...valuation.eq_autocallable_nx_valuer import EQAutoCallableNXValuer
from ...data.refdata import get_underlyings_map
from ...constants.underlying_type import UnderlyingType
from ...dates.utils import get_holidays, bdc_adjustment
from ...infrastructure import market_utils
from ...infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateInternalDataSource, SpotRateCitiDataSource
from ...infrastructure.correlation_matrix_container import CorrelationMatrixDataRequest, ConstantCorrelationMatrixDataSource
from ...infrastructure.market import Market
from ...infrastructure.fixing_table import FixingTable
from ...infrastructure.holiday_center import HolidayCenter
from ...infrastructure.volatility_surface import VolatilitySurface
from ...reporting.trade_reporter import TradeReporter
import pandas as pd


class TestAutoCallableNXValuer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutoCallableNXValuer, self).__init__(*args, **kwargs)
        self.src_folder = test_utils.get_test_src_folder("test_autocallable_nx_valuer")
        self.test_folder = test_utils.get_test_data_folder("test_autocallable_nx_valuer")

        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_single_underlying(self):
        # 1) create trade
        und_list = ["SPX Index"]
        inception_date = datetime(2024, 10, 1)
        expiration = datetime(2029, 10, 1)
        # inception_date = datetime(2024, 4, 5)
        # expiration = datetime(2026, 4, 2)
        type = "AUTOCALL_SWAP"
        dt = inception_date
        # load vol surface:
        market = Market(base_datetime=dt)
        ref_time = 'NYC|CLOSE|{}'.format(dt.strftime('%Y%m%d'))
        und_map = get_underlyings_map(return_df=False)
        for und in und_list:
            und_id = und_map[und]
            surface = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time, underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
            surface = surface.override_num_regular(num_regular=252)
            market.add_item(market_utils.create_vol_surface_key(und), surface)
        start_spots = [market.get_spot(und) for und in und_list]
        currency = "USD"
        notional = 1000000
        call_put = "PUT"
        strike = 0.55
        lower_strike = 0
        exercise_type = "EUROPEAN"
        knock_in_barrier = 0.55
        knock_in_barrier_obs = "European"
        knock_in_put_strike = 0.55
        put_grearing = 1
        autocall_dates = []
        tmp = inception_date + relativedelta(months=6)
        while tmp < expiration:
            bd = bdc_adjustment(tmp)
            autocall_dates.append(bd)
            tmp = tmp + relativedelta(months=6)
        autocall_dates.append(expiration)
        num_autocall_dates = len(autocall_dates)
        autocall_barriers = [1.1] * num_autocall_dates
        coupon_barrier = 0.75
        coupon_dates = autocall_dates
        coupon_down = 0
        coupon_up = 0.015
        coupon_is_memory = True
        coupon_is_snowball = False
        float_dates = coupon_dates
        float_fixed_dates = float_dates
        float_start_date = inception_date
        funding_spread = 0
        rate_index = "SOFRRATE"
        rate_index_ccy = "USD"
        glider_event = False
        guaranteed_coupon = 0.0
        autocallable = AutoCallable(
            und_list=und_list, type=type, start_spots=start_spots, start_date=inception_date, expiration=expiration,
            currency=currency, notional=notional, call_put=call_put, strike=strike, lower_strike=lower_strike,
            exercise_type=exercise_type, knock_in_barrier=knock_in_barrier, knock_in_barrier_obs=knock_in_barrier_obs,
            knock_in_put_strike=knock_in_put_strike, put_grearing=put_grearing, autocall_dates=autocall_dates,
            autocall_barriers=autocall_barriers, coupon_barrier=coupon_barrier, coupon_dates=coupon_dates,
            coupon_down=coupon_down, coupon_up=coupon_up, coupon_is_memory=coupon_is_memory,
            coupon_is_snowball=coupon_is_snowball, float_dates=float_dates, float_fixed_dates=float_fixed_dates,
            float_start_date=float_start_date, funding_spread=funding_spread, rate_index=rate_index,
            rate_index_ccy=rate_index_ccy, glider_event=glider_event, guaranteed_coupon=guaranteed_coupon)

        # 2) add other market data
        index_curve_data = SpotRateInternalDataSource().initialize(SpotRateRequest(dt, dt, rate_index_ccy, rate_index))
        market.add_item(index_curve_data.get_market_key(), index_curve_data.get_market_item(dt))

        # check if fixing is required
        bbg_ticker_map = {"SOFRRATE": "SOFRRATE Index"}
        inverse_bbg_ticker_map = {v: k for k, v in bbg_ticker_map.items()}
        fixing_requirement = TradeReporter(autocallable).get_fixing_requirement(market)
        if len(fixing_requirement) > 0:
            # create fixing table from bbg
            fixings = []
            for fixing_req in fixing_requirement:
                und = bbg_ticker_map.get(fixing_req.underlying, fixing_req.underlying)
                fixing = datalake.get_bbg_history(
                    [und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                fixing["date"] = pd.to_datetime(fixing["date"]).dt.date
                fixing["ticker"] = fixing["ticker"].apply(lambda ticker: inverse_bbg_ticker_map.get(ticker, ticker))
                fixing = fixing.rename(columns={"PX_LAST": "fixing", "ticker": "underlying"}).set_index(
                    ["date", "underlying"])
                fixings.append(fixing)

            fixings = pd.concat(fixings).reset_index()
            fixings.loc[fixings.underlying == rate_index, "fixing"] /= 100
            fixing_table = FixingTable(fixings)
            market.add_item(market_utils.create_fixing_table_key(), fixing_table)

        # check if holiday is required
        holiday_requirement = TradeReporter(autocallable).get_holiday_requirement(market, EQAutoCallableNXValuer())
        hol_center = {}
        if len(holiday_requirement) > 0:
            for hol_req in holiday_requirement:
                holidays = get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
                code_hols = hol_center.get(hol_req.code, []) + holidays
                hol_center[hol_req.code] = sorted(set(code_hols))

        if len(hol_center) > 0:
            key = market_utils.create_holiday_center_key()
            market.add_item(key, HolidayCenter(hol_center))

        # 3) calculate PV using default DUPIRE model
        model_param_overrides = dict(
            sim_path=15000,
            calib_time_steps=400,
            pricing_time_steps=400,
        )
        [pv, delta, vega] = EQAutoCallableNXValuer(override_model_params=model_param_overrides).price(autocallable, market, calc_types=['price', 'delta', 'vega'])
        print(pv, delta, vega)
        self.assertAlmostEqual(pv, -48812.25230455919)
        self.assertAlmostEqual(delta, 2031.3992621647956)
        self.assertAlmostEqual(vega, 0)

    def test_single_underlying_bt(self):
        # 1) create trade
        und_list = ["SPX Index"]
        inception_date = datetime(2024, 4, 5)
        expiration = datetime(2026, 4, 2)
        type = "AUTOCALL_SWAP"
        # dt = datetime(2024, 7, 8)
        dt = inception_date
        # load vol surface:
        market = Market(base_datetime=dt)
        market_inception_date = Market(base_datetime=inception_date)
        ref_time = 'NYC|CLOSE|{}'.format(dt.strftime('%Y%m%d'))
        ref_time_inception_date = 'NYC|CLOSE|{}'.format(inception_date.strftime('%Y%m%d'))
        und_map = get_underlyings_map(return_df=False)
        for und in und_list:
            und_id = und_map[und]
            surface = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time, underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
            surface = surface.override_num_regular(num_regular=252)
            market.add_item(market_utils.create_vol_surface_key(und), surface)
            surface_inception_date = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time_inception_date, underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
            surface_inception_date = surface_inception_date.override_num_regular(num_regular=252)
            market_inception_date.add_item(market_utils.create_vol_surface_key(und), surface_inception_date)
        start_spots = [market_inception_date.get_spot(und) for und in und_list]
        currency = "USD"
        notional = 1000000
        call_put = "PUT"
        strike = 0.75
        lower_strike = 0
        exercise_type = "EUROPEAN"
        knock_in_barrier = 0.75
        knock_in_barrier_obs = "European"
        knock_in_put_strike = 0.75
        put_grearing = 1
        autocall_dates = []
        tmp = inception_date + relativedelta(months=6)
        while tmp < expiration:
            bd = bdc_adjustment(tmp)
            autocall_dates.append(bd)
            tmp = tmp + relativedelta(months=6)
        autocall_dates.append(expiration)
        num_autocall_dates = len(autocall_dates)
        autocall_barriers = [1.1] * num_autocall_dates
        coupon_barrier = 0.75
        coupon_dates = autocall_dates
        coupon_down = 0
        coupon_up = 0.035
        coupon_is_memory = True
        coupon_is_snowball = False
        float_dates = coupon_dates
        float_fixed_dates = float_dates
        float_start_date = inception_date
        funding_spread = 0
        rate_index = "SOFRRATE"
        rate_index_ccy = "USD"
        glider_event = False
        guaranteed_coupon = 0.0
        autocallable = AutoCallable(
            und_list=und_list, type=type, start_spots=start_spots, start_date=inception_date, expiration=expiration,
            currency=currency, notional=notional, call_put=call_put, strike=strike, lower_strike=lower_strike,
            exercise_type=exercise_type, knock_in_barrier=knock_in_barrier, knock_in_barrier_obs=knock_in_barrier_obs,
            knock_in_put_strike=knock_in_put_strike, put_grearing=put_grearing, autocall_dates=autocall_dates,
            autocall_barriers=autocall_barriers, coupon_barrier=coupon_barrier, coupon_dates=coupon_dates,
            coupon_down=coupon_down, coupon_up=coupon_up, coupon_is_memory=coupon_is_memory,
            coupon_is_snowball=coupon_is_snowball, float_dates=float_dates, float_fixed_dates=float_fixed_dates,
            float_start_date=float_start_date, funding_spread=funding_spread, rate_index=rate_index,
            rate_index_ccy=rate_index_ccy, glider_event=glider_event, guaranteed_coupon=guaranteed_coupon)

        # 2) add other market data
        index_curve_data = SpotRateInternalDataSource().initialize(SpotRateRequest(dt, dt, rate_index_ccy, rate_index))
        market.add_item(index_curve_data.get_market_key(), index_curve_data.get_market_item(dt))

        # check if fixing is required
        bbg_ticker_map = {"SOFRRATE": "SOFRRATE Index"}
        inverse_bbg_ticker_map = {v: k for k, v in bbg_ticker_map.items()}
        fixing_requirement = TradeReporter(autocallable).get_fixing_requirement(market)
        if len(fixing_requirement) > 0:
            # create fixing table from bbg
            fixings = []
            for fixing_req in fixing_requirement:
                und = bbg_ticker_map.get(fixing_req.underlying, fixing_req.underlying)
                fixing = datalake.get_bbg_history(
                    [und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                fixing["date"] = pd.to_datetime(fixing["date"]).dt.date
                fixing["ticker"] = fixing["ticker"].apply(lambda ticker: inverse_bbg_ticker_map.get(ticker, ticker))
                fixing = fixing.rename(columns={"PX_LAST": "fixing", "ticker": "underlying"}).set_index(
                    ["date", "underlying"])
                fixings.append(fixing)

            fixings = pd.concat(fixings).reset_index()
            fixings.loc[fixings.underlying == rate_index, "fixing"] /= 100
            fixing_table = FixingTable(fixings)
            market.add_item(market_utils.create_fixing_table_key(), fixing_table)

        # check if holiday is required
        holiday_requirement = TradeReporter(autocallable).get_holiday_requirement(market, EQAutoCallableNXValuer())
        hol_center = {}
        if len(holiday_requirement) > 0:
            for hol_req in holiday_requirement:
                holidays = get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
                code_hols = hol_center.get(hol_req.code, []) + holidays
                hol_center[hol_req.code] = sorted(set(code_hols))

        if len(hol_center) > 0:
            key = market_utils.create_holiday_center_key()
            market.add_item(key, HolidayCenter(hol_center))

        # 3) calculate PV using default DUPIRE model
        model_param_overrides = dict(
            sim_path=30000,
            calib_time_steps=400,
            pricing_time_steps=400,
        )
        [pv, funding_leg, eq_leg, delta, vega] = EQAutoCallableNXValuer(override_model_params=model_param_overrides).price(autocallable, market, calc_types=['price', 'funding_leg', 'eq_leg', 'delta', 'vega'])
        print(pv, funding_leg, eq_leg, delta, vega)
        self.assertAlmostEqual(pv, -3520.0354499036625)
        self.assertAlmostEqual(funding_leg, -62262.217650490195)
        self.assertAlmostEqual(eq_leg, 58742.182200611845)
        self.assertAlmostEqual(delta, -214.26617869698134)
        self.assertAlmostEqual(vega, 0)

    def test_single_underlying_long_date_bt(self):
        # 1) create trade
        und_list = ["SPX Index"]
        inception_date = datetime(2024, 4, 5)
        expiration = datetime(2026, 4, 2)
        swap_type = "AUTOCALL_SWAP"
        # load vol surface:
        dt = inception_date
        market = Market(base_datetime=inception_date)
        ref_time = 'NYC|CLOSE|{}'.format(inception_date.strftime('%Y%m%d'))
        und_map = get_underlyings_map(return_df=False)
        for und in und_list:
            und_id = und_map[und]
            surface = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time,
                                             underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
            surface = surface.override_num_regular(num_regular=252)
            market.add_item(market_utils.create_vol_surface_key(und), surface)
        start_spots = [market.get_spot(und) for und in und_list]
        currency = "USD"
        notional = 1000000
        call_put = "PUT"
        strike = 0.75
        lower_strike = 0
        exercise_type = "EUROPEAN"
        knock_in_barrier = 0.75
        knock_in_barrier_obs = "European"
        knock_in_put_strike = 0.75
        put_grearing = 1
        autocall_dates = []
        tmp = inception_date + relativedelta(months=6)
        while tmp < expiration:
            bd = bdc_adjustment(tmp)
            autocall_dates.append(bd)
            tmp = tmp + relativedelta(months=6)
        autocall_dates.append(expiration)
        print(f"ac dates: {' '.join([x.strftime('%Y-%m-%d') for x in autocall_dates])}")
        num_autocall_dates = len(autocall_dates)
        # low strike barrier
        autocall_barriers = [1.] * num_autocall_dates
        coupon_barrier = 0.0425
        coupon_dates = autocall_dates
        coupon_down = 0
        coupon_up = 0.0375
        coupon_is_memory = False    # no memory
        coupon_is_snowball = False
        float_dates = coupon_dates
        float_fixed_dates = float_dates
        float_start_date = inception_date
        funding_spread = 0
        # rate_index = "SOFRRATE"
        rate_index = "SWAP_SOFR" # source sofr curve from citi rather internal
        rate_index_ccy = "USD"
        glider_event = False
        guaranteed_coupon = 0.0
        autocallable = AutoCallable(
            und_list=und_list, type=swap_type, start_spots=start_spots, start_date=inception_date,
            expiration=expiration,
            currency=currency, notional=notional, call_put=call_put, strike=strike, lower_strike=lower_strike,
            exercise_type=exercise_type, knock_in_barrier=knock_in_barrier, knock_in_barrier_obs=knock_in_barrier_obs,
            knock_in_put_strike=knock_in_put_strike, put_grearing=put_grearing, autocall_dates=autocall_dates,
            autocall_barriers=autocall_barriers, coupon_barrier=coupon_barrier, coupon_dates=coupon_dates,
            coupon_down=coupon_down, coupon_up=coupon_up, coupon_is_memory=coupon_is_memory,
            coupon_is_snowball=coupon_is_snowball, float_dates=float_dates, float_fixed_dates=float_fixed_dates,
            float_start_date=float_start_date, funding_spread=funding_spread, rate_index=rate_index,
            rate_index_ccy=rate_index_ccy,
            glider_event=glider_event, guaranteed_coupon=guaranteed_coupon)

        # 2) add other market data
        if rate_index == "SOFRRATE":
            index_curve_data = SpotRateInternalDataSource().initialize(SpotRateRequest(dt, dt, rate_index_ccy, rate_index))
        elif rate_index == "SWAP_SOFR":
            index_curve_data = SpotRateCitiDataSource().initialize(SpotRateRequest(dt, dt, rate_index_ccy, rate_index))
        else:
            raise Exception(f"Unexpected rate index {rate_index}")
        market.add_item(index_curve_data.get_market_key(), index_curve_data.get_market_item(dt))

        # check if fixing is required
        bbg_ticker_map = {"SOFRRATE": "SOFRRATE Index", "SWAP_SOFR": "SOFRRATE Index"}
        inverse_bbg_ticker_map = {v: k for k, v in bbg_ticker_map.items() if k == rate_index}
        fixing_requirement = TradeReporter(autocallable).get_fixing_requirement(market)
        if len(fixing_requirement) > 0:
            # create fixing table from bbg
            fixings = []
            for fixing_req in fixing_requirement:
                und = bbg_ticker_map.get(fixing_req.underlying, fixing_req.underlying)
                fixing = datalake.get_bbg_history(
                    [und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                fixing["date"] = pd.to_datetime(fixing["date"]).dt.date
                fixing["ticker"] = fixing["ticker"].apply(lambda ticker: inverse_bbg_ticker_map.get(ticker, ticker))
                fixing = fixing.rename(columns={"PX_LAST": "fixing", "ticker": "underlying"}).set_index(
                    ["date", "underlying"])
                fixings.append(fixing)

            fixings = pd.concat(fixings).reset_index()
            fixings.loc[fixings.underlying == rate_index, "fixing"] /= 100
            fixing_table = FixingTable(fixings)
            market.add_item(market_utils.create_fixing_table_key(), fixing_table)

        # check if holiday is required
        holiday_requirement = TradeReporter(autocallable).get_holiday_requirement(market, EQAutoCallableNXValuer())
        hol_center = {}
        if len(holiday_requirement) > 0:
            for hol_req in holiday_requirement:
                holidays = get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
                code_hols = hol_center.get(hol_req.code, []) + holidays
                hol_center[hol_req.code] = sorted(set(code_hols))

        if len(hol_center) > 0:
            key = market_utils.create_holiday_center_key()
            market.add_item(key, HolidayCenter(hol_center))

        # test solve coupon
        coupon, y = EQAutoCallableNXValuer().solve_coupon_up(autocallable, market)
        print(coupon)
        self.assertAlmostEqual(coupon, 0.0374)

        # 3) calculate PV using default DUPIRE model
        model_param_overrides = dict(
            sim_path=30000,
            calib_time_steps=400,
            pricing_time_steps=400,
        )
        [pv, funding_leg, eq_leg, delta, vega] = EQAutoCallableNXValuer(override_model_params=model_param_overrides).price(autocallable, market, calc_types=['price', 'funding_leg', 'eq_leg', 'delta', 'vega'])
        print(pv, funding_leg, eq_leg, delta, vega)
        self.assertAlmostEqual(pv, 996.914434674012)

    def test_three_underlying(self):
        # 1) create trade
        inception_date = datetime(2022, 2, 8)
        dt = inception_date
        und_list = ["SPX Index", "RTY Index", "NDX Index"]
        # load vol surface:
        market = Market(base_datetime=dt)
        ref_time = 'NYC|CLOSE|{}'.format(dt.strftime('%Y%m%d'))
        und_map = get_underlyings_map(return_df=False)
        for und in und_list:
            und_id = und_map[und]
            surface = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time, underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
            surface = surface.override_num_regular(num_regular=252)
            market.add_item(market_utils.create_vol_surface_key(und), surface)
        type = "AUTOCALL_SWAP"
        start_spots = [market.get_spot(und) for und in und_list]
        expiration = datetime(2024, 2, 8)
        currency = "USD"
        notional = 1000000
        call_put = "PUT"
        strike = 1
        lower_strike = 0
        exercise_type = "EUROPEAN"
        knock_in_barrier = 0.6
        knock_in_barrier_obs = "European"
        knock_in_put_strike = 1
        put_grearing = 1
        autocall_dates = []
        tmp = inception_date + relativedelta(months=6)
        while tmp < expiration:
            bd = bdc_adjustment(tmp)
            autocall_dates.append(bd)
            tmp = tmp + relativedelta(months=6)
        autocall_dates.append(expiration)
        print(autocall_dates)
        num_autocall_dates = len(autocall_dates)
        autocall_barriers = [1] * num_autocall_dates
        coupon_barrier = 0.6
        coupon_dates = autocall_dates
        coupon_down = 0
        coupon_up = 0.066/2
        coupon_is_memory = False
        coupon_is_snowball = False
        float_dates = coupon_dates
        float_fixed_dates = float_dates
        float_start_date = inception_date
        funding_spread = 0.0036
        rate_index = "SWAP_SOFR"
        rate_index_ccy = "USD"
        glider_event = False
        guaranteed_coupon = 0.0
        autocallable = AutoCallable(
            und_list=und_list, type=type, start_spots=start_spots, start_date=inception_date, expiration=expiration,
            currency=currency, notional=notional, call_put=call_put, strike=strike, lower_strike=lower_strike,
            exercise_type=exercise_type, knock_in_barrier=knock_in_barrier, knock_in_barrier_obs=knock_in_barrier_obs,
            knock_in_put_strike=knock_in_put_strike, put_grearing=put_grearing, autocall_dates=autocall_dates,
            autocall_barriers=autocall_barriers, coupon_barrier=coupon_barrier, coupon_dates=coupon_dates,
            coupon_down=coupon_down, coupon_up=coupon_up, coupon_is_memory=coupon_is_memory,
            coupon_is_snowball=coupon_is_snowball, float_dates=float_dates, float_fixed_dates=float_fixed_dates,
            float_start_date=float_start_date, funding_spread=funding_spread, rate_index=rate_index,
            rate_index_ccy=rate_index_ccy, glider_event=glider_event, guaranteed_coupon=guaranteed_coupon)

        # 2) add other market data
        index_curve_data = SpotRateCitiDataSource().initialize(SpotRateRequest(dt, dt, rate_index_ccy, rate_index))
        market.add_item(index_curve_data.get_market_key(), index_curve_data.get_market_item(dt))

        # create correlation matrix
        correlation_shift = 0.05
        correlation_data = ConstantCorrelationMatrixDataSource().initialize(CorrelationMatrixDataRequest(dt, dt, und_list, correlation_shift=correlation_shift))
        market.add_item(correlation_data.get_market_key(), correlation_data.get_market_item(dt))

        # check if fixing is required
        bbg_ticker_map = {"SOFRRATE": "SOFRRATE Index", "SWAP_SOFR": "SOFRRATE Index"}
        inverse_bbg_ticker_map = {v: k for k, v in bbg_ticker_map.items()}
        fixing_requirement = TradeReporter(autocallable).get_fixing_requirement(market)
        if len(fixing_requirement) > 0:
            # create fixing table from bbg
            fixings = []
            for fixing_req in fixing_requirement:
                und = bbg_ticker_map.get(fixing_req.underlying, fixing_req.underlying)
                fixing = datalake.get_bbg_history(
                    [und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                fixing["date"] = pd.to_datetime(fixing["date"]).dt.date
                fixing["ticker"] = fixing["ticker"].apply(lambda ticker: inverse_bbg_ticker_map.get(ticker, ticker))
                fixing = fixing.rename(columns={"PX_LAST": "fixing", "ticker": "underlying"}).set_index(
                    ["date", "underlying"])
                fixings.append(fixing)

            fixings = pd.concat(fixings).reset_index()
            fixings.loc[fixings.underlying == rate_index, "fixing"] /= 100
            fixing_table = FixingTable(fixings)
            market.add_item(market_utils.create_fixing_table_key(), fixing_table)

        # check if holiday is required
        holiday_requirement = TradeReporter(autocallable).get_holiday_requirement(market, EQAutoCallableNXValuer())
        hol_center = {}
        if len(holiday_requirement) > 0:
            for hol_req in holiday_requirement:
                holidays = get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
                code_hols = hol_center.get(hol_req.code, []) + holidays
                hol_center[hol_req.code] = sorted(set(code_hols))

        if len(hol_center) > 0:
            key = market_utils.create_holiday_center_key()
            market.add_item(key, HolidayCenter(hol_center))

        # 3) calculate PV using default DUPIRE model
        pv = EQAutoCallableNXValuer().price(autocallable, market)
        px = pv/notional
        print(pv, px)
        self.assertAlmostEqual(px, -0.04678259856015997)