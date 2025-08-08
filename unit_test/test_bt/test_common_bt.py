import unittest
from datetime import datetime
from ...tools import test_utils
from ...analytics.symbology import option_calendar_from_ticker
from ...backtest_nb.common_bt_factory import CommonBTFactory
from ...backtest_nb.common_bt_cr_options_config import *
from ...backtest_nb.common_bt_future_options_config import TENOR_DICT, IV_DICT, COST_DICT
from ...tools.test_utils import get_temp_folder
from ...backtest.indicator import Indicator
from ...backtest.tranche import MonthlyTranche, WeeklyTranche, DailyTranche, RollingAtExpiryTranche
import os

from ...tradable.forwardstartcds import ForwardStartCDS
from ...tradable.option import Option
import warnings


class TestCommonBT(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCommonBT, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_common_bt")

        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_cr_options_base_case(self):

        test_name = "test_cr_options_base_case"
        st = datetime(2024, 4, 29)
        et = datetime(2025, 7, 11)
        asset = "CDX_NA_IG"
        ccy = "USD"
        params = dict(asset=asset)
        params['legs'] = {
            'leg1': {
                'currency': 'USD',
                'underlying': asset,
                'type': 'C',
                'tenor': '3M',
                'strike': 0.5,
                'sizing': {'quantity': 100},
                'hedged': True,
            }
        }
        params['tranches'] = {
            'M1': {
                'initial_tenor': '3M',
            },
        }
        params['price_absolute_costs'] = {
            Option: 1 / 10000,
            ForwardStartCDS: 0.1 / 10000
        }

        bt = CommonBTFactory().create(leg_type="cr_options", start_date=st, end_date=et, currency=ccy,
                                      parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_stock_options_daily_base_case(self):

        test_name = "test_stock_options_daily_base_case"
        st = datetime(2022, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "SPX Index"
        ccy = "USD"
        params = dict(asset=asset)

        calendar = option_calendar_from_ticker(asset)
        tranche_type = 'monthly'
        if tranche_type == 'monthly':
            tranche = MonthlyTranche(st, et, 3, 4, 3, 'previous', calendar, trade_first_day=True)
        elif tranche_type == 'weekly':
            tranche = WeeklyTranche(st, et, 4, 3, 'previous', calendar, trade_first_day=True)
        elif tranche_type == 'daily':
            tranche = DailyTranche(st, et, 3, 'previous', calendar, trade_first_day=True)
        else:
            raise RuntimeError(f"unknown tranche type {tranche_type}")

        params["legs"] = {
            'leg': {
                'underlying': asset,
                'expiry': '3M',
                'strike_type': 'delta',
                'strike': 0.4,
                'type': 'C',
                'sizing_measure': 'price',
                'sizing_target': 100 / 4,
                "hedge": True,
                'tranche': tranche
            },
        }
        params["tc_factor"] = 1

        bt = CommonBTFactory().create(leg_type="stock_options_daily", start_date=st, end_date=et, currency=ccy,
                                      parameters=params, force_run=False,
                                      replay=True, replay_file="test_stock_options_daily_replay.py")
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_stock_options_daily_rolling_at_exp_tranche(self):

        test_name = "test_stock_options_daily_rolling_at_exp_tranche"
        st = datetime(2022, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "SPX Index"
        ccy = "USD"
        params = dict(asset=asset)

        params["legs"] = {
            'leg': {
                'underlying': asset,
                'expiry': '3M',
                'strike_type': 'delta',
                'strike': 0.4,
                'type': 'C',
                'sizing_measure': 'price',
                'sizing_target': 100 / 4,
                "hedge": True,
                "tranche": "RollingAtExpiryTranche",
            },
        }

        bt = CommonBTFactory().create(leg_type="stock_options_daily", start_date=st, end_date=et, currency=ccy,
                                      parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_stock_options_daily_unwind(self):

        test_name = self._testMethodName
        st = datetime(2022, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "SPX Index"
        ccy = "USD"
        params = dict(asset=asset)

        params["legs"] = {
            'leg': {
                'underlying': asset,
                'expiry': '3M',
                'strike_type': 'delta',
                'strike': 0.4,
                'type': 'C',
                'sizing_measure': 'price',
                'sizing_target': 100 / 4,
                "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="delta", one_range=(0, 0.02)),
            },
        }

        bt = CommonBTFactory().create(leg_type="stock_options_daily", start_date=st, end_date=et, currency=ccy,
                                      parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_stock_options_daily_legacy_params(self):

        test_name = "test_stock_options_daily_legacy_params"
        st = datetime(2022, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "SPX Index"
        ccy = "USD"
        tenor = '3M'
        selection_by = 'delta'
        selection_value = 0.4
        option_type = 'C'
        sizing_measure = 'price'
        sizing_target = 100 / 4

        params = {}

        calendar = option_calendar_from_ticker(asset)
        tranche_type = 'monthly'
        if tranche_type == 'monthly':
            tranche = MonthlyTranche(st, et, 3, 4, 3, 'previous', calendar, trade_first_day=True)
        elif tranche_type == 'weekly':
            tranche = WeeklyTranche(st, et, 4, 3, 'previous', calendar, trade_first_day=True)
        elif tranche_type == 'daily':
            tranche = DailyTranche(st, et, 3, 'previous', calendar, trade_first_day=True)
        else:
            raise RuntimeError(f"unknown tranche type {tranche_type}")

        params["legs"] = {
            'leg': {
                'underlying': asset,
                'tenor': tenor,
                'selection_by': selection_by,
                selection_by: selection_value,
                'type': option_type,
                'sizing_measure': sizing_measure,
                'sizing_target': sizing_target,
                'tranche': tranche
            },
        }

        bt = CommonBTFactory().create(leg_type="stock_options_daily", start_date=st, end_date=et, currency=ccy,
                                      parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_future_option_base_case(self):
        test_name = "test_future_option_base_case"
        st = datetime(2022, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "TY"
        ccy = "USD"
        vega_size = 1
        params = dict(asset=asset)
        params["legs"] = {
            "call": {"type": "C", "strike_type": "delta", "strike": 0.4, "expiry": "1M",
                     "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                     "tranche": "RollingAtExpiryTranche",
                     "unwind": Indicator(name="delta", one_range=(0, 0.02)),
                     }
        }
        params["tc_factor"] = 1
        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_future_option_2d(self):
        test_name = "test_future_option_2d"
        st = datetime(2020, 3, 1)
        et = datetime(2021, 1, 1)
        ccy = "USD"
        params = dict(currency=ccy)
        params["legs"] = {
            "option1": {"type": "P", "strike_type": "delta", "strike": -0.5, "expiry": "2D",
                        "sizing_measure": "vega", "sizing_target": -0.01, "hedge": True,
                        "tranche": "RollingAtExpiryTranche", "fut_tgt_tenor": "45d", "fut_min": "45d"}
        }
        params["asset"] = "TY"
        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_future_option_weekly(self):
        test_name = "test_future_option_weekly"
        st = datetime(2024, 6, 7)
        et = datetime(2024, 6, 20)
        ccy = "USD"
        params = dict(currency=ccy)
        trade_first_day = False
        tranche = WeeklyTranche(st, et, 2, 1, 'previous', calendar="FCBT-CME", trade_first_day=trade_first_day)
        params["legs"] = {
            "option1": {"type": "P", "strike_type": "atmf", "strike": 0.85, "expiry": "1W",
                        "sizing_measure": "units", "sizing_target": 1, "hedge": False,
                        "tranche": tranche}
        }
        params["asset"] = "TY"
        params["weekly_expiry_filter"] = ["Wed"]
        params["skip_future"] = True
        params["trade_first_day"] = trade_first_day
        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_future_option_daily_selling(self):
        test_name = "test_future_option_daily_selling"
        st = datetime(2025, 1, 6)
        et = datetime(2025, 4, 22)
        asset = "TY"
        ccy = "USD"
        vega_size = -1
        params = dict(asset=asset)
        params["legs"] = {
            "call": {"type": "C", "strike_type": "delta", "strike": 0.4, "expiry": "1M",
                     "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                     "tranche": "RollingAtExpiryDailyTranche",
                     "unwind": Indicator(name="delta", one_range=(0, 0.02)),
                     }
        }

        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_custom_future_tenor(self):
        st = datetime(2025, 1, 4)
        et = datetime(2025, 4, 22)
        asset = "TY"
        ccy = "USD"
        vega_size = 1
        params = dict(asset=asset)
        params["legs"] = {
            "call": {"type": "P", "strike_type": "delta", "strike": -0.4, "expiry": "2M", "fut_tgt_tenor": "45D", "fut_min": "45D",
                     "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                     "tranche": "RollingAtExpiryTranche",
                     "unwind": Indicator(name="delta", one_range=(0, 0.02)),
                     }
        }

        df = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False).run().get_nav()
        print(df)

    def test_swaption_base_case(self):
        test_name = "test_swaption_base_case"
        st = datetime(2022, 1, 4)
        et = datetime(2024, 12, 31)
        currency = "USD"
        vega_size = -1
        asset = "SWAP_SOFR.1Y"
        params = dict(currency=currency, asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "atmf", "strike": 15, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="deltapct", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = True
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="swaption", start_date=st, end_date=et, currency=currency, parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_swaption_delta_strike(self):
        test_name = "test_swaption_delta_strike"
        st = datetime(2022, 1, 4)
        et = datetime(2022, 1, 4)
        currency = "USD"
        vega_size = -1
        asset = "SWAP_SOFR.1Y"
        params = dict(currency=currency, asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "delta", "strike": 0.4, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="deltapct", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = True
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="swaption", start_date=st, end_date=et, currency=currency, parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_pfo_df()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_swaption_jpy(self):
        test_name = "test_swaption_jpy"
        st = datetime(2021, 2, 1)
        et = datetime(2021, 2, 11)
        currency = "JPY"
        vega_size = -1
        asset = "SWAP_OIS.1Y"
        params = dict(currency=currency, asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "atmf", "strike": 15, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="deltapct", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = False
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="swaption", start_date=st, end_date=et, currency=currency, parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_fx_option_base_case(self):
        test_name = "test_fx_option_base_case"
        st = datetime(2022, 1, 4)
        et = datetime(2022, 6, 30)
        vega_size = -1
        asset = "USDJPY"
        params = dict(asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "delta", "strike": 0.25, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="delta", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = False
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="fx_option", start_date=st, end_date=et, currency=asset[3:], parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_fx_option_daily_selling(self):
        test_name = "test_fx_option_daily_selling"
        st = datetime(2022, 1, 4)
        et = datetime(2022, 6, 30)
        vega_size = -1
        asset = "AUDJPY"
        params = dict(asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "delta", "strike": 0.25, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryDailyTranche",
                "unwind": Indicator(name="delta", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = False
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="fx_option", start_date=st, end_date=et, currency=asset[3:], parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_fx_option_audjpy(self):
        test_name = "test_fx_option_audjpy"
        st = datetime(2022, 1, 4)
        et = datetime(2022, 1, 30)
        vega_size = -1
        asset = "AUDJPY"
        params = dict(asset=asset)
        params["legs"] = {
            "call": {
                "type": "C", "strike_type": "delta", "strike": 0.25, "expiry": "1M",
                "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
                "tranche": "RollingAtExpiryTranche",
                "unwind": Indicator(name="delta", one_range=(0, 0.02)),
            },
        }
        params["tc_factor"] = 1
        cache_market_data = False
        data_cache_path = get_temp_folder()
        bt_base = CommonBTFactory().create(leg_type="fx_option", start_date=st, end_date=et, currency=asset[3:], parameters=params, force_run=False, cache_market_data=cache_market_data, data_cache_path=data_cache_path)
        bt_base.run()

        actual_res = bt_base.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_future_option_legacy_params(self):
        test_name = "test_future_option_legacy_params"
        st = datetime(2025, 1, 6)
        et = datetime(2025, 4, 22)
        asset = "TY"
        # ccy = "USD"
        # vega_size = -1
        # params = dict(asset=asset)
        # params["legs"] = {
        #     "call": {"type": "C", "strike_type": "delta", "strike": 0.4, "expiry": "1M",
        #              "sizing_measure": "revega", "sizing_target": vega_size, "hedge": True,
        #              "tranche": "RollingAtExpiryDailyTranche",
        #              "unwind": Indicator(name="delta", one_range=(0, 0.02)),
        #              }
        # }

        from ...tradable.future import Future
        from ...tradable.option import Option
        from ...valuation.future_data_valuer import FutureDataValuer, FutureDataIntradayValuer
        from ...valuation.option_data_valuer import OptionDataValuer_Zero_Px
        from ...valuation.future_option_black76 import FutureOptionIntradayBlack76Valuer

        root = "TY"
        currency = 'USD'
        used_tenor = TENOR_DICT[root]['fut_tgt']
        used_min_tenor = TENOR_DICT[root]['fut_min']
        option_tenor = TENOR_DICT[root]['opt_tgt']
        hedge_intraday = True
        price_overrides = {}
        parameters = {
            'legs': {
                'Call_1': {
                    'root': root, 'tenor': used_tenor, 'min_tenor': used_min_tenor, 'type': 'C', 'delta': 0.40,
                    'sizing_measure': 'vega', 'sizing_target': -0.50, 'hedge': True, 'hedge_intraday': hedge_intraday,
                    'target_option_tenor': option_tenor,
                    'days_before_expiry': 1,
                    # 'tranche': custom_tranche # MonthlyTranche(start_date, end_date, 2, 1, 1, 'previous', calendar),
                },
                'Put_1': {
                    'root': root, 'tenor': used_tenor, 'min_tenor': used_min_tenor, 'type': 'P', 'delta': -0.40,
                    'target_option_tenor': option_tenor,
                    'sizing_measure': 'vega', 'sizing_target': -0.50, 'hedge': True, 'hedge_intraday': hedge_intraday,
                    'days_before_expiry': 1,
                    # 'tranche': custom_tranche # MonthlyTranche(start_date, end_date, 2, 1, 1, 'previous', calendar),
                },
            },
            'roll_style': 'expiry',
            'valuer_map': {
                Option: OptionDataValuer_Zero_Px(otm_threshold=-0.0, itm_threshold=0.125, raise_if_zero_px=True,
                                                 underlying_valuer=FutureDataValuer(price_name='price',
                                                                                    imply_delta_from_spot=False)),
                # OptionDataValuer(),
                Future: FutureDataValuer(price_name='price', imply_delta_from_spot=False, overrides=price_overrides),
            },
            'intraday_valuer_map': {
                Option: FutureOptionIntradayBlack76Valuer(),
                Future: FutureDataIntradayValuer(price_name='close'),
            },
            'iv_ticker_for_trigger': IV_DICT[root],
            'flat_vega_charge_per_unit_delta_charge': COST_DICT[root],
            'allow_fill_forward_missing_data': 0,
            'trade_first_day': True
        }

        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=currency, parameters=parameters, force_run=False)
        bt.run()

        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_bad_price(self):
        test_name = "test_bad_price"
        # bad price (0) on 2025-01-09
        st = datetime(2025, 1, 1)
        et = datetime(2025, 1, 13)
        ccy = "USD"
        newParams = dict(currency=ccy)
        newParams["legs"] = {
            "option1": {"type": "P", "strike_type": "atmf", "strike": 0.85, "expiry": "1M",
                        "sizing_measure": "units", "sizing_target": 1, "hedge": False,
                        "tranche": "RollingAtExpiryTranche", }
        }
        newParams["asset"] = "ES"
        bt = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=newParams, force_run=True).run()
        actual_res = bt.get_strategy_daily_returns()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, True)
