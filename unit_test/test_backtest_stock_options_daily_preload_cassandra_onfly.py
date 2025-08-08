import unittest
from ..tools import test_utils
import os
from datetime import datetime
import pandas as pd

from ..analytics.symbology import option_root_from_ticker
from ..backtest.tranche import MonthlyTranche, WeeklyTranche, DailyTranche
from ..infrastructure.eq_vol_data_container import VolaEqVolDataSource
from ..infrastructure.option_data_container import CassandraDSPreLoadFromPickle,CassandraDSPreLoadFromCacheByYear,CassandraDSPreLoadFromCacheByDay,CassandraDSOnDemandFromCacheByYear
from ..tradable.constant import Constant
from ..backtest.functions.stock_options_daily import stock_options_daily


TEST_CASES = [
    [datetime(2007, 1, 3), datetime(2009, 3, 10), ['XCBO'], 'SPX Index', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2022, 1, 3), datetime(2023, 3, 10), ['XCBO'], 'VIX Index', 0, 'strike', 0.75, 'weekly', True],
    [datetime(2022, 1, 3), datetime(2023, 3, 10), ['XCBO'], 'RTY Index', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2022, 11, 20), datetime(2023, 3, 10), ['XOSE'], 'NKY Index', 0, 'strike', 0.75, 'weekly', True],
    [datetime(2022, 1, 3), datetime(2023, 3, 10), ['XFRA'], 'SX5E Index', 0, 'strike', 0.75, 'daily', True],
    [datetime(2022, 6, 10), datetime(2023, 3, 10), ['XCBO'], 'UUP US Equity', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2019, 5, 9), datetime(2021, 3, 10), ['XCBO', datetime(2020, 4, 29)], 'USO US Equity', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2022, 1, 2), datetime(2023, 3, 10), ['XCBO'], 'TLT US Equity', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2022, 8, 20), datetime(2023, 3, 10), ['XCBO'], 'HYG US Equity', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2013, 1, 3), datetime(2015, 3, 10), ['XCBO'], 'QQQ US Equity', 0, 'strike', 0.75, 'monthly', True],
    [datetime(2022, 3, 1), datetime(2023, 3, 10), ['XCBO'], 'TQQQ US Equity', 0, 'strike', 0.75, 'monthly', True],
]


class TestStockOptionsDaily(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestStockOptionsDaily, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_stock_options_daily_onfly")

        self.rebase = False

    def test_Single_Underlying(self):
        for item in TEST_CASES:
            print(item)
            start_date = item[0]
            end_date = item[1]
            calendar = item[2]
            underlying = item[3]
            allow_fill_forward_missing_data = item[4]

            selection_by = item[5]
            selection_value = item[6]
            tranche_type = item[7]

            use_listed = item[8]

            test_name = f"test_Single_Underlying {underlying}"

            hedged = True

            if tranche_type == 'monthly':
                tranche = MonthlyTranche(start_date, end_date, 3, 4, 3, 'previous', calendar, trade_first_day=True)
            elif tranche_type == 'weekly':
                tranche = WeeklyTranche(start_date, end_date, 4, 3, 'previous', calendar, trade_first_day=True)
            elif tranche_type == 'daily':
                tranche = DailyTranche(start_date, end_date, 3, 'previous', calendar, trade_first_day=True)
            else:
                raise RuntimeError(f"unknown tranche type {tranche_type}")
            legs = {
                'leg': {
                    'underlying': underlying,
                    'tenor': '3M',
                    'selection_by': selection_by,
                    selection_by: selection_value,
                    'type': 'P',
                    'other_filters': [lambda x: x[x['price'] >= 0.05]],
                    'sizing_measure': 'price',
                    'sizing_target': 100 / 4,
                    'tranche': tranche
                },
            }
            if use_listed:
                option_data_source = CassandraDSOnDemandFromCacheByYear()
            else:
                option_data_source = VolaEqVolDataSource()

            results, nav_series, portfolio_expanded = stock_options_daily(
                start_date, end_date, calendar, legs, hedged=hedged, max_option_expiry_days=100,
                option_data_source=option_data_source,
                allow_fill_forward_missing_data=allow_fill_forward_missing_data,
                use_listed=use_listed,
            )

            records = [{'date': x.time_stamp, 'nav': x.price} for x in results]
            nav_series = pd.DataFrame.from_dict(records)

            target_file = os.path.join(self.test_folder, f"{test_name}_nav_series.csv")
            test_utils.assert_dataframe(nav_series, target_file, self.rebase, ignore_cols=[])

            portfolio_expanded = []
            for state in results:
                for k, v in state.portfolio.net_positions().items():
                    if not isinstance(v.tradable, Constant):
                        portfolio_expanded.append({
                            'date': state.time_stamp,
                            'position': k,
                            'quantity': v.quantity,
                            'price': v.price
                        })
            portfolio_expanded = pd.DataFrame.from_records(portfolio_expanded)

            # target_file = os.path.join(self.test_folder, f"{test_name}_portfolio_expanded.csv")
            # test_utils.assert_dataframe(portfolio_expanded, target_file, self.rebase, ignore_cols=[])

