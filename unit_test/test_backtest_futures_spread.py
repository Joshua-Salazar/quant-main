import os
import pandas as pd
import unittest
from datetime import datetime
from ..tools import test_utils
from ..backtest.backtester import LocalBacktester
from ..backtest.strategies.futures_spread import FuturesSpread, FuturesSpreadState
from ..infrastructure.future_data_container import DatalakeBBGFuturesDataSource, FutureDataRequest
from ..infrastructure.fx_data_container import DatalakeBBGFXDataSource, FXDataRequest
from ..tradable.constant import Constant
from ..tradable.portfolio import Portfolio


class TestFuturesSpread(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFuturesSpread, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_futures_spread")

        self.rebase = False

    def test(self):

        test_name = "test"

        missing_dates = []
        missing_dates = list(set(missing_dates))
        missing_dates = [datetime.strptime(x, "%Y-%m-%d") for x in missing_dates]

        start_date = datetime(2022, 1, 4)
        end_date = datetime(2022, 12, 16)
        long_root = 'RX'
        short_root = 'IK'
        suffix = 'Comdty'
        calendar = ['XFRA'] + missing_dates
        currency = 'USD'

        # define instance of a strategy
        futures_spread_strategy = FuturesSpread(
            start_date=start_date,
            end_date=end_date,
            calendar=calendar,
            currency=currency,
            parameters={
                'long': {
                    'underlying': long_root,
                    'target_notional': 100,
                },
                'short': {
                    'underlying': short_root,
                    'target_notional': -100,
                },
                'roll_offset': 3,
                'tc_rate': 0.0005,
            },
            data_requests={
                'long': (
                    FutureDataRequest(start_date, end_date, calendar, long_root, suffix, 2),
                    DatalakeBBGFuturesDataSource()
                ),
                'short': (
                    FutureDataRequest(start_date, end_date, calendar, short_root, suffix, 2),
                    DatalakeBBGFuturesDataSource()
                ),
                'fx': (
                    FXDataRequest(start_date, end_date, calendar, currency='EUR', denominator_currency=currency, fixing_type='L160'),
                    DatalakeBBGFXDataSource()
                )
            }
        )

        # run backtest
        runner = LocalBacktester()
        results = runner.run(futures_spread_strategy, start_date, end_date, FuturesSpreadState(start_date, Portfolio([]), 0.0, 0.0))
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
                        'price': v.price #* v.fx
                    })
        portfolio_expanded = pd.DataFrame.from_records(portfolio_expanded)

        target_file = os.path.join(self.test_folder, f"{test_name}_portfolio_expanded.csv")
        test_utils.assert_dataframe(portfolio_expanded, target_file, self.rebase, ignore_cols=[])



