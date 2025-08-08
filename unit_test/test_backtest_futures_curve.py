import unittest
from ..tools import test_utils
import os
from datetime import datetime, time
import pandas as pd
from ..backtest.strategies.futures_curve import FuturesCurve, FuturesCurveState
from ..infrastructure.future_data_container import DatalakeBBGFuturesDataSource, FutureDataRequest
from ..infrastructure.fx_data_container import DatalakeBBGFXDataSource, FXDataRequest
from ..backtest.backtester import LocalBacktester
from ..tradable.portfolio import Portfolio
from ..tradable.constant import Constant
from ..dates.utils import minus_tenor


class TestFuturesCurve(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFuturesCurve, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_futures_curve")

        self.rebase = False

    def test(self):

        test_name = "test"
        missing_dates = ['2017-1-2', '2023-1-2', '2022-9-29', '2022-9-30', '2022-10-3', '2022-10-4', '2022-10-5', '2022-10-6' ]
        missing_dates = list(set(missing_dates))
        missing_dates = [datetime.strptime(x, "%Y-%m-%d") for x in missing_dates]

        start_date = datetime(2023, 1, 1)
        end_date = datetime( 2023, 2, 7 )
        while end_date.weekday() > 4:
            end_date = minus_tenor(end_date, "2D")
        end_date = datetime.combine(end_date, time(0, 0))

        und = 'VHO'
        calendar = ['XFRA'] + missing_dates
        currency = 'USD'

        # equal notional
        target_fut_short = 1
        target_fut_long = 2

        # define instance of a strategy
        futures_spread_strategy = FuturesCurve(
            start_date=start_date,
            end_date=end_date,
            calendar=calendar,
            currency=currency,
            parameters={
                'long': {
                    'underlying': und,
                    'target_future': target_fut_long,
                    'target_notional': 250,

                },
                'short': {
                    'underlying': und,
                    'target_future': target_fut_short,
                    'target_notional': -100,
                },
                'roll_offset': 3,
                'tc_rate': 0.0005,
                'skip_months': ['H', 'M', 'U'],
            },
            data_requests={
                'underlier': (
                    FutureDataRequest(start_date, end_date, calendar, und, 'Index', 6 * (target_fut_short + target_fut_long),
                                      skip_months=['H', 'M', 'U']),
                    DatalakeBBGFuturesDataSource()
                ),
                'fx': (
                    FXDataRequest(start_date, end_date, calendar, currency='EUR', denominator_currency=currency,
                                  fixing_type='L160'),
                    DatalakeBBGFXDataSource()
                )
            }

        )
        # run backtest
        runner = LocalBacktester()
        # sometime Datalake throws an error => re-run
        fails = 0
        while fails < 4:
            try:
                results = runner.run(futures_spread_strategy, start_date, end_date,
                                     FuturesCurveState(start_date, Portfolio([]), 0.0, 0.0))
                fails = 5
            except:
                fails += 1
                print('fails: %d' % fails)

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

