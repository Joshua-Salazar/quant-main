import os
import pandas as pd
import unittest
from datetime import datetime
from ..tools import test_utils
from ..backtest.strategies.rolling_swaptions import RollingSwaptions, RollingSwaptionsState
from ..backtest.backtester import LocalBacktester
from ..infrastructure.df_curve_data_container import DFCurveRequest, JPDFCurveDataSource, DFCurveJPDataSource
from ..infrastructure.forward_rate_data_container import ForwardRateRequest, JPForwardRateDataSource
from ..infrastructure.rate_vol_data_container import JPRateVolDataSource, RateVolRequest
from ..tradable.portfolio import Portfolio


class TestRollingSwaptions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRollingSwaptions, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_rolling_swaptions")
        self.src_folder = test_utils.get_test_src_folder()

        self.rebase = False

    def test(self):

        test_name = "test"

        missing_dates = ['2021-12-24', '2020-07-03', '2018-12-05', '2015-07-03', '2010-12-24', '2009-07-03', '2004-12-24', '1999-12-24', '1999-01-18', '2001-09-11', '2001-09-11', '2001-09-12', '2001-09-12', '2004-06-11', '2004-06-11', '2012-07-23', '2012-07-23', '2012-07-24', '2012-07-24', '2012-10-30', '2012-10-30', '2012-11-20', '2012-11-20', '2012-11-21', '2012-11-21', '2012-11-23', '2012-11-23', '2013-03-08', '2013-03-08', '2013-03-27', '2013-03-27', '2013-03-28', '2013-03-28', '2014-05-21', '2014-05-21', '2014-05-22', '2014-05-22', '2014-05-23', '2014-05-23', '2016-12-27', '2016-12-27', '2018-02-21', '2018-02-21', '2018-03-12', '2018-03-12', '2018-03-13', '2018-03-13', '2018-03-27', '2018-03-27', '2018-07-23', '2018-07-23', '2018-07-24', '2018-07-24', '2018-11-20', '2018-11-20', '2019-04-02', '2019-04-02', '2019-04-03', '2019-04-03', '2020-10-12', '2020-10-12', '2014-02-21', '2014-02-21', '2014-06-30', '2014-06-30', '2017-11-21', '2017-11-21', '2017-11-22', '2017-11-22', '2017-11-28', '2017-11-28', '2017-11-29', '2017-11-29', '2020-07-23', '2020-07-23', '2020-07-24', '2020-07-24', '2022-06-20', '2022-06-20', '2002-11-26', '2002-11-26', '2003-05-28', '2003-05-28', '2003-11-25', '2003-11-25', '2004-05-26', '2004-05-26', '2009-10-28', '2009-10-28', '2010-10-27', '2010-10-27', '2017-09-27', '2017-09-27', '2018-09-26', '2018-09-26', '2019-03-27', '2019-03-27', '2021-10-27', '2001-09-07', '2001-09-10']
        missing_dates = list(set(missing_dates))
        missing_dates = [datetime.strptime(x, "%Y-%m-%d") for x in missing_dates]

        start_date = datetime(2021, 1, 4)
        end_date = datetime(2022, 12, 16)

        # define instance of a strategy
        rolling_swaptions_strategy = RollingSwaptions(
            start_date=start_date,
            end_date=end_date,
            calendar=['NYC', 'LON'] + missing_dates,
            currency='USD',
            parameters={
                'legs': {
                    'leg1': {
                        'currency': 'USD',
                        'style': 'Payer',
                        'expiry': '1Y',
                        'strike': 100,
                        'strike_type': 'spot', # spot or forward
                        'tenor': '15y', # lower case years
                        'sizing': -10000000/4.0,
                        'delta_hedge': "Unhedged"
                    },
                    },
                'tranches': {
                    'tranche1': {
                        'roll_months': [3],
                        'roll_day': 31},
                    'tranche2': {
                        'roll_months': [6],
                        'roll_day': 31},
                    'tranche3': {
                        'roll_months': [9],
                        'roll_day': 31},
                    'tranche4': {
                        'roll_months': [12],
                        'roll_day': 31},
                },
                'tc_rate': 0.00,
            },
            data_requests={
                'vol_cube': (
                    RateVolRequest(start_date, end_date, 'USD'),
                    JPRateVolDataSource()
                ),
                'atmf_yields': (
                    ForwardRateRequest(start_date, end_date, 'USD'),
                    JPForwardRateDataSource()
                ),
                'spot_rates': (
                    DFCurveRequest(start_date, end_date, 'USD', 'SWAP'),
                    DFCurveJPDataSource()
                ),
                'df_curve': (
                    DFCurveRequest(start_date, end_date, 'USD', 'OIS'),
                    DFCurveJPDataSource()
                ),
            }
        )

        # run backtest
        runner = LocalBacktester()
        results = runner.run(rolling_swaptions_strategy, start_date, end_date,
                             RollingSwaptionsState(start_date, Portfolio([]), 0.0, 0.0, None))

        records = [{'date': x.time_stamp, 'nav': x.price} for x in results]
        nav_series = pd.DataFrame.from_dict(records)

        target_file = os.path.join(self.test_folder, f"{test_name}_nav_series.csv")
        test_utils.assert_dataframe(nav_series, target_file, self.rebase, ignore_cols=[])
