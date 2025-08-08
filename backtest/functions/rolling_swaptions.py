from datetime import datetime
import pandas as pd
from ...backtest.backtester import LocalBacktester
from ...backtest.strategies.rolling_swaptions import RollingSwaptions, RollingSwaptionsState
from ...infrastructure.df_curve_data_container import DFCurveRequest
from ...infrastructure.forward_rate_data_container import ForwardRateRequest
from ...infrastructure.rate_vol_data_container import RateVolRequest
from ...tradable.portfolio import Portfolio
from ...tradable.constant import Constant


def rolling_swaptions(
        start_date: datetime, end_date: datetime, calendar: list[str], currency: str,
        legs: dict, tranches: dict,
        tc_rate: float,
        rate_vols_data_source,
        fwd_rates_data_source,
        spot_rates_data_source,
        df_rates_data_source,
        df_rates_type="OIS",
        spot_rates_type="SWAP",
        forward_rates_type=None,
):
    missing_dates = ['2021-12-24', '2020-07-03', '2018-12-05', '2015-07-03', '2010-12-24', '2009-07-03', '2004-12-24',
                     '1999-12-24', '1999-01-18', '2001-09-11', '2001-09-11', '2001-09-12', '2001-09-12', '2004-06-11',
                     '2004-06-11', '2012-07-23', '2012-07-23', '2012-07-24', '2012-07-24', '2012-10-30', '2012-10-30',
                     '2012-11-20', '2012-11-20', '2012-11-21', '2012-11-21', '2012-11-23', '2012-11-23', '2013-03-08',
                     '2013-03-08', '2013-03-27', '2013-03-27', '2013-03-28', '2013-03-28', '2014-05-21', '2014-05-21',
                     '2014-05-22', '2014-05-22', '2014-05-23', '2014-05-23', '2016-12-27', '2016-12-27', '2018-02-21',
                     '2018-02-21', '2018-03-12', '2018-03-12', '2018-03-13', '2018-03-13', '2018-03-27', '2018-03-27',
                     '2018-07-23', '2018-07-23', '2018-07-24', '2018-07-24', '2018-11-20', '2018-11-20', '2019-04-02',
                     '2019-04-02', '2019-04-03', '2019-04-03', '2020-10-12', '2020-10-12', '2014-02-21', '2014-02-21',
                     '2014-06-30', '2014-06-30', '2017-11-21', '2017-11-21', '2017-11-22', '2017-11-22', '2017-11-28',
                     '2017-11-28', '2017-11-29', '2017-11-29', '2020-07-23', '2020-07-23', '2020-07-24', '2020-07-24',
                     '2022-06-20', '2022-06-20', '2002-11-26', '2002-11-26', '2003-05-28', '2003-05-28', '2003-11-25',
                     '2003-11-25', '2004-05-26', '2004-05-26', '2009-10-28', '2009-10-28', '2010-10-27', '2010-10-27',
                     '2017-09-27', '2017-09-27', '2018-09-26', '2018-09-26', '2019-03-27', '2019-03-27', '2021-10-27',
                     '2001-09-07', '2001-09-10', '2020-01-31']
    missing_dates = list(set(missing_dates))
    missing_dates = [datetime.strptime(x, "%Y-%m-%d") for x in missing_dates]

    if not isinstance(calendar, list):
        calendar = [calendar]

    rolling_swaptions_strategy = RollingSwaptions(
        start_date=start_date,
        end_date=end_date,
        calendar=calendar + missing_dates,
        currency=currency,
        parameters={
            'legs': legs,
            'tranches': tranches,
            'tc_rate': tc_rate,
            'df_rates_type': df_rates_type,
        },
        data_requests={
            'vol_cube': (
                RateVolRequest(start_date, end_date, currency),
                rate_vols_data_source
                # JPRateVolDataSource(rate_data_file=rate_data_file, vol_data_file=vol_data_file)
            ),
            'atmf_yields': (
                ForwardRateRequest(start_date, end_date, currency, forward_rates_type),
                fwd_rates_data_source,
                # JPForwardRateDataSource(data_file=rate_data_file)
            ),
            'spot_rates': (
                DFCurveRequest(start_date, end_date, currency, spot_rates_type),
                spot_rates_data_source,
                # JPDFCurveDataSource(data_file=rate_data_file)
            ),
            'df_curve': (
                DFCurveRequest(start_date, end_date, currency, df_rates_type),
                df_rates_data_source
                # JPDFCurveDataSource(data_file=rate_data_file)
            ),
        }
    )

    # run backtest
    runner = LocalBacktester()
    results = runner.run(rolling_swaptions_strategy, start_date, end_date,
                         RollingSwaptionsState(start_date, Portfolio([]), 0.0, 0.0, None))

    records = [{'date': x.time_stamp, 'pnl': x.price} for x in results]
    pnl_series = pd.DataFrame.from_dict(records)

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

    return results, pnl_series, portfolio_expanded
