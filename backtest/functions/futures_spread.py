import pandas as pd
from ...backtest.strategies.futures_spread import FuturesSpread, FuturesSpreadState
from ...backtest.backtester import LocalBacktester
from ...infrastructure.future_data_container import DatalakeBBGFuturesDataSource, FutureDataRequest
from ...infrastructure.fx_data_container import DatalakeBBGFXDataSource, FXDataRequest
from ...tradable.portfolio import Portfolio
from ...tradable.constant import Constant


def futures_spread(start_date, end_date, calendar, currency,
                   long_root, long_suffix, long_currency, long_notional,
                   short_root, short_suffix, short_currency, short_notional,
                   roll_offset, tc_rate, fx_fixing_type):
    # define instance of a strategy
    data_requests = {
        'long': (
            FutureDataRequest(start_date, end_date, calendar, long_root, long_suffix, 2),
            DatalakeBBGFuturesDataSource()
        ),
        'short': (
            FutureDataRequest(start_date, end_date, calendar, short_root, short_suffix, 2),
            DatalakeBBGFuturesDataSource()
        ),
    }
    if currency != long_currency:
        pair = f'{long_currency}{currency}'
        data_requests[pair] = (
            FXDataRequest(start_date, end_date, calendar, currency=long_currency, denominator_currency=currency, fixing_type=fx_fixing_type),
            DatalakeBBGFXDataSource()
        )
    if currency != long_currency and long_currency != short_currency:
        pair = f'{short_currency}{currency}'
        data_requests[pair] = (
            FXDataRequest(start_date, end_date, calendar, currency=short_currency, denominator_currency=currency, fixing_type=fx_fixing_type),
            DatalakeBBGFXDataSource()
        )

    futures_spread_strategy = FuturesSpread(
        start_date=start_date,
        end_date=end_date,
        calendar=calendar,
        currency=currency,
        parameters={
            'long': {
                'underlying': long_root,
                'target_notional': long_notional,
            },
            'short': {
                'underlying': short_root,
                'target_notional': -short_notional,
            },
            'roll_offset': roll_offset,
            'tc_rate': tc_rate,
        },
        data_requests=data_requests
    )
    # run backtest
    runner = LocalBacktester()
    results = runner.run(futures_spread_strategy, start_date, end_date, FuturesSpreadState(start_date, Portfolio([]), 0.0, 0.0))

    records = [{'date': x.time_stamp, 'pnl': x.price} for x in results]
    nav_series = pd.DataFrame.from_dict(records)

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

    return results, nav_series, portfolio_expanded
