from datetime import datetime, timedelta
import pandas as pd

from ...analytics.symbology import future_root_suffix_from_ticker, currency_from_option_root, \
    hedging_instrument_from_ticker, option_root_from_ticker, option_underlying_type_from_ticker
from ...backtest.strategies.stock_options_daily import StockOptionsDailyState, StockOptionsDaily
from ...backtest.backtester import LocalBacktester
from ...data.datalake_cassandra import ExpiryFilterByDateOffset, ExpirationRuleByList
from ...infrastructure.future_data_container import FutureDataRequest, DatalakeBBGFuturesDataSource, \
    Datalake2BBGFuturesDataSource
from ...infrastructure.option_data_container import OptionDataRequest, CassandraDSPreLoadFromPickle, \
    CassandraDSPreLoadFromDict, CassandraDSPreLoadFromCacheByYear, CassandraDSPreLoadFromCacheByDay, \
    CassandraDSOnDemandFromCacheByYear
from ...infrastructure.equity_data_container import EquityDataRequest, IVolEquityDataSource
from ...infrastructure.dividend_data_container import DividendDataRequest, IVolDividendDataSource
from ...infrastructure.corpaction_data_container import CorpActionDataRequest, IVolCorpActionDataSource
from ...tradable.portfolio import Portfolio
from ...tradable.future import Future
from ...tradable.constant import Constant
from ...tradable.option import Option
from ...tradable.stock import Stock
from ...valuation.future_data_valuer import FutureDataValuer
from ...valuation.future_from_option_data_valuer import FutureFromOptionDataValuer
from ...valuation.option_data_valuer import OptionDataValuer
from ...valuation.stock_data_valuer import StockDataValuer
from ...valuation.constant_valuer import ConstantValuer

from ...dates.utils import add_business_days
from ...tradable.constant import Constant
from ...tradable.option import Option
from ...tradable.stock import Stock
from ...tradable.cash import Cash
from ...tradable.position import Position
from ...dates.holidays import get_holidays


def stock_options_daily(start_date: datetime, end_date: datetime, calendar: list[str],
                        legs: dict,
                        hedged: bool,
                        max_option_expiry_days: int,
                        option_data_source,
                        extra_data_requests: dict = {},
                        allow_fill_forward_missing_data: int = 0,
                        prev_state=[],
                        use_listed: bool = True,
                        cost_params: dict = {},
                        expiration_rules=None,
                        keep_hedges_in_tranche_portfolio=False,
                        hedge_future_expiry_at_option_expiry=False,
                        number_of_futures_to_load=2,
                        trade_first_day=False,
                        greeks_to_include: list[str] = ['delta'],
                        inc_greeks=False,
                        scale_by_nav=False,
                        data_start_date_shift=0,
                        allow_fix_option_price_from_settlement=False,
                        allow_reprice=False,
                        inc_trd_dts=False,
                        return_strategy_and_initial_state=False
                        ):
    if not use_listed:
        from ...infrastructure.eq_vol_data_container import EqVolRequest, VolaEqVolDataSource, VolaEqVolDataSourceDict
        from ...valuation.option_vola_valuer import OptionVolaValuer
    if isinstance(option_data_source, dict):
        for v in list(option_data_source.values()):
            if not use_listed:
                assert isinstance(v, VolaEqVolDataSource) or isinstance(v, VolaEqVolDataSourceDict)
    else:
        if not use_listed:
            assert isinstance(option_data_source, VolaEqVolDataSource)

    data_start_date = add_business_days(start_date, -data_start_date_shift,
                                        get_holidays(calendar, start_date - timedelta(2 * data_start_date_shift),
                                                     start_date))

    parameters = {
        'legs': legs,
        'allow_fill_forward_missing_data': allow_fill_forward_missing_data,
        'use_listed': use_listed,
        'keep_hedges_in_tranche_portfolio': keep_hedges_in_tranche_portfolio,
        'greeks_to_include': greeks_to_include,
    }
    if len(cost_params) > 0:
        assert len(cost_params) == 1
        assert 'variable_costs' in cost_params.keys() or 'flat_costs' in cost_params.keys() \
               or 'flat_vega_charge' in cost_params.keys() or 'bid_offer_charge' in cost_params.keys()
        for k, v in cost_params.items():
            parameters[k] = v

    if scale_by_nav:
        parameters['scale_by_nav'] = True

    underlyings = []
    parameters['process_dividends_and_corpactions'] = False
    if not hedged:
        parameters['naked'] = True

    data_requests = {}
    option_underlying_types = []
    for leg_name, leg_info in legs.items():
        underlying = leg_info['underlying']
        leg_hedged = hedged[leg_name] if isinstance(hedged, dict) else hedged
        root = option_root_from_ticker(underlying)
        hedging_instrument = hedging_instrument_from_ticker(underlying)
        future_root, future_suffix = future_root_suffix_from_ticker(underlying)
        currency = currency_from_option_root(root)

        if not hedging_instrument == 'future':
            parameters['process_dividends_and_corpactions'] = True

        option_underlying_types.append(option_underlying_type_from_ticker(underlying))

        if underlying not in underlyings:
            if use_listed:
                data_requests.update({
                    f'{root} spots': (
                        EquityDataRequest(data_start_date, end_date, calendar, underlying),
                        IVolEquityDataSource()
                    ),
                    f'{root} options': (
                        OptionDataRequest(data_start_date, end_date, calendar, root,
                                          expiry_filter=ExpiryFilterByDateOffset(max_option_expiry_days),
                                          expiration_rule_filter=ExpirationRuleByList(
                                              expiration_rules) if expiration_rules is not None else None,
                                          allow_fix_option_price_from_settlement=allow_fix_option_price_from_settlement,
                                          allow_reprice=allow_reprice, ),
                        option_data_source[root] if isinstance(option_data_source, dict) else option_data_source.clone()
                    ),
                })
            else:
                data_requests.update({
                    f'{root} vola': (
                        EqVolRequest(underlying, data_start_date, end_date),
                        option_data_source[root] if isinstance(option_data_source, dict) else option_data_source.clone()
                    ),
                })

        if leg_hedged:
            if hedging_instrument == 'future':
                parameters['legs'][leg_name]['hedging_instrument'] = {
                    'type': 'future_expiry_at_option_expiry' if hedge_future_expiry_at_option_expiry else 'future',
                    'root': future_root,
                    'expiry_offset': 3,
                }
                if underlying not in underlyings:
                    data_requests[f'{root} future'] = (
                        FutureDataRequest(data_start_date, end_date, calendar, future_root, future_suffix,
                                          number_of_futures_to_load),
                        Datalake2BBGFuturesDataSource()
                    )
            elif hedging_instrument == 'stock':
                parameters['legs'][leg_name]['hedging_instrument'] = {
                    'type': 'stock',
                    'ticker': underlying,
                    'currency': currency,
                }
                if underlying not in underlyings:
                    data_requests[f'{root} div'] = (
                        DividendDataRequest(data_start_date, end_date, calendar, underlying),
                        IVolDividendDataSource()
                    )
                    data_requests[f'{root} corpaction'] = (
                        CorpActionDataRequest(data_start_date, end_date, calendar, underlying),
                        IVolCorpActionDataSource()
                    )
            else:
                raise RuntimeError('Unknown hedging instrument')

        underlyings.append(underlying)

    # for now only support same type of uncerlyings

    assert all([x == 'future' for x in option_underlying_types]) or all(
        [x == 'equity' for x in option_underlying_types])

    parameters['valuer_map'] = {
        Option: OptionDataValuer() if use_listed else OptionVolaValuer(),
        # future data valuer's delta: is it delta on future or on spot?
        Future: FutureDataValuer(imply_delta_from_spot=False if all([x == 'future'
                                                                     for x in option_underlying_types]) else True),
        # FutureFromOptionDataValuer('VIX', 'VIX Index'),
        Stock: StockDataValuer(),
    }

    # extra data in addition to those implied by the parameters
    if extra_data_requests is not None:
        for k, v in extra_data_requests.items():
            assert k not in data_requests
            data_requests[k] = v

    # define instance of a strategy
    parameters['trade_first_day'] = trade_first_day
    daily = StockOptionsDaily(start_date=start_date, end_date=end_date, calendar=calendar, currency=currency,
                              parameters=parameters, data_requests=data_requests, inc_trd_dts=inc_trd_dts)
    stock_options_daily_strategy = daily

    # for use in common_bt_stock_options_daily.py
    if return_strategy_and_initial_state:
        if scale_by_nav:
            pfo = Portfolio([])
            cash = Position(Cash(currency), 100)
            pfo.add_position(cash.tradable, cash.quantity)
            pfo_price = 100
            return stock_options_daily_strategy, StockOptionsDailyState(start_date, pfo, pfo_price, 0.0)
        else:
            return stock_options_daily_strategy, StockOptionsDailyState(start_date, Portfolio([]), 0.0, 0.0)

    # run backtest
    runner = LocalBacktester()
    if isinstance(prev_state, StockOptionsDailyState):
        results = runner.run(stock_options_daily_strategy, start_date, end_date, prev_state)
    else:
        if scale_by_nav:
            pfo = Portfolio([])
            cash = Position(Cash(currency), 100)
            pfo.add_position(cash.tradable, cash.quantity)
            pfo_price = 100
            results = runner.run(stock_options_daily_strategy, start_date, end_date,
                                 StockOptionsDailyState(start_date, pfo, pfo_price, 0.0))
        else:
            results = runner.run(stock_options_daily_strategy, start_date, end_date,
                                 StockOptionsDailyState(start_date, Portfolio([]), 0.0, 0.0))

    records = [{'date': x.time_stamp, 'pnl': x.price} for x in results]
    nav_series = pd.DataFrame.from_dict(records)

    portfolio_expanded = []
    for state in results:
        for k, v in state.portfolio.net_positions().items():
            if not isinstance(v.tradable, Constant):
                item = {
                    'date': state.time_stamp,
                    'position': k,
                    'quantity': v.quantity,
                    'price': v.price,
                }
                for g in greeks_to_include:
                    item[g] = getattr(v, g, None)
                portfolio_expanded.append(item)

    portfolio_expanded = pd.DataFrame.from_records(portfolio_expanded)

    return results, nav_series, portfolio_expanded
