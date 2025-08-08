from ...interface.idatarequest import IDataRequest
from ...interface.idatasource import IDataSource
from ...infrastructure.bmarket import BMarket
from ...tradable.portfolio import Portfolio
from ...tradable.option import Option
from ...tradable.future import Future
from ...dates.utils import add_tenor, add_business_days
from dateutil.relativedelta import relativedelta, FR
from ...infrastructure import market_utils
from ...backtest.strategy import StrategyState, Event, Strategy
from ...valuation.option_data_valuer import OptionDataValuer
from ...valuation.future_data_valuer import FutureDataValuer
from ...backtest.backtester import LocalBacktester
from ...analytics.option_selection_tool import SelectionTool
from ...analytics.symbology import option_root_from_ticker, ticker_from_option_root
from ...infrastructure.minute_option_data_container import OptionMinuteDataRequest, CassandraOptionMinuteDataScource
from ...infrastructure.future_intraday_data_container import FutureMinuteDataRequest, \
    CassandraFutureMinuteDataSource
from ...infrastructure.equity_intraday_data_container import EquityIntradayDataRequest, IVolEquityIntradayDataSource
from ...backtest.expires import ExpireContractsAtPrice, ExpireOptionAtIntrinsic
from ...backtest.tranche import DailyTranche, MinuteTranche
from ...data.market import find_nearest_listed_options_intrday
from ...analytics.symbology import currency_from_option_root, option_root_from_ticker
from ...backtest.costs import FlatTradingCost, FlatVegaCostStock, BidOfferCost
from ...tradable.cash import Cash
from ...dates.utils import is_business_day
from ...analytics.utils import EPSILON

from tqdm import tqdm
from datetime import datetime
import datetime as dt

MIN_SIZING_MEASURE = {
    'price': 0.05,
    'vega': 0.005,
    'theta': 0.0001
}


class StockOptionsIntradayState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, portfolio_before_trades=None):
        self.price = price
        self.cost = cost
        self.portfolio_before_trades = portfolio_before_trades
        super().__init__(time_stamp, portfolio)


def make_tranche_key(entry_date, exit_date):
    if exit_date is None:
        exit_date = datetime.max
    return f"{entry_date.strftime('%Y-%m-%d %H:%M:%S')}_{exit_date.strftime('%Y-%m-%d %H:%M:%S')}"


def parse_tranche_key(tranche_name):
    [entry_str, exit_str] = tranche_name.split('_')
    return datetime.strptime(entry_str, '%Y-%m-%d %H:%M:%S'), datetime.strptime(exit_str, '%Y-%m-%d %H:%M:%S')


class ExpirationEvent(Event):
    def execute(self, state: StrategyState, daily_states=[]):

        portfolio = state.portfolio.clone()
        # expire options of each leg, cash in each leg portfolio
        for leg_name, leg in self.parameters['legs'].items():
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                self.parameters['expires'].expire(self.time_stamp, leg_portfolio, cash_path=())
        return StockOptionsIntradayState(self.time_stamp, portfolio, 0.0, 0.0)


class OptionTradeEvent(Event):
    def __init__(self, dt, strategy):
        super().__init__(dt, strategy)

    def get_timestamp(self):
        return self.time_stamp

    @staticmethod
    def parse_option_type(type_string):
        return type_string.upper()[0]

    @staticmethod
    def select_listed_option(market, root, sizing_measure, other_filters, target_expiration, selection_value,
                             selection_by, option_type, expiration_search_method='absolute'):
        option_universe = market.get_option_universe(root, return_as_dict=False).copy()
        if sizing_measure not in ['units', 'notional', 'strike_notional', None]:
            filters = other_filters + [
                lambda x: x[abs(x[sizing_measure]) >= MIN_SIZING_MEASURE.get(sizing_measure, EPSILON)]]
        else:
            filters = other_filters

        option_universe['expiration_date'] = option_universe['expiration_date'].apply(
            lambda x: datetime.combine(x.date(), datetime.min.time()).isoformat()).astype(str)
        option_to_trade = find_nearest_listed_options_intrday(target_expiration, selection_value, option_type,
                                                              option_universe,
                                                              other_filters=filters,
                                                              return_as_tradables=True,
                                                              select_by=selection_by,
                                                              expiration_search_method=expiration_search_method)
        assert len(option_to_trade) >= 1
        # TODO: more than one option expires on same day we have to choose
        option_to_trade = option_to_trade[0]
        return option_to_trade

    @staticmethod
    def option_selection_simple(
            # generic input
            strategy, strategy_state,
            # underlying
            underlying,
            # listed or otc
            use_listed,
            # type
            option_type,
            # strike selection
            selection_by, selection_value,
            # expiration selection
            tenor,
            # expiration selection overrides
            override_target_expiry=None, force_3rd_Fri=False,
            # additional filter on option selection
            other_filters=[],
            # sizing measure to filter selection
            sizing_measure_to_filter_selection=None,
            spread_strike=None,

    ):
        time_stamp = strategy_state.time_stamp
        market = strategy.backtest_market.get_market(time_stamp)
        spot = market.get_spot(underlying)
        valuer = strategy.valuer_map[Option]
        root = option_root_from_ticker(underlying)

        # find the option to trade
        if tenor.endswith('+'):
            tenor_used = tenor[:-1]
            expiration_search_method = 'geq'
        elif tenor.endswith('-'):
            tenor_used = tenor[:-1]
            expiration_search_method = 'leq'
        else:
            tenor_used = tenor
            expiration_search_method = 'absolute'
        target_expiration = add_tenor(market.base_datetime, tenor_used)

        if force_3rd_Fri:
            fri3 = target_expiration.replace(day=1)
            fri3 = fri3 + relativedelta(weekday=FR(3))
            target_expiration = fri3

        option_type = OptionTradeEvent.parse_option_type(option_type)

        if override_target_expiry is not None:
            target_expiration = override_target_expiry
            # expiration_search_method = 'absolute'

        if selection_by == 'strike':
            target_strike = spot * selection_value
            if use_listed:
                option_to_trade = OptionTradeEvent.select_listed_option(market, root,
                                                                        sizing_measure_to_filter_selection,
                                                                        other_filters, target_expiration, target_strike,
                                                                        selection_by, option_type,
                                                                        expiration_search_method=expiration_search_method)
            else:
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                         target_strike, option_type == 'C', False, 1, '')
        elif selection_by == 'delta':
            if use_listed:
                option_to_trade = OptionTradeEvent.select_listed_option(market, root,
                                                                        sizing_measure_to_filter_selection,
                                                                        other_filters, target_expiration,
                                                                        selection_value, selection_by, option_type,
                                                                        expiration_search_method=expiration_search_method)
            else:
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                         spot, option_type == 'C', False, 1, '')
                strike = valuer.solve(option_to_trade, market, given='delta', solve_for='strike',
                                      value_given=selection_value)
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                         strike, option_type == 'C', False, 1, '')
        else:
            raise RuntimeError(f"Option selection by {selection_by} has not been implemented")

        if spread_strike is None:
            return [option_to_trade]
        else:
            if selection_by == 'strike':
                target_strike = spot * spread_strike
                if use_listed:
                    sprd_option_to_trade = OptionTradeEvent.select_listed_option(market, root,
                                                                                 sizing_measure_to_filter_selection,
                                                                                 other_filters, target_expiration,
                                                                                 target_strike, selection_by,
                                                                                 option_type,
                                                                                 expiration_search_method=expiration_search_method)
                else:
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root),
                                                  target_expiration,
                                                  target_strike, option_type == 'C', False, 1, '')
            elif selection_by == 'delta':
                if use_listed:
                    sprd_option_to_trade = OptionTradeEvent.select_listed_option(market, root,
                                                                                 sizing_measure_to_filter_selection,
                                                                                 other_filters, target_expiration,
                                                                                 spread_strike, selection_by,
                                                                                 option_type,
                                                                                 expiration_search_method=expiration_search_method)
                else:
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root),
                                                  target_expiration,
                                                  spot, option_type == 'C', False, 1, '')
                    strike = valuer.solve(sprd_option_to_trade, market, given='delta', solve_for='strike',
                                          value_given=spread_strike)
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root),
                                                  target_expiration,
                                                  strike, option_type == 'C', False, 1, '')
            else:
                raise RuntimeError(f"Option selection by {selection_by} has not been implemented")

            return [option_to_trade, sprd_option_to_trade]

    @staticmethod
    def option_sizing_simple(
            # generic input
            strategy, strategy_state,
            # option selected
            options_to_trade,
            # sizing
            sizing_measure, sizing_target,
            market
    ):
        assert len(options_to_trade) == 1
        option_to_trade = options_to_trade[0]

        time_stamp = strategy_state.time_stamp
        underlying = ticker_from_option_root(option_to_trade.underlying)
        spot = market.get_spot(underlying)
        valuer = strategy.valuer_map[Option]

        if sizing_measure in ['units', 'notional', 'strike_notional']:
            bid, ask, option_price = option_to_trade.price(market, valuer, calc_types=['bid', 'ask', 'price'])
        else:
            bid, ask, option_price, option_sizing_measure_value = option_to_trade.price(market, valuer,
                                                                                        calc_types=['bid', 'ask',
                                                                                                    'price',
                                                                                                    sizing_measure])

        # sizing
        if sizing_measure == 'units':
            sizing_measure_value = 1
        elif sizing_measure == 'notional':
            sizing_measure_value = spot
        elif sizing_measure == 'strike_notional':
            sizing_measure_value = option_to_trade.strike
        else:
            sizing_measure_value = option_sizing_measure_value
        units_to_trade = sizing_target / sizing_measure_value

        return [(units_to_trade, option_price)]

    @staticmethod
    def trade_one_leg(portfolio, tranche_fraction,
                      strategy, strategy_state,
                      option_selection_function,
                      option_sizing_function,
                      position_path, exit_before_expiry, holidays=[],
                      scale_by_nav=None,
                      spread_strike=None,
                      unique_holding=False):
        # if unique_holding == True:
        #     if portfolio.get_position((position_path[0],)) is not None:
        #         if len(portfolio.get_position((position_path[0],)).filter_tradable_type(Option).root) > 0:
        #             return

        ts = strategy_state.time_stamp
        options_selected_list = option_selection_function(strategy, strategy_state, spread_strike)
        if spread_strike is None:
            option_sizing_list = option_sizing_function(strategy, strategy_state, options_selected_list)
        else:
            option_sizing_list = option_sizing_function(strategy, strategy_state, [options_selected_list[0]])
            sprd_option_sizing_list = option_sizing_function(strategy, strategy_state, [options_selected_list[1]])
            option_sizing_list = option_sizing_list + [(option_sizing_list[0][0], sprd_option_sizing_list[0][1])]

        if scale_by_nav is not None:
            option_sizing_list = [(x[0] * scale_by_nav, x[1]) for x in option_sizing_list]

        time_stamp = strategy_state.time_stamp
        market = strategy.backtest_market.get_market(time_stamp)
        valuer = strategy.valuer_map[Option]

        def _build_options_to_trade_list(_x, _y):
            if isinstance(_y, tuple):
                assert len(_y) == 2
                _units = _y[0]
                _execution_price = _y[1]
            else:
                _units = _y
                _execution_price = _x.price(market, valuer, calc_types='price')
            return _x, _units, _execution_price

        option_to_trade_list = [_build_options_to_trade_list(x, y) for x, y in
                                zip(options_selected_list, option_sizing_list)]
        
        if exit_before_expiry is not None:
            min_exp = option_to_trade_list[0][0].expiration
            for o, u, e in option_to_trade_list:
                if o.expiration < min_exp:
                    min_exp = o.expiration


            exit_minute=parse_tranche_key(position_path[1])[1]
            unwind_date = add_business_days(min_exp, exit_before_expiry, holidays=holidays).replace(hour=exit_minute.hour, minute=exit_minute.minute)
            strategy.parameters['legs'][position_path[0]]['tranche'].add_tranche(unwind_date,unwind_date,1)
            position_path = (position_path[0], make_tranche_key(ts, unwind_date))

        
        for o, u, p in option_to_trade_list:
            portfolio.trade(o, u * tranche_fraction, p, o.currency, position_path=position_path,
                            cash_path=(position_path[0],))
        

    def execute(self, state: StockOptionsIntradayState, daily_states=[]):


        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        if len(daily_states) > 0:
            prev_nav = daily_states[-1].price
        else:
            prev_nav = None

        # price before any trade
        portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        portfolio_before_trades = portfolio.clone()

        for leg_name, leg in self.parameters['legs'].items():
            # get market data for this day

            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                leg_position_names = list(leg_portfolio.get_positions().keys())
                for tranche_name in leg_position_names:
                    tranche_portfolio = leg_portfolio.get_position(tranche_name)
                    if isinstance(tranche_portfolio, Portfolio):
                        # move tranche portfolio with cash only to leg level
                        if self.parameters.get('keep_hedges_in_tranche_portfolio', False):
                            pos_list = list(tranche_portfolio.get_positions().items())
                            cash_only = all(
                                [isinstance(x[1].tradable, Cash) or isinstance(x[1].tradable, Constant) for x in
                                 pos_list])
                            if cash_only:
                                for cash_pos in pos_list:
                                    if isinstance(cash_pos[1].tradable, Cash) or isinstance(cash_pos[1].tradable,
                                                                                            Constant):
                                        leg_portfolio.move(
                                            cash_pos[1].tradable,
                                            cash_pos[1].quantity,
                                            (tranche_name,), ()
                                        )
                                    else:
                                        raise RuntimeError(f"found non cash in an unwound tranche portfolio")
                        # unwind portfolio when it is the unwind date

                        
                        if parse_tranche_key(tranche_name)[1] == self.time_stamp:
                            tranche_position_names = list(tranche_portfolio.get_positions().keys())
                            for option_position_name in tranche_position_names:
                                option_position = tranche_portfolio.get_position(option_position_name)
                                if self.parameters.get('keep_hedges_in_tranche_portfolio', False):
                                    if not isinstance(option_position.tradable, Cash) and not isinstance(
                                            option_position.tradable, Constant):
                                        unwind_price = option_position.tradable.price(market, self.strategy.valuer_map[
                                            type(option_position.tradable)], calc_types='price')
                                        portfolio.unwind((leg_name, tranche_name, option_position_name), unwind_price,
                                                         option_position.tradable.currency,
                                                         cash_path=(leg_name, tranche_name,))
                                else:
                                    unwind_price = option_position.tradable.price(market,
                                                                                  self.strategy.valuer_map[Option],
                                                                                  calc_types='price')
                                    portfolio.unwind((leg_name, tranche_name, option_position_name), unwind_price,
                                                     option_position.tradable.currency, cash_path=(leg_name,))
            # trade new tranches
            if leg['tranche'].is_entry_datetime(self.time_stamp):
                # if there is no customize function we must have selection method as leg parameters
                if "option_selection_function" not in leg:
                    if 'selection_by' in leg:
                        selection_by = leg['selection_by']
                    elif 'strike' in leg:
                        selection_by = 'strike'
                    elif 'delta' in leg:
                        selection_by = 'delta'
                    else:
                        raise RuntimeError(
                            "Cannot find option selection method. In the leg config please specify 'selection_by' and the corresponding measure.")
                    selection_value = leg[selection_by]

                # TODO: remove the case where the exit datetime can be None by making the tranche class logic take care of future exit dates
                if isinstance(leg['tranche'].get_exit_datetime(self.time_stamp), datetime) or leg[
                    'tranche'].get_exit_datetime(self.time_stamp) is None:
                    exit_dates = [leg['tranche'].get_exit_datetime(self.time_stamp)]
                    fractions = [leg['tranche'].get_tranche_fraction(self.time_stamp)]
                else:
                    # this is when for one entry date, there are multiple exit dates
                    print("tranching with multiple exit dates on same entry date")
                    exit_dates = leg['tranche'].get_exit_datetime(self.time_stamp)
                    fractions = leg['tranche'].get_tranche_fraction(self.time_stamp)

                # run through tranches
                if leg['tranche'].get_target_expiry(self.time_stamp) is None:
                    target_exps = [None for x in exit_dates]
                else:
                    if isinstance(leg['tranche'].get_target_expiry(self.time_stamp), datetime):
                        target_exps = [leg['tranche'].get_target_expiry(self.time_stamp)]
                    else:
                        target_exps = leg['tranche'].get_target_expiry(self.time_stamp)

                for _tranche_exit, _tranche_fraction, _target_exp_from_tranche in zip(exit_dates, fractions,
                                                                                      target_exps):
                    if "option_selection_function" in leg:
                        OptionTradeEvent.trade_one_leg(
                            # portfolio obj to trade in, and fraction due to tranche
                            portfolio,
                            _tranche_fraction,
                            # the following three define the option selection and sizing
                            self.strategy,
                            state,
                            leg['option_selection_function'],
                            leg['option_sizing_function'],
                            # other trading parameters
                            position_path=(leg_name, make_tranche_key(self.time_stamp, _tranche_exit)),
                            exit_before_expiry=leg.get("exit_before_expiry", None),
                            holidays=self.strategy.holidays,
                            scale_by_nav=prev_nav if ('scale_by_nav' in self.parameters.keys()
                                                      and self.parameters['scale_by_nav']) else None,
                            spread_strike=leg['spread_strike'] if 'spread_strike' in leg else None,
                            unique_holding=leg.get('unique_holding', False),
                        )
                    else:
                        spread_stk = leg['spread_strike'] if 'spread_strike' in leg else None
                        OptionTradeEvent.trade_one_leg(
                            # portfolio obj to trade in, and fraction due to tranche
                            portfolio,
                            _tranche_fraction,
                            # the following four define the option selection and sizing
                            self.strategy,
                            state,
                            lambda _strategy, _state, spread_stk:
                            OptionTradeEvent.option_selection_simple(
                                _strategy, _state,
                                leg['underlying'],
                                self.parameters.get('use_listed', True),
                                leg['type'],
                                selection_by, selection_value,
                                leg['tenor'],
                                _target_exp_from_tranche,
                                'force_3rd_Fri' in leg,
                                leg.get('other_filters', []),
                                leg['sizing_measure'],
                                spread_strike=spread_stk
                            ),
                            lambda _strategy, _state, _options:
                            OptionTradeEvent.option_sizing_simple(
                                _strategy, _state, _options,
                                leg['sizing_measure'], leg['sizing_target'],
                                market
                            ),
                            # other trading parameters
                            position_path=(leg_name, make_tranche_key(self.time_stamp, _tranche_exit)),
                            exit_before_expiry=leg.get("exit_before_expiry", None),
                            holidays=self.strategy.holidays,
                            scale_by_nav=prev_nav if ('scale_by_nav' in self.parameters.keys()
                                                      and self.parameters['scale_by_nav']) else None,
                            spread_strike=leg['spread_strike'] if 'spread_strike' in leg else None,
                            unique_holding=leg.get('unique_holding', False),
                        )

        if 'flat_vega_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'vega', 'theta'],
                                              valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price[0], 0.0)
        elif 'bid_offer_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'bid', 'ask'],
                                              valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price[0], 0.0)
        else:
            price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        # value portfolio
        px = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)

        return StockOptionsIntradayState(self.time_stamp, portfolio, px, 0.0, portfolio_before_trades)


class OptionDeltaHedgeEvent(Event):

    def hedge_simple(self, strategy, strategy_state, leg_portfolio):

        market = strategy.backtest_market.get_market(strategy_state.time_stamp)

        leg_portfolio.price_at_market(market, 'delta', self.strategy.valuer_map, default=0)
        portfolio_delta = leg_portfolio.aggregate('delta')

        if portfolio_delta != 0:

            return True

        else:

            return False

    def execute(self, state: StrategyState, daily_states=[]):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day

        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # use stale data to forward fill missing price
        if self.parameters.get('use_listed', True):
            backfill_markets = []
            for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
                back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
                if back_date >= self.strategy.start_date:
                    backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
            self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        for leg_name in list(self.parameters['legs'].keys()):

            hedge_now = False

            if 'hedge_function' in self.parameters['legs'][leg_name]:

                hedge_now = self.parameters['legs'][leg_name]['hedge_function'](self.strategy, state)

            elif 'hedge_schedule' in self.parameters['legs'][leg_name]:

                hedge_now = (self.time_stamp.to_pydatetime()in self.parameters['legs'][leg_name]['hedge_schedule'])

            else:

                hedge_now = self.hedge_simple(self.strategy, state, leg_portfolio)


            if 'hedging_instrument' not in self.parameters['legs'][leg_name]:
                continue
            
            if hedge_now:
                # hedge future
                if self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future':
                    hedge_instrument_type = Future
                    container = market_utils.create_future_data_container_key(
                        self.parameters['legs'][leg_name]['hedging_instrument']['root'])
                    hedge_instrument = market.storage[container].get_lead_future(market.base_datetime,
                                                                                self.parameters['legs'][leg_name][
                                                                                    'hedging_instrument']['expiry_offset'])
                    hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market,
                                                                                            self.strategy.valuer_map[
                                                                                                Future],
                                                                                            calc_types=['price', 'delta'])
                elif self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'stock':
                    hedge_instrument_type = Stock
                    hedge_instrument = Stock(self.parameters['legs'][leg_name]['hedging_instrument']['ticker'],
                                            self.parameters['legs'][leg_name]['hedging_instrument']['currency'])
                    hedge_instrument_price = market.get_spot(
                        self.parameters['legs'][leg_name]['hedging_instrument']['ticker'])
                    hedge_instrument_delta = 1.0
                else:
                    assert self.parameters.get('keep_hedges_in_tranche_portfolio', False)
                    assert self.parameters['legs'][leg_name]['hedging_instrument'][
                            'type'] == 'future_expiry_at_option_expiry', 'Unknown hedge instrument type'
                leg_portfolio = portfolio.get_position(leg_name)

                if leg_portfolio is not None:
                    # first unwind any hedge that is not the selected hedge future (the only reason this is the case is it is close to expiry)
                    if self.parameters.get('keep_hedges_in_tranche_portfolio', False):
                        leg_position_names = list(leg_portfolio.get_positions().keys())
                        for tranche_name in leg_position_names:
                            tranche_portfolio = leg_portfolio.get_position(tranche_name)
                            if isinstance(tranche_portfolio, Portfolio):

                                # find hedge instrument for this tranche
                                if self.parameters['legs'][leg_name]['hedging_instrument'][
                                    'type'] == 'future_expiry_at_option_expiry':
                                    option_positions = tranche_portfolio.find_children_of_tradable_type(Option)
                                    if len(option_positions):
                                        option_expiry = option_positions[0][1].tradable.expiration
                                        hedge_instrument_type = Future
                                        hedge_instrument = market.storage[container].get_lead_future(market.base_datetime,
                                                                                                    self.parameters[
                                                                                                        'legs'][leg_name][
                                                                                                        'hedging_instrument'][
                                                                                                        'expiry_offset'])
                                        hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market,
                                                                                                                self.strategy.valuer_map[
                                                                                                                    Future],
                                                                                                                calc_types=[
                                                                                                                    'price',
                                                                                                                    'delta'])
                                        net_positions = tranche_portfolio.net_positions()
                                        for position_name, position in net_positions.items():
                                            if isinstance(position.tradable,
                                                        hedge_instrument_type) and not position.tradable == hedge_instrument:
                                                unwind_price = position.tradable.price(market, self.strategy.valuer_map[
                                                    hedge_instrument_type], calc_types='price')
                                                tranche_portfolio.trade(position.tradable, -position.quantity, unwind_price,
                                                                        position.tradable.currency)
                                else:
                                    net_positions = tranche_portfolio.net_positions()
                                    for position_name, position in net_positions.items():
                                        if isinstance(position.tradable,
                                                    hedge_instrument_type) and not position.tradable == hedge_instrument:
                                            unwind_price = position.tradable.price(market, self.strategy.valuer_map[
                                                hedge_instrument_type], calc_types='price')
                                            tranche_portfolio.trade(position.tradable, -position.quantity, unwind_price,
                                                                    position.tradable.currency)
                    else:
                        net_positions = leg_portfolio.net_positions()
                        for position_name, position in net_positions.items():
                            if isinstance(position.tradable,
                                        hedge_instrument_type) and not position.tradable == hedge_instrument:
                                unwind_price = position.tradable.price(market,
                                                                    self.strategy.valuer_map[hedge_instrument_type],
                                                                    calc_types='price')
                                leg_portfolio.trade(position.tradable, -position.quantity, unwind_price,
                                                    position.tradable.currency)



                    # then hedge the portfolio delta with front future
                    leg_portfolio.price_at_market(market, 'delta', self.strategy.valuer_map, default=0)
                    portfolio_delta = leg_portfolio.aggregate('delta')
                    leg_portfolio.trade(hedge_instrument, -portfolio_delta / hedge_instrument_delta,
                                        hedge_instrument_price, hedge_instrument.currency)

        # value portfolio
        if 'flat_vega_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'vega'],
                                              valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price[0], 0.0)
        elif 'bid_offer_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'bid', 'ask'],
                                              valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price[0], 0.0)
        else:
            price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsIntradayState(self.time_stamp, portfolio, price, 0.0)


        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        # value portfolio
        calc_types = ['price'] + self.parameters.get('greeks_to_include', ['delta'])
        pricing_results = portfolio.price_at_market(market, fields=calc_types,
                                                    valuer_map_override=self.strategy.valuer_map)

        return StockOptionsIntradayState(self.time_stamp, portfolio, pricing_results[0], 0.0,
                                         state.portfolio_before_trades)


class StockOptionsIntraday(Strategy):

    def __init__(self, parameters, schedule, holidays, data_requests: dict[str, (IDataRequest, IDataSource)]):
        super().__init__(start_date=None, end_date=None, calendar=None, currency=None,
                         parameters=parameters, data_requests=data_requests)
        self.schedule = schedule
        self.holidays = holidays

    def preprocess(self):
        super().preprocess()

        self.valuer_map = self.parameters['valuer_map']

        if not 'expires' in self.parameters:
            self.parameters['expires'] = ExpireContractsAtPrice(self.backtest_market, self.valuer_map)
        elif self.parameters['expires'] == ExpireOptionAtIntrinsic:
            self.parameters['expires'] = ExpireOptionAtIntrinsic(self.backtest_market)
        elif self.parameters['expires'] == ExpireContractsAtPrice:
            self.parameters['expires'] = ExpireContractsAtPrice(self.backtest_market, self.valuer_map)

        if 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters['flat_costs']
            self.parameters['trading_cost'] = FlatTradingCost(cost_params['tc_delta'], cost_params['tc_vega'])
        elif 'flat_vega_charge' in self.parameters.keys():
            cost_params = self.parameters['flat_vega_charge']
            self.parameters['trading_cost'] = FlatVegaCostStock(cost_params['tc_delta'], cost_params['tc_vega'])
        elif 'bid_offer_charge' in self.parameters.keys():
            cost_params = self.parameters['bid_offer_charge']
            self.parameters['trading_cost'] = BidOfferCost(cost_params['tc_delta'], cost_params['tc_vega'])

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data_container in self.data_containers.items():
            backtest_market.add_item(data_container.get_market_key(), data_container)
        self.backtest_market = backtest_market
        self.valuer_map = self.parameters['valuer_map']

    def generate_events(self, dt):
        if dt.time() == self.parameters['expiration_time']:
            return [ExpirationEvent(dt, self), OptionTradeEvent(dt, self), OptionDeltaHedgeEvent(dt, self)]
        else:
            return [OptionTradeEvent(dt, self), OptionDeltaHedgeEvent(dt, self)]

    def evolve(self, start_date, end_date, start_state: StrategyState):
        daily_states = [start_state]
        time_grid = self.schedule
        time_grid = time_grid[(time_grid >= start_date) & (time_grid <= end_date)]
        for dt in tqdm(time_grid):
            if is_business_day(dt, self.holidays):
                today_events = self.generate_events(dt)
                state = daily_states[-1]
                for event in today_events:
                    state = event.execute(state)
                daily_states.append(state)
        return daily_states


if __name__ == '__main__':
    from datetime import datetime
    import pandas as pd

    from ..valuation.option_data_valuer import OptionDataValuer
    from ..valuation.option_vola_valuer import OptionVolaValuer
    from ..valuation.stock_data_valuer import StockDataValuer

    from ..tradable.future import Future
    from ..tradable.constant import Constant
    from ..tradable.option import Option
    from ..tradable.stock import Stock
    from tqdm import tqdm

    from ..dates.holidays import get_holidays

    import warnings

    warnings.filterwarnings("ignore")

    start_date = datetime(2018, 8, 1, 0)
    end_date = datetime(2018, 8, 10, 0)


    def calculate_holidays(calendar, start_date, end_date):
        if not isinstance(calendar, list):
            calendar = [calendar]
        holiday_days = []
        for cal in calendar:
            if isinstance(cal, str):
                holiday_days = holiday_days + get_holidays(cal, start_date, end_date)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')
        return holiday_days


    calendar = ['XCBO']
    holidays = calculate_holidays(calendar, start_date, end_date)
    schedule = pd.date_range(start_date, end_date, freq='15T')
    schedule = schedule[schedule.day_of_week < 5]

    data_requests = {
        'SPX spots': (
            EquityIntradayDataRequest(ticker='SPX Index', look_times=schedule[
                schedule.indexer_between_time('9:30:00', '16:00:00')].strftime("%Y-%m-%d %H:%M:%S").to_list()),
            IVolEquityIntradayDataSource()),
        'SPX options': (
            OptionMinuteDataRequest(root='SPX'),
            CassandraOptionMinuteDataScource()),
        'SPX futures': (
            FutureMinuteDataRequest(root='ES', suffix='Index', allow_back_fill=60),
            CassandraFutureMinuteDataSource())
    }

    strategy = StockOptionsIntraday(parameters={
        'legs': {'short_put': {'underlying': 'SPX Index',
                               'tenor': '1M',
                               'delta': -0.25,
                               'type': 'P',
                               'other_filters': [lambda x: x[x['price'] >= 0.05]],
                               'sizing_measure': 'notional',
                               'sizing_target': -100 / 4,
                               'hedging_instrument': {'type': 'future', 'root': 'ES', 'expiry_offset': 3},
                               'tranche': MinuteTranche(start_date, end_date, 1, 'previous', calendar, '9:30:00',
                                                        '16:00:00', interval=390)}},
        'valuer_map': {
            Option: OptionDataValuer(),
            Future: FutureDataValuer(imply_delta_from_spot=False if all([x == 'future' for x in ['Equity']]) else True),
            Stock: StockDataValuer(),
        }
    },
        data_requests=data_requests,
        schedule=schedule[schedule.indexer_between_time('9:30:00', '16:00:00')],
        holidays=holidays
    )

    runner = LocalBacktester()
    results = runner.run(strategy=strategy, start_date=start_date, end_date=end_date,
                         initial_state=StockOptionsIntradayState(datetime(2018, 8, 1, 9, 30), Portfolio([]), 0.0, 0.0))

    for i in results:
        print(i.__dict__)

