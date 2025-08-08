from datetime import datetime, timedelta

from ...backtest.costs import FlatTradingCost
from ...backtest.rolls import RollFutureContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...tradable.future import Future
from ...tradable.portfolio import Portfolio
from ...valuation.future_data_valuer import FutureDataValuer


class RollingFuturesBasketState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost):
        self.price = price
        self.cost = cost
        super().__init__(time_stamp, portfolio)


class EODTradeEvent(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        if portfolio.is_empty():
            notionals_function = self.parameters["notionals_function"]
            weights = notionals_function(self.strategy, state)
            # initial positions
            for leg_name, leg_params in self.parameters['legs'].items():
                # find first future, but if we are too close to expiry, we trade in the next future
                current_future = self.parameters['roll'][leg_name].find_initial_contract(self.time_stamp, root=leg_params['underlying'])
                current_future_price = market.get_future_data(current_future)['close']

                units = weights[leg_name] / current_future_price
                portfolio.trade(current_future, units, current_future_price, position_path=(leg_name,))
        else:
            # roll if need to
            for leg_name, leg_params in self.parameters['legs'].items():
                leg_portfolio = Portfolio([]) if portfolio.get_position(leg_name) is None else portfolio.get_position(leg_name)
                self.parameters['roll'][leg_name].roll(self.time_stamp, leg_portfolio)

            # then rebalance if it is a rebalance date
            current_state = RollingFuturesBasketState(self.time_stamp, portfolio, None, None)
            is_rebalance_day = self.parameters["is_rebalance_day_function"]
            if is_rebalance_day(self.strategy, current_state):
                notionals_function = self.parameters["notionals_function"]
                weights = notionals_function(self.strategy, state)
                for leg_name, leg_params in self.parameters['legs'].items():
                    # find the future in this leg
                    key, pos = portfolio.get_position(leg_name).find_children_of_tradable_type(Future)[0]
                    # unwind it
                    current_future_price = market.get_future_data(pos.tradable)['close']
                    portfolio.unwind((leg_name, key), current_future_price)
                    # enter new position with correct sizing
                    units = weights[leg_name] / current_future_price
                    portfolio.trade(pos.tradable, units, current_future_price, position_path=(leg_name,))

        # cost
        price_pre_cost = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)
        for leg_name, leg_params in self.parameters['legs'].items():
            leg_portfolio_before = Portfolio([]) if state.portfolio.get_position(leg_name) is None else state.portfolio.get_position(leg_name)
            leg_portfolio_after = Portfolio([]) if portfolio.get_position(leg_name) is None else portfolio.get_position(leg_name)
            self.parameters['trading_cost'][leg_name].apply(leg_portfolio_after, leg_portfolio_before, reprice=True, market=market, valuer_map=self.strategy.valuer_map)
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)

        return RollingFuturesBasketState(self.time_stamp, portfolio, price, price - price_pre_cost)


class RollingFuturesBasket(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.valuer_map = {
            Future: FutureDataValuer('close'),
        }

        self.parameters['roll'] = {}
        self.parameters['trading_cost'] = {}
        for leg_name, leg_params in self.parameters['legs'].items():
            self.parameters['roll'][leg_name] = RollFutureContracts(self.backtest_market, self.valuer_map, 'close', leg_params['roll_offset'], 'last tradable date', self.holidays)
            self.parameters['trading_cost'][leg_name] = FlatTradingCost(leg_params['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
