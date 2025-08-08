from datetime import datetime, timedelta

from ...backtest.costs import FlatTradingCost
from ...backtest.rolls import RollFutureContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.future import Future
from ...dates.schedules import MonthDaySchedule
from ...valuation.future_data_valuer import FutureDataValuer


class FuturesSpreadState(StrategyState):
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
            # initial positions
            for leg_side in ['long', 'short']:
                # find first future, but if we are too close to expiry, we trade in the next future
                current_future = self.parameters['roll'].find_initial_contract(self.time_stamp, root=self.parameters[leg_side]['underlying'])

                # handle missing data
                current_futures = market.get_future_universe(current_future.root)
                ix = 1
                date_with_data = self.time_stamp
                while current_future.listed_ticker not in current_futures:
                    print( 'no data for %s on %s' %(current_future.listed_ticker, self.time_stamp.date() ) )
                    current_futures = self.strategy.backtest_market.get_market(self.time_stamp - timedelta(days=ix)).get_future_universe(current_future.root)
                    date_with_data = self.time_stamp - timedelta(days=ix)
                    ix += 1
                current_future_price = self.strategy.backtest_market.get_market(date_with_data).get_future_data(current_future)['close']

                #current_future_price = market.get_future_data(current_future)['close']

                units = self.parameters[leg_side]['target_notional'] / current_future_price
                portfolio.trade(current_future, units, current_future_price, position_path=(leg_side,))
        else:
            # roll if need to
            self.parameters['roll'].roll(self.time_stamp, portfolio)

            # then rebalance if it is a rebalance date
            if self.time_stamp in self.parameters['rebalance_days']:
                for leg_side in ['long', 'short']:
                    # find the future in this leg
                    key, pos = portfolio.get_position(leg_side).find_children_of_tradable_type(Future)[0]
                    # unwind it
                    current_future_price = market.get_future_data(pos.tradable)['close']
                    portfolio.unwind((leg_side, key), current_future_price)
                    # enter new position with correct sizing
                    units = self.parameters[leg_side]['target_notional'] / current_future_price
                    portfolio.trade(pos.tradable, units, current_future_price, position_path=(leg_side,))

        # cost
        price_pre_cost = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)
        self.parameters['trading_cost'].apply(portfolio, state.portfolio, reprice=True, market=market, valuer_map=self.strategy.valuer_map)
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)

        return FuturesSpreadState(self.time_stamp, portfolio, price, price - price_pre_cost)


class FuturesSpread(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.valuer_map = {
            Future: FutureDataValuer('close'),
        }

        self.parameters['rebalance_days'] = MonthDaySchedule(15, 'following', self.holidays).schedule_days(self.start_date, self.end_date)
        self.parameters['roll'] = RollFutureContracts(self.backtest_market, self.valuer_map, 'close', self.parameters['roll_offset'], 'last tradable date', self.holidays)
        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
