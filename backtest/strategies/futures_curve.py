from datetime import datetime, timedelta

from ...backtest.costs import FlatTradingCost
from ...backtest.rolls import RollFutureContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.future import Future
from ...dates.schedules import MonthDaySchedule
from ...valuation.future_data_valuer import FutureDataValuer


class FuturesCurveState(StrategyState):
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

        if 'skip_months' in self.parameters:
            skip_months = self.parameters[ 'skip_months' ]
        else:
            skip_months = []

        if portfolio.is_empty():
            # initial positions
            for leg_side in ['long', 'short']:
                # target future, but if we are too close to expiry, we trade in the next future
                current_future = self.parameters['roll'].find_target_contract(self.time_stamp, target_tenor = self.parameters[ leg_side ][ 'target_future' ], root = self.parameters[ leg_side ][ 'underlying' ], skip_months = skip_months )

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

                units = self.parameters[leg_side]['target_notional'] / current_future_price
                portfolio.trade(current_future, units, current_future_price, position_path=(leg_side,))
        else:
            # roll if need to
            long_target = self.parameters[ 'long' ][ 'target_future' ]
            short_target = self.parameters[ 'short' ][ 'target_future' ]
            all_pos = portfolio.net_positions()
            all_fut = { k:v for k,v in all_pos.items() if isinstance( v.tradable, Future ) }
            min_exp = min( [ x.tradable.expiration for x in all_fut.values() ] )
            min_fut = [v for v in all_fut.values() if v.tradable.expiration == min_exp][0]
            # use <min_fut> expiration to roll long and short legs synchronously
            self.parameters['roll'].roll_target( self.time_stamp, portfolio, long_target, short_target, min_fut, skip_months )

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

        # add fx
        def fx(tradable):
            if tradable.currency == self.strategy.currency:
                fx = 1.0
            else:
                fx_pair_name = f'{tradable.currency}{self.strategy.currency}'
                fx = market.get_fx_spot(fx_pair_name)
            return {'fx': fx}

        # TODO: simplify the pricing with fx on portfolios

        # cost
        portfolio.value_positions(fx)
        portfolio.value_positions_at_market(market, ['price'], valuer_map_override=self.strategy.valuer_map)
        price_pre_cost = portfolio.aggregate(lambda x: getattr(x, 'price') * getattr(x, 'fx'))
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)

        portfolio.value_positions(fx)
        portfolio.value_positions_at_market(market, ['price'], valuer_map_override=self.strategy.valuer_map)
        price = portfolio.aggregate(lambda x: getattr(x, 'price') * getattr(x, 'fx'))

        return FuturesCurveState(self.time_stamp, portfolio, price, price - price_pre_cost)


class FuturesCurve(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        self.valuer_map = {
            Future: FutureDataValuer('close'),
        }

        self.parameters['rebalance_days'] = MonthDaySchedule( 25, 'following', self.holidays).schedule_days(self.start_date, self.end_date)
        self.parameters['roll'] = RollFutureContracts(self.backtest_market, self.valuer_map, 'close', self.parameters['roll_offset'], 'last tradable date',
                                                      DailyStrategy.calculate_holidays(self.calendar, self.start_date, self.end_date + timedelta(days=self.parameters['roll_offset'] * 2)))
        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
