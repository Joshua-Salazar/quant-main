from datetime import datetime, timedelta

from ...backtest.costs import FlatTradingCost
from ...backtest.rolls import RollFutureContracts, RollFXForwardContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.FXforward import FXforward
from ...dates.schedules import MonthDaySchedule
from ...valuation.fx_forward_fx_vol_surface_valuer import FXForwardDataValuer


class RollingFXForwardsState(StrategyState):
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

        # initial positions
        if portfolio.is_empty():
            first_contract = self.parameters['roll'].find_next_contract(self.time_stamp, FXforward(self.parameters['underlying'], self.parameters['underlying'], self.parameters['currency'], self.time_stamp, 'America/New_York'))
            forward_price = first_contract.price(market, calc_types='price', currency=self.parameters['currency'])
            quantity = self.parameters['notional'] / forward_price
            portfolio.trade(first_contract, quantity, execution_price=forward_price, execution_currency=self.parameters['currency'])

        # roll if need to
        self.parameters['roll'].roll(self.time_stamp, portfolio)

        # cost
        price_pre_cost = portfolio.price_at_market(market, fields='price')
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)
        price_after_cost = portfolio.price_at_market(market, fields='price')

        return RollingFXForwardsState(self.time_stamp, portfolio, price_after_cost, price_after_cost - price_pre_cost)


class RollingFXForwards(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.parameters['roll'] = RollFXForwardContracts(self.backtest_market, self.parameters['roll_offset'], self.holidays)
        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
