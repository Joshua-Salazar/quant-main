import numbers
from datetime import datetime

from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.equity_data_container import EquityDataRequest, EquityPricesDataSource
from ...tradable.cash import Cash
from ...tradable.portfolio import Portfolio
from ...dates.utils import bdc_adjustment
from ...tradable.stock import Stock
from ...constants.business_day_convention import BusinessDayConvention
from ...analytics.utils import float_equal
from ...valuation.stock_data_valuer import StockDataValuer


class BasketState(StrategyState):
    def __init__(self, time_stamp, portfolio, price):
        self.price = price
        super().__init__(time_stamp, portfolio)


class BasketRebalanceEvent(Event):
    def equal_weights(self, dt):
        w = {}
        live_constituent_names = list(filter(lambda x: x[1]['start_date'] <= dt <= x[1]['end_date'], self.strategy.parameters['constituents'].items()))
        live_constituent_names = [x[0] for x in live_constituent_names]
        n = len(live_constituent_names)
        for k, v in self.strategy.parameters['constituents'].items():
            if k in live_constituent_names:
                w[k] = 1 / n
            else:
                w[k] = 0
        return w

    def is_rebalance_date(self, dt):
        if isinstance(self.strategy.parameters['rebalance_schedule'], tuple):
            if self.strategy.parameters['rebalance_schedule'][0] == 'monthly':
                day_of_month = self.strategy.parameters['rebalance_schedule'][1]
                this_month_rebalance_day = datetime(dt.year, dt.month, day_of_month)
                this_month_rebalance_day = bdc_adjustment(this_month_rebalance_day, convention=BusinessDayConvention.FOLLOWING, holidays=self.strategy.holidays)
                if dt.date() == this_month_rebalance_day.date():
                    return True
                else:
                    return False
            else:
                RuntimeError(f"Cannot handle rebalance_schedule as tuple with first element being {self.strategy.parameters['rebalance_schedule'][0]}")
        else:
            raise RuntimeError(f"Unknown type of rebalance_schedule")

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        if self.is_rebalance_date(self.time_stamp):
            if self.strategy.parameters['weights_method'] == 'Equal':
                weights_dict = self.equal_weights(self.time_stamp)
            else:
                raise RuntimeError(f"Unknown weights calculation method {self.strategy.parameters['weights_method']}")

            # what weight means depends on what it refers to for basket
            if isinstance(self.strategy.parameters['basket_weights_reference'], numbers.Number):
                basket_weights_reference = self.strategy.parameters['basket_weights_reference']
            elif isinstance(self.strategy.parameters['basket_weights_reference'], dict):
                basket_weights_reference = self.strategy.parameters['basket_weights_reference'][self.time_stamp]
            elif isinstance(self.strategy.parameters['basket_weights_reference'], tuple):
                if self.strategy.parameters['basket_weights_reference'][0] == 'nav':
                    basket_weights_reference = self.strategy.parameters['basket_weights_reference'][1] + state.price
                else:
                    raise RuntimeError(f"Cannot handle basket_weights_reference as tuple with first element being {self.strategy.parameters['basket_weights_reference'][0]}")
            else:
                raise RuntimeError(f"Unknown type of basket_weights_reference")

            # trade each underlying
            for name, weight in weights_dict.items():
                # it is only possible to trade an underlying when that underlying is live
                # if the underlying is not live the weight has to be zero
                if not(self.strategy.parameters['constituents'][name]['start_date'] <= self.time_stamp <= self.strategy.parameters['constituents'][name]['end_date']):
                    assert float_equal(weight, 0)
                else:
                    if self.strategy.parameters['look_through']:
                        sub_strategy = self.strategy.data_requests[name][0].strategy
                        currency = sub_strategy.currency
                        sub_portfolio = market.get_portfolio(name)
                        if hasattr(sub_strategy, 'valuer_map'):
                            sub_strategy_valuer_map = sub_strategy.valuer_map
                        else:
                            sub_strategy_valuer_map = {}
                        execution_price = sub_portfolio.price_at_market(market, fields='price', valuer_map_override=sub_strategy_valuer_map, currency=currency)
                    else:
                        currency = self.strategy.parameters['constituents'][name]['currency']
                        tradable = Stock(name, currency)
                        execution_price = tradable.price(market, calc_types='price', valuer=self.strategy.valuer_map[type(tradable)], currency=currency)

                    # what weight means depends on what it refers to for constituents
                    if isinstance(self.strategy.parameters['constituent_weights_reference'], numbers.Number):
                        constituent_weights_reference = self.strategy.parameters['constituent_weights_reference']
                    elif isinstance(self.strategy.parameters['constituent_weights_reference'], dict):
                        if self.time_stamp in self.strategy.parameters['constituent_weights_reference']:
                            constituent_weights_reference = self.strategy.parameters['constituent_weights_reference'][self.time_stamp][name]
                        elif name in self.strategy.parameters['constituent_weights_reference']:
                            constituent_weights_reference = self.strategy.parameters['constituent_weights_reference'][name][self.time_stamp]
                        else:
                            raise RuntimeError(f"Cannot find the constituent weight reference for {name} on {self.time_stamp}")
                    elif isinstance(self.strategy.parameters['constituent_weights_reference'], tuple):
                        if self.strategy.parameters['constituent_weights_reference'][0] == 'nav':
                            if self.strategy.parameters['look_through']:
                                constituent_price = sub_portfolio.price_at_market(market, fields='price', valuer_map_override=sub_strategy_valuer_map, currency=self.strategy.currency)
                            else:
                                constituent_price = tradable.price(market, calc_types='price', valuer=self.strategy.valuer_map[type(tradable)], currency=self.strategy.currency)
                            constituent_weights_reference = self.strategy.parameters['constituent_weights_reference'][1] + constituent_price
                        else:
                            raise RuntimeError(
                                f"Cannot handle constituent_weights_reference as tuple with first element being {self.strategy.parameters['constituent_weights_reference'][0]}")
                    else:
                        raise RuntimeError(f"Unknown type of constituent_weights_reference")

                    new_units = weight * basket_weights_reference / constituent_weights_reference
                    if 'units_cap' in self.parameters:
                        new_units = min(new_units, self.parameters['units_cap'].get(name, float('inf')))

                    if self.strategy.parameters['look_through']:
                        current_sub_portfolio = portfolio.get_position(name)
                        if current_sub_portfolio is None:
                            current_sub_portfolio_execution_price = 0
                        else:
                            current_sub_portfolio_execution_price = current_sub_portfolio.price_at_market(market, fields='price', valuer_map_override=sub_strategy_valuer_map, currency=currency)
                            # remove current portfolio
                            portfolio.set_position(name, Portfolio([]))
                            portfolio.remove_zero_positions()
                        # add in new portfolio
                        sub_portfolio.scale(new_units)
                        portfolio.set_position(name, sub_portfolio)
                        # add the cash in this transaction
                        portfolio.add_position(Cash(currency), current_sub_portfolio_execution_price - execution_price * new_units)
                    else:
                        pos = portfolio.get_position(tradable.name())
                        if pos is None and abs(new_units) > 0:
                            print(f"{self.time_stamp}: New {name} position entered")
                        trade_units = new_units - (0 if pos is None else pos.quantity)

                        portfolio.trade(tradable, trade_units, execution_price)

        current_portfolio_price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)

        return BasketState(self.time_stamp, portfolio, current_portfolio_price)


class BasketMtMEvent(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        current_portfolio_price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map, currency=self.strategy.currency)

        return BasketState(self.time_stamp, portfolio, current_portfolio_price)


class Basket(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        if self.parameters['look_through']:
            for strategy_name in self.parameters['constituents'].keys():
                sub_strategy = self.data_requests[strategy_name][0].strategy
                sub_strategy_data_containers = sub_strategy.data_containers.values()
                for data in sub_strategy_data_containers:
                    self.backtest_market.add_item(data.get_market_key(), data)
        else:
            for strategy_name in self.parameters['constituents'].keys():
                states = self.data_containers[strategy_name].get_strategy_states()
                data_request = EquityDataRequest(
                    self.parameters['constituents'][strategy_name]['start_date'],
                    self.parameters['constituents'][strategy_name]['end_date'],
                    self.calendar, strategy_name)
                data_source = EquityPricesDataSource([x.time_stamp for x in states], [x.price for x in states])
                self.data_requests[strategy_name + "_price"] = (data_request, data_source)
                data = data_source.initialize(data_request)
                self.data_containers[strategy_name + "_price"] = data
                self.backtest_market.add_item(data.get_market_key(), data)

        self.valuer_map = self.parameters.get("valuer_map", {
            Stock: StockDataValuer(),
        })

    def generate_events(self, dt: datetime):
        return [BasketRebalanceEvent(dt, self), BasketMtMEvent(dt, self)]
