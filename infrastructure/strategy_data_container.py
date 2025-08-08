from datetime import datetime

from ..backtest.backtester import LocalBacktester
from ..backtest.strategy import StrategyState
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..tradable.portfolio import Portfolio
from ..tradable.position import Position
from ..tradable.stock import Stock


class StrategyDataContainer(DataContainer):
    def __init__(self, name=None):
        self.name = name
        self.market_key = market_utils.create_portfolio_key(name)

    def get_market_key(self):
        return self.market_key

    def get_strategy_prices(self, dt=None):
        return self._get_strategy_prices(dt)

    def get_strategy_states(self, dt=None):
        return self._get_strategy_states(dt)

    def get_spot(self, base_date: datetime) -> float:
        """
        :param dt: current date for spot value
        :return: scalar spot value
        """
        return self.get_strategy_prices(base_date)

    def get_market_item(self, dt):
        state = self._get_strategy_states(dt)
        return state.portfolio


class StrategyDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, name=None, currency=None, strategy=None, initial_state=None):
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.currency = currency
        self.strategy = strategy
        self.initial_state = initial_state


class StrategyRunDataSource(IDataSource):
    def __init__(self):
        self.runner = LocalBacktester()
        self.data_dict = {}

    def initialize(self, data_request):
        states_series = self.runner.run(data_request.strategy, data_request.start_date, data_request.end_date, data_request.initial_state)
        return StrategyStatesDataSource.initialize_from_states(data_request.name, states_series, self.data_dict)


class StrategyStateWithPrice(StrategyState):
    def __init__(self, time_stamp, portfolio, price):
        self.price = price
        super().__init__(time_stamp, portfolio)


class StrategyPricesDataSource(IDataSource):
    def __init__(self, dates, prices):
        self.prices = prices
        self.dates = dates
        self.data_dict = dict(zip(dates, prices))

    def initialize(self, data_request):
        def _get_strategy_prices(dt):
            if dt is None:
                return self.prices
            else:
                return self.data_dict.get(dt, None)

        def _get_strategy_states(dt):
            if dt is None:
                return [StrategyStateWithPrice(d, Portfolio([Position(Stock(data_request.name, data_request.currency), 1)]), p) for d, p in zip(self.dates, self.prices)]
            else:
                return StrategyStateWithPrice(dt, Portfolio([Position(Stock(data_request.name, data_request.currency), 1)]), self.data_dict.get(dt, None))

        container = StrategyDataContainer(data_request.name)
        container._get_strategy_prices = _get_strategy_prices
        container._get_strategy_states = _get_strategy_states

        return container


class StrategyStatesDataSource(IDataSource):
    def __init__(self, states_series):
        self.states_series = states_series
        self.data_dict = {}

    @staticmethod
    def initialize_from_states(name, states_series, data_dict):
        dates = [x.time_stamp for x in states_series]
        data_dict = dict(zip(dates, states_series))

        def _get_strategy_prices(dt):
            if dt is None:
                return [x.price for x in states_series]
            else:
                return data_dict[dt].price if dt in data_dict else None

        def _get_strategy_states(dt):
            if dt is None:
                return states_series
            else:
                return data_dict[dt]

        container = StrategyDataContainer(name)
        container._get_strategy_prices = _get_strategy_prices
        container._get_strategy_states = _get_strategy_states

        return container

    def initialize(self, data_request):
        return StrategyStatesDataSource.initialize_from_states(data_request.name, self.states_series, self.data_dict)
