from ..infrastructure.data_container import DataContainer
from ..infrastructure.option_data_container import OptionDataContainer
from ..infrastructure.market import Market
from datetime import timedelta


class GenericMarketContainer(DataContainer):
    def __init__(self, item):
        self.item = item

    def get_market_key(self):
        pass

    def get_market_item(self, dt):
        return self.item


class BMarket(Market):
    """
    Backtest market
    It stores time series of data mostly in the form of DataContainers
    get_market(dt) will return the Market for a given date
    """

    def __init__(self):
        self.storage = {}

    @staticmethod
    def from_market(dt, market):
        backtest_market = BMarket()
        for key, item in market.storage.items():
            if not backtest_market.has_item(key):
                backtest_market.add_item(key, GenericMarketContainer(item))
        return backtest_market

    def get_market_with_prev(self, dt, num_prev_days=14):
        market = Market(dt)
        market.storage = {}

        keys = list(self.storage.keys())
        for k in keys:
            v = self.storage[k]
            # TODO: move getting market data for particular day outside Market for Option Data Container
            if not isinstance(v, OptionDataContainer):
                if v.get_market_item(dt) is not None:
                    market.storage[k] = v.get_market_item(dt)
                else:
                    # take previous day's value if not available
                    item = None
                    prev_dt = dt - timedelta(days=1)
                    num_days = 1
                    while item is None and num_days <= num_prev_days:
                        item = v.get_market_item(prev_dt)
                        prev_dt = prev_dt - timedelta(days=1)
                        num_days = num_days + 1
                    if item is not None:
                        market.storage[k] = item
            else:
                market.storage[k] = v

        return market

    def get_market(self, dt, filter_keys=[], force_keys=[]):
        market = Market(dt)
        market.storage = {}

        keys = list(self.storage.keys()) if len(filter_keys) == 0 else filter_keys
        for k in keys:
            v = self.storage[k]
            # TODO: move getting market data for particular day outside Market for Option Data Container
            if not isinstance(v, OptionDataContainer):
                # temp fix
                if False: #os.getlogin() == 'hstone':
                    if isinstance(v, dict):
                        market.storage[k] = v
                    else:
                        market.storage[k] = v.get_market_item(dt)
                elif v.get_market_item(dt) is not None:
                    market.storage[k] = v.get_market_item(dt)
                else:
                    # allow fallback look up
                    if k in force_keys:
                        item = None
                        max_days = 14
                        prev_dt = dt - timedelta(days=1)
                        num_days = 1
                        while item is None and num_days <= max_days:
                            item = v.get_market_item(prev_dt)
                            prev_dt = prev_dt - timedelta(days=1)
                            num_days = num_days + 1
                        if item is None:
                            raise Exception(f"{k} Cannot find last available data between {prev_dt} and {dt}")
                        market.storage[k] = item
            else:
                market.storage[k] = v

        return market
