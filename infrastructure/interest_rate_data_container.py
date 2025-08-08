from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem
from ..language import format_isodate
from ..data.datalake_cassandra import DatalakeCassandra


class InterestRateDataContainer(DataContainer):
    def __init__(self, currency):
        self.market_key = market_utils.create_interest_rate_data_container_key(currency)

    def get_market_key(self):
        return self.market_key

    def get_interest_rate(self, dt, tenor):
        return self._get_interest_rate(dt, tenor)

    def get_market_item(self, dt):
        def _get_interest_rate(tenor):
            return self.get_interest_rate(dt, tenor)
        return InterestRateData(self.get_market_key(), _get_interest_rate)


class InterestRateDataRequest(IDataRequest):
    def __init__(self, currency):
        self.currency = currency

class IVOLInterestRateDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}
        self.dlc = DatalakeCassandra()

    def initialize(self, data_request):
        # market data
        currency = data_request.currency
        container = InterestRateDataContainer(currency)

        def _get_interest_rate(dt, tenor: int):
            dt = format_isodate(dt)
            if dt not in self.data_dict:
                self.data_dict[dt] = {}
            if tenor not in self.data_dict[dt]:
                self.data_dict[dt][tenor] = self.dlc.get_ivol_interest_rate(currency=currency, tenor=tenor, date=dt)

            return self.data_dict[dt][tenor]

        container._get_interest_rate = _get_interest_rate

        return container


# we make distinction between DataContainers and MarketItem as for some case data container has too much data
# and it is less efficient to have it in market whereas MarketItem can contain only one day worth of data
# each data container should have a get_market_item function, the return of which will be contained in the market object
class InterestRateData(MarketItem):
    def __init__(self, market_key, data_func):
        self.market_key = market_key
        self.get_interest_rate = data_func

    def get_market_key(self):
        return self.market_key

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        raise Exception("Not implemented yet")
