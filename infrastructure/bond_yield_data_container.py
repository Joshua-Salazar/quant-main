from datetime import datetime
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource


class BondYieldDataContainer(DataContainer):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_bond_yield_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_bond_yield(self, dt):
        return self._get_bond_yield(dt)

    def get_market_item(self, dt):
        return self.get_bond_yield(dt)


class BondYieldDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker


class DatalakeBBGBondYieldDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = get_bbg_history([ticker], 'PX_LAST', data_request.start_date, data_request.end_date)
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in data_df['date'].values],
                                  [{ticker: x}for x in data_df['PX_LAST'].values]))

        def _get_bond_yield(dt):
            return self.data_dict[dt]

        container = BondYieldDataContainer(ticker)
        container._get_bond_yield = _get_bond_yield
        return container
