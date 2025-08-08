from datetime import datetime
import pandas as pd
import numpy as np
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_items.ispot import ISpot


class IndexDataContainer(DataContainer, ISpot):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_spot_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_index_data(self, dt):
        return self._get_index_data(dt)

    def get_spot(self, base_date: datetime) -> float:
        """
        :param dt: current date for spot value
        :return: scalar spot value
        """
        return self.get_index_data(base_date)[self.ticker]

    def get_market_item(self, dt):
        return self.get_spot(dt)


class IndexDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker


class DatalakeBBGIndexDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = get_bbg_history([ticker], 'PX_LAST', data_request.start_date, data_request.end_date)
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in data_df['date'].values],
                                  [{ticker: x}for x in data_df['PX_LAST'].values]))

        def _get_index_data(dt):
            return self.data_dict[dt]

        container = IndexDataContainer(ticker)
        container._get_index_data = _get_index_data
        return container


class BPipeBBGIndexDataSource(IDataSource):
    def __init__(self, bpipe_get_bbg_history):
        self.data_dict = {}
        self.bpipe_get_bbg_history = bpipe_get_bbg_history

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = self.bpipe_get_bbg_history(ticker, ['PX_LAST'], data_request.start_date, data_request.end_date)
        self.data_dict = dict(zip([pd.to_datetime(x) for x in data_df['date'].values],
                                  [{ticker: x}for x in data_df['PX_LAST'].values]))

        def _get_index_data(dt):
            return self.data_dict.get(dt, {ticker: np.nan})

        container = IndexDataContainer(ticker)
        container._get_index_data = _get_index_data
        return container
