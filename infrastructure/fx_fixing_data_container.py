from datetime import datetime
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..infrastructure.fx_pair import FXPair
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
import pandas as pd


class FXFixingDataContainer(DataContainer):
    def __init__(self, pair: FXPair):
        self.market_key = market_utils.create_fx_fixing_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_fixing_data(self, dt=None):
        return self._get_fx_fixing_data(dt)

    def get_market_item(self, dt):
        return self.get_fx_fixing_data(dt)


class FXFixingDataRequest(IDataRequest):
    def __init__(self, start_date: datetime, end_date: datetime, pair: FXPair, fixing_type: str):
        self.start_date = start_date
        self.end_date = end_date
        self.pair = pair
        self.fixing_type = fixing_type

    def get_bbg_ticker(self):
        return f'{self.pair.to_string()} {self.fixing_type} Curncy'


class DatalakeBBGFXFixingDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        pair = data_request.pair
        bbg_ticker = data_request.get_bbg_ticker()
        df = get_bbg_history([bbg_ticker], 'PX_LAST', data_request.start_date, data_request.end_date)
        df["date"] = pd.to_datetime(df["date"])
        self.data = df[["date", "PX_LAST"]].rename(columns={"PX_LAST": "fixing"}).set_index("date")

        def get_fx_fixing_data(dt):
            if dt is None:
                return self.data
            else:
                return self.data[self.data.index < dt]

        container = FXFixingDataContainer(pair)
        container.get_fx_fixing_data = get_fx_fixing_data
        return container
