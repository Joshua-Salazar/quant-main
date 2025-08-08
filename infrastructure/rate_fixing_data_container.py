from ..data.datalake import DATALAKE
from ..constants.ccy import Ccy
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource

import pandas as pd


class RateFixingDataContainer(DataContainer):
    def __init__(self, ccy: Ccy, tenor: str):
        self.market_key = market_utils.create_rate_fixing_key(ccy, tenor)

    def get_market_key(self):
        return self.market_key

    def get_rate_fixing_data(self, dt=None):
        return self._get_rate_fixing_data(dt)

    def get_market_item(self, dt):
        return self.get_rate_fixing_data(dt)


class RateFixingDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, ccy: Ccy, tenor: str):
        if tenor != "3M":
            raise Exception(f"Only support 3M curve but found {tenor}")
        self.tenor = tenor
        self.support_ccy_list = [Ccy.AUD, Ccy.EUR, Ccy.JPY, Ccy.USD]
        if ccy not in self.support_ccy_list:
            raise Exception(f"Not support ccy {ccy}, on support: {','.join(self.support_ccy_list)}")
        self.ccy = ccy
        self.start_date = start_date
        self.end_date = end_date
        self.bbg_ticker_map = {
            "3M": {
                Ccy.AUD: "RATES.SWAP_LIBOR.AUD.PAR.3M", Ccy.EUR: "RATES.SWAP_LIBOR.EUR.PAR.3M",
                Ccy.JPY: "RATES.SWAP_LIBOR.JPY.PAR.3M", Ccy.USD: "RATES.SWAP_LIBOR.USD.PAR.3M",
            }
        }

    def get_bbg_ticker(self):
        return self.bbg_ticker_map[self.tenor][self.ccy]


class DatalakeBBGRateFixingDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        ccy = data_request.ccy
        bbg_ticker = data_request.get_bbg_ticker()
        df = DATALAKE.getData("CITI_VELOCITY", bbg_ticker, 'VALUE', data_request.start_date, data_request.end_date)
        df = df[["tstamp", "VALUE"]].rename(columns={"tstamp": "date", "VALUE": "fixing"})
        df["date"] = pd.to_datetime(df["date"])
        self.data = df.set_index("date")
        def _get_rate_fixing_data(dt):
            if dt is None:
                return self.data
            else:
                return self.data[self.data.index < dt]

        container = RateFixingDataContainer(ccy, data_request.tenor)
        container._get_rate_fixing_data = _get_rate_fixing_data
        return container
