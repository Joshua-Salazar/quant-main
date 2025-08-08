from datetime import datetime
from ..data.datalake import DATALAKE
from ..data.market import get_expiry_in_year
from ..constants.ccy import Ccy
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..infrastructure.fx_pair import FXPair

import pandas as pd


class FXFwdDataContainer(DataContainer):
    def __init__(self, pair):
        if isinstance(pair, FXPair):
            self.pair = pair
        else:
            self.pair = FXPair(base_ccy=Ccy(pair[:3]), term_ccy=Ccy(pair[-3:]))
        self.market_key = market_utils.create_fx_fwd_key(self.pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_fwd(self, dt=None):
        return self._get_fx_fwd(dt)

    def get_market_item(self, dt):
        return self.get_fx_fwd(dt)


class FXFwdDataRequest(IDataRequest):
    def __init__(self, start_date: datetime, end_date: datetime, pair):
        self.start_date = start_date
        self.end_date = end_date
        if isinstance(pair, FXPair):
            self.pair = pair
        else:
            self.pair = FXPair(base_ccy=Ccy(pair[:3]), term_ccy=Ccy(pair[-3:]))

    def get_citi_tickers(self):
        tenors = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y",
                  "7Y", "8Y", "9Y", "10Y"]
        tickers = ""
        for tenor in tenors:
            tickers += f"FX.FORWARD.FWD_OUTRIGHT.{self.pair.base_ccy.value}.{self.pair.term_ccy.value}.{tenor}.CITI,"
        return tickers

    def get_inverse_citi_tickers(self):
        tenors = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y",
                  "7Y", "8Y", "9Y", "10Y"]
        tickers = ""
        for tenor in tenors:
            tickers += f"FX.FORWARD.FWD_OUTRIGHT.{self.pair.term_ccy.value}.{self.pair.base_ccy.value}.{tenor}.CITI,"
        return tickers


class DatalakeCitiFXFwdDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        tickers = data_request.get_citi_tickers()
        df = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
        if df.empty:
            inverse_tickers = data_request.get_inverse_citi_tickers()
            df = DATALAKE.getData('CITI_VELOCITY', inverse_tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
            if df.empty:
                raise RuntimeError(f"cannot find fx forward history for {data_request.pair.base_ccy.value}{data_request.pair.term_ccy.value}")
            else:
                df['VALUE'] = 1 / df['VALUE']
        df["tstamp"] = pd.to_datetime(df["tstamp"])

        df["tenor"] = df.ticker.apply(lambda x: get_expiry_in_year(x.replace("ON", "1D").split('.')[-2][:-1],
                                                                   x.replace("ON", "1D").split('.')[-2][-1]))
        self.data_dict = {dt: grp.loc[dt, "VALUE"].to_dict() for dt, grp in
                          df.set_index(["tstamp", "tenor"]).groupby(level="tstamp")}

        container = FXFwdDataContainer(data_request.pair)

        def _get_fx_fwd(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_fx_fwd = _get_fx_fwd
        return container
