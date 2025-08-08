from datetime import datetime
from ..data.datalake import DATALAKE
from ..data.market import get_expiry_in_year
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..infrastructure.fx_pair import FXPair
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
import pandas as pd


class FXFwdPointDataContainer(DataContainer):
    def __init__(self, pair: FXPair):
        self.market_key = market_utils.create_fx_fwd_point_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_fwd_point(self, dt=None):
        return self._get_fx_fwd_point(dt)

    def get_market_item(self, dt):
        return self.get_fx_fwd_point(dt)


class FXFwdPointDataRequest(IDataRequest):
    def __init__(self, start_date: datetime, end_date: datetime, pair: FXPair):
        self.start_date = start_date
        self.end_date = end_date
        self.pair = pair

    def get_bbg_tickers(self):
        tenors = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "15M", "18M", "2Y", "3Y", "4Y", "5Y", "6Y",
                  "7Y", "8Y", "9Y", "10Y"]
        tickers = ""
        for tenor in tenors:
            tickers += f"FX.FORWARD.FWD_POINT.{self.pair.base_ccy.value}.{self.pair.term_ccy.value}.{tenor}.CITI,"
        return tickers


class DatalakeCitiFXFwdPointDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        tickers = data_request.get_bbg_tickers()
        df = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
        if not df.empty:
            df["tstamp"] = pd.to_datetime(df["tstamp"])
            df = df.rename(columns={"ON": "1D"})
            df["tenor"] = df.ticker.apply(lambda x: get_expiry_in_year(x.replace("ON", "1D").split('.')[-2][:-1],
                                                                       x.replace("ON", "1D").split('.')[-2][-1]))
            self.data_dict = {dt: grp.loc[dt, "VALUE"].to_dict() for dt, grp in
                              df.set_index(["tstamp", "tenor"]).groupby(level="tstamp")}

        container = FXFwdPointDataContainer(data_request.pair)

        def _get_fx_fwd_point(dt):
            if dt is None:
                return self.data_dict
            else:
                # allowed missing since we only have short history for fwd points
                return self.data_dict.get(dt, None)

        container._get_fx_fwd_point = _get_fx_fwd_point
        return container
