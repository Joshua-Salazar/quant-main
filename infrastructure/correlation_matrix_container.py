from ..data import datalake
from ..infrastructure import market_utils
from ..infrastructure.correlation_matrix import CorrelationMatrix
from ..infrastructure.data_container import DataContainer
from ..interface.idatasource import IDataSource
from ..interface.idatarequest import IDataRequest

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
idx = pd.IndexSlice


class CorrelationMatrixDataContainer(DataContainer):

    def __init__(self, tickers):
        self.tickers = tickers
        self.market_key = market_utils.create_correlation_matrix(self.tickers)

    def get_market_key(self):
        return self.market_key

    def get_correlation_martrix(self, dt):
        return self._get_correlation_martrix(dt)

    def get_market_item(self, dt):
        return self.get_correlation_martrix(dt)


class CorrelationMatrixDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, tickers, correlation_shift=0):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.correlation_shift = correlation_shift


class ConstantCorrelationMatrixDataSource(IDataSource):
    def __init__(self):
        self.ret_data = None
        self.correlation_shift = 0

    def initialize(self, data_request):
        tickers = data_request.tickers
        # last 5 years
        st = data_request.start_date - relativedelta(years=5) - relativedelta(days=4)
        ts = datalake.get_bbg_history(tickers, "PX_LAST", st, data_request.end_date)
        ts["date"] = pd.to_datetime(ts["date"])
        self.ret_data = np.log(ts.pivot_table(index="date", columns="ticker", values="PX_LAST")).diff()
        self.correlation_shift = data_request.correlation_shift
        container = CorrelationMatrixDataContainer(tickers)
        container._get_correlation_martrix = self._get_correlation_martrix
        return container

    def _get_correlation_martrix(self, dt):
        # 5 year return
        st = dt - relativedelta(years=5)
        ret = self.ret_data.loc[st:dt]
        # 6 months rolling windows
        corr_ts = ret.rolling(int(252/2)).corr()
        corr_ts = corr_ts.dropna()
        unds = corr_ts.columns.values
        corr_matrix = pd.DataFrame(index=unds, columns=unds)
        for und1 in unds:
            for und2 in unds:
                if und1 == und2:
                    corr_matrix.loc[und1, und2] = 1
                else:
                    corr_matrix.loc[und1, und2] = min(corr_ts.loc[idx[:, und1], und2].quantile(0.5) + self.correlation_shift, 1)
        # corr_matrix.loc["NDX Index", "RTY Index"] = 0.7898
        # corr_matrix.loc["RTY Index", "NDX Index"] = 0.7898
        # corr_matrix.loc["NDX Index", "SPX Index"] = 0.9303
        # corr_matrix.loc["SPX Index", "NDX Index"] = 0.9303
        # corr_matrix.loc["SPX Index", "RTY Index"] = 0.8515
        # corr_matrix.loc["RTY Index", "SPX Index"] = 0.8515

        return CorrelationMatrix(corr_matrix)


class FlatCorrelationMatrixDataSource(IDataSource):
    def __init__(self, corr):
        self.corr = corr
        self.tickers = None
        self.correlation_shift = 0
        self.validate()

    def validate(self):
        if isinstance(self.corr, pd.DataFrame):
            if not self.corr.index.equals(self.corr.columns):
                raise Exception("Correlation matrix must have the same ordered indices and columns")
            for und in self.corr.index:
                if self.corr.loc[und, und] != 1:
                    raise Exception("Correlation diag matrix must be 1")
            pos_def = np.all(np.linalg.eigvals(self.corr) > 0)
            if not pos_def:
                raise Exception("Correlation matrix must positive definite")
            self.tickers = self.corr.index.to_list()

    def initialize(self, data_request):
        tickers = data_request.tickers
        if self.tickers is None:
            self.tickers = data_request.tickers
            self.corr = pd.DataFrame(self.corr, index=self.tickers, columns=self.tickers)
            for und in self.tickers:
                self.corr.loc[und, und] = 1
        else:
            if not np.all([x in self.tickers for x in data_request.tickers]):
                raise Exception("Required tickers not all in correlation matrix")
        self.correlation_shift = data_request.correlation_shift
        container = CorrelationMatrixDataContainer(tickers)
        container._get_correlation_martrix = self._get_correlation_martrix
        return container

    def _get_correlation_martrix(self, dt):
        corr_matrix = pd.DataFrame(index=self.tickers, columns=self.tickers)
        for und1 in self.tickers:
            for und2 in self.tickers:
                if und1 == und2:
                    corr_matrix.loc[und1, und2] = 1
                else:
                    corr_matrix.loc[und1, und2] = self.corr.loc[und1, und2] + self.correlation_shift
        return CorrelationMatrix(corr_matrix)