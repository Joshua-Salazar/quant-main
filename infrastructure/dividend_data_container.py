from datetime import datetime

import pandas as pd

from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..data.datalake_cassandra import DatalakeCassandra
from ..analytics.symbology import option_root_from_ticker


class Dividend:
    def __init__(self, ex_datetime, payment_datetime, ticker, amount, currency):
        self.ex_datetime = ex_datetime
        self.payment_datetime = payment_datetime
        self.ticker = ticker
        self.amount = amount
        self.currency = currency


class DividendDataContainer(DataContainer):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_dividend_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_dividend(self, dt):
        return self._get_dividend(dt)

    def get_market_item(self, dt):
        return self.get_dividend(dt)


class DividendDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker


class DividendDataSourceFile(IDataSource):
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = pd.read_csv(self.data_file)
        data_df = data_df[data_df['ticker'] == ticker]

        # TODO: this is assuming each day there can only be one div
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in data_df['ex_date'].values],
                                  [[Dividend(ex_date, payment_date, ticker, amount, currency)]
                                   for ex_date, payment_date, amount, currency in zip(data_df['ex_date'].values, data_df['payment_date'].values, data_df['amount'].values, data_df['currency'].values)]))

        def _get_dividend(dt):
            if dt in self.data_dict:
                return self.data_dict[dt]
            else:
                return None

        container = DividendDataContainer(ticker)
        container._get_dividend = _get_dividend
        return container


class IVolDividendDataSource(IDataSource):
    def __init__(self):
        self.cassandra = DatalakeCassandra()
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker

        self.data_dict = {}
        data_df = self.cassandra.get_divs(option_root_from_ticker(ticker))

        if not data_df.empty:
            ex_dates = [datetime.combine(x.date(), datetime.min.time()) for x in data_df.index.to_pydatetime()]
            self.data_dict = dict(zip(ex_dates,
                                      [[Dividend(ex_date, ex_date, ticker, amount, currency)]
                                       for ex_date, amount, currency in zip(ex_dates, list(data_df['real_last_dvd_amount'].values), list(data_df['dvd_ccy'].values))]))

        def _get_dividend(dt):
            if dt in self.data_dict:
                return self.data_dict[dt]
            else:
                return None

        container = DividendDataContainer(ticker)
        container._get_dividend = _get_dividend
        return container
