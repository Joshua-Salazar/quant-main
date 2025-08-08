from datetime import datetime

import pandas as pd
import math

from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..data.datalake_cassandra import DatalakeCassandra
from ..analytics.symbology import option_root_from_ticker, currency_from_option_root


class SpecialCash:
    def __init__(self, ex_datetime, payment_datetime, ticker, amount, currency):
        self.ex_datetime = ex_datetime
        self.payment_datetime = payment_datetime
        self.ticker = ticker
        self.amount = amount
        self.currency = currency


class Split:
    def __init__(self, ex_datetime, payment_datetime, ticker, r_factor):
        self.ex_datetime = ex_datetime
        self.payment_datetime = payment_datetime
        self.ticker = ticker
        self.r_factor = r_factor


class CorpActionDataContainer(DataContainer):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_corpaction_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_corpaction(self, dt):
        return self._get_corpaction(dt)

    def get_market_item(self, dt):
        return self.get_corpaction(dt)


class CorpActionDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker


class CorpActionDataSourceFile(IDataSource):
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = pd.read_csv(self.data_file)
        data_df = data_df[data_df['ticker'] == ticker]

        self.data_dict = {}
        grouped = data_df.groupby('ex_date')
        for group_name, group in grouped:
            ex_date = group_name
            corpactions = []
            for _, row in group.iterrows():
                if row['type'] == 'Special Cash':
                    corpactions.append(SpecialCash(row['ex_date'], row['payment_date'], ticker, row['amount'], row['currency']))
                elif row['type'] == 'Split':
                    corpactions.append(Split(row['ex_date'], row['payment_date'], ticker, row['r_factor']))
                else:
                    raise RuntimeError(f"unsupported corp action type {row['type']}")
            self.data_dict[ex_date] = corpactions

        def _get_corpaction(dt):
            if dt in self.data_dict:
                return self.data_dict[dt]
            else:
                return None

        container = CorpActionDataContainer(ticker)
        container._get_corpaction = _get_corpaction
        return container


class IVolCorpActionDataSource(IDataSource):
    def __init__(self):
        self.cassandra = DatalakeCassandra()
        self.data_dict = {}

    def initialize(self, data_request):
        ticker = data_request.ticker

        data_df = self.cassandra.get_corp_actions(option_root_from_ticker(ticker))

        self.data_dict = {}
        if not data_df.empty:
            grouped = data_df.reset_index().groupby('t_date')
            for group_name, group in grouped:
                ex_date = datetime.combine(group_name.to_pydatetime().date(), datetime.min.time())
                corpactions = []
                for _, row in group.iterrows():
                    if row['amount'] is not None and not math.isnan(row['amount']):
                        corpactions.append(SpecialCash(ex_date, ex_date, ticker, row['amount'], currency_from_option_root(option_root_from_ticker(ticker))))
                    else:
                        corpactions.append(Split(ex_date, ex_date, ticker, row['factor']))

                # TODO: hack for various data issue
                if ex_date == datetime(2020, 4, 29) and ticker == 'USO US Equity':
                    ex_date = datetime(2020, 4, 30)
                if ex_date == datetime(2008, 7, 24) and ticker == 'EEM US Equity':
                    ex_date = datetime(2008, 7, 25)
                if ex_date == datetime(2008, 7, 24) and ticker == 'FXI US Equity':
                    ex_date = datetime(2008, 7, 25)
                if ex_date == datetime(2019, 12, 16) and ticker == 'EEM US Equity':
                    ex_date = datetime(2019, 12, 17)
                if ex_date == datetime(2023, 6, 7) and ticker == 'FXI US Equity':
                    ex_date = datetime(2023, 6, 8)

                self.data_dict[ex_date] = corpactions

        def _get_corpaction(dt):
            if dt in self.data_dict:
                return self.data_dict[dt]
            else:
                return None

        container = CorpActionDataContainer(ticker)
        container._get_corpaction = _get_corpaction
        return container
