import pandas as pd
import numpy as np
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_item import MarketItem
from ..infrastructure import market_utils
from ..language import dataframe_to_records
from ..interface.ishock import IShock
from ..data.datalake_cassandra import DatalakeCassandra, ExpiryFilterByDateRange, StrikeFilterByValue, \
    CallPutFilterByValue, ExerciseFilterByValue, ExpiryFilterByDateOffset, ExpiryFilterByDate, StrikeFilterByRange, \
    CallPutFilterByList


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


class OptionMinuteDataRequest(IDataRequest):
    def __init__(self, root, expiry_filter=ExpiryFilterByDateOffset(max_expiry_offset=365),
                 strike_filter=StrikeFilterByRange(lb=0, ub=10000),
                 call_put_filter=None,
                 exercise_filter=ExerciseFilterByValue('E'),
                 market_start='9:00:00',
                 market_end='16:00:00',):
        self.root = root
        self.expiry_filter = expiry_filter
        self.strike_filter = strike_filter
        self.call_put_filter = call_put_filter
        self.exercise_filter = exercise_filter
        self.market_start = market_start
        self.market_end = market_end

        super().__init__()


class OptionMinuteDataContainer(MarketItem):

    def __init__(self, root: str):
        self.market_key = market_utils.create_option_data_container_key(root)

    def get_market_key(self):
        return self.market_key

    def get_option_universe(self, dt, return_as_dict):
        return self._get_option_universe(dt)

    def get_option_data(self, dt, option, return_as_dict=False):
        return self._get_option_data(dt, option, return_as_dict)

    def get_option_price_based_on_ticker(self, dt, listed_ticker):
        return self._get_option_price_based_on_ticker(dt, listed_ticker)

    def get_option_series(self, start_date, end_date, option):
        return self._get_option_series(start_date, end_date, option)

    def get_market_item(self, dt):
        return self

    def apply(self, shocks: [IShock], **kwargs) -> MarketItem:
        raise Exception("Not implemented yet")


class CassandraOptionMinuteDataScource(IDataSource):
    def __init__(self):
        super().__init__()
        self.universe = {}
        self.data_dict = {}
        self.cassandra = DatalakeCassandra()

    def initialize(self, data_request):

        data_container = OptionMinuteDataContainer(data_request.root)

        def _get_option_universe(dt):

            # expiry_filter=ExpiryFilterByDateRange(dt,(dt + datetime.timedelta(days=self.max_expiry_offset)))
            if dt not in self.universe:
                self.universe = {dt: {}}

            if data_request.root not in self.universe[dt]:

                data = self.cassandra.get_intraday_option_universe(dt=dt,
                                                                   root_security=data_request.root,
                                                                   expiry_filter=data_request.expiry_filter,
                                                                   strike_filter=data_request.strike_filter,
                                                                   call_put_filter=data_request.call_put_filter,
                                                                   exercise_filter=data_request.exercise_filter, )

                data.index = data['option_symbol']
                data.loc[:, 'price'] = data[['price_ask', 'price_bid']].mean(axis=1)
                self.universe[dt][data_request.root] = data

                return self.universe[dt][data_request.root]

            else:

                return self.universe[dt][data_request.root]

        def _get_option_data(dt, option, return_as_dict):

            if dt not in self.data_dict:
                self.data_dict[dt] = {}

            if option.listed_ticker not in self.data_dict[dt]:

                data = self.cassandra.get_intraday_option(dt, option, data_request.market_start,
                                                          data_request.market_end)
                for minute_data in data:

                    if minute_data['t_date'] not in self.data_dict:
                        self.data_dict[minute_data['t_date']] = {}
                    self.data_dict[minute_data['t_date']][option.listed_ticker] = minute_data

                return self.data_dict[dt][option.listed_ticker]

            else:

                return self.data_dict[dt][option.listed_ticker]

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        return data_container


if __name__ == '__main__':
    import datetime
    from ..tradable.option import Option

    request = OptionMinuteDataRequest(root='SPX', frequency='15T')

    source = CassandraOptionMinuteDataScource()

    data_container = source.initialize(request)

    df = data_container.get_option_universe(datetime.datetime(2023, 1, 23, 9, 0))

    print(df)

    option = Option('SPX', 'SPX Index', 'USD', datetime.datetime(2023, 1, 24, 0, 0), 5000, True, False, 1, None,
                    'SPX Index2023-01-24T00:00:001000.00000000CallEUSD1')

    df = data_container.get_option_series(datetime.datetime(2023, 1, 23, 9, 0), datetime.datetime(2023, 1, 23, 10),
                                          option)

    print(df)



