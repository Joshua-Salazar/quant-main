from datetime import datetime
from ..dates.utils import add_tenor
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem
from ..language import format_isodate
from ..tradable.future import Future
from ..data.datalake_cassandra import DatalakeCassandra
from ..dates.utils import add_business_days
import os
import pandas as pd
import time

FUTURE_MONTH_CODE = {
    'F': 1,
    'G': 2,
    'H': 3,
    'J': 4,
    'K': 5,
    'M': 6,
    'N': 7,
    'Q': 8,
    'U': 9,
    'V': 10,
    'X': 11,
    'Z': 12,
}


def expiration_months_number_from_skip_months(skip_months):
    skip_months_number = [FUTURE_MONTH_CODE[x] for x in skip_months]
    return list(filter(lambda x: x not in skip_months_number, FUTURE_MONTH_CODE.values()))


class FutureMinuteDataRequest(IDataRequest):
    def __init__(self, root, suffix, market_start='9:00:00',
                 market_end='16:00:00', schedule=None, rolling_field='last_tradeable_dt'):
        self.root = root
        self.suffix = suffix
        self.schedule = schedule
        self.market_start = market_start
        self.market_end = market_end
        self.rolling_field = rolling_field
        super().__init__()


class FutureMinuteDataContainer(MarketItem):
    def __init__(self, underlying: str, ref_data,rolling_field):
        self.market_key = market_utils.create_future_data_container_key(underlying)
        self.ref_data = ref_data
        self.rolling_field = rolling_field

    def get_market_key(self):
        return self.market_key

    # def get_lead_future_price(self, s_dt, e_dt):
    #     return self._get_lead_future_price(s_dt, e_dt)

    # def get_lead_future(self):
    #     return self._get_lead_future(self.base_datetime)
    
    # taking account of holidays, different field for offset columns
    def get_lead_future(self, dt, offset, holidays=[]):
        dt_v = add_business_days(dt, offset, holidays)
        start = time.time()
        data_short = self.ref_data[self.ref_data[self.rolling_field] >= dt_v].iloc[0]
        
        return Future(data_short['root'], data_short['crncy'],
                      data_short[self.rolling_field], data_short['fut_exch_name_short'],
                      data_short['ticker'])

    def apply(self, shocks: [IShock], **kwargs) -> MarketItem:
        raise Exception("Not implemented yet")

    def get_market_item(self, dt):
        def get_future_data(future, return_as_dict=True):
            return self._get_future_data(dt, future)

        self.get_future_data = get_future_data

        return self

    def get_future_data_for_date(self, dt, future):
        return self._get_future_data(dt, future, True)


class CassandraFutureMinuteDataSource(IDataSource):
    def __init__(self):
        super().__init__()
        self.cassandra = DatalakeCassandra()
        self.data_dict = {}

    def initialize(self, data_request):

        self.current_minute = datetime(1900, 1, 1)
        ref_data = self.cassandra.get_future_ref(data_request.root, data_request.suffix)
        ref_data[data_request.rolling_field] = pd.to_datetime(ref_data[data_request.rolling_field])
        ref_data = ref_data.sort_values(data_request.rolling_field)
        data_container = FutureMinuteDataContainer(data_request.root, ref_data, data_request.rolling_field)
        root = data_request.root
        calendar = data_request.schedule

        # def _get_lead_future_price(s_dt, e_dt):
        #     data = self.cassandra.get_lead_future_series(underlying, s_dt, e_dt, calendar)
        #     return data

        def _get_future_data(dt, future, return_whole_day=False):

            end_date=ref_data[ref_data['ticker'] == future.listed_ticker][data_request.rolling_field].item()

            if future.listed_ticker not in self.data_dict:

                data = self.cassandra.get_intraday_future_series(future, dt, end_date, data_request.market_start, data_request.market_end)

                self.data_dict[future.listed_ticker] = data

            if return_whole_day:
                return self.data_dict[future.listed_ticker]
            else:
                return self.data_dict[future.listed_ticker][dt]

        # data_container._get_lead_future_price = _get_lead_future_price
        # data_container._get_lead_future = _get_lead_future
        data_container._get_future_data = _get_future_data
        return data_container


if __name__ == '__main__':
    from ..tradable.future import Future
    from datetime import datetime

    request = FutureMinuteDataRequest(root='ES',suffix='Index')

    source = CassandraFutureMinuteDataSource()

    data_container = source.initialize(request)

    future = Future('ES', 'USD', datetime(2023, 3, 17), "CME", "ESH23 Index")

    df = data_container._get_future_data(datetime(2023, 1, 23, 9, 0), future)

    print(df)