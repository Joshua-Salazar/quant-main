from datetime import datetime, time
from ..data.datalake import DATALAKE
from datalake.datalakeapi import DataLakeAPI

from ..data.datalake_cassandra import DatalakeCassandra
from ..dates.holidays import get_holidays
from ..dates.utils import get_business_days, EXCHANGE_TZ, set_timezone, minus_tenor, add_business_days
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem
from ..language import dict_of_dict_to_dataframe, format_isodate
from ..tradable.future import Future
import time as t

import pandas as pd
import pickle


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


class FutureDataContainer(DataContainer):
    def __init__(self, root):
        # we create spot key instead of future key because it contains futures with different maturities
        self.market_key = market_utils.create_future_data_container_key(root)

    def get_market_key(self):
        return self.market_key

    def get_future_universe(self, dt, return_as_dict=True):
        return self._get_future_universe(dt, return_as_dict)

    def get_future_data(self, dt, future, return_as_dict=True):
        return self._get_future_data(dt, future, return_as_dict)

    def get_future_price(self, dt, future):
        return self._get_future_price(dt, future)

    def get_market_item(self, dt):
        def _get_future_universe(return_as_dict=True):
            return self.get_future_universe(dt, return_as_dict)
        def _get_future_data(future, return_as_dict=True):
            return self.get_future_data(dt, future, return_as_dict)
        def _get_future_price(future):
            return self.get_future_price(dt, future)
        return FutureData(self.get_market_key(), _get_future_universe, _get_future_data, price_func=_get_future_price)


class FutureDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, root, suffix, expiry_counts=float('inf'), skip_months=[],
                 bbg_fall_back=None,
                 inc_prev_day = False ):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.root = root
        self.suffix = suffix
        self.expiry_counts = expiry_counts
        self.skip_months = skip_months
        self.bbg_fall_back = bbg_fall_back,
        self.inc_prev_day = inc_prev_day

    def data_cache_name(self):
        return f'future_data_{self.root}_daily'


class DatalakeBBGFuturesDataSource(IDataSource):
    def __init__(self, force_reload=True, update_cache=False, data_cache_path=None):
        self.ref_dict = {}
        self.data_dict = {}
        self.force_reload = force_reload
        self.update_cache = update_cache
        self.data_cache_path = data_cache_path

    def read_cache(self, data_request):
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_data_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_cache(self, data_request, cache_data):
        data_cache_request_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_request.pickle'
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_request_file, 'w+b') as f:
            pickle.dump(data_request, f, pickle.HIGHEST_PROTOCOL)
        with open(data_cache_data_file, 'w+b') as f:
            pickle.dump(cache_data, f, pickle.HIGHEST_PROTOCOL)

    def initialize(self, data_request):

        if self.force_reload:
            start=t.time()
            # ref data
            source = 'BBG_CONTRACT_DESC'

            if data_request.inc_prev_day:
                start_date = minus_tenor( data_request.start_date, '1W' )
            else:
                start_date = data_request.start_date
            end_date = data_request.end_date
            ticker = f'root:{data_request.root}'
            ref_df = DATALAKE.getData(source=source, ticker=ticker, fields=DATALAKE.getAvailableFields(source),
                                      start=start_date, end=end_date,
                                      extra_fields='suffix', extra_values=data_request.suffix)
            if data_request.root == 'RTY':
                additional_ref_df = DATALAKE.getData(source=source, ticker='root:RTA',
                                                     fields=DATALAKE.getAvailableFields(source),
                                                     start=start_date, end=end_date,
                                                     extra_fields='suffix', extra_values=data_request.suffix)
                additional_ref_df['root'] = 'RTY'
                ref_df = pd.concat([ref_df, additional_ref_df], axis=0)
            if ref_df.empty:
                print(f"Missing future data {ticker}")
                return None
            ref_df = ref_df.sort_values('last_tradeable_dt')
            ref_df = ref_df[(ref_df['last_tradeable_dt'] >= start_date.strftime('%Y-%m-%d')) & (ref_df['fut_first_trade_dt'] <= end_date.strftime('%Y-%m-%d'))]

            # market data
            source = 'BBG_PRICE'
            all_tickers = list(set(ref_df['ticker'].values))
            if len(data_request.skip_months) > 0:
                all_tickers = [x for x in all_tickers if x.replace(' Index', '').replace(data_request.root, '')[
                    0] not in data_request.skip_months]
                ref_df = ref_df[ref_df.ticker.isin(all_tickers)]

            all_tickers_data = {}
            # full query not work, e.g. DATALAKE.getAvailableFields(source)
            # instead, we only  query fields required
            fields = ['PX_LAST', 'SETTLMNT_PX_ALTRNTVE_NOTTN']
            for ticker in all_tickers:
                all_tickers_data[ticker] = DATALAKE.getData(source=source, ticker=ticker, fields=fields, start=start_date, end=end_date)

            # TODO: check formatting so that single API call actually works.
            '''
            all_tickers = ','.join( all_tickers )
            all_tickers_data = DATALAKE.getData(source=source, ticker=all_tickers, fields=DATALAKE.getAvailableFields(source),
                                                start=start_date, end=end_date, extra_fields=None, extra_values=None)
            '''

            # organize them in dictionaries
            for dt in get_business_days(start_date, end_date,
                                        get_holidays(data_request.calendar, start_date, end_date)):
                selected_ref = ref_df[ref_df['last_tradeable_dt']
                                      >= dt.strftime('%Y-%m-%d')].head(data_request.expiry_counts)

                if data_request.root == 'RTY':
                    select_count = data_request.expiry_counts + 1
                    while len(list(set(selected_ref['last_tradeable_dt'].values))) < data_request.expiry_counts:
                        selected_ref = ref_df[ref_df['last_tradeable_dt']
                                              >= dt.strftime('%Y-%m-%d')].head(select_count)
                        select_count = select_count + 1

                for index, row in selected_ref.iterrows():
                    ticker = row['ticker']
                    self.ref_dict.setdefault(dt, {})[ticker] = {
                        'root': row['root'],
                        'ticker': ticker,
                        'currency': row['crncy'],
                        'contract size': row['fut_cont_size'],
                        'month code': row['month_code'],
                        # 'last tradable date': datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                        'last tradable date': set_timezone(datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                                                           EXCHANGE_TZ[row['fut_exch_name_short']]),
                        'exchange': row['fut_exch_name_short'],
                    }
                    data_df = all_tickers_data[ticker]
                    # TODO: check formatting so that single API call actually works.
                    # data_df = all_tickers_data[ [ col for col in all_tickers_data.columns if ticker in col or col == 'tstamp' ] ]
                    data_df = data_df[data_df['tstamp'].str.slice(stop=10) == dt.strftime('%Y-%m-%d')]

                    # TODO: once we have calendar sorted this should be strictly equal to 1
                    assert data_df.shape[0] <= 1

                    # temp fix for VHO
                    if data_df.shape[0] == 1 and data_request.root == 'VHO' \
                            and datetime.strptime(data_df.tstamp.values[0][:10], "%Y-%m-%d") >= datetime(2022, 9, 29):
                        settle_px = DATALAKE.getData(source=source, ticker=ticker, fields='SETTLMNT_PX_ALTRNTVE_NOTTN',
                                                     start=start_date, end=end_date)
                        if len(settle_px) > 0:
                            data_df = settle_px[settle_px['tstamp'].str.slice(stop=10) == dt.strftime('%Y-%m-%d')]
                        else:
                            data_df = pd.DataFrame()
                        if data_df.shape[0] == 1:
                            self.data_dict.setdefault(dt, {})[ticker] = {
                                'close': float(data_df['SETTLMNT_PX_ALTRNTVE_NOTTN'].values[0]),
                            }

                    else:
                        if data_df.shape[0] == 1 and data_df['PX_LAST'].values[0] != "":
                            self.data_dict.setdefault(dt, {})[ticker] = {
                                    'close': float(data_df['PX_LAST'].values[0]), }
            if self.update_cache:
                self.save_cache(data_request, [self.ref_dict, self.data_dict])

        else:
            [self.ref_dict, self.data_dict] = self.read_cache(data_request)

        container = FutureDataContainer(data_request.root)

        def _get_future_universe(dt, return_as_dict):
            if dt is None:
                return list(self.ref_dict.keys())
            else:
                # fix for calling the function with intraday data
                try:
                    return self.ref_dict[dt] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[dt], key_column_name='ticker')
                except:
                    return self.ref_dict[datetime.combine(dt, time(0,0))] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[datetime.combine(dt, time(0,0))], key_column_name='ticker')

        def _get_future_data(dt, future, return_as_dict):
            """
            support fallback method to look up ticker like ESZ3 using ESZ23 with year component "2"
            """
            def convert_ctp_ticker_to_permanent_ticker(original_ticker):
                year = str(dt.year)[-2]
                return original_ticker.split()[0][:-1] + str(year) + original_ticker.split()[0][-1:] + " " \
                    + original_ticker.split()[1]

            ticker = future.listed_ticker
            if dt not in self.data_dict:
                print(dt, future.root)
            if ticker not in self.data_dict[dt]:
                ticker = convert_ctp_ticker_to_permanent_ticker(ticker)
                if ticker not in self.data_dict[dt] and future.root == 'RTY':
                    ticker = future.listed_ticker.replace('RTY', 'RTA')
                    if ticker not in self.data_dict[dt]:
                        ticker = convert_ctp_ticker_to_permanent_ticker(ticker)

            if ticker not in self.data_dict[dt]:
                print(ticker)

            return self.data_dict[dt][ticker] if return_as_dict \
                else dict_of_dict_to_dataframe(self.data_dict[dt][ticker], key_column_name='ticker')

        container._get_future_universe = _get_future_universe
        container._get_future_data = _get_future_data
        return container


# we make distinction between DataContainers and MarketItem as for some case data container has too much data
# and it is less efficient to have it in market whereas MarketItem can contain only one day worth of data
# each data container should have a get_market_item function, the return of which will be contained in the market object
class FutureData(MarketItem):
    def __init__(self, market_key, universe_func, data_func, price_func=None):
        self.market_key = market_key
        self.get_future_universe = universe_func
        self.get_future_data = data_func
        if price_func is not None:
            self.get_future_price = price_func

    def get_market_key(self):
        return self.market_key

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        raise Exception("Not implemented yet")


class Datalake2BBGFuturesDataSource(IDataSource):
    def __init__(self, force_reload=True, update_cache=False, data_cache_path=None):
        self.ref_dict = {}
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')
        self.force_reload = force_reload
        self.update_cache = update_cache
        self.data_cache_path = data_cache_path

    def read_cache(self, data_request):
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_data_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_cache(self, data_request, cache_data):
        data_cache_request_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_request.pickle'
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_request_file, 'w+b') as f:
            pickle.dump(data_request, f, pickle.HIGHEST_PROTOCOL)
        with open(data_cache_data_file, 'w+b') as f:
            pickle.dump(cache_data, f, pickle.HIGHEST_PROTOCOL)

    def initialize(self, data_request):
        if self.force_reload:
            start=t.time()
            # ref data
            source = 'BBG_CONTRACT_DESC'

            if data_request.inc_prev_day:
                start_date = minus_tenor( data_request.start_date, '1W' )
            else:
                start_date = data_request.start_date
            end_date = data_request.end_date
            ticker = f'root:{data_request.root}'
            ref_df = self.datalake.getData(source=source, ticker=ticker, fields=self.datalake.getAvailableFields(source),
                                        start=start_date, end=end_date, kwargs={'suffix':data_request.suffix})

            if data_request.root == 'RTY':
                additional_ref_df = self.datalake.getData(source=source, ticker='root:RTA', fields=self.datalake.getAvailableFields(source),
                            start=start_date, end=end_date, kwargs={'suffix':data_request.suffix})
                additional_ref_df['root'] = 'RTY'
                ref_df = pd.concat([ref_df, additional_ref_df], axis=0)
            if ref_df.empty:
                print(f"Missing future data {ticker}")
                return None
            ref_df = ref_df.sort_values('last_tradeable_dt')
            ref_df = ref_df[(ref_df['last_tradeable_dt'] >= start_date.strftime('%Y-%m-%d')) & (ref_df['fut_first_trade_dt'] <= end_date.strftime('%Y-%m-%d'))]

            # market data
            source = 'BBG_PRICE'
            all_tickers = list(set(ref_df['ticker'].values))
            if len(data_request.skip_months) > 0:
                all_tickers = [x for x in all_tickers if x.replace(' Index', '').replace(data_request.root, '')[
                    0] not in data_request.skip_months]
                ref_df = ref_df[ref_df.ticker.isin(all_tickers)]

            all_tickers_data = self.datalake.getData(source=source, ticker=all_tickers, fields='PX_LAST',
                                    start=start_date, end=end_date).reset_index().pivot(index='tstamp', columns='ticker', values='PX_LAST')

            # TODO: check formatting so that single API call actually works.
            '''
            all_tickers = ','.join( all_tickers )
            all_tickers_data = DATALAKE.getData(source=source, ticker=all_tickers, fields=DATALAKE.getAvailableFields(source),
                                                start=start_date, end=end_date, extra_fields=None, extra_values=None)
            '''

            # organize them in dictionaries
            for dt in get_business_days(start_date, end_date,
                                        get_holidays(data_request.calendar, start_date, end_date)):
                selected_ref = ref_df[ref_df['last_tradeable_dt']
                                        >= dt.strftime('%Y-%m-%d')].head(data_request.expiry_counts)

                if data_request.root == 'RTY':
                    select_count = data_request.expiry_counts + 1
                    while len(list(set(selected_ref['last_tradeable_dt'].values))) < data_request.expiry_counts:
                        selected_ref = ref_df[ref_df['last_tradeable_dt']
                                                >= dt.strftime('%Y-%m-%d')].head(select_count)
                        select_count = select_count + 1

                for index, row in selected_ref.iterrows():
                    ticker = row['ticker']
                    self.ref_dict.setdefault(dt, {})[ticker] = {
                        'root': row['root'],
                        'ticker': ticker,
                        'currency': row['crncy'],
                        'contract size': row['fut_cont_size'],
                        'month code': row['month_code'],
                        # 'last tradable date': datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                        'last tradable date': set_timezone(datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                                                            EXCHANGE_TZ[row['fut_exch_name_short']]),
                        'exchange': row['fut_exch_name_short'],
                    }
                    data_df = all_tickers_data[ticker]
                    if dt in data_df.index:
                        data_df = data_df.loc[[dt]]
                        # TODO: once we have calendar sorted this should be strictly equal to 1
                        assert data_df.shape[0] <= 1

                        if data_df.shape[0] == 1 and data_df.values[0] != "":
                            self.data_dict.setdefault(dt, {})[ticker] = {'close': float(data_df.values[0])}
                    else:
                        self.data_dict.setdefault(dt, {})[ticker] = {'close': float('nan')}
            if self.update_cache:
                self.save_cache(data_request, [self.ref_dict, self.data_dict])
        else:
            [self.ref_dict, self.data_dict] = self.read_cache(data_request)

        container = FutureDataContainer(data_request.root)

        def _get_future_universe(dt, return_as_dict):
            if dt is None:
                return list(self.ref_dict.keys())
            else:
                # fix for calling the function with intraday data
                try:
                    return self.ref_dict[dt] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[dt], key_column_name='ticker')
                except:
                    return self.ref_dict[datetime.combine(dt, time(0,0))] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[datetime.combine(dt, time(0,0))], key_column_name='ticker')

        def _get_future_data(dt, future, return_as_dict):
            """
            support fallback method to look up ticker like ESZ3 using ESZ23 with year component "2"
            """
            def convert_ctp_ticker_to_permanent_ticker(original_ticker):
                year = str(dt.year)[-2]
                return original_ticker.split()[0][:-1] + str(year) + original_ticker.split()[0][-1:] + " " \
                    + original_ticker.split()[1]

            ticker = future.listed_ticker
            if dt not in self.data_dict:
                print(dt, future.root)
            if ticker not in self.data_dict[dt]:
                ticker = convert_ctp_ticker_to_permanent_ticker(ticker)
                if ticker not in self.data_dict[dt] and future.root == 'RTY':
                    ticker = future.listed_ticker.replace('RTY', 'RTA')
                    if ticker not in self.data_dict[dt]:
                        ticker = convert_ctp_ticker_to_permanent_ticker(ticker)

            if ticker not in self.data_dict[dt]:
                print(ticker)

            return self.data_dict[dt][ticker] if return_as_dict \
                else dict_of_dict_to_dataframe(self.data_dict[dt][ticker], key_column_name='ticker')

        container._get_future_universe = _get_future_universe
        container._get_future_data = _get_future_data
        return container


class IntradayFutureDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, root, suffix, expiry_counts=float('inf'), skip_months=[],snap_time='9:30', inc_prev_day = False ):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.root = root
        self.suffix = suffix
        self.expiry_counts = expiry_counts
        self.skip_months = skip_months
        self.snap_time = snap_time
        self.inc_prev_day = inc_prev_day

class Datalake2KibotMinuteFuturesDataSource(IDataSource):
    def __init__(self):
        self.ref_dict = {}
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    def initialize(self, data_request):
            
        start=t.time()
        # ref data
        source = 'BBG_CONTRACT_DESC'

        if data_request.inc_prev_day:
            start_date = minus_tenor( data_request.start_date, '1W' )
        else:
            start_date = data_request.start_date
        end_date = data_request.end_date
        ticker = f'root:{data_request.root}'
        ref_df = self.datalake.getData(source=source, ticker=ticker, fields=self.datalake.getAvailableFields(source),
                                    start=start_date, end=end_date, kwargs={'suffix':data_request.suffix})

        if data_request.root == 'RTY':
            additional_ref_df = self.datalake.getData(source=source, ticker='root:RTA', fields=self.datalake.getAvailableFields(source),
                        start=start_date, end=end_date, kwargs={'suffix':data_request.suffix})
            additional_ref_df['root'] = 'RTY'
            ref_df = pd.concat([ref_df, additional_ref_df], axis=0)
        if ref_df.empty:
            print(f"Missing future data {ticker}")
            return None
        ref_df = ref_df.sort_values('last_tradeable_dt')
        ref_df = ref_df[(ref_df['last_tradeable_dt'] >= start_date.strftime('%Y-%m-%d')) & (ref_df['fut_first_trade_dt'] <= end_date.strftime('%Y-%m-%d'))]

        # market data
        source = 'KIBOT_MINUTE'
        all_tickers = list(set(ref_df['ticker'].values))
        if len(data_request.skip_months) > 0:
            all_tickers = [x for x in all_tickers if x.replace(' Index', '').replace(data_request.root, '')[
                0] not in data_request.skip_months]
            ref_df = ref_df[ref_df.ticker.isin(all_tickers)]
        
        all_tickers_data = self.datalake.getData(source=source, ticker=[i.split(' ')[0] for i in all_tickers], fields=['close'],
                                start=start_date, end=end_date, kwargs = {'exact_time':data_request.snap_time}).reset_index().pivot(index='trade_tstamp', columns='ticker', values='close')
        all_tickers_data.index=all_tickers_data.index.strftime('%Y-%m-%d')

        # TODO: check formatting so that single API call actually works.
        '''
        all_tickers = ','.join( all_tickers )
        all_tickers_data = DATALAKE.getData(source=source, ticker=all_tickers, fields=DATALAKE.getAvailableFields(source),
                                            start=start_date, end=end_date, extra_fields=None, extra_values=None)
        '''

        # organize them in dictionaries
        holidays=get_holidays(data_request.calendar, start_date, end_date)
        for dt in get_business_days(start_date, end_date, holidays):

            selected_ref = ref_df[ref_df['last_tradeable_dt']
                                    >= dt.strftime('%Y-%m-%d')].head(data_request.expiry_counts)

            if data_request.root == 'RTY':
                select_count = data_request.expiry_counts + 1
                while len(list(set(selected_ref['last_tradeable_dt'].values))) < data_request.expiry_counts:
                    selected_ref = ref_df[ref_df['last_tradeable_dt']
                                            >= dt.strftime('%Y-%m-%d')].head(select_count)
                    select_count = select_count + 1

            for index, row in selected_ref.iterrows():
                ticker = row['ticker']
                self.ref_dict.setdefault(dt, {})[ticker] = {
                    'root': row['root'],
                    'ticker': ticker,
                    'currency': row['crncy'],
                    'contract size': row['fut_cont_size'],
                    'month code': row['month_code'],
                    # 'last tradable date': datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                    'last tradable date': set_timezone(datetime.strptime(row['last_tradeable_dt'], '%Y-%m-%d'),
                                                        EXCHANGE_TZ[row['fut_exch_name_short']]),
                    'exchange': row['fut_exch_name_short'],
                }
                data_df = all_tickers_data[ticker.split(" ")[0]].dropna()
                # TODO: check formatting so that single API call actually works.
                # data_df = all_tickers_data[ [ col for col in all_tickers_data.columns if ticker in col or col == 'tstamp' ] ]
                try:
                    data_df = data_df.loc[[dt.strftime('%Y-%m-%d')]]
                except:
                    fix_data=self.datalake.getData(source=source, ticker=[ticker.split(" ")[0]], fields=['close'],
                                start=dt, end=dt.replace(hour=int(data_request.snap_time.split(':')[0]), minute=int(data_request.snap_time.split(':')[1]))).reset_index().pivot(index='trade_tstamp', columns='ticker', values='close')

                    if len(fix_data) > 0:
                        fix_data=fix_data.iloc[[-1]]
                        print("data missing for ticker {} on date {} at snaptime {}, replace with closest timestamp {}". format(ticker.split(" ")[0],dt.strftime('%Y-%m-%d'),data_request.snap_time,fix_data.index[0]))
                        fix_data.index=fix_data.index.strftime('%Y-%m-%d')
                        data_df=fix_data
                    else:
                        data_df = pd.DataFrame(None)
                        print("Warning: data missing for ticker {} on date {} at snaptime {} with no possible replacment.". format(ticker.split(" ")[0],dt.strftime('%Y-%m-%d'),data_request.snap_time))


                # TODO: once we have calendar sorted this should be strictly equal to 1
                assert data_df.shape[0] <= 1

                if data_df.shape[0] == 1 and data_df.values[0] != "":
                    self.data_dict.setdefault(dt, {})[ticker] = {
                            'close': float(data_df.values[0]), }

        container = FutureDataContainer(data_request.root)

        def _get_future_universe(dt, return_as_dict):
            if dt is None:
                return list(self.ref_dict.keys())
            else:
                # fix for calling the function with intraday data
                try:
                    return self.ref_dict[dt] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[dt], key_column_name='ticker')
                except:
                    return self.ref_dict[datetime.combine(dt, time(0,0))] if return_as_dict \
                        else dict_of_dict_to_dataframe(self.ref_dict[datetime.combine(dt, time(0,0))], key_column_name='ticker')

        def _get_future_data(dt, future, return_as_dict):
            """
            support fallback method to look up ticker like ESZ3 using ESZ23 with year component "2"
            """
            def convert_ctp_ticker_to_permanent_ticker(original_ticker):
                year = str(dt.year)[-2]
                return original_ticker.split()[0][:-1] + str(year) + original_ticker.split()[0][-1:] + " " \
                    + original_ticker.split()[1]

            ticker = future.listed_ticker
            if dt not in self.data_dict:
                print(dt, future.root)
            if ticker not in self.data_dict[dt]:
                ticker = convert_ctp_ticker_to_permanent_ticker(ticker)
                if ticker not in self.data_dict[dt] and future.root == 'RTY':
                    ticker = future.listed_ticker.replace('RTY', 'RTA')
                    if ticker not in self.data_dict[dt]:
                        ticker = convert_ctp_ticker_to_permanent_ticker(ticker)

            if ticker not in self.data_dict[dt]:
                print(ticker)

            return self.data_dict[dt][ticker] if return_as_dict \
                else dict_of_dict_to_dataframe(self.data_dict[dt][ticker], key_column_name='ticker')

        container._get_future_universe = _get_future_universe
        container._get_future_data = _get_future_data
        return container


class IVOLFutureDataSource(IDataSource):
    def __init__(self):
        self.ref_dict = {}
        self.data_dict = {}
        self.dlc = DatalakeCassandra()

    def initialize(self, data_request):
        # ref data
        self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        # self.bbg_fall_back = True if data_request.bbg_fall_back is None else  data_request.bbg_fall_back

        expiration_months = expiration_months_number_from_skip_months(data_request.skip_months)

        ref_df = self.dlc.get_ivol_futures(data_request.root,
                                           expiration_months=expiration_months).set_index(['futures_id'])
        self.ref_dict = self.ref_dict | ref_df.to_dict('index')

        # market data
        container = FutureDataContainer(data_request.root)

        def _get_future_universe(dt, return_as_dict_of_dict=True):
            return self.ref_dict if return_as_dict_of_dict \
                else dict_of_dict_to_dataframe(self.ref_dict, key_column_name='futures_id')

        def _get_future_data(dt, future: Future, return_as_dict=True):
            dt_str = format_isodate(dt)
            futures_id = future.get_ivol_futures_id(self.dlc)
            if future not in self.data_dict:
                df = self.dlc.get_futures_price(futures_id=futures_id, start_date=self.start_date, end_date=self.end_date)
                self.data_dict[future] = df.set_index('tstamp').to_dict('index')

            result = self.data_dict[future].get(dt_str)

            # if self.bbg_fall_back and (result is None or result['price'] <= 0):
            #     session = get_session()
            #     price = bpipe.get_bbg_history(session, future.bbg_ticker, 'PX_SETTLE', dt, dt)['PX_SETTLE'].iloc[0]
            #     print('went to bbg for ' + future.name() + ' on ' + dt_str)
            #     result = {'price': price}
            #     self.data_dict[future][dt_str] = result
            return result

        def _get_future_price(dt, future: Future):
            return _get_future_data(dt, future).get('price')

        container._get_future_universe = _get_future_universe
        container._get_future_data = _get_future_data
        container._get_future_price = _get_future_price

        return container
