import pickle
import time
from datetime import datetime, timedelta
import pandas as pd
import os
import io
import json
from cassandra.query import named_tuple_factory

from ..analytics.symbology import CURRENCY_FROM_OPTION_ROOT, currency_from_option_root
from ..data.datalake_cassandra import DatalakeCassandra, ExpiryFilterByDateRange, StrikeFilterByValue, \
    CallPutFilterByValue, ExerciseFilterByValue, ExpiryFilterByDateOffset, ExpiryFilterByDate
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..language import dataframe_to_records, format_isodate
from ..dates.utils import bdc_adjustment, add_business_days
from ..dates.holidays import get_holidays
from ..tools.timer import Timer
from ..tradable.option import Option
from ..analytics.utils import BSPricer

pd.options.mode.chained_assignment = None

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


class OptionDataContainer(DataContainer):
    def __init__(self, root: str):
        self.market_key = market_utils.create_option_data_container_key(root)

    def get_market_key(self):
        return self.market_key

    def get_option_universe(self, dt, return_as_dict=False):
        return self._get_option_universe(dt, return_as_dict)

    def get_option_data(self, dt, option, return_as_dict=False):
        return self._get_option_data(dt, option, return_as_dict)

    def get_option_data_prev_eod(self, dt, option):
        return self._get_option_data_prev_eod(dt, option)

    def get_market_item(self, dt):
        raise RuntimeError('To be implemented -- right now code should not reach this point.')

    def get_full_option_universe(self):
        return self._get_full_option_universe()


class OptionDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, root, expiry_filter=None,
                 strike_filter=None, call_put_filter=None, exercise_filter=None, contract_filter=None,
                 expiration_rule_filter=None,
                 frequency='daily', use_1545table=False, dates=[], allow_fix_option_price_from_settlement=False,
                 allow_reprice=False):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.root = root
        self.expiry_filter = expiry_filter
        self.strike_filter = strike_filter
        self.call_put_filter = call_put_filter
        self.exercise_filter = exercise_filter
        self.contract_filter = contract_filter
        self.expiration_rule_filter = expiration_rule_filter
        self.frequency = frequency
        self.use_1545table = use_1545table
        self.dates = dates
        self.allow_fix_option_price_from_settlement=allow_fix_option_price_from_settlement
        self.allow_reprice=allow_reprice

    def data_cache_name(self):
        return f'option_data_{self.root}_{self.frequency}'

    def box_diff(self, other):
        """
        returns None, None if it cannot compare with existing, meaning you have to do a fresh load but don't want to write to cache as it overwrites
        otherwise it can compare, and returns the missing tiles (which may be empty if asked data is already contained in cache)
        @param other:
        @return:
        """
        if self.root == other.root and self.strike_filter == other.strike_filter and self.call_put_filter == other.call_put_filter and self.exercise_filter == other.exercise_filter and self.contract_filter == other.contract_filter and self.frequency == other.frequency:
            if isinstance(self.expiry_filter, ExpiryFilterByDateOffset) and isinstance(other.expiry_filter, ExpiryFilterByDateOffset):
                missing_tiles = []
                # first dimension to expand is time
                if self.end_date.date() > other.end_date.date():
                    assert self.frequency == 'daily'
                    missing_tiles.append(OptionDataRequest(other.end_date + timedelta(days=1), self.end_date, self.calendar, self.root,
                                         ExpiryFilterByDateOffset(max(self.expiry_filter.max_expiry_offset, other.expiry_filter.max_expiry_offset), min(self.expiry_filter.min_expiry_offset, other.expiry_filter.min_expiry_offset)),
                                         self.strike_filter, self.call_put_filter, self.exercise_filter, self.contract_filter, self.frequency))
                if self.start_date.date() < other.start_date.date():
                    assert self.frequency == 'daily'
                    missing_tiles.append(OptionDataRequest(self.start_date, other.start_date - timedelta(days=1), self.calendar, self.root,
                                         ExpiryFilterByDateOffset(max(self.expiry_filter.max_expiry_offset, other.expiry_filter.max_expiry_offset), min(self.expiry_filter.min_expiry_offset, other.expiry_filter.min_expiry_offset)),
                                         self.strike_filter, self.call_put_filter, self.exercise_filter, self.contract_filter, self.frequency))
                if self.expiry_filter.max_expiry_offset > other.expiry_filter.max_expiry_offset:
                    missing_tiles.append(OptionDataRequest(other.start_date, other.end_date, self.calendar, self.root,
                                         ExpiryFilterByDateOffset(self.expiry_filter.max_expiry_offset, other.expiry_filter.max_expiry_offset + 1),
                                         self.strike_filter, self.call_put_filter, self.exercise_filter, self.contract_filter, self.frequency))
                if self.expiry_filter.min_expiry_offset < other.expiry_filter.min_expiry_offset:
                    missing_tiles.append(OptionDataRequest(other.start_date, other.end_date, self.calendar, self.root,
                                         ExpiryFilterByDateOffset(other.expiry_filter.min_expiry_offset - 1, self.expiry_filter.min_expiry_offset),
                                         self.strike_filter, self.call_put_filter, self.exercise_filter, self.contract_filter, self.frequency))
                consolidated_data_request = OptionDataRequest(min(self.start_date, other.start_date), max(self.end_date, other.end_date), self.calendar, self.root,
                                                              ExpiryFilterByDateOffset(max(self.expiry_filter.max_expiry_offset, other.expiry_filter.max_expiry_offset), min(self.expiry_filter.min_expiry_offset, other.expiry_filter.min_expiry_offset)),
                                                              self.strike_filter, self.call_put_filter, self.exercise_filter, self.contract_filter, self.frequency)
                return missing_tiles, consolidated_data_request
            else:
                return None, None
        else:
            return None, None


# base class for pre load type of data source
class CassandraDSPreLoad(IDataSource):
    def __init__(self, adjust_expiration=False,
                 force_reload=False, update_cache=False):
        self.cassandra = DatalakeCassandra()
        self.adjust_expiration = adjust_expiration
        self.force_reload = force_reload
        self.update_cache = update_cache

    def save_cache(self, data_request, cache_data_request, cache_data):
        pass

    def check_cache(self, data_request):
        pass

    def read_cache(self, data_request):
        pass

    def initialize(self, data_request):
        if data_request.exercise_filter is not None and isinstance(data_request.exercise_filter, ExpiryFilterByDateOffset):
            holiday_request_end_date = data_request.end_date + timedelta(days=data_request.exercise_filter.max_expiry_offset)
        else:
            holiday_request_end_date = datetime(2050, 12, 31)
        self.holidays = get_holidays(data_request.calendar, data_request.start_date, holiday_request_end_date)

        assert data_request.frequency == 'daily', 'Non daily option data has not been implemented by this DataSource'
        assert data_request.contract_filter is None, 'Contract filter has not been implemented by this DataSource'
        assert data_request.expiration_rule_filter is None

        data_container = OptionDataContainer(data_request.root)

        if data_request.use_1545table:
            if not self.force_reload or self.update_cache:
                raise Exception("Only support force reload without update cache for using 1545 table")
        if self.force_reload:
            # we just load the data and save it to cache (overwriting anything existing)
            self.data = self.cassandra.load_option_data(
                start_date=data_request.start_date,
                end_date=data_request.end_date,
                calendar=data_request.calendar,
                root_security=data_request.root,
                expiry_filter=data_request.expiry_filter,
                strike_filter=data_request.strike_filter,
                call_put_filter=data_request.call_put_filter,
                exercise_filter=data_request.exercise_filter,
                file_cache_path=None,
                use_1545table=data_request.use_1545table,
                dates=data_request.dates,
            )
            if not self.data.empty:
                self.data['root'] = data_request.root
                self.data = convert_ivol_df(self.data, True, adjust_expiration=self.adjust_expiration, holidays=self.holidays)
                consolidated_data_request = data_request
                if self.update_cache:
                    self.save_cache(data_request, consolidated_data_request, self.data)
        else:
            # check cache, load any missing data blocks, and save the increased data into cache
            fresh_load, missing_tiles, consolidated_data_request = self.check_cache(data_request)

            if fresh_load:
                self.data = self.cassandra.load_option_data(
                    data_request.start_date,
                    data_request.end_date,
                    data_request.calendar,
                    data_request.root,
                    data_request.expiry_filter,
                    data_request.strike_filter,
                    data_request.call_put_filter,
                    data_request.exercise_filter,
                )
                self.data['root'] = data_request.root
                if self.data.empty:
                    raise Exception(f"Missing option data for {data_request.root}")
                self.data = convert_ivol_df(self.data, True, adjust_expiration=self.adjust_expiration, holidays=self.holidays)
                consolidated_data_request = data_request
            else:
                self.data = self.read_cache(data_request)
                for missing_dq in missing_tiles:
                    tile = self.cassandra.load_option_data(
                        missing_dq.start_date,
                        missing_dq.end_date,
                        missing_dq.calendar,
                        missing_dq.root,
                        missing_dq.expiry_filter,
                        missing_dq.strike_filter,
                        missing_dq.call_put_filter,
                        missing_dq.exercise_filter,
                    )
                    if not tile.empty:
                        tile['root'] = data_request.root
                        tile = convert_ivol_df(tile, True, adjust_expiration=self.adjust_expiration, holidays=self.holidays)

                        # merge tile with self.data
                        for k, v in tile.items():
                            if k in self.data:
                                self.data[k]['df'] = pd.concat([self.data[k]['df'], v['df']]).drop_duplicates()
                                self.data[k]['indexing'] = dict(zip(self.data[k]['df']['ticker'].values, range(self.data[k]['df'].shape[0])))
                            else:
                                self.data[k] = v
                        #self.data = merge_dict(self.data, tile)
                assert consolidated_data_request is not None

            # caching
            # caching for each fresh load may cause too many overriding when people load different flavors of data request
            # so for now we need to seed the cache with a standard format of data request
            # each time someone ask for something that is an enlargement we cache the enlarged data
            if self.update_cache:
                if not fresh_load:
                    # fresh load means the ask is not comparable to cache, we don't overwrite the cache
                    if len(missing_tiles) > 0:
                        # if not missing tiles we don't need to write back as there is nothing new
                        self.save_cache(data_request, consolidated_data_request, self.data)

            # TODO: self.data can be a superset of what is asked
            # filter here for only data that is asked

        self.data_np = {}
        for k, v in self.data.items():
            self.data_np[k] = v['df'].to_numpy()

        def _get_option_universe(dt, return_as_dict):
            if dt not in self.data:
                return None
            if return_as_dict:
                return dataframe_to_records(self.data[dt]['df'])
            else:
                return self.data[dt]['df']
                # option_universe_df = pd.DataFrame.from_records(self.data[dt])
                # option_universe_df['date'] = dt.isoformat()
                # return option_universe_df

        def _get_full_option_universe():
            return list( self.data.keys() )


        def _get_option_data(dt, option: Option, return_as_dict):
            option_universe = self.data[dt]['df']
            option_indexing = self.data[dt]['indexing']
            cols = list(option_universe.columns)

            if option.listed_ticker not in option_indexing:
                return None
            index = option_indexing[option.listed_ticker]
            # option_data = option_universe.iloc[index].values
            option_data = self.data_np[dt][index]
            if return_as_dict:
                option_data_dict = dict(zip(cols, option_data))
                assert option_data_dict['ticker'] == option.listed_ticker
                return option_data_dict
            else:
                return pd.DataFrame().from_records([dict(zip(cols, option_data))])

        def _get_option_data_prev_eod(dt: datetime, option, return_as_dict=True):
            dt = datetime(dt.year, dt.month, dt.day)
            prev_eod = add_business_days(dt, -1, self.holidays)
            return _get_option_data(prev_eod, option, return_as_dict)

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        data_container._get_full_option_universe = _get_full_option_universe
        data_container._get_option_data_prev_eod = _get_option_data_prev_eod
        return data_container


# preload from pickle file
class CassandraDSPreLoadFromPickle(CassandraDSPreLoad):
    def __init__(self, adjust_expiration=False,
                 data_cache_path='/misc/Traders/Solutions/backtests/data_cache/options',
                 force_reload=False, update_cache=False):
        self.data_cache_path = data_cache_path
        super().__init__(adjust_expiration, force_reload, update_cache)

    def clone(self):
        return CassandraDSPreLoadFromPickle(self.adjust_expiration, self.data_cache_path, self.force_reload, self.update_cache)

    def save_cache(self, data_request, cache_data_request, cache_data):
        data_cache_request_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_request.pickle'
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_request_file, 'w+b') as f:
            pickle.dump(cache_data_request, f, pickle.HIGHEST_PROTOCOL)
        with open(data_cache_data_file, 'w+b') as f:
            pickle.dump(cache_data, f, pickle.HIGHEST_PROTOCOL)

    def check_cache(self, data_request):
        data_cache_request_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_request.pickle'
        fresh_load = True
        missing_tiles = None
        consolidated_data_request = None
        if os.path.isfile(data_cache_request_file):
            with open(data_cache_request_file, 'rb') as f:
                cached_data_request = pickle.load(f)
            missing_tiles, consolidated_data_request = data_request.box_diff(cached_data_request)
            if missing_tiles is not None:
                fresh_load = False
        return fresh_load, missing_tiles, consolidated_data_request

    def read_cache(self, data_request):
        data_cache_data_file = f'{self.data_cache_path}/{data_request.data_cache_name()}_data.pickle'
        with open(data_cache_data_file, 'rb') as f:
            data = pickle.load(f)
        return data


# preload from dictionary
class CassandraDSPreLoadFromDict(CassandraDSPreLoad):
    def __init__(self, adjust_expiration=False,
                 data_dict={},
                 data_key_path=(),
                 force_reload=False):
        self.data_dict = data_dict
        self.data_key_path = data_key_path
        cache = self.data_dict
        for k in self.data_key_path:
            if k not in cache:
                cache = None
                break
            else:
                cache = cache[k]
        self.cache = cache
        super().__init__(adjust_expiration, force_reload, False)

    def save_cache(self, data_request, cache_data_request, cache_data):
        raise RuntimeError('cannot save in memory data cache')

    def check_cache(self, data_request):
        fresh_load = True

        if self.cache is None:
            return True, None, None
        else:
            cached_data_request = self.cache['data_request']
            missing_tiles, consolidated_data_request = data_request.box_diff(cached_data_request)
            if missing_tiles is not None:
                fresh_load = False
            return fresh_load, missing_tiles, consolidated_data_request

    def read_cache(self, data_request):
        if self.cache is None:
            return None
        else:
            data = self.cache['data']
            return data


# on demand from database itself
class CassandraDSOnDemand(IDataSource):
    def __init__(self, adjust_expiration=False):
        self.cassandra = DatalakeCassandra()
        self.adjust_expiration = adjust_expiration
        self.data_dict = {}

    def load_from_cassandra(self, start_date, end_date=None, expiry=None, strike=None, call_put=None, exercise=None):
        end_date = start_date if end_date is None else end_date
        expiry_filter = None if expiry is None else ExpiryFilterByDateRange(expiry - timedelta(days=3), expiry + timedelta(days=3))
        strike_filter = None if strike is None else StrikeFilterByValue(strike)
        call_put_filter = None if call_put is None else CallPutFilterByValue(call_put)
        exercise_filter = None if exercise is None else ExerciseFilterByValue(exercise)

        success = False
        count = 1
        while not success and count < 50:
            try:
                df = self.cassandra.load_option_data(
                                start_date,
                                end_date,
                                self.calendar,
                                self.root_security,
                                expiry_filter,
                                strike_filter,
                                call_put_filter,
                                exercise_filter,
                            )
                success = True
            except:
                print(f"cassandra load option data error. retrying #{count}")
                count = count + 1
                time.sleep(1)

        if not df.empty:
            # override root to consolidate weekly, monthly option root.
            if self.root_security is not None:
                df['root'] = self.root_security
            df = format_ivol_df(df, adjust_expiration=self.adjust_expiration, holidays=self.holidays)
            if expiry is not None:
                df = df[df['expiration_date'] == datetime.combine(expiry.date(), datetime.min.time()).isoformat()]
            df.set_index('t_date', inplace=True)
        return df

    def initialize(self, data_request):
        if data_request.exercise_filter is not None and isinstance(data_request.exercise_filter, ExpiryFilterByDateOffset):
            holiday_request_end_date = data_request.end_date + timedelta(days=data_request.exercise_filter.max_expiry_offset)
        else:
            holiday_request_end_date = datetime(2050, 12, 31)
        self.holidays = get_holidays(data_request.calendar, data_request.start_date, holiday_request_end_date)
        self.use_1545table = data_request.use_1545table
        self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        self.calendar = data_request.calendar
        self.root_security = data_request.root

        assert data_request.frequency == 'daily', 'Non daily option data has not been implemented by this DataSource'
        assert data_request.contract_filter is None, 'Contract filter has not been implemented by this DataSource'
        assert data_request.expiration_rule_filter is None

        data_container = OptionDataContainer(data_request.root)

        def _get_option_universe(dt, return_as_dict):
            df = self.load_from_cassandra(dt)
            if return_as_dict:
                return df.to_dict('index')
            else:
                return df

        def _get_full_option_universe():
            return NotImplementedError('not supported as no guarantee all possible dates have been pulled in')

        def _get_option_data(dt, option: Option, return_as_dict):
            strike = option.strike
            call_put = "C" if option.is_call else "P"
            expiry_date = option.expiration
            dt_str = format_isodate(dt)
            if (option not in self.data_dict) or (option in self.data_dict and dt_str not in self.data_dict[option]):
                df = self.load_from_cassandra(start_date=dt, end_date=expiry_date, expiry=expiry_date,
                                              strike=strike, call_put=call_put)
                df.index = df.index.astype(str)
                option_dict = df.to_dict('index')
                self.data_dict[option] = option_dict
            result = self.data_dict[option].get(dt_str, None)
            if return_as_dict:
                return result
            else:
                return pd.DataFrame.from_dict(result, orient='index')

        def _get_option_data_prev_eod(dt: datetime, option, return_as_dict=True):
            dt = datetime(dt.year, dt.month, dt.day)
            prev_eod = add_business_days(dt, -1, self.holidays)
            return _get_option_data(prev_eod, option, return_as_dict)

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        data_container._get_full_option_universe = _get_full_option_universe
        data_container._get_option_data_prev_eod = _get_option_data_prev_eod
        return data_container


# on demand from daily cache
class CassandraDSOnDemandFromCacheByDay(IDataSource):
    def __init__(self, data_cache_path=None):
        self.data_cache_path = data_cache_path
        self.cassandra = DatalakeCassandra()
        self.root_security = None
        self.date = None
        self.data = None

    def clone(self):
        return CassandraDSOnDemandFromCacheByDay(self.data_cache_path)

    def load_from_cassandra(self, dt):
        timer = Timer(f"load_from_cassandra {self.root_security}", verbose=False)
        timer.start()
        if self.data_cache_path is not None:
            data_cache_data_file = os.path.join(self.data_cache_path, dt.strftime("%Y-%m-%d"), f"option_data_{self.root_security}_daily_data.pickle")
            if os.path.exists(data_cache_data_file):
                timer.print("load pickle")
                with open(data_cache_data_file, "rb") as f:
                    data = pickle.load(f)
                timer.end()
                return data[dt]["df"]
        session = self.cassandra.get_session()
        session.row_factory = pandas_factory
        sql = "select * from datacache.solution_{}_cache_pickle where year={} and month={} and date='{}'".format(
            self.root_security, dt.year, dt.month, dt.strftime("%Y-%m-%d"))
        rslt = session.execute_async(sql)
        data = []
        tmp = rslt.result()._current_rows.set_index('date')
        for index in tmp.index:
            data.append(pickle.loads(tmp.loc[index]['data'])["df"])
        data = pd.concat(data).set_index('date') if len(data) > 0 else None
        session.row_factory = named_tuple_factory
        timer.end()
        return data

    def initialize(self, data_request):
        self.root_security = data_request.root
        data_container = OptionDataContainer(data_request.root)

        assert data_request.expiration_rule_filter is None
        
        def _get_option_universe(dt, return_as_dict):
            if dt != self.date:
                self.data = self.load_from_cassandra(dt)
                self.date = dt
            if return_as_dict:
                return self.data.to_dict('index')
            else:
                return self.data

        def _get_option_data(dt, option: Option, return_as_dict):
            strike = option.strike
            call_put = "C" if option.is_call else "P"
            expiry_date = option.expiration
            if dt != self.date:
                self.data = self.load_from_cassandra(dt)
                self.date = dt
            self.data["expiration_datetime"] = pd.to_datetime(self.data["expiration"])
            data = self.data[(self.data["expiration_datetime"].dt.date == expiry_date.date())
                             & (self.data["strike"] == strike)
                             & (self.data["call_put"] == call_put)].drop(columns=["expiration_datetime"])
            if return_as_dict:
                return data.to_dict('records')[0]
            else:
                raise Exception("Not have user case yet")

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        return data_container


# preload from daily cache
class CassandraDSPreLoadFromCacheByDay(CassandraDSPreLoad):
    def __init__(self, adjust_expiration=False,
                 data_cache_path='',
                 force_reload=False, update_cache=False):
        self.data_cache_path = data_cache_path
        super().__init__(adjust_expiration, force_reload, update_cache)

    def clone(self):
        return CassandraDSPreLoadFromCacheByDay(self.adjust_expiration, self.data_cache_path, self.force_reload, self.update_cache)

    def save_cache(self, data_request, cache_data_request, cache_data):
        session=self.cassandra.get_session() 
        for key in self.data.keys():
            byte=io.BytesIO()
            pickle.dump(self.data[key],byte)
            df=byte.getvalue()
            dfs=[]
            if len(df) > 10000000:
                i=0
                while True:
                    if i*10000000 > len(df):
                        break
                    dfs.append(df[i*10000000:(i+1)*10000000])
                    i+=1
            else:
                dfs.append(df)
            for i in range(len(dfs)):
                sql="update datacache.solution_{}_cache_pickle set data=? where year=? and month=? and date=? and part=?".format(data_request.root.split(' ')[0])
                info=(dfs[i],key.year,key.month,key,i)
                prepared = session.prepare(sql)
                session.execute(prepared,info)
        dq={'start':'2007-01-03',
        'end':cache_data_request.end_date.strftime("%Y-%m-%d"),
        'calendar':cache_data_request.calendar,
        'root':cache_data_request.calendar,
        'expiry':cache_data_request.expiry_filter.max_expiry_offset}
        json_byte = json.dumps(dq, indent=2).encode('utf-8')
        sql="update datacache.solution_{}_cache_pickle set data=? where year=? and month=? and date=? and part=?".format(data_request.root.split(' ')[0])
        info=(json_byte,2099,12,'2099-12-31',0)
        prepared = session.prepare(sql)
        session.execute(prepared,info)
        

    def check_cache(self, data_request):
        session=self.cassandra.get_session()
        session.row_factory = pandas_factory
        data_cache_request_file_sql = "select * from datacache.solution_{}_cache_pickle where year=2099 and month=12 and date='2099-12-31'".format(data_request.root.split(' ')[0])
        data_cache_request_file=session.execute(data_cache_request_file_sql)._current_rows['data'][0]
        fresh_load = True
        missing_tiles = None
        consolidated_data_request = None
        cached_request_info = json.loads(data_cache_request_file)
        cached_data_request = OptionDataRequest(datetime.strptime(cached_request_info['start'],"%Y-%m-%d"), datetime.strptime(cached_request_info['end'],"%Y-%m-%d"), cached_request_info['calendar'], cached_request_info['root'],expiry_filter=ExpiryFilterByDateOffset(int(cached_request_info['expiry'])))
        missing_tiles, consolidated_data_request = data_request.box_diff(cached_data_request)
        if missing_tiles is not None:
            fresh_load = False
        session.row_factory = named_tuple_factory
        return fresh_load, missing_tiles, consolidated_data_request

    def read_cache(self, data_request):
        session=self.cassandra.get_session()
        session.row_factory = pandas_factory
        sqls=[]
        sql="select * from datacache.solution_{}_cache_pickle where year=? and month = ? and date = ?".format(data_request.root.split(' ')[0])
        prepared = session.prepare(sql)
        dates=pd.date_range(data_request.start_date, data_request.end_date)
        for date in dates:
            info=(date.year,date.month,date)
            sqls.append(session.execute_async(prepared,info))
        data={}
        for rslt in sqls:
            tmp=rslt.result()._current_rows.set_index('date')
            for index in tmp.index:
                dt=datetime(index.date().year,index.date().month,index.date().day)
                data[dt]=pickle.loads(tmp.loc[index]['data'])
        session.row_factory = named_tuple_factory
        return data


# preload from yearly cache
class CassandraDSPreLoadFromCacheByYear(CassandraDSPreLoad):
    def __init__(self, adjust_expiration=False,
                 data_cache_path='',
                 force_reload=False, update_cache=False):
        self.data_cache_path = data_cache_path
        super().__init__(adjust_expiration, force_reload, update_cache)
    
    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    def clone(self):
        return CassandraDSPreLoadFromCacheByYear(self.adjust_expiration, self.data_cache_path, self.force_reload, self.update_cache)

    def save_cache(self, data_request, cache_data_request, cache_data):
        session=self.cassandra.get_session() 
        if (data_request.start_date.year == data_request.end_date.year):
            years=[data_request.start_date + pd.offsets.YearEnd()]
        else:
            years=pd.date_range(data_request.start_date,data_request.end_date + pd.tseries.offsets.DateOffset(years=1),freq='Y')
        for i in range(len(years)):
            datas=[]
            for key in self.data.keys():
                if i ==0:
                    if key<=years[0]:
                        datas.append(self.data[key]['df'])
                else:
                    if key<=years[i] and key>years[i-1]:
                        datas.append(self.data[key]['df'])
            if len(datas)>0:
                tmp=io.BytesIO()
                test=pd.concat(datas)
                test.to_parquet(tmp,compression='gzip')
                df=tmp.getvalue()
                dfs=[]
                if len(df) > 10000000:
                    j=0
                    while True:
                        if j*10000000 > len(df):
                            break
                        dfs.append(df[j*10000000:(j+1)*10000000])
                        j+=1
                else:
                    dfs.append(df)
                sql="delete from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],years[i].year)
                session.execute(sql)
                for j in range(len(dfs)):
                    sql="update datacache.solution_datacache_{}_option set byte=? where name='{}' and part=?".format(data_request.root.split(' ')[0],years[i].year)
                    info=(dfs[j],j)
                    prepared = session.prepare(sql)
                    session.execute(prepared,info)
        for i in range(len(years)):
            datas={}
            for key in self.data.keys():
                if i ==0:
                    if key<=years[0]:
                        datas.update({key:self.data[key]['indexing']})
                else:
                    if key<=years[i] and key>years[i-1]:
                        datas.update({key:self.data[key]['indexing']})
            if len(datas)>0:
                tmp=io.BytesIO()
                pickle.dump(datas,tmp)
                df=tmp.getvalue()
                dfs=[]
                if len(df) > 10000000:
                    j=0
                    while True:
                        if j*10000000 > len(df):
                            break
                        dfs.append(df[j*10000000:(j+1)*10000000])
                        j+=1
                else:
                    dfs.append(df)
                sql="delete from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],str(years[i].year)+'_indexing')
                session.execute(sql)
                for j in range(len(dfs)):
                    sql="update datacache.solution_datacache_{}_option set byte=? where name='{}' and part=?".format(data_request.root.split(' ')[0],str(years[i].year)+'_indexing')
                    info=(dfs[j],j)
                    prepared = session.prepare(sql)
                    session.execute(prepared,info)
        dq={'start':cache_data_request.start_date.strftime("%Y-%m-%d"),
        'end':cache_data_request.end_date.strftime("%Y-%m-%d"),
        'calendar':cache_data_request.calendar,
        'root':cache_data_request.root,
        'expiry':cache_data_request.expiry_filter.max_expiry_offset}
        json_byte = json.dumps(dq, indent=2).encode('utf-8')         
        sql="update datacache.solution_datacache_{}_option set byte=? where name='{}' and part=?".format(data_request.root.split(' ')[0],'request_json')
        info=(json_byte,0)
        prepared = session.prepare(sql)
        session.execute(prepared,info)

    def check_cache(self, data_request):
        session=self.cassandra.get_session() 
        session.row_factory = pandas_factory
        data_cache_request_file_sql = "select * from datacache.solution_datacache_{}_option where name='request_json'".format(data_request.root.split(' ')[0])
        data_cache_request_file=session.execute(data_cache_request_file_sql)._current_rows['byte'][0]
        fresh_load = True
        missing_tiles = None
        consolidated_data_request = None
        cached_request_info = json.loads(data_cache_request_file)
        cached_data_request = OptionDataRequest(datetime.strptime(cached_request_info['start'],"%Y-%m-%d"), datetime.strptime(cached_request_info['end'],"%Y-%m-%d"), cached_request_info['calendar'], cached_request_info['root'],expiry_filter=ExpiryFilterByDateOffset(int(cached_request_info['expiry'])))
        missing_tiles, consolidated_data_request = data_request.box_diff(cached_data_request)
        if missing_tiles is not None:
            fresh_load = False
        session.row_factory = named_tuple_factory
        return fresh_load, missing_tiles, consolidated_data_request

    def read_cache(self, data_request):
        start=time.time()
        if (data_request.start_date.year == data_request.end_date.year):
            years=[data_request.start_date + pd.offsets.YearEnd()]
        else:
            years=pd.date_range(data_request.start_date,data_request.end_date + pd.tseries.offsets.DateOffset(years=1),freq='Y')
        session=self.cassandra.get_session()
        session.row_factory = pandas_factory
        sqls=[]
        for date in years:
            sqls.append(session.execute_async("select * from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],date.year)))
        for date in years:
            sqls.append(session.execute_async("select * from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],str(date.year)+'_indexing')))

        result=[]
        data={}
        for i in range(len(sqls)):

            retry=0
            retry_result=None

            while retry<10:
                try:
                    if retry == 0:
                        result.append(sqls[i].result()._current_rows)
                    else:
                        result[i]=retry_result._current_rows
                    break
                except:
                    time.sleep(1)
                    retry_result=session.execute(sqls[i].query._query_string)
                    retry+=1
            byte = bytearray()
            for line in result[i]['byte']:
                byte.extend(line)
            if i < len(years):
                byte=io.BytesIO(byte)
                tmp=pd.read_parquet(byte)
                for index,df in tmp.groupby(tmp['date']):
                    index=datetime.fromisoformat(index)
                    if index not in data.keys():
                        data[index]={}
                    data[index]['df']=df
            else:
                tmp=pickle.loads(bytes(byte))
                for index in tmp.keys():
                    if index not in data.keys():
                        data[index]={}
                    data[index]['indexing']=tmp[index]
        session.row_factory = named_tuple_factory
        return data


# on demand from yearly cache
class CassandraDSOnDemandFromCacheByYear(IDataSource):

    def __init__(self,adjust_expiration=False):
        self.cassandra = DatalakeCassandra()
        self.adjust_expiration=adjust_expiration
    
    def clone(self):
        return CassandraDSOnDemandFromCacheByYear(self.adjust_expiration)
    
    def read_cache(self,data_request,dt):
        session=self.cassandra.get_session()
        session.row_factory = pandas_factory

        sqls=[]
        sqls.append(session.execute_async("select * from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],dt.year)))
        sqls.append(session.execute_async("select * from datacache.solution_datacache_{}_option where name='{}'".format(data_request.root.split(' ')[0],str(dt.year)+'_indexing')))

        result=[]
        data={}
        for i in range(len(sqls)):
            result.append(sqls[i].result()._current_rows)
            byte = bytearray()
            for line in result[i]['byte']:
                byte.extend(line)
            if i < 1:
                byte=io.BytesIO(byte)
                tmp=pd.read_parquet(byte)
                for index,df in tmp.groupby(tmp['date']):
                    index=datetime.fromisoformat(index)
                    if index not in data.keys():
                        data[index]={}
                    data[index]['df']=df
            else:
                tmp=pickle.loads(bytes(byte))
                for index in tmp.keys():
                    if index not in data.keys():
                        data[index]={}
                    data[index]['indexing']=tmp[index]
        session.row_factory = named_tuple_factory
        return data

    def check_cache(self, data_request):
        session=self.cassandra.get_session() 
        session.row_factory = pandas_factory
        data_cache_request_file_sql = "select * from datacache.solution_datacache_{}_option where name='request_json'".format(data_request.root.split(' ')[0])
        data_cache_request_file=session.execute(data_cache_request_file_sql)._current_rows['byte'][0]
        fresh_load = True
        missing_tiles = None
        consolidated_data_request = None
        cached_request_info = json.loads(data_cache_request_file)
        cached_data_request = OptionDataRequest(datetime.strptime(cached_request_info['start'],"%Y-%m-%d"), datetime.strptime(cached_request_info['end'],"%Y-%m-%d"), cached_request_info['calendar'], cached_request_info['root'],expiry_filter=ExpiryFilterByDateOffset(int(cached_request_info['expiry'])))
        missing_tiles, consolidated_data_request = data_request.box_diff(cached_data_request)
        if missing_tiles is not None:
            fresh_load = False
        session.row_factory = named_tuple_factory
        return fresh_load, missing_tiles, consolidated_data_request
        

    def initialize(self, data_request):

        if data_request.exercise_filter is not None and isinstance(data_request.exercise_filter, ExpiryFilterByDateOffset):
            holiday_request_end_date = data_request.end_date + timedelta(days=data_request.exercise_filter.max_expiry_offset)
        else:
            holiday_request_end_date = datetime(2050, 12, 31)
        self.holidays = get_holidays(data_request.calendar, data_request.start_date, holiday_request_end_date)
        self.use_1545table = data_request.use_1545table
        self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        self.calendar = data_request.calendar
        self.root_security = data_request.root

        assert data_request.frequency == 'daily', 'Non daily option data has not been implemented by this DataSource'
        assert data_request.contract_filter is None, 'Contract filter has not been implemented by this DataSource'
        assert data_request.expiration_rule_filter is None

        data_container = OptionDataContainer(data_request.root)

        self.data={}

        fresh_load, missing_tiles, consolidated_data_request = self.check_cache(data_request)


        for missing_dq in missing_tiles:
            tile = self.cassandra.load_option_data(
                missing_dq.start_date,
                missing_dq.end_date,
                missing_dq.calendar,
                missing_dq.root,
                missing_dq.expiry_filter,
                missing_dq.strike_filter,
                missing_dq.call_put_filter,
                missing_dq.exercise_filter,
            )
            if not tile.empty:
                tile['root'] = data_request.root
                tile = convert_ivol_df(tile, True, adjust_expiration=self.adjust_expiration, holidays=self.holidays)

                # merge tile with self.data
                for k, v in tile.items():
                    if k in self.data:
                        self.data[k]['df'] = pd.concat([self.data[k]['df'], v['df']])
                        self.data[k]['indexing'] = dict(zip(self.data[k]['df']['ticker'].values, range(self.data[k]['df'].shape[0])))
                    else:
                        self.data[k] = v

        def _get_option_universe(dt,return_as_dict):

            if dt not in self.data:

                self.data={}

                self.data=self.read_cache(data_request,dt)
            
            if return_as_dict:
                return dataframe_to_records(self.data[dt]['df'])
            else:
                return self.data[dt]['df']

        def _get_full_option_universe():
            return list( self.data.keys() )


        def _get_option_data(dt, option: Option, return_as_dict):
            
            if dt not in self.data:
                
                self.data={}

                self.data=self.read_cache(data_request,dt)

            option_universe = self.data[dt]['df']

            strike = option.strike
            call_put = "C" if option.is_call else "P"
            expiry_date = option.expiration
            expiration_rule = option.expiration_rule

            #option_universe["expiration_time"] = pd.to_datetime(option_universe["expiration"])
            option_data = option_universe[(option_universe["expiration"] == expiry_date.strftime("%Y-%m-%dT%H:%M:%S"))
                             & (option_universe["strike"] == strike)
                             & (option_universe["call_put"] == call_put)]

            if expiration_rule:

                option_data=option_data[option_data['expiration_rule']==expiration_rule]

            if len(option_data) > 1:

                
                if len(option_data['expiration_rule'].unique()) > 1:

                    raise Exception('Duplicate option found at {}, please specify expiration_rule'.format(dt.strftime("%Y-%m-%d")))
                
                else:

                    # print('Warning: Duplicate option with same expiration_rule found at {}, drop duplicates'.format(dt.strftime("%Y-%m-%d")))
                    option_data=option_data.drop_duplicates(subset=['expiration','strike','call_put','expiration_rule'])
            
            # if none revert the expiration_rule filter with warning
            elif len(option_data) == 0:
                print('Warning: option expiration rule changed at {}, revert to no expiration_rule filter'.format(dt.strftime("%Y-%m-%d")))
                option_data = option_universe[(option_universe["expiration"] == expiry_date.strftime("%Y-%m-%dT%H:%M:%S"))
                            & (option_universe["strike"] == strike)
                            & (option_universe["call_put"] == call_put)]       

            if (option_data['price'].item() <=0.001) and (data_request.allow_fix_option_price_from_settlement):

                option_data['price']=option_data['settlement'].item()

            if (option_data['price'].item() <=0.001) and (data_request.allow_reprice):
                
                if option_data['price'].item() <=0.001:

                    print('Warning: option {} price is 0 at {}, reprice with BS formula'.format(option.listed_ticker,dt.strftime("%Y-%m-%d")))

                    time_to_maturity = option.expiration - dt
                    T = time_to_maturity.days + time_to_maturity.seconds / (3600 * 24)

                    call_put_flag = option_data['call_put'].item()
                    strike = option_data['strike'].item()
                    iv = option_data['iv'].item()

                    backfill_dt=dt
                    while iv==-1:

                        print('Warning: option {} iv is -1 at {}, reprice with previous day iv'.format(option.listed_ticker,dt.strftime("%Y-%m-%d")))

                        backfill_dt = add_business_days(backfill_dt, -1, self.holidays)
                        option_data = _get_option_data(backfill_dt, option, return_as_dict=False)
                        iv = option_data['iv'].item()
                    
                    S = self.cassandra.get_stock_data(option.root, dt, dt).drop_duplicates(subset=['price_close'])['price_close'].item()
                    q = self.cassandra.get_option_dividend(option,dt,S)
                    r = self.cassandra.get_option_interest_rate(option,dt)
                    bs = BSPricer(call_put_flag, S, strike, T, r, q, iv)
                    option_data['price'] = bs.get_value('p')
                    

            if return_as_dict:
                return option_data.to_dict('records')[0]
            else:
                return option_data

        def _get_option_data_prev_eod(dt: datetime, option, return_as_dict=True):
            dt = datetime(dt.year, dt.month, dt.day)
            prev_eod = add_business_days(dt, -1, self.holidays)
            return _get_option_data(prev_eod, option, return_as_dict)
        
        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        data_container._get_full_option_universe = _get_full_option_universe
        data_container._get_option_data_prev_eod = _get_option_data_prev_eod
        return data_container


# on demand, universe from daily cache, individual option read from database for all dates to expiry at first query
class CassandraDSOnDemandMixed(IDataSource):
    def __init__(self):
        self.cassandra = DatalakeCassandra()
        self.adjust_expiration = True
        self.calendar = None
        self.holidays = []
        self.root_security = None
        self.expiration_rule_filter = None
        self.universe_date = None
        self.universe_data = {}
        self.options_data = {}

    def clone(self):
        return CassandraDSOnDemandMixed()

    def load_universe_from_cassandra_cache(self, root, dt):
        timer = Timer(f"load_universe_from_cassandra_cache {root}", verbose=False)
        timer.start()

        session = self.cassandra.get_session()
        session.row_factory = pandas_factory
        sql = "select * from datacache.solution_{}_cache_pickle where year={} and month={} and date='{}'".format(
            root, dt.year, dt.month, dt.strftime("%Y-%m-%d"))
        rslt = session.execute_async(sql)
        data = []
        tmp = rslt.result()._current_rows.set_index('date')
        for index in tmp.index:
            data.append(pickle.loads(tmp.loc[index]['data'])["df"])
        data = pd.concat(data).set_index('date') if len(data) > 0 else None
        session.row_factory = named_tuple_factory
        timer.end()
        return data

    def load_one_option_data(self, dt, option):
        end_date = option.expiration
        # we have to search the immediate following holidays as some early options has expiration in weekend in database
        expiry_filter = ExpiryFilterByDateRange(option.expiration, add_business_days(option.expiration, 1, self.holidays) - timedelta(days=1))
        strike_filter = StrikeFilterByValue(option.strike)
        call_put_filter = CallPutFilterByValue('C' if option.is_call else 'P')
        exercise_filter = ExerciseFilterByValue('A' if option.is_american else 'E')

        success = False
        count = 1
        while not success and count < 50:
            try:
                df = self.cassandra.load_option_data(
                                dt,
                                end_date,
                                self.holidays,
                                option.root,
                                expiry_filter,
                                strike_filter,
                                call_put_filter,
                                exercise_filter,
                            )
                success = True
            except:
                print(f"cassandra load option data error. retrying #{count}")
                print(f"{dt.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')} {option.name_str}")
                count = count + 1
                time.sleep(1)

        if not df.empty:
            # override root to consolidate weekly, monthly option root.
            df['root'] = option.root
            df = format_ivol_df(df, adjust_expiration=self.adjust_expiration, holidays=self.holidays)
            df = df[df['expiration_date'] == datetime.combine(option.expiration.date(), datetime.min.time()).isoformat()]
            df['t_date'] = df['t_date'].apply(lambda x: datetime.combine(x.date(), datetime.min.time()).isoformat())
            df = df[df['expiration_date'] == option.expiration.isoformat()]
        return df

    def from_df_first_row_to_dict(self, df):
        #return df.to_dict('records')[0]

        keys = list(df.columns)
        values = list(df.values[0])
        return dict(zip(keys, values))

    def initialize(self, data_request):
        self.root_security = data_request.root
        self.expiration_rule_filter = data_request.expiration_rule_filter
        data_container = OptionDataContainer(data_request.root)

        self.calendar = data_request.calendar
        holiday_request_end_date = datetime(2050, 12, 31)
        self.holidays = get_holidays(self.calendar, data_request.start_date, holiday_request_end_date)

        def _get_option_universe(dt, return_as_dict):
            if dt != self.universe_date:
                new_data = self.load_universe_from_cassandra_cache(self.root_security, dt)

                # only support filter by list now
                if self.expiration_rule_filter is not None:
                    new_data = new_data[new_data['expiration_rule'].isin(self.expiration_rule_filter.list)]

                self.universe_data = {dt: new_data}
                self.universe_date = dt
            if return_as_dict:
                return self.universe_data[dt].to_dict('index')
            else:
                return self.universe_data[dt]

        def _get_option_data(dt, option: Option, return_as_dict):
            option_ticker = option.name_str
            if dt in self.options_data and option_ticker in self.options_data[dt]:
                # first check if it is in the options data cache
                data = self.options_data[dt][option_ticker]
            else:
                # if not see if it is in universe cache
                # otherwise load from db until expiry and put the data in options data cache
                strike = option.strike
                call_put = "C" if option.is_call else "P"
                expiry_date = option.expiration

                if dt in self.universe_data:
                    data = self.universe_data[dt]
                    data = data[(data["expiration"] == expiry_date.isoformat()) & (data["strike"] == strike) & (data["call_put"] == call_put)]
                else:
                    data = pd.DataFrame()

                if not data.empty:
                    data = self.from_df_first_row_to_dict(data)
                else:
                    # purge older data
                    for k in list(self.options_data.keys()):
                        if k < dt:
                            del self.options_data[k]

                    new_data = self.load_one_option_data(dt, option)

                    if self.expiration_rule_filter is not None:
                        new_data = new_data[new_data['expiration_rule'].isin(self.expiration_rule_filter.list)]

                    new_data = {datetime.fromisoformat(k): {option_ticker: self.from_df_first_row_to_dict(g.set_index('t_date'))} for k, g in new_data.groupby('t_date')}
                    for d, vs in new_data.items():
                        for t, v in vs.items():
                            self.options_data.setdefault(d, vs).setdefault(t, v)
                    data = new_data[dt][option_ticker]

            if return_as_dict:
                return data
            else:
                raise Exception("Not have user case yet")

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        return data_container


def convert_ivol_df(df, include_option_indexing, adjust_expiration=True, holidays=[]):
    df = format_ivol_df(df, adjust_expiration=adjust_expiration, holidays=holidays)
    result_dict = {}
    grouped = df.groupby('t_date')
    ticker_universe = []
    for group_name, group in grouped:
        this_df = group.drop(['t_date'], axis=1)
        key = datetime.combine(group_name.date(), datetime.min.time())
        this_df['date'] = key.isoformat()

        # check shrinking universe outside of expiration
        # new_ticker_universe = this_df['ticker'].values
        # missing_tickers = set(ticker_universe) - set(new_ticker_universe)
        # if len(missing_tickers) > 0:
        #     print(f'{key.date()}: {missing_tickers} missing from universe')
        # ticker_universe = this_df[this_df['expiration'] > key.isoformat()]['ticker'].values

        if include_option_indexing:
            indexing = dict(zip(this_df['ticker'].values, range(this_df.shape[0])))
            result_dict[key] = {'df': this_df, 'indexing': indexing}
        else:
            result_dict[key] = this_df
    return result_dict


def format_ivol_df(df, adjust_expiration=True, holidays=[]):
    if not adjust_expiration:
        _exp_adjustment = lambda x: x
    else:
        def _exp_adjustment(_exp):
            return bdc_adjustment(_exp, convention='previous', holidays=holidays)

    def imply_ticker(row):
        option = Option(row['root'], row['underlying'], row['currency'],
                        datetime.fromisoformat(row['expiration']), row['strike'],
                        row['call_put'] == 'C', row['exercise_type'] == 'A',
                        row['contract_size'], row['tz_name'])
        return option.name()

    # format and transform df
    drop_cols = [x for x in
                 ['calc_date', 'expiration_id', 'is_interpolated', 'root_id', 'shift', 'status', ]
                 if x in df.columns]
    df = df.drop(drop_cols, axis=1)

    df = df.rename(columns={'expiration_date': 'expiration', 'root_style': 'exercise_type', 'style': 'exercise_type'})

    # to take care of data issues where expiration date is not a business day
    expirations = list(set(df['expiration'].values))
    adjusted_expiration_map = {exp: _exp_adjustment(datetime.combine(exp.date(), datetime.min.time())).isoformat()
                               for exp in expirations}
    df['expiration'] = df['expiration'].apply(lambda x: adjusted_expiration_map[x])

    # sometimes the underlying is in a shortened format e.g. GLD Equity instead of GLD US Equity
    # df['underlying'] = df['underlying'].apply(lambda x: ticker_from_ticker_short(x))

    # TODO: we make difference between expiration date and the expiration which take into account time
    # need to add logic to handle expiration_time
    df['expiration_date'] = df['expiration']
    root = df['root'].values[0]
    df['currency'] = '' if root not in CURRENCY_FROM_OPTION_ROOT else currency_from_option_root(root)
    df['contract_size'] = 1
    df['tz_name'] = ''
    df['ticker'] = df.apply(imply_ticker, axis=1)
    return df
