from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import WhiteListRoundRobinPolicy
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd
import os
import re
import numpy as np
import time
from dateutil import parser

from ..dates.utils import is_business_day, date_range, set_timezone, n_th_weekday_of_month, count_business_days
from ..dates.holidays import get_holidays
from ..analytics.symbology import ticker_from_option_root
from ..language import format_isodate
from ..analytics.utils import BSPricer

"""
query = "SELECT * FROM ivol.optionvalue_1545 where t_date='2020-12-31' and stock_id=627 and expiration_date='2021-1-15' and strike > 395 and strike < 405"
df = pd.DataFrame(session.execute(query))
# Rearranging the columns
cols=list(df.columns)
dic_replace = {'tstamp':'t_date'}
cols=[dic_replace.get(n, n) for n in cols]
df=df[cols]
df.head()
"""

COMMODITY_ROOT_ID_AND_UNDERLYING_ID_PAIR = {
    "CL": ('107', '680'),
    "HG": ('36', '97'),
}

ROOT_TO_IVOL_STOCK_ID = {
    'NKY':
        (
            '20590',    # bbg NKY Index, name NIKKEI 225 -OSE
            '32150',    # bbg NKYM Index, name NIKKEI 225 MINI -OSE
        ),
    'USO': '19370',
    'GLD': '18838',
    'EMB': '22189',
    'TIP': '18209',
    'V2X': '22073',
    'HSCEI': '20533',
    'FTSEMIB': '19124',
    'AS51': '20591',
    'CAC': '10564',
    'SX7E': '20046',
    'SX5E':
        (
            '10562',   # bbg SX5E Index, name DJ Euro STOXX 50 - null
            '25757',   # bbg SX5EM Index, name DJ Euro STOXX 50 (month-end option)
            '32229',   # bbg SX5EOM Index, name DJ Euro STOXX 50 (daily option)
        ),
    'SMI': '10567',
    'DAX': '10565',
    'SXXP': '20074',
    'FXF': '19376'
    # 'SPX': ('9327','22192')
}


class ExpiryFilter:
    def __init__(self):
        pass

    def is_date_dependent(self):
        pass

    def cassandra_filter_string(self, dt=None):
        pass


class ExpiryFilterByDateOffset(ExpiryFilter):
    def __init__(self, max_expiry_offset, min_expiry_offset=0):
        self.max_expiry_offset = max_expiry_offset
        self.min_expiry_offset = min_expiry_offset

    def is_date_dependent(self):
        return True

    def cassandra_filter_string(self, dt=None):
        max_exp_date_str = (dt + datetime.timedelta(days=self.max_expiry_offset)).strftime('%Y-%m-%d')
        min_exp_date_str = (dt + datetime.timedelta(days=self.min_expiry_offset)).strftime('%Y-%m-%d')
        return f" and expiration_date >= '{min_exp_date_str}' and expiration_date <= '{max_exp_date_str}'"


class ExpiryFilterByDate(ExpiryFilter):
    def __init__(self, expiration_date):
        self.expiration_date = expiration_date

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        return f" and expiration_date = '{self.expiration_date.strftime('%Y-%m-%d')}'"


class ExpiryFilterByDateRange(ExpiryFilter):
    def __init__(self, expiration_date_lb, expiration_date_ub):
        self.expiration_date_lb = expiration_date_lb
        self.expiration_date_ub = expiration_date_ub

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        # return f" and expiration_date >= '{self.expiration_date_lb.strftime('%Y-%m-%d')}' and expiration_date <= '{self.expiration_date_ub.strftime('%Y-%m-%d')}'"
        all_dates = date_range(self.expiration_date_lb, self.expiration_date_ub)
        date_needed = "','".join(i.strftime("%Y-%m-%d") for i in all_dates)
        return f" and expiration_date in ('{date_needed}')"


class StrikeFilter:
    def __init__(self):
        pass

    def is_date_dependent(self):
        pass

    def cassandra_filter_string(self, dt=None):
        pass


class StrikeFilterByRange(StrikeFilter):
    def __init__(self, lb, ub, attribute='strike'):
        self.lb = lb
        self.ub = ub
        self.attribute = attribute

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        if self.ub == float('inf'):
            return f" and {self.attribute} >= {self.lb}"
        else:
            return f" and {self.attribute} >= {self.lb} and {self.attribute} <= {self.ub}"


class StrikeFilterByValue(StrikeFilter):
    def __init__(self, strike):
        self.strike = strike

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        return f" and strike = {self.strike}"


class CallPutFilter:
    def __init__(self):
        pass

    def is_date_dependent(self):
        pass

    def cassandra_filter_string(self, dt=None):
        pass


class CallPutFilterByList(CallPutFilter):
    def __init__(self, list):
        self.list = list

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        joined = "','".join(self.list)
        return f" and call_put in ('{joined}')"


class CallPutFilterByValue(CallPutFilter):
    def __init__(self, value):
        self.value = value

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        return f" and call_put = '{self.value}'"


class ExerciseFilter:
    def __init__(self):
        pass

    def is_date_dependent(self):
        pass

    def cassandra_filter_string(self, dt=None):
        pass


class ExerciseFilterByList(ExerciseFilter):
    def __init__(self, list):
        self.list = list

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        joined = "','".join(self.list)
        return f" and root_style in ('{joined}')"


class ExerciseFilterByValue(ExerciseFilter):
    def __init__(self, value):
        self.value = value

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        return f" and root_style = '{self.value}'"


class ExpirationRuleFilter:
    def __init__(self):
        pass

    def is_date_dependent(self):
        pass

    def cassandra_filter_string(self, dt=None):
        pass


class ExpirationRuleByList(ExpirationRuleFilter):
    def __init__(self, list):
        self.list = list

    def is_date_dependent(self):
        return False

    def cassandra_filter_string(self, dt=None):
        raise RuntimeError(f"expiration rule filter in cassandra query is not implemented")


class intrday_option_data_cleaning():
    
    def __init__(self, option, data:pd.DataFrame, market_start, market_end ,replace_with_intrinsic_value = True):
        
        option.expiration=option.expiration.replace(hour=int(market_end.split(':')[0]), minute=int(market_end.split(':')[1]))
        self.option = option
        self.data = data.sort_values(by='t_date')
        self.replace_with_intrinsic_value = replace_with_intrinsic_value
        if 'error' not in self.data.columns:
            self.data.loc[:, 'error'] = ''
        if 'reprice_source' not in self.data.columns:
            self.data.loc[:, 'reprice_source'] = None
        self.start_datetime = self.data['t_date'].dt.strftime("%Y-%m-%d").values[0]
        self.sql = DatalakeCassandra()
        self.wrong_time = pd.Index([])
        self.cassandra = DatalakeCassandra()
    
    def process(self):
        
        data = self.data.copy()
        
        data = self._detect_nan_value(data)
        data = self._detect_minus_one_iv(data)
        data = self._detect_bid_larger_than_ask(data)
        data = self._detect_non_changing_underlyer_price(data)
        
        if self.replace_with_intrinsic_value:
                data = self._replace_with_intrinsic_value(data)
                
        data = self.fix(data)
        
        return data
    
    def fix(self, data):

        data = self._reprice(self.wrong_time, data)

        return data
    
    def get_q(self,data):
        
        q_amount=self.cassandra.get_dividend(self.option.root,self.start_datetime,self.option.expiration.strftime("%Y-%m-%d"))['dividend_amount'].item()
        t_diff = parser.parse(str(self.option.expiration)) - parser.parse(str(self.start_datetime))
        days_to_maturity = t_diff.days + t_diff.seconds / 86400
        q = q_amount * 365 / days_to_maturity / data.loc[:, 'underlyer_price_mid'][0]
        
        return q
    
    def get_r(self,t):
        
        days_to_maturity = (parser.parse(str(self.option.expiration)) - parser.parse(str(t))).days
        day = t.date()
        r = self.cassandra.get_interest_rate(day, days_to_maturity, self.option.currency)
        
        return r['interest_rate'].values[0]
        
        
    
    def _reprice(self, wrong_time, data):
        
        if not wrong_time.empty:
            data = self._replace_iv_with_previous_subsequent(wrong_time, data)
            self.q=self.get_q(data)
    
            def cal(x):
                t = x['t_date']
                time_to_maturity = parser.parse(str(self.option.expiration)) - parser.parse(str(t))
                T = time_to_maturity.days + time_to_maturity.seconds / (3600 * 24)

                S = x['underlyer_price_mid']
                call_put_flag = x['call_put']
                strike = x['strike']
                r = self.get_r(t)
                iv = x['iv']

                bs = BSPricer(call_put_flag, S, strike, T, r, self.q, iv)
                price = bs.get_value('p')
                delta = bs.get_value('d')
                gamma = bs.get_value('g')
                vega = bs.get_value('v')
                theta = bs.get_value('t')

                return [price, delta, gamma, vega, theta]

            temp = data.loc[wrong_time]
            data.loc[wrong_time, ['price', 'delta', 'gamma', 'vega', 'theta']] = temp.apply(cal, axis=1,
                                                                                                result_type='expand').values
            data.loc[wrong_time, 'reprice_source'] = 'BS'

        return data
    
    def _detect_nan_value(self, data):
        ffill_columns = ['symbol', 'call_put', 'expiration_date', 'strike', 'iv']
        data.loc[:, ffill_columns] = data.loc[:, ffill_columns].fillna(method='ffill')

        wrong_time = data[data['price'].isnull()].index
        self.wrong_time = self.wrong_time.union(wrong_time)
        data.loc[wrong_time, 'volume'] = 0
        data.loc[wrong_time, 'error'] += '1'
        return data
    
    def _detect_minus_one_iv(self, data):

        # if iv is -1, replace with previous value and reprice
        minue_one_iv = data[data['iv'] == -1].index
        wrong_time = data[(data['iv'] == -1) & (data['price_bid'] == 0) & (data['t_date'] != self.option.expiration)].index
        data.loc[minue_one_iv, 'iv'] = np.nan
        data.loc[:, 'iv'] = data.loc[:, 'iv'].fillna(method='ffill')
        data.loc[:, 'iv'] = data.loc[:, 'iv'].fillna(0.15)
        self.wrong_time = self.wrong_time.union(wrong_time)
        # data = self.reprice(wrong_time, data)
        data.loc[wrong_time, 'error'] += '2'

        return data
    
    def _detect_zero_option_price(self, data):

        price_is_zero = data['price'] == 0.0
        not_small_delta = np.abs(data['delta']) > 0.05
        strike_spot_ratio = data['strike'] / data['underlyer_price_mid']
        is_call = data['call_put'] == 'C'
        is_put = data['call_put'] == 'P'
        not_small_moneyness = ((is_call & (strike_spot_ratio < 1.15)) | (is_put & (strike_spot_ratio > 0.85))).astype(
            int)
        vol_jump_not_crazy = data['iv'] / data['iv'].shift() < 5

        wrong_time = data[price_is_zero & (not_small_delta | not_small_moneyness | vol_jump_not_crazy) & (data['t_date'] != self.maturity_date)].index
        self.wrong_time = self.wrong_time.union(wrong_time)
        # data = self.reprice(wrong_time, data)
        data.loc[wrong_time, 'error'] += '3'
        return data
    
    
    def _detect_bid_larger_than_ask(self, data):

        wrong_time = data[data['price_bid'] > data['price_ask']].index
        self.wrong_time = self.wrong_time.union(wrong_time)
        data.loc[wrong_time, 'error'] += 'D'
        return data
    
    def _detect_wide_bid_ask_spread(self, data):
        bid_ask_spread = abs(data['price_ask'] - data['price_bid'])
        is_wide_spread = bid_ask_spread>50
        not_small_price = (data['price_ask'] / data['underlyer_price_mid'] > 0.0020)|(data['price_bid'] / data['underlyer_price_mid'] > 0.0020)
        #spread_changed_alot = bid_ask_spread > 2 * bid_ask_spread.shift()

        wrong_time = data[is_wide_spread & not_small_price].index
        self.wrong_time = self.wrong_time.union(wrong_time)
        # data = self.reprice(wrong_time, data)

        data.loc[wrong_time, 'error'] += '5'
        return data
    
    
    def _detect_non_changing_underlyer_price(self, data):

        non_changing_underlyer_price = data['underlyer_price_mid'].diff() == 0
        non_changing_option_price = (data['price_ask'].diff() == 0)&(data['price_bid'].diff() == 0)
        non_changing_price = non_changing_underlyer_price&non_changing_option_price

        wrong_time = data[non_changing_price].index
        data.loc[wrong_time, 'error'] += '4'
            # data = self.reprice(wrong_time, data)
        # else:
        #     if len(wrong_time) > 0:
        #         print('data Error 4, cant fix without future data')

        return data
    
    def _replace_with_intrinsic_value(self, data):

        # replace zero price of deep in the money option with intrinsic value
        is_call = data['call_put'] == 'C'
        is_put = data['call_put'] == 'P'
        deep_ITM = np.abs(data['delta']) >= 0.95
        intrinsic_valueC = np.maximum(0, data['underlyer_price_mid'] - data['strike']) * is_call.astype(int)
        intrinsic_valueP = np.maximum(0, data['strike'] - data['underlyer_price_mid']) * is_put.astype(int)
        intrinsic_value = intrinsic_valueC + intrinsic_valueP
        price_smaller_than_intrinsic_value = intrinsic_value > data['price']

        # replace with intrinsic value for deep in the money
        replace_with_intrinsic_value = data[price_smaller_than_intrinsic_value & deep_ITM].index
        data.loc[replace_with_intrinsic_value, 'price'] = intrinsic_value.loc[replace_with_intrinsic_value]
        data.loc[replace_with_intrinsic_value, 'error'] += '6'
        data.loc[replace_with_intrinsic_value, 'reprice_source'] = 'ITV'

        is_maturity_date = (data['t_date'] == self.option.expiration).astype(int)
        # price at maturity use intrinsic value
        data.loc[:, 'price'] = is_maturity_date * intrinsic_value + (1 - is_maturity_date) * data['price']
        data.loc[:, 'price_ask'] = is_maturity_date * intrinsic_value + (1 - is_maturity_date) * data['price_ask']
        data.loc[:, 'price_bid'] = is_maturity_date * intrinsic_value + (1 - is_maturity_date) * data['price_bid']
        data.loc[is_maturity_date == 1, 'reprice_source'] += 'EXPIRY'

        # reprice the others
        need_reprice = data[price_smaller_than_intrinsic_value & (~deep_ITM) &(is_maturity_date!=1)].index
        self.wrong_time = self.wrong_time.union(need_reprice)
        # data = self.reprice(need_reprice, data)
        data.loc[need_reprice, 'error'] += '7'

        return data
    
    def _replace_iv_with_previous_subsequent(self, wrong_time, data):

        if len(wrong_time)==len(data):
            return data

        data.loc[:, 'iv'] = data.loc[:, 'iv'].replace(-1, np.nan).fillna(method='ffill')

        origin_iv = data['iv'].copy()
        data.loc[wrong_time, 'iv'] = np.nan
        none_na_data=data.dropna(subset=['iv'])
        def check_small_time_diff(x, ffill=True):
            
            if ffill:
                return (-(none_na_data[none_na_data['t_date']<=x].iloc[-1]['t_date']-x) <= pd.to_timedelta('10min'))
            else:
                return ((none_na_data[none_na_data['t_date']>=x].iloc[0]['t_date']-x) <= pd.to_timedelta('10min'))
        
        small_time_diff = data['t_date'].apply(lambda x:check_small_time_diff(x)).astype(int)
        ffill_iv = data['iv'].fillna(method='ffill').fillna(method='bfill') * small_time_diff + origin_iv * (
                    1 - small_time_diff)
        small_stime_diff = data['t_date'].apply(lambda x:check_small_time_diff(x,False)).astype(int)
        bfill_iv = data['iv'].fillna(method='bfill').fillna(method='ffill') * small_stime_diff + origin_iv * (
                    1 - small_stime_diff)
        data.loc[:, 'iv'] = (ffill_iv + bfill_iv) / 2

        return data
    


class DatalakeCassandra:
    """
    a direct connection to datalake db for faster access

    """

    def __init__(self, db_user=None, db_pwd=None):
        if db_user is None:
            db_user = 'capinternal_read'
        if db_pwd is None:
            db_pwd = 'Capstone123!'

        cluster_ip = "10.0.21.160,10.0.21.242,10.0.21.49,10.0.21.165".split(",")
        # packages
        auth_provider = PlainTextAuthProvider(db_user, password=db_pwd)
        self._cluster = Cluster(cluster_ip, load_balancing_policy=WhiteListRoundRobinPolicy(cluster_ip),
                                auth_provider=auth_provider)
        self._cluster.protocol_version = 4
        self.session = self._cluster.connect()
        self.session.default_consistency_level = 6
        self.session.default_timeout = 20

    def get_session(self):
        """To implement check when connection drops"""
        return self.session

    def get_intraday_all_option_data(self, stock_id, look_times):

        session = self.get_session()

        cql = "SELECT * FROM ivol.option_minute where"

        if isinstance(stock_id, int):
            cql = cql + " stockid={}".format(stock_id)
        elif isinstance(stock_id, list):
            cql = cql + " stockid IN {}".format(tuple(stock_id))
        else:
            raise Exception('Incorrect Stock Id, has to be either list or int')

        if isinstance(look_times, str):
            cql = cql + " and t_date='{}'".format(look_times)
        elif isinstance(look_times, list):
            if len(look_times) == 1:
                cql = cql + " and t_date='{}'".format(look_times[0])
            else:
                cql = cql + " and t_date IN {}".format(tuple(look_times))
        else:
            raise Exception('Incorrect look_times, has to be either list or str')

        df = pd.DataFrame(session.execute(cql))
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        return df

    def get_intraday_option_data(self, stock_id, expiration_times, look_times):

        session = self.get_session()

        cql = "SELECT * FROM ivol.option_minute where"

        if isinstance(stock_id, int):
            cql = cql + " stockid={}".format(stock_id)
        elif isinstance(stock_id, list):
            cql = cql + " stockid IN {}".format(tuple(stock_id))
        else:
            raise Exception('Incorrect Stock Id, has to be either list or int')

        if isinstance(look_times, str):
            cql = cql + " and t_date='{}'".format(look_times)
        elif isinstance(look_times, list):
            if len(look_times) == 1:
                cql = cql + " and t_date='{}'".format(look_times[0])
            else:
                cql = cql + " and t_date IN {}".format(tuple(look_times))
        else:
            raise Exception('Incorrect look_times, has to be either list or str')

        if isinstance(expiration_times, str):
            cql = cql + " and expiration_date='{}'".format(expiration_times)
        elif isinstance(stock_id, list):
            if len(expiration_times) == 1:
                cql = cql + " and expiration_date='{}'".format(expiration_times[0])
            else:
                cql = cql + " and expiration_date IN {}".format(tuple(expiration_times))
        else:
            raise Exception('Incorrect expiration_times, has to be either list or str')

        df = pd.DataFrame(session.execute(cql))
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        return df

    def get_option_data(self, source='optionvalue', stock_id=None, start_date=None,
                        end_date=None, expiration=None, strike=None,
                        call_put=None, strike_range=None, expiration_range=None):
        """
        :param source:
        :param stock_id:
        :param start_date:
        :param end_date:
        :param expiration:
        :param strike:
        :param call_put:
        :param strike_range:
        :param expiration_range:
        :return:
        """
        session = self.get_session()
        source = source.lower()
        if start_date is not None:
            if isinstance(start_date, datetime.date) or isinstance(start_date, datetime.datetime):
                start_date = start_date.strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.datetime.today().strftime("%Y-%m-%d")
        else:
            if isinstance(end_date, datetime.date) or isinstance(end_date, datetime.datetime):
                end_date = end_date.strftime("%Y-%m-%d")
        cql = "SELECT * FROM ivol.{} where".format(source, stock_id)

        if isinstance(stock_id, int):
            cql = cql + " stock_id={}".format(stock_id)
        elif isinstance(stock_id, list):
            cql = cql + " stock_id IN {}".format(tuple(stock_id))
        else:
            raise Exception('Incorrect Stock Id, has to be either list or int')

        if expiration is not None and expiration_range is None:
            cql = cql + " and expiration_date='{}'".format(expiration)
        elif expiration is None and expiration_range is not None:
            cql = cql + " and expiration_date>='{}' and expiration_date<='{}'".format(expiration_range[0],
                                                                                      expiration_range[1])
        if strike is not None and strike_range is None:
            cql = cql + " and strike={}".format(strike)
        elif strike is None and strike_range is not None:
            cql = cql + " and strike>={} and strike<={}".format(strike_range[0], strike_range[1])
        if call_put is not None:
            cql = cql + " and call_put='{}'".format(call_put)
        if start_date == end_date:
            cql = cql + " and t_date='{}'".format(start_date)
        else:
            d_range = pd.bdate_range(start_date, end_date)
            dlist = "("
            for z in d_range:
                if z == d_range[-1]:
                    dlist += "'{}'".format(z.strftime("%Y-%m-%d"))
                else:
                    dlist += "'{}',".format(z.strftime("%Y-%m-%d"))
            dlist += ")"
            cql = cql + " and t_date in {}".format(dlist)

        df = pd.DataFrame(session.execute(cql))
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        return df

    @staticmethod
    def find_stock_ids(session, root_security):
        # find stock_id
        if root_security in ROOT_TO_IVOL_STOCK_ID:
            stock_id_from_map = ROOT_TO_IVOL_STOCK_ID[root_security]
            stock_ids = f'({stock_id_from_map})' if isinstance(stock_id_from_map, str) else \
                '(' + ','.join(stock_id_from_map) + ')'
            cql = f"select * from ivol.stocksymbol where stock_id in {stock_ids} allow filtering"
        else:
            cql = f"select * from ivol.rootproperty where symbol='{root_security}' allow filtering"
        root_df = pd.DataFrame(session.execute(cql))
        if root_df.shape[0] < 1:
            raise Exception(f"Missing stock ids for {root_security}")
        return root_df.stock_id

    @staticmethod
    def find_futures_ids(session, underlying_id):
        cql = f"select * from ivol.fut_futures where underlying_id = {underlying_id} allow filtering"
        root_df = pd.DataFrame(session.execute(cql))
        if root_df.shape[0] < 1:
            raise Exception(f"Missing future ids for underlying id {underlying_id}")
        return root_df.futures_id

    @staticmethod
    def load_option_data_actual(session, start_date, end_date, calendar, root_security,
                                expiry_filter, strike_filter, call_put_filter, exercise_filter, use_1545table,
                                dates=[]):
        start = time.time()
        holidays = get_holidays(calendar, start_date, end_date)
        dts = filter(lambda x: is_business_day(x, holidays), date_range(start_date, end_date)) \
            if len(dates) == 0 else filter(lambda x: is_business_day(x, holidays), dates)
        # convert filter to list to avoid empty after iteration
        dts = list(dts)

        # find stock_id
        is_com = root_security in COMMODITY_ROOT_ID_AND_UNDERLYING_ID_PAIR
        if is_com:
            (root_id, underlying_id) = COMMODITY_ROOT_ID_AND_UNDERLYING_ID_PAIR[root_security]
            futures_ids = DatalakeCassandra.find_futures_ids(session, underlying_id)
            futures_id = '(' + ','.join(map(lambda x: str(x), list(futures_ids))) + ')'
        else:
            stock_ids = DatalakeCassandra.find_stock_ids(session, root_security)
            stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        date_needed = "','".join(i.strftime("%Y-%m-%d") for i in dts)

        # load data
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        table_name = "ivol.fut_option" if is_com else ("ivol.optionvalue_1545" if use_1545table else "ivol.optionvalue")

        if (expiry_filter is not None and expiry_filter.is_date_dependent()) \
                or (strike_filter is not None and strike_filter.is_date_dependent()) \
                or (call_put_filter is not None and call_put_filter.is_date_dependent()) \
                or (exercise_filter is not None and exercise_filter.is_date_dependent()):
            dfs = []
            for dt in dts:
                dt_str = dt.strftime('%Y-%m-%d')
                expiry_filter_str = "" if expiry_filter is None else expiry_filter.cassandra_filter_string(dt)
                strike_filter_str = "" if strike_filter is None else strike_filter.cassandra_filter_string(dt)
                call_put_filter_str = "" if call_put_filter is None else call_put_filter.cassandra_filter_string(dt)
                exercise_filter_str = "" if exercise_filter is None else exercise_filter.cassandra_filter_string(dt)
                if is_com:
                    cql = f"SELECT * FROM {table_name} where root_id={root_id} and futures_id in {futures_id} " \
                          f"and t_date = '{dt_str}'{expiry_filter_str}{strike_filter_str}{call_put_filter_str}{exercise_filter_str} allow filtering"
                else:
                    cql = f"SELECT * FROM {table_name} where stock_id in {stock_id} and t_date = '{dt_str}'" \
                          f"{expiry_filter_str}{strike_filter_str}{call_put_filter_str}{exercise_filter_str} allow filtering"
                dfs.append(pd.DataFrame(session.execute(cql)))
            df = pd.concat(dfs, ignore_index=True)
        else:
            expiry_filter_str = "" if expiry_filter is None else expiry_filter.cassandra_filter_string()
            strike_filter_str = "" if strike_filter is None else strike_filter.cassandra_filter_string()
            call_put_filter_str = "" if call_put_filter is None else call_put_filter.cassandra_filter_string()
            exercise_filter_str = "" if exercise_filter is None else exercise_filter.cassandra_filter_string()
            if start_date_str == end_date_str:
                date_str = f"and t_date = '{start_date_str}'"
            else:
                date_str = f"and t_date in ('{date_needed}')"
            if is_com:
                cql = f"""SELECT * FROM {table_name} where root_id={root_id} and futures_id in {futures_id} {date_str}
                        {expiry_filter_str}{strike_filter_str}{call_put_filter_str}{exercise_filter_str} allow filtering"""
            else:
                cql = f"""SELECT * FROM {table_name} where stock_id in {stock_id} {date_str}
                        {expiry_filter_str}{strike_filter_str}{call_put_filter_str}{exercise_filter_str} allow filtering"""
            df = pd.DataFrame(session.execute(cql))

        if not is_com:
            df = df[~df["expiration_id"].isna()]

        if df.empty:
            return df

        if not is_com:
            df["expiration_id"] = df["expiration_id"].apply(int)
            expiration_id = '(' + ','.join(map(lambda x: str(x), df["expiration_id"].dropna().unique())) + ')'
            cql = f"SELECT * FROM ivol.expirations where expiration_id in {expiration_id} allow filtering"
            expiration_id_df = pd.DataFrame(session.execute(cql))
            exp_row = '(' + ','.join(map(lambda x: str(x), expiration_id_df["exp_row"].unique())) + ')'
            cql = f"SELECT * FROM ivol.expiration_rules where exp_row in {exp_row} allow filtering"
            exp_row_df = pd.DataFrame(session.execute(cql))

            expiration_rule = expiration_id_df.set_index("exp_row").join(exp_row_df.set_index("exp_row"),
                                                                         how="left").set_index("expiration_id")
            if root_security == "SX5E":
                df["expiration_rule"] = df.apply(lambda x: expiration_rule.loc[9967, "description"]
                if x.expiration_id == 9968 else expiration_rule.loc[x.expiration_id, "description"], axis=1)
            else:
                df["expiration_rule"] = df['expiration_id'].apply(lambda x: expiration_rule.loc[x, "description"])

        df['underlying'] = df.apply(lambda row: ticker_from_option_root(root_security), axis=1)

        if is_com:
            return df
        # forward
        forward_cql = f"select t_date, expiration_date, forward_price from ivol.fut_spot_forward_price where stock_id in {stock_id} and t_date in ('{date_needed}') allow filtering"
        forward_df = pd.DataFrame(session.execute(forward_cql))

        # some underliers have fwd listed in <IVP_Parameterization>: try pulling data from here
        if forward_df.empty:
            forward_cql = forward_cql.replace('fut_spot_forward_price', 'IVP_Parameterization').replace('_price', '')
            forward_df = pd.DataFrame(session.execute(forward_cql))
            forward_df.rename(columns={'forward': 'forward_price'}, inplace=True)

        if forward_df.empty:
            return df
        else:
            df = pd.merge(df, forward_df, how='left', on=['t_date', 'expiration_date'])
            return df

    def load_option_data(self, start_date, end_date, calendar, root_security,
                         expiry_filter, strike_filter, call_put_filter, exercise_filter,
                         file_cache_path=None, use_1545table=False, dates=[]):
        session = self.get_session()
        loaded = False
        if file_cache_path is not None and isinstance(expiry_filter, ExpiryFilterByDateOffset):
            for f in os.listdir(file_cache_path):
                match = re.match(r'([\w|\s]+)@(\d{4}-\d{2}-\d{2})@(\d{4}-\d{2}-\d{2})@(\d+).pkl', f)
                if match is not None:
                    groups = tuple(match.groups())
                    if root_security == groups[0] and start_date >= datetime.datetime.strptime(groups[1],
                                                                                               '%Y-%m-%d') and end_date <= datetime.datetime.strptime(
                        groups[2], '%Y-%m-%d') and expiry_filter.max_expiry_offset <= float(groups[3]):
                        loaded = True
                        file_cache = f
                        break
        if loaded:
            print('load from ' + file_cache)
            return pd.read_pickle(file_cache_path + file_cache)
        else:
            df = DatalakeCassandra.load_option_data_actual(
                session, start_date, end_date, calendar, root_security,
                expiry_filter, strike_filter, call_put_filter, exercise_filter, use_1545table, dates
            )
            if file_cache_path is not None and isinstance(expiry_filter, ExpiryFilterByDateOffset):
                file_name = f"{root_security}@{start_date.strftime('%Y-%m-%d')}@{end_date.strftime('%Y-%m-%d')}@{str(expiry_filter.max_expiry_offset)}.pkl"
                df.to_pickle(file_cache_path + file_name)
            return df

    def get_intraday_stock_data(self, stock_id, look_times):

        session = self.get_session()

        cql = "SELECT * FROM ivol.stock_minute where"

        if isinstance(stock_id, int):
            cql = cql + " stock_id={}".format(stock_id)
        elif isinstance(stock_id, list):
            if '9327' in stock_id:
                stock_id=['9327']
            cql = cql + " stock_id IN ({})".format(','.join(stock_id))
        else:
            raise Exception('Incorrect Stock Id, has to be either list or int')

        if isinstance(look_times, str):
            cql = cql + " and t_date='{}'".format(look_times)
        elif isinstance(look_times, list):
            cql = cql + " and t_date IN {}".format(tuple(look_times))
        else:
            raise Exception('Incorrect look_times, has to be either list or str')

        df = pd.DataFrame(session.execute(cql))
        if df.empty:
            return df
        df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        return df

    def get_stock_data(self, root_security, start_date, end_date):
        """

        :param stock_id: int
        :param start_date: datetime.datetime
        :param end_date: datetime.datetime
        :return: pd.DataFrame()
        """
        session = self.get_session()
        # if start_date is not None:
        #     if isinstance(start_date, datetime.date) or isinstance(start_date, datetime.datetime):
        #         start_date = start_date.strftime("%Y-%m-%d_%H:%M:%S")
        # if end_date is None:
        #     end_date = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        # else:
        #     if isinstance(end_date, datetime.date) or isinstance(end_date, datetime.datetime):
        #         end_date = end_date.strftime("%Y-%m-%d_%H:%M:%S")

        # find stock_id
        stock_ids = DatalakeCassandra.find_stock_ids(session, root_security)
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        cql = "SELECT * FROM ivol.stockprice where stock_id in {}".format(stock_id)
        if start_date == end_date:
            cql = cql + " and t_date='{}'".format(start_date.strftime("%Y-%m-%d"))
        else:
            d_range = pd.bdate_range(start_date, end_date)
            dlist = "("
            for z in d_range:
                if z == d_range[-1]:
                    dlist += "'{}'".format(z.strftime("%Y-%m-%d"))
                else:
                    dlist += "'{}',".format(z.strftime("%Y-%m-%d"))
            dlist += ")"
            cql = cql + " and t_date in {}".format(dlist)

        df = pd.DataFrame(session.execute(cql))
        df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        # TODO add adjustment data
        return df

    def get_divs(self, root_security):
        session = self.get_session()

        # find stock_id
        stock_ids = DatalakeCassandra.find_stock_ids(session, root_security)
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        cql = "SELECT * FROM ivol.dividend where stock_id in {}".format(stock_id)
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
        # cql = cql + " and last_dvd_date<='{}'".format(end_date)
        df = pd.DataFrame(session.execute(cql))
        df.rename(columns={'last_dvd_date': 'tstamp'}, inplace=True)
        if len(df) > 0:
            df.set_index('tstamp', inplace=True, drop=True)
            df.index = pd.to_datetime(df.index.astype(str), infer_datetime_format=True, utc=True)
            df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
            df.term_date = pd.to_datetime(df.term_date.astype(str), infer_datetime_format=True, utc=True)
            df = df[df.t_date <= df.index]
            df = df[df.index < end_date]

        return df

    def get_corp_actions(self, root_security):
        session = self.get_session()

        # find stock_id
        stock_ids = DatalakeCassandra.find_stock_ids(session, root_security)
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        cql = "SELECT * FROM ivol.split where stock_id in {} and t_date >= '1980-01-01' and cause >= 0 ALLOW FILTERING".format(
            stock_id)
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
        # cql = cql + " and last_dvd_date<='{}'".format(end_date)
        df = pd.DataFrame(session.execute(cql))
        # df.rename(columns={'last_dvd_date': 'tstamp'}, inplace=True)
        if len(df) > 0:
            df.set_index('t_date', inplace=True, drop=True)
            df.index = pd.to_datetime(df.index.astype(str), infer_datetime_format=True, utc=True)
            # df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True, utc=True)
            # df.term_date = pd.to_datetime(df.term_date.astype(str), infer_datetime_format=True, utc=True)
            # df = df[df.t_date <= df.index]
            # df = df[df.index < end_date]

        return df

    def get_futures_price(self, source='fut_futures_price', futures_id=None, start_date=None, end_date=None):
        session = self.get_session()
        source = source.lower()
        if start_date is not None:
            start_date = format_isodate(start_date)
        if end_date is None:
            end_date = datetime.datetime.today().strftime("%Y-%m-%d")
        else:
            end_date = format_isodate(end_date)
        cql = "SELECT * FROM ivol.{} where".format(source)

        if isinstance(futures_id, int) or isinstance(futures_id, np.int64):
            cql += " futures_id={}".format(futures_id)
        elif isinstance(futures_id, list):
            cql += " futures_id IN {}".format(tuple(futures_id))
        elif futures_id is None:
            pass
        else:
            raise Exception('Incorrect futures_id, has to be either list or int or None')

        if start_date is not None:
            cql += " and t_date>='{}'".format(start_date)
        if end_date is not None:
            cql += " and t_date<='{}'".format(end_date)
        cql += ' allow filtering'

        df = pd.DataFrame(session.execute(cql))
        # df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True)
        df.t_date = df.t_date.astype(str)
        df.rename(columns={'t_date': 'tstamp', 'price_opt': 'price'}, inplace=True)
        return df

    def get_ivol_futures(self, root, expiration_months=None, expiration_years=None):
        """
        :param root: str
        :param expiration_months: list[int]
        :param expiration_years: list[int] 4 digit yr
        :return: dataframe
        """
        session = self.get_session()

        fut_root_id = self.get_ivol_fut_root_id(root)

        source = 'fut_futures'
        cql = "SELECT * FROM ivol.{} where underlying_id={} allow filtering".format(source, fut_root_id)
        df = pd.DataFrame(session.execute(cql))

        # handle CO double listing
        if root == 'CO':
            df_eb = df[[datetime.datetime.strptime(str(df.iloc[ix].expiration_date), '%Y-%m-%d') <
                        datetime.datetime(2015, 1, 30) for ix in range(len(df))]]
            df_br = df[['BR' in x for x in df.symbol.values]]
            df = pd.concat([df_eb, df_br])
            df = df[df.futures_id != 456063]
        elif root == "SR1":
            df = df[~df.contract_month.isna()]

        if (expiration_months is not None) or (expiration_years is not None):
            df_mask = (True if expiration_months is None else df.expiration_month_id.isin(expiration_months)) & \
                      (True if expiration_years is None else df.expiration_year.isin(expiration_years))
            df = df[df_mask]
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        return df

    def get_ivol_future_id(self, future):
        session = self.get_session()

        fut_root_id = self.get_ivol_fut_root_id(future.root)

        source = 'fut_futures'
        cql = """
              SELECT * FROM ivol.{} where underlying_id={} and expiration_date='{}' allow filtering
              """.format(source, fut_root_id, format_isodate(future.expiration))
        df = pd.DataFrame(session.execute(cql))
        assert len(df) == 1
        return df.futures_id.iat[0]

    def get_ivol_fut_root_id(self, root_symbol):

        # session = self.get_session()
        #
        # source = 'fut_underlying'
        # cql = "SELECT * FROM ivol.{} where symbol = '{}' allow filtering".format(source, root_symbol)
        # df = pd.DataFrame(session.execute(cql))
        # if df.shape[0] > 1:
        #     cql = "SELECT * FROM IVOL.EXCHANGE"
        #     exchange_df = pd.DataFrame(session.execute(cql))
        #     df = df.join(exchange_df.rename(columns={"name": "exchange_name"}).set_index("exchange_id"), on="exchange_id")
        # print(df.transpose())
        # assert len(df) == 1
        # return df.underlying_id.at[0]
        root_to_id = {'CL': 680, 'W': 27, 'C': 5, 'CO': 6467, 'VX': 1108, 'GC': 96, 'NG': 684, 'S': 22, 'HG': 97,
                      'TU': 24, 'TY': 25, 'US': 26, 'FV': 8,
                      'FF': 7,
                      'SR1': 2094, 'SR3': 6855,
                      'ES': 215}
        return root_to_id[root_symbol]

    def get_ivol_opt_root_id(self, root_symbol, date, include_weekly, weekly_expiry_filter):
        """
        mapping to identify correct underlier in the ivol db
        """
        if include_weekly:
            weekly_tickers = ["FV", "TU", "TY", "US"]
            if root_symbol not in weekly_tickers:
                raise Exception(f"Only support weekly {','.join(weekly_tickers)} for now but found {root_symbol}")
            if weekly_expiry_filter is None:
                # underlying_id = self.get_ivol_fut_root_id(root_symbol)
                # print(underlying_id)
                # session = self.get_session()
                # ref_source = 'fut_option_root'
                # cql = "SELECT * FROM ivol.{} where underlying_id = {} allow filtering".format(ref_source, underlying_id)
                # df = pd.DataFrame(session.execute(cql))
                # root_ids = sorted(df[df.production_name.str.contains("Weekly") | (df.symbol == root_symbol)].root_id.tolist())
                # print(root_ids)
                root_ids_map = {
                    "FV": [5, 623, 624, 625, 641, 651, 1780, 1785, 1796, 1804, 1826, 2861, 2862, 2879, 2885, 2922, 3240, 3241, 3242, 3258, 3264, 3270, 3276],
                    "TY": [13, 629, 630, 631, 643, 653, 1779, 1784, 1795, 1803, 1829, 2865, 2866, 2881, 2887, 2924, 3246, 3247, 3248, 3260, 3266, 3272, 3278],
                    "TU": [12, 626, 627, 628, 642, 652, 1776, 1787, 1797, 1805, 1827, 2863, 2864, 2880, 2886, 2923, 3243, 3244, 3245, 3259, 3265, 3271, 3277],
                    "US": [14, 632, 633, 634, 644, 654, 1775, 1786, 1799, 1801, 1828, 2867, 2868, 2882, 2888, 2925, 3249, 3250, 3251, 3261, 3267, 3273, 3279],
                }
                root_ids = root_ids_map[root_symbol]
            else:
                underlying_id = self.get_ivol_fut_root_id(root_symbol)
                session = self.get_session()
                ref_source = 'fut_option_root'
                cql = "SELECT * FROM ivol.{} where underlying_id = {} allow filtering".format(ref_source, underlying_id)
                df = pd.DataFrame(session.execute(cql))
                if weekly_expiry_filter is None:
                    root_ids = sorted(df[df.production_name.str.contains("Weekly") | (df.symbol == root_symbol)].root_id.tolist())
                else:
                    filter_str = "|".join([f"Weekly {x}" for x in weekly_expiry_filter])
                    df = df[df.production_name.str.contains(filter_str)]
                    df[["day", "n"]] = df["production_name"].str.split(expand=True)[[1, 5]]
                    df["n"] = df["n"].astype(int)
                    df["weekday"] = df["day"].map({"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4})
                    df["date"] = df.apply(lambda row: n_th_weekday_of_month(date.year, date.month, row["n"], row["weekday"]), axis=1)
                    date_next_month = date + relativedelta(months=1)
                    df["date_next_month"] = df.apply(lambda row: n_th_weekday_of_month(date_next_month.year, date_next_month.month, row["n"], row["weekday"]), axis=1)
                    df["bds_curr_month"] = df["date"].apply(lambda x: 0 if pd.isna(x) else count_business_days(date, x))
                    df["bds_next_month"] = df["date_next_month"].apply(lambda x: 0 if pd.isna(x) else count_business_days(date, x))
                    min_days = 2
                    if df[df.bds_curr_month > min_days].shape[0] >= 2:
                        root_ids = df[df.bds_curr_month > min_days].sort_values("date").iloc[:2]["root_id"].tolist()
                    else:
                        root_ids_curr_month = df[df.bds_curr_month > min_days]["root_id"].tolist()
                        num_root_ids_next_month = 2 - len(root_ids_curr_month)
                        root_ids_next_month = df[df.bds_next_month > min_days].sort_values("date").iloc[:num_root_ids_next_month]["root_id"].tolist()
                        root_ids = set(root_ids_curr_month + root_ids_next_month)
                    root_ids = sorted(root_ids)
        else:
            # session = self.get_session()
            # ref_source = 'fut_option_root'
            # cql = "SELECT * FROM ivol.{} where symbol = '{}' allow filtering".format(
            #     ref_source, root_symbol)
            # df = pd.DataFrame(session.execute(cql))
            # print(df.transpose())
            # assert len(df) == 1 # avoid ambiguous underliers
            # return df.root_id.at[0]
            opt_root_to_id = {'CL': 107, 'W': 15, 'C': 2,
                              'CO': 1438 if date < datetime.datetime(2015, 1, 9) else 1442,
                              'GC': 35, 'NG': 110,
                              'S': 10 if date < datetime.datetime(2017, 3, 21) else 1661, 'HG': 36,
                              'TU': 12, 'TY': 13, 'US': 14, 'FV': 5,
                              'FF': 4,
                              'SR1': 2216,
                              # 'SR3': 1950,
                              'SR3': 1958,
                              'ES': 63,
            }
            root_ids = opt_root_to_id[root_symbol]
        return root_ids

    def get_ivol_options_for_future(self, futures_id, root_symbol, date, include_weekly=False, weekly_expiry_filter=None, skip_future=False):

        session = self.get_session()

        # underlying_id = self.get_ivol_fut_root_id(root_symbol)
        opt_root_id = self.get_ivol_opt_root_id(root_symbol, date, include_weekly, weekly_expiry_filter)
        date = format_isodate(date)

        price_source = 'fut_option'

        if root_symbol == 'SR3':
            # SOFR futures vol data needs revisiting
            # assert 0 > 1

            SR3_root_ids = [1960, 1950, 2179, 1957,
                            2171, 2175, 2219, 1949,
                            1955, 1954, 1958, 1952,
                            1956, 2170, 2217, 1953,
                            2218, 2180, 2178, 2176,
                            1959, 2172, 1951, 2177]
            SR3_root_ids = [1958]   # Standard Options, exclude Mid Curve options for now
            cql = """SELECT *
                        FROM ivol.{} WHERE root_id in ({}) and futures_id={}
                    """.format(price_source, ','.join([str(x) for x in SR3_root_ids]), futures_id)
            cql += " AND t_date = '{}'".format(date)
            cql += ' allow filtering'
            df = pd.DataFrame(session.execute(cql))
            df = df[df.iv > 0]
        else:
            opt_root_id_list = opt_root_id if isinstance(opt_root_id, list) else [opt_root_id]
            if skip_future:
                df = []
                for fut_id in futures_id:
                    cql = f"SELECT * FROM ivol.{price_source} WHERE root_id in ({','.join([str(x) for x in opt_root_id_list])}) AND futures_id={fut_id} AND t_date = '{date}' allow filtering"
                    tmp = pd.DataFrame(session.execute(cql))
                    df.append(tmp)
                df = pd.concat(df)
            else:
                cql = """SELECT * FROM ivol.{} WHERE root_id in ({}) and futures_id={}
                        """.format(price_source, ','.join([str(x) for x in opt_root_id_list]), futures_id)
                cql += " AND t_date = '{}'".format(date)
                cql += ' allow filtering'
                df = pd.DataFrame(session.execute(cql))
        # df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True)
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        df.t_date = df.t_date.astype(str)
        df.rename(columns={'t_date': 'tstamp', 'settle': 'price'}, inplace=True)
        return df

    def get_ivol_options_price(self, futures_id, opt_root_id, strike, call_put,
                               expiry_date=None, start_date=None, end_date=None):
        """
        requires root symbol as there are sometimes look a like
        """
        session = self.get_session()

        if start_date is not None:
            start_date = format_isodate(start_date)
        if end_date is not None:
            end_date = format_isodate(end_date)
        if expiry_date is not None:
            expiry_date = format_isodate(expiry_date)

        price_source = 'fut_option'

        if opt_root_id == 1950:
            SR3_root_ids = [1960, 1950, 2179, 1957,
                            2171, 2175, 2219, 1949,
                            1955, 1954, 1958, 1952,
                            1956, 2170, 2217, 1953,
                            2218, 2180, 2178, 2176,
                            1959, 2172, 1951, 2177]
            cql = """SELECT *
                        FROM ivol.{} WHERE root_id in ({}) and futures_id={} and strike={} and call_put='{}'
                    """.format(price_source, ','.join([str(x) for x in SR3_root_ids]),
                               futures_id, strike, call_put.upper())
        else:
            opt_root_id_list = opt_root_id if isinstance(opt_root_id, list) else [opt_root_id]
            cql = """SELECT * FROM ivol.{} WHERE root_id in ({}) and futures_id={} and strike={} and call_put='{}'
            """.format(price_source, ','.join([str(x) for x in opt_root_id_list]), futures_id, strike, call_put.upper())

        if expiry_date is not None:
            cql += " and expiration_date='{}'".format(expiry_date)
        if start_date is not None:
            cql += " and t_date>='{}'".format(start_date)
        if end_date is not None:
            cql += " and t_date<='{}'".format(end_date)
        cql += ' allow filtering'

        df = pd.DataFrame(session.execute(cql))
        # df.t_date = pd.to_datetime(df.t_date.astype(str), infer_datetime_format=True)
        df.expiration_date = pd.to_datetime(df.expiration_date.astype(str), infer_datetime_format=True)
        df.t_date = df.t_date.astype(str)
        df.rename(columns={'t_date': 'tstamp', 'settle': 'price'}, inplace=True)
        return df

    def get_ivol_interest_rate(self, currency, tenor, date):
        session = self.get_session()

        source = 'InterestRate'
        cql = """
              SELECT * FROM ivol.{} where currency='{}' and period={} and t_date='{}' allow filtering
              """.format(source, currency, str(tenor), format_isodate(date))
        df = pd.DataFrame(session.execute(cql))
        assert len(df) == 1
        return df.rate.iat[0]

    def get_intraday_option_universe(self, dt, root_security, expiry_filter, strike_filter, call_put_filter,
                                     exercise_filter):

        start = time.time()

        session = self.get_session()

        expiry_filter_str = "" if expiry_filter is None else expiry_filter.cassandra_filter_string(dt)
        strike_filter_str = "" if strike_filter is None else strike_filter.cassandra_filter_string()
        call_put_filter_str = "" if call_put_filter is None else call_put_filter.cassandra_filter_string()
        exercise_filter_str = "" if exercise_filter is None else exercise_filter.cassandra_filter_string().replace(
            'root_style', 'style')

        stock_ids = DatalakeCassandra.find_stock_ids(session, root_security)
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        table_name = "ivol.option_minute"

        cql = f"SELECT * FROM {table_name} where stockid in {stock_id} and t_date = '{dt}'{expiry_filter_str}{strike_filter_str}{call_put_filter_str}{exercise_filter_str} allow filtering"
        df = pd.DataFrame(session.execute(cql))

        df['t_date'] = df['t_date'].dt.tz_localize('UTC').dt.tz_convert("US/Eastern").dt.tz_localize(None)
        df.rename(columns={'t_date': 'tstamp'}, inplace=True)
        df['stock_symbol'] = df['stock_symbol'].str.replace('SPXPM','SPX')

        return df

    def get_intraday_option_series(self, start_date, end_date, frequency, option):

        session = self.get_session()

        strike = option.strike
        expiration = option.expiration
        call_put = option.is_call
        root = option.root

        stock_ids = DatalakeCassandra.find_stock_ids(session, root)
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        minutes = pd.date_range(start_date, end_date, freq=frequency).tolist()
        utc_minutes = [f"{set_timezone(time.to_pydatetime(), 'UTC')}" for time in minutes]

        cql = '''select option_symbol as symbol, call_put, expiration_date, strike, t_date, price_ask, price_bid,
                   price_opt as underlyer_price_mid, volume, iv, delta, gamma, vega, theta from ivol.option_minute
                   where stockid in {} and t_date in ('{}') and expiration_date ='{}' and strike = {} and call_put = '{}' allow filtering
                    '''.format(stock_id, "','".join(utc_minutes),
                               str(expiration)[:10], strike, "C" if call_put else "P")

        df = pd.DataFrame(session.execute(cql))

        df.loc[:, 'price'] = df[['price_ask', 'price_bid']].mean(axis=1)
        df.sort_values('t_date', inplace=True)


        return df

    def get_intraday_option(self, dt, option, market_start, market_end):

        start = time.time()

        session = self.get_session()

        strike = option.strike
        expiration = option.expiration
        call_put = option.is_call
        root = option.root
        listed_ticker = option.listed_ticker
        schedule = pd.date_range(dt, option.expiration + relativedelta(days=1), freq='1T')
        schedule = schedule[schedule.indexer_between_time(market_start, market_end)].strftime("%Y-%m-%d %H:%M:%S")
    
        stock_ids = DatalakeCassandra.find_stock_ids(session, root)
        
        stock_id = '(' + ','.join(map(lambda x: str(x), list(stock_ids))) + ')'

        cql = '''select option_symbol as symbol, call_put, expiration_date, strike, t_date, price_ask, price_bid,
                   price_opt as underlyer_price_mid, volume, iv, delta, gamma, vega, theta from ivol.option_minute
                   where stockid in {} and t_date in ('{}') and expiration_date ='{}' and strike = {} and call_put = '{}' and option_symbol='{}' allow filtering
                    '''.format(stock_id, "','".join(schedule),
                               str(expiration)[:10], strike, "C" if call_put else "P", listed_ticker)

        df = pd.DataFrame(session.execute(cql))

        df.loc[:, 'price'] = df[['price_ask', 'price_bid']].mean(axis=1)
        df.sort_values('t_date', inplace=True)

        df['t_date'] = df['t_date'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.tz_localize(None)

        cleanning=intrday_option_data_cleaning(option,df,'9:35:00','16:00:00')
        df=cleanning.process()

        return df.to_dict('record')

    def get_intraday_future_data(self, dt, future, allow_back_fill):

        session = self.get_session()
        ticker = future.listed_ticker
        dates = pd.date_range(dt - datetime.timedelta(minutes=allow_back_fill), dt, freq='1min').strftime(
            "%Y-%m-%d %H:%M:%S")

        cql = '''select ticker, trade_tstamp as t_date, open, high, low, close from kibot.minute_data where ticker='{}' and
                    trade_tstamp in ('{}') 
                    '''.format(ticker.split(' ')[0], "','".join(dates))

        data = pd.DataFrame(session.execute(cql))

        retry=0

        while len(data) == 0:
            print(retry)
            retry+=1
            if retry > 10:
                print(f'Empty data for future {ticker} at time {dt}')
                data = pd.DataFrame(index=[0],
                                    columns=['ticker', 't_date', 'open',
                                            'high', 'low', 'close'],
                                    data=[[np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan]])
                break
            else:
                dates = pd.date_range(dt - datetime.timedelta(minutes=60*retry), dt, freq='1min').strftime(
                    "%Y-%m-%d %H:%M:%S")

                cql = '''select ticker, trade_tstamp as t_date, open, high, low, close from kibot.minute_data where ticker='{}' and
                        trade_tstamp in ('{}') 
                        '''.format(ticker.split(' ')[0], "','".join(dates))

                data = pd.DataFrame(session.execute(cql))

        return data.loc[data.index[0]]
    
    def get_intraday_future_series(self, future, start_date, end_date, market_start, market_end):

        session = self.get_session()
        ticker = future.listed_ticker
        dates = pd.date_range(start_date, end_date, freq='1T')
        dates = dates[dates.indexer_between_time(market_start, market_end)].strftime("%Y-%m-%d %H:%M:%S")

        cql = '''select ticker, trade_tstamp as t_date, open, high, low, close from kibot.minute_data where ticker='{}' and
                    trade_tstamp in ('{}') 
                    '''.format(ticker.split(' ')[0], "','".join(dates))

        data = pd.DataFrame(session.execute(cql)).set_index('t_date').sort_index().drop(columns='ticker')
        
        data.index = data.index.tz_localize('UTC').tz_convert('US/Eastern').tz_localize(None)

        return data.to_dict('index')

    def get_future_ref(self, root, suffix):

        session = self.get_session()

        cql = """
            select * from bbg.futures_contract_definition where root='{}' and suffix='{}' allow filtering
        """.format(root, suffix)

        data = pd.DataFrame(session.execute(cql))

        return data
    
    def get_interest_rate(self, t_date, maturity, currency):

        session = self.get_session()

        if maturity > 360:
            print('For table ivol.InterestRate, period of insterest rate should be no larger than 360.')
            maturity = 360

        maturity = int(maturity)
        cql = ''' SELECT t_date, rate as interest_rate from ivol.interestrate where 
                      t_date = '{}' and currency='{}' and period = {}
                      '''.format(t_date,currency, max(1, maturity))
        
        r = pd.DataFrame(session.execute(cql))
        r['interest_rate'] = r['interest_rate']/100

        return r
    
    def get_dividend(self, root, start_date, end_date):

        session = self.get_session()

        stock_ids = ','.join(DatalakeCassandra.find_stock_ids(session, root).astype(str).to_list())

        cql="""SELECT sum(real_last_dvd_amount) as dividend_amount from ivol.dividend where stock_id in ({}) and last_dvd_date >= '{}' and last_dvd_date <= '{}' allow filtering""".format(stock_ids,start_date,end_date)

        dividend = pd.DataFrame(session.execute(cql))
        
        return dividend

    def get_option_dividend(self,option,start_datetime, price):
        
        q_amount=self.get_dividend(option.root,start_datetime.strftime("%Y-%m-%d"),option.expiration.strftime("%Y-%m-%d"))['dividend_amount'].item()
        t_diff = option.expiration - start_datetime
        days_to_maturity = t_diff.days + t_diff.seconds / 86400
        q = q_amount * 365 / days_to_maturity / price
        
        return q
    
    def get_option_interest_rate(self,option,t):
        
        days_to_maturity = (option.expiration - t).days
        day = t.date()
        r = self.get_interest_rate(day, days_to_maturity, option.currency)
        
        return r['interest_rate'].values[0]
    



if __name__ == '__main__':
    from ..tradable.option import Option
    from ..tradable.future import Future
    import datetime as dt
    import time
    start=time.time()
    cassandra = DatalakeCassandra()
    # ref = cassandra.get_interest_rate('2024-01-18', 360, 'USD')
    # print(ref)
    # q=cassandra.get_dividend('SPX', '2024-01-01', '2024-06-01')
    # print(q)
    # option=Option(root='SPX',is_call=False,is_american=False,underlying='SPX',currency='USD',tz_name=None,contract_size=None,expiration=dt.datetime(2024, 1, 18, 0, 0),strike=4785,listed_ticker='SPXW  240118P04785000')

    # data=pd.DataFrame(cassandra.get_intraday_option(dt.datetime(2024,1,11,16,0,0),option,'9:35:00','16:00:00'))

    # print(data)
    # print(time.time()-start)