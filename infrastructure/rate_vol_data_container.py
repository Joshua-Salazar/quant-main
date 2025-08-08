import math
import re

import pandas as pd
import numpy as np
import os
from datetime import datetime
import numbers

from ..infrastructure.data_container import DataContainer
from ..data.market import read_jp_swaption_vol_cubes, remove_slice_with_too_few_strikes
from ..infrastructure import market_utils, jp_rate_vol_config
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from datalake.datalakeapi import DataLakeAPI
from ..data.market import get_expiry_in_year


class RateVolContainer(DataContainer):
    def __init__(self, currency: str):
        self.market_key = market_utils.create_swaption_vol_cube_key(currency)

    def get_market_key(self):
        return self.market_key

    def get_vol_cube(self, dt=None):
        return self._get_vol_cube(dt)

    def get_market_item(self, dt):
        return self.get_vol_cube(dt)


class RateVolRequest(IDataRequest):
    def __init__(self, start_date, end_date, currency, strikes_relative_to_atmf=True):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.strikes_relative_to_atmf = strikes_relative_to_atmf


class RateVolDataSourceDict(IDataSource):
    def __init__(self, data_dict={}, data_key_path=()):
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

    def initialize(self, data_request):
        container = RateVolContainer(data_request.currency)

        def _get_vol_cube(dt):
            if dt is None:
                return self.cache['data']
            else:
                return self.cache['data'][dt]

        container._get_vol_cube = _get_vol_cube
        return container


class JPRateVolDataSource(IDataSource):
    def __init__(self, rate_data_file=None, vol_data_file=None):
        self.rate_data_file = rate_data_file
        self.vol_data_file = vol_data_file
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    @staticmethod
    def load_with_data_cache(data_request, file_path, tgt_start_date, tgt_end_date, data_tickers, update_cache, datalake):
        if os.path.exists(file_path):
            data_df = pd.read_csv(file_path, index_col='tstamp')
            file_start_date = pd.Timestamp(data_df.index[0])
            file_end_date = pd.Timestamp(data_df.index[-1])
            if file_end_date >= tgt_end_date and file_start_date <= tgt_start_date:
                return data_df.reset_index()
            else:
                if file_end_date < tgt_end_date and file_start_date > tgt_start_date:
                    load_db_from = tgt_start_date
                    load_db_to = tgt_end_date
                elif file_end_date < tgt_end_date:
                    load_db_from = pd.Timestamp(data_df.index[-1])
                    load_db_to = tgt_end_date
                else:
                    load_db_from = tgt_start_date
                    load_db_to = pd.Timestamp(data_df.index[0])
        else:
            load_db_from = tgt_start_date
            load_db_to = tgt_end_date

        # load new data from db
        if os.path.exists(file_path):
            tickers = data_df.columns
        else:
            tickers = data_tickers

        ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                       np.split(tickers, [i * 100 for i in range(1, len(tickers) // 100 + 1)])]
        data_list = []
        for ticker_str in ticker_strs:
            data_list.append(datalake.getData('JPM_DATAQUERY', ticker_str, "VALUE", start=load_db_from, end=load_db_to))
        result_df = pd.concat(data_list).reset_index()

        if data_request.currency == 'USD':
            ## add SOFR data post 10Jun24 (JPM discontinued LIBOR Swaption data)
            sofr_tickers = [ x.replace('_SWAPTION', '_SOFRSWAPTION') for x in tickers ]
            ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                           np.split(sofr_tickers, [i * 100 for i in range(1, len(sofr_tickers) // 100 + 1)])]
            data_list = []
            for ticker_str in ticker_strs:
                data_list.append(datalake.getData('JPM_DATAQUERY', ticker_str, "VALUE", start=pd.Timestamp(data_df.index[-1]), end=tgt_end_date))
            sofr_result_df = pd.concat(data_list).reset_index()
            sofr_result_df.ticker = [x.replace('SOFR', '') for x in sofr_result_df.ticker.values]
            if not result_df.empty:
                result_df = result_df[result_df.tstamp <= pd.Timestamp(data_df.index[-1])]
                sofr_result_df = sofr_result_df[[x != result_df.tstamp.max() for x in sofr_result_df.tstamp.values]]
            result_df = pd.concat([result_df,sofr_result_df])

        result_df['tstamp'] = result_df['tstamp'].dt.strftime("%Y-%m-%d 00:00:00")

        new_data = result_df.pivot(index='tstamp', columns='ticker', values='VALUE').copy()
        missing_tickers = list(set(tickers) - set(new_data.columns))
        available_tickers = list(set(tickers) - set(missing_tickers))
        new_data = new_data[available_tickers]
        for t in missing_tickers:
            new_data[t] = np.NaN

        # join with existing if available
        if os.path.exists(file_path):
            data_df = data_df[~data_df.index.isin(new_data.index)]
            full_df = pd.concat([data_df, new_data]).sort_index()
        else:
            full_df = new_data.sort_index()

        if update_cache:
            full_df.to_csv(file_path)

        return full_df.reset_index()

    def initialize(self, data_request):
        container = RateVolContainer(data_request.currency)

        assert data_request.currency == 'USD'
        assert data_request.strikes_relative_to_atmf

        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        vol_data_file_default = f'{data_file_folder}Solutions/backtests/data_cache/rates/{data_request.currency}_vol_jp_data_cache.csv'
        rate_data_file_default = f'{data_file_folder}Solutions/backtests/data_cache/rates/{data_request.currency}_rates_jp_data_cache.csv'
        self.vol_data_file = vol_data_file_default if (self.vol_data_file is None) else self.vol_data_file
        self.rate_data_file = rate_data_file_default if (self.rate_data_file is None) else self.rate_data_file

        vol_data_df = JPRateVolDataSource.load_with_data_cache(data_request, self.vol_data_file, pd.Timestamp(data_request.start_date), pd.Timestamp(data_request.end_date), jp_rate_vol_config.rate_vol_config['rate_vol_tickers'], update_cache=True, datalake=self.datalake)
        rate_data_df = JPRateVolDataSource.load_with_data_cache(data_request, self.rate_data_file, pd.Timestamp(data_request.start_date), pd.Timestamp(data_request.end_date), jp_rate_vol_config.rate_vol_config['rate_tickers'], update_cache=True, datalake=self.datalake)

        vol_data_df = vol_data_df[(vol_data_df['tstamp'] >= data_request.start_date.strftime('%Y-%m-%d %H:%M:%S')) & (vol_data_df['tstamp'] <= data_request.end_date.strftime('%Y-%m-%d %H:%M:%S'))]
        rate_data_df = rate_data_df[(rate_data_df['tstamp'] >= data_request.start_date.strftime('%Y-%m-%d %H:%M:%S')) & (rate_data_df['tstamp'] <= data_request.end_date.strftime('%Y-%m-%d %H:%M:%S'))]

        self.data_dict = read_jp_swaption_vol_cubes(vol_data_df, rate_data_df, strike_relative_to_atmf=True)

        def _get_vol_cube(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_vol_cube = _get_vol_cube
        return container


class RateVolCitiDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    @staticmethod
    def get_citi_vol_tickers(data_request):
        expiries = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "15M", "18M", "21M",
                    "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "20Y",
                    "25Y", "30Y", "40Y", "50Y"]
        tenors = expiries
        strikes = ["M200", "M100", "M75", "M50", "M25", "M10", "10", "25", "50", "75", "100", "200"]
        tickers = [f"RATES.VOL.{data_request.currency}.OTM.NORMALABSOLUTE.ANNUAL.OTM_{strike}.{expiry}.{tenor}" for strike in strikes for expiry in expiries for tenor in tenors]
        tickers += [f"RATES.VOL.{data_request.currency}.ATM.NORMAL.ANNUAL.{expiry}.{tenor}" for expiry in expiries for tenor in tenors]
        return tickers

    @staticmethod
    def parse_citi_swaption_expiry_tenor_strike(ticker):
        left_pattern = r"RATES.VOL.(?:\w{3}).OTM.NORMALABSOLUTE.ANNUAL.OTM_(M\d{2,3}).(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))"
        right_pattern = r"RATES.VOL.(?:\w{3}).OTM.NORMALABSOLUTE.ANNUAL.OTM_(\d{2,3}).(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))"
        atm_pattern = r"RATES.VOL.(?:\w{3}).ATM.NORMAL.ANNUAL.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))"

        if re.match(left_pattern, ticker) is not None:
            groups = re.match(left_pattern, ticker).groups()
            strike = -float(groups[0][1:])
            expiry = groups[1]
            tenor = groups[2]
        elif re.match(right_pattern, ticker) is not None:
            groups = re.match(right_pattern, ticker).groups()
            strike = float(groups[0])
            expiry = groups[1]
            tenor = groups[2]
        elif re.match(atm_pattern, ticker) is not None:
            groups = re.match(atm_pattern, ticker).groups()
            strike = 0
            expiry = groups[0]
            tenor = groups[1]
        else:
            raise RuntimeError(f"cannot parse ticker {ticker}")
        expiry = get_expiry_in_year(expiry[:-1], expiry[-1])
        return expiry, tenor, strike

    @staticmethod
    def read_citi_swaption_vol_cubes(df):
        cubes = {}
        for record in df.to_dict('records'):
            dt = record['date']
            cube = {}
            for k, v in record.items():
                if k != 'date':
                    if isinstance(v, numbers.Number) and not math.isnan(v):
                        expiry, tenor, strike = RateVolCitiDataSource.parse_citi_swaption_expiry_tenor_strike(k)
                        strike = strike / 100
                        cube.setdefault(tenor, {}).setdefault(expiry, {}).setdefault(strike, v / 100.0)
            cubes[dt] = cube

        remove_slice_with_too_few_strikes(cubes, min_num_strikes=3)
        return cubes

    def initialize(self, data_request):
        assert data_request.strikes_relative_to_atmf

        container = RateVolContainer(data_request.currency)
        tickers = RateVolCitiDataSource.get_citi_vol_tickers(data_request)

        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        data_file = f'{data_file_folder}QuantResearch/CitiRateVols/{data_request.currency}.csv'
        df = pd.read_csv(data_file)
        selected_columns = df.columns.intersection(['date'] + tickers)
        df = df[selected_columns]

        if df.empty:
            raise Exception(f"Not found {tickers}")

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= data_request.start_date) & (df["date"] <= data_request.end_date)]

        self.data_dict = RateVolCitiDataSource.read_citi_swaption_vol_cubes(df)

        container._get_vol_cube = self._get_vol_cube
        return container

    def _get_vol_cube(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict[dt]


class RateVolDictDataSource(IDataSource):
    def __init__(self, data_dict=None):
        self.data_dict = data_dict

    def initialize(self, data_request):
        container = RateVolContainer(data_request.currency)

        def _get_vol_cube(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_vol_cube = _get_vol_cube
        return container


class RateVolFlatFileDataSource(IDataSource):
    """
    the columns of the file is following
    date: as of date in %Y-%m-%d format, e.g. 2024-01-24
    [expiry in years] [tenor string] [strike in bps relative to atmf rate], for example, 0.5 10y, -100
    the vol numbers are in percentage (e.g. it needs to be devided by 100)
    """
    def __init__(self, vol_data_file=None):
        self.vol_data_file = vol_data_file
        self.data_dict = {}

    def initialize(self, data_request):
        container = RateVolContainer(data_request.currency)

        vol_data_df = pd.read_csv(self.vol_data_file)

        cubes = {}
        for record in vol_data_df.to_dict('records'):
            dt = datetime.strptime(str(record['date']), "%Y-%m-%d")
            cube = {}
            for k, v in record.items():
                if k != 'date':
                    if isinstance(v, numbers.Number):
                        keys = k.split(' ')
                        tenor = keys[1]
                        expiry = float(keys[0])
                        strike = float(keys[2]) / 100
                        if not math.isnan(v):
                            cube.setdefault(tenor, {}).setdefault(expiry, {}).setdefault(strike, v / 100.0)
            cubes[dt] = cube

        self.data_dict = cubes

        def _get_vol_cube(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_vol_cube = _get_vol_cube
        return container


if __name__ == '__main__':
    from ..constants.ccy import Ccy
    start = datetime(2023, 5, 1)
    # start = datetime(2020, 7, 3)
    end = datetime(2023, 6, 20)

    # rate_vol_request = RateVolRequest(start, end, "USD")
    # rate_vol_container = JPRateVolDataSource().initialize(rate_vol_request)

    rate_vol_request = RateVolRequest(start, end, "GBP")
    rate_vol_container = RateVolFlatFileDataSource('/misc/Traders/Solutions/backtests/data_cache/rates/gbp_vol_manual.csv').initialize(rate_vol_request)

    print(rate_vol_container.get_vol_cube(end))
