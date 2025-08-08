import pandas as pd
import numpy as np
import numbers
from ..data.market import read_jp_rates
from ..infrastructure import market_utils, df_curve_config, citi_df_curve_config, jp_df_curve_config
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_item import MarketItem
from ..data.market import get_expiry_in_year
from ..dates.utils import add_business_days, datetime64_to_datetime
from datetime import datetime
from datalake.datalakeapi import DataLakeAPI
import json
import os
import requests


class DFCurveContainer(DataContainer):
    def __init__(self, currency: str, name: str, day_convention=None):
        name = "SWAP" if "OISSWAP" == name else name
        self.market_key = market_utils.create_spot_rates_key(currency, name)
        self.day_convention = day_convention

    def get_market_key(self):
        return self.market_key

    def get_spot_rate_curves(self, dt=None):
        return self._get_spot_rate_curves(dt)

    def get_market_item(self, dt):
        return DFCurveData(self.get_market_key(), self.get_spot_rate_curves(dt), self.day_convention)


# we make distinction between DataContainers and MarketItem as for some case data container has too much data
# and it is less efficient to have it in market whereas MarketItem can contain only one day worth of data
# each data container should have a get_market_item function, the return of which will be contained in the market object
class DFCurveData(MarketItem):
    def __init__(self, market_key, data_dict, day_convention=None):
        self.market_key = market_key
        self.data_dict = data_dict
        self.day_convention = day_convention

    def get_market_key(self):
        return self.market_key

    def clone(self):
        return DFCurveData(self.market_key, self.data_dict, self.day_convention)

    def apply(self, shocks, original_market, **kwargs) -> MarketItem:
        # TODO: for now just return original
        return DFCurveData(self.market_key, self.data_dict, self.day_convention)


class DFCurveRequest(IDataRequest):
    def __init__(self, start_date, end_date, currency, name, tenor=None, src=None):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.name = name
        self.tenor = tenor
        self.src = src


class DFCurveDataSourceDict(IDataSource):
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
        container = DFCurveContainer(data_request.currency, data_request.name)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.cache['data']
            else:
                return self.cache['data'][dt]

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class JPDFCurveDataSource(IDataSource):
    def __init__(self, data_file=None, allow_stale_df=False):
        """
        :param data_file: load date from given file
        :param load_ds: indicate to load data from data store rather than shared folder
        """
        self.data_file = data_file
        self.data_dict = {}
        # self.allow_stale_df=allow_stale_df
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    def update_data_file(self, file_path, tgt_end_date, tgt_start_date=None, currency=None, name=None) -> pd.DataFrame:
        if os.path.exists(file_path):
            data_df = pd.read_csv(file_path)
            file_end_date = pd.Timestamp(data_df["tstamp"].iloc[-1])
            if file_end_date >= tgt_end_date:
                return data_df

            data_df.set_index("tstamp", inplace=True)
            tickers = data_df.columns
            tgt_start_date = pd.Timestamp(data_df.index[-1])
        else:
            data_df = None
            tickers = df_curve_config.df_curve_config[currency]['tickers'].split(',')
        ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                       np.split(tickers, [i * 100 for i in range(1, len(tickers) // 100 + 1)])]
        data_list = []
        for ticker_str in ticker_strs:
            data_list.append(self.datalake.getData('JPM_DATAQUERY', ticker_str, fields="VALUE", start=tgt_start_date,
                                                   end=tgt_end_date))
        result_df = pd.concat(data_list).reset_index()
        new_data = result_df.pivot(index='tstamp', columns='ticker', values='VALUE')
        missing_tickers = list(set(tickers) - set(new_data.columns))
        available_tickers = list(set(tickers) - set(missing_tickers))
        new_data = new_data[available_tickers]
        for t in missing_tickers:
            new_data[t] = np.NaN
        if data_df is not None:
            new_data.drop(data_df.index[-1], inplace=True)
            full_df = pd.concat([data_df, new_data])
        else:
            full_df = new_data
        full_df.to_csv(file_path)
        return full_df.reset_index()

    def initialize(self, data_request):
        container = DFCurveContainer(data_request.currency, data_request.name)

        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        data_file_default = f'{data_file_folder}Solutions/backtests/data_cache/rates/{data_request.currency}_df_rates_jp_data_cache.csv'
        self.data_file = data_file_default if (self.data_file is None) else self.data_file
        data = self.update_data_file(self.data_file, pd.Timestamp(data_request.end_date), pd.Timestamp(data_request.start_date), data_request.currency, data_request.name)
        self.data_dict = read_jp_rates(data, data_request.name)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

                # try:
                #     return self.data_dict[dt]
                # except KeyError as e:
                #     if self.allow_stale_df:
                #         latest_day = max(k for k in self.data_dict if k < dt)
                #         return self.data_dict[latest_day]
                #     else:
                #         raise e

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class DFCurveCitiDataSource(IDataSource):
    def __init__(self, data_file=None):
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    @staticmethod
    def process_Citi_rates(rates_data):
        rates_data['tenor'] = [x.split('.')[-1] for x in rates_data.ticker]
        all_rates = {}
        for dt in rates_data.tstamp.unique():
            data_dt = rates_data[rates_data.tstamp == dt]
            dt = datetime64_to_datetime(dt)
            rates = {}
            for index, row in data_dt.iterrows():
                expiry = get_expiry_in_year(row.tenor[:-1], row.tenor[-1])
                v = float(row.VALUE)
                if isinstance(v, numbers.Number) and v < 1e20:
                    rates.setdefault(expiry, row.VALUE)
            all_rates[dt] = rates
        return all_rates

    def initialize(self, data_request):
        container = DFCurveContainer(data_request.currency, data_request.name)

        df_config = citi_df_curve_config.df_curve_config[data_request.currency]
        if data_request.name not in df_config:
            raise RuntimeError(f"{data_request.currency} {data_request.name} is not configured for Citi data source")

        tickers = df_config[data_request.name]

        rates_data = self.datalake.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date).reset_index()
        self.data_dict = DFCurveCitiDataSource.process_Citi_rates(rates_data)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class DFCurveJPDataSource(IDataSource):
    def __init__(self, data_file=None):
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    @staticmethod
    def process_JP_rates(rates_data, curve_name, pattern):
        return read_jp_rates(rates_data, curve_name, pattern_override=pattern)

    def initialize(self, data_request):
        container = DFCurveContainer(data_request.currency, data_request.name)

        df_config = jp_df_curve_config.df_curve_config[data_request.currency]
        if data_request.name not in df_config:
            raise RuntimeError(f"{data_request.currency} {data_request.name} is not configured for JP data source")
        pattern = jp_df_curve_config.df_curve_config_pattern[data_request.currency][data_request.name]

        tickers = df_config[data_request.name]

        tickers = tickers.split(',')
        ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                       np.split(tickers, [i * 100 for i in range(1, len(tickers) // 100 + 1)])]
        data_list = []
        for ticker_str in ticker_strs:
            data_list.append(self.datalake.getData('JPM_DATAQUERY', ticker_str, 'VALUE', data_request.start_date, data_request.end_date))
        result_df = pd.concat(data_list).reset_index()
        rates_data = result_df.pivot(index='tstamp', columns='ticker', values='VALUE')
        missing_tickers = list(set(tickers) - set(rates_data.columns))
        available_tickers = list(set(tickers) - set(missing_tickers))
        rates_data = rates_data[available_tickers]
        for t in missing_tickers:
            rates_data[t] = np.NaN
        rates_data = rates_data.reset_index()

        # data seems to contain some strings => update
        rates_data = rates_data.replace('','0')
        for col in rates_data.columns:
            if col != 'tstamp':
                rates_data[col] = [ float(x) if isinstance(x, str) else x for x in rates_data[col].values ]

        self.data_dict = DFCurveJPDataSource.process_JP_rates(rates_data, data_request.name, pattern)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class DFCurveBBGDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}
        self.currency_country_map = \
            {
                "AUD": "AU",
                "CHF": "CH",
                "CNY": "CN",
                "DKK": "DK",
                "EUR": "EU",
                "GBP": "GB",
                "JPY": "JP",
                "ILS": "IL",
                "MAD": "MA",
                "NOK": "NO",
                "USD": "US",
                "CAD": "CA",
                "HKD": "HK",
                "SEK": "SK",
                "CNH": "CN",
            }

    def initialize(self, data_request):

        dt = data_request.start_date
        live = dt.date() == datetime.today().date()
        if live and data_request.start_date != data_request.end_date:
            raise Exception("Only support single date request for Live")
        # Live per country
        # http://cpiceregistry:6703/all_rate_curves?server_name=data_cache_interest_rates&country=EU&environment=CAPSTONE&source=BBG_ZERO_RATES
        # EOD per country
        # http://cpiceregistry:6703/all_rate_curves.history?server_name=data_cache_interest_rates&country=EU&PIT_SOURCE=CLOSE&environment=CAPSTONE&PIT_DATE=2024-04-23&PIT_LOCATION=NYC&source=BBG_ZERO_RATES
        payload = {
            "server_name": "data_cache_interest_rates",
            "country": self.currency_country_map[data_request.currency],
            "environment": "CAPSTONE",
            "source": data_request.name,
        }
        if live:
            r = requests.get("http://cpiceregistry:6703/all_rate_curves", params=payload)
            rates = json.loads(r.content.decode("utf-8"))[0]
            self.data_dict = {dt: dict(zip(rates["days"], rates["rates"]))}
        else:
            payload.update({
                "PIT_SOURCE": "CLOSE",
                "PIT_LOCATION": "NYC",
            })
            self.data_dict = {}
            while dt <= data_request.end_date:
                payload["PIT_DATE"] = dt.strftime("%Y-%m-%d"),
                r = requests.get("http://cpiceregistry:6703/all_rate_curves.history", params=payload)
                if len(r.content) > 0:
                    rates = json.loads(r.content.decode("utf-8"))[0]
                    self.data_dict[dt] = dict(zip(rates["days"], rates["rates"]))
                else:
                    print(f"No {data_request.currency} {data_request.name} Curve found on {dt.strftime('%Y-%m-%d')}")
                dt = add_business_days(dt, 1)

        container = DFCurveContainer(data_request.currency, data_request.name, rates["dayConvention"])
        container._get_spot_rate_curves = self._get_spot_rate_curves
        return container

    def _get_spot_rate_curves(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict.get(dt, None)


if __name__ == '__main__':
    from ..constants.ccy import Ccy
    start = datetime(2001, 5, 1)
    # start = datetime(2020, 7, 3)
    end = datetime(2001, 5, 18)

    def _print_rates(_rates):
        keys = [x for x in list(_rates.keys())]
        keys = sorted(keys)
        for k in keys:
            print(f"{k}: {_rates[k]}")

    # usd_disc_curve_request = DFCurveRequest(start, end, "USD", 'SWAP')
    # usd_disc_curve_container = JPDFCurveDataSource().initialize(usd_disc_curve_request)
    #
    # _print_rates(usd_disc_curve_container.get_spot_rate_curves(end))

    usd_disc_curve_request = DFCurveRequest(start, end, "GBP", 'SWAP')
    usd_disc_curve_container = DFCurveCitiDataSource().initialize(usd_disc_curve_request)
    _print_rates(usd_disc_curve_container.get_spot_rate_curves(end))

    # usd_disc_curve_request = DFCurveRequest(start, end, "USD", 'SWAP')
    # usd_disc_curve_container = DFCurveJPDataSource().initialize(usd_disc_curve_request)
    #
    # _print_rates(usd_disc_curve_container.get_spot_rate_curves(end))
