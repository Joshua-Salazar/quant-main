import numbers
import re

import pandas as pd
import numpy as np
from datetime import datetime

from ..dates.utils import datetime64_to_datetime
from ..infrastructure.data_container import DataContainer
from ..data.market import get_expiry_in_year
from ..data.market import read_jp_swaption_atmf_yields
from ..infrastructure import market_utils
from ..infrastructure.spot_rate_data_container import SpotRateCitiDataSource
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from datalake.datalakeapi import DataLakeAPI
import os


class ForwardRateContainer(DataContainer):
    def __init__(self, currency: str, curve_name: str = None):
        self.market_key = market_utils.create_forward_rates_key(currency, curve_name)

    def get_market_key(self):
        return self.market_key

    def get_forward_rate_curves(self, dt=None):
        return self._get_forward_rate_curves(dt)

    def get_market_item(self, dt):
        return self.get_forward_rate_curves(dt)

    def save_rates_to_file(self, file_name):
        data_dict = self.get_forward_rate_curves()
        records = []
        for dt, v in data_dict.items():
            for expiry, v1 in v.items():
                for tenor, rate in v1.items():
                    records.append({
                        'date': dt,
                        'expiry': expiry,
                        'tenor': tenor,
                        'rate': rate,
                    })
        df = pd.DataFrame.from_records(records)
        df.to_csv(file_name)


class ForwardRateRequest(IDataRequest):
    def __init__(self, start_date, end_date, currency, curve_name=None, tenor=None):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.curve_name = None if curve_name is None else curve_name.upper()
        self.tenor = None if tenor is None else tenor.upper()

    def get_citi_tickers(self):
        start_tenors = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "15M", "18M", "21M",
                        "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "20Y",
                        "25Y", "30Y", "40Y", "50Y"]
        tickers = ""
        prefix = f"RATES.{self.curve_name}.{self.currency}.FWD"
        for start_tenor in start_tenors:
            tickers += f"{prefix}.{start_tenor}.{self.tenor},"
        return tickers

class ForwardRateDataSourceDict(IDataSource):
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
        container = ForwardRateContainer(data_request.currency)

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.cache['data']
            else:
                return self.cache['data'][dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


class JPForwardRateDataSource(IDataSource):
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    def update_data_file(self, file_path, tgt_end_date):
        data_df = pd.read_csv(file_path, index_col='tstamp')
        file_end_date = pd.Timestamp(data_df.index[-1])
        if file_end_date >= tgt_end_date:
            return

        tickers = data_df.columns
        ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                       np.split(tickers, [i * 100 for i in range(1, len(tickers) // 100 + 1)])]
        data_list = []
        for ticker_str in ticker_strs:
            data_list.append(self.datalake.getData('JPM_DATAQUERY', ticker_str, "VALUE", start=pd.Timestamp(data_df.index[-1]), end=tgt_end_date))
        result_df = pd.concat(data_list).reset_index()

        ## add SOFR data post 10Jun24 (JPM discontinued LIBOR Swaption data)
        sofr_tickers = [ x.replace('_SWAPTION', '_SOFRSWAPTION') for x in tickers ]
        ticker_strs = [','.join(ticker_chunk) for ticker_chunk in
                       np.split(sofr_tickers, [i * 100 for i in range(1, len(sofr_tickers) // 100 + 1)])]
        data_list = []
        for ticker_str in ticker_strs:
            data_list.append(self.datalake.getData('JPM_DATAQUERY', ticker_str, "VALUE", start=pd.Timestamp(data_df.index[-1]), end=tgt_end_date))
        sofr_result_df = pd.concat(data_list).reset_index()
        sofr_result_df.ticker = [x.replace('SOFR', '') for x in sofr_result_df.ticker.values]
        if not result_df.empty:
            sofr_result_df = sofr_result_df[[x != result_df.tstamp.max() for x in sofr_result_df.tstamp.values]]
        result_df = pd.concat([result_df,sofr_result_df])

        new_data = result_df.pivot(index='tstamp', columns='ticker', values='VALUE')
        missing_tickers = list(set(tickers) - set(new_data.columns))
        available_tickers = list(set(tickers) - set(missing_tickers))
        new_data = new_data[available_tickers]
        for t in missing_tickers:
            new_data[t] = np.NaN
        new_data.drop(data_df.index[-1], inplace=True)
        full_df = pd.concat([data_df, new_data])
        full_df.to_csv(file_path)

    def initialize(self, data_request):
        container = ForwardRateContainer(data_request.currency)

        assert data_request.currency == 'USD'

        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        data_file_default = f'{data_file_folder}Solutions/backtests/data_cache/rates/{data_request.currency}_rates_jp_data_cache.csv'
        self.data_file = data_file_default if (self.data_file is None) else self.data_file
        self.update_data_file(self.data_file, pd.Timestamp(data_request.end_date))
        self.data_dict = read_jp_swaption_atmf_yields(pd.read_csv(self.data_file))

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


class ForwardRateJPDataSource(IDataSource):
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    @staticmethod
    def get_config(data_request):
        currency = data_request.currency
        curve_name = data_request.curve_name
        usd_liborswap_start_tenors = ["1D", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y"]
        usd_liborswap_tenors = ["1D", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y"]
        usd_sofrswap_start_tenors = usd_liborswap_start_tenors
        usd_sofrswap_tenors = usd_liborswap_tenors
        if data_request.tenor is not None:
            usd_sofrswap_tenors = usd_liborswap_tenors = [data_request.tenor]
        if curve_name == "SWAP_LIBOR":
            if currency == "USD":
                tickers = [f"FCRV_SWAP_PAR_{x}_{y}_RATE" for x in usd_liborswap_tenors for y in usd_liborswap_start_tenors]
                tenor_func = lambda x: re.match(r"FCRV_SWAP_PAR_(\d{1,2}(?:Y|M|W|D))_(\d{1,2}(?:Y|M|W|D))_RATE", x).groups()[::-1]
            else:
                raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
        else:
            if currency == "USD":
                if curve_name == "SWAP_SOFR":
                    tickers = [f"FCRV_SOFR_SWAP_PAR_{x}_{y}_RATE" for x in usd_sofrswap_tenors for y in usd_sofrswap_start_tenors]
                    tenor_func = lambda x: re.match(r"FCRV_SOFR_SWAP_PAR_(\d{1,2}(?:Y|M|W|D))_(\d{1,2}(?:Y|M|W|D))_RATE", x).groups()[::-1]
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            else:
                raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")

        return tickers, tenor_func

    @staticmethod
    def process_JP_rates(rates_data, tenor_func, fixed_tenor=None):
        rates_data['start_tenor'] = [tenor_func(x)[0] for x in rates_data.ticker]
        rates_data['tenor'] = [tenor_func(x)[1] for x in rates_data.ticker]
        all_rates = {}
        for dt in rates_data.tstamp.unique():
            data_dt = rates_data[rates_data.tstamp == dt]
            dt = datetime64_to_datetime(dt)
            rates = {}
            for index, row in data_dt.iterrows():
                expiry = get_expiry_in_year(row.start_tenor[:-1], row.start_tenor[-1])
                tenor = row.tenor
                if fixed_tenor is not None:
                    assert tenor == fixed_tenor
                v = float(row.VALUE)
                if isinstance(v, numbers.Number) and v < 1e20:
                    if fixed_tenor is not None:
                        rates.setdefault(expiry, v)
                    else:
                        rates.setdefault(expiry, {})
                        rates[expiry].setdefault(tenor, v)
            all_rates[dt] = rates
        return all_rates

    def initialize(self, data_request):
        container = ForwardRateContainer(data_request.currency, data_request.curve_name)
        tickers, tenor_func = ForwardRateJPDataSource.get_config(data_request)
        rates_data = self.datalake.getData('JPM_DATAQUERY', ",".join(tickers), 'VALUE', data_request.start_date, data_request.end_date).reset_index()
        self.data_dict = ForwardRateJPDataSource.process_JP_rates(rates_data, tenor_func, data_request.tenor)

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


class ForwardRateCitiDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}
        self.credentials = {
            "username": "quant-research",
            "token":"YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV"
        }


    @staticmethod
    def get_config(data_request):
        currency = data_request.currency
        curve_name = data_request.curve_name
        start_tenors = ["1D", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "15M", "18M", "21M",
                        "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
                        "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y", "19Y", "20Y", "25Y", "30Y", "35Y", "40Y", "45Y", "50Y"]
        tenors = ["1D", "1W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "15M", "18M", "21M",
                  "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
                  "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y", "19Y", "20Y", "25Y", "30Y", "35Y", "40Y", "45Y", "50Y"]
        if data_request.tenor is not None:
            tenors = [data_request.tenor]

        ois_names_map = {"AUD": "AONIA", "CAD": "CORRA", "CHF": "SARON", "DKK": "TNDKK", "GBP": "SONIA", "JPY": "TONAR", "NOK": "NINA", "NZD": "NZIONA", "SEK": "STINA", "SGD": "SORA", "THB": "THOR"}
        if curve_name == "SWAP_LIBOR":
            tickers = [f"RATES.SWAP_LIBOR.{currency}.FWD.{x}.{y}" for x in start_tenors for y in tenors]
            tenor_func = lambda x: re.match(r"RATES.SWAP_LIBOR." + currency + r".FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
        else:
            if currency == "USD":
                if curve_name == "SWAP_SOFR":
                    tickers = [f"RATES.OIS.USD_SOFR.FWD.{x}.{y}" for x in start_tenors for y in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.USD_SOFR.FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
                elif curve_name == "SWAP_FEDFUND":
                    tickers = [f"RATES.OIS.USD_FEDFUND.FWD.{x}.{y}" for x in start_tenors for y in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.USD_FEDFUND.FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            elif currency == "EUR":
                if curve_name == "SWAP_EUROSTR":
                    tickers = [f"RATES.OIS.EUR_EUROSTR.FWD.{x}.{y}" for x in start_tenors for y in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.EUR_EUROSTR.FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
                elif curve_name == "SWAP_EONIA":
                    tickers = [f"RATES.OIS.EUR_EONIA.FWD.{x}.{y}" for x in start_tenors for y in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.EUR_EONIA.FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            else:
                ois_name = ois_names_map[currency]
                if curve_name == "SWAP_OIS" or curve_name == f"SWAP_{ois_name}":
                    tickers = [f"RATES.OIS.{currency}_{ois_name}.FWD.{x}.{y}" for x in start_tenors for y in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS." + f"{currency}_{ois_name}" + r".FWD.(\d{1,2}(?:Y|M|W|D)).(\d{1,2}(?:Y|M|W|D))", x).groups()
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
        return tickers, tenor_func

    @staticmethod
    def process_Citi_rates(rates_data, tenor_func, fixed_tenor=None):
        rates_data['start_tenor'] = [tenor_func(x)[0] for x in rates_data.ticker]
        rates_data['tenor'] = [tenor_func(x)[1] for x in rates_data.ticker]

        all_rates = {}
        for record in rates_data.to_dict('records'):
            dt = record['tstamp']
            expiry = get_expiry_in_year(record["start_tenor"][:-1], record["start_tenor"][-1])
            tenor = record["tenor"]
            if fixed_tenor is not None:
                assert tenor == fixed_tenor
            v = float(record["VALUE"])
            if isinstance(v, numbers.Number) and v < 1e20:
                all_rates.setdefault(dt, {}).setdefault(expiry, {}).setdefault(tenor, v)

        return all_rates

    def initialize(self, data_request):
        datalake = DataLakeAPI(username=self.credentials["username"], token=self.credentials["token"])
        if isinstance(data_request, IDataRequest):
            container = ForwardRateContainer(data_request.currency, data_request.curve_name)
            tickers, tenor_func = ForwardRateCitiDataSource.get_config(data_request)
            rates_data = datalake.getData("CITI_VELOCITY", ",".join(tickers), 'VALUE', data_request.start_date, data_request.end_date).reset_index()
            self.data_dict = ForwardRateCitiDataSource.process_Citi_rates(rates_data, tenor_func, data_request.tenor)
        else:
            [data_request1, data_request2] = data_request[0]
            [mix_start, mix_end] = data_request[1]
            assert data_request1.currency == data_request2.currency
            # use curve name from second request
            container = ForwardRateContainer(data_request2.currency, data_request2.curve_name)

            tickers1, tenor_func1 = ForwardRateCitiDataSource.get_config(data_request1)
            rates_data1 = datalake.getData('CITI_VELOCITY', ",".join(tickers1), 'VALUE', data_request1.start_date, min(mix_end, data_request1.end_date)).reset_index()
            rates1 = ForwardRateCitiDataSource.process_Citi_rates(rates_data1, tenor_func1, data_request1.tenor)

            tickers2, tenor_func2 = ForwardRateCitiDataSource.get_config(data_request2)
            rates_data2 = datalake.getData('CITI_VELOCITY', ",".join(tickers2), 'VALUE', max(data_request2.start_date, mix_start), data_request2.end_date).reset_index()
            rates2 = ForwardRateCitiDataSource.process_Citi_rates(rates_data2, tenor_func2, data_request2.tenor)
            self.data_dict = SpotRateCitiDataSource.mix_rates(rates1, rates2, mix_start, mix_end)

        container._get_forward_rate_curves = self._get_forward_rate_curves
        return container

    def _get_forward_rate_curves(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict[dt]


class ForwardRateFlatFileDataSource(IDataSource):
    def __init__(self, forward_rates_file=None):
        self.forward_rates_file = forward_rates_file
        self.data_dict = {}

    def initialize(self, data_request):
        container = ForwardRateContainer(data_request.currency, data_request.curve_name)

        forward_rates_df = pd.read_csv(self.forward_rates_file)

        all_rates = {}
        for record in forward_rates_df.to_dict('records'):
            dt = record['date']
            dt = datetime64_to_datetime(dt)
            expiry = record["expiry"]
            tenor = record["tenor"]
            v = float(record["rate"])
            if isinstance(v, numbers.Number) and v < 1e20:
                all_rates.setdefault(dt, {}).setdefault(expiry, {}).setdefault(tenor, v)

        self.data_dict = all_rates

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


class ForwardRateDictDataSource(IDataSource):
    def __init__(self, data_dict=None):
        self.data_dict = data_dict

    def initialize(self, data_request):
        container = ForwardRateContainer(data_request.currency, data_request.curve_name)

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


class ForwardRateFlatFileDataSourceOld(IDataSource):
    def __init__(self, forward_rates_file=None):
        self.forward_rates_file = forward_rates_file
        self.data_dict = {}

    def initialize(self, data_request):
        container = ForwardRateContainer(data_request.currency, data_request.curve_name)

        forward_rates_df = pd.read_csv(self.forward_rates_file)

        forward_rates_curves = {}
        for record in forward_rates_df.to_dict('records'):
            dt = datetime.strptime(str(record['date']), "%Y-%m-%d")
            curve = {}
            for k, v in record.items():
                if k != 'date':
                    if isinstance(v, numbers.Number):
                        keys = k.split(' ')
                        tenor = keys[1]
                        expiry = float(keys[0])
                        curve.setdefault(tenor, {}).setdefault(expiry, v)
            forward_rates_curves[dt] = curve

        self.data_dict = forward_rates_curves

        def _get_forward_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_forward_rate_curves = _get_forward_rate_curves
        return container


if __name__ == '__main__':
    from ..constants.ccy import Ccy
    start = datetime(2023, 5, 1)
    # start = datetime(2020, 7, 3)
    end = datetime(2023, 6, 20)

    rate_request = ForwardRateRequest(start, end, "USD")
    rate_container = JPForwardRateDataSource().initialize(rate_request)

    print(rate_container.get_forward_rate_curves(end))

    rate_request = ForwardRateRequest(start, end, "GBP")
    rate_container = ForwardRateFlatFileDataSource('/misc/Traders/Solutions/backtests/data_cache/rates/gbp_forward_rates_manual.csv').initialize(rate_request)

    print(rate_container.get_forward_rate_curves(end))
