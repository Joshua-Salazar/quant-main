import re

import pandas as pd
import numbers
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..infrastructure.shock import ShockType, DatetimeShiftType
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_item import MarketItem
from ..data.market import get_expiry_in_year
from ..dates.utils import add_business_days, datetime64_to_datetime
from datetime import datetime
from datalake.datalakeapi import DataLakeAPI
import json
import requests


class SpotRateContainer(DataContainer):
    def __init__(self, currency: str, name: str):
        self.market_key = market_utils.create_spot_rates_key(currency, name)

    def get_market_key(self):
        return self.market_key

    def get_spot_rate_curves(self, dt=None):
        return self._get_spot_rate_curves(dt)

    def save_rates_to_file(self, file_name):
        data_dict = self.get_spot_rate_curves()
        records = []
        for dt, v in data_dict.items():
            for expiry, rate in v.items():
                records.append({
                    'date': dt,
                    'expiry': expiry,
                    'rate': rate,
                })
        df = pd.DataFrame.from_records(records)
        df.to_csv(file_name)

    def get_market_item(self, dt):
        return SpotRateData(self.get_market_key(), self.get_spot_rate_curves(dt))


# we make distinction between DataContainers and MarketItem as for some case data container has too much data
# and it is less efficient to have it in market whereas MarketItem can contain only one day worth of data
# each data container should have a get_market_item function, the return of which will be contained in the market object
class SpotRateData(MarketItem):
    def __init__(self, market_key, data_dict):
        self.market_key = market_key
        self.data_dict = data_dict

    def get_market_key(self):
        return self.market_key

    def clone(self):
        return SpotRateData(self.market_key, self.data_dict)

    def apply(self, shocks, original_market, **kwargs) -> MarketItem:
        cloned_spot = self.clone()
        for shock in shocks:
            if shock.type == ShockType.RATESHOCK:
                # assume rate in pct in data_dict
                assert shock.method == "level"
                data_dict = {k: v + 0.0001 * shock.size_bps for k, v in self.data_dict.items()}
                cloned_spot.data_dict = data_dict
            elif shock.type == ShockType.DATETIMESHIFT:
                if shock.shift_type in [DatetimeShiftType.STICKY_DATE, DatetimeShiftType.VOLA_STICKY_DATE]:
                    raise Exception(f"{shock.shift_type.value} not supported yet")
        return cloned_spot


class SpotRateRequest(IDataRequest):
    def __init__(self, start_date, end_date, currency, curve_name, tenor=None):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.curve_name = curve_name
        self.tenor = tenor


class SpotRateCitiDataSource(IDataSource):
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
        tenors = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y", "15M", "18M", "21M",
                  "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
                  "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y", "19Y", "20Y", "25Y", "30Y", "35Y", "40Y", "45Y", "50Y"]
        ois_names_map = {"AUD": "AONIA", "CAD": "CORRA", "CHF": "SARON", "DKK": "TNDKK", "GBP": "SONIA", "JPY": "TONAR", "NOK": "NINA", "NZD": "NZIONA", "SEK": "STINA", "SGD": "SORA", "THB": "THOR"}
        if curve_name == "SWAP_LIBOR":
            tickers = [f"RATES.SWAP_LIBOR.{currency}.PAR.{x}" for x in tenors]
            tenor_func = lambda x: re.match(r"RATES.SWAP_LIBOR." + currency + r".PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
        else:
            if currency == "USD":
                if curve_name == "SWAP_SOFR":
                    tickers = [f"RATES.OIS.USD_SOFR.PAR.{x}" for x in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.USD_SOFR.PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
                elif curve_name == "SWAP_FEDFUND":
                    tickers = [f"RATES.OIS.USD_FEDFUND.PAR.{x}" for x in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.USD_FEDFUND.PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            elif currency == "EUR":
                if curve_name == "SWAP_EUROSTR":
                    tickers = [f"RATES.OIS.EUR_EUROSTR.PAR.{x}" for x in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.EUR_EUROSTR.PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
                elif curve_name == "SWAP_EONIA":
                    tickers = [f"RATES.OIS.EUR_EONIA.PAR.{x}" for x in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS.EUR_EONIA.PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            else:
                ois_name = ois_names_map[currency]
                if curve_name == "SWAP_OIS" or curve_name == f"SWAP_{ois_name}":
                    tickers = [f"RATES.OIS.{currency}_{ois_name}.PAR.{x}" for x in tenors]
                    tenor_func = lambda x: re.match(r"RATES.OIS." + f"{currency}_{ois_name}" + r".PAR.(\d{1,2}(?:Y|M|W|D))", x).groups()[0]
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
        return tickers, tenor_func

    @staticmethod
    def process_Citi_rates(rates_data, tenor_func):
        rates_data['tenor'] = [tenor_func(x) for x in rates_data.ticker]
        all_rates = {}
        for dt in rates_data.tstamp.unique():
            data_dt = rates_data[rates_data.tstamp == dt]
            dt = datetime64_to_datetime(dt)
            rates = {}
            for index, row in data_dt.iterrows():
                expiry = get_expiry_in_year(row.tenor[:-1], row.tenor[-1])
                v = float(row.VALUE)
                if isinstance(v, numbers.Number) and v < 1e20:
                    rates.setdefault(expiry, v)
            all_rates[dt] = rates
        return all_rates

    @staticmethod
    def mix_rates(rates1, rates2, mix_start, mix_end):
        all_rates = {}
        for dt, data_dt in rates1.items():
            if dt < mix_start:
                all_rates[dt] = data_dt
            elif dt >= mix_end:
                continue
            else:
                rates = {}
                for exp, r in data_dt.items():
                    r2 = rates2.get(dt, {}).get(exp, None)
                    if r2 is not None:
                        r_m = (dt - mix_start).days / (mix_end - mix_start).days * r2 + (mix_end - dt).days / (mix_end - mix_start).days * r
                        rates.setdefault(exp, r_m)
                all_rates[dt] = rates
        for dt, data_dt in rates2.items():
            if dt >= mix_end:
                all_rates[dt] = data_dt
            else:
                continue
        return all_rates

    def initialize(self, data_request):
        datalake = DataLakeAPI(username=self.credentials["username"], token=self.credentials["token"])
        if isinstance(data_request, IDataRequest):
            container = SpotRateContainer(data_request.currency, data_request.curve_name)
            tickers, tenor_func = SpotRateCitiDataSource.get_config(data_request)
            rates_data = datalake.getData('CITI_VELOCITY', ",".join(tickers), 'VALUE', data_request.start_date, data_request.end_date).reset_index()
            self.data_dict = SpotRateCitiDataSource.process_Citi_rates(rates_data, tenor_func)
        else:
            assert data_request[0][0].currency == data_request[0][1].currency
            container = SpotRateContainer(data_request[0][0].currency, [data_request[0][0].curve_name, data_request[0][1].curve_name])
            tickers1, tenor_func1 = SpotRateCitiDataSource.get_config(data_request[0][0])
            rates_data1 = datalake.getData('CITI_VELOCITY', ",".join(tickers1), 'VALUE', data_request[0][0].start_date, data_request[0][0].end_date).reset_index()
            rates1 = SpotRateCitiDataSource.process_Citi_rates(rates_data1, tenor_func1)
            tickers2, tenor_func2 = SpotRateCitiDataSource.get_config(data_request[0][1])
            rates_data2 = datalake.getData('CITI_VELOCITY', ",".join(tickers2), 'VALUE', data_request[0][1].start_date, data_request[0][1].end_date).reset_index()
            rates2 = SpotRateCitiDataSource.process_Citi_rates(rates_data2, tenor_func2)
            self.data_dict = SpotRateCitiDataSource.mix_rates(rates1, rates2, data_request[1][0], data_request[1][1])
        container._get_spot_rate_curves = self._get_spot_rate_curves
        return container

    def _get_spot_rate_curves(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict.get(dt, None)


class SpotRateJPDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}
        self.datalake = DataLakeAPI('quant-research','YFmDOZhsjibMKDwRfIzmKHAzmhMBOdrrGRBEEsoDSlUSGtulHaUbtiLUfAMquantJkCUqElreSearchBuaXVZKxbPSpsnbXCsAXHYbVmzPmmRdnKvlJcStrNIuFUeERV')

    @staticmethod
    def get_config(data_request):
        currency = data_request.currency
        curve_name = data_request.curve_name
        usd_libswap_tenors = ["1D", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "1Y"]
        usd_parswap_tenors = ["18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
                              "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y", "19Y", "20Y", "25Y", "30Y", "35Y", "40Y", "45Y", "50Y", "60Y"]
        usd_ois_tenors = ["1d", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "10m", "11m", "12m", "13m", "14m", "15m",
                          "16m", "17m", "18m", "19m", "20m", "21m", "22m", "23m", "24m",
                          "27m", "30m", "33m", "36m", "39m", "42m", "45m", "48m", "51m", "54m", "57m", "60m",
                          "6y", "7y", "8y", "9y", "10y", "15y", "20y", "25y", "30y", "40y", "50y", "60y"]
        usd_sofr_tenors = usd_libswap_tenors + usd_parswap_tenors
        if curve_name == "SWAP_LIBOR":
            if currency == "USD":
                tickers = [f"FDER_PARSWAP_{x}_RT_MID" for x in usd_parswap_tenors] + [f"FDER_LIBSWAP_{x}_RT_MID" for x in usd_libswap_tenors]
                tenor_func = lambda x: re.match(r"FDER_(?:PAR|LIB)SWAP_(\d{1,2}(?:Y|M|W|D))_RT_MID", x).groups()[0]
            else:
                raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
        else:
            if currency == "USD":
                if curve_name == "SWAP_SOFR":
                    tickers = [f"MTE_usd/sofr/daily/{x}/rate" for x in usd_sofr_tenors]
                    tenor_func = lambda x: re.match(r"MTE_usd/sofr/daily/(\d{1,2}(?:Y|M|W|D))/rate", x).groups()[0]
                elif curve_name == "SWAP_FEDFUND":
                    tickers = [f"MTE_usd/ois/fomc/{x}/rate" for x in usd_ois_tenors]
                    tenor_func = lambda x: re.match(r"MTE_usd/ois/fomc/(\d{1,2}(?:y|m|w|d))/rate", x).groups()[0]
                else:
                    raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")
            else:
                raise RuntimeError(f"Un-scrubbed spot rate specification {data_request.currency} {data_request.curve_name}")

        return tickers, tenor_func

    @staticmethod
    def process_JP_rates(rates_data, tenor_func):
        rates_data['tenor'] = [tenor_func(x) for x in rates_data.ticker]
        all_rates = {}
        for dt in rates_data.tstamp.unique():
            data_dt = rates_data[rates_data.tstamp == dt]
            dt = datetime64_to_datetime(dt)
            rates = {}
            for index, row in data_dt.iterrows():
                expiry = get_expiry_in_year(row.tenor[:-1], row.tenor[-1])
                if row.VALUE != '':
                    v = float(row.VALUE)
                    if isinstance(v, numbers.Number) and v < 1e20:
                        rates.setdefault(expiry, v)
            all_rates[dt] = rates
        return all_rates

    def initialize(self, data_request):
        container = SpotRateContainer(data_request.currency, data_request.curve_name)

        tickers, tenor_func = SpotRateJPDataSource.get_config(data_request)

        rates_data = self.datalake.getData('JPM_DATAQUERY', ",".join(tickers), 'VALUE', data_request.start_date, data_request.end_date).reset_index()
        self.data_dict = SpotRateJPDataSource.process_JP_rates(rates_data, tenor_func)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class SpotRateInternalDataSource(IDataSource):
    """
    We construct curves internally based on the mkt data quotes for each instrument used in construction of a curve.
    The zero rates for each curve are exposed via data_cache where we store intraday and EOD snaps as well as live data.
    The supported curves are available from:
    http://cpiceregistry:6703/curvenames?server_name=data_cache_zerorates&environment=CAPSTONE
    """
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        dt = data_request.start_date
        live = dt.date() == datetime.today().date()
        if live and data_request.start_date != data_request.end_date:
            raise Exception("Only support single date request for Live")
        # Live
        # http://cpiceregistry:6703/zerorates?server_name=data_cache_zerorates&environment=CAPSTONE&curvename=FEDL01
        # EOD
        # http://cpiceregistry:6703/zerorates.history?server_name=data_cache_zerorates&PIT_SOURCE=CLOSE&environment=CAPSTONE&PIT_DATE=2024-04-23&PIT_LOCATION=NYC&curvename=FEDL01
        payload = {
            "server_name": "data_cache_zerorates",
            "curvename": data_request.curve_name,
            "environment": "CAPSTONE",
        }
        if live:
            r = requests.get("http://cpiceregistry:6703/zerorates", params=payload)
            rates = json.loads(r.content.decode("utf-8"))[0]
            rates["days"] = [(datetime.strptime(str(et), "%Y%m%d") - datetime.strptime(str(st), "%Y%m%d")).days for st, et in rates["tenor"]]
            self.data_dict = {dt: dict(zip(rates["days"], rates["rates"]))}
        else:
            payload.update({
                "PIT_SOURCE": "CLOSE",
                "PIT_LOCATION": "NYC",
            })
            self.data_dict = {}
            while dt <= data_request.end_date:
                payload["PIT_DATE"] = dt.strftime("%Y-%m-%d"),
                r = requests.get("http://cpiceregistry:6703/zerorates.history", params=payload)
                if len(r.content) > 0:
                    rates = json.loads(r.content.decode("utf-8"))[0]
                    rates["days"] = [(datetime.strptime(str(et), "%Y%m%d") - datetime.strptime(str(st), "%Y%m%d")).days for st, et in rates["tenor"]]
                    self.data_dict[dt] = dict(zip(rates["days"], rates["rates"]))
                else:
                    print(f"No {data_request.currency} {data_request.curve_name} Curve found on {dt.strftime('%Y-%m-%d')} "
                          f"from \n {r.url}")
                dt = add_business_days(dt, 1)

        container = SpotRateContainer(data_request.currency, data_request.curve_name)
        container._get_spot_rate_curves = self._get_spot_rate_curves
        return container

    def _get_spot_rate_curves(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict.get(dt, None)


class SpotRateFlatFileDataSource(IDataSource):
    def __init__(self, spot_rates_file=None):
        self.spot_rates_file = spot_rates_file
        self.data_dict = {}

    def initialize(self, data_request):
        container = SpotRateContainer(data_request.currency, data_request.curve_name)

        spot_rates_df = pd.read_csv(self.spot_rates_file)

        all_rates = {}
        for record in spot_rates_df.to_dict('records'):
            dt = record['date']
            dt = datetime64_to_datetime(dt)
            expiry = record["expiry"]
            v = float(record["rate"])
            if isinstance(v, numbers.Number) and v < 1e20:
                all_rates.setdefault(dt, {}).setdefault(expiry, v)

        self.data_dict = all_rates

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


class SpotRateDictDataSource(IDataSource):
    def __init__(self, data_dict=None):
        self.data_dict = data_dict

    def initialize(self, data_request):
        container = SpotRateContainer(data_request.currency, data_request.curve_name)

        def _get_spot_rate_curves(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        container._get_spot_rate_curves = _get_spot_rate_curves
        return container


if __name__ == '__main__':
    start = datetime(2023, 5, 1)
    end = datetime(2023, 6, 20)

    rate_request = SpotRateRequest(start, end, "USD", "SWAP_LIBOR")
    rate_container = SpotRateCitiDataSource().initialize(rate_request)

    print(rate_container.get_spot_rate_curves(end))

    rate_container = SpotRateFlatFileDataSource('/misc/Traders/Solutions/backtests/data_cache/rates/usd_spot_rates_manual.csv').initialize(rate_request)

    print(rate_container.get_spot_rate_curves(end))
