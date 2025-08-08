from datetime import datetime, time, timedelta
import json
import numpy as np
import pandas as pd
from ..dates.holidays import get_holidays
from ..dates.utils import add_business_days
from ..infrastructure import market_utils
from ..infrastructure.cr_spot import CRSpot
from ..infrastructure.data_container import DataContainer
from ..interface.market_items.ispot import ISpot
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
import requests


class CRSpotDataContainer(DataContainer, ISpot):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_spot_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_cr_spots(self, dt):
        return self._get_cr_spots(dt)

    def get_spot(self, base_date: datetime) -> float:
        """
        :param base_date: current date for spot value
        :return: scalar spot value
        """
        data = self.get_cr_spots(base_date)
        if data is None:
            return None
        else:
            return data[self.ticker]

    def get_market_item(self, dt):
        spot = self.get_spot(dt)
        return None if spot is None else CRSpot(self.ticker, spot)

    def get_data_dts(self):
        return self._get_data_dts()


class CRSpotDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, underlier,
                 inc_prev_day=False):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.underlier = underlier
        self.inc_prev_day = inc_prev_day


class CRSpotsDataSource(IDataSource):
    def __init__(self):
        self.pit_location = None
        self.calendar = None
        self.cdxid_series_version_tenor_map = None
        self.cdx_index_name = None
        self.cdx_index_ticker = None
        self.start_date = None
        self.end_date = None
        self.data_dict = dict()
        self.payload = {
            "server_name": "data_cache_creditvol",
            "environment": "CAPSTONE",
        }
        # can move into a config file
        self.MAP_GENERIC_CDS_INDEX = {
            'CDX North America High Yield Index': 12206218,
            'CDX North America Investment Grade Index': 12206219,
            'CDX Emerging Markets Index': 12206217,
            'iTraxx Europe Subordinated Financial Index': 12206222,
            'iTraxx Europe Crossover Index': 12206223,
            'iTraxx Europe Index': 12206220,
            'iTraxx Europe Senior Financial Index': 12206221,
            'iTraxx Asia ex-Japan Investment Grade Index': 38770679,
            'iTraxx Japan Index': 12206224,
            'iTraxx Australia Index': 12206225
        }
        # can move into a config file
        self.MAP_GENERIC_CDS_INDEX_SYMBOL_NAME = {
            'CDX_NA_HY': 'CDX North America High Yield Index',
            'CDX_NA_IG': 'CDX North America Investment Grade Index',
            'ITRAXX_XOVER': 'iTraxx Europe Crossover Index',
            'ITRAXX_EUROPE': 'iTraxx Europe Index',
        }

    def get_holiday_days(self):
        holiday_days = []
        for cal in self.calendar:
            if isinstance(cal, str):
                holiday_days = holiday_days + get_holidays(cal, self.start_date, self.end_date)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')
        return holiday_days

    def fetch_spots_from_shared_drive(self, dt):
        cdxids = list(self.cdxid_series_version_tenor_map.keys())
        spot_data = pd.read_csv("/mnt/tradersny/CR_QuoteVision_Vol_Surfaces/CDX_240429_250711.csv")
        spot_data_filt = spot_data[(spot_data['TRADE_DATE'] == dt.strftime('%Y-%m-%d')) &
                                   (spot_data['INST_ID'].isin(cdxids)) &
                                   (spot_data['LOCATION'] == self.pit_location)].sort_values('TRADE_DATE')
        cds_dict_list = []
        inst_ids = spot_data_filt['INST_ID'].to_list()

        for inst_id in inst_ids:
            series, version, tenor = self.cdxid_series_version_tenor_map[inst_id].split('_')
            quote_dict = {'CAPSTONE': spot_data_filt.loc[spot_data_filt['INST_ID'] == inst_id, 'VALUE'].values[0]}
            cds_dict_list.append({'series': series,
                                  'version': version,
                                  'tenor': tenor,
                                  'quotes': quote_dict})
        return cds_dict_list

    # time this, is essentially static or at least can be set to be automated daily and then fetched if already updated
    def get_cdxid_series_version_tenor_map(self):
        cdxid_series_version_tenor_map = dict()
        for cds_index_name in self.MAP_GENERIC_CDS_INDEX.keys():
            cdx_id = self.MAP_GENERIC_CDS_INDEX[cds_index_name]
            url = f'http://ntprctp01:6684/getCDSIndices?id={cdx_id}'
            results = requests.get(url)
            if cds_index_name in results.json():
                quotes = results.json()[cds_index_name]
                for quote in quotes:
                    key = f"{quote['series']}_{quote['version']}_{quote['tenor']}"
                    cdxid_series_version_tenor_map.setdefault(cds_index_name, {})[quote['instId']] = key
        return cdxid_series_version_tenor_map

    def fetch_surfaces_from_data_cache_api(self, live=False):
        surfaces = {}
        for cdxid, s_v_t in self.cdxid_series_version_tenor_map.items():
            self.payload["cdxid"] = cdxid
            url = "http://cpiceregistry:6703/creditvol" if live else "http://cpiceregistry:6703/creditvol.history"
            r = requests.get(url, params=self.payload)
            r_json = json.loads(r.content)[0] if len(r.content) > 0 else {}
            if 'cdxid' in r_json:
                surf_dict = {k: v for k, v in r_json.items() if isinstance(v, list) or isinstance(v, int)}
                surfaces[s_v_t] = surf_dict
        return surfaces

    def set_dt_spots(self, dt, cds_dict_list):
        for cds_dict in cds_dict_list:
            # retrieve CAPSTONE aggregated quote, else take the average of the rest of the available quotes
            cds_key = f"{cds_dict['series']}_{cds_dict['version']}_{cds_dict['tenor']}"
            quote = cds_dict['quotes'].get("CAPSTONE",
                                           np.mean(list(cds_dict['quotes'].values())))
            if np.isnan(quote):
                continue
            # cds_market_key = f"{self.cdx_index_ticker}.{cds_key}"
            self.data_dict.setdefault(dt, {}).setdefault(self.cdx_index_ticker, {})[cds_key] = quote
        pass

    def fetch_cds_indices_from_api(self):
        url = "http://ntprctp01:6684/getCDSIndices"
        r = requests.get(url, params={'id': self.MAP_GENERIC_CDS_INDEX[self.cdx_index_name]})
        cds_dict_list = json.loads(r.content)[self.cdx_index_name] if len(r.content) > 0 else {}
        return cds_dict_list

    def initialize(self, data_request):

        dt = self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        live = dt.date() == datetime.today().date()
        if live and self.start_date != self.end_date:
            raise Exception("Only support single date request for Live")

        self.cdx_index_ticker = data_request.underlier
        self.cdx_index_name = self.MAP_GENERIC_CDS_INDEX_SYMBOL_NAME[self.cdx_index_ticker]
        # this is fetched live, so hopefully it contains entire history of the index names various cdx ids (i.e. s_v_t)
        self.cdxid_series_version_tenor_map = self.get_cdxid_series_version_tenor_map()[self.cdx_index_name]
        # cdxid -> all the specific series version tenor key inst ids that we need to associate to vols
        self.calendar, self.pit_location = data_request.calendar, data_request.calendar[0]

        if live:
            cds_dict_list = self.fetch_cds_indices_from_api()
            self.set_dt_spots(dt, cds_dict_list)
        else:
            holiday_days = self.get_holiday_days()
            while dt <= min(self.end_date, datetime.today() + timedelta(days=-1)):
                # not live, so we check the csv (we cannot fill in the recent gaps past July 11th)
                cds_dict_list = self.fetch_spots_from_shared_drive(dt)
                self.set_dt_spots(dt, cds_dict_list)
                dt = add_business_days(dt, 1, holiday_days)

        def _get_cr_spots(_dt):
            return self.data_dict if _dt is None else self.data_dict.get(_dt, None)

        def _get_data_dts():
            return list(self.data_dict.keys())

        container = CRSpotDataContainer(data_request.underlier)
        container._get_cr_spots = _get_cr_spots
        container._get_data_dts = _get_data_dts

        return container
