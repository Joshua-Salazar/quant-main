import json
import numpy as np
import os
import pandas as pd
import pickle
import requests
from ..analytics.cr_vol_surface import CRVolSurfaceFromQuotedVols
from ctp.models.vol import ExpiryStrikeVolPoint, CubicSplineVolatility
from ctp.specifications.daycount import Actual360
from ctp.specifications.defs import RecoveryRate, Strike, Volatility, BasePrice, AnnualRate, Notional
from ctp.utils.time import datetime_to_timepoint, Date, DayDuration, FrequencyAnnual, \
    BusinessDayConvention, TimePoint, MicroDuration, GregorianDate, HolidayCalendarCpp
from ..data.datalake import DATALAKE
from ..dates.holidays import get_holidays
from ..dates.utils import get_business_days, add_business_days, datetime64_to_datetime
from datetime import datetime, time, timedelta
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource


def filter_keys(keys, filter_func):
    rule = filter_func(keys)
    return list(filter(rule, keys))[0]


class CRVolDataContainer(DataContainer):
    def __init__(self, und: str):
        self.market_key = market_utils.create_cr_vol_surface_key(und)

    def get_market_key(self):
        return self.market_key

    def get_cr_surface(self, dt=None):
        return self._get_cr_surface(dt)

    def get_market_item(self, dt):
        return self.get_cr_surface(dt)

    def get_option_data(self, dt, option, dummy_arg):
        return self._get_option_data(dt, option)

    def get_trading_days(self):
        return self._get_trading_days()


class CRVolDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, underlier, cds_tenor=None):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.underlier = underlier
        self.cds_tenor = cds_tenor if cds_tenor is not None else '5'


UND_TO_SPOT_TICKER = {
    'CDX_NA_IG': 'CREDIT.CDX.CDS_IDX_GRP_IG.CDS_INDEX_NAIG.OTR.5Y.CITI_SPREAD',
    'CDX_NA_HY': 'CREDIT.CDX.CDS_IDX_GRP_HY.CDS_INDEX_NAHY.OTR.5Y.CITI_SPREAD',
    'ITRAXX_EUROPE': 'CREDIT.ITRAXX.CDS_IDX_GRP_EUROPE.CDS_INDEX_EUROPE.OTR.5Y.CITI_SPREAD',
    'ITRAXX_XOVER': 'CREDIT.ITRAXX.CDS_IDX_GRP_EUROPE.CDS_INDEX_XOVER.OTR.5Y.CITI_SPREAD',
}


class DatalakeCitiCRVolDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        fixed_tenors = ['1M', '2M', '3M', '6M']
        put_strikes = ['05'] + [str(n) for n in range(10, 100, 5)]

        # load vol, strike, fwd data
        vol_data_dfs = []
        for tenor in fixed_tenors:
            tickers = ''
            tickers += 'CREDIT.CVOL.%s.%s.CVOL_ATM.CVOL_FWD.CVOL_ROLLING_MAT,' % (data_request.underlier, tenor)
            for stk in put_strikes:
                tickers += 'CREDIT.CVOL.%s.%s.CVOL_DELTA_%s.CVOL_VOL.CVOL_ROLLING_MAT,' % (
                data_request.underlier, tenor, stk)
                tickers += 'CREDIT.CVOL.%s.%s.CVOL_DELTA_%s.CVOL_STRIKE.CVOL_ROLLING_MAT,' % (
                data_request.underlier, tenor, stk)

            vol_data_dfs.append(
                DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date,
                                 None).rename(columns={'tstamp': 'date'}))

        vol_data = pd.concat(vol_data_dfs, ignore_index=True)
        vol_data = vol_data.rename(columns={'tstamp': 'date'})
        vol_data['date'] = vol_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())
        vol_data['tenor'] = [x.split('.')[3] for x in vol_data.ticker]
        vol_data['delta'] = [x.split('.')[4][-2:].replace('TM', '50') for x in vol_data.ticker]
        # vols = vol_data[['_VOL.' in x for x in vol_data.ticker]]
        # Ks = vol_data[['_VOL.' not in x for x in vol_data.ticker]]
        # Fwds = vol_data[['FWD' in x for x in vol_data.ticker]]
        vols = vol_data[['.CVOL_VOL.' in x for x in vol_data.ticker]]
        Ks = vol_data[['.CVOL_STRIKE.' in x for x in vol_data.ticker]]
        Fwds = vol_data[['.CVOL_FWD.' in x for x in vol_data.ticker]]

        if data_request.underlier in UND_TO_SPOT_TICKER:
            spot_str = UND_TO_SPOT_TICKER[data_request.underlier]
        else:
            raise RuntimeError('Spot data missing for %s' % data_request.underlier)
        spot_data = DATALAKE.getData('CITI_VELOCITY', spot_str, 'VALUE', data_request.start_date, data_request.end_date,
                                     None).rename(columns={'tstamp': 'date'})
        spot_data['date'] = spot_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())

        pd.options.mode.chained_assignment = None

        holiday_days = []
        for cal in data_request.calendar:
            if isinstance(cal, str):
                holiday_days = holiday_days + get_holidays(cal, data_request.start_date, data_request.end_date)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')

        for dt in get_business_days(data_request.start_date, data_request.end_date, holidays=holiday_days):
            forwards = {}
            volatilities = {}
            abs_strike = {}
            missing_vol_tenors = []
            missing_fwd_tenors = []
            for tenor in fixed_tenors:
                data = vols[(vols['date'] == dt.isoformat()) & (vols['tenor'] == tenor)]
                data_k = Ks[(Ks['date'] == dt.isoformat()) & (Ks['tenor'] == tenor)]
                data['abs_strike'] = [data_k[data_k.delta == x].copy().VALUE.values[0] for x in data.delta]
                data['delta'] = [float(x) / 100 for x in data.delta.values]
                if data.empty:
                    missing_vol_tenors.append(tenor)
                    missing_fwd_tenors.append(tenor)
                else:
                    abs_strike[21 * int(tenor[0]) / 252] = dict(zip(data.delta.values, data.abs_strike.values))
                    vol = data['VALUE'].values / 100.0
                    volatilities[21 * int(tenor[0]) / 252] = dict(zip(data.delta.values, vol))
                    data = Fwds[(Fwds['date'] == dt.isoformat()) & (Fwds['tenor'] == tenor)]
                    if data.empty:
                        missing_fwd_tenors.append(tenor)
                    else:
                        forward = data['VALUE'].values[0]
                        forwards[21 * int(tenor[0]) / 252] = forward
            if len(missing_vol_tenors) or len(missing_fwd_tenors):
                print(f"{dt}: " + (
                    f"Missing vols on tenors {missing_vol_tenors}. " if len(missing_vol_tenors) else "") + (
                          f"Missing fwds on tenors {missing_fwd_tenors}. " if len(missing_fwd_tenors) else ""))

            spots = spot_data[spot_data['date'] == dt.isoformat()]['VALUE']
            if len(spots) == 1:
                spot = spots.values[0]
            else:
                print(f'No CR SPOT data found at {dt}')

            self.data_dict.setdefault(dt, {})[data_request.underlier] = \
                CRVolSurfaceFromQuotedVols(data_request.underlier, dt,
                                           abs_strike,
                                           forwards,
                                           volatilities,
                                           spot,
                                           holiday_days)

        container = CRVolDataContainer(data_request.underlier)

        def _get_cr_surface(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        def _get_trading_days():
            all_dt = list(self.data_dict.keys())
            all_dt.sort()
            return all_dt

        container._get_cr_surface = _get_cr_surface
        container._get_trading_days = _get_trading_days
        return container


class QuotedCRVolDataSource(IDataSource):

    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.calendar = None
        self.pit_location = None
        self.cdx_index_ticker = None
        self.cdx_index_name = None
        self.cdxid_series_version_tenor_map = None
        self.cdxid_map = None
        self.data_dict = {}
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
        self.day_counter_actual_360 = Actual360()
        self.payload = {
            "server_name": "data_cache_creditvol",
            "environment": "CAPSTONE",
        }

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

    def match_cdxid_tenor(self, cdxid, cds_tenor):
        cdxid_series_version_tenor = self.cdxid_map.get(cdxid, None)
        if cdxid_series_version_tenor is None:
            print(f'CDXID {cdxid} not found in cdxid_map.')
            return False
        else:
            return cdxid_series_version_tenor.split('_')[2] == cds_tenor

    @staticmethod
    def max_series_filter(keys):
        max_series = max([key.split('_')[0] for key in keys])
        return lambda key: key.split('_')[0] == max_series

    def fetch_surfaces_from_shared_drive(self, dt, pit_location):
        surfaces_cache_path = f'/mnt/tradersny/CR_QuoteVision_Vol_Surfaces/{self.cdx_index_name}/{dt.strftime("%Y-%m-%d")}/{pit_location}_CLOSE/surfaces.pkl'
        if not os.path.exists(surfaces_cache_path):
            return None
        with open(surfaces_cache_path, 'rb') as f:
            surfaces = pickle.load(f)
        return surfaces

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

    @staticmethod
    def filter_surfaces(surfaces, filter_method):
        filtered_key = filter_keys(surfaces.keys(), filter_method)
        surface_points = [val for val in surfaces[filtered_key].values() if isinstance(val, list)]
        return surface_points

    def set_dt_vol_surfaces(self, dt, surfaces):

        for surface_key, surface_dict in surfaces.items():
            surface_points = [val for val in surface_dict.values() if isinstance(val, list)]
            valuation_time_point = TimePoint(Date(dt.year,
                                                  dt.month,
                                                  dt.day),
                                             MicroDuration(dt.hour, 0, 0))
            vol_points = []
            expiries, strikes, vols = surface_points
            for i, expiry in enumerate(expiries):
                vol_date = datetime.strptime(str(expiry), '%Y%m%d')
                vol_points.append(ExpiryStrikeVolPoint(Date(vol_date.year,
                                                            vol_date.month,
                                                            vol_date.day),
                                                       Strike(strikes[i] / 10000),
                                                       Volatility(vols[i] / 100)))
            if len(vol_points) > 0:
                vol_surface = CubicSplineVolatility(self.day_counter_actual_360, valuation_time_point, vol_points)
                # cds_market_key = '.'.join([self.cdx_index_ticker, surface_key])
                self.data_dict.setdefault(dt, {}).setdefault(self.cdx_index_ticker, {})[surface_key] = vol_surface
        pass

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

    def save_surfaces_to_shared_drive(self, dt, surfaces):
        dir_path = os.path.join('/mnt/tradersny/CR_QuoteVision_Vol_Surfaces',
                                self.cdx_index_name,
                                dt.strftime("%Y-%m-%d"),
                                f'{self.pit_location}_CLOSE')
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, 'surfaces.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(surfaces, f)
        pass

    def initialize(self, data_request):

        dt = self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        live = dt.date() == datetime.today().date()
        if live and self.start_date != self.end_date:
            raise Exception("Only support single data request for Live")
        # Live
        # http://cpiceregistry:6703/creditvol?server_name=data_cache_creditvol&environment=CAPSTONE&cdxid=74356613
        # EOD
        # http://cpiceregistry:6703/creditvol.history?server_name=data_cache_creditvol&environment=CAPSTONE&PIT_SOURCE=CLOSE&PIT_DATE=2025-04-23&PIT_LOCATION=NYC&cdxid=74356613

        self.cdx_index_ticker = data_request.underlier
        self.cdx_index_name = self.MAP_GENERIC_CDS_INDEX_SYMBOL_NAME[self.cdx_index_ticker]
        self.cdxid_series_version_tenor_map = self.get_cdxid_series_version_tenor_map()[self.cdx_index_name]
        # cdxid -> all the specific series version tenor key inst ids that we need to associate to vols
        self.calendar, self.pit_location = data_request.calendar, data_request.calendar[0]

        if live:
            # fetch all live cdxid surfaces that exist for the index
            surfaces = self.fetch_surfaces_from_data_cache_api(live=True)
            # now filter for the one we want based on filter rule
            self.set_dt_vol_surfaces(dt, surfaces)
        else:
            self.payload.update({
                "PIT_SOURCE": "CLOSE",
                "PIT_LOCATION": self.pit_location,
            })
            holiday_days = self.get_holiday_days()

            # try fetch historical surface points on dt from cache in shared drive first
            while dt <= min(self.end_date, datetime.today() + timedelta(days=-1)):

                surfaces = self.fetch_surfaces_from_shared_drive(dt, self.pit_location)
                if surfaces is None:
                    # fetch from API and cache into shared drive folder
                    self.payload["PIT_DATE"] = dt.strftime("%Y-%m-%d")
                    # fetch all historical cdxid surfaces on dt that exist for the index
                    surfaces = self.fetch_surfaces_from_data_cache_api()
                    # save method
                    self.save_surfaces_to_shared_drive(dt, surfaces)

                self.set_dt_vol_surfaces(dt, surfaces)
                dt = add_business_days(dt, 1, holiday_days)

        container = CRVolDataContainer(data_request.underlier)

        def _get_cr_surface(_dt):
            if _dt is None:
                return self.data_dict
            else:
                return self.data_dict[_dt]

        def _get_trading_days():
            all_dt = list(self.data_dict.keys())
            all_dt.sort()
            return all_dt

        container._get_cr_surface = _get_cr_surface
        container._get_trading_days = _get_trading_days
        return container


