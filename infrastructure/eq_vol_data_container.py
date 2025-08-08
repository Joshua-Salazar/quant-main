from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
from datetime import datetime, timedelta
import os
import pandas as pd

from ..analytics.symbology import option_calendar_from_ticker, option_underlying_type_from_ticker
from ..dates.utils import get_business_days
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..infrastructure.bs_vol import BSVol
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..infrastructure.volatility_surface import VolatilitySurface
from ..data.refdata import get_underlyings_map
from ..dates.holidays import get_holidays
from ..dates.utils import is_business_day
import glob


class EqVolDataContainer(DataContainer):
    def __init__(self, underlying: str):
        self.market_key = market_utils.create_vol_surface_key(underlying)

    def get_market_key(self):
        return self.market_key

    def get_vol_surface(self, dt=None):
        return self._get_vol_surface(dt)

    def get_market_item(self, dt):
        return self.get_vol_surface(dt)


class EqVolRequest(IDataRequest):
    def __init__(self, underlying: str,  # i.e. 'SPX Index'
                 start_date: datetime, end_date: datetime, requested_dates=None, num_regular=None):
        self.underlying = underlying
        self.start_date = start_date
        self.end_date = end_date
        self.requested_dates = requested_dates
        self.num_regular = num_regular


class VolaEqVolDataSourceDict(IDataSource):
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
        underlying = data_request.underlying
        container = EqVolDataContainer(underlying)

        def _get_vol_surface(dt):
            if dt is None:
                return self.cache['data']
            else:
                return self.cache['data'][dt]

        container._get_vol_surface = _get_vol_surface
        return container


class FlatEqVolDataSourceDict(IDataSource):
    def __init__(self, data_path_dict):
        self.data_path_dict = data_path_dict

    def initialize(self, data_request):
        underlying = data_request.underlying
        if underlying in self.data_path_dict:
            all_data = pd.read_csv(self.data_path_dict[underlying], parse_dates=["date"])
            cols = ["date", "Spot", "1M", "2M",	"3M", "6M",	"9M", "12M"]
            for col in cols:
                if col not in all_data.columns:
                    raise Exception(f"Not found column {col} in source {self.data_path_dict[underlying]}. "
                                    f"Source file must contain all columns: {','.join(cols)}.")
            if not all_data["date"].is_monotonic_increasing:
                raise Exception("Found date not monotonic increasing. Make sure that csv parse date is correct.")
            if not all_data["date"].is_unique:
                raise Exception("Found duplicate date.")
            all_data = all_data.set_index("date")
        else:
            raise Exception(f"Not found underlying {underlying}")

        all_data = all_data.loc[data_request.start_date:data_request.end_date]

        container = EqVolDataContainer(underlying)

        data_dict = {}
        for dt, row in all_data.iterrows():
            ts = row[row.index != "Spot"].to_dict()
            surface = BSVol(underlying_type=UnderlyingType.EQUITY, underlying=underlying, ts=ts, base_dt=dt, spot=row.Spot)
            data_dict[dt] = {underlying: surface}
        self.vol_data_dict = data_dict

        def _get_vol_surface(dt):
            if dt not in self.vol_data_dict:
                return None
            return self.vol_data_dict[dt][underlying]

        container._get_vol_surface = _get_vol_surface
        return container


class VolaEqVolDataSourceOnDemand(IDataSource):
    def __init__(self, skip_pattern_search=False, load_shared_file=True):
        self.skip_pattern_search = skip_pattern_search
        self.load_shared_file = load_shared_file
        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        self.eq_vol_surface_file_root_location = f'{data_file_folder}Solutions/VOLA_Surfaces/'
        # TODO: reftime could be different for non US names, there might be other variations of the file name
        self.eq_vol_surface_file_name_pattern = '{folder}/{underlying}_{date}-[0-9][0-9][0-9][0-9][0-9][0-9]-EDT_vs-eq.yml'
        # TODO: only load Equity surface for now
        self.underlyings_map = get_underlyings_map(return_df=False)
        self.data = None
        self.date = None
        self.underlying = None
        self.file_names = []
        self.vol_surface_file_location = None
        self.underlying_vol_surface_exists = False
        self.holidays = None
        self.num_regular = None

    def clone(self):
        return VolaEqVolDataSourceOnDemand()

    def load_surface(self, dt):
        factory = vola.makeFactoryAnalytics()
        vol_surface = None
        underlying_id = self.underlyings_map[self.underlying]
        is_future = option_underlying_type_from_ticker(self.underlying) == "future"
        if is_business_day(dt, self.holidays):
            try:
                if self.underlying_vol_surface_exists:
                    if self.skip_pattern_search:
                        target_file_name = f"{self.underlying}_{dt.strftime('%Y%m%d')}"
                        vol_surface_files_list = [
                            os.path.join(self.vol_surface_file_location, file_name) for file_name in self.file_names
                            if target_file_name in file_name and file_name[-4:] == ".yml"]
                    else:
                        vol_surface_files_list = glob.glob(
                            self.eq_vol_surface_file_name_pattern.format(folder=self.vol_surface_file_location,
                                                                         underlying=self.underlying,
                                                                         date=dt.strftime('%Y%m%d'))
                        )
                    assert len(vol_surface_files_list) <= 1
                    if len(vol_surface_files_list) == 1:
                        vola_surface = factory.makeVolSurfaceFutures(vol_surface_files_list[0]) \
                            if is_future else factory.makeVolSurfaceEquity(vol_surface_files_list[0])
                        vol_surface = VolatilitySurface.create_from_vola_surface(vola_surface, self.underlying)
                    else:
                        print('No Vola surface exists for {} | {}'.format(dt, self.underlying))
                else:
                    target_date_query_str = "Live" if dt.date() == datetime.today().date() else f"NYC|CLOSE|{dt.strftime('%Y%m%d')}"
                    underlying_type = UnderlyingType.FUTURES if is_future else UnderlyingType.EQUITY
                    vol_surface = VolatilitySurface.load(underlying_id, target_date_query_str, underlying_type, self.underlying)
            except Exception as e:
                print('Error at date {} : {}'.format(dt, e))
        if self.num_regular is not None:
            vol_surface = vol_surface.override_num_regular(self.num_regular)
        return vol_surface

    def get_vol_surface(self, dt):
        if dt is None:
            raise Exception("Not support")

        if dt == self.date:
            return self.data
        else:
            self.data = self.load_surface(dt)
            self.dt = dt
            return self.data

    def initialize(self, data_request):
        self.underlying = data_request.underlying
        self.num_regular = data_request.num_regular
        calendar = option_calendar_from_ticker(self.underlying)
        self.holidays = get_holidays(calendar, data_request.start_date, data_request.end_date)
        self.underlying_vol_surface_exists = self.load_shared_file and os.path.exists(self.eq_vol_surface_file_root_location + self.underlying)
        if self.underlying_vol_surface_exists:
            self.vol_surface_file_location = self.eq_vol_surface_file_root_location + self.underlying
            self.file_names = os.listdir(self.vol_surface_file_location)
        container = EqVolDataContainer(self.underlying)
        container._get_vol_surface = self.get_vol_surface
        return container


class VolaEqVolDataSource(IDataSource):
    def __init__(self, skip_pattern_search=False, load_shared_file=True):
        self.skip_pattern_search = skip_pattern_search
        self.load_shared_file = load_shared_file
        data_file_folder = "/misc/Traders/" if os.path.exists("/misc/Traders/") else "/mnt/tradersny/"
        self.eq_vol_surface_file_root_location = f'{data_file_folder}Solutions/VOLA_Surfaces/'
        # TODO: reftime could be different for non US names, there might be other variations of the file name
        self.eq_vol_surface_file_name_pattern = '{folder}/{underlying}_{date}-[0-9][0-9][0-9][0-9][0-9][0-9]-EDT_vs-eq.yml'
        # TODO: only load Equity surface for now
        self.factory = vola.makeFactoryAnalytics()
        self.underlyings_map = get_underlyings_map(return_df=False)
        self.data_dict = {}

    def clone(self):
        return VolaEqVolDataSource()

    def initialize(self, data_request):
        underlying = data_request.underlying
        underlying_vol_surface_exists = self.load_shared_file and os.path.exists(self.eq_vol_surface_file_root_location + underlying)
        file_names = []
        if underlying_vol_surface_exists:
            vol_surface_file_location = self.eq_vol_surface_file_root_location + underlying
            file_names = os.listdir(vol_surface_file_location)
        else:
            vol_surface_file_location = None
        underlying_id = self.underlyings_map[underlying]

        calendar = option_calendar_from_ticker(underlying)
        holidays = get_holidays(calendar, data_request.start_date, data_request.end_date)

        business_dates = get_business_days(data_request.start_date, data_request.end_date, holidays=holidays)

        is_future = option_underlying_type_from_ticker(underlying) == "future"
        for target_date in business_dates:
            if data_request.requested_dates is not None and target_date not in data_request.requested_dates:
                continue
            try:
                if underlying_vol_surface_exists:
                    if self.skip_pattern_search:
                        target_file_name = f"{underlying}_{target_date.strftime('%Y%m%d')}"
                        vol_surface_files_list = [
                            os.path.join(vol_surface_file_location, file_name) for file_name in file_names
                            if target_file_name in file_name]
                    else:
                        vol_surface_files_list = glob.glob(
                            self.eq_vol_surface_file_name_pattern.format(folder=vol_surface_file_location,
                                                                         underlying=underlying,
                                                                         date=target_date.strftime('%Y%m%d'))
                        )
                    assert len(vol_surface_files_list) <= 1
                    if len(vol_surface_files_list) == 1:
                        vola_surface = self.factory.makeVolSurfaceFutures(vol_surface_files_list[0]) \
                            if is_future else self.factory.makeVolSurfaceEquity(vol_surface_files_list[0])
                        vol_surface = VolatilitySurface.create_from_vola_surface(vola_surface, underlying)
                        self.data_dict[target_date] = vol_surface
                    else:
                        print('No Vola surface exists for {} | {}'.format(target_date, underlying))
                else:
                    target_date_query_str = "Live" if target_date.date() == datetime.today().date() else f"NYC|CLOSE|{target_date.strftime('%Y%m%d')}"
                    underlying_type = UnderlyingType.FUTURES if is_future else UnderlyingType.EQUITY
                    vol_surface = VolatilitySurface.load(underlying_id, target_date_query_str, underlying_type, underlying)
                    self.data_dict[target_date] = vol_surface
            except Exception as e:
                print('Error at date {} : {}'.format(target_date, e))
                continue
        container = EqVolDataContainer(underlying)

        def _get_vol_surface(dt):
            if dt is None:
                return self.data_dict
            else:
                if dt in self.data_dict:
                    return self.data_dict[dt]
                else:
                    return None
        container._get_vol_surface = _get_vol_surface
        return container

