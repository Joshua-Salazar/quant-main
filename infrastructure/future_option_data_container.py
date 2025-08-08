import datetime
import pandas as pd

from ..data.datalake_cassandra import DatalakeCassandra
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..language import dataframe_to_records, format_isodate
from ..dates.utils import add_business_days
from ..dates.holidays import get_holidays


class FutureOptionDataContainer(DataContainer):
    def __init__(self, root: str):
        self.market_key = market_utils.create_option_data_container_key(root)

    def get_market_key(self):
        return self.market_key

    def get_option_universe(self, dt, future, return_as_list_of_dict=False):
        return self._get_option_universe(dt, future, return_as_list_of_dict=return_as_list_of_dict)

    def get_option_data(self, dt, option):
        return self._get_option_data(dt, option)

    def get_option_data_prev_eod(self, dt, option):
        return self._get_option_data_prev_eod(dt, option)

    def get_market_item(self, dt):
        return self


class IVOLOptionDataRequest(IDataRequest):
    def __init__(self, root, start_date=None, end_date=None, calendar=None,
                 include_weekly=False,  # include weekly option for TY
                 weekly_expiry_filter=None,    # filter weekly expiry by day, e.g. Friday
                 skip_future=False,     # allow to return option chain without specific future details
                 ):
        self.start_date = start_date
        self.end_date = end_date
        self.root = root
        self.calendar = calendar if calendar is not None else []
        self.include_weekly = include_weekly
        self.weekly_expiry_filter = weekly_expiry_filter
        self.skip_future = skip_future


class IVOLOptionDataSource(IDataSource):
    def __init__(self):
        self.dlc = DatalakeCassandra()
        self.data_dict = {}
        self.universe_dict = {}
        self.include_weekly = False
        self.weekly_expiry_filter = None
        self.skip_future = False
        self.ivol_futures_id = None
        self.num_next_futs = 3

    def get_next_future_ids(self, dt):
        # find next future
        curr_df = self.ivol_futures_id[(self.ivol_futures_id.expiration_year == dt.year) & (self.ivol_futures_id.expiration_month_id >= dt.month)].sort_values("expiration_month_id")
        df = curr_df.iloc[:self.num_next_futs]
        num_next_futs = self.num_next_futs - df.shape[0]
        if num_next_futs > 0:
            next_df = self.ivol_futures_id[self.ivol_futures_id.expiration_year == dt.year + 1].sort_values("expiration_month_id").iloc[:num_next_futs]
            df = pd.concat([df, next_df])
        next_future_ids = list(df.futures_id.unique())
        return next_future_ids

    def initialize(self, data_request):
        self.start_date = data_request.start_date
        self.end_date = data_request.end_date
        self.calendar = data_request.calendar
        self.root = data_request.root
        self.include_weekly = data_request.include_weekly
        self.weekly_expiry_filter = data_request.weekly_expiry_filter
        self.skip_future = data_request.skip_future
        self.holidays = get_holidays(self.calendar, self.start_date, self.end_date)
        self.ivol_fut_root_id = self.dlc.get_ivol_fut_root_id(self.root)
        if self.skip_future:
            # preload ivol fut id between start and end date
            expiration_years = list(range(self.start_date.year, self.end_date.year + 2))
            self.ivol_futures_id = self.dlc.get_ivol_futures(self.root, expiration_years=expiration_years)
        # self.ivol_opt_root_id = self.dlc.get_ivol_opt_root_id(self.root, self.ivol_fut_root_id, self.include_weekly, self.weekly_expiry_filter)

        data_container = FutureOptionDataContainer(data_request.root)

        def _get_option_universe(dt, future, return_as_list_of_dict=False):
            if (dt, future) not in self.universe_dict:
                futures_id = self.get_next_future_ids(dt) if self.skip_future else future.get_ivol_futures_id(self.dlc)
                self.universe_dict[(dt, future)] = self.dlc.get_ivol_options_for_future(futures_id, self.root, dt, self.include_weekly, self.weekly_expiry_filter, self.skip_future)
            df = self.universe_dict[(dt, future)]
            return dataframe_to_records(df) if return_as_list_of_dict else df

        def _get_option_data(dt, option):
            futures_id = option.underlying.get_ivol_futures_id(self.dlc)
            strike = option.strike
            call_put = "C" if option.is_call else "P"
            expiry_date = option.expiration

            if self.root == 'CL':
                expiry_date = None  # handles changing expiry dates around Juneteenth

            dt_str = format_isodate(dt)
            if option not in self.data_dict or dt_str > max(self.data_dict[option].keys()) :
                ivol_opt_root_id = self.dlc.get_ivol_opt_root_id(self.root, dt, self.include_weekly, self.weekly_expiry_filter)
                df = self.dlc.get_ivol_options_price(futures_id, ivol_opt_root_id, strike, call_put,
                                                     expiry_date=expiry_date, start_date=self.start_date,
                                                     end_date=self.end_date).set_index(['tstamp'])
                option_dict = df.to_dict('index')
                if option in self.data_dict:
                    self.data_dict[option].update(option_dict)
                else:
                    self.data_dict[option] = option_dict
            return self.data_dict[option].get(dt_str, None)

        def _get_option_data_prev_eod(dt: datetime, option):
            if option not in self.data_dict:
                raise RuntimeError('cannot prev eod stale data as first pull')
            dt = datetime.datetime(dt.year, dt.month, dt.day)
            prev_eod = add_business_days(dt, -1, self.holidays)
            prev_eod = format_isodate(prev_eod)
            # prev_eod = max(k for k in self.data_dict[option] if k < dt) # TODO: look up actual yesterday
            return self.data_dict[option].get(prev_eod, None)

        data_container._get_option_universe = _get_option_universe
        data_container._get_option_data = _get_option_data
        data_container._get_option_data_prev_eod = _get_option_data_prev_eod
        return data_container


if __name__ == '__main__':
    req = IVOLOptionDataRequest('CL', start_date=datetime.datetime(2022, 1, 1), end_date=datetime.datetime(2023, 1, 1))
    src = IVOLOptionDataSource()
    cont = src.initialize(req)
    fut_id = 562537
    universe = cont.get_option_universe(datetime.datetime(2022,11,16),futures_id=fut_id, return_as_list_of_dict=False)
    stk, cp = universe[['strike', 'call_put']].iloc[0]
    opt_data = cont.get_option_data(fut_id, stk, cp)
    print(universe.head())
    print(opt_data.head())
