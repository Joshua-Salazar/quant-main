from datetime import datetime
import copy
from multiprocessing import Pool
from ..dates.holidays import get_holidays
from ..dates.utils import date_range, is_business_day
from ..infrastructure.bmarket import BMarket
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..tradable.portfolio import Portfolio
from ..tools.timer import Timer
import os
import pandas as pd
import pickle
import traceback


class StrategyState:
    def __init__(self, time_stamp: datetime, portfolio: Portfolio):
        self.time_stamp = time_stamp
        self.portfolio = portfolio
        self.error_dates = []
        self.errors = []


class Event:
    def __init__(self, dt, strategy):
        self.strategy = strategy
        self.parameters = strategy.parameters
        self.data_containers = strategy.data_containers
        self.time_stamp = dt

    def get_time_stamp(self):
        return self.time_stamp

    def execute(self, state: StrategyState):
        pass


class Strategy:
    def __init__(self, start_date, end_date, calendar, currency, parameters, data_requests: dict[str, (IDataRequest, IDataSource)],
                 force_run=False, trading_days=[], inc_daily_states = False, is_SOFR = False, inc_trd_dts = False, cache_market_data=False, data_cache_path=None):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.currency = currency
        self.parameters = parameters
        self.data_requests = data_requests
        self.data_containers = {}
        self.force_run = force_run
        self.trading_days = trading_days
        self.inc_daily_states = inc_daily_states
        self.states = []
        self.is_SOFR = is_SOFR
        self.inc_trd_dts = inc_trd_dts
        self.cache_market_data = cache_market_data
        self.data_cache_path = data_cache_path
        if self.cache_market_data and not os.path.exists(self.data_cache_path):
            os.makedirs(self.data_cache_path, mode=0o777)

    def preprocess(self):
        timer = Timer("strategy preprocess ", unit="min", verbose=False)
        timer.start()
        i = 0
        for name, (data_request, data_source) in self.data_requests.items():
            if self.cache_market_data:
                data_cache_pkl = os.path.join(self.data_cache_path, f"{name}.pkl")
                if os.path.exists(data_cache_pkl):
                    with open(data_cache_pkl, 'rb') as f:
                        data = pickle.load(f)
                else:
                    data = data_source.initialize(data_request)
                    with open(data_cache_pkl, 'wb') as f:
                        try:
                            pickle.dump(data, f)
                        except:
                            if os.path.exists(data_cache_pkl):
                                os.remove(data_cache_pkl)
            else:
                data = data_source.initialize(data_request)
            #Todo: temporary allow to skip missing data as waiting for CA future data populated in datalake
            if data is not None:
                self.data_containers[name] = data
            timer.reset(f"Completed loading ({i+1}/{len(self.data_requests)}) underlying {name}")
            i = i + 1

        self.set_backtest_market()
        timer.end()

    def set_backtest_market(self):
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market

    def evolve(self, start: datetime, end: datetime, start_state: StrategyState):
        pass

    def postprocess(self):
        pass

    @staticmethod
    def get_pfo_df(results):
        res = []
        attr_cols = []
        for state in results:
            for pfo_name, pfo in state.portfolio.root.items():
                for pos_name, pos in pfo.root.items():
                    tmp = [state.time_stamp, pfo_name, pos_name, pos.quantity, pos.price, pos.quantity * pos.price]
                    attr_dict = pos.get_additional_attributes()
                    if len(attr_cols) == 0:
                        attr_cols = [k for k in attr_dict.keys() if k not in ["price"]]
                    for k in attr_cols:
                        tmp.append(attr_dict[k])
                    res.append(tmp)
        res = pd.DataFrame(res, columns=["dt", "pfo", "trd", "qty", "px", "price"] + attr_cols)
        return res


class DailyStrategy(Strategy):
    @staticmethod
    def calculate_holidays(calendar, start_date, end_date):
        if not isinstance(calendar, list):
            calendar = [calendar]
        holiday_days = []
        for cal in calendar:
            if isinstance(cal, str):
                holiday_days = holiday_days + get_holidays(cal, start_date, end_date)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')
        return holiday_days

    def __init__(self, start_date, end_date, calendar, currency, parameters, data_requests: dict[str, (IDataRequest, IDataSource)],
                 force_run=False, low_memory_mode=False, stateless_parallel_run=False, trading_days=[], logging=False,
                 is_SOFR=False,
                 inc_trd_dts=False,
                 log_file="", cache_market_data=False, data_cache_path=None):
        start_dates_from_data = [v[0].start_date for k, v in data_requests.items() if hasattr(v[0], 'start_date')]
        calendar_start_date = min(start_date, min(start_dates_from_data) if len(start_dates_from_data) else start_date)
        end_dates_from_data = [v[0].end_date for k, v in data_requests.items() if hasattr(v[0], 'end_date')]
        calendar_end_date = max(end_date, max(end_dates_from_data) if len(end_dates_from_data) else end_date)
        self.holidays = DailyStrategy.calculate_holidays(calendar, calendar_start_date, calendar_end_date)
        self.force_run = force_run
        self.low_memory_mode = low_memory_mode
        self.stateless_parallel_run = stateless_parallel_run
        self.logging = logging
        self.log_file = log_file
        self.is_SOFR = is_SOFR
        self.inc_trd_dts = inc_trd_dts

        super().__init__(start_date, end_date, calendar, currency, parameters, data_requests, force_run,
                         trading_days=trading_days,
                         is_SOFR=is_SOFR,
                         inc_trd_dts=inc_trd_dts, cache_market_data=cache_market_data, data_cache_path=data_cache_path)

    def generate_events(self, dt: datetime):
        pass

    def execute_stateless_event(self, dt, start_state):
        today_events = self.generate_events(dt)
        state = copy.deepcopy(start_state)
        try:
            for event in today_events:
                state = event.execute(state=state)
        except Exception as e:
            if self.logging:
                with open(self.log_file, "a+") as f:
                    f.write(dt.strftime("%Y-%m-%d") + "," + event.__class__.__name__ + "," + str(traceback.format_exc())
                            + str(e) + "\n")

    def evolve(self, start: datetime, end: datetime, start_state: StrategyState):
        daily_states = [start_state]
        if self.stateless_parallel_run:
            dates = self.trading_days if len(self.trading_days) > 0 else \
                [dt for dt in date_range(start, end) if is_business_day(dt, self.holidays)]
            num_processes = self.parameters["num_processes"]
            # chunk_size = max(math.ceil(len(dates)/num_processes), 15)
            # remove engine from parameter setting if we run parallel
            if "engine" in self.parameters:
                self.parameters["engine"] = None
            params = [(dt, start_state) for dt in dates]
            if num_processes == 1:
                res = []
                for param in params:
                    res.append(self.execute_stateless_event(*param))
            else:
                with Pool(processes=num_processes) as pool:
                    res = pool.starmap(self.execute_stateless_event, params)
        else:
            if len(self.trading_days) > 0:
                # can use intersection of days where data is available to
                # determine when to run the backtest
                trading_days = self.trading_days
            else:
                trading_days = date_range(start, end)

            from tqdm import tqdm
            timer = Timer("strategy evolve", unit="sec", verbose=False)
            timer.start()
            progress = tqdm(trading_days)
            for dt in progress:
                progress.set_postfix_str(dt.date())
                if is_business_day(dt, self.holidays):
                    today_events = self.generate_events(dt)
                    state = daily_states[-1]
                    if self.force_run:
                        try:
                            for event in today_events:
                                state = event.execute(state)
                        except Exception as e:
                            # reset state as the last available one
                            state = copy.deepcopy(daily_states[-1])
                            state.time_stamp = dt
                            state.error_dates.append(dt)
                            # fix for states that do not have <errors> attribute.
                            if hasattr(state, 'errors'):
                                state.errors.append(str(e))
                                if self.logging:
                                    if self.log_file == "":
                                        print(dt.strftime("%Y-%m-%d") + "," + event.__class__.__name__ + "," + str(traceback.format_exc()) + str(e) + "\n")
                                    else:
                                        with open(self.log_file, "a+") as f:
                                            f.write(dt.strftime("%Y-%m-%d") + "," + event.__class__.__name__ + "," + str(traceback.format_exc()) + str(e) + "\n")
                    else:
                        for event in today_events:
                            if self.inc_daily_states:
                                state = event.execute(state, daily_states)
                            else:
                                state = event.execute(state)
                    if self.low_memory_mode:
                        daily_states[-1] = state
                    else:
                        daily_states.append(state)

                    self.states = daily_states

                    timer.reset(f"Completed evolve on {dt.strftime('%Y-%m-%d')}")
            timer.end()
        return daily_states if self.low_memory_mode else daily_states[1:]


# TODO: Abstract out the periodic evolution
