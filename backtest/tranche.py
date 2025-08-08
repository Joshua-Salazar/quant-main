import itertools
import numbers
from datetime import datetime

import calendar as calendar_lib
from ..constants.business_day_convention import BusinessDayConvention
from ..dates.schedules import MonthlySchedule, WeeklySchedule, DailySchedule, QuarterlySchedule, MinuteSchedule, \
    MinuteEntrySchedule, DailyHoldSchedule, QuarterlyMonthDaySchedule
from ..dates.utils import bdc_adjustment, count_business_days
from ..dates.holidays import get_holidays
from ..dates.utils import add_tenor, is_business_day, add_business_days, n_th_weekday_of_month, \
    which_n_th_weekday_of_month
from ..tradable.cash import Cash
from ..tradable.constant import Constant


class RollingTrancheExpiryOffset:
    def __init__(self, start_date, end_date, initial_entry_dates, offset, calendar):
        self.initial_entry_dates = [initial_entry_dates] if isinstance(initial_entry_dates, datetime) else initial_entry_dates
        self.offset = offset
        self.holidays = get_holidays(calendar, start_date, end_date)

    def get_roll_date(self, **kwargs):
        expiry = kwargs['expiry']
        roll_date = add_business_days(expiry, self.offset, self.holidays)
        return roll_date


class RollingAtExpiryTranche:
    """
    Dynamic next roll date. User case is set next roll date as contract expiry date when entry the contract.
    In the case contract unwind earlier we still roll the original expiry date.
    """
    def __init__(self):
        self.next_roll_datetime = None

    def is_entry_datetime(self, dt):
        return self.next_roll_datetime == dt

    def get_exit_datetime(self, entry_datetime):
        return self.next_roll_datetime

    def get_tranche_fraction(self, dt):
        return 1.

    def set_next_roll_date(self, dt, holidays):
        # ensure roll date is one of evolve dates in backtester, i.e. skip holidays
        roll_date = bdc_adjustment(dt, convention=BusinessDayConvention.PREVIOUS, holidays=holidays)
        self.next_roll_datetime = roll_date


class RollingAtExpiryDailyTranche:
    """
    Dynamic update next batch entry/exist schedule. User case is set daily entry schedule when roll into next contract.
    In the case contract unwind earlier we still roll the original expiry date.
    """
    def __init__(self):
        self.entry_exit_map = {}
        self.entry_fraction_map = {}

    def is_entry_datetime(self, dt):
        return dt in self.entry_exit_map.keys()

    def get_exit_datetime(self, entry_datetime):
        return self.entry_exit_map[entry_datetime]

    def get_tranche_fraction(self, entry_datetime):
        return self.entry_fraction_map[entry_datetime]

    def update_tranche(self, dt, et, holidays):
        tranche_entry_schedule = []
        bds = count_business_days(dt, et, holidays=holidays)
        tranche_entry_schedule.append(dt)
        dt = add_business_days(dt, 1, holidays=holidays)
        tranche_entry_schedule.append(dt)
        tranche_exit_schedule = [et] * len(tranche_entry_schedule)
        # update fraction for current trade and set dummy fraction for next entry trade.
        tranche_fraction = [1 / bds, None]
        self.update_tranche_schedule(tranche_entry_schedule, tranche_exit_schedule, tranche_fraction)

    def update_tranche_schedule(self, tranche_entry_schedule, tranche_exit_schedule, tranche_fraction):
        self.entry_exit_map.update(dict(zip(tranche_entry_schedule, tranche_exit_schedule)))
        self.entry_fraction_map.update(dict(zip(tranche_entry_schedule, tranche_fraction)))


class Tranche:
    def __init__(self, tranche_entry_schedule=None, tranche_exit_schedule=None, tranche_fraction=None,
                 tranche_target_expiry = None ):

        self.tranche_entry_schedule = tranche_entry_schedule
        self.tranche_exit_schedule = None if tranche_entry_schedule is None else tranche_exit_schedule + [None] * (len(tranche_entry_schedule) - len(tranche_exit_schedule))
        if isinstance(tranche_fraction, numbers.Number):
            self.tranche_fraction = [tranche_fraction] * len(tranche_entry_schedule)
        else:
            self.tranche_fraction = tranche_fraction

        self.entry_exit_map = {}
        self.entry_fraction_map = {}
        if self.tranche_entry_schedule is not None:
            for entry, exit, fraction in zip(self.tranche_entry_schedule, self.tranche_exit_schedule, self.tranche_fraction):
                if entry in self.entry_exit_map:
                    self.entry_exit_map[entry] = self.entry_exit_map[entry] + [exit] if isinstance(self.entry_exit_map[entry], list) else [self.entry_exit_map[entry], exit]
                else:
                    self.entry_exit_map[entry] = exit
                if entry in self.entry_fraction_map:
                    self.entry_fraction_map[entry] = self.entry_fraction_map[entry] + [fraction] if isinstance(self.entry_fraction_map[entry], list) else [self.entry_fraction_map[entry], fraction]
                else:
                    self.entry_fraction_map[entry] = fraction

        self.tranche_target_expiry = tranche_target_expiry

        if tranche_target_expiry is None:
            self.entry_to_expiry = None
        else:
            self.entry_to_expiry = {}
            for entry, expiry in zip(self.tranche_entry_schedule, self.tranche_target_expiry):
                if entry in self.entry_to_expiry:
                    self.entry_to_expiry[entry] = self.entry_to_expiry[entry] + [expiry] if isinstance(self.entry_to_expiry[entry], list) else [self.entry_to_expiry[entry], expiry]
                else:
                    self.entry_to_expiry[entry] = expiry

    def get_entry_exit_map(self):
        return self.entry_exit_map

    def get_exit_datetime(self, entry_datetime):
        return self.entry_exit_map[entry_datetime]

    def is_entry_datetime(self, dt):
        return dt in self.entry_exit_map

    def get_tranche_fraction(self, entry_datetime):
        return self.entry_fraction_map[entry_datetime]

    def get_target_expiry(self, entry_datetime):
        if self.entry_to_expiry is None:
            return None
        else:
            return self.entry_to_expiry[entry_datetime]

    def add_tranche(self, entry_datetime, exit_datetime, fraction):
        self.entry_exit_map[entry_datetime] = exit_datetime
        self.entry_fraction_map[entry_datetime] = fraction
        self.tranche_entry_schedule.append(entry_datetime)
        self.tranche_exit_schedule.append(exit_datetime)
        self.tranche_fraction.append(fraction)

    @staticmethod
    def make_tranche_key(entry_date, exit_date):
        if exit_date is None:
            exit_date = datetime.max
        return f"{entry_date.strftime('%Y-%m-%d')}_{exit_date.strftime('%Y-%m-%d')}"

    @staticmethod
    def parse_tranche_key(tranche_name):
        elements = tranche_name.split('_')
        entry_str = elements[0]
        exit_str = elements[1]
        return datetime.strptime(entry_str, '%Y-%m-%d'), datetime.strptime(exit_str, '%Y-%m-%d')

    @staticmethod
    def unwind_tranche_portfolio(leg_name, tranche_name, portfolio, leg_portfolio, tranche_portfolio, market, strategy):
        tranche_position_names = [x for x in tranche_portfolio.get_positions().keys() if x != "delta_hedge"]
        tranche_position_names = list(filter(
            lambda x: not isinstance(tranche_portfolio.get_position(x).tradable, Cash) and not isinstance(
                tranche_portfolio.get_position(x).tradable, Constant), tranche_position_names))

        pos_path = (leg_name,)
        pos_path += tranche_name if isinstance(tranche_name, tuple) else (tranche_name,)
        for position_name in tranche_position_names:
            position = tranche_portfolio.get_position(position_name)
            unwind_price = position.tradable.price(market, strategy.valuer_map[type(position.tradable)],
                                                   calc_types='price')
            portfolio.unwind(pos_path + (position_name,), unwind_price, position.tradable.currency,
                             cash_path=pos_path)
        delta_hedge_pfo = tranche_portfolio.get_position("delta_hedge")
        delta_hedge_tranche_name = (tranche_name, "delta_hedge")
        if delta_hedge_pfo is None:
            return

        Tranche.unwind_tranche_portfolio(leg_name, delta_hedge_tranche_name, portfolio, leg_portfolio, delta_hedge_pfo, market, strategy)

    @staticmethod
    def move_tranche_portfolio_with_cash_only_to_leg_level(tranche_name, leg_portfolio, tranche_portfolio):
        # move tranche portfolio with cash only to leg level
        pos_list = list(tranche_portfolio.get_positions().items())
        cash_only = all(
            [isinstance(x[1].tradable, Cash) or isinstance(x[1].tradable, Constant) for x in pos_list])
        if cash_only:
            for cash_pos in pos_list:
                if isinstance(cash_pos[1].tradable, Cash) or isinstance(cash_pos[1].tradable, Constant):
                    leg_portfolio.move(
                        cash_pos[1].tradable,
                        cash_pos[1].quantity,
                        (tranche_name,), ()
                    )
                else:
                    raise RuntimeError(f"found non cash in an unwound tranche portfolio")




class DailyTranche(Tranche):
    def __init__(self, start_date, end_date, hold_days, bdc, calendar, trade_first_day = False):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = DailySchedule(bdc, holidays).schedule_days(start_date, end_date)
        if trade_first_day:
            entry_override = []
            for dt in [start_date] + [add_tenor(start_date, '%dD' % x) for x in range(len(tranche_entry_schedule))[1:]]:
                if is_business_day(dt, holidays):
                    entry_override.append(dt)
                else:
                    previous_dt = add_business_days(dt, -1, holidays=holidays)
                    if previous_dt not in entry_override:
                        entry_override.append(previous_dt)
            super().__init__(entry_override, tranche_entry_schedule[hold_days:], 1 / hold_days)
        else:
            super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_days:], 1 / hold_days)


class MinuteTranche(Tranche):

    def __init__(self, start_date, end_date, hold_intervals, bdc, calendar, start_time, end_time, interval=1):

        holidays = get_holidays(calendar, start_date, end_date)

        tranche_entry_schedule = MinuteSchedule(bdc, holidays, start_time, end_time,interval).schedule_minutes(start_date, end_date)

        super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_intervals:], 1 / hold_intervals)


class MinuteEntry(Tranche):

    def __init__(self, start_date, end_date, bdc, calendar, entry_time, interval=1):

        holidays = get_holidays(calendar, start_date, end_date)

        tranche_entry_schedule, tranche_exit_schedule = MinuteEntrySchedule(bdc,holidays,entry_time,interval).schedule_minutes(start_date, end_date)

        super().__init__(tranche_entry_schedule, tranche_exit_schedule, 1 )


def calculate_target_expiry(base_date, tenor):
    if tenor == "next_month_same_week_friday":
        week_number = which_n_th_weekday_of_month(base_date)
        base_date_plus_1m = add_tenor(base_date, "1M")
        target_expiration = n_th_weekday_of_month(base_date_plus_1m.year, base_date_plus_1m.month, week_number, calendar_lib.FRIDAY)
        if target_expiration is None: # there is no nth week in next month, we have to return the first week of the following month
            base_date_plus_2m = add_tenor(base_date, "2M")
            target_expiration = n_th_weekday_of_month(base_date_plus_2m.year, base_date_plus_2m.month, 1, calendar_lib.FRIDAY)
        return target_expiration
    else:
        if tenor.endswith('+'):
            tenor_used = tenor[:-1]
        elif tenor.endswith('-'):
            tenor_used = tenor[:-1]
        else:
            tenor_used = tenor

        target_expiration = add_tenor(base_date, tenor_used)
        return target_expiration


class WeeklyTranche(Tranche):
    def __init__(self, start_date, end_date, which_week_day, hold_weeks, bdc, calendar,
                 trade_first_day=False, trade_first_day_method="shift_all_entry_dates"):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = WeeklySchedule(which_week_day, bdc, holidays).schedule_days(start_date, end_date)
        if trade_first_day:
            if trade_first_day_method == "shift_all_entry_dates":
                entry_override = []
                for dt in [start_date] + [add_tenor(start_date, '%dW' % x) for x in range(len(tranche_entry_schedule))[1:]]:
                    if is_business_day(dt, holidays):
                        entry_override.append(dt)
                    else:
                        entry_override.append(add_business_days(dt, -1, holidays=holidays))
                assert len(entry_override) == len(tranche_entry_schedule)
                super().__init__(entry_override, tranche_entry_schedule[hold_weeks:], 1 / hold_weeks)
            elif trade_first_day_method == "shift_first_entry_date_and_trade_all_tranches" or trade_first_day_method == "shift_first_entry_date_and_trade_all_tranches_but_exclude_5th_week_entry_date_if_the_same_week_friday_does_not_exist":
                unadjusted_tranches = WeeklyTranche(start_date, end_date, which_week_day, hold_weeks, bdc, calendar, trade_first_day=False)
                backdated_start_day = add_tenor(unadjusted_tranches.tranche_entry_schedule[0], '%dW' % -(hold_weeks - 1))
                backdated_unadjusted_tranches = WeeklyTranche(backdated_start_day, end_date, which_week_day, hold_weeks, bdc, calendar, trade_first_day=False)
                tranche_entry_schedule = [x if x > unadjusted_tranches.tranche_entry_schedule[0] else start_date for x in backdated_unadjusted_tranches.tranche_entry_schedule]

                # some very specific rule to be removed!
                if trade_first_day_method == "shift_first_entry_date_and_trade_all_tranches_but_exclude_5th_week_entry_date_if_the_same_week_friday_does_not_exist":
                    chosen_flags = []
                    for d in tranche_entry_schedule:
                        week_number = which_n_th_weekday_of_month(d)
                        same_week_friday = n_th_weekday_of_month(d.year, d.month, week_number, calendar_lib.FRIDAY)
                        if week_number == 5 and same_week_friday is not None:
                            chosen_flags.append(False)
                        else:
                            chosen_flags.append(True)
                    super().__init__(list(itertools.compress(tranche_entry_schedule, chosen_flags)),
                                     list(itertools.compress(backdated_unadjusted_tranches.tranche_exit_schedule, chosen_flags)),
                                     list(itertools.compress(backdated_unadjusted_tranches.tranche_fraction, chosen_flags)))
                else:
                    super().__init__(tranche_entry_schedule, backdated_unadjusted_tranches.tranche_exit_schedule, backdated_unadjusted_tranches.tranche_fraction)
            else:
                raise RuntimeError(f"Unknown trade first day method {trade_first_day_method}")
        else:
            super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_weeks:], 1 / hold_weeks)


class MonthlyTranche(Tranche):
    def __init__(self, start_date, end_date, which_week, which_week_day, hold_months, bdc, calendar,
                 trade_first_day = False,
                 shift_all_days = True,
                 custom_entry = [],
                 custom_exit = [], ):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = MonthlySchedule(which_week, which_week_day, bdc, holidays).schedule_days(start_date, end_date)
        if len( custom_entry ) > 0:
            if len(custom_exit) > 0:
                super().__init__(custom_entry, custom_exit, 1 / hold_months)
            else:
                trade_exit = []
                for dt in custom_entry:
                    standard_entry = [x for x in tranche_entry_schedule if (x.month == dt.month) and (x.year == dt.year)]
                    assert len(standard_entry) in [ 0, 1]
                    if len(standard_entry) > 0:
                        idx = tranche_entry_schedule.index( standard_entry[0] ) + hold_months
                        if idx in range(len(tranche_entry_schedule)):
                            trade_exit.append( tranche_entry_schedule[ idx ] )
                super().__init__( custom_entry, trade_exit, 1 / hold_months)
        else:
            if trade_first_day:
                if shift_all_days:
                    entry_override = []
                    for dt in [start_date] + [ add_tenor( start_date, '%dM'%x ) for x in range(len(tranche_entry_schedule))[1:] ]:
                        if is_business_day( dt, holidays):
                            entry_override.append(dt)
                        else:
                            entry_override.append(add_business_days(dt, -1, holidays=holidays))
                    assert len(entry_override) == len(tranche_entry_schedule)
                else:
                    entry_override = tranche_entry_schedule
                    entry_override[ 0 ] = start_date
                super().__init__(entry_override, tranche_entry_schedule[hold_months:], 1 / hold_months)
            else:
                super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_months:], 1 / hold_months)


class QuarterlyTranche(Tranche):
    def __init__(self, start_date, end_date, which_month, which_week, which_week_day, hold_quarters, bdc, calendar,
                 trade_first_day = False,
                 shift_all_days = True,
                 custom_entry = [],
                 custom_exit = [], ):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = QuarterlySchedule(which_month, which_week, which_week_day, bdc, holidays).schedule_days(start_date, end_date)
        if len( custom_entry ) > 0:
            raise RuntimeError('custom_entry is not supported')
        else:
            if trade_first_day:
                raise RuntimeError('trade_first_day is not supported')
            else:
                super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_quarters:], 1 / hold_quarters)


class QuarterlyMonthDayTranche(Tranche):
    def __init__(self, start_date, end_date, which_month, month_day, hold_quarters, bdc, calendar,
                 trade_first_day=False,
                 trade_first_day_method=None,
                 custom_entry = [],
                 custom_exit = [], ):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = QuarterlyMonthDaySchedule(which_month, month_day, bdc, holidays).schedule_days(start_date, end_date)
        if len( custom_entry ) > 0:
            raise RuntimeError('custom_entry is not supported')
        else:
            if trade_first_day:
                if trade_first_day_method == "add_first_day":
                    tranche_entry_schedule.insert(0, start_date)
                    super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_quarters:], 1 / hold_quarters)
                else:
                    raise RuntimeError(f'trade_first_day with {trade_first_day_method} is not supported')
            else:
                super().__init__(tranche_entry_schedule, tranche_entry_schedule[hold_quarters:], 1 / hold_quarters)


class CustomTranche( Tranche ):
    def __init__( self, custom_entry,
                  custom_exit,
                  custom_expiry ):
            super().__init__( custom_entry, custom_exit, 1, tranche_target_expiry = custom_expiry )


class DailyHold(Tranche):
    def __init__(self, start_date, end_date, hold_days, bdc, calendar, trade_first_day = False):
        holidays = get_holidays(calendar, start_date, end_date)
        tranche_entry_schedule = DailyHoldSchedule(bdc, holidays, hold_days).schedule_days(start_date, end_date)
        if trade_first_day:
            entry_override = []
            for dt in [start_date] + [add_tenor(start_date, '%dD' % x) for x in range(len(tranche_entry_schedule))[1:]]:
                if is_business_day(dt, holidays):
                    entry_override.append(dt)
                else:
                    previous_dt = add_business_days(dt, -1, holidays=holidays)
                    if previous_dt not in entry_override:
                        entry_override.append(previous_dt)
            super().__init__(entry_override, tranche_entry_schedule[1:],1)
        else:
            super().__init__(tranche_entry_schedule, tranche_entry_schedule[1:],1)