from datetime import datetime
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
from ..dates.utils import bdc_adjustment, add_business_days, has_day
import abc


class Schedule:
    def __init__(self):
        pass

    @abc.abstractmethod
    def next_schedule_day_on_or_after(self, dt):
        pass

    def next_schedule_day(self, dt):
        return self.next_schedule_day_on_or_after(dt + relativedelta(days=1))

    def schedule_days(self, start_date, end_date):
        days = []
        dt = self.next_schedule_day_on_or_after(start_date)
        while dt <= end_date:
            if dt >= start_date:
                days.append(dt)
            new_dt = self.next_schedule_day(dt)
            assert new_dt > dt
            dt = new_dt
        return days


class MonthDaySchedule(Schedule):
    def __init__(self, day, bdc, holidays):
        self.day = day
        self.bdc = bdc
        self.holidays = holidays

    def _get_datetime_at_day_of_month(self, dt):
        if self.day == 'Last':
            for d in [31, 30, 29, 28]:
                if has_day(dt.year, dt.month, d):
                    this_month_date = dt.replace(day=d)
                    return this_month_date
        else:
            month_day = self.day
            if has_day(dt.year, dt.month, month_day):
                this_month_date = dt.replace(day=month_day)
                return this_month_date
            else:
                return None

    def next_schedule_day_on_or_after(self, dt):
        this_month_date = self._get_datetime_at_day_of_month(dt)
        if this_month_date is None:
            this_month_date = dt + relativedelta(months=1)
            this_month_date = self._get_datetime_at_day_of_month(this_month_date)

        while bdc_adjustment(this_month_date, convention=self.bdc, holidays=self.holidays) < dt:
            this_month_date = this_month_date + relativedelta(months=1)
            this_month_date = self._get_datetime_at_day_of_month(this_month_date)
            if this_month_date is None:
                this_month_date = dt + relativedelta(months=1)
                this_month_date = self._get_datetime_at_day_of_month(this_month_date)

        return bdc_adjustment(this_month_date, convention=self.bdc, holidays=self.holidays)


class QuarterlySchedule(Schedule):
    def __init__(self, which_month, which_week, which_week_day, bdc, holidays):
        self.which_month = which_month
        self.which_week = which_week
        which_week_day = which_week_day if isinstance(which_week_day, int) else which_week_day.value
        self.which_week_day = {0: MO, 1: TU, 2: WE, 3: TH, 4: FR, 5: SA, 6: SU}[which_week_day]
        self.bdc = bdc
        self.holidays = holidays

    def next_schedule_day_on_or_after(self, dt):
        this_quarty_date = self._schedule_day_this_quarter(dt)

        while bdc_adjustment(this_quarty_date, convention=self.bdc, holidays=self.holidays) < dt:
            this_quarty_date = this_quarty_date + relativedelta(months=3)
            this_quarty_date = self._schedule_day_this_quarter(this_quarty_date)

        return bdc_adjustment(this_quarty_date, convention=self.bdc, holidays=self.holidays)

    def _schedule_day_this_quarter(self, dt: datetime):
        if dt.month in [1, 2, 3]:
            dt = dt.replace(month=self.which_month)
        elif dt.month in [4, 5, 6]:
            dt = dt.replace(month=self.which_month + 3)
        elif dt.month in [7, 8, 9]:
            dt = dt.replace(month=self.which_month + 6)
        elif dt.month in [10, 11, 12]:
            dt = dt.replace(month=self.which_month + 9)
        else:
            raise RuntimeError('invalid month number')
        dt = dt.replace(day=1)
        dt = dt + relativedelta(weekday=self.which_week_day(self.which_week))
        return bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)


class QuarterlyMonthDaySchedule(Schedule):
    def __init__(self, which_month, month_day, bdc, holidays):
        self.which_month = which_month
        self.month_day = month_day
        self.bdc = bdc
        self.holidays = holidays

    def next_schedule_day_on_or_after(self, dt):
        this_quarty_date = self._schedule_day_this_quarter(dt)

        while bdc_adjustment(this_quarty_date, convention=self.bdc, holidays=self.holidays) < dt:
            this_quarty_date = this_quarty_date + relativedelta(months=3)
            this_quarty_date = self._schedule_day_this_quarter(this_quarty_date)

        return bdc_adjustment(this_quarty_date, convention=self.bdc, holidays=self.holidays)

    def _schedule_day_this_quarter(self, dt: datetime):
        if dt.month in [1, 2, 3]:
            dt = dt.replace(month=self.which_month)
        elif dt.month in [4, 5, 6]:
            dt = dt.replace(month=self.which_month + 3)
        elif dt.month in [7, 8, 9]:
            dt = dt.replace(month=self.which_month + 6)
        elif dt.month in [10, 11, 12]:
            dt = dt.replace(month=self.which_month + 9)
        else:
            raise RuntimeError('invalid month number')
        dt = dt.replace(day=1)
        this_month_date = None
        if self.month_day == 'Last':
            for d in [31, 30, 29, 28]:
                if has_day(dt.year, dt.month, d):
                    this_month_date = dt.replace(day=d)
                    break
        else:
            month_day = self.month_day
            if has_day(dt.year, dt.month, month_day):
                this_month_date = dt.replace(day=month_day)
            else:
                for d in [31, 30, 29, 28]:
                    if has_day(dt.year, dt.month, d):
                        this_month_date = dt.replace(day=d)
                        break

        return bdc_adjustment(this_month_date, convention=self.bdc, holidays=self.holidays)


class MonthlySchedule(Schedule):
    def __init__(self, which_week, which_week_day, bdc, holidays):
        self.which_week = which_week
        which_week_day = which_week_day if isinstance(which_week_day, int) else which_week_day.value
        self.which_week_day = {0: MO, 1: TU, 2: WE, 3: TH, 4: FR, 5: SA, 6: SU}[which_week_day]
        self.bdc = bdc
        self.holidays = holidays

    def next_schedule_day_on_or_after(self, dt):
        this_month_date = self._schedule_day_this_month(dt)

        while bdc_adjustment(this_month_date, convention=self.bdc, holidays=self.holidays) < dt:
            this_month_date = this_month_date + relativedelta(months=1)
            this_month_date = self._schedule_day_this_month(this_month_date)

        return bdc_adjustment(this_month_date, convention=self.bdc, holidays=self.holidays)

    def _schedule_day_this_month(self, dt: datetime):
        dt = dt.replace(day=1)
        dt = dt + relativedelta(weekday=self.which_week_day(self.which_week))
        return bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)


class WeeklySchedule(Schedule):
    def __init__(self, which_week_day, bdc, holidays):
        which_week_day = which_week_day if isinstance(which_week_day, int) else which_week_day.value
        self.which_week_day = which_week_day
        self.bdc = bdc
        self.holidays = holidays

    def next_schedule_day_on_or_after(self, dt):
        this_week_date = self._schedule_day_this_week(dt)

        while bdc_adjustment(this_week_date, convention=self.bdc, holidays=self.holidays) < dt:
            next_week_date = this_week_date + relativedelta(weeks=1)
            next_week_date = self._schedule_day_this_week(next_week_date)
            if next_week_date == this_week_date:
                next_week_date = this_week_date + relativedelta(weeks=2)
                next_week_date = self._schedule_day_this_week(next_week_date)
            this_week_date = next_week_date

        return bdc_adjustment(this_week_date, convention=self.bdc, holidays=self.holidays)

    def _schedule_day_this_week(self, dt: datetime):
        offset = self.which_week_day - dt.weekday()
        dt = dt + relativedelta(days=offset)
        return bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)


class DailySchedule(Schedule):
    def __init__(self, bdc, holidays):
        self.bdc = bdc
        self.holidays = holidays

    def next_schedule_day_on_or_after(self, dt):
        this_dt = bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)
        while this_dt < dt:
            this_dt = add_business_days(this_dt, 1, self.holidays)

        return this_dt


class MinuteSchedule():
    def __init__(self, bdc, holidays, start_time, end_time, interval=1):
        self.bdc = bdc
        self.holidays = holidays
        self.interval = interval
        self.start_time=start_time
        self.end_time=end_time

    def next_schedule_day(self, dt):
        return self.next_schedule_minutes_on_or_after(dt + relativedelta(minutes=self.interval))

    def schedule_minutes(self, start_date, end_date):
        days = []
        start_hour=int(self.start_time.split(':')[0])
        start_minute=int(self.start_time.split(':')[1])
        end_hour=int(self.end_time.split(':')[0])
        end_minute=int(self.end_time.split(':')[1])
        dt = self.next_schedule_minutes_on_or_after(datetime(start_date.year,start_date.month,start_date.day,start_hour,start_minute))
        while dt <= end_date:
            day_start=datetime(dt.year,dt.month,dt.day,start_hour,start_minute)
            day_end=datetime(dt.year,dt.month,dt.day,end_hour,end_minute)
            if dt >= start_date and dt >= day_start and dt < day_end:
                days.append(dt)
            new_dt = self.next_schedule_day(dt)
            if new_dt > day_end:
                new_dt = add_business_days(day_start, 1, self.holidays)
            assert new_dt > dt
            dt = new_dt
        return days

    def next_schedule_minutes_on_or_after(self, dt):
        this_dt = bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)
        while this_dt < dt:
            this_dt = add_business_days(this_dt, 1, self.holidays)

        return this_dt


class MinuteEntrySchedule():
    def __init__(self, bdc, holidays, entry_time, interval=1):
        self.bdc = bdc
        self.holidays = holidays
        self.interval = interval
        self.entry_time=entry_time

    def exit_day(self, dt):
        return self.next_schedule_minutes_on_or_after(dt + relativedelta(minutes=self.interval))

    def schedule_minutes(self, start_date, end_date):
        days = []
        exit_dates=[]
        entry_hour=int(self.entry_time.split(':')[0])
        entry_minute=int(self.entry_time.split(':')[1])
        dt = self.next_schedule_minutes_on_or_after(datetime(start_date.year,start_date.month,start_date.day,entry_hour,entry_minute))
        while dt <= end_date:
            days.append(dt)
            exit_dt = self.exit_day(dt)
            exit_dates.append(exit_dt)
            dt = add_business_days(dt, 1, self.holidays)
        return days,exit_dates

    def next_schedule_minutes_on_or_after(self, dt):
        this_dt = bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)
        while this_dt < dt:
            this_dt = add_business_days(this_dt, 1, self.holidays)

        return this_dt
    

class DailyHoldSchedule(Schedule):
    def __init__(self, bdc, holidays, date_gap):
        self.bdc = bdc
        self.holidays = holidays
        self.date_gap = date_gap

    def next_schedule_day(self, dt):
        return self.next_schedule_day_on_or_after(dt + relativedelta(days=self.date_gap))

    def next_schedule_day_on_or_after(self, dt):
        this_dt = bdc_adjustment(dt, convention=self.bdc, holidays=self.holidays)
        while this_dt < dt:
            this_dt = add_business_days(this_dt, 1, self.holidays)

        return this_dt


    def schedule_days(self, start_date, end_date):
        days = []
        dt = self.next_schedule_day_on_or_after(start_date)
        while dt <= end_date:
            if dt >= start_date:
                days.append(dt)
            new_dt = self.next_schedule_day(dt)
            assert new_dt > dt
            dt = new_dt
        return days