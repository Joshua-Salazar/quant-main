import calendar
import json
import numbers
import re
from datetime import datetime, date, timedelta, timezone
from math import ceil

import numpy as np
import pandas as pd
import pytz
import requests
from dateutil.relativedelta import relativedelta

from ..dates import holidays
from ..analytics.constants import MICROSECONDS_PER_DAY, SECONDS_PER_DAY, YEAR_CALENDAR_DAY_COUNT
from ..analytics.symbology import OPTION_CALENDAR_FROM_TICKER
from ..constants.business_day_convention import BusinessDayConvention

EXCHANGE_TZ = {
    'CME': 'America/Chicago',
    'CBF': 'America/Chicago',
    'CBT': 'America/Chicago',
    'NYF': 'America/Chicago',
    'CMX': 'America/Chicago',
    'ICF': 'Europe/London',
    'EUX': 'Europe/Berlin',
    'EOP': 'Europe/Paris',
    'MIL': 'Europe/Berlin',
    'EOE': 'Europe/Amsterdam',
    'HKG': 'Asia/Hong_Kong',
    'KFE': 'Asia/Seoul',
    'OSE': 'Asia/Tokyo',
    'SFE': 'Australia/Sydney',
    'MSE': 'America/Montreal',
    'IST': 'Europe/Istanbul',
    'SGX': 'Asia/Singapore',
    'SAF': 'Africa/Johannesburg',
    'MFM': 'Europe/Madrid',
    'BMF': 'America/Sao_Paulo',
    'FTX': 'Asia/Taipei',
    'SSE': 'Europe/Stockholm',
    'TEF': 'Asia/Bangkok',
    'NGC': 'Asia/Kolkata',
    'WSE': 'Europe/Warsaw',
}

UNDERLYING_TIMEZONE = {
    "SPX Index": "America/New_York",
    "SX5E Index": "Europe/Berlin",
    "NKY Index": "Asia/Tokyo",
}

MAX_DATETIME = datetime.max.replace(tzinfo=timezone.utc)


def isoformat(datetime_like, format_str=''):
    if isinstance(datetime_like, str):
        return datetime.strptime(datetime_like, format_str).isoformat()
    elif isinstance(datetime_like, date):
        return datetime.combine(datetime_like, datetime.min.time()).isoformat()
    elif isinstance(datetime_like, datetime):
        return datetime_like.isoformat()
    else:
        raise RuntimeError('Cannot handle input type ' + type(datetime_like))


def timestamp_to_datetime(timestamp):
    return timestamp.to_pydatetime()


def vola_datetime_to_datetime(dt):
    return set_timezone(datetime.strptime(dt.toString(format='%Y-%m-%dT%H:%M:%S'), '%Y-%m-%dT%H:%M:%S.%f'),
                        'America/New_York')


def datetime_to_vola_datetime(dt: datetime, tz_from_dt=False, tz_name=None):
    """
    If tz_from_dt is true we assume dt has a named timezone info and we use that to construct vola datetime.
    If not we look at tz_name and use that to construct vola datetime if tz_name is given,
    otherwise we use vola's default timezone EDT
    In any case the constructed vola datetime represents the same point of time as the input datetime
    With default tz_from_dt=False, tz_name=None, any timezone info from dt is lost
    @param dt:
    @param tz_from_dt:
    @param tz_name:
    @return:
    """
    import pyvolar as vola
    if tz_from_dt:
        assert tz_name is None
        return vola.DateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, str(dt.tzinfo))
    else:
        if tz_name is None:
            return vola.DateTime(int(dt.timestamp() * 1e9))
        else:
            dt_as_named_tz = dt.astimezone(pytz.timezone(tz_name))
            return vola.DateTime(dt_as_named_tz.year, dt_as_named_tz.month, dt_as_named_tz.day,
                                 dt_as_named_tz.hour, dt_as_named_tz.minute, dt_as_named_tz.second,
                                 tz_name)


def datetime64_to_datetime(dt64):
    return timestamp_to_datetime(pd.Timestamp(dt64))


def is_aware(d):
    return d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None


def set_timezone(d, tz):
    """
    if d is tz aware we set the tz to the timezone given by tz (converting the timezone)
    if d is tz naive we localize it with the timezone given by tz
    @param d:
    @param tz_name:
    @return:
    """
    if isinstance(tz, str):
        tzinfo = pytz.timezone(tz)
    else:
        tzinfo = tz
    if is_aware(d):
        return d.astimezone(tzinfo)
    else:
        return tzinfo.localize(d)


def bdc_adjustment(dt, convention=BusinessDayConvention.PREVIOUS, holidays=[]):
    if is_business_day(dt, holidays=holidays):
        return dt
    if isinstance(convention, str):
        convention = BusinessDayConvention(convention.upper())

    if convention == BusinessDayConvention.PREVIOUS:
        return add_business_days(dt, -1, holidays)
    elif convention == BusinessDayConvention.FOLLOWING:
        return add_business_days(dt, 1, holidays)
    elif convention == BusinessDayConvention.MODIFIEDFOLLOWING:
        dt_adjust = add_business_days(dt, 1, holidays)
        if dt_adjust.month != dt.month:
            dt_adjust = add_business_days(dt, -1, holidays)
        return dt_adjust
    elif convention == BusinessDayConvention.MODIFIEDPREVIOUS:
        dt_adjust = add_business_days(dt, -1, holidays)
        if dt_adjust.month != dt.month:
            dt_adjust = add_business_days(dt, 1, holidays)
        return dt_adjust
    else:
        raise RuntimeError('Unknown business day adjustment convention ' + convention)


def is_business_day(dt, holidays=[]):
    if len(holidays) > 0 and not isinstance(holidays[0], datetime):
        holidays_dates = holidays
    else:
        holidays_dates = list(map(lambda x: x.date(), holidays))
    return not (dt.weekday() > 4 or (dt.date() in holidays_dates))


def add_business_days(base_date, shift, holidays=[]):
    """
    if shift>0, it returns the shift-th business day after base_date
    if shift<0, it returns the |shift|-th business day before base_date
    if shift=0, it returns the base_date even if it is not a business day
    @param base_date: datetime
    @param shift:
    @param holidays: a list of datetimes
    @return:
    """
    if len(holidays) > 0 and not isinstance(holidays[0], datetime):
        holidays_dates = holidays
    else:
        holidays_dates = list(map(lambda x: x.date(), holidays))
    if shift >= 0:
        dt = base_date
        for i in range(shift):
            dt = dt + timedelta(1)
            while dt.weekday() > 4 or dt.date() in holidays_dates:
                dt = dt + timedelta(1)
        return dt
    else:
        dt = base_date
        for i in range(abs(shift)):
            dt = dt + timedelta(-1)
            while dt.weekday() > 4 or dt.date() in holidays_dates:
                dt = dt + timedelta(-1)
        return dt


def get_holidays(cdr_code, start_date, end_date):
    """
    get the holiday days for a given calendar between start_date and end_date (both inclusive)
    @param cdr_code:
    @param start_date: datetime
    @param end_date: datetime
    @return: a list of holidays (datetime)
    """
    env_name = 'cpcapdata'
    payload = {
        'ExchIds': cdr_code,
        'StartDate': (start_date - timedelta(days=1)).strftime("%Y%m%d"),
        'EndDate': (end_date + timedelta(days=1)).strftime("%Y%m%d"),
        'format': 'json',
    }
    r = requests.get('http://' + env_name + ':5555/api/ctp/calendar', params=payload)
    json_string = r.content.decode("utf-8")
    json_obj = json.loads(json_string)
    if len(json_obj):
        # handle no json_obj[0]["Holidays"] error
        if "Holidays" in json_obj[0]:
            hols = json_obj[0]["Holidays"]
            return list(map(lambda x: datetime.strptime(str(x), '%Y%m%d'), hols))
        else:
            return []
    else:
        return []

def date_range(date1, date2):
    """
    iterator for all dates between two days (both inclusive)
    @param date1: datetime
    @param date2: datetime
    @return:
    """
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


def get_business_days(start_date, end_date, holidays=[]):
    """
    all business days between start_date and end_date (both inclusive if it is a business day)
    @param start_date: datetime
    @param end_date: datetime
    @param holidays: a list of datetimes
    @return:
    """
    all_dates = list(date_range(start_date, end_date))
    holidays_dates = list(map(lambda x: x.date(), holidays))
    biz_dates = []
    for dt in all_dates:
        if not (dt.weekday() > 4 or dt.date() in holidays_dates):
            biz_dates.append(dt)
    return biz_dates


def count_business_days(start_date, end_date, holidays=[]):
    """
    count the number of business days between start_date and end_date (both inclusive)
    this exludes all weekend days and any day in holidays
    @param start_date: datetime
    @param end_date: datetime
    @param holidays: a list of datetime
    @return:
    """
    count = 0
    all_dates = list(date_range(start_date, end_date))
    holidays_dates = list(map(lambda x: x.date(), holidays))
    for dt in all_dates:
        if not (dt.weekday() > 4 or dt.date() in holidays_dates):
            count += 1
    return count


def count_business_days_vectorized(start_dates, end_dates, holidays=[]):
    """
    Count the number of business days between start_dates and end_dates (both inclusive).
    This excludes all weekend days and any day in holidays.
    @param start_dates: array-like of datetime
    @param end_dates: array-like of datetime
    @param holidays: a list of datetime
    @return: Series of counts of business days
    """

    # Ensure holidays are in datetime format
    holidays_dates = pd.to_datetime(holidays).date

    # Create a DataFrame to hold start and end dates
    date_df = pd.DataFrame({'start': start_dates, 'end': end_dates})

    # Vectorized function to calculate business days
    def calculate_business_days(row):
        if pd.isna(row['start']) or pd.isna(row['end']):
            return np.nan
        # Calculate business days using pd.bdate_range
        business_days = pd.bdate_range(start=row['start'], end=row['end'], freq='B')
        # Exclude holidays
        business_days = business_days.difference(holidays_dates)
        return len(business_days)

    # Apply the function to calculate business days for each row
    date_df['business_days_count'] = date_df.apply(calculate_business_days, axis=1)

    return date_df['business_days_count']


def bus_day_in_month(date, holidays=[], pos_count=True):
    if pos_count:
        month_start = bdc_adjustment(datetime(date.year, date.month, 1), convention=BusinessDayConvention.FOLLOWING,
                                     holidays=holidays)
        day_count = count_business_days(month_start, date, holidays=holidays)
    else:
        month_end = bdc_adjustment(datetime(date.year, date.month, calendar.monthrange(date.year, date.month)[1]),
                                   convention=BusinessDayConvention.PREVIOUS, holidays=holidays)
        day_count = -count_business_days(date, month_end, holidays=holidays)
    return day_count

def add_tenor(base_date, tenor):
    if isinstance(tenor, numbers.Number):
        return base_date + relativedelta(microseconds=tenor * YEAR_CALENDAR_DAY_COUNT * MICROSECONDS_PER_DAY)
    elif isinstance(tenor, str):
        tenor = tenor.upper()
        elements = re.match("(|-)(\d+)(D|W|M|Y)", tenor).groups()
        sign = 1 if elements[0] == "" else -1
        delta = sign * int(elements[1])
        if elements[2] == 'D':
            return base_date + relativedelta(days=delta)
        elif elements[2] == 'W':
            return base_date + relativedelta(weeks=delta)
        elif elements[2] == 'M':
            return base_date + relativedelta(months=delta)
        elif elements[2] == 'Y':
            return base_date + relativedelta(years=delta)
        else:
            raise RuntimeError('The parsed tenor unit is not recognized ' + elements[2])
    else:
        raise RuntimeError('Unknown type of input tenor ' + type(tenor))

def add_bus_tenor(base_date, tenor, **kwargs):
    tgt_date = add_tenor(base_date, tenor)
    return bdc_adjustment(tgt_date, **kwargs)

def minus_tenor(base_date, tenor):
    if isinstance(tenor, numbers.Number):
        return base_date - relativedelta(microseconds=tenor * YEAR_CALENDAR_DAY_COUNT * MICROSECONDS_PER_DAY)
    elif isinstance(tenor, str):
        elements = re.match("(\d+)(D|W|M|Y)", tenor).groups()
        if elements[1] == 'D':
            return base_date - relativedelta(days=int(elements[0]))
        elif elements[1] == 'W':
            return base_date - relativedelta(weeks=int(elements[0]))
        elif elements[1] == 'M':
            return base_date - relativedelta(months=int(elements[0]))
        elif elements[1] == 'Y':
            return base_date - relativedelta(years=int(elements[0]))
        else:
            raise RuntimeError('The parsed tenor unit is not recognized ' + elements[1])
    else:
        raise RuntimeError('Unknown type of input tenor ' + type(tenor))


def tenor_to_days(tenor):
    elements = re.match("(\d+)(D|d|W|w|M|m|Y|y)", tenor).groups()
    if elements[1] in ['D', 'd']:
        return int(elements[0])
    elif elements[1] in ['W', 'w']:
        return int(elements[0]) * 7
    elif elements[1] in ['M', 'm']:
        return int(elements[0]) * 30
    elif elements[1] in ['Y', 'y']:
        return int(elements[0]) * 360
    else:
        raise RuntimeError('The parsed tenor unit is not recognized ' + elements[1])


def tenor_to_years(tenor):
    elements = re.match("(\d+)(D|d|W|w|M|m|Y|y)", tenor).groups()
    if elements[1] in ['D', 'd']:
        return int(elements[0]) / 360.
    elif elements[1] in ['W', 'w']:
        return int(elements[0]) / 52.
    elif elements[1] in ['M', 'm']:
        return int(elements[0]) / 12.
    elif elements[1] in ['Y', 'y']:
        return int(elements[0])
    else:
        raise RuntimeError('The parsed tenor unit is not recognized ' + elements[1])


def coerce_timezone(dt1, dt2):
    if is_aware(dt1) and not is_aware(dt2):
        aware_dt2 = set_timezone(dt2, dt1.tzinfo)
        return dt1, aware_dt2
    elif is_aware(dt2) and not is_aware(dt1):
        aware_dt1 = set_timezone(dt1, dt2.tzinfo)
        return aware_dt1, dt2
    else:
        return dt1, dt2


def datetime_diff(dt1, dt2):
    dt1, dt2 = coerce_timezone(dt1, dt2)
    return dt1 - dt2


def datetime_to_tenor(dt, base_date, coerce_tz=True):
    if coerce_tz:
        dt, base_date = coerce_timezone(dt, base_date)
        return (dt - base_date).total_seconds() / (YEAR_CALENDAR_DAY_COUNT * SECONDS_PER_DAY)
    else:
        return (dt - base_date).total_seconds() / (YEAR_CALENDAR_DAY_COUNT * SECONDS_PER_DAY)


def tenor_to_datetime(tenor, base_date):
    return add_tenor(base_date, tenor)


def datetime_equal(left: datetime, right: datetime, tol_days: int = 15, und=None) -> bool:
    """
    datetime equal with default tolerance 2 minutes
    """
    # return abs(left - right) <= timedelta(seconds=tol_secs)
    # return abs(left - right) <= timedelta(seconds=tol_secs)
    # ToDo remove usage of base date from benchmark surface spx and replace with real vol surface base day
    #  as we have seen 5 hours closing time difference between spx vol surface and v2x vol surface
    equal = abs(left - right) < timedelta(days=tol_days)
    if not equal and und is not None:
        st = left if left < right else right
        et = right if left < right else left
        hols = holidays.get_holidays(OPTION_CALENDAR_FROM_TICKER[und], st, et)
        gbd = count_business_days(st, et, hols)
        equal = gbd < tol_days
    return equal


def get_ny_timezone():
    return pytz.timezone("America/New_York")


def get_ny_now():
    return datetime.now(tz=get_ny_timezone())


def convert_to_ny(dt: datetime):
    return dt.astimezone(tz=get_ny_timezone())


def UTCtoEST(time):
    utc = pytz.utc.localize(time)
    est = utc.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
    est = datetime.strptime(est, '%Y-%m-%d %H:%M:%S')
    return est


def ESTtoUTC(time):
    est = pytz.timezone('US/Eastern').localize(time)
    utc = est.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S') + '.000+0000'
    return utc


def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))


def n_th_weekday_of_month(year, month, n, weekday):
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    next_month_n_th_weekdays = [day for week in monthcal for day in week if
                                day.weekday() == weekday and day.month == month]
    if n <= len(next_month_n_th_weekdays):
        return datetime.combine(next_month_n_th_weekdays[n - 1], datetime.min.time())
    else:
        return None


def which_n_th_weekday_of_month(dt):
    for n in [1, 2, 3, 4, 5]:
        n_th = n_th_weekday_of_month(dt.year, dt.month, n, dt.weekday())
        if n_th is not None and dt.date() == n_th.date():
            return n
    raise RuntimeError(f"cannot find which weekday of the given date")


def get_t_minus_one_bd(today=datetime.today(), calendar="#A", date_offset=-1):
    """
    get good business date using bbg calendar
    """
    bd = add_business_days(today, date_offset, holidays=get_holidays(calendar, today + timedelta(date_offset * 10 - 20),
                                                                     today + timedelta(10)))
    return bd


def date_to_datetime(dt):
    return datetime(dt.year, dt.month, dt.day)


def get_last_friday(today=date_to_datetime(date.today()), holidays=[]):
    offset = 7 if today.weekday() == 4 else (today.weekday() - 4) % 7
    friday = today - timedelta(days=offset)
    adj_friday = bdc_adjustment(friday, convention=BusinessDayConvention.PREVIOUS, holidays=holidays)
    return adj_friday


def get_fx_spot_date(base_date, pair, market=None, hols=None):
    codes = [pair[:3], pair[3:]]
    st = base_date
    et = base_date + timedelta(days=15)
    if hols is None:
        if market is None:
            hols = holidays.get_holidays_by_currency(codes, st, et)
        else:
            hols = []
            for code in codes:
                hols += market.get_holidays(code, st.date(), et.date())
    shift = 1 if pair in ["USDCAD"] else 2
    spot_date = add_business_days(base_date=base_date, shift=shift, holidays=hols)
    return spot_date


def get_fx_expiry_date(dt, tenor, pair, hol_dates):
    term = int(tenor[:-1])
    unit = tenor[-1]
    if unit in ["M", "Y"]:
        spot_date = get_fx_spot_date(dt, pair, market=None, hols=hol_dates)
        if unit == "M":
            delivery = spot_date + relativedelta(months=term)
        else:
            assert unit == "Y"
            delivery = spot_date + relativedelta(years=term)
        delivery = bdc_adjustment(delivery, convention=BusinessDayConvention.MODIFIEDFOLLOWING, holidays=hol_dates)
        expiry = get_fx_expiry_from_delivery(delivery, pair, market=None, hols=hol_dates)
    else:
        assert unit in ["D", "W"]
        if unit == "D":
            expiry = dt + relativedelta(days=term)
        else:
            assert unit == "W"
            expiry = dt + relativedelta(weeks=term)
    return expiry


def get_fx_expiry_from_delivery(delivery, pair, market=None, hols=None):
    expiry = delivery
    while not is_business_day(expiry, hols) or get_fx_spot_date(expiry, pair, market, hols) > delivery:
        expiry -= relativedelta(days=1)
    return expiry


def is_last_business_day_of_week(date, holidays):
    """
    Determine if the given date is the last business day of the week, considering holidays.

    Parameters:
    date (datetime): The date to check.
    holidays (list): A list of holiday dates (datetime objects).

    Returns:
    bool: True if the date is the last business day of the week, False otherwise.
    """
    # Convert holidays to a set for faster lookup
    holidays_set = set(holidays)

    # Check if the given date is a business day
    if date.weekday() >= 5 or date in holidays_set:
        return False

    # Check the next days in the week to see if they are business days
    next_day = date + timedelta(days=1)
    while next_day.weekday() < 5:
        if next_day not in holidays_set:
            return False
        next_day += timedelta(days=1)

    return True

# n-th last business day of the month
def is_last_nth_business_day(date, n, holiday):

    # Convert holidays to a set for faster lookup
    holidays_set = set(holiday)

    # Get the last day of the month
    last_day_of_month = calendar.monthrange(date.year, date.month)[1]

    # Start from the last day of the month and count backwards
    business_days_count = 0
    for day_offset in range(last_day_of_month, 0, -1):
        current_date = datetime(date.year, date.month, day_offset)

        # Check if it is a weekday (Monday to Friday)
        if current_date.weekday() < 5:
            business_days_count += 1

        # If the current date is the n-th last business day
        if business_days_count == n and current_date not in holidays_set:
            return current_date == date

    return False


def has_day(year, month, day):
    # Get the number of days in the month
    _, num_days = calendar.monthrange(year, month)
    # Check if the day is within the range of days in the month
    return 1 <= day <= num_days


def extract_date_only_from_datetime(_x):
    return datetime(_x.year, _x.month, _x.day)


def find_nearest_datetime(expiration, expiration_dates, method='absolute'):
    # We sort the string list of expiration dates here first.
    # This is to avoid unexpected behavior when two different expiries have the same distance to target date.
    # After sorted, we always select the one with shorter expiry.
    expiration_dates = sorted(expiration_dates)

    if method == 'leq':
        expiration_dates_filtered = list(filter(lambda x: x <= expiration, expiration_dates))
    elif method == 'geq':
        expiration_dates_filtered = list(filter(lambda x: x >= expiration, expiration_dates))
    elif method == "absolute":
        expiration_dates_filtered = expiration_dates
    else:
        raise RuntimeError(f"Unknown method {method}")

    if len(expiration_dates_filtered) == 0:
        raise RuntimeError(f"No datetime in the list {expiration_dates} is {method} the given datetime {expiration}")

    nearest_expiration_date = min(expiration_dates_filtered, key=lambda x: abs(datetime_diff(x, expiration)))
    return nearest_expiration_date


if __name__ == "__main__":
    cdr = '#A'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 11, 7)

    # Get list of business days for #A cdr
    # holiday_list = get_holidays(cdr, start_date, end_date)
    # bday_list = get_business_days(start_date, end_date, holidays=holiday_list)
    # print(type(bday_list), bday_list)
