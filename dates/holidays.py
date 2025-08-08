from datetime import datetime

from ..data.utils import read_sql


def get_holidays_by_exchange_trading(codes, start_date, end_date):
    if not isinstance(codes, list):
        codes = [codes]

    codes_str = "','".join(codes)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    sql_query = f"SELECT ISOCOUNTRYCODE, ISO_MIC_CODE, EXCHANGENAME, EVENTNAME, EVENTDATE FROM DBO.COPP_EXCHANGETRADING WHERE ISO_MIC_CODE in ('{codes_str}') AND EVENTDATE>='{start_date_str}' AND EVENTDATE<='{end_date_str}' ORDER BY EVENTDATE"
    exchange_holidays = read_sql(sql_query, server='DBCTPPROD.capstoneco.com', database='capstone', userid='ctp', password='ctppw')
    holiday_dates = exchange_holidays['EVENTDATE'].values
    holiday_dates = [datetime.combine(x, datetime.min.time()) for x in holiday_dates]
    return holiday_dates


def get_holidays_by_financial_centre(codes, start_date, end_date):
    if not isinstance(codes, list):
        codes = [codes]

    codes_str = "','".join(codes)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    sql_query = f"SELECT ISOCOUNTRYCODE, UN_LOCODE, FINANCIALCENTER, EVENTNAME, EVENTDATE FROM DBO.COPP_FINANCIALCENTRES WHERE UN_LOCODE in ('{codes_str}') AND EVENTDATE>='{start_date_str}' AND EVENTDATE<='{end_date_str}' ORDER BY EVENTDATE"
    exchange_holidays = read_sql(sql_query, server='DBCTPPROD.capstoneco.com', database='capstone', userid='ctp', password='ctppw')
    holiday_dates = exchange_holidays['EVENTDATE'].values
    holiday_dates = [datetime.combine(x, datetime.min.time()) for x in holiday_dates]
    return holiday_dates


def get_holidays_by_currency(codes, start_date, end_date):
    if not isinstance(codes, list):
        codes = [codes]

    codes_str = "','".join(codes)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    sql_query = f"SELECT ISOCOUNTRYCODE, ISOCURRENCYCODE, EVENTNAME, EVENTDATE FROM DBO.COPP_CURRENCIES WHERE ISOCURRENCYCODE in ('{codes_str}') AND EVENTDATE>='{start_date_str}' AND EVENTDATE<='{end_date_str}' ORDER BY EVENTDATE"
    exchange_holidays = read_sql(sql_query, server='DBCTPPROD.capstoneco.com', database='capstone', userid='ctp', password='ctppw')
    holiday_dates = exchange_holidays['EVENTDATE'].values
    holiday_dates = [datetime.combine(x, datetime.min.time()) for x in holiday_dates]
    return holiday_dates


def get_holidays(codes, start_date, end_date):
    if not isinstance(codes, list):
        codes = [codes]

    extra_days = list(filter(lambda x: isinstance(x, datetime), codes))
    codes = list(filter(lambda x: not isinstance(x, datetime), codes))

    if len(codes) > 0:
        holiday_dates = get_holidays_by_exchange_trading(codes, start_date, end_date)
        if len(holiday_dates):
            return holiday_dates + extra_days

        holiday_dates = get_holidays_by_financial_centre(codes, start_date, end_date)
        if len(holiday_dates):
            return holiday_dates + extra_days

        holiday_dates = get_holidays_by_currency(codes, start_date, end_date)
        if len(holiday_dates):
            return holiday_dates + extra_days

    return [] + extra_days


def find_calendar_code_by_exchange(*key_words):
    sql_query = "SELECT ISO_MIC_CODE, EXCHANGENAME, ISOCOUNTRYCODE FROM DBO.COPP_EXCHANGETRADING ORDER BY EVENTDATE"
    hols = read_sql(sql_query, server='DBCTPPROD.capstoneco.com', database='capstone', userid='ctp', password='ctppw')
    hols = hols.drop_duplicates()
    for k in key_words:
        hols = hols[hols['EXCHANGENAME'].str.contains(k)]
    return hols


def find_calendar_code_by_financial_centre(*key_words):
    sql_query = "SELECT UN_LOCODE, FINANCIALCENTER, ISOCOUNTRYCODE FROM DBO.COPP_FINANCIALCENTRES ORDER BY EVENTDATE"
    hols = read_sql(sql_query, server='DBCTPPROD.capstoneco.com', database='capstone', userid='ctp', password='ctppw')
    hols = hols.drop_duplicates()
    for k in key_words:
        hols = hols[hols['FINANCIALCENTER'].str.contains(k)]
    return hols
