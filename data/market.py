import numbers

import requests
import pandas as pd
import json
import re
from datetime import datetime
from ..data.refdata import get_instruments_info
from ..dates.utils import timestamp_to_datetime, vola_datetime_to_datetime, datetime_diff
from ..tradable.option import Option
from ..analytics.symbology import UNDERLYING_PRICING_ROOT_MAP

JP_DATA_NA_TEST_BOUND = 1e20

def is_call(text):
    lower_text = text.lower()
    if lower_text == 'call':
        return True
    elif lower_text == 'put':
        return False
    else:
        raise RuntimeError('Unknown input text ' + text)


def is_american(text):
    lower_text = text.lower()
    if lower_text == 'american':
        return True
    elif lower_text == 'european':
        return False
    else:
        raise RuntimeError('Unknown input text ' + text)


def get_vola_surface(underlying_id, ref_time, type="Equity"):
    """
    Get the Vola surface (and associated market data) using API
    @param underlying_id:
    @param ref_time:
    @return:
    """
    server_name = 'volar_surface_server'
    env_name = 'cpiceregistry'
    payload = {
        'server_name': server_name,
        'underlyingId': underlying_id,
        'refTime': ref_time,
    }
    if type == "Futures":
        payload["optionType"] = "itOptionOnFuture"

    if underlying_id in UNDERLYING_PRICING_ROOT_MAP:
        payload["pricingRoot"] = UNDERLYING_PRICING_ROOT_MAP[underlying_id]

    r = requests.get('http://' + env_name + ':6703/vol_surface', params=payload, timeout=60)
    # if len(r.content) == 0 or r.content[0] != '{':
    #     print('Error: ', r.content)
    import pyvolar as vola
    factory = vola.makeFactoryAnalytics()
    if type == 'Equity':
        try:
            surface = factory.makeVolSurfaceEquity(r.content, vola.FormatIO.JSON)
        except:
            print(f"Unable to load vola equity surface id: {underlying_id}, {r.content.decode()}")
            surface = None
    elif type == 'Futures':
        try:
            surface = factory.makeVolSurfaceFutures(r.content, vola.FormatIO.JSON)
        except:
            print(f"Unable to load vola future surface id: {underlying_id}, {r.content.decode()}")
            surface = None
    else:
        print(f"nknow vol surface type {type}")
        surface = None
    return surface


def get_intraday_vola_surface(underlying_id, dt, type="Equity"):
    """
    Get the Vola surface (and associated market data) using API
    @param underlying_id:
    @param ref_time:
    @return:
    """
    server_name = 'data_cache_volar_obj'
    env_name = 'cpiceregistry'
    ts = dt.strftime("%Y_%m_%d_%H_%M_%S")
    payload = {
        'server_name': server_name,
        "optionType": "itOption" if type == "Equity" else "itOptionOnFuture",
        'environment': 'CAPSTONE',
        'TIMESTAMP': ts,
        'underlyingId': underlying_id,
        'fitSource': "AUTO",
        'settlementStyle': 'ssCash',
        'exerciseStyle': 'esEuropean',
        'fitMethod': 'Volar',
    }
    if underlying_id in UNDERLYING_PRICING_ROOT_MAP:
        payload["pricingRoot"] = UNDERLYING_PRICING_ROOT_MAP[underlying_id]
    r = requests.get('http://' + env_name + ':6703/vol_surfaces.intraday', params=payload)
    import pyvolar as vola
    factory = vola.makeFactoryAnalytics()
    if type == 'Equity':
        try:
            surface = factory.makeVolSurfaceEquity(r.json()[0]["result"], vola.FormatIO.JSON)
        except:
            print(f"Unable to load vola equity surface id: {underlying_id}, {r.content.decode()}")
            surface = None
    elif type == 'Futures':
        try:
            surface = factory.makeVolSurfaceFutures(r.json()[0]["result"], vola.FormatIO.JSON)
        except:
            print(f"Unable to load vola future surface id: {underlying_id}, {r.content.decode()}")
            surface = None
    else:
        print(f"nknow vol surface type {type}")
        surface = None
    return surface


def extract_expiration_date_isoformat(_x):
    """
    Extract the date information only from the input datetime
    This means the date of the input datetime using the tz in the input datetime (if it has one)
    @param _x: input datetime string in iso format
    @return: extracted date datetime in iso format
    """
    _y = datetime.fromisoformat(_x);
    return datetime(_y.year, _y.month, _y.day).isoformat()


def get_option_chain(as_of_time, underlying_id):
    """
    Get the listed option chain
    The price is internal price at as_of_time, not exchange settlement price
    All datetime outputs are stored as string in iso format
    We don't assume timezone info (or named timezone info for that matter) in datetimes
    @param as_of_time:
    @param underlying_id:
    @return:
    """
    # option chain prices from vola using API
    server_name = 'volar_surface_server'
    env_name = 'cpiceregistry'
    payload = {
        'server_name': server_name,
        'underlyingId': underlying_id,
        'refTime': as_of_time,
        'format': 'json',
        'column_names': 'ID,RefTime,VALUE',
    }
    if underlying_id == 4015054:
        payload["pricingRoot"] = "CL"
    if underlying_id == 4015060:
        payload["optionType"] = "itOptionOnFuture"

    result = requests.get('http://' + env_name + ':6703/price', params=payload)
    try:
        json_obj = json.loads(result.content.decode("utf-8"))[0]
    except:
        print(f"Unable to load option chain for underlying id: {underlying_id}")
        return pd.DataFrame()

    option_ids = json_obj['ids']
    if len(option_ids) == 0:
        return pd.DataFrame()

    # get option contract details
    options_info = get_instruments_info(option_ids)

    df = pd.DataFrame.from_dict({
        'id': option_ids,
        'symbol': options_info['Symbol'],
        'underlying_id': underlying_id,
        # use time zone info to make the datetime with full info
        # assuming the utc offset in the MaturityDate column is
        # consistent with TimeZone column, using replace or astimezone should give same result
        # 'expiration': list(map(lambda x, y: timestamp_to_datetime(pd.Timestamp(x)).astimezone(pytz.timezone(y)).isoformat(),
        #                        options_info['MaturityDate'].values, options_info['TimeZone'].values)),
        'expiration': list(map(lambda x: timestamp_to_datetime(pd.Timestamp(x)).isoformat(),
                               options_info['MaturityDate'].values)),
        # the TimeZone only indicate the named time zone of the datetime in MaturityDate
        # not the exchange's time zone
        'time_zone': options_info['TimeZone'],
        'strike': options_info['Strike'],
        'call_put': options_info['PutCall'],
        'exercise_type': options_info['ExerciseStyle'],
        'currency': options_info['CCY'],
        'multiplier': options_info['Multiplier'],
        'price': list(map(lambda x: x[0], json_obj['values'])),
    })

    df['expiration_date'] = list(map(extract_expiration_date_isoformat, df['expiration']))
    df['as_of_time'] = as_of_time
    return df


def find_listed_options(expiration, strike, call_put, universe):
    """
    Find listed contracts using the given input specs.
    If an optional filter is None, it will not filter on that spec.
    All datetime outputs are stored as string in iso format
    We don't assume timezone info (or named timezone info for that matter) in datetimes
    @param expiration: we will only use the date information in this input
    @param strike:
    @param call_put:
    @param universe:
    @return:
    """
    df = universe

    df = df[df['strike'] == strike]
    df = df[df['expiration_date'] == extract_expiration_date_isoformat(expiration.isoformat())]
    df = df[df['call_put'] == call_put]
    return df


def bracket_options(strike, slice):
    strikes = list(sorted(set(slice['strike'].values)))
    lower_strikes = list(filter(lambda x: x <= strike, strikes))
    if len(lower_strikes) == 0:
        lower_option = None
    else:
        lower_strike = lower_strikes[-1]
        lower_option = slice[slice['strike'] == lower_strike]
        if lower_option.shape[0] > 1:
            lower_option = lower_option[lower_option['call_put'] == 'P']
    upper_strikes = list(filter(lambda x: x >= strike, strikes))
    if len(upper_strikes) == 0:
        upper_option = None
    else:
        upper_strike = upper_strikes[0]
        upper_option = slice[slice['strike'] == upper_strike]
        if upper_option.shape[0] > 1:
            upper_option = upper_option[upper_option['call_put'] == 'C']
    return lower_option, upper_option


def box_options(expiration, strike, universe):
    expiration_dates = list(sorted(set(universe['expiration_date'].values)))
    lower_expirations = list(filter(lambda x: datetime.fromisoformat(x) <= expiration, expiration_dates))
    if len(lower_expirations) == 0:
        lower_options = None, None
    else:
        lower_expiration = lower_expirations[-1]
        lower_options = bracket_options(strike, universe[universe['expiration_date'] == lower_expiration])
    upper_expirations = list(filter(lambda x: datetime.fromisoformat(x) >= expiration, expiration_dates))
    if len(upper_expirations) == 0:
        upper_options = None, None
    else:
        upper_expiration = upper_expirations[0]
        upper_options = bracket_options(strike, universe[universe['expiration_date'] == upper_expiration])
    return lower_options, upper_options


def find_nearest_expiration(expiration, universe, method='absolute'):
    if not isinstance(universe, pd.DataFrame):
        expiration_dates = list(
            map(lambda x: extract_expiration_date_isoformat(vola_datetime_to_datetime(x).isoformat()),
                universe.expiryTimes))
    else:
        expiration_dates = list(set(universe['expiration_date'].values))

    # We sort the string list of expiration dates here first.
    # This is to avoid unexpected behavior when two different expiries have the same distance to target date.
    # After sorted, we always select the one with shorter expiry.
    expiration_dates = sorted(expiration_dates)
    expiration_date = datetime.fromisoformat(extract_expiration_date_isoformat(expiration.isoformat()))

    if method == 'leq':
        expiration_dates = list(filter(lambda x: datetime.fromisoformat(x) <= expiration_date, expiration_dates))
    elif method == 'geq':
        expiration_dates = list(filter(lambda x: datetime.fromisoformat(x) >= expiration_date, expiration_dates))

    nearest_expiration_date = min(expiration_dates,
                                  key=lambda x: abs(datetime_diff(datetime.fromisoformat(x), expiration_date)))
    return datetime.fromisoformat(nearest_expiration_date)


def find_nearest_listed_options(expiration, strike, call_put, universe, other_filters=[], return_as_tradables=False,
                                select_by = 'strike', expiration_search_method='absolute',expiration_rule=None):
    """
    Find the listed option nearest to the specified expiration_date and strike
    it first finds the closest expiration date, and then find the closest strike in that slice
    @param expiration: we will only use the date information in this input
    @param strike:
    @param call_put:
    @param universe:
    @param select_by \in {'strike', 'delta'}
    @return:
    """
    if expiration_rule:
        universe = universe[universe['expiration_rule'] == expiration_rule]

    df = universe

    found = False
    while not found:
        nearest_expiration_date = find_nearest_expiration(expiration, df, method=expiration_search_method).isoformat()
        df_selected = df[(df['expiration_date'] == nearest_expiration_date) & (df['call_put'] == call_put)]
        for other_filter in other_filters:
            df_selected = other_filter(df_selected)
        if df_selected.empty:
            df = df[df['expiration_date'] != nearest_expiration_date]
            print(f'Nearest expiry is {nearest_expiration_date} but found no qualifying {call_put} at this expiry')
        else:
            found = True

    strikes = list(set(df_selected[select_by].values))
    nearest_strike = min(strikes, key=lambda x: abs(x - strike))
    df_selected = df_selected[df_selected[select_by] == nearest_strike]

    if return_as_tradables:
        tradables = []
        for option_to_trade in df_selected.to_dict('records'): #df_selected.to_dict('index').items():
            tradables.append(Option(option_to_trade['root'], option_to_trade['underlying'], option_to_trade['currency'],
                                    datetime.fromisoformat(option_to_trade['expiration']), option_to_trade['strike'],
                                    option_to_trade['call_put'] == 'C', option_to_trade['exercise_type'] == 'A',
                                    option_to_trade['contract_size'], option_to_trade['tz_name'],
                                    option_to_trade['ticker'],
                                    option_to_trade['expiration_rule']))
        return tradables
    else:
        return df_selected

def find_nearest_listed_options_intrday(expiration, strike, call_put, universe, other_filters=[], return_as_tradables=False,
                                select_by = 'strike', expiration_search_method='absolute'):
    """
    Find the listed option nearest to the specified expiration_date and strike
    it first finds the closest expiration date, and then find the closest strike in that slice
    @param expiration: we will only use the date information in this input
    @param strike:
    @param call_put:
    @param universe:
    @param select_by \in {'strike', 'delta'}
    @return:
    """
    df = universe

    found = False
    while not found:
        nearest_expiration_date = find_nearest_expiration(expiration, df, method=expiration_search_method).isoformat()
        df_selected = df[(df['expiration_date'] == nearest_expiration_date) & (df['call_put'] == call_put)]
        for other_filter in other_filters:
            df_selected = other_filter(df_selected)
        if df_selected.empty:
            df = df[df['expiration_date'] != nearest_expiration_date]
            print(f'Nearest expiry is {nearest_expiration_date} but found no qualifying {call_put} at this expiry')
        else:
            found = True

    strikes = list(set(df_selected[select_by].values))
    nearest_strike = min(strikes, key=lambda x: abs(x - strike))
    df_selected = df_selected[df_selected[select_by] == nearest_strike]

    if return_as_tradables:
        tradables = []
        for option_to_trade in df_selected.to_dict('records'): #df_selected.to_dict('index').items():
            tradables.append(Option(option_to_trade['stock_symbol'], option_to_trade['stock_symbol'], 'USD',
                                    datetime.fromisoformat(option_to_trade['expiration_date']), option_to_trade['strike'],
                                    option_to_trade['call_put'] == 'C', option_to_trade['style'] == 'A',
                                    None, None,
                                    option_to_trade['option_symbol']))
        return tradables
    else:
        return df_selected


def get_jp_swaption_tenor(key):
    if key[1] in ['Day', 'Days', 'D', 'd']:
        return key[0].lstrip('0') + 'd'
    elif key[1] in ['Week', 'Weeks', 'W', 'w']:
        return key[0].lstrip('0') + 'w'
    elif key[1] in ['Month', 'Months', 'M', 'm']:
        return key[0].lstrip('0') + 'm'
    elif key[1] in ['Year', 'Years', 'Y', 'y']:
        return key[0].lstrip('0') + 'y'
    else:
        raise RuntimeError('Cannot get tenor from key ' + key)


def get_expiry_in_year_from_string(tenor):
    tenor_elements = re.match(r'(\d{1,2})(Y|M|W|D|y|m|w|d)', tenor).groups()
    tenor_years = get_expiry_in_year(tenor_elements[0], tenor_elements[1])
    return tenor_years


def get_expiry_in_year(number_str, unit_str):
    if unit_str in ['Day', 'Days', 'D', 'd']:
        return round(float(number_str) / 365, 10)
    elif unit_str in ['Week', 'Weeks', 'W', 'w']:
        return round(float(number_str) * 7 / 365, 10)
    elif unit_str in ['Month', 'Months', 'M', 'm']:
        return round(float(number_str) / 12.0, 10)
    elif unit_str in ['Year', 'Years', 'Y', 'y']:
        return round(float(number_str), 10)
    else:
        raise RuntimeError('Cannot get expiry from key ' + number_str + unit_str)


def get_jp_swaption_expiry(key):
    return get_expiry_in_year(key[2], key[3])


def get_jp_swaption_strike(key):
    return float(key[5])
    # if key[4] == 'ATMF':
    #     return 0.0
    # elif key[4].startswith('OTMF'):
    #     return float(key[6])
    # elif key[4].startswith('ITMF'):
    #     return -float(key[6])
    # else:
    #     raise RuntimeError('Cannot get expiry from key ' + key)


def read_jp_rates(data=None, rate_type='OIS', pattern_override = None ):
    if data is None:
        data = r'/var/ctp/data_cache/rates/2023-02-22 new_rates_data.csv'
    if isinstance(data, str):
        data = pd.read_csv(data)

    if rate_type == 'OIS':
        # pattern = r'OIS (\d{1,2}) (Day|Days|Week|Weeks|Month|Months|Year|Years) Rate'
        # pattern = r'\(DB\(FCRV,SWAP,ZERO,(\d{1,2})(Y|M|W|D),RT_MID\)'
        # pattern = r'\(DB\(MTE,usd/ois/fomc/(\d{1,2})(y|m|w|d)/rate\)'
        pattern = r'MTE_usd/ois/fomc/(\d{1,2})(y|m|w|d)/rate'
    elif rate_type == 'SOFR':
        # pattern = r'SOFR (\d{1,2}) (Day|Days|Week|Weeks|Month|Months|Year|Years) Mid Rate'
        # pattern = r'\(DB\(FCRV,SOFR,SWAP,ZERO,(\d{1,2})(Y|M|W|D),RT_MID\)'
        pattern = r'FCRV_SOFR_SWAP_ZERO_(\d{1,2})(Y|M|W|D)_RT_MID'
    elif rate_type == 'SWAP':
        # pattern = r'\(DB\(FDER,PARSWAP,(\d{1,2})(Y|M|W|D),RT_MID\)'
        pattern = r'FDER_(PAR|LIB)SWAP_(\d{1,2})(Y|M|W|D)_RT_MID'
    else:
        raise RuntimeError('Unknown rate type ' + rate_type)

    if pattern_override is not None:
        pattern = pattern_override

    selected_cols = ['tstamp'] + list(filter(lambda x: re.match(pattern, x) is not None, data.columns))
    selected_data = data[selected_cols].rename(
        columns=lambda x: 'tstamp' if x == 'tstamp' else tuple(re.match(pattern, x).groups()))
    ois_rates = {}
    for record in selected_data.to_dict('records'):
        dt = datetime.strptime(str(record['tstamp'])[:10], "%Y-%m-%d")
        rates = {}
        for k, v in record.items():
            if k != 'tstamp':
                expiry = get_expiry_in_year(k[-2], k[-1])
                if isinstance(float(v), numbers.Number) and float(v) < JP_DATA_NA_TEST_BOUND:
                    rates.setdefault(expiry, float(v))
        ois_rates[dt] = rates
    return ois_rates


def read_jp_swaption_atmf_yields(data=None):
    # if data is None:
    #     data = r'/var/ctp/data_cache/rates/2023-02-21 rates_data_new_header.csv'
    # if isinstance(data, str):
    #     data = pd.read_csv(data)

    # pattern = r'(\d{1,2}) (Week|Weeks|Month|Months|Year|Years) (\d{1,2}) (Week|Weeks|Month|Months|Year|Years) Payer ATMF Strike Yield'
    # pattern = r'\(DB\(FDER,SWAPTION,(\d{1,2})(Y|M|W|D),(\d{1,2})(Y|M|W|D),3PT,Payer,ATMF,0,STRIKE\)'
    pattern = r'FDER_SWAPTION_(\d{1,2})(Y|M|W|D)_(\d{1,2})(Y|M|W|D)_3PT_Payer_ATMF_0_STRIKE'
    selected_cols = ['tstamp'] + list(filter(lambda x: re.match(pattern, x) is not None, data.columns))
    selected_data = data[selected_cols].rename(
        columns=lambda x: 'tstamp' if x == 'tstamp' else tuple(re.match(pattern, x).groups()))
    atmf_yields = {}
    for record in selected_data.to_dict('records'):
        dt = datetime.strptime(str(record['tstamp'])[:10], "%Y-%m-%d")
        yields = {}
        for k, v in record.items():
            if k != 'tstamp':
                tenor = get_jp_swaption_tenor(k)
                expiry = get_jp_swaption_expiry(k)
                if isinstance(v, numbers.Number) and v < JP_DATA_NA_TEST_BOUND:
                    yields.setdefault(tenor, {}).setdefault(expiry, v)
        atmf_yields[dt] = yields
    return atmf_yields


def remove_slice_with_too_few_strikes(cubes, min_num_strikes=3):
    dts_to_delete = []
    for dt, cube in cubes.items():
        tenors_to_delete = []
        for tenor, surface in cube.items():
            expiries_to_delete = []
            for expiry, slice in surface.items():
                if len(list(slice.keys())) < min_num_strikes:
                    expiries_to_delete.append(expiry)
            for expiry in expiries_to_delete:
                del surface[expiry]
            if len(cube[tenor]) == 0:
                tenors_to_delete.append(tenor)
        for tenor in tenors_to_delete:
            del cube[tenor]
        if len(cubes[dt]) == 0:
            dts_to_delete.append(dt)
    for dt in dts_to_delete:
        del cubes[dt]


def read_jp_swaption_vol_cubes(data, rate_data, strike_relative_to_atmf=True, verbose=False):
    # pattern = r'(\d{1,2}) (Week|Weeks|Month|Months|Year|Years) (\d{1,2}) (Week|Weeks|Month|Months|Year|Years) Payer (ATMF|OTMF,|ITMF,) ((\d{1,3}) bps )?Implied BP Vol \(Annualized\)'
    # pattern = r'\(DB\(FDER,SWAPTION,(\d{1,2})(Y|M|W|D),(\d{1,2})(Y|M|W|D),STOCH,Payer,(OTMF|ATMF),(-?\d{1,3}),ABPVOL\)'
    pattern = r'FDER_SWAPTION_(\d{1,2})(Y|M|W|D)_(\d{1,2})(Y|M|W|D)_STOCH_Payer_(OTMF|ATMF)_(-?\d{1,3})_ABPVOL'
    selected_cols = ['tstamp'] + list(filter(lambda x: re.match(pattern, x) is not None, data.columns))
    selected_data = data[selected_cols].rename(
        columns=lambda x: 'tstamp' if x == 'tstamp' else tuple(re.match(pattern, x).groups()))

    if not strike_relative_to_atmf:
        atmf_yields = read_jp_swaption_atmf_yields(rate_data)

    cubes = {}
    for record in selected_data.to_dict('records'):
        dt = datetime.strptime(str(record['tstamp'])[:10], "%Y-%m-%d")
        # dt = datetime.strptime(str(record['date']), '%Y%m%d')
        cube = {}
        for k, v in record.items():
            if k != 'tstamp':
                if isinstance(v, numbers.Number) and v < JP_DATA_NA_TEST_BOUND:
                    tenor = get_jp_swaption_tenor(k)
                    expiry = get_jp_swaption_expiry(k)
                    if strike_relative_to_atmf:
                        strike = get_jp_swaption_strike(k) / 100
                        cube.setdefault(tenor, {}).setdefault(expiry, {}).setdefault(strike, v / 100.0)
                    else:
                        atmf_yield = atmf_yields.get(dt, {}).get(tenor, {}).get(expiry, None)
                        if atmf_yield is None:
                            if verbose:
                                print(
                                    f'No atmf yield data found for {dt.strftime("%Y-%m-%d")} {tenor} {str(expiry)} -- no vol is read')
                        else:
                            strike = atmf_yields[dt][tenor][expiry] + get_jp_swaption_strike(k) / 100
                            cube.setdefault(tenor, {}).setdefault(expiry, {}).setdefault(strike, v / 100.0)
        cubes[dt] = cube

    remove_slice_with_too_few_strikes(cubes, min_num_strikes=3)
    return cubes
