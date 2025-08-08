from datetime import datetime, time
from ..tradable.option import Option
from ..data.market import find_nearest_listed_options
from ..dates.utils import tenor_to_datetime, set_timezone, bdc_adjustment
from typing import Union


def get_expiry(expiration: str, base_date, tz_name, expiration_time=time(16, 0)):
    expiry = bdc_adjustment(set_timezone(tenor_to_datetime(expiration, base_date), tz_name), 'previous')
    expiry = datetime.combine(expiry.date(), expiration_time)
    return expiry


def create_fixed_expiry_strike_option(expiration: Union[datetime, str], tz_name: str, strike: float, strike_type: str,
                                      is_call: bool, contract_size: float, root: str, underlying: str, currency: str,
                                      is_american: bool,
                                      base_date, spot, expiration_time=time(16, 0),
                                      use_listed_options=False, listed_options_universe=None, is_ivol_universe=False, surface=None):
    if isinstance(expiration, datetime):
        expiry = expiration
    else:
        expiry = get_expiry(expiration, base_date, tz_name, expiration_time)

    if use_listed_options:

        # for listed options, we expected surface expiries should exist in option universe always
        # but we found missing expiry from surface, e.g. VIX surface on 14/02/2023
        # to work around, we place check here to ensure expiry exist in surface.
        if surface is not None:
            expiry = surface.find_nearest_expiration(expiry)
        listed_option = find_nearest_listed_options(
            expiry,
            strike if strike_type == 'absolute' else spot * strike,
            ('C' if is_call else 'P') if is_ivol_universe else ('CALL' if is_call else 'PUT'),
            listed_options_universe
        )
        assert len(listed_option) >= 1
        # if there are two options we use am expiry
        if len(listed_option) > 1:
            listed_option = listed_option.sort_values('expiration')
        listed_option = listed_option.to_dict('records')[0]
        # check consistency of contract size
        listed_option_contract_size = listed_option['contract_size'] if is_ivol_universe else listed_option['multiplier']
        # Todo: check why time zone missing from ivol
        listed_option_time_zone = ("America/New_York" if listed_option['tz_name'] == "" else listed_option['tz_name']) \
            if is_ivol_universe else listed_option['time_zone']
        if contract_size != listed_option_contract_size:
            print(f"Found inconsistent size {contract_size} vs list size {listed_option_contract_size} for {underlying}")
        return Option(root, underlying, currency,
                      datetime.fromisoformat(listed_option['expiration']),
                      listed_option['strike'],
                      is_call,
                      (False if listed_option['exercise_type'] == 'E' else True) if is_ivol_universe
                      else (True if listed_option['exercise_type'] == 'American' else False),
                      listed_option_contract_size, listed_option_time_zone,
                      listed_ticker=listed_option['ticker'] if is_ivol_universe else listed_option['symbol'])
    else:
        return Option(root, underlying, currency, expiry,
                      strike if strike_type == 'absolute' else spot * strike,
                      is_call, is_american, contract_size, tz_name)