from datetime import datetime


from ..data.market import extract_expiration_date_isoformat
from ..dates.utils import datetime_diff

from ..tradable.option import Option


def find_nearest_expiration(expiration, options_list):
    expiration_dates = [x['expiration_date'] for x in options_list]
    expiration_dates = list(sorted(set(expiration_dates)))
    expiration_date = datetime.fromisoformat(extract_expiration_date_isoformat(expiration.isoformat()))
    nearest_expiration_date = min(expiration_dates,
                                  key=lambda x: abs(datetime_diff(datetime.fromisoformat(x), expiration_date)))
    return datetime.fromisoformat(nearest_expiration_date)


def find_nearest_listed_options(expiration, strike, call_put, options_list, other_filters=[], return_as_tradables=False):
    """
    Find the listed option nearest to the specified expiration_date and strike
    it first finds the closest expiration date, and then find the closest strike in that slice
    @param expiration: we will only use the date information in this input
    @param strike:
    @param call_put:
    @param universe:
    @return:
    """

    found = False
    while not found:
        nearest_expiration_date = find_nearest_expiration(expiration, options_list).isoformat()
        options_selected = list(filter(lambda x: x['expiration_date'] == nearest_expiration_date and x['call_put'] == call_put, options_list))

        for other_filter in other_filters:
            options_selected = other_filter(options_selected)

        if not options_selected:
            options_list = list(filter(lambda x: x['expiration_date'] != nearest_expiration_date, options_list))
            print(f'Nearest expiry is {nearest_expiration_date} but found no qualifying option at this expiry')
        else:
            found = True

    strikes = list(set(list(map(lambda x: x['strike'], options_selected))))
    nearest_strike = min(strikes, key=lambda x: abs(x - strike))
    options_selected = list(filter(lambda x: x['strike'] == nearest_strike, options_selected))

    if return_as_tradables:
        tradables = []
        for option_to_trade in options_selected:
            tradables.append(Option(option_to_trade['root'], option_to_trade['underlying'], option_to_trade['currency'],
                                    datetime.fromisoformat(option_to_trade['expiration']), option_to_trade['strike'],
                                    option_to_trade['call_put'] == 'C', option_to_trade['exercise_type'] == 'A',
                                    option_to_trade['contract_size'], option_to_trade['tz_name'],
                                    option_to_trade['ticker']))
        return tradables
    else:
        return options_selected
