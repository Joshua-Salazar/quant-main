from ..dates.utils import add_business_days, MAX_DATETIME, coerce_timezone, datetime_diff
from ..interface.ivaluer import IValuer
from ..tradable.future import Future
from ..valuation.future_data_valuer import FutureDataValuer
from ..valuation.future_valuer import FutureValuer
from datetime import datetime


def find_front_future(market, future_root, expiry_offset_reference, expiry_offset):
    dt = market.base_datetime
    next_future_ref = None
    next_future_expiry = MAX_DATETIME
    for k, v in market.get_future_universe(future_root).items():
        dt_v, this_expriy = coerce_timezone(dt, v[expiry_offset_reference])
        if future_root == v['root'] and dt_v <= add_business_days(this_expriy, -expiry_offset) \
                and this_expriy < next_future_expiry:
            next_future_ref = v
            next_future_expiry = this_expriy

    next_future = Future(next_future_ref['root'], next_future_ref['currency'],
                         next_future_ref['last tradable date'], next_future_ref['exchange'],
                         next_future_ref['ticker'])

    return next_future


def find_future_with_expiration_date(market, future_root, expiry_reference, expiration_date, method='exact'):
    if method == 'exact':
        selected_expiration_date = expiration_date
    elif method == 'nearest':
        selected_expiration_date = min([v[expiry_reference] for k, v in market.get_future_universe(future_root).items()], key=lambda x: abs((x.date() - expiration_date.date()).days))
    else:
        raise RuntimeError(f"unknown date selection method {method}")

    next_future_ref = None
    for k, v in market.get_future_universe(future_root).items():
        if future_root == v['root'] and v[expiry_reference].date() == selected_expiration_date.date():
            next_future_ref = v
            break

    next_future = Future(next_future_ref['root'], next_future_ref['currency'],
                         next_future_ref['last tradable date'], next_future_ref['exchange'],
                         next_future_ref['ticker'])

    return next_future


class FutureHedgeValuer(IValuer):
    """
    Valuer for futures used as a hedge. In this case, we use on the run future for non-vix underlying and
    use the same option underlying for vix option. Delta will be different as vix future has delta 1 in this case
    """
    def price(self, tradable, market, calc_types='price', **kwargs):
        is_vix = tradable.root in ["VIX Index", "V2X Index"]
        if is_vix:
            return FutureValuer().price(tradable, market, calc_types)
        else:
            return FutureDataValuer().price(tradable, market, calc_types)
