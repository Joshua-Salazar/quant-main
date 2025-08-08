from ..dates.utils import coerce_timezone, add_business_days
from ..tradable.future import Future


def find_nth_future(market, root, n, expiry_reference, expiry_offset):
    dt = market.base_datetime

    def expiry_filter(_x):
        dt_v, this_expriy = coerce_timezone(dt, _x[expiry_reference])
        if root == _x['root'] and dt_v <= add_business_days(this_expriy, -expiry_offset):
            return True
        else:
            return False

    future_universe = list(market.get_future_universe(root).values())
    future_universe = sorted(future_universe, key=lambda x: x[expiry_reference])
    future_universe = list(filter(expiry_filter, future_universe))

    future_ref = future_universe[n - 1]

    future = Future(future_ref['root'], future_ref['currency'],
                    future_ref['last tradable date'], future_ref['exchange'],
                    future_ref['ticker'])

    return future
