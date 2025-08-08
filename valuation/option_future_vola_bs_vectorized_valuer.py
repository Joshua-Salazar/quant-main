from datetime import datetime
from ..analytics.symbology import option_underlying_type_from_ticker
from ..dates.utils import datetime_diff
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..tradable.future import Future
from ..interface.ivaluer import IValuer
from ..analytics.options import BlackScholes
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
import numpy as np

from ..valuation import valuer_utils

VOLA_FACTOR_MAP = {
    vola.PricerResultsType.VEGA: 1/100,
    vola.PricerResultsType.THETA: 1/365,
}


class OptionFutureVolaBSVectorizedValuer(IValuer):

    def __init__(self):
        pass

    def price(self, option_or_future, market: Market, calc_types='price', **kwargs):
        date = market.get_base_datetime()

        if 'spots' in kwargs:
            spots = kwargs['spots']
        else:
            if isinstance(option_or_future, Future) and option_underlying_type_from_ticker(option_or_future.root) == 'future':
                spots = market.get_future_price(option_or_future.root, option_or_future.expiration)
            else:
                spots = market.get_underlying_price(option_or_future)
            # surface = market.get_vol_surface(underlying=option_or_future.underlying)
            # spots = surface.get_spot(base_date=date)

        if isinstance(option_or_future, Option):
            if 'vols' in kwargs:
                vols = kwargs['vols']
            else:
                vols = market.get_vol(underlying=option_or_future.root, expiry_dt=option_or_future.expiration, strike=option_or_future.strike)

            if 'strikes' in kwargs:
                strikes = kwargs['strikes']
            else:
                strikes = option_or_future.strike

        if option_or_future.is_expired(market):
            TTM = 0
            disc = 1
            forward_factor = 1
            if isinstance(option_or_future, Option):
                if vols is None:
                    vols = 0.001
        else:
            # can only price a concrete option
            assert isinstance(option_or_future.expiration, datetime)

            TTM = datetime_diff(option_or_future.expiration, date).days / 365

            if isinstance(option_or_future, Option) and isinstance(option_or_future.underlying, Future):
                forward_factor = 1
            elif isinstance(option_or_future, Future) \
                    and option_underlying_type_from_ticker(option_or_future.root, market) == 'future':
                forward_factor = 1
            else:
                surface = market.get_vol_surface(underlying=option_or_future.root)
                spot_original = surface.get_spot(base_date=date)
                fwd_original = surface.get_forward(option_or_future.expiration)
                forward_factor = fwd_original / spot_original

            if isinstance(option_or_future, Option):
                r = market.get_discount_rate(option_or_future.root, option_or_future.expiration)
                disc = np.exp(-r * TTM)
                if vols is None:
                    vols = market.get_vol(underlying=option_or_future.root, expiry_dt=option_or_future.expiration, strike=option_or_future.strike)

        if isinstance(option_or_future, Option):
            results = BlackScholes(np.array(strikes), option_or_future.expiration, option_or_future.is_call, option_or_future.is_american, date,
                                   np.array(spots), np.array(spots) * forward_factor, np.array(vols), disc=disc, TTM=TTM)
            results = valuer_utils.return_results_based_on_dictionary(calc_types, results)

            if isinstance(calc_types, list):
                return [x * option_or_future.contract_size for x in results]
            else:
                return results * option_or_future.contract_size
        else:
            # note for futures like VIX (or other market where the option has future underlyings),
            # the delta is 1, that is to be consistent with the delta on the options on these markets
            prices = forward_factor * np.array(spots)
            deltas = prices / np.array(spots)
            results = {
                'price': prices,
                'delta': deltas,
            }
            results = valuer_utils.return_results_based_on_dictionary(calc_types, results)
            return results
