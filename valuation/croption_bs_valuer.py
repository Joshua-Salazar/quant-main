from ..analytics.constants import EPSILON
from ..analytics.swaptions import df_interpolate, atmf_yields_interpolate
from ..analytics.bs import BlackScholes
from ..dates.utils import datetime_to_tenor, add_tenor, count_business_days, coerce_timezone
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
import math


class CROptionBSValuer(IValuer):
    def __init__(self, discount_curve_name):
        self.discount_curve_name = discount_curve_name

    def price(self, option: Option, market: Market, calc_types='price', return_struc=False, vol_override=None, **kwargs):
        dt = market.get_base_datetime()
        vol_surface = market.get_cr_vol_surface(option.underlying)
        df_curve = {dt: market.get_spot_rates(option.currency, self.discount_curve_name).data_dict}
        spot = vol_surface.spots

        dt1, dt2 = coerce_timezone(dt, option.expiration)
        TTM = datetime_to_tenor(dt2, dt1)

        # interpolate forward
        forward = vol_surface.get_forward(TTM)

        if vol_override is not None:
            vol = vol_override
        else:
            vol = vol_surface.get_vol_interpolation(TTM, option.strike)

        # interpolate df
        if TTM < EPSILON:
            df = 1.0
        else:
            df = df_interpolate(df_curve, dt, option.expiration,
                                flat_upper_expiry_extrapolation=True,
                                flat_lower_expiry_extrapolation=True,
                                compounding_convention='annual',
                                backfill_rate=True)
        results = BlackScholes(option.strike/10000, option.expiration, option.is_call, option.is_american, dt, forward/10000, forward/10000, vol, df, TTM)

        # this improves the original pricer Henry wrote, but still a very rough pricer overall (no default probability no rate credit convexity etc)
        # needs to use the pricing library's pricer when they are exposed
        convPayPeriod = 4
        swap_tenor = 5
        fwd_rate_curve = {dt: market.get_forward_rates(option.currency, self.discount_curve_name)}
        atmf = atmf_yields_interpolate(fwd_rate_curve, df_curve, dt, f"{str(swap_tenor)}Y", TTM)
        annuity = (1 - (1 / ((1 + (atmf / 100 / convPayPeriod)) ** (swap_tenor * convPayPeriod)))) / (atmf / 100)

        for k in list(results.keys()):
            if k not in ['fwd', 'vol', 'spot']:
                results[k] *= annuity

        if return_struc:
            return results
        else:
            return valuer_utils.return_results_based_on_dictionary(calc_types, results)
