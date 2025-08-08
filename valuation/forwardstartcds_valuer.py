from ..analytics.constants import EPSILON
from ..analytics.swaptions import df_interpolate, atmf_yields_interpolate
from ..analytics.bs import BlackScholes
from ..dates.utils import datetime_to_tenor, add_tenor, count_business_days, coerce_timezone
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..tradable.forwardstartcds import ForwardStartCDS
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
import math


class ForwardStartCDSSytheticValuer(IValuer):
    def __init__(self, discount_curve_name):
        self.discount_curve_name = discount_curve_name

    def price(self, fwdstartcds: ForwardStartCDS, market: Market, calc_types='price', return_struc=False, vol_override=None, **kwargs):
        dt = market.get_base_datetime()
        vol_surface = market.get_cr_vol_surface(fwdstartcds.spread)
        df_curve = {dt: market.get_spot_rates(fwdstartcds.currency, self.discount_curve_name).data_dict}
        spot = vol_surface.spots

        dt1, dt2 = coerce_timezone(dt, fwdstartcds.expiration)
        TTM = datetime_to_tenor(dt2, dt1)

        # interpolate forward
        forward = vol_surface.get_forward(TTM)

        if vol_override is not None:
            vol = vol_override
        else:
            vol = vol_surface.get_vol_interpolation(TTM, fwdstartcds.strike)

        # interpolate df
        if TTM < EPSILON:
            df = 1.0
        else:
            df = df_interpolate(df_curve, dt, fwdstartcds.expiration,
                                flat_upper_expiry_extrapolation=True,
                                flat_lower_expiry_extrapolation=True,
                                compounding_convention='annual',
                                backfill_rate=True)

        results_call = BlackScholes(fwdstartcds.strike/10000, fwdstartcds.expiration, True, False, dt, forward/10000, forward/10000, vol, df, TTM)
        results_put = BlackScholes(fwdstartcds.strike/10000, fwdstartcds.expiration, False, False, dt, forward / 10000, forward / 10000, vol, df, TTM)

        # this improves the original pricer Henry wrote, but still a very rough pricer overall (no default probability no rate credit convexity etc)
        # needs to use the pricing library's pricer when they are exposed
        convPayPeriod = 4
        swap_tenor = 5
        fwd_rate_curve = {dt: market.get_forward_rates(fwdstartcds.currency, self.discount_curve_name)}
        atmf = atmf_yields_interpolate(fwd_rate_curve, df_curve, dt, f"{str(swap_tenor)}Y", TTM)
        annuity = (1 - (1 / ((1 + (atmf / 100 / convPayPeriod)) ** (swap_tenor * convPayPeriod)))) / (atmf / 100)

        results = {}
        for k in list(results_call.keys()):
            if k not in ['fwd', 'vol', 'spot']:
                results[k] = (results_call[k] - results_put[k]) * annuity
            else:
                assert abs(results_call[k] - results_put[k]) < EPSILON
                results[k] = results_call[k]

        assert abs(results['price'] - df * (forward/10000 - fwdstartcds.strike/10000) * annuity) < EPSILON
        assert abs(results['delta'] - df * annuity) < EPSILON or TTM < EPSILON
        assert abs(results['vega']) < EPSILON
        assert abs(results['gamma']) < EPSILON

        if return_struc:
            return results
        else:
            return valuer_utils.return_results_based_on_dictionary(calc_types, results)
