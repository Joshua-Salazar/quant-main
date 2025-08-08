from ..analytics.constants import EPSILON
from ..analytics.swaptions import df_interpolate
from ..analytics.options import BlackScholes
from ..analytics.utils import interpolate_curve
from ..dates.utils import datetime_to_tenor
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
import numpy as np


class FXOptionSABRBSValuer(IValuer):
    def __init__(self):
        pass

    def price(self, option: Option, market: Market, calc_types='price', return_struc=False, **kwargs):
        assert isinstance(option, Option)
        dt = market.get_base_datetime()
        vol_surface = market.get_fx_sabr_vol_surface(option.underlying)

        if 'spots' in kwargs:
            spots = kwargs['spots']
        else:
            spots = 1. / vol_surface.spot if vol_surface.is_inversed else vol_surface.spot

        if 'vols' in kwargs:
            vols = kwargs['vols']
        else:
            # sabr vol
            vols = vol_surface.get_vol(
                option.expiration,
                1 / option.strike if vol_surface.is_inversed else option.strike
            )

        if 'strikes' in kwargs:
            strikes = kwargs['strikes']
        else:
            strikes = option.strike

        if option.is_expired(market):
            TTM = 0
            disc = 1
            forward_factor = 1
        else:
            # discount in term currency
            df_curve = {dt: market.get_spot_rates(option.underlying[-3:], 'DEFAULT').data_dict}

            TTM = (option.expiration.date() - dt.date()).days / 360
            spot_original = vol_surface.spot
            fwd_original = interpolate_curve(vol_surface.forwards, TTM, flat_extrapolate_lower=True, flat_extrapolate_upper=True)
            forward_factor = spot_original / fwd_original if vol_surface.is_inversed else fwd_original / spot_original

            # interpolate df
            if TTM < EPSILON:
                disc = 1.0
            else:
                disc = df_interpolate(df_curve, dt, option.expiration,
                                      flat_upper_expiry_extrapolation=True,
                                      flat_lower_expiry_extrapolation=True,
                                      compounding_convention='annual')

        # first results in term currency
        results = BlackScholes(np.array(strikes), option.expiration, option.is_call, option.is_american, dt,
                               spot=np.array(spots),
                               fwd=np.array(spots) * forward_factor,
                               vol=np.array(vols),
                               disc=disc,
                               TTM=TTM)

        # then results in option premium currency
        if option.underlying[-3:] != option.currency:
            fx_rate = vol_surface.spot if vol_surface.is_inversed else 1 / vol_surface.spot
            results = {k: fx_rate * v for k, v in results.items()}

        results["revega"] = results["vega"] * results["vol"]
        if return_struc:
            return results
        else:
            return valuer_utils.return_results_based_on_dictionary(calc_types, results)

