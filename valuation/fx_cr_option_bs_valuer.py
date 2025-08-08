from ..analytics.constants import EPSILON
from ..analytics.swaptions import df_interpolate
from ..analytics.bs import BlackScholes
from ..dates.utils import datetime_to_tenor, add_tenor, count_business_days, coerce_timezone
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer


class FXOptionBSValuer(IValuer):
    def __init__(self, discount_curve_name, trade_date_info=None):
        self.discount_curve_name = discount_curve_name
        self.trade_date_info = trade_date_info

    def price(self, option: Option, market: Market, calc_types='price', return_struc=False, vol_override=None, **kwargs):
        dt = market.get_base_datetime()
        if len( [ x for x in list(market.storage.keys()) if 'CRVol' in x ] ) > 0 :
            vol_surface = market.storage['CRVolatility.' + option.underlying][option.underlying]
            df_curve = market.storage['SpotRateCurve.' + option.currency + '.' + self.discount_curve_name].data_dict
            spot = vol_surface.spots
            is_CR = True
        else:
            is_CR = False
            fx_pair = option.underlying
            if hasattr( market, 'storage' ):
                try:
                    spot = market.storage['FXSpot.'+fx_pair][dt][fx_pair]
                except:
                    spot = market.get_fx_spot(fx_pair)

                try:
                    vol_surface = market.storage['FXVolatility.'+fx_pair][dt][fx_pair]
                except:
                    vol_surface = market.get_fx_vol_surface(fx_pair)

                try:
                    df_curve = market.storage['SpotRateCurve.' + option.currency + '.' + self.discount_curve_name]
                except:
                    df_curve = market.get_spot_rates(option.currency, self.discount_curve_name)
                if not isinstance( df_curve, dict ):
                    df_curve = df_curve.data_dict
            else:
                spot = market.get_fx_spot(fx_pair)
                vol_surface = market.get_fx_vol_surface(fx_pair)
                df_curve = market.get_spot_rates(option.currency, self.discount_curve_name).data_dict

        if is_CR:
            dt1, dt2 = coerce_timezone(dt, option.expiration)
            TTM = ( count_business_days(dt1, dt2, self.trade_date_info['holidays'] ) - 1) / 252
        else:
            # temp fix for bug when pricing 9M options
            TTM = min( datetime_to_tenor(option.expiration, dt), 0.75 )

        # TODO: re-examine the vol fitting and interpolation method

        # interpolate forward
        forward = vol_surface.get_forward( TTM if is_CR else option.expiration )

        if vol_override is not None:
            vol = vol_override
        else:
            if self.trade_date_info is not None and option.name() in self.trade_date_info and \
                    dt == self.trade_date_info[option.name()]['trade_date']:
                vol = self.trade_date_info[option.name()]['trade_date_vol']
            else:
                vol = vol_surface.get_vol( TTM if is_CR else add_tenor(dt, TTM), option.strike )

        # interpolate df
        if TTM < EPSILON:
            df = 1.0
        else:
            df = df_interpolate(df_curve, dt, option.expiration,
                                flat_upper_expiry_extrapolation=True,
                                flat_lower_expiry_extrapolation=True,
                                compounding_convention='annual',
                                backfill_rate = True )
        if is_CR:
            results = BlackScholes(option.strike/10000, option.expiration, option.is_call, option.is_american, dt, forward/10000,
                                   forward/10000, vol, df, TTM )
        else:
            results = BlackScholes(option.strike, option.expiration, option.is_call, option.is_american, dt, spot, forward, vol, df)
        if return_struc:
            return results
        else:
            return valuer_utils.return_results_based_on_dictionary(calc_types, results)

