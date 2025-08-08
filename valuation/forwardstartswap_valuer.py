from ..analytics.swaptions import fwdstartswapcalc, atmf_yields_interpolate, rate_interpolate, atmf_yields_interpolate_old, fwdstartswapcalc_old
from ..data.market import get_expiry_in_year
from ..infrastructure.market import Market
from ..tradable.forwardstartswap import ForwardStartSwap
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
from datetime import datetime
import re


class ForwardStartSwapValuer(IValuer):
    def __init__(self, discount_rate_curve_type='OIS'):
        self.discount_rate_curve_type = discount_rate_curve_type

    def price(self, fwdstartswap: ForwardStartSwap, market: Market, calc_types='price', **kwargs):
        dt = market.get_base_datetime()

        spot_rate_curve = {dt: market.get_spot_rates(fwdstartswap.currency, fwdstartswap.curve).data_dict}
        df_curve = {dt: market.get_spot_rates(fwdstartswap.currency, self.discount_rate_curve_type).data_dict}

        if fwdstartswap.style == 'Receiver':
            rps = 'r'
        elif fwdstartswap.style == 'Payer':
            rps = 'p'
        else:
            raise RuntimeError('Unknown swap style ' + fwdstartswap.style)

        # remove timezone
        expiry = datetime.combine(fwdstartswap.expiration.date(), datetime.min.time())

        rate_curve = {dt: market.get_forward_rates(fwdstartswap.currency, fwdstartswap.curve)}
        atmf = atmf_yields_interpolate(rate_curve, spot_rate_curve, dt, fwdstartswap.tenor, expiry)

        results = fwdstartswapcalc(dt, None, None, df_curve, fwdstartswap.currency,
                                   expiry, fwdstartswap.tenor, fwdstartswap.strike, rps, atmf_override=atmf)

        return valuer_utils.return_results_based_on_dictionary(calc_types, {
            'price': results['pv'],
            'delta': results['delta'],
            'gamma': results['gamma'],
            'vega': 0,
            'forward_rate': results['fwd'],
            'theta': results['theta'],
        })


class ForwardStartSwapValuerOld(IValuer):
    def __init__(self, spot_rate_curve_type='SWAP', discount_rate_curve_type='OIS', forward_rate_curve_type='SWAPTION'):
        self.spot_rate_curve_type = spot_rate_curve_type
        self.discount_rate_curve_type = discount_rate_curve_type
        self.forward_rate_curve_type = forward_rate_curve_type

    def price(self, fwdstartswap: ForwardStartSwap, market: Market, calc_types='price', **kwargs):
        dt = market.get_base_datetime()

        spot_rate_curve = {dt: market.get_spot_rates(fwdstartswap.currency, self.spot_rate_curve_type).data_dict}
        df_curve = {dt: market.get_spot_rates(fwdstartswap.currency, self.discount_rate_curve_type).data_dict}

        if fwdstartswap.style == 'Receiver':
            rps = 'r'
        elif fwdstartswap.style == 'Payer':
            rps = 'p'
        else:
            raise RuntimeError('Unknown swap style ' + fwdstartswap.style)

        # remove timezone
        expiry = datetime.combine(fwdstartswap.expiration.date(), datetime.min.time())

        if self.forward_rate_curve_type == 'SWAPTION':
            rate_curve = {dt: market.get_forward_rates(fwdstartswap.currency)}
            atmf = atmf_yields_interpolate_old(rate_curve, spot_rate_curve, dt, fwdstartswap.tenor, expiry)
        elif self.forward_rate_curve_type == 'SPOTRATE':
            # use spot rate as forward rate
            tenor_elements = re.match(r'(\d{1,2})(Y|M|W|D|y|m|w|d)', fwdstartswap.tenor).groups()
            tenor_years = get_expiry_in_year(tenor_elements[0], tenor_elements[1])
            atmf = rate_interpolate(spot_rate_curve, dt, tenor_years)
        else:
            raise RuntimeError(f'Unknown forward rate curve type {self.forward_rate_curve_type}')

        results = fwdstartswapcalc_old(dt, None, None, df_curve, fwdstartswap.currency,
                                   expiry, fwdstartswap.tenor, fwdstartswap.strike, rps, atmf_override=atmf)

        return valuer_utils.return_results_based_on_dictionary(calc_types, {
            'price': results['pv'],
            'delta': results['delta'],
            'gamma': results['gamma'],
            'forward_rate': results['fwd']
        })
