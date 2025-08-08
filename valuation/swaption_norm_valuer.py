from ..analytics.swaptions import swaptioncalc, swaptioncalc_old
from ..infrastructure.market import Market
from ..tradable.swaption import Swaption
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
from scipy.optimize import fsolve


class SwaptionNormValuer(IValuer):
    def __init__(self, df_rates_type='OIS'):
        self.df_rates_type = df_rates_type

    def price(self, swaption: Swaption, market: Market, calc_types='price', **kwargs):
        dt = market.get_base_datetime()

        vol_cube = {dt: market.get_swaption_vol_cube(swaption.currency)}
        rate_curve = {dt: market.get_forward_rates(swaption.currency, swaption.curve)}
        spot_rate_curve = {dt: market.get_spot_rates(swaption.currency, swaption.curve).data_dict}
        df_curve = {dt: market.get_spot_rates(swaption.currency, self.df_rates_type).data_dict}

        if swaption.style == 'Receiver':
            rps = 'r'
        elif swaption.style == 'Payer':
            rps = 'p'
        elif swaption.style == 'Straddle':
            rps = 's'
        else:
            raise RuntimeError('Unknown swaption style ' + swaption.style)
        results = swaptioncalc(dt, vol_cube, rate_curve, spot_rate_curve, df_curve,
                               swaption.currency, swaption.expiration, swaption.tenor, swaption.strike,
                               rps)
        return valuer_utils.return_results_based_on_dictionary(calc_types, {
            'price': results['pv'],
            'delta': results['delta'],
            'vega': results['vega'],
            'revega': results['vega'] * results['vol'],
            'gamma': results['gamma'],
            'forward_rate': results['fwd'],
            'theta': results['theta'],
            'deltapct': results['deltapct'],

        })

    def get_strike_from_delta(self, swaption: Swaption, delta_strike: float, market: Market):
        if not isinstance(swaption, Swaption):
            raise Exception(f"Only support swaption")

        def func(strike):
            tradable_clone = swaption.override_strike(strike)
            delta = self.price(tradable_clone, market, calc_types='deltapct')
            return (delta - delta_strike)**2

        sol = fsolve(lambda x: func(x[0]), swaption.strike)
        fair_strike = sol[0]
        return fair_strike


class SwaptionNormValuerOld(IValuer):
    def __init__(self, df_rates_type='OIS'):
        self.df_rates_type = df_rates_type

    def price(self, swaption: Swaption, market: Market, calc_types='price', **kwargs):
        dt = market.get_base_datetime()

        vol_cube = {dt: market.get_swaption_vol_cube(swaption.currency)}
        rate_curve = {dt: market.get_forward_rates(swaption.currency)}
        spot_rate_curve = {dt: market.get_spot_rates(swaption.currency, 'SWAP').data_dict}
        df_curve = {dt: market.get_spot_rates(swaption.currency, self.df_rates_type).data_dict}

        if swaption.style == 'Receiver':
            rps = 'r'
        elif swaption.style == 'Payer':
            rps = 'p'
        elif swaption.style == 'Straddle':
            rps = 's'
        else:
            raise RuntimeError('Unknown swaption style ' + swaption.style)
        results = swaptioncalc_old(dt, vol_cube, rate_curve, spot_rate_curve, df_curve,
                               swaption.currency, swaption.expiration, swaption.tenor, swaption.strike,
                               rps)

        return valuer_utils.return_results_based_on_dictionary(calc_types, {
            'price': results['pv'],
            'delta': results['delta'],
            'vega': results['vega'],
            'gamma': results['gamma'],
            'forward_rate': results['fwd'],

        })