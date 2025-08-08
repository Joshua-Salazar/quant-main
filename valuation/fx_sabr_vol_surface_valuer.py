from ..infrastructure.market import Market
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer


class FXSABRVolSurfaceValuer(IValuer):
    def __init__(self):
        pass

    def price(self, tradable, market: Market, calc_types='price', **kwargs):
        fx_pair_name = tradable.ticker
        vol_surface = market.get_fx_sabr_vol_surface(fx_pair_name)

        data = {
            'price': vol_surface.spot,
            'delta': 1.0,
            'gamma': 0,
            'vega': 0,
            'vanna': 0,
            'volga': 0,
            'rho': 0,
            'theta': 0,

        }
        return valuer_utils.return_results_based_on_dictionary(calc_types, data)
