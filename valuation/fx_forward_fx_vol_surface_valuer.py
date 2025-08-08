from ..infrastructure.market import Market
from ..tradable.FXforward import FXforward
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer


class FXForwardDataValuer(IValuer):
    def __init__(self):
        pass

    def price(self, forward: FXforward, market: Market, calc_types='price', return_struc=False, vol_override = None, **kwargs):
        fx_pair_name = f'{forward.underlying}'
        vol_surface = market.get_fx_vol_surface(fx_pair_name)
        fwd = vol_surface.get_forward(forward.expiration)
        return valuer_utils.return_results_based_on_dictionary(calc_types, {'price': fwd,
                                                                            'forward_delta': 1.0,
                                                                            'theta': 0.0,
                                                                            'vega': 0.0,
                                                                            'gamma': 0.0 })
