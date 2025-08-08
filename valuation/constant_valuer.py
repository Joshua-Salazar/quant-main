from ..infrastructure.market import Market
from ..tradable.constant import Constant
from ..interface.ivaluer import IValuer
from ..valuation.utils import find_fx_for_tradable


class ConstantValuer(IValuer):
    def __init__(self):
        pass

    def price(self, constant: Constant, market: Market, calc_types='price', currency=None, **kwargs):
        fx = find_fx_for_tradable(market, constant, currency)

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        values = tuple()
        for calc_type in calc_types_list:
            if calc_type == 'price':
                values = values + (1.0 * fx,)
            elif calc_type in ['delta', 'gamma', 'vega', 'theta', 'vanna', 'volga', 'rho']:
                values = values + (0.0,)
            else:
                values = values + (None,)

        if isinstance(calc_types, list):
            return values
        else:
            return values[0]