from ..interface.ivaluer import IValuer
from ..interface.imarket import IMarket
from ..tradable.stock import Stock
from ..valuation.utils import find_fx_for_tradable


class StockDataValuer(IValuer):
    def price(self, tradable: Stock, market: IMarket, calc_types='price', currency=None, **kwargs):
        fx = find_fx_for_tradable(market, tradable, currency)

        data = {
            'price': market.get_spot(tradable.ticker) * fx,
            'delta': 1.0,
            'gamma': 0.0,
            'vega': 0.0,
            'vanna': 0.0,
            'volga': 0.0,
            'rho': 0.0,
            'theta': 0.0,
        }
        if isinstance(calc_types, list):
            return [data[calc_type] for calc_type in calc_types]
        else:
            return data[calc_types]
