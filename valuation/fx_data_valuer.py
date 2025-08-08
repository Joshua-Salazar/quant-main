from ..interface.ivaluer import IValuer
from ..interface.imarket import IMarket
from ..tradable.fxspot import FXSpot
from ..valuation import valuer_utils


class FXDataValuer(IValuer):
    def price(self, tradable: FXSpot, market: IMarket, calc_types='price', **kwargs):
        data = {
            'price': market.get_fx_spot(tradable.ticker),
            'delta': 1.0,
        }
        return valuer_utils.return_results_based_on_dictionary(calc_types, data)
