from ..interface.ivaluer import IValuer
from ..interface.imarket import IMarket
from ..tradable.future import Future
from ..valuation import valuer_utils
from ..analytics.symbology import ticker_from_future_root
from ..valuation.utils import find_fx_for_tradable


class FutureDataValuer(IValuer):
    def __init__(self, price_name='close', imply_delta_from_spot=True, overrides={}):
        # overrides are a tuple of futures name and date mapped to a price
        self.price_name = price_name
        self.imply_delta_from_spot = imply_delta_from_spot
        self.overrides = overrides

    def price(self, tradable: Future, market: IMarket, calc_types='price', currency=None, **kwargs):
        if (tradable.name(), market.get_base_datetime()) in self.overrides.keys():
            future_price = self.overrides[(tradable.name(), market.get_base_datetime())]
        else:
            future_price = market.get_future_data(tradable)[self.price_name]

        fx = find_fx_for_tradable(market, tradable, currency)

        data = {
            'price': future_price * tradable.contract_size * fx,
            'delta': 1. * tradable.contract_size,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
        }

        if (calc_types == 'delta' or 'delta' in calc_types) and self.imply_delta_from_spot:
            spot = market.get_spot(ticker_from_future_root(tradable.root))
            data['delta'] = future_price / spot

        return valuer_utils.return_results_based_on_dictionary(calc_types, data)


class FutureDataIntradayValuer(IValuer):
    def __init__(self, price_name='close'):
        self.price_name = price_name

    def price(self, tradable: Future, market: IMarket, calc_types='price', currency=None, **kwargs):
        future_price = market.get_future_intraday_data(tradable).get(self.price_name, float('nan'))

        fx = find_fx_for_tradable(market, tradable, currency)

        data = {
            'price': future_price * tradable.contract_size * fx,
            'delta': 1. * tradable.contract_size,
            'gamma': 0,
            'vega': 0,
        }

        return valuer_utils.return_results_based_on_dictionary(calc_types, data)
