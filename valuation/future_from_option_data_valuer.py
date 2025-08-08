from ..analytics.utils import float_equal
from ..analytics.option_finder import find_nearest_listed_options
from ..interface.ivaluer import IValuer
from ..interface.imarket import IMarket
from ..tradable.future import Future
from ..valuation import valuer_utils
from ..valuation.option_data_valuer import OptionDataValuer


class FutureFromOptionDataValuer(IValuer):
    def __init__(self, option_root, underlying):
        self.option_root = option_root
        self.underlying = underlying

    def price(self, tradable: Future, market: IMarket, calc_types='price', **kwargs):
        option_universe = market.get_option_universe(self.option_root)
        spot = market.get_spot(self.underlying)

        option_to_use_1_C = find_nearest_listed_options(tradable.expiration, spot * 1.05, 'C', option_universe, return_as_tradables=True)
        option_to_use_1_P = find_nearest_listed_options(tradable.expiration, spot * 1.05, 'P', option_universe, return_as_tradables=True)
        option_to_use_2_C = find_nearest_listed_options(tradable.expiration, spot * 0.95, 'C', option_universe, return_as_tradables=True)
        option_to_use_2_P = find_nearest_listed_options(tradable.expiration, spot * 0.95, 'P', option_universe, return_as_tradables=True)

        assert option_to_use_1_C[0].expiration.date() == tradable.expiration.date() and option_to_use_1_P[0].expiration.date() == tradable.expiration.date() and option_to_use_2_C[0].expiration.date() == tradable.expiration.date() and option_to_use_2_P[0].expiration.date() == tradable.expiration.date()
        assert float_equal(option_to_use_1_C[0].strike, option_to_use_1_P[0].strike) and float_equal(option_to_use_2_C[0].strike, option_to_use_2_P[0].strike)

        call_1_price = option_to_use_1_C[0].price(market, OptionDataValuer(), calc_types='price')
        call_2_price = option_to_use_2_C[0].price(market, OptionDataValuer(), calc_types='price')
        put_1_price = option_to_use_1_P[0].price(market, OptionDataValuer(), calc_types='price')
        put_2_price = option_to_use_2_P[0].price(market, OptionDataValuer(), calc_types='price')
        k_1 = option_to_use_1_C[0].strike
        k_2 = option_to_use_2_C[0].strike
        synthetic_1 = call_1_price - put_1_price
        synthetic_2 = call_2_price - put_2_price
        df = (synthetic_1 - synthetic_2) / (k_2 - k_1)
        forward_price = k_1 + synthetic_1 / df

        assert spot * 0.9 < forward_price < spot * 1.1

        data = {
            'price': forward_price * tradable.contract_size,
            'delta': forward_price / spot * tradable.contract_size
        }

        return valuer_utils.return_results_based_on_dictionary(calc_types, data)
