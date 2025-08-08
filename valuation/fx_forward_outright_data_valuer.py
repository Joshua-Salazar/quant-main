from ..constants.ccy import Ccy
from ..infrastructure.fx_pair import FXPair
from ..infrastructure.market import Market
from ..tradable.FXforward import FXforward
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer


class FXForwardOutrightDataValuer(IValuer):
    def __init__(self):
        pass

    def price(self, forward: FXforward, market: Market, calc_types='price', **kwargs):
        fwd = market.get_fx_fwd(FXPair(Ccy(forward.underlying), Ccy(forward.currency)), forward.expiration)
        return valuer_utils.return_results_based_on_dictionary(calc_types, {'price': fwd})
