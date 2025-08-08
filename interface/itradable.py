from ..interface.imarket import IMarket
from typing import final, Union


class ITradable:
    def __init__(self):

        pass

    def clone(self):
        pass

    def has_expiration(self):
        pass

    def is_expired(self, market: IMarket) -> bool:
        pass

    def intrinsic_value(self, market: IMarket) -> float:
        pass

    def get_underlyings(self):
        return []

    def get_rate_underlyings(self):
        return []

    @final
    def price(self, market: IMarket, valuer=None, calc_types: Union[str, list] = 'price', currency=None, **kwargs):
        if valuer is None:
            from ..valuation.valuer_factory import ValuerFactory
            valuer = ValuerFactory().get_valuer(self)
        return valuer.price(self, market, calc_types, currency=currency, **kwargs)

    @final
    def ask_keys(self, valuer=None, market: IMarket=None, **kwargs):
        if valuer is None:
            from ..valuation.valuer_factory import ValuerFactory
            valuer = ValuerFactory().get_valuer(self)
        return valuer.ask_keys(self, market=market, **kwargs)

    def name(self):
        pass
