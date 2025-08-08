from abc import ABC, abstractmethod
from ..infrastructure.fmarket import FMarket
from ..infrastructure.market import Market
from ..interface.itradable import ITradable


class IValuer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def price(self, tradable: ITradable, market: Market, calc_types='price', **kwargs):
        pass

    def ask_keys(self, tradable: ITradable,  market: Market=None, **kwargs):
        return []

    def fmtm(self, market: FMarket) -> float:
        pass