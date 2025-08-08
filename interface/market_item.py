from __future__ import annotations
from abc import ABC, abstractmethod
from ..interface.ishock import IShock


class MarketItem(ABC):
    @abstractmethod
    def get_market_key(self):
        pass

    @abstractmethod
    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        pass
