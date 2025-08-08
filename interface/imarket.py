from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from ..constants.underlying_type import UnderlyingType
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem


class IMarket(ABC):

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def add_item(self, key: str, item: MarketItem):
        pass

    @abstractmethod
    def has_item(self, key: str) -> bool:
        pass

    @abstractmethod
    def get_item(self, key: str) -> MarketItem:
        pass

    @abstractmethod
    def get_base_datetime(self) -> datetime:
        pass

    @abstractmethod
    def get_discount_rate(self, underlying: str, expiry_dt: datetime) -> float:
        pass

    @abstractmethod
    def get_forward_discount_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        pass

    @abstractmethod
    def get_borrow_rate(self, underlying: str, expiry_dt: datetime) -> float:
        pass

    @abstractmethod
    def get_forward_borrow_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        pass

    @abstractmethod
    def get_spot(self, underlying: str) -> float:
        pass

    @abstractmethod
    def get_future_price(self, underlying: str, expiry_dt: datetime) -> float:
        pass

    @abstractmethod
    def get_vol(self, underlying: str, expiry_dt: datetime, strike: float, strike_type=None) -> float:
        pass

    @abstractmethod
    def get_vol_type(self, underlying: str) -> UnderlyingType:
        pass

    @abstractmethod
    def get_vol_surface(self, underlying: str):
        pass

    @abstractmethod
    def apply(self, shock_map: {str: [IShock]}, **kwargs) -> IMarket:
        pass



