from abc import ABC, abstractmethod
from datetime import datetime
from ...constants.strike_type import StrikeType
from ...constants.underlying_type import UnderlyingType


class IVolatilitySurface(ABC):

    @abstractmethod
    def get_base_datetime(self) -> datetime:
        return self.base_dt

    @abstractmethod
    def get_underlying_type(self) -> UnderlyingType:
        pass

    @abstractmethod
    def get_vol(self, expiry_dt: datetime, strike: float, strike_type: StrikeType = StrikeType.K) -> float:
        pass
