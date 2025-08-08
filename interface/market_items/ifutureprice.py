from abc import ABC, abstractmethod
from datetime import datetime


class IFuturePrice(ABC):

    @abstractmethod
    def get_future_price(self, base_date: datetime, expiry_date: datetime) -> float:
        pass
