from abc import ABC, abstractmethod
from datetime import datetime


class ISpot(ABC):

    @abstractmethod
    def get_spot(self, base_date: datetime) -> float:
        pass
