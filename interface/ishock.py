from abc import ABC, abstractmethod
from enum import Enum


class ShockType(Enum):
    DATETIMESHIFT = 1
    DIVSHOCK = 2
    RATESHOCK = 3
    REPOSHOCK = 4
    SPOTSHOCK = 5
    VOLSHOCK = 6
    VOLFUTURESHOCK = 7

class IShock(ABC):
    def __init__(self, type: ShockType):
        self.type = type

    @abstractmethod
    def is_market_shock(self):
        pass

    def get_base_datetime(self):
        raise Exception("unsupported base datetime")