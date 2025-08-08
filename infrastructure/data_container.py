from abc import ABC, abstractmethod
from datetime import timedelta


class DataContainer(ABC):
    @abstractmethod
    def get_market_key(self):
        pass

    @abstractmethod
    def get_market_item(self, dt):
        pass

    def get_market_item_with_fill_forward(self, dt, max_days=10):
        market_item = self.get_market_item(dt)
        # allow fallback look up
        if market_item is None:
            prev_dt = dt
            num_days = 0
            while market_item is None and num_days <= max_days:
                prev_dt = prev_dt - timedelta(days=1)
                market_item = self.get_market_item(prev_dt)
                num_days += 1
            if market_item is None:
                raise Exception(f"{dt}, Cannot find market item {self.get_market_key()} "
                                f"between {prev_dt} and {dt}")
        return market_item
