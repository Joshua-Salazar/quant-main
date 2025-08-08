from enum import Enum


class BuySell(Enum):
    BUY = "BUY"
    SELL = "SELL"

    def inverse(self):
        return BuySell.SELL if self == BuySell.BUY else BuySell.BUY
