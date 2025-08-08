from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class FXCorrelationSwap(ITradable):
    def __init__(self, pair1: str, pair2: str, inception: datetime, expiration: datetime, strike: float,
                 notional: float, fixing_src: str, cdr_code: str = ""):
        self.pair1 = pair1
        self.pair2 = pair2
        self.inception = inception
        self.expiration = expiration
        self.strike = strike
        self.notional = notional
        self.fixing_src = fixing_src
        self.cdr_code = cdr_code

    def clone(self):
        return FXCorrelationSwap(
            self.pair1, self.pair2, self.inception, self.expiration, self.strike, self.notional,  self.fixing_src,
            self.cdr_code)

    def has_expiration(self):
        return True

    def name(self):
        return self.pair1 + self.pair2 + self.inception.isoformat() + self.expiration.isoformat() + \
               format_number(self.strike, 8) + format_number(self.notional, 8) + self.fixing_src + self.cdr_code

    def get_underlyings(self):
        return [self.pair1, self.pair2]

