from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class GeoDispVolSwap(ITradable):
    def __init__(self, underlyings: list, weights: list, inception: datetime, expiration: datetime,
                 strike_in_vol: float, notional: float, currency: str, lag: int, cap: float = 0):
        self.underlyings = underlyings
        self.weights = weights
        self.inception = inception
        self.expiration = expiration
        self.strike_in_vol = strike_in_vol
        self.strike_in_var = strike_in_vol * strike_in_vol
        self.notional = notional
        self.currency = currency
        self.lag = lag
        self.cap = cap

    def clone(self):
        return GeoDispVolSwap(
            self.underlyings, self.weights, self.inception, self.expiration,
            self.strike_in_vol, self.notional, self.currency, self.lag
        )

    def has_expiration(self):
        return True

    def name(self):
        name = [self.weights[idx] + self.underlyings[idx] for idx in range(len(self.weights))]
        return name + self.inception.isoformat() + self.expiration.isoformat() + \
               format_number(self.strike_in_vol, 8) + format_number(self.notional, 8) + self.currency + self.lag

    def get_underlyings(self):
        return self.underlyings

    def has_cap(self):
        return self.cap != 0

    def get_cap(self):
        return self.cap
