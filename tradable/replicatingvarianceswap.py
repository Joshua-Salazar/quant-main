from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class ReplVarianceSwap(ITradable):
    def __init__(self, underlying: str, inception: datetime, expiration: datetime,
                 strike_in_vol: float, notional: float, currency: str, strike_star: float, strike_step: int,
                 strike_min: float, strike_max: float, reference_vol_level: float):
        self.underlying = underlying
        self.inception = inception
        self.expiration = expiration
        self.strike_in_vol = strike_in_vol
        self.strike_in_var = strike_in_vol * strike_in_vol
        self.notional = notional
        self.currency = currency
        self.strike_star = strike_star
        self.strike_step = strike_step
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.reference_vol_level = reference_vol_level

    def clone(self):
        return ReplVarianceSwap(
            self.underlying, self.inception, self.expiration, self.strike_in_vol, self.notional, self.currency,
            self.strike_star, self.strike_step, self.strike_min, self.strike_max, self.reference_vol_level)

    def has_expiration(self):
        return True

    def name(self):
        return self.underlying + self.inception.isoformat() + self.expiration.isoformat() + \
            format_number(self.strike_in_vol, 8) + format_number(self.notional, 8) + self.currency + \
            self.strike_star + self.strike_step + self.strike_min + self.strike_max + self.reference_vol_level

    def get_underlyings(self):
        return [self.underlying]

    def has_cap(self):
        return False

    def get_cap(self):
        return self.cap