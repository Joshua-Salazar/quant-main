from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class ForwardStartOption(ITradable):
    def __init__(self, underlying: str, currency: str, strike_date: datetime, expiration: datetime, strike: float, is_call: bool):
        self.underlying = underlying
        self.currency = currency
        self.strike_date = strike_date
        self.expiration = expiration
        self.strike = strike
        self.is_call = is_call

    def clone(self):
        return ForwardStartOption(
            self.underlying, self.currency, self.strike_date, self.expiration, self.strike, self.is_call
        )

    def has_expiration(self):
        return True

    def name(self):
        return 'ForwardStartOption' + self.underlying + self.currency + self.strike_date.isoformat() + \
               self.expiration.isoformat() + format_number(self.strike, 8) + self.is_call

    def get_underlyings(self):
        return [self.underlying]
