from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class ForwardStartCDS(ITradable):
    """Unless necessary, only use payers (call on yield) so underlier is consistent across swaptions"""
    def __init__(self, currency: str, expiration: datetime, tenor: str, strike: float, style: str, spread: str):
        self.currency = currency
        self.expiration = expiration
        self.tenor = tenor
        self.strike = strike
        self.style = style
        self.spread = spread

    def clone(self):
        return ForwardStartCDS(
            self.currency, self.expiration, self.tenor, self.strike, self.style, self.spread
        )

    def has_expiration(self):
        return True

    def name(self):
        return 'ForwardStartCDS' + self.currency + self.spread + self.expiration.isoformat() + self.tenor + format_number(self.strike, 8) + self.style
