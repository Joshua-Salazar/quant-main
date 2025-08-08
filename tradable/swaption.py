from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable
from ..tradable.forwardstartswap import ForwardStartSwap


class Swaption(ITradable):
    def __init__(self, currency: str, expiration: datetime, tenor: str, strike: float, style: str, curve: str):
        self.currency = currency
        self.expiration = expiration
        self.tenor = tenor
        self.strike = strike
        self.style = style
        self.curve = curve
        #Hard coding payers (call on yield) for clean netting
        self.underlying = ForwardStartSwap(currency, expiration, tenor, strike, 'Payer', self.curve)

    def clone(self):
        return Swaption(
            self.currency, self.expiration, self.tenor, self.strike, self.style, self.curve
        )

    def has_expiration(self):
        return True

    def name(self):
        return 'Swaption' + self.currency + self.expiration.isoformat() + self.tenor + format_number(self.strike, 8) + self.style

    def override_strike(self, strike):
        return Swaption(self.currency, self.expiration, self.tenor, strike, self.style, self.curve)