from datetime import datetime
from ..dates.utils import set_timezone
from ..interface.itradable import ITradable

class FXforward(ITradable):
    def __init__(self, root: str, underlying: str, currency: str, expiration: datetime, tz_name: str=""):
        self.root = root
        self.underlying = underlying
        self.currency = currency
        self.tz_name = tz_name
        self.expiration = expiration if self.tz_name is None else set_timezone(expiration, self.tz_name)

    def clone(self):
        return FXforward( self.root, self.underlying, self.currency, self.expiration, self.tz_name)

    def has_expiration(self):
        return True

    def name(self):
        if isinstance(self.underlying, ITradable):
            return self.underlying.name() + 'Fwd' + self.expiration.isoformat() if isinstance(self.expiration, datetime) else self.expiration \
                                                                                               + self.currency
        else:
            return self.underlying + (
                self.expiration.isoformat() if isinstance(self.expiration, datetime) else self.expiration) + 'Fwd' + self.currency

    def __eq__(self, other):
        if not isinstance(other, FXforward ):
            return False
        return self.name() == other.name()