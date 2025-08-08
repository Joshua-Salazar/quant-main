from datetime import datetime
from ..dates.utils import coerce_timezone, set_timezone
from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
from ..language import format_number


class Option(ITradable):
    def __init__(self, root: str, underlying, currency: str,
                 expiration: datetime, strike: float, is_call: bool, is_american: bool, contract_size: float,
                 tz_name: str,
                 listed_ticker=None,
                 expiration_rule: str = None, specialisation: str = None):
        self.root = root
        self.is_call = is_call
        self.is_american = is_american
        self.underlying = underlying
        self.currency = currency
        self.tz_name = tz_name
        self.contract_size = contract_size
        if tz_name is not None and len(tz_name) and isinstance(expiration, datetime):
            self.expiration = set_timezone(expiration, tz_name)
        else:
            self.expiration = expiration
        self.strike = strike
        self.listed_ticker = listed_ticker
        self.expiration_rule = expiration_rule
        self.specialisation = specialisation
        self.name_str = self.name()

    def clone(self):
        expiration_rule = self.expiration_rule if hasattr(self, 'expiration_rule') else None
        return Option(
            self.root, self.underlying, self.currency,
            self.expiration, self.strike, self. is_call, self.is_american, self.contract_size,
            self.tz_name, self.listed_ticker, expiration_rule)

    def has_expiration(self):
        return True

    def is_expired(self, market: IMarket) -> bool:
        dt, expiry = coerce_timezone(market.get_base_datetime(), self.expiration)
        expired = dt >= expiry
        return expired

    def intrinsic_value(self, market_or_spot, underlying_valuer=None) -> float:
        if isinstance(market_or_spot, IMarket):
            if isinstance(self.underlying, str):
                spot = market_or_spot.get_spot(self.root)
            elif underlying_valuer is not None:
                spot = self.underlying.price(market_or_spot, calc_types='price', valuer=underlying_valuer)
            else:
                spot = market_or_spot.get_future_price(self.root, self.underlying.expiration, future=self.underlying)
        else:
            spot = market_or_spot

        phi = 1 if self.is_call else -1
        res = max(phi * (spot - self.strike), 0)
        return res

    def name(self):
        if self.listed_ticker is None:
            if isinstance(self.underlying, ITradable):
                return self.underlying.name() \
                       + format_number(self.strike, 8) \
                       + ('Call' if self.is_call else 'Put') + ('A' if self.is_american else 'E') \
                       + self.expiration.isoformat() if isinstance(self.expiration, datetime) else self.expiration \
                       + self.currency + str(self.contract_size)
            else:
                return self.underlying + (self.expiration.isoformat() if isinstance(self.expiration, datetime) else self.expiration) \
                       + format_number(self.strike, 8) \
                       + ('Call' if self.is_call else 'Put') + ('A' if self.is_american else 'E') \
                       + self.currency + str(self.contract_size)
        else:
            return self.listed_ticker

    def __eq__(self, other):
        if not isinstance(other, Option):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())

    def get_underlyings(self):
        return [self.underlying]
