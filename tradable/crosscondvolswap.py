from datetime import datetime
from ..language import format_number
from ..interface.itradable import ITradable


class CrossCondVolSwap(ITradable):
    def __init__(self, underlying: str, barrier_underlying: str, inception: datetime, expiration: datetime,
                 strike_in_vol: float, notional: float, currency: str, barrier_condition: str, barrier_type: str,
                 down_var_barrier: float = None,
                 up_var_barrier: float = None,
                 cap: float = 0,
                 ):
        self.underlying = underlying
        self.barrier_underlying = barrier_underlying
        self.inception = inception
        self.expiration = expiration
        self.strike_in_vol = strike_in_vol
        self.notional = notional
        self.currency = currency
        self.barrier_condition = barrier_condition
        self.barrier_type = barrier_type
        self.down_var_barrier = down_var_barrier
        self.up_var_barrier = up_var_barrier
        self.cap = cap

        self.validate()

    def validate(self):
        if self.down_var_barrier is None and self.up_var_barrier is None:
            raise Exception("At least one barrier has to be set")
        if self.down_var_barrier is not None and self.up_var_barrier is not None \
                and self.down_var_barrier < self.up_var_barrier:
            raise Exception(f"down var barrier {self.down_var_barrier} must be greater "
                            f"than up var barrier {self.up_var_barrier}")

    def clone(self):
        return CrossCondVolSwap(
            self.underlying, self.barrier_underlying, self.inception, self.expiration, self.strike_in_vol,
            self.notional, self.currency, self.barrier_condition, self.barrier_type, self.down_var_barrier,
            self.up_var_barrier
        )

    def has_expiration(self):
        return True

    def name(self):
        return self.underlying + self.barrier_underlying, self.inception.isoformat() + self.expiration.isoformat() + \
               format_number(self.strike_in_vol, 8) + format_number(self.notional, 8) + self.currency + \
               self.barrier_condition + self.barrier_type + self.down_var_barrier + self.up_var_barrier

    def get_underlyings(self):
        return [self.underlying, self.barrier_underlying]

    def has_cap(self):
        return self.cap != 0

    def get_cap(self):
        return self.cap