from datetime import datetime
from ..analytics import math
from ..constants.asset_class import AssetClass
from ..dates.utils import coerce_timezone
from ..language import format_number
from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
from ..tradable.irealisedvol import IRealisedVol
import numpy as np


class VolOption(ITradable, IRealisedVol):
    def __init__(self, underlying: str, inception: datetime, expiration: datetime, vol_strike: float,
                 notional: float, currency: str, lag: int = 1, is_cap: bool = False, cap: float = 0,
                 cdr_code="", fixing_src="", asset_class=AssetClass.EQUITY):
        IRealisedVol.__init__(self, underlying=underlying, inception=inception, expiration=expiration, lag=lag, cdr_code=cdr_code, fixing_src=fixing_src, asset_class=asset_class)
        self.underlying = underlying
        self.inception = inception
        self.expiration = expiration
        self.vol_strike = vol_strike
        self.notional = notional
        self.currency = currency
        self.lag = lag
        self.is_cap = is_cap    # indicate if it is cap or floor
        self.cap = cap
        self.cdr_code = cdr_code
        self.fixing_src = fixing_src
        self.asset_class = asset_class
        self.name_str = self.name()
        self.root = self.underlying
        self.contract_size = 1
        self.option_strike = self.vol_strike * self.cap

    def clone(self):
        return VolOption(
            self.underlying, self.inception, self.expiration, self.vol_strike, self.notional, self.currency,
            self.lag, self.is_cap, self.cap, self.cdr_code, self.fixing_src, self.asset_class
        )

    def override_strike(self, vol_strike):
        return VolOption(
            self.underlying, self.inception, self.expiration, vol_strike, self.notional, self.currency,
            self.lag, self.is_cap, self.cap, self.cdr_code, self.fixing_src, self.asset_class
        )

    def has_expiration(self):
        return True

    def is_expired(self, market: IMarket) -> bool:
        dt, expiry = coerce_timezone(market.get_base_datetime(), self.expiration)
        expired = dt >= expiry
        return expired

    def intrinsic_value(self, market: IMarket, underlying_valuer=None) -> float:
        if underlying_valuer is None:
            assert self.is_expired(market) or self.expiration.date() <= market.get_base_datetime().date()
            fixing_data = self.get_fixings(market)
            ln_ret = np.log(fixing_data["fixings"]).diff(self.lag)
            realised_vol = math.calculate_realised_vol_from_returns(ln_ret, self.lag)
        else:
            assert isinstance(underlying_valuer, float)
            realised_vol = underlying_valuer
        if self.is_cap:
            total_vol = min(realised_vol - self.vol_strike, self.option_strike - self.vol_strike)
        else:
            total_vol = max(realised_vol - self.vol_strike, self.option_strike - self.vol_strike)
        res = total_vol * self.notional * self.contract_size
        return res

    def name(self):
        return self.underlying + self.inception.isoformat() + self.expiration.isoformat() \
               + self.asset_class.value + "VOLOPTION" + \
               format_number(self.vol_strike, 8) + format_number(self.notional, 8) + self.currency + str(self.lag) +\
               self.fixing_src + ("CAP" if self.is_cap else "FLOOR" + str(self.cap))

    def get_underlyings(self):
        return [self.underlying]

    def is_cap(self):
        return self.is_cap()

    def get_cap(self):
        return self.cap

    def __eq__(self, other):
        if not isinstance(other, VolOption):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())
