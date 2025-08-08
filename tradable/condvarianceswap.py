from datetime import datetime
from ..analytics import math
from ..constants.asset_class import AssetClass
from ..dates.utils import coerce_timezone
from ..language import format_number
from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
from ..tradable.irealisedvol import IRealisedVol
import numpy as np


class CondVarianceSwap(ITradable, IRealisedVol):
    def __init__(self, underlying: str, inception: datetime, expiration: datetime,
                 strike_in_vol: float, notional: float, currency: str, barrier_condition: str, barrier_type: str,
                 down_var_barrier: float = None, up_var_barrier: float = None, cap: float = 0, cdr_code="#A", fixing_src="", asset_class=AssetClass.EQUITY,
                 inst_id=None,
                 ):
        IRealisedVol.__init__(self, underlying=underlying, inception=inception, expiration=expiration, lag=1, cdr_code=cdr_code, fixing_src=fixing_src, asset_class=asset_class)
        self.underlying = underlying
        self.inception = inception
        self.expiration = expiration
        self.strike_in_vol = strike_in_vol
        self.strike_in_var = strike_in_vol * strike_in_vol
        self.notional = notional
        self.currency = currency
        self.barrier_condition = barrier_condition
        self.barrier_type = barrier_type
        self.down_var_barrier = down_var_barrier
        self.up_var_barrier = up_var_barrier
        self.cap = cap
        self.cdr_code = cdr_code
        self.fixing_src = fixing_src
        self.asset_class = asset_class
        self.name_str = self.name()
        self.contract_size = 1
        self.inst_id = inst_id

        self.validate()

    def validate(self):
        if self.down_var_barrier is None and self.up_var_barrier is None:
            raise Exception("At least one barrier has to be set")
        if self.down_var_barrier is not None and self.up_var_barrier is not None \
                and self.down_var_barrier < self.up_var_barrier:
            raise Exception(f"down var barrier {self.down_var_barrier} must be greater "
                            f"than up var barrier {self.up_var_barrier}")

    def clone(self):
        return CondVarianceSwap(
            self.underlying, self.inception, self.expiration, self.strike_in_vol, self.notional, self.currency,
            self.barrier_condition, self.barrier_type, self.down_var_barrier, self.up_var_barrier, cap=self.cap, cdr_code=self.cdr_code, fixing_src=self.fixing_src, asset_class=self.asset_class
        )

    def override_strike(self, strike_in_vol):
        return CondVarianceSwap(
            self.underlying, self.inception, self.expiration, strike_in_vol, self.notional, self.currency,
            self.barrier_condition, self.barrier_type, self.down_var_barrier, self.up_var_barrier, cap=self.cap, cdr_code=self.cdr_code, fixing_src=self.fixing_src, asset_class=self.asset_class
        )

    def has_expiration(self):
        return True

    def is_expired(self, market: IMarket) -> bool:
        dt, expiry = coerce_timezone(market.get_base_datetime(), self.expiration)
        expired = dt >= expiry
        return expired

    def intrinsic_value(self, market: IMarket, underlying_valuer=None) -> float:
        assert self.is_expired(market) or self.expiration.date() <= market.get_base_datetime().date()
        fixing_data = self.get_fixings(market)
        und_fixings = fixing_data.set_index("date").rename(columns={"fixings": "T"})
        if self.barrier_type == "UpVar":
            if self.barrier_condition == "bcTandTMinusOneConvention":
                und_fixings.loc[:, "T-1"] = und_fixings["T"].shift(periods=1)
                und_fixings.dropna(inplace=True)
                valid_und_fixings = und_fixings[
                    (und_fixings["T"] > self.up_var_barrier)
                    & (und_fixings["T-1"] > self.up_var_barrier)]
            else:
                raise Exception(f"Unsupport barrier condition {self.instrument.tradable.barrier_condition}")
        elif self.barrier_type == "Corridor":
            if self.barrier_condition == "bcTandTMinusOneConvention":
                und_fixings.loc[:, "T-1"] = und_fixings["T"].shift(periods=1)
                und_fixings.dropna(inplace=True)
                valid_und_fixings = und_fixings[
                    (und_fixings["T"] > self.up_var_barrier)
                    & (und_fixings["T-1"] > self.up_var_barrier)
                    & (und_fixings["T"] < self.down_var_barrier)
                    & (und_fixings["T-1"] < self.down_var_barrier)]
            else:
                raise Exception(f"Unsupport barrier condition {self.barrier_condition}")

        ln_ret = np.log(valid_und_fixings["T"]) - np.log(valid_und_fixings["T-1"])
        realised_vol = math.calculate_realised_vol_from_returns(ln_ret, self.lag)
        total_var = realised_vol * realised_vol - self.strike_in_vol * self.strike_in_vol
        if self.has_cap():
            total_var = min(total_var, (self.cap - 1) * self.strike_in_vol * self.strike_in_vol)
        res = total_var * self.notional * self.contract_size
        return res

    def name(self):
        return self.underlying + self.inception.isoformat() + self.expiration.isoformat() \
               + self.asset_class.value + "CONDVARSWAP" + \
               format_number(self.strike_in_vol, 8) + format_number(self.notional, 8) + self.currency + \
               self.barrier_condition + self.barrier_type + str(self.down_var_barrier) + str(self.up_var_barrier)

    def get_underlyings(self):
        return [self.underlying]

    def has_cap(self):
        return self.cap != 0

    def get_cap(self):
        return self.cap

    def __eq__(self, other):
        if not isinstance(other, CondVarianceSwap):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())