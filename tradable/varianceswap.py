from datetime import datetime
from ..analytics import math
from ..constants.asset_class import AssetClass
from ..dates.utils import coerce_timezone
from ..language import format_number
from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
from ..tradable.irealisedvol import IRealisedVol
import numpy as np
import pandas as pd


class VarianceSwap(ITradable, IRealisedVol):
    def __init__(self, underlying: str, inception: datetime, expiration: datetime,
                 strike_in_vol: float, notional: float, currency: str, lag: int = 1, cap: float = 0,
                 rep_ptf=pd.DataFrame(), synth_fwd=pd.DataFrame(), cdr_code="#A", fixing_src="",
                 asset_class=AssetClass.EQUITY, inst_id=None):
        IRealisedVol.__init__(self, underlying=underlying, inception=inception, expiration=expiration, lag=lag, cdr_code=cdr_code, fixing_src=fixing_src, asset_class=asset_class)
        self.underlying = underlying
        self.inception = inception
        self.expiration = expiration
        self.strike_in_vol = strike_in_vol
        self.strike_in_var = strike_in_vol * strike_in_vol
        self.notional = notional
        self.currency = currency
        self.lag = lag
        self.cap = cap
        self.rep_ptf = rep_ptf
        self.synth_fwd = synth_fwd
        self.cdr_code = cdr_code
        self.fixing_src = fixing_src
        self.asset_class = asset_class
        self.name_str = self.name()
        self.root = self.underlying
        self.contract_size = 1
        self.inst_id = inst_id

    def clone(self):
        return VarianceSwap(
            self.underlying, self.inception, self.expiration,
            self.strike_in_vol, self.notional, self.currency, self.lag, self.cap,
            self.rep_ptf, self.synth_fwd, self.cdr_code, fixing_src=self.fixing_src, asset_class=self.asset_class
        )

    def override_strike(self, strike_in_vol):
        return VarianceSwap(
            self.underlying, self.inception, self.expiration,
            strike_in_vol, self.notional, self.currency, self.lag, self.cap,
            self.rep_ptf, self.synth_fwd, self.cdr_code, fixing_src=self.fixing_src, asset_class=self.asset_class
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
        ln_ret = np.log(fixing_data["fixings"]).diff(self.lag)
        realised_vol = math.calculate_realised_vol_from_returns(ln_ret, self.lag)
        total_var = realised_vol * realised_vol - self.strike_in_vol * self.strike_in_vol
        if self.has_cap():
            total_var = min(total_var, (self.cap - 1) * self.strike_in_vol * self.strike_in_vol)
        res = total_var * self.notional * self.contract_size
        return res

    def name(self):
        return self.underlying + self.inception.isoformat() + self.expiration.isoformat() \
               + self.asset_class.value + "VARSWAP" + \
               format_number(self.strike_in_vol, 8) + format_number(self.notional, 8) + self.currency + str(self.lag)

    def get_underlyings(self):
        return [self.underlying]

    def has_cap(self):
        return self.cap != 0

    def get_cap(self):
        return self.cap

    def __eq__(self, other):
        if not isinstance(other, VarianceSwap):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())
