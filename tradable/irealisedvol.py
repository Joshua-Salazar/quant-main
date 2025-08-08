from datetime import datetime
from ..analytics import math
from ..constants.asset_class import AssetClass
from ..dates import utils as date_utils
from ..dates.utils import count_business_days
from ..interface.imarket import IMarket
import numpy as np
import pandas as pd


class IRealisedVol:
    def __init__(self, underlying: str, inception: datetime, expiration: datetime, lag: int = 1, cdr_code="#A", fixing_src="", asset_class=AssetClass.EQUITY):
        self.underlying = underlying
        self.inception = inception
        self.expiration = expiration
        self.lag = lag
        self.cdr_code = cdr_code
        self.fixing_src = fixing_src
        self.asset_class = asset_class

    def get_cdr_codes(self):
        codes = []
        if len(self.cdr_code) > 0:
            codes.append(self.cdr_code)
        if self.asset_class == AssetClass.FX:
            codes += [self.underlying[:3], self.underlying[3:]]
        return codes

    def get_fixings(self, market: IMarket):
        codes = self.get_cdr_codes()
        holidays = []
        for code in codes:
            holidays += market.get_holidays(code, self.inception.date(), self.expiration.date())
        fixing_data = []
        dt = self.inception
        while dt < market.get_base_datetime():
            fixing = market.get_fixing_from_fixing_table(self.fixing_src, dt.date())
            fixing_data.append([dt, fixing])
            dt = date_utils.add_business_days(dt, 1, holidays)
        return pd.DataFrame(fixing_data, columns=["date", "fixings"]) if len(fixing_data) > 1 else pd.DataFrame()

    def get_minimum_realised_vol(self, market: IMarket):
        # minimum realised vol based on assumption that the rest of spot remains the same level
        fixing_data = self.get_fixings(market)
        if fixing_data.empty:
            realised_vol = 0
        else:
            ln_ret = np.log(fixing_data["fixings"]).diff(self.lag)
            realised_vol = math.calculate_realised_vol_from_returns(ln_ret, self.lag)
        done_days = fixing_data.shape[0]
        cdr_codes = self.get_cdr_codes()
        holidays = []
        for code in cdr_codes:
            holidays += market.get_holidays(code, self.inception.date(), self.expiration.date())
        left_days = count_business_days(market.get_base_datetime(), self.expiration, holidays)
        done_ratio = done_days / (done_days + left_days)
        min_realised_vol = realised_vol * np.sqrt(done_ratio)
        return min_realised_vol