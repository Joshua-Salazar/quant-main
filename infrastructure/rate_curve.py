from datetime import datetime
from ..analytics import swaptions
from ..dates.utils import datetime_to_tenor
from ..constants.ccy import Ccy
from ..constants.day_count_convention import DayCountConvention
import math


class RateCurve:
    def __init__(self, ccy: Ccy, ts: dict, base_dt: datetime):
        self.ccy = ccy
        self.ts = dict(sorted(ts.items()))
        self.base_dt = base_dt

    def get_rate(self, expiry: datetime):
        rate = swaptions.rate_curve_interpolate(self.ts, self.base_dt, expiry)
        rate /= 100.
        return rate

    def get_df(self, expiry: datetime):
        """
        get discount factor 1/(1+r)^n
        """
        t = datetime_to_tenor(expiry, self.base_dt)
        rate = self.get_rate(expiry)
        df = 1.0 / math.pow(1.0 + rate, t)
        return df

    def get_forward_rate(self, st: datetime, et: datetime, dc: DayCountConvention = DayCountConvention.Actual360):
        dt = (et - st).days / dc.value
        return 0. if st == et else self.get_forward_rate_with_dt(st, et, dt)

    def get_forward_rate_with_dt(self, st: datetime, et: datetime, dt: float):
        """
        get forward rate [st, et] = (DF_st / DF_et - 1) / dt
        """
        rt = self.get_df(st) / self.get_df(et) - 1
        rate = rt / dt
        return rate


