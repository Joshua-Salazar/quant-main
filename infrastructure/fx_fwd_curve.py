from datetime import datetime
from ..analytics import swaptions
from ..data.market import get_expiry_in_year
from ..constants.ccy import Ccy
from ..infrastructure.fx_pair import FXPair


class FXFwdCurve:
    def __init__(self, pair: FXPair, ts: dict, base_dt: datetime, fx_spot: float, inverse: bool, use_fwd_points: bool):
        """
        :param pair: target pair
        :param ts: term structure for either forward curve or forward basis on target pair or inversed target pair
        :param base_dt: base datetime
        :param fx_spot: fx spot on target pair
        :param inverse: indicate if term structure on target pair or inversed target pair
        :param use_fwd_points: indicate if term structure for forward curve or forward basis curve
        """
        self.pair = pair
        self.base_dt = base_dt
        if use_fwd_points:
            self.ts = {dt: 1./(1./fx_spot + value) if inverse else fx_spot + value for dt, value in ts.items()}
        else:
            # we avoid short tenor, e.g. 1D, 1W, 2W as those tenors quiet noisey back to 2018-09-01 for EURUSD
            # for example.
            if (self.pair == FXPair(Ccy.EUR, Ccy.USD) or self.pair == FXPair(Ccy.USD, Ccy.EUR))\
                    and self.base_dt < datetime(2018, 9, 1):
                short_tenors = [get_expiry_in_year(tenor[:-1], tenor[-1]) for tenor in ["1D", "1W", "2W"]]
                self.ts = {dt: 1./value if inverse else value for dt, value in ts.items() if dt not in short_tenors}
            else:
                self.ts = {dt: 1./value if inverse else value for dt, value in ts.items()}
        self.ts[0] = fx_spot  # add spot into term structure to tie up the front end of curve

    def get_fwd_rate(self, expiry: datetime):
        fwd = swaptions.rate_curve_interpolate(self.ts, self.base_dt, expiry)
        return fwd
