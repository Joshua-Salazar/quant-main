from datetime import datetime
from ..analytics.utils import float_equal, find_bracket_bounds
from ..dates import utils as dates_utils
from ..constants.strike_type import StrikeType
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..interface.ishock import IShock, ShockType
from ..interface.market_item import MarketItem
from ..interface.market_items.ispot import ISpot
from ..interface.market_items.ivolatilitysurface import IVolatilitySurface
import numpy as np


class BSVol(MarketItem, ISpot, IVolatilitySurface):

    def __init__(self, underlying_type: UnderlyingType, underlying: str, ts: dict, base_dt: datetime, spot: float):
        self.underlying_type = underlying_type
        self.underlying = underlying
        self.market_key = market_utils.create_vol_surface_key(underlying)
        self.base_dt = base_dt
        self.ts = {dates_utils.tenor_to_years(tenor) if isinstance(tenor, str) else tenor: value for tenor, value in ts.items()}
        self.spot = spot

    def clone(self):
        return BSVol(self.underlying_type, self.underlying, self.ts, self.base_dt, self.spot)

    def get_base_datetime(self) -> datetime:
        return self.base_dt

    def get_market_key(self):
        return self.market_key

    def get_spot(self, base_date: datetime) -> float:
        assert base_date == self.base_dt
        return self.spot

    def get_underlying_type(self) -> UnderlyingType:
        return self.underlying_type

    def get_vol(self, expiry_dt: datetime, strike: float, strike_type: StrikeType = StrikeType.K) -> float:
        expiry_yrs = dates_utils.datetime_to_tenor(expiry_dt, self.base_dt)
        (lb, ub) = find_bracket_bounds(sorted(list(self.ts.keys())), expiry_yrs)
        if lb == -float('inf') and ub == float('inf'):
            raise Exception(f'Cannot find vol slice on {self.base_dt.strftime("%Y-%m-%d")} for expiry {str(expiry_dt)}')

        if ub == float('inf'):
            ub = lb
        if lb == -float('inf'):
            lb = ub

        vol_lb = self.ts[lb]
        vol_ub = self.ts[ub]
        if float_equal(lb, ub):
            return vol_lb
        vol = np.sqrt(vol_lb * vol_lb + (expiry_yrs - lb) * (vol_ub * vol_ub - vol_lb * vol_lb) / (ub - lb))
        return vol

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        cloned_vol = self.clone()
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                # return same vol for now.
                continue
            else:
                raise Exception(f"Not implemented yet: {shock.type.value}")
        return cloned_vol







