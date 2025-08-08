from datetime import datetime
from enum import Enum
from ..interface.ishock import IShock, ShockType


class SpotShock(IShock):
    """
    how to shock equity spot, only applicable when vola_surface is VolSurfaceEquity
    """
    def __init__(self, size, spot_beta=None, benchmark_underlying="SPX Index", method='percentage', sticky_strike=True, pvol=None):
        super().__init__(ShockType.SPOTSHOCK)
        self.size = size
        self.spot_beta = spot_beta
        self.benchmark_underlying = benchmark_underlying
        self.method = method
        self.sticky_strike = sticky_strike
        self.pvol = pvol

    def is_market_shock(self):
        return False


class VolFutureShock(IShock):
    """
    how to shock VIX/V2X future, only applicable when vola_surface is VolSurfaceFutures
    """
    def __init__(self, method, parameters, sticky_strike=True, benchmark_underlying="SPX Index", benchmark_parameters=None, use_forward_vol_level=False,
                 use_shifted_spot=False, spot_shift_size=None):
        super().__init__(ShockType.VOLFUTURESHOCK)
        self.method = method
        self.parameters = parameters
        self.sticky_strike = sticky_strike
        self.benchmark_underlying = benchmark_underlying
        self.benchmark_parameters = benchmark_parameters
        self.use_forward_vol_level = use_forward_vol_level
        self.use_shifted_spot = use_shifted_spot
        self.spot_shift_size = spot_shift_size

    def is_market_shock(self):
        return False


class VolShock(IShock):
    """
    how to shock volatility surface, applicable for both VolSurfaceEquity and VolSurfaceFutures
    """
    def __init__(self, method, parameters, vol_beta=None, benchmark_underlying="SPX Index", min_vol0=0.03, sticky_strike=False):
        super().__init__(ShockType.VOLSHOCK)
        self.method = method
        self.parameters = parameters
        self.vol_beta = vol_beta
        self.benchmark_underlying = benchmark_underlying
        self.min_vol0 = min_vol0
        self.sticky_strike = sticky_strike

    def is_market_shock(self):
        return False


class DatetimeShiftType(Enum):
    STICKY_TENOR = "Sticky Tenor"               # shift vol surface such that 1M tenor in 1M is 1M tenor on base date
    STICKY_DATE = "Sticky Date"                 # shift vol surface such that 1M tenor in 1M is 2M tenor on base date
    VOLA_STICKY_TENOR = "Vola Sticky Tenor"     # using vola fixed by tenor method based on normalised strike
    VOLA_STICKY_DATE = "Vola Sticky Date"       # using vola fixed by date method based on normalised strike


class DatetimeShift(IShock):
    def __init__(self, shifted_datetime: datetime, shift_type: DatetimeShiftType, roll_future_price: bool,
                 shift_days: int=None):
        super().__init__(ShockType.DATETIMESHIFT)
        self.shifted_datetime = shifted_datetime
        self.shift_type = shift_type
        self.roll_future_price = roll_future_price
        self.shift_days = shift_days

    def is_market_shock(self):
        return True

    def get_base_datetime(self):
        return self.shifted_datetime

class DivShock(IShock):
    """
    how to shock equity div rate, only applicable when vola_surface is VolSurfaceEquity
    """
    def __init__(self, size_bps = 1, method = 'level'):
        super().__init__(ShockType.DIVSHOCK)
        self.size_bps = size_bps
        self.method = method

    def is_market_shock(self):
        return False

class RateShock(IShock):
    """
    how to shock equity interst rate, only applicable when vola_surface is VolSurfaceEquity
    """
    def __init__(self, size_bps=1, method='level'):
        super().__init__(ShockType.RATESHOCK)
        self.size_bps = size_bps
        self.method = method

    def is_market_shock(self):
        return False

class RepoShock(IShock):
    """
    how to shock equity borrow rate, only applicable when vola_surface is VolSurfaceEquity
    """
    def __init__(self, size_bps = 1, method = 'level'):
        super().__init__(ShockType.REPOSHOCK)
        self.size_bps = size_bps
        self.method = method

    def is_market_shock(self):
        return False