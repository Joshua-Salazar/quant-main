from datetime import datetime
from ..analytics import fx_sabr
from ..analytics.utils import float_equal, find_bracket_bounds
from ..constants.ccy import Ccy
from ..constants.strike_type import StrikeType
from ..constants.underlying_type import UnderlyingType
from ..dates.utils import datetime_to_tenor
from ..infrastructure import market_utils
from ..infrastructure.fx_pair import FXPair
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem
from ..interface.market_items.ispot import ISpot
from ..interface.market_items.ivolatilitysurface import IVolatilitySurface
import numpy as np
import os
import pandas as pd


class FXSOLSABRVolSurface(MarketItem, ISpot, IVolatilitySurface):
    FX_SOL_SABR_SURFACE_FODLER = "/misc/Traders/Solutions/FX_SABR_Surface/"

    def __init__(self, base_ccy: Ccy, term_ccy: Ccy, base_date: datetime, inversed=False):
        self.pair = FXPair(base_ccy, term_ccy)
        self.base_date = base_date
        self.is_inversed = inversed
        self.ts = None
        self.quotes = None
        self.spot = None
        self.forwards = None
        self.load_model_params()

    def load_model_params(self):
        file = os.path.join(self.FX_SOL_SABR_SURFACE_FODLER, f"{self.pair.to_string()}/{self.pair.to_string()}_{self.base_date.strftime('%Y%m%d')}.csv")
        if not os.path.exists(file):
            raise Exception(f"Not found vol surface {self.pair.to_string()} on {self.base_date.strftime('%Y%m%d')} in {file}")
        self.ts = pd.read_csv(file)
        self.ts["expiry"] = pd.to_datetime(self.ts["expiry"])
        file = os.path.join(self.FX_SOL_SABR_SURFACE_FODLER, f"{self.pair.to_string()}/{self.pair.to_string()}_{self.base_date.strftime('%Y%m%d')}_quotes.csv")
        if not os.path.exists(file):
            raise Exception(f"Not found vol surface {self.pair.to_string()} on {self.base_date.strftime('%Y%m%d')} in {file}")
        self.quotes = pd.read_csv(file)
        self.quotes["expiry"] = pd.to_datetime(self.quotes["expiry"])
        self.spot = self.ts["spot"].values[0]
        self.forwards = self.ts[["tte", "fwd"]].set_index("tte").to_dict()["fwd"]

    def get_base_datetime(self) -> datetime:
        return self.base_date

    def get_underlying_type(self) -> UnderlyingType:
        return UnderlyingType.FX

    def get_tte_bounds(self, tte):
        (lb, ub) = find_bracket_bounds(self.ts["tte"].sort_values().tolist(), tte)
        if lb == -float('inf') and ub == float('inf'):
            raise Exception(f"Cannot find vol slice on {self.base_date.strftime('%Y%m%d')} for {tte} yrs expiry")
        # flat expiry extrapolation
        if ub == float('inf'):
            ub = lb
        if lb == -float('inf'):
            lb = ub
        return lb, ub

    def is_quote(self, tenor, delta):
        res = self.quotes[(self.quotes["tenor"] == tenor)&(self.quotes["delta"] == delta)]
        return not res.empty

    def get_quote(self, tenor, delta):
        res = self.quotes[(self.quotes["tenor"] == tenor)&(self.quotes["delta"] == delta)]
        if res.empty:
            raise Exception(f"Not found quotes for {tenor} {delta} on {self.base_date.strftime('%Y%m%d')}")
        else:
            return res.iloc[0]

    def get_vol(self, expiry: datetime, strike: float, strike_type: StrikeType = StrikeType.K) -> float:
        assert strike_type == StrikeType.K
        tte = (expiry.date() - self.base_date.date()).days / 360
        (lb, ub) = self.get_tte_bounds(tte)
        ts = self.ts.set_index("tte")
        if float_equal(lb, ub):
            slice_t = ts.loc[lb]
            vol = fx_sabr.get_vol_from_raw_params(k=strike, fwd=slice_t["fwd"], tte=lb, sigma0=slice_t["sigma0"], vov=slice_t["vov"], rho=slice_t["rho"])
            return vol

        # interpolate sabr raw parameter
        slice_t0 = ts.loc[lb]
        slice_t1 = ts.loc[ub]
        sigma0_t0 = slice_t0["sigma0"]
        sigma0_t1 = slice_t1["sigma0"]
        vov_t0 = slice_t0["vov"]
        vov_t1 = slice_t1["vov"]
        rho_t0 = slice_t0["rho"]
        rho_t1 = slice_t1["rho"]

        sigma0 = np.sqrt(sigma0_t0 * sigma0_t0 + (tte - lb) * (sigma0_t1 * sigma0_t1 - sigma0_t0 * sigma0_t0) / (ub - lb))
        ratio = (tte - lb) / (ub - lb)
        vov = vov_t0 + ratio * (vov_t1 - vov_t0)
        rho = rho_t0 + ratio * (rho_t1 - rho_t0)

        fwd_t0 = slice_t0["fwd"]
        fwd_t1 = slice_t1["fwd"]
        fwd = fwd_t0 + ratio * (fwd_t1 - fwd_t0)
        vol = fx_sabr.get_vol_from_raw_params(k=strike, fwd=fwd, tte=tte, sigma0=sigma0, vov=vov, rho=rho)
        return vol

    def get_spot(self, base_date: datetime) -> float:
        return self.spot

    def get_forward(self, expiry: datetime) -> float:
        tte = datetime_to_tenor(expiry, self.base_date)
        (lb, ub) = self.get_tte_bounds(tte)
        ts = self.ts.set_index("tte")
        if float_equal(lb, ub):
            fwd = ts.loc[lb]["fwd"]
        else:
            fwd_t0 = ts.loc[lb]["fwd"]
            fwd_t1 = ts.loc[ub]["fwd"]
            ratio = (tte - lb) / (ub - lb)
            fwd = fwd_t0 + ratio * (fwd_t1 - fwd_t0)
        return fwd

    def get_market_key(self):
        return market_utils.create_fx_vol_surface_key(self.pair)

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        pass
