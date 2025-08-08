from __future__ import annotations
from datetime import datetime, date
from ..constants.market_item_type import MarketItemType
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem


class FMarket(Market):
    """
    Future Market contains simulation market data:
    1. equity spot
    2. equity future spot
    Note that more simulated data can be added later, e.g. volatility
    """

    def __init__(self, base_datetime, original_market: Market, sim_data,
                 sim_dates):
        self.base_datetime = base_datetime
        self.original_market = original_market
        self.sim_data = sim_data
        self.sim_dates = sim_dates
        # index of date and path for current state
        self.current_date = None
        self.current_path = None

        self.use_cache = True
        self.cached_spot = {}
        self.cached_future_price = {}

    def set_current_state(self, date, path: int=0):
        self.base_datetime = date
        self.current_date = self.sim_dates.index(date)
        self.current_path = path

    def add_item(self, key: str, item: MarketItem):
        raise

    def has_item(self, key: str) -> bool:
        return self.original_market.has_item(key)

    def get_item(self, key: str) -> MarketItem:
        raise

    def get_discount_rate(self, underlying: str, expiry_dt: datetime) -> float:
        return self.original_market.get_discount_rate(underlying, expiry_dt)

    def get_forward_discount_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        return self.original_market.get_forward_discount_rate(underlying, st_dt, ed_dt)

    def get_borrow_rate(self, underlying: str, expiry_dt: datetime) -> float:
        return self.original_market.get_borrow_rate(underlying, expiry_dt)

    def get_forward_borrow_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        return self.original_market.get_forward_borrow_rate(underlying, st_dt, ed_dt)

    def get_original_spot(self, underlying: str) -> float:
        cache_key = underlying
        if self.use_cache and cache_key in self.cached_spot:
            spot = self.cached_spot[cache_key]
        else:
            key = market_utils.create_spot_key(underlying)
            base_date = self.original_market.get_base_datetime()
            if self.has_item(key):
                spot = self.get_item(key).get_spot(base_date)
            else:
                vol_key = market_utils.create_vol_surface_key(underlying)
                volsurface = self.original_market.get_item(vol_key)
                spot = volsurface.get_spot(base_date)
            if self.use_cache:
                self.cached_spot[cache_key] = spot
        return spot

    def get_spot(self, underlying: str) -> float:
        spot = self.get_original_spot(underlying)
        if MarketItemType.SPOT in self.sim_data:
            m = self.sim_data[MarketItemType.SPOT][self.current_path][self.current_date]
            spot *= m
        return spot

    def get_future_price(self, underlying: str, expiry_dt: datetime, **kwargs) -> float:
        cache_key = (underlying, expiry_dt)
        if self.use_cache and cache_key in self.cached_future_price:
            spot = self.cached_future_price[cache_key]
        else:
            key = market_utils.create_future_key(underlying, expiry_dt)
            base_date = self.original_market.get_base_datetime()
            if self.has_item(key):
                spot = self.get_item(key).get_future_price(base_date, expiry_dt)
            else:
                vol_key = market_utils.create_vol_surface_key(underlying)
                volsurface = self.original_market.get_item(vol_key)

                # try to find future prices observed in original market. If not found, get spot instead
                if expiry_dt in volsurface.future_prices:
                    spot = volsurface.future_prices[expiry_dt]
                else:
                    spot = volsurface.get_future_price(base_date, volsurface.get_base_datetime())
            if self.use_cache:
                self.cached_future_price[cache_key] = spot

        if MarketItemType.SPOT in self.sim_data:
            m = self.sim_data[MarketItemType.SPOT][self.current_path][self.current_date]
            spot *= m
        return spot

    def has_fixing(self, underlying: str, obs_date: date):
        found = True
        if MarketItemType.SPOT in self.sim_data:
            dt_idx = len([dt for dt in self.sim_dates if dt.date() <= obs_date]) - 1
            if dt_idx < 0 or dt_idx > self.current_date:
                found = False
        return found

    def get_fixing_from_fixing_table(self, underlying: str, obs_date: date):
        spot = self.get_original_spot(underlying)
        if MarketItemType.SPOT in self.sim_data:
            dt_idx = len([dt for dt in self.sim_dates if dt.date() <= obs_date]) - 1
            assert dt_idx >= 0 and dt_idx <= self.current_date
            m = self.sim_data[MarketItemType.SPOT][self.current_path][dt_idx]
            spot *= m
        return spot

    def get_vol(self, underlying: str, expiry_dt: datetime, strike: float, strike_type=None) -> float:
        return self.original_market.get_vol(underlying, expiry_dt, strike, strike_type)

    def get_vol_type(self, underlying: str) -> UnderlyingType:
        return self.original_market.get_vol_type(underlying)

    def get_vol_surface(self, underlying: str):
        return self.original_market.get_vol_surface(underlying)

    def get_option_universe(self, underlying: str):
        raise

    def apply(self, shock_map: {str: [IShock]}, **kwargs) -> Market:
        for key, shocks in shock_map.items():
            if key != "":
                raise Exception("Not support non market shock")
        self.original_market = self.original_market.apply(shock_map)
        return self

