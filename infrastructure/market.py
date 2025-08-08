from __future__ import annotations
from datetime import date, datetime
from ..analytics.symbology import OPTION_ROOT_FROM_TICKER
from ..constants.ccy import Ccy
from ..constants.day_count_convention import DayCountConvention
from ..constants.market_item_type import MarketItemType
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..infrastructure.eq_spot import EQSpot
from ..infrastructure.data_container import DataContainer
from ..infrastructure.rate_curve import RateCurve
from ..infrastructure.fixing_table import FixingTable
from ..infrastructure.fx_fwd_curve import FXFwdCurve
from ..infrastructure.fx_pair import FXPair
from ..infrastructure.fx_spot import FXSpot
from ..interface.imarket import IMarket
from ..interface.ishock import IShock
from ..interface.market_item import MarketItem
from ..interface.market_items.ivolatilitysurface import IVolatilitySurface
from ..tradable.option import Option
from ..tradable.varianceswap import VarianceSwap
from ..tradable.volswap import VolSwap
from ..tradable.voloption import VolOption
from ..tradable.future import Future
from typing import Union
import pandas as pd


class Market(IMarket):

    def __init__(self, base_datetime: datetime):
        self.storage = {}
        self.base_datetime = base_datetime

    @classmethod
    def create(cls, base_datetime: datetime, storage: dict):
        market = cls(base_datetime=base_datetime)
        for key, item in storage.items():
            market.add_item(key, item)
        return market

    def is_empty(self):
        return not len(self.storage)

    def add_item(self, key: str, item: MarketItem, cummulative=False):
        if key in self.storage:
            if cummulative:
                if isinstance(self.storage[key], list):
                    self.storage[key] = self.storage[key] + [item]
                else:
                    self.storage[key] = [self.storage[key], item]
            else:
                self.storage[key] = item
        else:
            self.storage[key] = item

    def has_item(self, key: str) -> bool:
        return key in self.storage

    def get_item(self, key: str) -> MarketItem:
        if not self.storage or key not in self.storage:
            raise Exception(f"Cannot find {key} in market on {self.base_datetime.strftime('%Y-%m-%d')}.")
        return self.storage[key]

    def get_base_datetime(self) -> datetime:
        return self.base_datetime

    def get_discount_rate(self, underlying: str, expiry_dt: datetime) -> float:
        curve_key = market_utils.create_discount_curve_key(underlying)
        if self.has_item(curve_key):
            return self.get_item(curve_key).get_discount_rate(expiry_dt)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            return self.get_item(vol_key).get_discount_rate(expiry_dt)

    def get_forward_discount_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        curve_key = market_utils.create_discount_curve_key(underlying)
        if self.has_item(curve_key):
            return self.get_item(curve_key).get_forward_discount_rate(st_dt, ed_dt)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            return self.get_item(vol_key).get_forward_discount_rate(st_dt, ed_dt)

    def get_borrow_rate(self, underlying: str, expiry_dt: datetime) -> float:
        curve_key = market_utils.create_borrow_curve_key(underlying)
        if self.has_item(curve_key):
            return self.get_item(curve_key).get_borrow_rate(expiry_dt)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            return self.get_item(vol_key).get_borrow_rate(expiry_dt)

    def get_forward_borrow_rate(self, underlying: str, st_dt: datetime, ed_dt: datetime) -> float:
        curve_key = market_utils.create_borrow_curve_key(underlying)
        if self.has_item(curve_key):
            return self.get_item(curve_key).get_forward_borrow_rate(st_dt, ed_dt)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            return self.get_item(vol_key).get_forward_borrow_rate(st_dt, ed_dt)

    def get_xccy_basis(self, ccy: Ccy, swap_term: str):
        key = market_utils.create_xccy_basis_key(ccy, swap_term)
        if self.get_item(key) is None:
            raise Exception(f"Missing {ccy.value} xccy basis on {self.base_datetime.strftime('%Y-%m-%d')} ")
        return self.get_item(key)[ccy]

    def get_fx_spot(self, fx_pair_name: Union[str, FXPair]) -> float:
        pair = FXPair.from_string(fx_pair_name) if isinstance(fx_pair_name, str) else fx_pair_name
        if pair.base_ccy == pair.term_ccy:
            return 1.0

        spot_key = market_utils.create_fx_spot_key(pair)
        vol_key = market_utils.create_fx_vol_surface_key(pair)
        inverse = False
        if not self.has_item(spot_key) and not self.has_item(vol_key):
            # try to inverse pair
            pair = pair.inverse()
            spot_key = market_utils.create_fx_spot_key(pair)
            vol_key = market_utils.create_fx_vol_surface_key(pair)
            inverse = True
        if self.has_item(spot_key):
            item = self.get_item(spot_key)
            if isinstance(item, FXSpot):
                spot = item.get_spot()
            else:
                if isinstance(item, DataContainer):
                    item = item.get_market_item(self.base_datetime)
                spot = item[pair.to_string()]
        else:
            spot = self.get_item(vol_key).get_spot(self.base_datetime)
        return 1 / spot if inverse else spot

    def get_fx_fixing(self, pair: FXPair, fixing_date: datetime) -> float:
        if pair.base_ccy == pair.term_ccy:
            return 1.0
        key = market_utils.create_fx_fixing_key(pair)
        inverse = False
        if not self.has_item(key):
            # try to inverse pair
            pair = pair.inverse()
            key = market_utils.create_fx_fixing_key(pair)
            inverse = True
        item = self.get_item(key)
        if fixing_date not in item.index:
            raise Exception(f"Missing {key} for fixing date {fixing_date.strftime('%Y-%m-%d')} "
                            f"on {self.base_datetime.strftime('%Y-%m-%d')} ")

        spot = item.loc[fixing_date, "fixing"]
        return 1 / spot if inverse else spot

    def get_fx_fwd_curve(self, pair: FXPair) -> RateCurve:
        spot = self.get_fx_spot(pair)
        # always try fwd point first
        pair_used = pair.clone()
        key = market_utils.create_fx_fwd_point_key(pair_used)
        found = self.has_item(key)
        if not found:
            pair_used = pair_used.inverse()
            key = market_utils.create_fx_fwd_point_key(pair_used)
            found = self.has_item(key)
        if not found:
            key = market_utils.create_fx_fwd_key(pair_used)
            found = self.has_item(key)
        if not found:
            pair_used = pair_used.inverse()
            key = market_utils.create_fx_fwd_key(pair_used)
            found = self.has_item(key)
        if not found:
            raise Exception(f"Not found FX Fwd Curve or FX Fwd Point Curve in market for {pair.to_string()}")

        ts = self.get_item(key)
        assert ts is not None
        inverse = key.split(".")[-1] != pair.to_string()
        use_fwd_points = key.split(".")[0] == MarketItemType.FXFORWARDPOINT.value
        if isinstance(ts, DataContainer):
            ts = ts.get_market_item(self.base_datetime)
        curve = FXFwdCurve(pair, ts, self.base_datetime, spot, inverse, use_fwd_points)
        return curve

    def get_fx_fwd(self, pair: FXPair, expiry_dt: datetime) -> float:
        if pair.base_ccy == pair.term_ccy:
            return 1.0
        curve = self.get_fx_fwd_curve(pair)
        fwd = curve.get_fwd_rate(expiry_dt)
        return fwd

    def get_underlying_price(self, derivative):
        """
        getting the price of the underlying of a given derivative from market
        the underlying can be a future
        @param derivative:
        @return:
        """
        if isinstance(derivative, Option):
            underlying = derivative.underlying
            if isinstance(underlying, str):
                return self.get_spot(underlying=underlying)
            elif isinstance(underlying, Future):
                return self.get_future_price(underlying.root, underlying.expiration)
            else:
                raise RuntimeError(
                    f"The underlying of derivative is of type {type(underlying)}, which is not yet supported")
        elif isinstance(derivative, VarianceSwap) or isinstance(derivative, VolSwap) or isinstance(derivative, VolOption):
            underlying = derivative.underlying
            if isinstance(underlying, str):
                return self.get_spot(underlying=underlying)
            else:
                raise RuntimeError(
                    f"The underlying of derivative is of type {type(underlying)}, which is not yet supported")
        elif isinstance(derivative, Future):
            return self.get_spot(underlying=derivative.underlying)
        else:
            raise RuntimeError(f'Derivative of type {type(derivative)} does not yet support get_underlying_price')

    def get_spot(self, underlying: str) -> float:
        key = market_utils.create_spot_key(underlying)
        if self.has_item(key):
            item = self.get_item(key)
            if isinstance(item, EQSpot):
                return item.get_spot()
            else:
                return self.get_item(key)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            if self.has_item(vol_key):
                return self.get_item(vol_key).get_spot(self.base_datetime)
            else:
                fx_vol_key = market_utils.create_fx_vol_surface_key(underlying)
                if self.has_item(fx_vol_key):
                    return self.get_item(fx_vol_key).get_spot(self.base_datetime)
                else:
                    raise RuntimeError(f"Cannot find spot for {underlying} on {self.base_datetime.strftime('%Y-%m-%d')}.")

    def get_dividend(self, underlying: str):
        dividend_key = market_utils.create_dividend_key(underlying)
        if self.has_item(dividend_key):
            return self.get_item(dividend_key)
        else:
            return None

    def get_corpaction(self, underlying: str):
        corpaction_key = market_utils.create_corpaction_key(underlying)
        if self.has_item(corpaction_key):
            return self.get_item(corpaction_key)
        else:
            return None

    def get_future_price(self, underlying: str, expiry_dt: datetime, future: Future = None) -> float:
        if future is not None:
            key = market_utils.create_future_data_container_key(future.root)
            if self.has_item(key):
                return self.get_item(key).get_future_price(future=future)
        key = market_utils.create_future_key(underlying, expiry_dt)
        if self.has_item(key):
            return self.get_item(key).get_future_price(self.base_datetime, expiry_dt)
        else:
            vol_key = market_utils.create_vol_surface_key(underlying)
            return self.get_item(vol_key).get_future_price(self.base_datetime, expiry_dt)

    def get_vol(self, underlying: str, expiry_dt: datetime, strike: float, strike_type=None) -> float:
        if strike_type is None:
            from ..constants.strike_type import StrikeType
            strike_type = StrikeType.K
        return self.get_vol_surface(underlying).get_vol(expiry_dt, strike, strike_type)

    def get_vol_type(self, underlying: str) -> UnderlyingType:
        return self.get_vol_surface(underlying).get_underlying_type()

    def get_vol_surface(self, underlying: str):
        vol_key = market_utils.create_vol_surface_key(underlying)
        if self.has_item(vol_key):
            return self.get_item(vol_key)
        else:
            fx_vol_key = market_utils.create_fx_vol_surface_key(underlying)
            if self.has_item(fx_vol_key):
                return self.get_item(fx_vol_key)
            else:
                raise RuntimeError(f"Cannot find vol surface for {underlying} on {self.base_datetime.strftime('%Y-%m-%d')}.")

    def get_cr_vol_surface(self, underlying: str):
        vol_key = market_utils.create_cr_vol_surface_key(underlying)
        return self.get_item(vol_key)[underlying]

    def get_cr_vol_surface_with_specialisation(self, underlying: str, specialisation: str = None):
        vol_key = market_utils.create_cr_vol_surface_key(underlying)
        return self.get_item(vol_key).get(underlying).get(specialisation)

    def get_fx_vol_surface(self, fx_pair: str):
        vol_key = market_utils.create_fx_vol_surface_key(fx_pair)
        item = self.get_item(vol_key)
        if isinstance(item, IVolatilitySurface):
            return item
        else:
            return self.get_item(vol_key)[fx_pair]

    def get_fx_sabr_vol_surface(self, fx_pair: str):
        vol_key = market_utils.create_fx_vol_surface_key(fx_pair)
        return self.get_item(vol_key)

    def get_swaption_vol_cube(self, currency: str):
        vol_key = market_utils.create_swaption_vol_cube_key(currency)
        return self.get_item(vol_key)

    def get_option_universe(self, root: str, return_as_dict=True):
        key = market_utils.create_option_data_container_key(root)
        if self.has_item(key):
            option_universe_item = self.get_item(key)
        else:
            key = market_utils.create_option_data_container_key(OPTION_ROOT_FROM_TICKER[root])
            option_universe_item = self.get_item(key)
        if isinstance(option_universe_item, list):
            # TODO: support return_as_dict=False
            assert return_as_dict
            option_universe = []
            for item in option_universe_item:
                option_universe = option_universe + item.get_option_universe(self.base_datetime, return_as_dict)
        else:
            option_universe = option_universe_item.get_option_universe(self.base_datetime, return_as_dict)
        return option_universe

    def get_option_data(self, option: Option, return_as_dict=True):
        key = market_utils.create_option_data_container_key(option.root)
        if self.has_item(key):
            option_data_item = self.get_item(key)
        else:
            key = market_utils.create_option_data_container_key(OPTION_ROOT_FROM_TICKER[option.root])
            option_data_item = self.get_item(key)
        if isinstance(option_data_item, list):
            # TODO: support return_as_dict=False
            assert return_as_dict
            option_data_all = []
            for item in option_data_item:
                x = item.get_option_data(self.base_datetime, option, return_as_dict)
                if x is not None:
                    option_data_all.append(x)
            assert len(option_data_all) == 1
            option_data = option_data_all[0]
        elif isinstance(option_data_item, dict):
            opts = {k: v for k, v in option_data_item.items() if k.name() == option.name()}
            if len(opts) == 0:
                print('missing option data on %s for %s' % (self.base_datetime, option.name()))
                assert 0 > 1
            else:
                opt_key = list(opts.keys())[0]
                assert opt_key.expiration == option.expiration
                assert opt_key.is_call == option.is_call
                assert opt_key.strike == option.strike
                assert opt_key.underlying == option.underlying

                option_data = opts[opt_key]
        else:
            option_data = option_data_item.get_option_data(self.base_datetime, option, return_as_dict)

        return option_data

    def get_future_universe(self, root: str, return_as_dict=True):
        key = market_utils.create_future_data_container_key(root)
        future_universe = self.get_item(key).get_future_universe(return_as_dict)
        return future_universe

    def get_lead_future(self, root: str, return_as_dict=False):
        key = market_utils.create_future_data_container_key(root)
        future = self.get_item(key).get_lead_future()
        return future

    def get_future_data(self, future: Future, return_as_dict=True):
        key = market_utils.create_future_data_container_key(future.root)
        future_data = self.get_item(key).get_future_data(future, return_as_dict)
        return future_data

    def get_future_intraday_data(self, future: Future):
        key = market_utils.create_future_intraday_data_container_key(future.root)
        future_data = self.get_item(key).get_future_intraday_data(future)
        return future_data

    def get_future_option_universe(self, future: Future, return_as_list_of_dict=False):
        key = market_utils.create_option_data_container_key(future.root)
        future_option_universe = self.get_item(key).get_option_universe(self.base_datetime, future,
                                                                        return_as_list_of_dict=return_as_list_of_dict)
        return future_option_universe

    def get_future_option_data(self, option: Option) -> dict:
        key = market_utils.create_option_data_container_key(option.root)
        future_option_data = self.get_item(key).get_option_data(self.base_datetime, option)
        return future_option_data

    def get_future_option_data_prev_eod(self, option: Option) -> dict:
        key = market_utils.create_option_data_container_key(option.root)
        future_option_data = self.get_item(key).get_option_data_prev_eod(self.base_datetime, option)
        return future_option_data

    def get_interest_rate(self, currency, tenor: int) -> dict:
        key = market_utils.create_interest_rate_data_container_key(currency)
        interest_rate = self.get_item(key).get_interest_rate(tenor)
        return interest_rate

    def get_forward_rates(self, currency: str, curve: str = None) -> float:
        key = market_utils.create_forward_rates_key(currency, curve)
        return self.get_item(key)

    def get_spot_rates(self, currency: str, curve: str) -> float:
        if curve.upper() == "DEFAULT":
            key = market_utils.create_spot_rates_key(currency, "")
            keys = [x for x in self.storage.keys() if key in x]
            if len(keys) == 0:
                raise Exception(f"Missing {currency} spot rate on {self.base_datetime.strftime('%Y-%m-%d')}")
            elif len(keys) > 1:
                raise Exception(f"Found multiple {currency} spot rate keys {','.join(keys)} on {self.base_datetime.strftime('%Y-%m-%d')}")
            key = keys[0]
        else:
            key = market_utils.create_spot_rates_key(currency, curve)
        return self.get_item(key)

    def get_rate_fixing(self, ccy: Ccy, tenor: str, fixing_date: datetime) -> float:
        key = market_utils.create_rate_fixing_key(ccy, tenor)
        item = self.get_item(key)
        if fixing_date not in item.index:
            raise Exception(f"Missing {key} for fixing date {fixing_date.strftime('%Y-%m-%d')} "
                            f"on {self.base_datetime.strftime('%Y-%m-%d')} ")
        rate = self.get_item(key).loc[fixing_date, "fixing"] / 100.
        return rate

    def get_spot_rate_curve(self, ccy: Ccy, curve: str) -> RateCurve:
        ts = self.get_spot_rates(ccy.value, curve)
        if isinstance(ts, DataContainer):
            ts = ts.get_market_item(self.base_datetime)
        assert isinstance(ts, MarketItem)
        # now ts is a market item
        # TODO: why do we need RateCurve rather than put any extra function it has inside the market item?
        rate_curve = RateCurve(ccy, ts.data_dict, self.base_datetime)
        return rate_curve

    def get_forward_rate_curve(self, ccy: Ccy, curve: str) -> RateCurve:
        ts = self.get_forward_rates(ccy.value, curve)
        if isinstance(ts, DataContainer):
            ts = ts.get_market_item(self.base_datetime)
        ts1 = {}
        for k, v in ts.items():
            if isinstance(v, dict):
                assert len(v.values()) == 1
                ts1[k] = list(v.values())[0]
            else:
                ts1[k] = v
        rate_curve = RateCurve(ccy, ts1, self.base_datetime)
        return rate_curve

    def get_df(self, ccy: Ccy, curve: str, expiry: datetime) -> float:
        """
        get discount factor 1/(1+r)^n
        """
        if expiry == self.base_datetime:
            return 1.
        rate_curve = self.get_spot_rate_curve(ccy, curve)
        return rate_curve.get_df(expiry)

    def get_forward_rate(self, ccy: Ccy, curve: str, st: datetime, et: datetime = None,
                         dc: DayCountConvention = DayCountConvention.Actual360) -> float:
        """
        get forward rate [st, et] = (DF_st / DF_et - 1)/ dt from spot rate curve
        or return rate directly from forward rate curve
        """
        if et is None:
            rate_curve = self.get_forward_rate_curve(ccy, curve)
            rate = rate_curve.get_rate(st)
        else:
            rate_curve = self.get_spot_rate_curve(ccy, curve)
            rate = rate_curve.get_forward_rate(st, et, dc)
        return rate

    def get_portfolio(self, portfolio_name: str):
        key = market_utils.create_portfolio_key(portfolio_name)
        return self.get_item(key)

    def get_bond_yield(self, ticker: str):
        key = market_utils.create_bond_yield_key(ticker)
        return self.get_item(key)[ticker]

    def has_fixing(self, underlying: str, obs_date: date):
        key = market_utils.create_fixing_table_key()
        if not self.has_item(key):
            return False
        fixing_table = self.get_item(key)
        return fixing_table.has_fixing(underlying, obs_date)

    def get_fixing_from_fixing_table(self, underlying: str, obs_date: date):
        key = market_utils.create_fixing_table_key()
        if not self.has_item(key):
            raise Exception(f"Missing fixing table in market on {self.base_datetime.strftime('%Y-%m-%d')} ")
        fixing_table = self.get_item(key)
        fixing = fixing_table.get_fixing(underlying, obs_date)
        return fixing

    def add_fixing_into_fixing_table(self, underlying: str, obs_date: date, fixing: float):
        key = market_utils.create_fixing_table_key()
        if self.has_item(key):
            fixing_table = self.get_item(key)
            fixing_table.add_fixing(underlying, obs_date, fixing)
        else:
            fixing_table = pd.DataFrame([[obs_date, underlying, fixing]], columns=["date", "underlying", "fixing"])
            self.storage[key] = FixingTable(fixing_table)

    def merge_fixing_table(self, other_fixing_table: FixingTable):
        key = market_utils.create_fixing_table_key()
        if self.has_item(key):
            fixing_table = self.get_item(key)
            fixing_table.merge_fixing_table(other_fixing_table)
        else:
            self.storage[key] = other_fixing_table

    def get_holidays(self, code: str, start_date: date, end_date: date):
        key = market_utils.create_holiday_center_key()
        if not self.has_item(key):
            return []
        holiday_center = self.get_item(key)
        return holiday_center.get_holidays(code, start_date, end_date)

    def get_correlation_matrix(self, und_list):
        key = market_utils.create_correlation_matrix(und_list)
        return self.get_item(key)

    def apply(self, shock_map: {str: [IShock]}, **kwargs) -> Market:
        new_market = Market(base_datetime=self.base_datetime)
        # apply market bump first
        non_market_shock_map = {}
        for key, shocks in shock_map.items():
            if key == "":
                if len(shocks) != 1:
                    raise Exception("unsupported multiple undefined shocks")

                if not shocks[0].is_market_shock():
                    raise Exception("not market bump")

                storage = self.storage if new_market.is_empty() else new_market.storage
                for key, item in storage.items():
                    if isinstance(item, MarketItem):
                        new_item = item.apply(shocks, self, **kwargs)
                    else:
                        new_item = item.clone()
                    new_market.add_item(key, new_item)
                # set market base datetime
                new_market.base_datetime = shocks[0].get_base_datetime()
            elif len(shocks):
                non_market_shock_map[key] = shocks

        storage = self.storage if new_market.is_empty() else new_market.storage
        for key, item in storage.items():
            if key in non_market_shock_map.keys():
                if not all(shock_map[key]):
                    raise Exception("Found invalid shock")
                new_item = item.apply(shock_map[key], self, **kwargs)
            else:
                new_item = item
            new_market.add_item(key, new_item)
        return new_market

    def clone(self):
        new_market = Market(base_datetime=self.base_datetime)
        for key, item in self.storage.items():
            new_market.add_item(key, item.clone())
        return new_market

    def extend(self, mkt: IMarket):
        new_market = self.clone()
        for key, item in mkt.storage.items():
            if not new_market.has_item(key):
                new_market.add_item(key, item)
        return new_market
