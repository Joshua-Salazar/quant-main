from ..infrastructure import market_utils
from ..infrastructure.shock import DatetimeShift, DatetimeShiftType, DivShock, RateShock, RepoShock, SpotShock, VolShock
from ..analytics.symbology import currency_from_ticker
from ..constants.numerical_greeks import NumericalGreeks
from ..dates.utils import add_business_days
from ..tradable.autocallable import AutoCallable
from dataclasses import dataclass
import pandas as pd


@dataclass
class NumericalGreeksShockSize:
    DeltaPercentageShockSize: float = 0.01  # 0.03  # 1% percentage shock size
    DivPerctShockSize: float = 25 # 25 bps div yield shock size (it's 0.25% div yield)
    RepoLevelShockSize: float = 25  # 25 bps repo shock size, test showed 25 bps bump size has closer result with bank
    RhoLevelShockSize: float = 25  # 25 bps rate shock size, test showed 25 bps bump size has closer result with bank 
    ThetaShockSize: float = 1  # 1 business day move
    VegaLevelShockSize: float = 0.01  # 1 vol shock size


@dataclass
class NumericalGreeksScalingFactor:
    VolBumpScalingFactor: float = 0.01


class DeltaShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und, sticky_strike=True, pvol=None, one_side=False):
        vol_key = market_utils.create_vol_surface_key(und)
        shock_size = NumericalGreeksShockSize.DeltaPercentageShockSize
        shock_up = {vol_key: [SpotShock(shock_size, method="percentage", sticky_strike=sticky_strike, pvol=pvol)]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {vol_key: [SpotShock(-shock_size, method="percentage", sticky_strike=sticky_strike, pvol=pvol)]}
            market_down = market.apply(shock_down)
        return market_up, market_down


class VegaShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und, use_time_weighted_vega, one_side=False, use_sticky_strike=False):
        vol_key = market_utils.create_vol_surface_key(und)
        shock_type = "time_weighted_percentage" if use_time_weighted_vega else "level"
        shock_up = {vol_key: [VolShock(shock_type, NumericalGreeksShockSize.VegaLevelShockSize, sticky_strike=use_sticky_strike)]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {vol_key: [VolShock(shock_type, -NumericalGreeksShockSize.VegaLevelShockSize, sticky_strike=use_sticky_strike)]}
            market_down = market.apply(shock_down)
        return market_up, market_down


class VannaShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und):
        vol_key = market_utils.create_vol_surface_key(und)
        spot_shock_size = NumericalGreeksShockSize.DeltaPercentageShockSize
        vol_shock_size = NumericalGreeksShockSize.VegaLevelShockSize
        shock_up_up = {vol_key: [SpotShock(spot_shock_size, method="percentage"), VolShock("level", vol_shock_size)]}
        shock_up_down = {vol_key: [SpotShock(spot_shock_size, method="percentage"), VolShock("level", -vol_shock_size)]}
        shock_down_up = {vol_key: [SpotShock(-spot_shock_size, method="percentage"), VolShock("level", vol_shock_size)]}
        shock_down_down = {vol_key: [SpotShock(-spot_shock_size, method="percentage"), VolShock("level", -vol_shock_size)]}
        market_up_up = market.apply(shock_up_up)
        market_up_down = market.apply(shock_up_down)
        market_down_up = market.apply(shock_down_up)
        market_down_down = market.apply(shock_down_down)
        return market_up_up, market_up_down, market_down_up, market_down_down

class EqDivShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und, one_side = False):
        vol_key = market_utils.create_vol_surface_key(und)
        shock_size = NumericalGreeksShockSize.DivPerctShockSize
        shock_up = {vol_key: [DivShock(size_bps = shock_size, method = "percentage")]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {vol_key: [DivShock(size_bps = - shock_size, method = "percentage")]}
            market_down = market.apply(shock_down)
        return market_up, market_down

class EqRepoShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und, one_side = False):
        vol_key = market_utils.create_vol_surface_key(und)
        shock_size = NumericalGreeksShockSize.RepoLevelShockSize
        shock_up = {vol_key: [RepoShock(size_bps = shock_size, method = "level")]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {vol_key: [RepoShock(size_bps = -shock_size, method = "level")]}
            market_down = market.apply(shock_down)
        return market_up, market_down

class EqRhoShockedMarket:
    @staticmethod
    def get_shocked_markets(market, und, one_side=False):
        vol_key = market_utils.create_vol_surface_key(und)
        shock_size = NumericalGreeksShockSize.RhoLevelShockSize
        shock_up = {vol_key: [RateShock(size_bps=shock_size, method="level")]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {vol_key: [RateShock(size_bps=-shock_size, method="level")]}
            market_down = market.apply(shock_down)
        return market_up, market_down


class RateRhoShockedMarket:
    @staticmethod
    def get_shocked_markets(market, ccy, curve_name, one_side=False):
        rate_key = market_utils.create_spot_rates_key(currency=ccy, curve=curve_name)
        shock_size = NumericalGreeksShockSize.RhoLevelShockSize
        shock_up = {rate_key: [RateShock(size_bps=shock_size, method="level")]}
        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {rate_key: [RateShock(size_bps=-shock_size, method="level")]}
            market_down = market.apply(shock_down)
        return market_up, market_down


# apply a list of rate curve and eq rate curve shock
class RhoShockedMarket:
    @staticmethod
    def get_shocked_markets(market, ccy_list, curve_name_list, und_list, one_side=False):
        shock_up = {}
        shock_size = NumericalGreeksShockSize.RhoLevelShockSize
        for ccy, curve_name in zip(ccy_list, curve_name_list):
            rate_key = market_utils.create_spot_rates_key(currency=ccy, curve=curve_name)
            shock_up[rate_key] = [RateShock(size_bps=shock_size, method="level")]
        for und in und_list:
            vol_key = market_utils.create_vol_surface_key(und)
            shock_up[vol_key] = [RateShock(size_bps=shock_size, method="level")]

        market_up = market.apply(shock_up)
        if one_side:
            market_down = None
        else:
            shock_down = {}
            for ccy, curve_name in zip(ccy_list, curve_name_list):
                rate_key = market_utils.create_spot_rates_key(currency=ccy, curve=curve_name)
                shock_down[rate_key] = [RateShock(size_bps=-shock_size, method="level")]
            for und in und_list:
                vol_key = market_utils.create_vol_surface_key(und)
                shock_down[vol_key] = [RateShock(size_bps=-shock_size, method="level")]
            market_down = market.apply(shock_down)
        return market_up, market_down


class ThetaShockedMarket:
    @staticmethod
    def get_shocked_markets(market):
        shock_size = NumericalGreeksShockSize.ThetaShockSize
        holiday_center = market.get_item(market_utils.create_holiday_center_key())
        hols = []
        for k, v in holiday_center.holidays.items():
            hols += v
        hols = list(set(hols))
        base_dt = market.base_datetime
        dt_plus = add_business_days(base_dt, shock_size, holidays=hols)
        shock_plus = {'': [DatetimeShift(shifted_datetime=dt_plus, shift_type=DatetimeShiftType.VOLA_STICKY_TENOR, roll_future_price=False)]}
        market_plus = market.apply(shock_plus)
        return market_plus


def calculate_numerical_greeks(tradable, market, greeks_list, valuer=None, one_side=False,
                               same_vol_grid_in_delta=False, sticky_strike_in_vega=False, **kwargs):
    unds = tradable.get_underlyings()
    res = {}
    tmp_res = {}  # temp res of pv or market to share cross greeks
    if "origin_price" in kwargs:
        tmp_res["origin_price"] = kwargs["origin_price"]

    # calculate spot greeks
    calc_delta = NumericalGreeks.Delta.value in greeks_list
    calc_gamma = NumericalGreeks.Gamma.value in greeks_list
    if calc_delta or calc_gamma:
        shock_size = NumericalGreeksShockSize.DeltaPercentageShockSize
        if same_vol_grid_in_delta:
            vol_grid_spot = {und: market.get_spot(und) for und in unds}
        for und in unds:
            ccy = currency_from_ticker(und) # todo find the better way to determine currency
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            sticky_strike = kwargs.get("sticky_strike", True)
            pvol = kwargs.get("pvol", None)
            one_side_market = one_side and not calc_gamma # two side markets required for two side delta or gamma
            market_up, market_down = DeltaShockedMarket.get_shocked_markets(market, und, sticky_strike=sticky_strike, pvol=pvol, one_side=one_side_market)
            stock_price = market.get_spot(und)
            price_kwargs = {"trade_suffix": f"_{und}_spot_up"}
            if same_vol_grid_in_delta:
                price_kwargs["vol_grid_spot"] = vol_grid_spot
            price_up = tradable.price(market_up, valuer=valuer, calc_types="price", **price_kwargs)
            if one_side_market:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price", trade_suffix="_base")
                price = tmp_res["origin_price"]
                delta = (price_up - price) / shock_size
            else:
                if f"spot_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_spot_down"}
                    if same_vol_grid_in_delta:
                        price_kwargs["vol_grid_spot"] = vol_grid_spot
                    tmp_res[f"spot_down_price#{und}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"spot_down_price#{und}"]
                delta = (price_up - price_down) / (2 * shock_size)
            if NumericalGreeks.Delta.value not in res:
                res[NumericalGreeks.Delta.value] = {}
            res[NumericalGreeks.Delta.value][f"Delta#{und}"] = delta / stock_price
            res[NumericalGreeks.Delta.value][f"DeltaUSD#{und}"] = delta * usd_conv
            res[NumericalGreeks.Delta.value][f"SpotRef#{und}"] = stock_price
            if calc_gamma:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price", trade_suffix="_base")
                price = tmp_res["origin_price"]
                if f"spot_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_spot_down"}
                    if same_vol_grid_in_delta:
                        price_kwargs["vol_grid_spot"] = vol_grid_spot
                    tmp_res[f"spot_down_price#{und}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"spot_down_price#{und}"]
                gamma = ((price_up + price_down) - 2 * price) / (shock_size ** 2)
                if NumericalGreeks.Gamma.value not in res:
                    res[NumericalGreeks.Gamma.value] = {}
                res[NumericalGreeks.Gamma.value][f"Gamma#{und}"] = gamma / stock_price / stock_price
                # replicate ctp gamma dollar method
                # so multiply 1% to scale it down to a change in dollar delta per percentage  change in spot
                res[NumericalGreeks.Gamma.value][f"GammaUSD#{und}"] = gamma * 0.01 * usd_conv
                res[NumericalGreeks.Gamma.value][f"SpotRef#{und}"] = stock_price

    # calculate vol greeks
    calc_vega = NumericalGreeks.Vega.value in greeks_list
    calc_volga = NumericalGreeks.Volga.value in greeks_list
    if calc_vega or calc_volga:
        expiry = tradable.expiration
        use_time_weighted_vega = kwargs.get("use_time_weighted_vega", False)
        one_side_market = one_side and not calc_volga
        shock_size = NumericalGreeksShockSize.VegaLevelShockSize
        for und in unds:
            ccy = currency_from_ticker(und) # todo find the better way to determine currency
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            if isinstance(tradable, AutoCallable):
                strike = dict(zip(tradable.und_list, tradable.start_spots))[und] * tradable.knock_in_put_strike
            else:
                strike = tradable.strike if hasattr(tradable, "strike") else None
            vol_ref = market.get_vol(und, expiry, strike) if strike is not None else None
            market_up, market_down = VegaShockedMarket.get_shocked_markets(market, und, use_time_weighted_vega, one_side=one_side_market, use_sticky_strike=sticky_strike_in_vega)
            price_kwargs = {"trade_suffix": f"_{und}_vol_up"}
            price_up = tradable.price(market_up, valuer=valuer, calc_types="price", **price_kwargs)
            if one_side:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price", trade_suffix="_base")
                price = tmp_res["origin_price"]
                vega = (price_up - price) / shock_size * NumericalGreeksScalingFactor.VolBumpScalingFactor
            else:
                if f"vol_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_vol_down"}
                    tmp_res[f"vol_down_price#{und}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"vol_down_price#{und}"]
                vega = (price_up - price_down) / (2 * shock_size) * NumericalGreeksScalingFactor.VolBumpScalingFactor
            if NumericalGreeks.Vega.value not in res:
                res[NumericalGreeks.Vega.value] = {}
            res[NumericalGreeks.Vega.value][f"Vega#{und}"] = vega
            res[NumericalGreeks.Vega.value][f"VegaUSD#{und}"] = vega * usd_conv
            res[NumericalGreeks.Vega.value][f"VolRef#{und}"] = vol_ref
            if calc_volga:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price", trade_suffix="_base")
                price = tmp_res["origin_price"]
                if f"vol_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_vol_down"}
                    tmp_res[f"vol_down_price#{und}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"spot_down_price#{und}"]
                volga = ((price_up + price_down) - 2 * price) / (shock_size ** 2) * NumericalGreeksScalingFactor.VolBumpScalingFactor
                if NumericalGreeks.Volga.value not in res:
                    res[NumericalGreeks.Volga.value] = {}
                res[NumericalGreeks.Volga.value][f"Volga#{und}"] = volga
                res[NumericalGreeks.Volga.value][f"Volga#{und}"] = volga * usd_conv
                res[NumericalGreeks.Volga.value][f"VolRef#{und}"] = vol_ref

    # calculate spot vol greeks
    calc_vanna = NumericalGreeks.Vanna.value in greeks_list
    if calc_vanna:
        spot_shock_size = NumericalGreeksShockSize.DeltaPercentageShockSize
        vol_shock_size = NumericalGreeksShockSize.VegaLevelShockSize
        for und in unds:
            ccy = currency_from_ticker(und)
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            market_up_up, market_up_down, market_down_up, market_down_down = VannaShockedMarket.get_shocked_markets(market, und)
            price_up_up = tradable.price(market_up_up, valuer=valuer, calc_types="price")
            price_up_down = tradable.price(market_up_down, valuer=valuer, calc_types="price")
            price_down_up = tradable.price(market_down_up, valuer=valuer, calc_types="price")
            price_down_down = tradable.price(market_down_down, valuer=valuer, calc_types="price")
            stock_price = market.get_spot(und)
            vanna = (price_up_up - price_up_down - price_down_up + price_down_down) / (4 * spot_shock_size * vol_shock_size) * NumericalGreeksScalingFactor.VolBumpScalingFactor
            if NumericalGreeks.Vanna.value not in res:
                res[NumericalGreeks.Vanna.value] = {}
            res[NumericalGreeks.Vanna.value][f"Vanna#{und}"] = vanna / stock_price
            res[NumericalGreeks.Vanna.value][f"VannaUSD#{und}"] = vanna * usd_conv

    # calculate div greeks
    calc_div = NumericalGreeks.Div.value in greeks_list
    if calc_div:
        div_shock_size_pct = NumericalGreeksShockSize.DivPerctShockSize
        for und in unds:
            ccy = currency_from_ticker(und)
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            #spot = market.get_spot(und)
            market_up, market_down = EqDivShockedMarket.get_shocked_markets(market, und, one_side = one_side)
            price_kwargs = {"trade_suffix": f"_{und}_div_up"}
            price_up = tradable.price(market_up, valuer = valuer, calc_types = "price", **price_kwargs)
            if one_side:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer = valuer, calc_types = "price")
                price = tmp_res["origin_price"]
                div = (price_up - price) / div_shock_size_pct
            else:
                if f"div_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_div_down"}
                    tmp_res[f"div_down_price#{und}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"div_down_price#{und}"]
                div = (price_up - price_down) / (2 * div_shock_size_pct)
            if NumericalGreeks.Div.value not in res:
                res[NumericalGreeks.Div.value] = {}
            res[NumericalGreeks.Div.value][f"Div#{und}"] = div
            res[NumericalGreeks.Div.value][f"DivUSD#{und}"] = div * usd_conv

    # calculate repo greeks
    calc_repo = NumericalGreeks.Repo.value in greeks_list
    if calc_repo:
        repo_shock_size_bps = NumericalGreeksShockSize.RepoLevelShockSize

        for und in unds:
            ccy = currency_from_ticker(und)
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            market_up, market_down = EqRepoShockedMarket.get_shocked_markets(market, und, one_side = one_side)
            price_kwargs = {"trade_suffix": f"_{und}_repo_up"}
            price_up = tradable.price(market_up, valuer = valuer, calc_types = "price", **price_kwargs)
            if one_side:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer = valuer, calc_types = "price")
                price = tmp_res["origin_price"]
                repo = (price_up - price) / repo_shock_size_bps
            else:
                if f"repo_down_price#{und}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{und}_repo_down"}
                    tmp_res[f"repo_down_price#{und}"] = tradable.price(market_down, valuer = valuer, calc_types = "price", **price_kwargs)
                price_down = tmp_res[f"repo_down_price#{und}"]
                repo = (price_up - price_down) / (2 * repo_shock_size_bps)
            if NumericalGreeks.Repo.value not in res:
                res[NumericalGreeks.Repo.value] = {}
            res[NumericalGreeks.Repo.value][f"Repo#{und}"] = repo
            res[NumericalGreeks.Repo.value][f"RepoUSD#{und}"] = repo * usd_conv

    # calculate rate greeks
    calc_rho = NumericalGreeks.Rho.value in greeks_list
    if calc_rho:
        rate_shock_size_bps = NumericalGreeksShockSize.RhoLevelShockSize
        rate_unds = tradable.get_rate_underlyings()
        market_up = market
        market_down = market

        for ccy, curve_name in rate_unds:
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            market_up, market_down = RateRhoShockedMarket.get_shocked_markets(market, ccy, curve_name, one_side=one_side)
            price_kwargs = {"trade_suffix": f"_{ccy}_{curve_name}_rate_up"}
            price_up = tradable.price(market_up, valuer=valuer, calc_types="price", **price_kwargs)
            if one_side:
                if "origin_price" not in tmp_res:
                    tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price")
                price = tmp_res["origin_price"]
                rho = (price_up - price) / rate_shock_size_bps
            else:
                if f"rate_down_price#{curve_name}" not in tmp_res:
                    price_kwargs = {"trade_suffix": f"_{ccy}_{curve_name}_rate_down"}
                    tmp_res[f"rate_down_price#{curve_name}"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
                price_down = tmp_res[f"rate_down_price#{curve_name}"]
                rho = (price_up - price_down) / (2 * rate_shock_size_bps)
            if NumericalGreeks.Rho.value not in res:
                res[NumericalGreeks.Rho.value] = {}
            res[NumericalGreeks.Rho.value][f"Rho#{curve_name}"] = rho
            res[NumericalGreeks.Rho.value][f"RhoUSD#{curve_name}"] = rho * usd_conv

        for und in unds:
            ccy = currency_from_ticker(und)
            usd_conv = 1 if ccy == "USD" else market.get_spot(f"{ccy}USD")
            market_up, dummy = EqRhoShockedMarket.get_shocked_markets(market_up, und, True)
            if(not one_side):
                dummy, market_down = EqRhoShockedMarket.get_shocked_markets(market_down, und, False)

        price_kwargs = {"trade_suffix": f"interest_rate_up"}
        price_up = tradable.price(market_up, valuer=valuer, calc_types="price", **price_kwargs)
        if one_side:
            if "origin_price" not in tmp_res:
                tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price")
            price = tmp_res["origin_price"]
            rho = (price_up - price) / rate_shock_size_bps
        else:
            if f"interest_rate_down_price" not in tmp_res:
                price_kwargs = {"trade_suffix": f"_interest_rate_down"}
                tmp_res[f"interest_rate_down_price"] = tradable.price(market_down, valuer=valuer, calc_types="price", **price_kwargs)
            price_down = tmp_res[f"interest_rate_down_price"]
            rho = (price_up - price_down) / (2 * rate_shock_size_bps)
        if NumericalGreeks.Rho.value not in res:
            res[NumericalGreeks.Rho.value] = {}
        res[NumericalGreeks.Rho.value][f"eqRho"] = rho  # eqRho is sensitivity for bump SOFR/Yield/discounting curve all at once
        res[NumericalGreeks.Rho.value][f"eqRhoUSD"] = rho * usd_conv

    # calculate theta greeks
    calc_theta = NumericalGreeks.Theta.value in greeks_list
    if calc_theta:
        theta_shock_size_bds = NumericalGreeksShockSize.ThetaShockSize
        market_plus = ThetaShockedMarket.get_shocked_markets(market)
        price_plus = tradable.price(market_plus, valuer=valuer, calc_types="price")
        if "origin_price" not in tmp_res:
            tmp_res["origin_price"] = tradable.price(market, valuer=valuer, calc_types="price")
        price = tmp_res["origin_price"]
        theta = (price_plus - price) / theta_shock_size_bds
        if NumericalGreeks.Theta.value not in res:
            res[NumericalGreeks.Theta.value] = {}
        res[NumericalGreeks.Theta.value][f"Theta"] = theta
    return res


def flatten_numerical_greeks(greeks, calc_types):
    res = []
    for calc_type, calc_res in zip(calc_types, greeks):
        if "numerical#" in calc_type:
            for k, v in calc_res.items():
                res.append([k, v])
        else:
            res.append([calc_type, calc_res])
    res = pd.DataFrame(res, columns=["Name", "Value"])
    return res
