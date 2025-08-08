from datetime import datetime
from ..constants.ccy import Ccy
from ..constants.market_item_type import MarketItemType
from ..infrastructure.fx_pair import FXPair
from typing import Union


def create_vol_surface_key(underlying: str) -> str:
    return f"{MarketItemType.VOLATILITY.value}.{underlying}"


def create_fx_vol_surface_key(pair: Union[str, FXPair]) -> str:
    return f"{MarketItemType.FXVOLATILITY.value}.{pair if isinstance(pair, str) else pair.to_string()}"


def create_cr_vol_surface_key(underlying: str) -> str:
    return f"{MarketItemType.CRVOLATILITY.value}.{underlying}"


def create_cr_vol_surface_key_with_specialisation(underlying: str, specialisation: str = None) -> str:
    return f"{underlying}.{specialisation}"


def create_swaption_vol_cube_key(currency: str) -> str:
    return f"{MarketItemType.SWAPTIONVOLATILITY.value}.{currency}"

def create_interest_rate_data_container_key(underlying: str) -> str:
    return f"{MarketItemType.INTERESTRATE.value}.{underlying}"

def create_discount_curve_key(underlying: str) -> str:
    return f"{MarketItemType.DISCOUNTCURVE.value}.{underlying}"


def create_borrow_curve_key(underlying: str) -> str:
    return f"{MarketItemType.BORROWCURVE.value}.{underlying}"


def create_spot_key(underlying: str) -> str:
    return f"{MarketItemType.SPOT.value}.{underlying}"


def create_dividend_key(underlying: str) -> str:
    return f"{MarketItemType.DIVIDEND.value}.{underlying}"


def create_corpaction_key(underlying: str) -> str:
    return f"{MarketItemType.CORPACTION.value}.{underlying}"


def create_fx_spot_key(pair: Union[str, FXPair]) -> str:
    return f"{MarketItemType.FXSPOT.value}.{pair if isinstance(pair, str) else pair.to_string()}"


def create_fx_fwd_key(pair: Union[str, FXPair]) -> str:
    return f"{MarketItemType.FXFORWARD.value}.{pair if isinstance(pair, str) else pair.to_string()}"


def create_fx_fwd_point_key(pair: FXPair) -> str:
    return f"{MarketItemType.FXFORWARDPOINT.value}.{pair.to_string()}"


def create_xccy_basis_key(ccy: Ccy, swap_term: str) -> str:
    return f"{MarketItemType.XCCYBASIS.value}.{ccy.value}.{swap_term}"


def create_rate_fixing_key(ccy: Ccy, tenor: str) -> str:
    return f"{MarketItemType.FIXING.value}.{ccy.value}.{tenor}"


def create_fx_fixing_key(pair: FXPair) -> str:
    return f"{MarketItemType.FIXING.value}.{pair.to_string()}"


def create_fixing_table_key() -> str:
    return f"{MarketItemType.FIXINGTABLE.value}"


def create_future_key(underlying: str, expiry_dt: datetime) -> str:
    return f"{MarketItemType.FUTURE.value}.{underlying}.{expiry_dt.isoformat()}"


def is_vol_surface_key(key: str) -> str:
    return MarketItemType.VOLATILITY.value in key


def is_fx_vol_surface_key(key: str) -> str:
    return "FXVolatility" in key


def create_option_data_container_key(underlying: str) -> str:
    return f"{MarketItemType.OPTIONDATACONTAINER.value}.{underlying}"


def create_future_data_container_key(underlying: str) -> str:
    return f"{MarketItemType.FUTUREDATACONTAINER.value}.{underlying}"


def create_forward_rates_key(currency: str, curve: str = None) -> str:
    key = f"{MarketItemType.FORWARDRATECURVE.value}.{currency}"
    return key if curve is None else f"{key}.{curve}"


def create_future_intraday_data_container_key(underlying: str) -> str:
    return f"{MarketItemType.FUTUREINTRADAYDATACONTAINER.value}.{underlying}"


def create_spot_rates_key(currency: str, curve: str) -> str:
    return f"{MarketItemType.SPOTRATECURVE.value}.{currency}.{curve}"


def create_portfolio_key(portfolio_name: str) -> str:
    return f"{MarketItemType.PORTFOLIO.value}.{portfolio_name}"


def create_bond_yield_key(ticker: str):
    return f"{MarketItemType.BONDYIELD.value}.{ticker}"


def create_holiday_center_key():
    return f"{MarketItemType.HOLIDAYCENTER.value}"


def get_key_type(key: str):
    return MarketItemType(key.split(".")[0])


def create_correlation_matrix(tickers) -> str:
    return f"{MarketItemType.CORRELATIONMATRIX.value}.{'.'.join(tickers)}"
