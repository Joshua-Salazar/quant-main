import numpy as np


def calculate_realised_vol_from_returns(ln_ret, lag):
    sq = ln_ret * ln_ret
    sq.dropna(inplace=True)
    sq_mean = sq.mean()
    periods_per_year = 252
    var = np.sqrt(sq_mean * periods_per_year / lag) * 100
    return var


def calculate_var_vol_breakevens(var_strike, vol_strike, var_vol_ratio=1):
    # for unit vega notional, 1/(2*var_strike*var_vol_ratio) (x^2 - var_strike^2) - (x - vol_strike) = 0
    tmp = np.sqrt(var_strike**2 + (var_strike*var_vol_ratio)**2 - 2 * var_strike * var_vol_ratio * vol_strike)
    be_low = var_strike * var_vol_ratio - tmp
    be_high = var_strike * var_vol_ratio + tmp
    return be_low, be_high


def calculate_var_vol_max_loss(var_strike, vol_strike, var_vol_ratio=1):
    # for unit vega notional, 1/(2*var_strike*var_vol_ratio) (x^2 - var_strike^2) - (x - vol_strike)
    max_loss = vol_strike - 0.5 * var_strike / var_vol_ratio - 0.5 * var_strike * var_vol_ratio
    return max_loss


def calculate_var_vol_hit_rate(spots, swap_tenor, var_strike, vol_strike, lookback_yrs=10, var_vol_ratio=1):
    rtn = np.log(spots).diff()**2
    tenor_days_map = {"3m": 65, "4m": 86, "5m": 108, "6m": 130, "1y": 260}
    if swap_tenor.lower() not in tenor_days_map:
        raise Exception(f"Not support swap tenor: {swap_tenor}")
    window_days = tenor_days_map[swap_tenor.lower()]
    realised_vol = rtn.rolling(window_days).apply(lambda x: np.sqrt(x.mean() * 252) * 100)
    unit_vega_pnl = 1 / (2 * var_strike * var_vol_ratio) * (realised_vol ** 2 - var_strike ** 2) - (realised_vol - vol_strike)
    sub_unit_vega_pnl = unit_vega_pnl.iloc[-lookback_yrs * 252:]
    hit_rate = sum(sub_unit_vega_pnl > 0) / sub_unit_vega_pnl.count()
    return hit_rate
