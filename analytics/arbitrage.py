from ..analytics import options
from ..dates.holidays import get_holidays
from ..dates.utils import get_fx_expiry_date
from ..data import datalake
from ..tools.timer import Timer
import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

CCS_CCY = [
    "AUD",
    "CAD",
    "EUR",
    "JPY",
    "NOK",
    "NZD",
    "SGD",
    "SEK",
    "CHF",
    "GBP",
    "USD",
]

USD_PAIR = [
    "AUDUSD",
    "EURUSD",
    "GBPUSD",
    # "NZDUSD",
]

PREMIMUM_ADJUSTED_DELTA_PAIR = [
 "USDJPY",
]


def check_arbitrage(call_data, throw_if_failure):
    res = True
    msg = ""
    for tenor, grp in call_data.groupby("tenor"):
        # print(tenor)
        # call spread / and butterfly
        data = grp.sort_values("strike")
        data[["strike_prev", "call_prev"]] = data[["strike", "call"]].shift()
        data[["strike_post", "call_post"]] = data[["strike", "call"]].shift(-1)
        data["q"] = (data["call_prev"] - data["call"]) / (data["strike"] - data["strike_prev"])
        # check call spread
        succeed = ((data["q"].iloc[1:] >= 0) & (data["q"].iloc[1:] <= 1)).all()
        if not succeed:
            print(f"Failed to pass call spread check on {tenor}")
            # print(data)
            res = False
            msg += f"CS{tenor}|"
        # check butterfly
        data["alpha"] = (data["strike_post"] - data["strike_prev"]) / (data["strike_post"] - data["strike"])
        data["beta"] = (data["strike"] - data["strike_prev"]) / (data["strike_post"] - data["strike"])
        data["bs"] = data["call_prev"] - data["alpha"] * data["call"] + data["beta"] * data["call_post"]
        succeed = (data["bs"].iloc[1:-1] >= 0).all()
        if not succeed:
            print(f"Failed to pass butterfly check on {tenor}")
            # print(data)
            res = False
            msg += f"BF{tenor}|"

    # calendar spread
    cs = call_data.groupby(["delta"]).apply(lambda x: x[["tte", "call"]].sort_values("tte").set_index("tte").diff().iloc[1:])
    succeed = (cs.values >= 0).all()
    if not succeed:
        print(f"Failed to pass calendar spread check")
        delta_tenors = call_data.set_index(["delta", "tte"]).loc[cs[cs.values < 0].index]
        # print(call_data)
        res = False
        for (delta, tte), row in delta_tenors.iterrows():
            msg += f"CalS{tenor}{delta}|"
    if throw_if_failure and not res:
        raise
    return res, msg


def load_data(pair, st, et, force_reload, timer):
    # 1) query data
    tenors = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y"]
    delta_strikes = ["10DP", "25DP", "ATM", "25DC", "10DC"]
    fwd_ccy = pair[:3] if pair[3:] == "USD" else pair[3:]
    data_file = f"fx_datalake_data_{pair}_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.pkl"
    if force_reload or not os.path.exists(data_file):
        vol_tickers = []
        for tenor in tenors:
            for delta_strike in delta_strikes:
                ticker = f"{pair} {tenor} {delta_strike} VOL BVOL Curncy"
                vol_tickers.append(ticker)
        vol_data = datalake.get_bbg_history(vol_tickers, 'PX_LAST', st, et)
        vol_data["date"] = pd.to_datetime(vol_data["date"])
        timer.reset("completed vol data query")
        spot_data = datalake.get_bbg_history(f"{pair} BGN Curncy", 'PX_LAST', st, et)
        spot_data["date"] = pd.to_datetime(spot_data["date"])
        timer.reset("completed spot data query")
        fwd_pt_tickers = []
        for tenor in tenors:
            ticker = f"{fwd_ccy}{tenor.replace('1D', 'ON').replace('1Y', '12M')} BGN Curncy"
            fwd_pt_tickers.append(ticker)
        fwd_pt_data = datalake.get_bbg_history(fwd_pt_tickers, 'PX_LAST', st, et)
        timer.reset("completed fwd pnt data query")
        fwd_pt_data["date"] = pd.to_datetime(fwd_pt_data["date"])
        with open(data_file, 'wb') as f:
            pickle.dump([vol_data, spot_data, fwd_pt_data], f)
    else:
        with open(data_file, 'rb') as f:
            [vol_data, spot_data, fwd_pt_data] = pickle.load(f)
    timer.reset("completed load data query")

    # 2) process data
    vol_st = vol_data.date.min()
    vol_et = vol_data.date.max()
    hols = get_holidays([pair[:3], pair[3:]], vol_st, vol_et + relativedelta(years=2))
    hol_dates = list(map(lambda x: x.date(), hols))
    def parse_dalta(ticker):
        delta_cp = ticker.split()[2]
        if delta_cp == "ATM":
            return 0.5, delta_cp
        elif delta_cp[-1] == "C":
            return int(delta_cp[:-2]) / 100, delta_cp[-1]
        else:
            assert delta_cp[-1] == "P"
            return -int(delta_cp[:-2]) / 100, delta_cp[-1]
    vol_data["tenor"] = vol_data["ticker"].str.split(expand=True)[1]
    vol_data[["delta", "cp"]] = vol_data.apply(lambda row: parse_dalta(row["ticker"]), result_type="expand", axis=1)
    vol_data["expiry"] = vol_data.apply(lambda row: get_fx_expiry_date(row["date"], row["tenor"], pair, hol_dates), axis=1)
    timer.reset("completed expiry")
    vol_data["tte"] = (vol_data["expiry"] - vol_data["date"]).dt.days / 360
    vol_data = vol_data.rename(columns={"PX_LAST": "iv"})
    vol_data["iv"] /= 100
    spot_data = spot_data.rename(columns={"PX_LAST": "spot"})[["date", "spot"]]
    fwd_pt_data["tenor"] = fwd_pt_data["ticker"].str.split(expand=True)[0].str[3:]
    fwd_pt_data["tenor"] = fwd_pt_data["tenor"].str.replace('ON', '1D').replace('12M', '1Y')
    fwd_pt_data = fwd_pt_data.rename(columns={"PX_LAST": "fwd_pt"})
    data = vol_data.set_index(["date", "tenor"])[["expiry", "tte", "cp", "delta", "iv"]].join(fwd_pt_data.set_index(["date", "tenor"])[["fwd_pt"]])
    data = data.reset_index().set_index("date").join(spot_data.set_index("date"))
    fwd_pt_scalar = 1e-2 if fwd_ccy in ["JPY"] else 1e-4
    data["fwd"] = data["spot"] + data["fwd_pt"] * fwd_pt_scalar
    # calculate strike from delta

    prem_adj_factor = -1 if pair in PREMIMUM_ADJUSTED_DELTA_PAIR else 1
    prem_adj = pair in PREMIMUM_ADJUSTED_DELTA_PAIR
    data["strike"] = data.apply(lambda row: row.fwd * np.exp(prem_adj_factor * 0.5 * row.iv ** 2 * row.tte) if row.cp == "ATM" else options.get_strike_from_delta(row.delta, row.fwd, row.iv, row.tte, is_call=row.cp == "C", prem_adj=prem_adj), axis=1)
    # calculate call option price
    data["call"] = data.apply(lambda row: options.Black76(strike=row.strike, expiration=None, is_call=True, dt=None, fwd=row.fwd, vol=row.iv, disc=1, TTM=row.tte)["price"], axis=1)
    # print(data)
    timer.reset("completed process data")
    return data


if __name__ == '__main__':
    timer = Timer("arbitrage check")
    timer.start()
    st = datetime(1990, 1, 1)
    # st = datetime(2025, 4, 2)
    et = datetime(2025, 4, 10)
    log_file = "arbitrage.log"
    # os.remove(log_file)
    force_reload = False
    for pair in PREMIMUM_ADJUSTED_DELTA_PAIR:
        timer.print(f"processing pair {pair}")
        data = load_data(pair, st, et, force_reload, timer)
        # run check by date
        throw_if_failure = False
        for dt, grp in data.groupby("date"):
            # print(dt)
            vol_data = grp[["tenor", "tte", "delta", "strike", "call"]]
            succeed, msg = check_arbitrage(vol_data, throw_if_failure=throw_if_failure)
            if not succeed:
                with open(log_file, "a+") as f:
                    f.write(pair + "," + dt.strftime("%Y-%m-%d") + "," + msg + "," + timer.get_now().isoformat() + "\n")
        timer.reset(f"completed pair {pair}")
    timer.end()
