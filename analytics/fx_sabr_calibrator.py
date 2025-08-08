from ..analytics import fx_sabr
from ..analytics.arbitrage import load_data
from ..tools.timer import Timer
from datetime import datetime
from scipy.optimize import least_squares
import numpy as np
import os
import pandas as pd
import pickle


def fit_smile(vol_smile_data):
    def func(params):
        [sigma0, vov, rho] = params
        data = vol_smile_data[["strike", "fwd", "tte", "iv"]]
        data["fit"] = data.apply(lambda row: fx_sabr.get_vol_from_raw_params(row["strike"], row["fwd"], row["tte"], sigma0, vov, rho), axis=1)
        error = data["fit"] - data["iv"]
        return error

    sigma0_guess = vol_smile_data.set_index("delta").loc[0.5, "iv"]
    vov_guess = 0.5
    rho_guess = -0.1
    x0 = [sigma0_guess, vov_guess, rho_guess]
    bounds = ([0.00001, 0.00001, -0.9999], [float('inf'), float('inf'), 0.9999])
    res = least_squares(func, x0, bounds=bounds)
    sigma0, vov, rho = res.x
    vol_smile_data["fit"] = vol_smile_data.apply(lambda row: fx_sabr.get_vol_from_raw_params(row["strike"], row["fwd"], row["tte"], sigma0, vov, rho), axis=1)
    mse = np.sqrt(((vol_smile_data["fit"] - vol_smile_data["iv"])**2).mean())
    print(vol_smile_data.tenor.values[0], sigma0, vov, rho, mse)
    # print(vol_smile_data)
    return sigma0, vov, rho, mse


def calibrate_sabr_param(vol_data):
    params = []
    for tenor, grp in vol_data.groupby("tenor"):
        sigma0, vov, rho, mse = fit_smile(grp)
        params.append([tenor, grp.tte.values[0], grp.expiry.values[0], grp.fwd.values[0], sigma0, vov, rho, mse])
    res = pd.DataFrame(params, columns=["tenor", "tte", "expiry", "fwd", "sigma0", "vov", "rho", "mse"])
    return res


if __name__ == '__main__':
    timer = Timer("sabr calibration")
    timer.start()

    pair_info = {
        "AUDUSD": {"fwd_ccy": "AUD", "st": datetime(2007, 1, 1)},
        "EURUSD": {"fwd_ccy": "EUR", "st": datetime(2003, 1, 1)},
        "USDJPY": {"fwd_ccy": "USD", "st": datetime(2003, 1, 1)},
    }
    # 1) query data
    tenors = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y"]
    delta_strikes = ["10DP", "25DP", "ATM", "25DC", "10DC"]
    pair = "AUDUSD"
    fwd_ccy = pair_info[pair]["fwd_ccy"]
    st = pair_info[pair]["st"]
    print(pair, fwd_ccy, st)
    et = datetime(2025, 5, 27)

    reload_data = False
    data_file = f"fx_data_{pair}_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.pkl"
    if reload_data:
        force_reload = True
        data = load_data(pair, st, et, force_reload, timer)
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    timer.reset("completed process data")

    log_file = f"fx_sabr_calibrator_{pair}.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    # 3) run check by date
    surface_folder = f"{pair}"
    if os.path.exists(surface_folder):
        import shutil
        shutil.rmtree(surface_folder)
    if not os.path.exists(surface_folder):
        os.mkdir(surface_folder)
    res = []
    for dt, grp in data.groupby("date"):
        print(dt)
        vol_data = grp[["tenor", "tte", "expiry", "delta", "strike", "fwd", "iv"]]
        # check all tenor:
        tenors = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y"]
        missing_tenors = [tenor for tenor in tenors if tenor not in vol_data.tenor.tolist()]
        if len(missing_tenors) > 0:
            with open(log_file, "a+") as f:
                msg = f"missing tenors: {','.join(missing_tenors)}"
                f.write(pair + "," + dt.strftime("%Y-%m-%d") + "," + msg + "," + timer.get_now().isoformat() + "\n")
            continue
        if vol_data.isnull().values.any():
            with open(log_file, "a+") as f:
                msg = f"found nan"
                f.write(pair + "," + dt.strftime("%Y-%m-%d") + "," + msg + "," + timer.get_now().isoformat() + "\n")
            continue
        sabr_params = calibrate_sabr_param(vol_data)
        sabr_params["date"] = dt
        sabr_params["spot"] = grp["spot"].values[0]
        res.append(sabr_params)
        surface_file = os.path.join(surface_folder, f"{pair}_{dt.strftime('%Y%m%d')}.csv")
        sabr_params.sort_values("expiry").to_csv(surface_file, index=False)
        quotes_file = os.path.join(surface_folder, f"{pair}_{dt.strftime('%Y%m%d')}_quotes.csv")
        calib_res = grp.reset_index().set_index(["date", "tenor"]).join(sabr_params.set_index(["date", "tenor"])[["sigma0", "vov", "rho", "mse"]])
        calib_res["fit"] = calib_res.apply(lambda row: fx_sabr.get_vol_from_raw_params(row["strike"], row["fwd"], row["tte"], row["sigma0"], row["vov"], row["rho"]), axis=1)
        calib_res.to_csv(quotes_file)

    timer.end()
