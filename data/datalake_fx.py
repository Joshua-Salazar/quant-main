import pandas as pd
from ..data import datalake


def get_bbg_fx_spot(pair, start_date, end_date, fixings_source):
    inverse = False
    postfix = "Curncy" if len(fixings_source) == 0 else f"{fixings_source} Curncy"
    ticker = f"{pair} {postfix}"
    spots = datalake.get_bbg_history(ticker, "PX_LAST", start_date, end_date, throw_if_missing=False)
    if spots.empty:
        # fallback method to check if we can load inversed pair
        inversed_pair = pair[3:] + pair[:3]
        inversed_ticker = f"{inversed_pair} {postfix}"
        inversed_spots = datalake.get_bbg_history(inversed_ticker, "PX_LAST", start_date, end_date,
                                                  throw_if_missing=False)
        if not inversed_spots.empty:
            inverse = True
            spots = inversed_spots

    if not spots.empty:
        spots["date"] = pd.to_datetime(spots["date"])
        spots = spots[["date", "PX_LAST"]].set_index("date")
        if inverse:
            spots = 1. / spots
    else:
        # fallback method to use USD cross
        if "USD" not in pair:
            base_pair = pair[:3] + "USD"
            base_spots = get_bbg_fx_spot(base_pair, start_date, end_date, fixings_source)
            term_pair = pair[3:] + "USD"
            term_spots = get_bbg_fx_spot(term_pair, start_date, end_date, fixings_source)
            if not base_spots.empty and not term_spots.empty:
                spots = base_spots[base_pair] / term_spots[term_pair]
                spots = spots.to_frame()

    if not spots.empty:
        spots.columns = [pair]
    return spots


if __name__ == "__main__":
    from datetime import datetime
    st = datetime(2023, 10, 1)
    et = datetime(2023, 10, 20)
    pairs = ["USDCAD", "EURUSD", "EURGBP", "GBPUSD", "USDMXN", "USDNOK", "CADNOK", "USDSEK", "EURSEK", "USDCHF", 
             "EURCHF", "USDJPY", "AUDJPY", "AUDUSD", "USDNZD"]
    # fixing WMCO London 11PM Fixings
    fixings_source = "WMCO"
    for pair in pairs:
        s = get_bbg_fx_spot(pair, st, et, fixings_source)
        print(pair, s.iloc[-1].values[0])