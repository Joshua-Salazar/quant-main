import json
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import requests
import logging
from typing import List


# bpipe urls
BPIPE_SERVER = "http://ntprctp01:6082/"
BPIPE_URL = "http://ntprctp01:6082/getHistoricalData?user={};symbols=/ticker/{};fields={};startdate={};enddate={}"
BPIPE_INTRA_URL = "http://ntprctp01:6082/getIntradayBar?user={};symbol=/ticker/{};eventtype={};starttime={};endtime={};interval={}"
BPIPE_REF_URL = "http://ntprctp01:6082/getRefData?user={};symbols=/ticker/{};fields={};overrides=TIME_ZONE_OVERRIDE:22"
BPIPE_LIVE_URL = "http://ntprctp01:6082/getRefData?user={};symbols=/ticker/{};fields={};force=TRUE"


def __chunk(alist, n: int):
    for i in range(0, len(alist), n):
        yield alist[i:i+n]

def get_bbg_timeseries(symbol, user, field='PX_LAST', lookback=731, startdate=None, enddate=None,
                       convert_to_numeric=True):

    # process inputs
    if startdate is None and enddate is None:
        enddate = (date.today() + relativedelta(days=-1))
        startdate = (enddate + relativedelta(days=-lookback))
    elif startdate is None:
        startdate = (enddate + relativedelta(days=-lookback))
    elif enddate is None:
        enddate = (date.today() + relativedelta(days=-1))
    startdate = startdate.strftime("%Y%m%d")
    enddate = enddate.strftime("%Y%m%d")
    multi_field = isinstance(field, list)
    if multi_field:
        field = "|".join(field)

    # form URL and load
    url = BPIPE_URL.format(user, symbol.replace(" ", "%20"), field, startdate, enddate)
    response = json.loads(requests.get(url).text)

    # parse output
    for (ticker, data) in response.items():
        rows = []
        for (point, value) in data.items():
            p = point.split("|")
            rows.append({"value": value, "date": p[0], "field": p[1]})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        if convert_to_numeric:
            df["value"] = pd.to_numeric(df["value"])
        df = df.pivot_table(values="value", index="date", columns="field", aggfunc=np.sum)
        df = df.sort_index()

    # return
    return df if multi_field else df[df.columns[0]]


def get_bbg_ref_data(symbol, user, field):

    # process inputs
    if isinstance(field, list):
        field = "|".join(field)

    # form URL and load
    url = BPIPE_REF_URL.format(user, symbol.replace(" ", "%20"), field)
    response = json.loads(requests.get(url).text)

    # parse output
    for (ticker, data) in response.items():
        fields = []
        results = []
        for (f, value) in data.items():
            fields.append(f)
            results.append(value)

    # return
    return pd.Series(index=fields, data=results)


def get_bbg_live_data(symbol, user, field):

    # form URL and load
    url = BPIPE_LIVE_URL.format(user, symbol.replace(" ", "%20"), field)
    response = json.loads(requests.get(url).text)

    # parse output
    for (ticker, data) in response.items():
        results = []
        for (point, value) in data.items():
            results.append(value)

    # return
    return pd.Series(results)


def get_bbg_live_data_v2(symbols: List, user, fields: List, chunk_size=250):
    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(fields, str):
        fields = [fields]

    symbol_src = "ticker"
    BPIPE_URL = BPIPE_SERVER + "/getRefData?user={};symbols={};fields={};force=TRUE"
    prefix = f"/{symbol_src}/"

    all_dataframes = []
    for symchunk in __chunk(symbols, chunk_size):
        qsym = ["/{}/{}".format(symbol_src, x.replace(" ", "%20")) for x in symchunk]
        url = BPIPE_URL.format(user, "|".join(qsym), "|".join(fields))

        logging.info("making bbg rest request for {} names and {} fields".format(
            len(qsym), len(fields)))
        response = json.loads(requests.get(url).text)

        # trim the '/ticker/' that prefixes each symbol
        results_no_prefix = dict()
        for key, value in response.items():
            if key.startswith(prefix):
                key = key[len(prefix):]
            results_no_prefix[key] = value

        all_dataframes.append(pd.DataFrame.from_dict(results_no_prefix).T)

    df = pd.concat(all_dataframes)
    return df


def get_bbg_intra_timeseries(symbol, user, eventtype="TRADE", lookback=30, interval=15, periods=None):

    # process inputs
    endd = (date.today() + relativedelta(days=1))
    startd = (date.today() + relativedelta(days=-lookback))
    periods = int(lookback / 5) if periods is None else periods

    # range of dates to read
    dates = list(pd.date_range(startd, endd, periods))
    df = pd.DataFrame([])

    # loop over days
    for start, end in zip(dates[:-1], dates[1:]):

        # load actual data
        startdate = start.strftime("%Y-%m-%d 00:00:00")
        enddate = end.strftime("%Y-%m-%d 00:00:00")
        df_this_Date = __get_bbg_intra_dates(symbol, user, startdate, enddate, eventtype=eventtype, interval=interval)
        df = df.append(df_this_Date)

    # return
    return df


def __get_bbg_intra_dates(symbol, user, startdate, enddate, eventtype="TRADE", interval=15):

    # load
    url = BPIPE_INTRA_URL.format(user, symbol, eventtype, startdate, enddate, interval).replace(" ", "%20")
    resp_text = requests.get(url).text
    response = json.loads(resp_text)

    # check response
    if response is None:
        raise ValueError(f"text: {resp_text}\nquery:{url}")

    # read data
    rows = []
    for (ticker, data) in response.items():
        timestamp, field = ticker.split("|")
        rows.append([timestamp, field, data])

    # convert to dataframe
    df = pd.DataFrame(rows, columns=["timestamp", "field", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"])
    df = df.set_index(["timestamp", "field"]).unstack().sort_index()
    df.columns = df.columns.get_level_values(1)
    df.index.name = None
    df.columns.name = None
    df.rename({"numEvents": f"{eventtype}_events"}, axis=1, inplace=True)

    # return
    return df


