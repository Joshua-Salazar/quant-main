# imports
import logging
from datetime import datetime as dt
from datetime import time
from typing import Tuple

import pandas as pd

import arctic
from ..data.datastore import DataStore


def replace_time(tstamps, new_time):
    return [dt.combine(t.date(), new_time) for t in tstamps]


def make_timestamps_eod(tstamps, eod_time=time(21, 30)):
    """'
    Replace time on a list of datetime objects with a time representing EOD
    """
    return replace_time(tstamps, eod_time)


def make_item_eod(data):
    data.index = [t.to_pydatetime() for t in data.index]
    eod_dates = make_timestamps_eod(pd.bdate_range(min(data.index), max(data.index), freq="B"))
    data = data.reindex(index=eod_dates, method="ffill")
    return data


def compare_tstamps_with_arctic_index(tstamps: list, ds: DataStore, df_name: str) -> Tuple[list, pd.DataFrame]:
    """
    Compares a list of tstamps with a DataFrame index stored in arctic and
    returns the set difference between the two along with the DataFrame.

    Parameters
    ----------
    tstamps: list
        The list of tstamps to compare to an arctic dataframe.
    ds: DataStore
        The datastore containing the arctic dataframe.
    df_name: str
        The name of the dataframe stored in the datastore.

    Returns
    -------
    list
        The set difference between tstamps and the index of the dataframe.
    """

    # Check available data history and append new or missing obs.
    try:
        arctic_df = ds.read_item(df_name)
        arctic_tstamps = arctic_df.index.to_pydatetime().tolist() if len(arctic_df) > 0 else []
    except arctic.exceptions.NoDataFoundException:
        logging.warning(f"DataFrame {df_name} not found in arctic.")
        return tstamps, pd.DataFrame()

    # Compute the missing tstamps in arctic with a set difference.
    arctic_missing_tstamps: list = list(set(tstamps) - set(arctic_tstamps))

    return arctic_missing_tstamps, arctic_df
