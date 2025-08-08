import logging

import numpy as np
import pandas as pd
from ..util.tstamps import make_timestamps_eod


# TODO: review implementation, and add tolerance params
def approx_lt(x, y):
    return (x <= y) & (~np.isclose(x, y))


# TODO: review implementation, and add tolerance params
def approx_gt(x, y):
    return (x >= y) & (~np.isclose(x, y))


def approx_eq(df1, df2, tolerance=1e-6):
    """Compare two dataframes for approximate equality."""
    if id(df1) == id(df2):
        return True
    # check nan's match
    if not ((df1.isna() == df2.isna()).all().all()):
        return False
    # check values differences are within tolerance
    return not ((df1 - df2).abs() > tolerance).any().any()


def ffill_all_missing(df, limit=None):
    """
    Forward fill rows if the entire row is missing

    :param  df     : pd.DataFrame
    :param  limit  : int (fefaults to None)

    """

    # loop over rows
    c = 0
    df_l = []
    for i in range(df.shape[0]):
        if df.iloc[i].isnull().all() and ((limit is None) or (c < limit)) and len(df_l) > 0:
            copy_row = df_l[-1].copy()
            copy_row.name = df.index[i]
            df_l.append(copy_row)
            c += 1
        else:
            df_l.append(df.iloc[i, :].copy())
            c = 0
    df_f = pd.concat(df_l, axis=1).transpose()

    return df_f


def ewm_smoothing_keeping_nan_rows(
    df: pd.DataFrame, ewm_hl: int, num_tail_rows: int = 5, missing_obs: int = None
) -> pd.DataFrame:
    """
    Applies ewm smoothing to a dataframe. If the underlying data rows are all
    NaN towards the tail, then these will be set to NaN and not smoothed over.

    missing_obs sets a value to NaN if the past N observations are also NaN
    prior to the smoothing.

    Parameters
    ----------
    df: pd.DataFrame
        A TxN DataFrame that will be smoothed.
    ewm_hl: int
        Halflife of the ewm smoothing.
    num_tail_rows: int
        Number of rows at the tail of the df to preserve as NaN after smoothing if the entire row is NaN.
    missing_obs: int
        If the past missing_obs_thresh values are all NaN in a column, set the smoothed value to NaN.

    Returns
    -------
    df: pd.DataFrame
    """

    if num_tail_rows <= 0:
        raise ValueError(f"num_tails_rows has value: {num_tail_rows}")

    if missing_obs is not None and missing_obs <= 0:
        raise ValueError(f"missing_obs must be a positive integer or None, has value: {missing_obs}")

    # calculates a boolean mask that is True if a row has all NaN values
    nan_rows_mask = df.isnull().all(axis=1)
    nan_rows_mask = nan_rows_mask.tail(num_tail_rows)
    nan_rows_mask = nan_rows_mask[nan_rows_mask]

    # applies ewm smoothing to the input DataFrame
    df_smoothed = df.ewm(halflife=ewm_hl).mean()

    # sets rows with all NaN values at the tail of the original df to NaN in the smoothed df
    if len(nan_rows_mask) > 0:
        df_smoothed.loc[nan_rows_mask.index, :] = np.nan

    # sets a smoothed value to NaN if past N obs in the pre-smoothed df are also NaN
    # this prevents an ewm from smoothing a column into infinity without new data
    if missing_obs:
        missing_obs_mask = df.isnull().astype(float).rolling(missing_obs).sum().eq(missing_obs)
        df_smoothed[missing_obs_mask] = np.nan

    return df_smoothed


def mask_df_with_entry_exit_cond(
    df_in: pd.DataFrame, entry_val: float, exit_val: float, null_exits: bool = True
) -> pd.DataFrame:
    """
    Masks an input dataframe based on entry and exit thresholds.

        df_in.loc[i, j] >= entry_val  (entry condition)
        df_in.loc[i, j] <  exit_val   (exit condition)

    Parameters
    ----------
    df_in: pd.DataFrame
    entry_val: float
    exit_val: float
    null_exits: bool
        A null value will yield an exit from the mask.

    Returns
    -------
    mask: pd.DataFrame
    """

    if entry_val <= exit_val:
        raise ValueError("Entry value cannot be less than or equal to exit value.")

    mask: pd.DataFrame = pd.DataFrame(None, index=df_in.index, columns=df_in.columns)
    in_prev = pd.Series(False, index=df_in.columns)

    # loop over row index and build boolean mask sequentially
    for idx, row in df_in.iterrows():
        to_entr = row.ge(entry_val) | in_prev
        to_exit = row.lt(exit_val) | row.isnull() if null_exits else row.lt(exit_val)
        mask.loc[idx, :] = to_entr & ~to_exit
        in_prev = mask.loc[idx, :]

    return mask


def move_weekend_to_friday_eod(data, eod_time, remove_zeros=False):
    # move to a long table
    data.index.name = "DATE_"
    cols_melt = data.columns
    data = pd.melt(data.reset_index(), id_vars=["DATE_"], value_vars=cols_melt)
    data = data[~data["value"].isnull()]  # remove NaNs
    if remove_zeros:
        data = data[data["value"] != 0]
    data.columns = ["DATE_", "ID_", "VALUE"]

    # move Sunday to Friday
    td = pd.Timedelta("1 day")
    bd = pd.offsets.BusinessDay(n=1)
    data["DATE_MOVED"] = data["DATE_"] + td - bd
    data_tmp = data.groupby(["DATE_MOVED", "ID_"])["DATE_"].max().reset_index()
    data = data.merge(
        data_tmp,
        left_on=["DATE_MOVED", "ID_", "DATE_"],
        right_on=["DATE_MOVED", "ID_", "DATE_"],
        how="inner",
    )
    data["DATE_"] = data["DATE_MOVED"]
    data = data[["DATE_", "ID_", "VALUE"]]

    # pivot
    data = data.pivot(index="DATE_", columns="ID_", values="VALUE")

    # set time
    if eod_time is not None:
        data.index = make_timestamps_eod(data.index, eod_time=eod_time)

    return data


def extract_values_as_unique_list(data: pd.DataFrame) -> list:
    """
    Returns the values of a DataFrame as a unique list.

    Parameters
    ----------
    data: pd.DataFrame

    Returns
    -------
    all_values: list
    """

    melted_data = data.melt(value_name="value", ignore_index=True)
    melted_data = melted_data.dropna()
    melted_data = melted_data[["value"]].drop_duplicates()

    all_values: list = melted_data.values.ravel().tolist()

    return all_values


def cconcat(cols: list, labels: list = None) -> pd.DataFrame:
    """
    simple helper function to concatonate columns into a dataframe and add labels
    @param cols: the dataframes/series to concat
    @param labels: list of labels for the new dataframe
    @return: pd.DataFrame
    """

    data = pd.concat(cols, axis=1)
    if labels is not None:
        data.columns = labels
    return data
