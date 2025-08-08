import getpass
import logging
import warnings
from datetime import time
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from snowflake.connector import SnowflakeConnection
from snowflake.connector.pandas_tools import write_pandas

from ..util.auth import get_credentials
from ..util.tstamps import replace_time


def get_snowflake_creds(profile=None):
    if profile is None:
        profile = getpass.getuser()
    creds = get_credentials("snowflake_%s" % profile)
    return creds


def load_query(query_name, file_path):
    with open(file_path + "/" + query_name + ".sql") as f:
        return f.read()


def run_query_with_ids_tstamps(
    con: Any,
    query_str: str,
    ids: Optional[List] = None,
    tstamps: Optional[List] = None,
    additional_params: Dict = {},
    pivot_values: Optional[List] = None,
    database: str = "SP_CASH_EQUITY_CAPSTONE",
    id_format: str = "STRING",
    drop_duplicates: bool = False,
    parse_dates: List = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs a query optionally with IDs and timestamps, returning a DataFrame or a
    dictionary of DataFrames if pivoting.

    Parameters
    ----------
    con: Any
    query_str: str
    ids: Optional[List]
    tstamps: Optional[List]
    additional_params: Dict
    pivot_values: Optional[List]
    database: str
    id_format: str
    drop_duplicates: bool
    parse_dates: List

    Returns
    -------
    pd.DataFrame | Dict[str, pd.DataFrame]
    """

    all_params = {}
    if tstamps is not None:
        con.execute_string(f"CREATE OR REPLACE TEMPORARY TABLE {database}.PUBLIC.TMP_DATES (DATE_ DATE)")
        dates_to_insert = pd.DataFrame({"DATE_": [d.strftime("%Y-%m-%d") for d in tstamps]})
        write_pandas(con, dates_to_insert, "TMP_DATES", database, schema="PUBLIC")

    if ids is not None:
        con.execute_string(f"CREATE OR REPLACE TEMPORARY TABLE {database}.PUBLIC.TMP_IDS (ID_ {id_format})")
        ids_to_insert = pd.DataFrame({"ID_": ids})
        write_pandas(con, ids_to_insert, "TMP_IDS", database, schema="PUBLIC")

    if len(additional_params) > 0:
        all_params = {**all_params, **additional_params}

    data = pd.read_sql_query(query_str.format_map(all_params), con, parse_dates=parse_dates)

    # (optionally) pivot and re-index
    if pivot_values is None:
        return data

    else:
        # convert to list of items to extract
        passed_list = True
        if type(pivot_values) == str:
            pivot_values = [pivot_values]
            passed_list = False

        # convert IDs back to strings
        if id_format == "INT":
            data.ID_ = data.ID_.astype(str)

        # remove invalid rows
        if tstamps is not None:
            data = data[~data.DATE_.isnull()]
        if ids is not None:
            data = data[~data.ID_.isnull()]

        # remove dupes
        if drop_duplicates:
            dupes = data[data.duplicated(subset=["ID_", "DATE_"])]
            num_dupes = dupes.shape[0]
            if num_dupes > 0:
                num_dupe_ids = len(dupes.ID_.unique())
                num_dupe_dates = len(dupes.DATE_.unique())
                logging.warning(
                    f"dropping {num_dupes} duplicated rows, corresponding to "
                    f"{num_dupe_ids} IDs, {num_dupe_dates} tstamps"
                )
                data = data.drop_duplicates(["ID_", "DATE_"], keep="first")

        # pivot each value
        all_data = dict()
        for pivot_on in pivot_values:
            # pivot
            data_pivot = data.pivot(index="DATE_", columns="ID_", values=pivot_on)

            # re-index
            if tstamps is not None:
                data_pivot = data_pivot.reindex(index=replace_time(tstamps, time(0, 0)))
                data_pivot.index = tstamps
            if ids is not None:
                data_pivot = data_pivot.reindex(columns=ids)

            # keep
            all_data[pivot_on] = data_pivot

        # if only one DataFrame, return as a DataFrame
        if passed_list:
            return all_data
        else:
            return all_data[pivot_values[0]]


def count_rows_and_unique_rows_in_table(
    con: SnowflakeConnection,
    table_name: str,
) -> pd.DataFrame:
    """
    Returns the count of all rows and unique rows in a Snowflake table. This can be useful
    if testing for duplicate rows in a table, for instance. You should specify the full table
    name unless the database namespace is in your snowflake connection:

        QUANT.PUBLIC.YOUR_TABLE_NAME

    Parameters
    ----------
    con: SnowflakeConnection
    table_name: str

    Returns
    -------

    result: pd.DataFrame
    """

    _query_str: str = f"""
    WITH UNIQUE_ROW_COUNT AS (
      SELECT
          COUNT(*) AS UNIQUE_ROW_COUNT
      FROM (SELECT DISTINCT * FROM {table_name})
    ),
    ALL_ROW_COUNT AS (
        SELECT COUNT(*) AS ALL_ROW_COUNT FROM {table_name}
    )
    SELECT
        A.UNIQUE_ROW_COUNT,
        B.ALL_ROW_COUNT
    FROM
        UNIQUE_ROW_COUNT A FULL OUTER JOIN ALL_ROW_COUNT B
    ;
    """

    cursor = con.cursor()
    cursor.execute(_query_str)
    result: pd.DataFrame = cursor.fetch_pandas_all()

    return result


# deprecated
def query_with_ids_tstamps(
    con,
    query_name,
    query_path,
    ids=None,
    tstamps=None,
    additional_params={},
    pivot_values=None,
    database="SP_CASH_EQUITY_CAPSTONE",
    id_format="STRING",
    drop_duplicates=False,
    parse_dates=None,
):
    """
    This function is deprecated. Use run_query_with_ids_tstamps instead.
    """

    warnings.warn(
        "query_with_ids_tstamps is deprecated, use run_query_with_ids_tstamps instead", DeprecationWarning, stacklevel=2
    )

    # load query
    logging.info(f"executing query { query_name}")
    raw_query = load_query(query_name, query_path)

    # build dictionary
    all_params = {}
    if tstamps is not None:
        con.execute_string('CREATE OR REPLACE TEMPORARY TABLE "%s"."PUBLIC".TMP_DATES (DATE_ DATE)' % database)
        dates_to_insert = pd.DataFrame({"DATE_": [d.strftime("%Y-%m-%d") for d in tstamps]})
        write_pandas(con, dates_to_insert, "TMP_DATES", database, schema="PUBLIC")

    if ids is not None:
        con.execute_string('CREATE OR REPLACE TEMPORARY TABLE "%s"."PUBLIC".TMP_IDS (ID_ %s)' % (database, id_format))
        ids_to_insert = pd.DataFrame({"ID_": ids})
        write_pandas(con, ids_to_insert, "TMP_IDS", database, schema="PUBLIC")

    if len(additional_params) > 0:
        all_params = {**all_params, **additional_params}

    # execute
    data = pd.read_sql_query(raw_query.format_map(all_params), con, parse_dates=parse_dates)

    # (optionally) pivot and re-index
    if pivot_values is None:
        return data

    else:
        # convert to list of items to extract
        passed_list = True
        if type(pivot_values) == str:
            pivot_values = [pivot_values]
            passed_list = False

        # convert IDs back to strings
        if id_format == "INT":
            data.ID_ = data.ID_.astype(str)

        # remove invalid rows
        if tstamps is not None:
            data = data[~data.DATE_.isnull()]
        if ids is not None:
            data = data[~data.ID_.isnull()]

        # remove dupes
        if drop_duplicates:
            dupes = data[data.duplicated(subset=["ID_", "DATE_"])]
            num_dupes = dupes.shape[0]
            if num_dupes > 0:
                num_dupe_ids = len(dupes.ID_.unique())
                num_dupe_dates = len(dupes.DATE_.unique())
                logging.warning(
                    f"dropping {num_dupes} duplicated rows, corresponding to {num_dupe_ids} IDs, {num_dupe_dates} tstamps"
                )
                data = data.drop_duplicates(["ID_", "DATE_"], keep="first")

        # pivot each value
        all_data = dict()
        for pivot_on in pivot_values:
            # pivot
            data_pivot = data.pivot(index="DATE_", columns="ID_", values=pivot_on)

            # re-index
            if tstamps is not None:
                data_pivot = data_pivot.reindex(index=replace_time(tstamps, time(0, 0)))
                data_pivot.index = tstamps
            if ids is not None:
                data_pivot = data_pivot.reindex(columns=ids)

            # keep
            all_data[pivot_on] = data_pivot

        # if only one DataFrame, return as a DataFrame
        if passed_list:
            return all_data
        else:
            return all_data[pivot_values[0]]
