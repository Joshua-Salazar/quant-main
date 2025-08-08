from ..dates.utils import timestamp_to_datetime
import pandas as pd
from ..data.Datalake3 import Datalake
DATALAKE = Datalake()


def get_bbg_history(tickers, field, start_date, end_date, throw_if_missing=True):
    """
    :param tickers: a list of tickers or single ticker
    :param field: e.g. PX_LAST, support types: str, list
    :param start_date: start datetime
    :param end_date: end datetime
    :param throw_if_missing: indicate if throw for missing. If not, we return empty dataframe
    :return: price in range start dt <= dt <= end dt.
    For example: start=2023-03-20_00:00:00, end=2023-03-21_16:00:43 return TWO prices
                 start=2023-03-20_01:00:00, end=2023-03-21_16:00:43 return ONE price on 2023-03-21
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    dfs = []
    for ticker in tickers:
        df = DATALAKE.getData('BBG_PRICE', ticker, field, start_date, end_date, None)
        if not df.empty:
            df = df.rename(columns={'tstamp': 'date'})
        if "date" not in df.columns:
            if throw_if_missing:
                raise Exception(f"Missing date for ticker {ticker} resulted df {df}")
            else:
                # return empty df if missing ticket from datalake
                return pd.DataFrame()
        df['date'] = df['date'].apply(lambda x: timestamp_to_datetime(pd.Timestamp(x)).isoformat())
        dfs.append(df)
    return pd.concat(dfs)

def reformat_df( data ):
    data.set_index(['tstamp'], inplace=True)
    new_data = []
    value = [ x for x in data.columns if x != 'ticker' ]
    assert len( value ) == 1
    for col in data.ticker.unique():
        temp_df = data[data.ticker == col]
        temp_df = temp_df.rename( columns = { value[0]: col } )
        temp_df = temp_df.drop(columns=['ticker'])
        if len( new_data ) == 0:
            new_data = temp_df
        else:
            new_data = new_data.join( temp_df)
    return new_data