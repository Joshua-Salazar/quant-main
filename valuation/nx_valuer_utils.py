import pandas as pd


def get_table_res(nx_res, type):
    table_res = nx_res.getTableResult(type)
    df_res = pd.DataFrame({x.key(): list(x.data()) for x in table_res})
    return df_res