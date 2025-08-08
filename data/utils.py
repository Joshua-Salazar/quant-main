import urllib
import sqlalchemy as sa
import pandas as pd
from sqlalchemy import text


def read_sql(sql_query, server, database, userid, password):
    """
    query a sql database with a sql query
    @param sql_query:
    @param server:
    @param database:
    @param userid:
    @param password:
    @return: the dataframe returned from the query
    """

    if database=='DBCTPPROD.capstoneco.com':
        engine=create_ctp_sql_con
    else:
        engine = create_engine(server, database, userid, password)

    with engine.connect() as connection:
        df = pd.read_sql(text(sql_query), connection)

    return df


def create_engine(server, database, userid, password):
    """
    create db engine
    @param server:
    @param database:
    @param userid:
    @param password:
    @return: the dataframe returned from the query
    """

    params = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=" + server + ";"
        "DATABASE=" + database + ";"
        "UID=" + userid + ";"
        "PWD=" + password + ";"
    )
    engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

    return engine

def create_ctp_sql_con():

    con=pyodbc.connect('DRIVER={SQL Server};Server=DBCTPPROD.capstoneco.com;Port=1433;UID=ctp;PWD=ctppw;Database=capstone;Trusted_connection=no')

    return con



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
