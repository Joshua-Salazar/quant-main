import requests
import json
import pandas as pd


def get_trades(dt, pms, token):
    """
    read the boss positions for a date and a desk
    @param dt:
    @param pms:
    @param token:
    @return: dataframe for trades
    """
    datestr = dt.strftime('%Y%m%d')
    env_name = 'cpcapdata'
    payload = {
        'token': token,
        'PMs': pms,
        'AsOfDate': datestr,
        'format': 'json',
    }

    r = requests.get('http://' + env_name + ':5555/api/ctp/execs', params=payload)
    json_string = r.content.decode("utf-8")
    trades = pd.read_json(json_string)

    return trades


def get_do_not_trade_df(userid, token):
    pass
    # payload = {'token': token,
    #            'userid': userid,
    #            'format': 'json'}
    #
    #
    #
    # try:
    #     return pd.DataFrame.from_dict(res.json())
    # except:
    #     return pd.DataFrame(
    #         columns=['Id', 'InstId', 'Desk', 'RestrictionType', 'StartDate', 'EndDate', 'Notes', 'Modified',
    #                  'Symbol', 'BBGSymbol', 'Cusip', 'Isin', 'Sedol', 'Country'])


def create_order(acct_id, pfo_id, inst_id, order_side, qty, user, pmid, locate_broker_id):
    pass
