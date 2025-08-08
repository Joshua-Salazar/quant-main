import requests
import json
import pandas as pd


def get_boss_positions(desk, dt, pms, token):
    """
    read the boss positions for a date and a desk
    @param desk:
    @param dt:
    @param pms:
    @param token:
    @return: dataframe for the positions
    """
    datestr = dt.strftime('%Y-%m-%d')

    env_name = 'cpcapdata'
    payload = {
        'token': token,
        'PMs': pms,
        'Desks': desk,
        'instDetails': 'True',
        'MetaData': 'true',
        'AsOfDateStr': datestr,
        'format': 'json',
    }

    r = requests.get('http://' + env_name + ':5555/api/ctp/boss', params=payload)
    json_string = r.content.decode("utf-8")
    positions = pd.read_json(json_string)

    return positions


def get_ctp_positions(pms, userid, token, snapshot=False, as_of_date=None):
    """
    read tht ctp positions for a given pm and augment it with portfolio info
    @param pms:
    @param userid:
    @param token:
    @return: dataframe for the positions
    """
    env_name = 'cpcapdata'
    payload = {
        'token': token,
        'pms': pms,
        'instDetails': 'True',
        'MetaData': 'true',
        'location': 'NYC',
        'source': 'CLOSE',
        'environment': 'CAPSTONE',
        'format': 'json',
    }
    if snapshot:
        assert as_of_date is not None
        payload["snapshot"] = "true"
        payload["asOfDate"] = as_of_date.strftime("%Y%m%d")

    r = requests.get('http://' + env_name + ':5555/api/ctp/risk', params=payload)
    json_string = r.content.decode("utf-8")
    json_obj = json.loads(json_string)
    positions_id = pd.read_json(json_string)

    # portfolio
    payload = {
        'token': token,
        'userid': userid,
        'pmids': list(set(positions_id["PMId"])),
        'format': 'json',
    }

    r = requests.get('http://' + env_name + ':5555/api/ctp/portfolio', params=payload)
    json_string = r.content.decode("utf-8")
    portfolios = pd.read_json(json_string)

    # merge
    positions = pd.merge(positions_id, portfolios, how="left", left_on=['PortfolioId'], right_on=['Id'],
                         suffixes=(None, '_y'), validate='m:1')
    duplicate_cols = ['PfoGroup', 'Desk', 'PM', 'PMId']
    for col in duplicate_cols:
        assert positions[col].equals(positions[col + '_y'])
        positions = positions.drop(col + '_y', axis=1)
    return positions
