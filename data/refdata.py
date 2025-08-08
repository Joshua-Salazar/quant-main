import os
import pathlib

import pandas as pd
import requests

from ..data.utils import chunks


def get_underlyings_map(return_df):
    cache_file_pkl = "/var/ctp/data/underlyings_map.pkl"
    if os.path.exists(cache_file_pkl):
        underlyings_map = pd.read_pickle(cache_file_pkl)
    else:
        current_path = pathlib.Path(__file__).parent.resolve()

        cache_file = os.path.join(current_path, "underlyings_map.parquet.gzip")
        underlyings_map = pd.read_parquet(cache_file, engine='pyarrow')

    if return_df:
        return underlyings_map

    und_id_map = underlyings_map[['m_symbolMap_vBloomberg', 'm_id']].dropna().set_index("m_symbolMap_vBloomberg")
    und_id_map = und_id_map.to_dict()["m_id"]

    return und_id_map


def get_instruments_info(instrument_ids, interested_cols=None, chunk_size=200):
    """
    get the instruments information
    @param instrument_ids:
    @param interested_cols:
    @return: dataframe for instruments info with only the instrested columns
    """
    env_name = 'cpcapdata'
    results = []
    for chunk in chunks(instrument_ids, chunk_size):
        payload = {
            'instids': chunk,
            'format': 'json',
        }
        r = requests.get('http://' + env_name + ':5555/api/ctp/instrument', params=payload)
        json_string = r.content.decode("utf-8")
        instruments_info = pd.read_json(json_string)
        if interested_cols is not None:
            instruments_info = instruments_info[interested_cols]
        results.append(instruments_info)
    return pd.concat(results)
