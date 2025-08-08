import requests
import datetime

"""
    The first part of the code is the same as in request_cds_index_spread_example.py.
    Here we will request the same backend to get all the series, version and tenor for a specific index.
    Then you can get the CTP id and request data_cache_creditvol to get the vol points.
"""


class CreditVolPoints:
    def __init__(self, date: datetime.date, strike: float, vol: float):
        self.date = date
        self.strike = strike
        self.vol = vol


class CreditVolObject:
    def __init__(self, index_name: str, index_id: int, environment: str):
        self.index_name = index_name
        self.index_id = index_id
        self.environment = environment
        self.vol = []

    def add_vol_point(self, vol: CreditVolPoints):
        self.vol.append(vol)


MAP_GENERIC_CDS_INDEX = {
    'CDX North America High Yield Index': 12206218,
    'CDX North America Investment Grade Index': 12206219,
    'CDX Emerging Markets Index': 12206217,
    'iTraxx Europe Subordinated Financial Index': 12206222,
    'iTraxx Europe Crossover Index': 12206223,
    'iTraxx Europe Index': 12206220,
    'iTraxx Europe Senior Financial Index': 12206221,
    'iTraxx Asia ex-Japan Investment Grade Index': 38770679,
    'iTraxx Japan Index': 12206224,
    'iTraxx Australia Index': 12206225
}

# choose the index you are interested in
cds_index_name = 'CDX North America Investment Grade Index'

# choose the series, version and tenor you are interested in
series = 43
version = 1
tenor = 5

cdx_ig_id = MAP_GENERIC_CDS_INDEX[cds_index_name]

url = f'http://ntprctp01:6684/getCDSIndices?id={cdx_ig_id}'

results = requests.get(url)

"""
    We will fill map_ctp_id with the results from the API call.
    The key will be composed of the series, version and tenor:
        - For series 44, version 1 and tenor 5, the key will be '44_1_5' 
    The value will be:
        - CTP id in int
"""

map_ctp_id = dict()

if cds_index_name in results.json():
    quotes = results.json()[cds_index_name]

    for quote in quotes:
        key = f"{quote['series']}_{quote['version']}_{quote['tenor']}"
        map_ctp_id[key] = quote['instId']

# I will need to get the key as 'series_version_tenor' as built above
ctp_id = map_ctp_id[f'{series}_{version}_{tenor}']

# Now we will get the vol points for the CTP id
env = 'CAPSTONE'

# this is coming from data_cache_creditvol
url = f'http://cpiceregistry:6703/creditvol?server_name=data_cache_creditvol&cdxid={ctp_id}&environment={env}'

results = requests.get(url)

data = results.json()

for d in data:
    credit_vol_object = CreditVolObject(cds_index_name, ctp_id, env)

    if len(d['expiry']) != len(d['strike']) != len(d['vol']):
        print("Expiry, strike and vol arrays must have the same length.")
        continue

    for i, exp in enumerate(d['expiry']):
        # date is in format YYYYMMDD
        vol_date = datetime.datetime.strptime(exp, '%Y%m%d').date()
        vol_strike = d['strike'][i]
        v = d['vol'][i]

        # create the vol point object
        vol_point = CreditVolPoints(vol_date, vol_strike, v)

        # add it to the object
        credit_vol_object.add_vol_point(vol_point)
