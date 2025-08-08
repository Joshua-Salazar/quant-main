import requests

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
We will fill all_quotes with the results from the API call.
The key will be composed of the series, version and tenor:
    - For series 44, version 1 and tenor 5, the key will be '44_1_5' 
The value will be the quote itself:
    - The quote object is a dictionary with the following keys:
        - symbol: str
        - instId: int
        - series: int
        - version: int
        - tenor : int
        - ticker : str
        - indexType : str
            - "uSpreadBps" for index trade on spread
            - "uPrice" for index trade on price
        - rollId: str (roll associated)
        - rollSymbol: str
        - rollTicker: str
        - quotes: dict
            - key (Source: str)
                - CBBT: quote coming from BBG (Source MSG1)
                - CAPSTONE: quote that we compute using the OTR and the roll prices
            - value (float)
                - in BPS if indexType is "uSpreadBps"
                - in Price if indexType is "uPrice"
        - rollQuotes: dict
            - key (Broker source: str)
                - BMLE
                - BARX
                - EBNP
                - CCGC
                - DBVD
                - GSMX
                - JPGP
                - GSET
                - JCTT
                - RBCX
            - value (float)
"""

all_quotes = dict()

if cds_index_name in results.json():
    quotes = results.json()[cds_index_name]

    for quote in quotes:
        key = f"{quote['series']}_{quote['version']}_{quote['tenor']}"
        all_quotes[key] = quote

# I will need to get the key as 'series_version_tenor' as built above
quote = all_quotes[f'{series}_{version}_{tenor}']

spread = quote['quotes']

"""
    As source, You will have either
    CBBT: quote coming from BBG (Source MSG1)
    CAPSTONE: quote that we compute using the OTR and the roll prices
"""

if "CBBT" in spread:
    bbg_value = spread["CBBT"]
    print(f"CBBT: {bbg_value}\n")

if "CAPSTONE" in spread:
    capstone_value = spread["CAPSTONE"]
    print(f"CAPSTONE: {capstone_value}\n")

# print all brokers roll prices
rolls = quote['rollQuotes']
if len(rolls) > 0:
    print("Roll prices:")
    for broker, value in rolls.items():
        print(f"\t{broker}: {value}")
else:
    print("No roll prices available for this index.")
