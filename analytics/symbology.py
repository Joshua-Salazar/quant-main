from ..constants.asset_class import AssetClass
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
import pandas as pd


IVOL_EXPIRATION_RULES = {
    'SPX Index': [
        '3rd Friday (Saturday before Feb 1 2015)',
        'End-of-month options',
        'Quarterly LEAPs',
        "Weekly's options",
    ],
    "SX5E Index": [
        '3rd Friday (Saturday before Feb 1 2015)',
        'End-of-month options',
        "Weekly's options",
    ],
    "NKY Index": [
        'Asia & Pacific expirations rule',
    ]
}

UNDERLYING_PRICING_ROOT_MAP = {
    585828: ['VIX Index','UX'],
    13854720: ['V2X Index','FVS'],
    4015054: ['CLA Comdty','CL'],
    4015061: ['NGA Comdty','NG'],
    4015078: ['GCA Comdty','GC'],
    4015077: ['SIA Comdty','SI'],
    4015076: ['HGA Comdty','HG'],
    4015058: ['C A Comdty','C'],
    4015055: ['S A Comdty','S'],
    4015072: ['W A Comdty','W'],

}

UNDERLYING_UND_ID_MAP = {
    'SPX Index': 65,
    'RTY Index': 58,
    'NDX Index': 50,
    'VIX Index': 585828,
    'V2X Index': 13854720,
    'CLA Comdty': 4015054,
    'NGA Comdty': 4015061,
    'GCA Comdty': 4015078,
    'SIA Comdty': 4015077,
    'HGA Comdty': 4015076,
    'C A Comdty': 4015058,
    'S A Comdty': 4015055,
    'W A Comdty': 4015072,
}


OPTION_FUTURE_TICKER_MAP = pd.DataFrame(
    data=[
        ['DAX Index', 'GX', 'Index', 'DAX', 'EUR', 'future', 'DAX Index', 'XFRA', 'equity'],
        ['AEX Index', '', 'Index', 'AEX', 'EUR', 'future', 'AEX Index', '', 'equity'],
        ['NDX Index', 'NQ', 'Index', 'NDX', 'USD', 'future', 'NDX Index', 'XCBO', 'equity'],
        ['NKY Index', 'NK', 'Index', 'NKY', 'JPY', 'future', 'NKY Index', 'XOSE', 'equity'],
        ['SMI Index', 'SM', 'Index', 'SMI', 'CHF', 'future', 'SMI Index', 'XSWX', 'equity'],
        ['SPX Index', 'ES', 'Index', 'SPX', 'USD', 'future', 'SPX Index', 'XCBO', 'equity'],
        ['SX5E Index', 'VG', 'Index', 'SX5E', 'EUR', 'future', 'SX5E Index', 'XFRA', 'equity'],
        ['SX5ED Index', 'dummy', 'Index', 'SX5ED', 'EUR', 'future', 'SX5ED Index', 'XFRA', 'equity'],
        ['RTY Index', 'RTY', 'Index', 'RUT', 'USD', 'future', 'RTY Index', 'XCBO', 'equity'],
        ['UKX Index', 'Z', 'Index', 'ESX', 'GBP', 'future', 'UKX Index', 'IFLO', 'equity'],
        ['VIX Index', 'UX', 'Index', 'VIX', 'USD', 'future', 'VIX Index', 'XCBO', 'future'],

        ['EEM US Equity', None, None, 'EEM', 'USD', 'stock', 'EEM Equity', 'XCBO', 'equity'],
        ['EFA US Equity', None, None, 'EFA', 'USD', 'stock', 'EFA Equity', 'XCBO', 'equity'],
        ['FXI US Equity', None, None, 'FXI', 'USD', 'stock', 'FXI Equity', 'XCBO', 'equity'],
        ['GDX US Equity', None, None, 'GDX', 'USD', 'stock', 'GDX Equity', 'XCBO', 'equity'],
        ['GLD US Equity', None, None, 'GLD', 'USD', 'stock', 'GLD Equity', 'XCBO', 'equity'],
        ['HYG US Equity', None, None, 'HYG', 'USD', 'stock', 'HYG Equity', 'XCBO', 'equity'],
        ['IWM US Equity', None, None, 'IWM', 'USD', 'stock', 'IWM Equity', 'XCBO', 'equity'],
        ['QQQ US Equity', None, None, 'QQQ', 'USD', 'stock', 'QQQ Equity', 'XCBO', 'equity'],
        ['USO US Equity', None, None, 'USO', 'USD', 'stock', 'USO Equity', 'XCBO', 'equity'],
        ['UUP US Equity', None, None, 'UUP', 'USD', 'stock', 'UUP Equity', 'XCBO', 'equity'],
        ['TLT US Equity', None, None, 'TLT', 'USD', 'stock', 'TLT Equity', 'XCBO', 'equity'],
        ['TQQQ US Equity', None, None, 'TQQQ', 'USD', 'stock', 'TQQQ Equity', 'XCBO', 'equity'],
        ['AAPL US Equity', 'AAPL', 'Equity', 'AAPL', 'USD', 'stock', 'AAPL Equity', 'XNYM', 'equity'],
        ['SXXP Index', 'SXO', 'Index', 'SXXP', 'EUR', 'future', 'SXXP Index', 'XFRA', 'equity'],

        ['CL', 'CL', 'Comdty', 'CL', 'USD', 'future', 'CL', 'XCBO', 'future'],
        ['CRUDE', 'CL', 'Comdty', 'CL', 'USD', 'future', 'CL', 'XCBO', 'future'],
        ['CO', 'CO', 'Comdty', 'CO', 'USD', 'future', 'CO', 'XCBO', 'future'],
        ['BRENT', 'CO', 'Comdty', 'CO', 'USD', 'future', 'CO', 'XCBO', 'future'],
        ['HG', 'HG', 'Comdty', 'HG', 'USD', 'future', 'HG', 'XCBO', 'future'],
        ['COPPER', 'HG', 'Comdty', 'HG', 'USD', 'future', 'HG', 'XCBO', 'future'],
        ['S', 'S', 'Comdty', 'S', 'USD', 'future', 'S', 'XCBO', 'future'],
        ['SOYBEAN', 'S', 'Comdty', 'S', 'USD', 'future', 'S', 'XCBO', 'future'],
        ['C', 'C', 'Comdty', 'C', 'USD', 'future', 'C', 'XCBO', 'future'],
        ['CORN', 'C', 'Comdty', 'S', 'USD', 'future', 'C', 'XCBO', 'future'],

        # Todo list: verify the future root and calendar for the following block of underlying
        #  so far it is only use for root and ticker mapping
        ['V2X Index', 'FVS', 'Index', 'V2X', 'USD', 'future', 'V2X Index', 'XFRA', 'future'],
        ['HSI Index', 'HI', 'Index', 'HSI', 'HKD', 'future', 'HSI Index', 'XHKG', 'equity'],
        ['HSCEI Index', 'HC', 'Index', 'HSCEI', 'HKD', 'future', 'HSCEI Index', 'XHKG', 'equity'],
        ['CAC Index', 'CF', 'Index', 'CAC', 'EUR', 'future', 'CAC Index', 'XPAR', 'equity'],
        ['FTSEMIB Index', 'ST', 'Index', 'FTSEMIB', 'EUR', 'future', 'FTSEMIB Index', 'XEUR', 'equity'],
        ['SX7E Index', 'CA', 'Index', 'SX7E', 'EUR', 'future', 'SX7E Index', 'NEOE', 'equity'],
        ['AS51 Index', 'XP', 'Index', 'AS51', 'AUD', 'future', 'AS51 Index', 'XNEC', 'equity'],
        ['KOSPI2 Index', 'KM', 'Index', 'KOSPI2', 'KRW', 'future', 'KOSPI2 Index', 'XKFE', 'equity'],
        ['EWC US Equity', None, None, 'EWC', 'USD', 'stock', 'EWC Equity', 'XCBO', 'equity'],
        ['EWJ US Equity', None, None, 'EWJ', 'USD', 'stock', 'EWJ Equity', 'XCBO', 'equity'],
        ['EWZ US Equity', None, None, 'EWZ', 'USD', 'stock', 'EWZ Equity', 'XCBO', 'equity'],
        ['EWY US Equity', None, None, 'EWY', 'USD', 'stock', 'EWY Equity', 'XCBO', 'equity'],
        ['IYR US Equity', None, None, 'IYR', 'USD', 'stock', 'IYR Equity', 'XCBO', 'equity'],
        ['XME US Equity', None, None, 'XME', 'USD', 'stock', 'XME Equity', 'XCBO', 'equity'],
        ['XOP US Equity', None, None, 'XOP', 'USD', 'stock', 'XOP Equity', 'XCBO', 'equity'],
        ['XLK US Equity', None, None, 'XLK', 'USD', 'stock', 'XLK Equity', 'XCBO', 'equity'],
        ['SMH US Equity', None, None, 'SMH', 'USD', 'stock', 'SMH Equity', 'XCBO', 'equity'],
        ['XLB US Equity', None, None, 'XLB', 'USD', 'stock', 'XLB Equity', 'XCBO', 'equity'],
        ['XLE US Equity', None, None, 'XLE', 'USD', 'stock', 'XLE Equity', 'XCBO', 'equity'],
        ['XLF US Equity', None, None, 'XLF', 'USD', 'stock', 'XLF Equity', 'XCBO', 'equity'],
        ['XLI US Equity', None, None, 'XLI', 'USD', 'stock', 'XLI Equity', 'XCBO', 'equity'],
        ['XLP US Equity', None, None, 'XLP', 'USD', 'stock', 'XLP Equity', 'XCBO', 'equity'],
        ['XLU US Equity', None, None, 'XLU', 'USD', 'stock', 'XLU Equity', 'XCBO', 'equity'],
        ['XLV US Equity', None, None, 'XLV', 'USD', 'stock', 'XLV Equity', 'XCBO', 'equity'],
        ['XLY US Equity', None, None, 'XLY', 'USD', 'stock', 'XLY Equity', 'XCBO', 'equity'],
        ['XRT US Equity', None, None, 'XRT', 'USD', 'stock', 'XLB Equity', 'XCBO', 'equity'],
        ['KRE US Equity', None, None, 'KRE', 'USD', 'stock', 'KRE Equity', 'XCBO', 'equity'],
        ['FXE US Equity', None, None, 'FXE', 'USD', 'stock', 'FXE Equity', 'XCBO', 'equity'],
        ['FXA US Equity', None, None, 'FXA', 'USD', 'stock', 'FXA Equity', 'XCBO', 'equity'],
        ['FXY US Equity', None, None, 'FXY', 'USD', 'stock', 'FXY Equity', 'XCBO', 'equity'],
        ['FXC US Equity', None, None, 'FXC', 'USD', 'stock', 'FXC Equity', 'XCBO', 'equity'],
        ['FXB US Equity', None, None, 'FXB', 'USD', 'stock', 'FXB Equity', 'XCBO', 'equity'],
        ['FXF US Equity', None, None, 'FXF', 'USD', 'stock', 'FXF Equity', 'XCBO', 'equity'],
        ['IEF US Equity', None, None, 'IEF', 'USD', 'stock', 'IEF Equity', 'XCBO', 'equity'],
        ['SHY US Equity', None, None, 'SHY', 'USD', 'stock', 'SHY Equity', 'XCBO', 'equity'],
        ['TIP US Equity', None, None, 'TIP', 'USD', 'stock', 'TIP Equity', 'XCBO', 'equity'],
        ['EMB US Equity', None, None, 'EMB', 'USD', 'stock', 'EMB Equity', 'XCBO', 'equity'],
        ['LQD US Equity', None, None, 'LQD', 'USD', 'stock', 'LQD Equity', 'XCBO', 'equity'],
        ['BKLN US Equity', None, None, 'BKLN', 'USD', 'stock', 'BKLN Equity', 'XCBO', 'equity'],
        ['SLV US Equity', None, None, 'SLV', 'USD', 'stock', 'SLV Equity', 'XCBO', 'equity'],

        ['TWSE Index', 'dummy', 'Index', 'TWSE', 'TWD', 'stock', 'TWSE Index', 'dummy', 'equity'],
        ['ASHR US Equity', None, None, 'ASHR', 'CNY', 'stock', 'ASHR Index', 'dummy', 'equity'],
        ['APP US Equity', None, None, 'APP', 'USD', 'stock', 'APP Equity', 'XCBO', 'equity'],
    ],
    columns=['ticker', 'future_root', 'suffix', 'option_root', 'currency', 'hedging_instrument', 'ticker_short',
             'option_calendar', 'option_underlying_type'],
)

TICKER_FROM_TICKER_SHORT = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker_short'].values, OPTION_FUTURE_TICKER_MAP['ticker'].values))

TICKER_FROM_FUTURE_ROOT = dict(zip(OPTION_FUTURE_TICKER_MAP['future_root'].values, OPTION_FUTURE_TICKER_MAP['ticker'].values))

TICKER_FROM_OPTION_ROOT = dict(zip(OPTION_FUTURE_TICKER_MAP['option_root'].values, OPTION_FUTURE_TICKER_MAP['ticker'].values))

CURRENCY_FROM_OPTION_ROOT = dict(zip(OPTION_FUTURE_TICKER_MAP['option_root'].values, OPTION_FUTURE_TICKER_MAP['currency'].values))

CURRENCY_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['currency'].values))

OPTION_ROOT_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['option_root'].values))

OPTION_CALENDAR_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['option_calendar'].values))

OPTION_UNDERLYING_TYPE_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['option_underlying_type'].values))

HEDGING_INSTRUMENT_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['hedging_instrument'].values))

FUTURE_ROOT_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, OPTION_FUTURE_TICKER_MAP['future_root'].values))

FUTURE_ROOT_SUFFIX_FROM_TICKER = dict(zip(OPTION_FUTURE_TICKER_MAP['ticker'].values, zip(OPTION_FUTURE_TICKER_MAP['future_root'].values, OPTION_FUTURE_TICKER_MAP['suffix'].values)))


def ticker_from_ticker_short(ticker: str):
    return TICKER_FROM_TICKER_SHORT[ticker]


def hedging_instrument_from_ticker(ticker: str):
    return HEDGING_INSTRUMENT_FROM_TICKER[ticker]


def option_root_from_ticker(root: str, asset_type=AssetClass.EQUITY):
    if asset_type == AssetClass.FX:
        return root
    return OPTION_ROOT_FROM_TICKER[root]


def option_calendar_from_ticker(ticker: str):
    return OPTION_CALENDAR_FROM_TICKER[ticker]


def option_calendar_from_ticker_lower(ticker: str):
    for actual_key in OPTION_CALENDAR_FROM_TICKER.keys():
        if actual_key.lower() == ticker.lower():
            return OPTION_CALENDAR_FROM_TICKER[actual_key]


def option_underlying_type_from_ticker(ticker: str, market=None):
    if OPTION_UNDERLYING_TYPE_FROM_TICKER[ticker] != "future" and ticker == "HSCEI Index" and market is not None:
        if market.has_item(market_utils.create_vol_surface_key(ticker)):
            if market.get_vol_surface(ticker).underlying_type == UnderlyingType.FUTURES:
                return "future"
    return OPTION_UNDERLYING_TYPE_FROM_TICKER[ticker]


def future_root_from_ticker(ticker: str):
    return FUTURE_ROOT_FROM_TICKER[ticker]


def ticker_from_future_root(root: str):
    return TICKER_FROM_FUTURE_ROOT[root]


def ticker_from_option_root(root: str):
    root = "RUT" if root == "RTY" else root
    return TICKER_FROM_OPTION_ROOT[root]


def currency_from_option_root(root: str):
    return CURRENCY_FROM_OPTION_ROOT[root]


def currency_from_ticker(ticker: str):
    return CURRENCY_FROM_TICKER[ticker]


def currency_from_ticker_lower(ticker: str):
    for actual_key in CURRENCY_FROM_TICKER.keys():
        if actual_key.lower() == ticker.lower():
            return CURRENCY_FROM_TICKER[actual_key]


def future_root_suffix_from_ticker(ticker: str):
    return FUTURE_ROOT_SUFFIX_FROM_TICKER[ticker]
