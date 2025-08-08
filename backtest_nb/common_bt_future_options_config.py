from datetime import datetime, time

# TY: 10-Year T-Note
# FV: 5-Year T-Note
# TU: 2-Year T-Note
# US: T-Bound
# DU: Euro-Schatz 1.75Y-2.25Y
# OE: Euro-Bobl 4.5Y-5.5Y
# RX: Euro-Bund 8.5Y-10.5Y
# IK: Italy bond future
# SFR: sofr future option
# SR1: sofr 1m future option
# SR3: sofr 3m future option
# SFR: sofr future option
# FF: Fed fund future option

SKIP_MONTHS_DICT = {
    'GC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'HG': ['F', 'G', 'J', 'M', 'Q', 'U', 'V', 'X'],
    'SR3': ['F', 'G', 'J', 'K', 'N', 'Q', 'V', 'X'],  # 3-month SOFR only expiry on H, M, U, Z
}

TENOR_DICT = {
    'GC': {'fut_tgt': '2M', 'fut_min': '45D', 'opt_tgt': '1M'},
    'CL': {'fut_tgt': '1M', 'fut_min': '1D', 'opt_tgt': '1M'},
    'CO': {'fut_tgt': '1M', 'fut_min': '1D', 'opt_tgt': '1M'},
    'NG': {'fut_tgt': '1M', 'fut_min': '1D', 'opt_tgt': '1M'},
    'C': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'S': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'TY': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'HG': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'TU': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'FV': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'US': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'DU': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'OE': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'RX': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'IK': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    # 'SFR': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'SR1': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'SR3': {'fut_tgt': '4M', 'fut_min': '4M', 'opt_tgt': '1M'},
    'FF': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
    'ES': {'fut_tgt': '45D', 'fut_min': '45D', 'opt_tgt': '1M'},
}

CALENDAR_DICT = {
    'GC': ['XCEC'],
    'CL': ['XNYM'],
    'CO': ['IFEU'],
    'NG': ['XNYM'] + [datetime(2016, 8, 22)],
    'C': ['FCBT-CBA'] + [datetime(2010, 1, 12), datetime(2010, 4, 1), datetime(2010, 10, 8),
                         datetime(2011, 2, 11),
                         datetime(2011, 2, 14), datetime(2011, 2, 15),
                         datetime(2011, 2, 22), datetime(2011, 3, 15), datetime(2011, 3, 17),
                         datetime(2011, 3, 31), datetime(2013, 3, 28), datetime(2022, 12, 7)],
    'S': ['FCBT-CBA'] + [datetime(2011, 2, 11), datetime(2011, 2, 14), datetime(2011, 2, 15),
                         datetime(2012, 9, 17),
                         datetime(2016, 8, 22)],
    'TY': ['FCBT-CME'] + [datetime(2010, 4, 2), datetime(2011, 2, 11), datetime(2011, 2, 14),
                          datetime(2011, 2, 15),
                          datetime(2012, 4, 6),
                          datetime(2015, 4, 3), datetime(2021, 4, 2)],
    'HG': ['XCEC'] + [datetime(2020, 3, 23), datetime(2020, 10, 19)],
    'TU': ['FCBT-CME'],
    'SR3': ['FCBT-CME'],
    'ES': ['XCME'],
}

CLOSING_DICT = {
    'GC': (time(13, 15, 0), time(12, 45, 0)),
    'CL': (time(14, 15, 0), time(12, 45, 0)),
    'CO': (time(14, 15, 0), time(12, 45, 0)),
    'NG': (time(14, 15, 0), time(12, 45, 0)),
    'C': (time(14, 0, 0), time(12, 45, 0)),
    'S': (time(14, 0, 0), time(12, 45, 0)),
    'TY': (time(14, 45, 0), time(12, 45, 0)),
    'HG': (time(12, 45, 0), time(12, 45, 0)),
    'TU': (time(14, 45, 0), time(12, 45, 0)),
    'FV': (time(14, 45, 0), time(12, 45, 0)),
    'US': (time(14, 45, 0), time(12, 45, 0)),
    'DU': (time(14, 45, 0), time(12, 45, 0)),
    'OE': (time(14, 45, 0), time(12, 45, 0)),
    'RX': (time(14, 45, 0), time(12, 45, 0)),
    'IK': (time(14, 45, 0), time(12, 45, 0)),
    'SFR': (time(14, 45, 0), time(12, 45, 0)),
    'SR1': (time(14, 45, 0), time(12, 45, 0)),
    'SR3': (time(14, 45, 0), time(12, 45, 0)),
    'FF': (time(14, 45, 0), time(12, 45, 0)),
    'ES': (time(14, 15, 0), time(12, 45, 0)),
}

IV_DICT = {
    'GC': 'GC 1M 100 VOL BVOL Comdty',
    'CL': 'CL 1M 100 VOL BVOL Comdty',
    'NG': 'NG 1M 100 VOL BVOL Comdty',
    'C': 'C  1M 100 VOL BVOL Comdty',
    'S': 'S  1M 100 VOL BVOL Comdty',
    'TY': 'TY 1M 100 VOL BVOL Comdty',
    'HG': 'HG 1M 100 VOL BVOL Comdty',
    'CO': 'CO 1M 100 VOL BVOL Comdty',
}

COST_DICT = {
    'GC': {'tc_delta': 0.1 / 4, 'tc_vega': 0.5 / 2.21 / 4},
    'CL': {'tc_delta': 0.01 / 4, 'tc_vega': 0.04 / 0.099 / 4},
    'NG': {'tc_delta': 0.001 / 4, 'tc_vega': 0.004 / 0.0026 / 4},
    'C': {'tc_delta': 0.25 / 4, 'tc_vega': 0.25 / 0.36 / 4},
    'S': {'tc_delta': 0.25 / 4, 'tc_vega': 0.75 / 0.95 / 4},
    'TY': {'tc_delta': 1 / 32 / 4, 'tc_vega': 1 / 64 / 0.1 / 4},
    'HG': {'tc_delta': 0.0005 / 4, 'tc_vega': 0.4 / 0.42 / 4},
    'CO': {'tc_delta': 0.01 / 4, 'tc_vega': 0.04 / 0.099 / 4},
}