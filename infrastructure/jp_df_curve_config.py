df_curve_config = {
    'USD': {
        "OIS": 'MTE_usd/ois/fomc/1d/rate,MTE_usd/ois/fomc/1m/rate,MTE_usd/ois/fomc/2m/rate,MTE_usd/ois/fomc/3m/rate,MTE_usd/ois/fomc/4m/rate,MTE_usd/ois/fomc/5m/rate,MTE_usd/ois/fomc/6m/rate,' +
               'MTE_usd/ois/fomc/7m/rate,MTE_usd/ois/fomc/8m/rate,MTE_usd/ois/fomc/9m/rate,MTE_usd/ois/fomc/10m/rate,MTE_usd/ois/fomc/11m/rate,MTE_usd/ois/fomc/12m/rate,MTE_usd/ois/fomc/13m/rate,' +
               'MTE_usd/ois/fomc/14m/rate,MTE_usd/ois/fomc/15m/rate,MTE_usd/ois/fomc/16m/rate,MTE_usd/ois/fomc/17m/rate,MTE_usd/ois/fomc/18m/rate,MTE_usd/ois/fomc/19m/rate,MTE_usd/ois/fomc/20m/rate,' +
               'MTE_usd/ois/fomc/21m/rate,MTE_usd/ois/fomc/22m/rate,MTE_usd/ois/fomc/23m/rate,MTE_usd/ois/fomc/24m/rate,MTE_usd/ois/fomc/27m/rate,MTE_usd/ois/fomc/30m/rate,MTE_usd/ois/fomc/33m/rate,' +
               'MTE_usd/ois/fomc/36m/rate,MTE_usd/ois/fomc/39m/rate,MTE_usd/ois/fomc/42m/rate,MTE_usd/ois/fomc/45m/rate,MTE_usd/ois/fomc/48m/rate,MTE_usd/ois/fomc/51m/rate,MTE_usd/ois/fomc/54m/rate,' +
               'MTE_usd/ois/fomc/57m/rate,MTE_usd/ois/fomc/60m/rate,MTE_usd/ois/fomc/6y/rate,MTE_usd/ois/fomc/7y/rate,MTE_usd/ois/fomc/8y/rate,MTE_usd/ois/fomc/9y/rate,MTE_usd/ois/fomc/10y/rate,' +
               'MTE_usd/ois/fomc/15y/rate,MTE_usd/ois/fomc/20y/rate,MTE_usd/ois/fomc/25y/rate,MTE_usd/ois/fomc/30y/rate,MTE_usd/ois/fomc/40y/rate,MTE_usd/ois/fomc/50y/rate,MTE_usd/ois/fomc/60y/rate',
        'SWAP': 'FDER_LIBSWAP_1D_RT_MID,FDER_LIBSWAP_1M_RT_MID,FDER_LIBSWAP_2M_RT_MID,FDER_LIBSWAP_3M_RT_MID,FDER_LIBSWAP_4M_RT_MID,FDER_LIBSWAP_5M_RT_MID,FDER_LIBSWAP_6M_RT_MID,' +
                'FDER_LIBSWAP_9M_RT_MID,FDER_LIBSWAP_1Y_RT_MID,' +
                'FDER_PARSWAP_18M_RT_MID,FDER_PARSWAP_2Y_RT_MID,FDER_PARSWAP_3Y_RT_MID,FDER_PARSWAP_4Y_RT_MID,FDER_PARSWAP_5Y_RT_MID,FDER_PARSWAP_6Y_RT_MID,FDER_PARSWAP_7Y_RT_MID,FDER_PARSWAP_8Y_RT_MID,' +
                'FDER_PARSWAP_9Y_RT_MID,FDER_PARSWAP_10Y_RT_MID,FDER_PARSWAP_12Y_RT_MID,FDER_PARSWAP_15Y_RT_MID,FDER_PARSWAP_20Y_RT_MID,FDER_PARSWAP_25Y_RT_MID,FDER_PARSWAP_30Y_RT_MID,' +
                'FDER_PARSWAP_35Y_RT_MID,FDER_PARSWAP_40Y_RT_MID,FDER_PARSWAP_50Y_RT_MID,FDER_PARSWAP_60Y_RT_MID',
        'SOFR': 'FCRV_SOFR_SWAP_ZERO_1D_RT_MID,FCRV_SOFR_SWAP_ZERO_1M_RT_MID,FCRV_SOFR_SWAP_ZERO_2M_RT_MID,FCRV_SOFR_SWAP_ZERO_3M_RT_MID,FCRV_SOFR_SWAP_ZERO_4M_RT_MID,FCRV_SOFR_SWAP_ZERO_6M_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_9M_RT_MID,FCRV_SOFR_SWAP_ZERO_1Y_RT_MID,FCRV_SOFR_SWAP_ZERO_1Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_2Y_RT_MID,FCRV_SOFR_SWAP_ZERO_2Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_3Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_3Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_4Y_RT_MID,FCRV_SOFR_SWAP_ZERO_4Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_5Y_RT_MID,FCRV_SOFR_SWAP_ZERO_5Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_6Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_6Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_7Y_RT_MID,FCRV_SOFR_SWAP_ZERO_7Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_8Y_RT_MID,FCRV_SOFR_SWAP_ZERO_8Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_9Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_9Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_10Y_RT_MID,FCRV_SOFR_SWAP_ZERO_10Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_11Y_RT_MID,FCRV_SOFR_SWAP_ZERO_11Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_12Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_12Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_13Y_RT_MID,FCRV_SOFR_SWAP_ZERO_13Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_14Y_RT_MID,FCRV_SOFR_SWAP_ZERO_14Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_15Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_15Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_16Y_RT_MID,FCRV_SOFR_SWAP_ZERO_16Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_17Y_RT_MID,FCRV_SOFR_SWAP_ZERO_17Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_18Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_18Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_19Y_RT_MID,FCRV_SOFR_SWAP_ZERO_19Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_20Y_RT_MID,FCRV_SOFR_SWAP_ZERO_20Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_21Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_21Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_22Y_RT_MID,FCRV_SOFR_SWAP_ZERO_22Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_23Y_RT_MID,FCRV_SOFR_SWAP_ZERO_23Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_24Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_24Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_25Y_RT_MID,FCRV_SOFR_SWAP_ZERO_25Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_26Y_RT_MID,FCRV_SOFR_SWAP_ZERO_26Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_27Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_27Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_28Y_RT_MID,FCRV_SOFR_SWAP_ZERO_28Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_29Y_RT_MID,FCRV_SOFR_SWAP_ZERO_29Y6M_RT_MID,FCRV_SOFR_SWAP_ZERO_30Y_RT_MID,' +
                'FCRV_SOFR_SWAP_ZERO_35Y_RT_MID'
    },
    'CAD': {
        "OIS": 'MTE_cad/ois/0d/01D/rate,MTE_cad/ois/0d/01W/rate,MTE_cad/ois/0d/02W/rate,MTE_cad/ois/0d/01M/rate,MTE_cad/ois/0d/02M/rate,MTE_cad/ois/0d/03M/rate,' +
               'MTE_cad/ois/0d/04M/rate,MTE_cad/ois/0d/05M/rate,MTE_cad/ois/0d/06M/rate,MTE_cad/ois/0d/07M/rate,MTE_cad/ois/0d/08M/rate,MTE_cad/ois/0d/09M/rate,' +
               'MTE_cad/ois/0d/01Y/rate,MTE_cad/ois/0d/02Y/rate,MTE_cad/ois/0d/03Y/rate,MTE_cad/ois/0d/04Y/rate,MTE_cad/ois/0d/05Y/rate,MTE_cad/ois/0d/06Y/rate,' +
               'MTE_cad/ois/0d/07Y/rate,MTE_cad/ois/0d/08Y/rate,MTE_cad/ois/0d/09Y/rate,MTE_cad/ois/0d/10Y/rate',
    },
    'CHF': {
        "OIS": 'MTE_chf/ois/0d/01D/rate,MTE_chf/ois/0d/01W/rate,MTE_chf/ois/0d/02W/rate,MTE_chf/ois/0d/01M/rate,MTE_chf/ois/0d/02M/rate,MTE_chf/ois/0d/03M/rate,' +
               'MTE_chf/ois/0d/04M/rate,MTE_chf/ois/0d/05M/rate,MTE_chf/ois/0d/06M/rate,MTE_chf/ois/0d/07M/rate,MTE_chf/ois/0d/08M/rate,MTE_chf/ois/0d/09M/rate,' +
               'MTE_chf/ois/0d/01Y/rate,MTE_chf/ois/0d/02Y/rate,MTE_chf/ois/0d/03Y/rate,MTE_chf/ois/0d/04Y/rate,MTE_chf/ois/0d/05Y/rate,MTE_chf/ois/0d/06Y/rate,' +
               'MTE_chf/ois/0d/07Y/rate,MTE_chf/ois/0d/08Y/rate,MTE_chf/ois/0d/09Y/rate,MTE_chf/ois/0d/10Y/rate',
    },
    'GBP': {
        "OIS": 'MTE_gbp/ois/0d/01D/rate,MTE_gbp/ois/0d/01W/rate,MTE_gbp/ois/0d/02W/rate,MTE_gbp/ois/0d/01M/rate,MTE_gbp/ois/0d/02M/rate,MTE_gbp/ois/0d/03M/rate,' +
               'MTE_gbp/ois/0d/04M/rate,MTE_gbp/ois/0d/05M/rate,MTE_gbp/ois/0d/06M/rate,MTE_gbp/ois/0d/07M/rate,MTE_gbp/ois/0d/08M/rate,MTE_gbp/ois/0d/09M/rate,' +
               'MTE_gbp/ois/0d/01Y/rate,MTE_gbp/ois/0d/02Y/rate,MTE_gbp/ois/0d/03Y/rate,MTE_gbp/ois/0d/04Y/rate,MTE_gbp/ois/0d/05Y/rate,MTE_gbp/ois/0d/06Y/rate,' +
               'MTE_gbp/ois/0d/07Y/rate,MTE_gbp/ois/0d/08Y/rate,MTE_gbp/ois/0d/09Y/rate,MTE_gbp/ois/0d/10Y/rate',
    },
}

df_curve_config_pattern = {
    'USD': {
        'OIS': r'MTE_usd/ois/fomc/(\d{1,2})(y|m|w|d)/rate',
        'SWAP': r'FDER_(PAR|LIB)SWAP_(\d{1,2})(Y|M|W|D)_RT_MID',
        'SOFR': r'FCRV_SOFR_SWAP_ZERO_(\d{1,2})(Y|M|W|D)_RT_MID',
    },
    'CAD': {
        'OIS': r'MTE_cad/ois/0d/(\d{2})(Y|M|W|D)/rate',
    },
    'CHF': {
        'OIS': r'MTE_chf/ois/0d/(\d{2})(Y|M|W|D)/rate',
    },
    'GBP': {
        'OIS': r'MTE_gbp/ois/0d/(\d{2})(Y|M|W|D)/rate',
    },
}
