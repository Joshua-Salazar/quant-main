from ..infrastructure.option_data_container import CassandraDSOnDemandFromCacheByYear


STOCK_OPTIONS_DAILY_DEFAULTS = {
    'hedged': True,
    'max_option_expiry_days': 90,
    'extra_data_requests': {},
    'allow_fill_forward_missing_data': 0,
    'use_listed': True,
    'cost_params': {},
    'expiration_rules': None,
    'keep_hedges_in_tranche_portfolio': False,
    'hedge_future_expiry_at_option_expiry': False,
    'number_of_futures_to_load': 2,
    'trade_first_day': True,
    'greeks_to_include': ['delta'],
    'inc_greeks': False,
    'scale_by_nav': False,
    'data_start_date_shift': 0,
    'allow_fix_option_price_from_settlement': False,
    'allow_reprice': False,
    'inc_trd_dts': False,
    'option_data_source': CassandraDSOnDemandFromCacheByYear(),
    'other_filters': [lambda x: x[x['price'] >= 0.05]],
}
