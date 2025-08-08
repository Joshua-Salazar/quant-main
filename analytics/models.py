import numpy as np
from dateutil.relativedelta import relativedelta
from ..dates.utils import count_business_days, get_business_days
from ..analytics.constants import YEAR_BUSINESS_DAY_COUNT, YEAR_CALENDAR_DAY_COUNT, RANDOM_NUMBER_SEED


def monte_carlo_daily_simulation(annualized_drift, annualized_vol,
                                 start_date, end_date,
                                 n_paths, holidays=[], return_type='paths'):
    dates = get_business_days(start_date, end_date, holidays)
    simulation = monte_carlo_simulation(annualized_drift, annualized_vol, dates, n_paths,
                                        use_business_days=True, return_type=return_type)
    return dates, simulation


def monte_carlo_simulation(annualized_drift, annualized_vol, dates, n_paths,
                           use_business_days=True, return_type='paths'):
    n_dates = len(dates)
    if use_business_days:
        dcf = np.array(list(map(lambda x, y: count_business_days(y + relativedelta(days=1), x) / YEAR_BUSINESS_DAY_COUNT, dates[1:], dates[:-1])))
    else:
        dcf = np.array(list(map(lambda x, y: (x - y).days / YEAR_CALENDAR_DAY_COUNT, dates[1:], dates[:-1])))
    np.random.seed(RANDOM_NUMBER_SEED)
    log_returns = np.random.normal(dcf * annualized_drift, np.sqrt(dcf) * annualized_vol, (n_paths, n_dates - 1))
    if return_type == 'log returns':
        return log_returns
    elif return_type == 'performances':
        performances = np.exp(log_returns)
        return performances
    elif return_type == 'simple returns':
        performances = np.exp(log_returns)
        return performances - 1.0
    elif return_type == 'paths':
        performances = np.exp(log_returns)
        return np.cumprod(np.concatenate((np.ones((n_paths, 1)), performances), axis=1), axis=1)
    else:
        raise RuntimeError('Unknown return type ' + return_type)
