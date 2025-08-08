import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from datetime import datetime
import pandas as pd
from ..data.datalake import get_bbg_history

HISTORICAL_SCENARIOS = [
    {'name': '2000-2002', 'period': [datetime(2000, 3, 24), datetime(2002, 10, 30)]},
    {'name': '2007-2009', 'period': [datetime(2007, 10, 1), datetime(2009, 3, 31)]},
    {'name': 'April-May 2010', 'period': [datetime(2010, 4, 13), datetime(2010, 5, 28)]},
    {'name': 'Summer 2011', 'period': [datetime(2011, 7, 1), datetime(2011, 10, 3)]},
    {'name': 'August 2015', 'period': [datetime(2015, 8, 3), datetime(2015, 8, 31)]},
    {'name': 'February-March 2018', 'period': [datetime(2018, 1, 31), datetime(2018, 2, 28)]},
    {'name': 'Q4 2018', 'period': [datetime(2018, 9, 28), datetime(2018, 12, 31)]},
    {'name': 'Q1 2020', 'period': [datetime(2019, 12, 31), datetime(2020, 4, 30)]},
]


def truncate_historical_scenario(series, period, date_column_name='date'):
    scenario_start = period[0]
    scenario_end = period[1]
    dates = sorted(series[date_column_name].dt.to_pydatetime())
    for d in dates:
        if d >= scenario_end:
            break
    end = d
    for d in reversed(dates):
        if d <= scenario_start:
            break
    start = d
    if start <= scenario_start and end >= scenario_end:
        return series[
            (pd.Timestamp(start) <= series[date_column_name]) & (series[date_column_name] <= pd.Timestamp(end))]
    else:
        return pd.DataFrame()


def levels_df(tickers, start_date, end_date, how='left', get_levels_func=None):
    """
    return a df with column date and each of the tickers, with row with any NA removed
    @param tickers:
    @param start_date:
    @param end_date:
    @return:
    """
    levels = pd.DataFrame()
    for ticker in list(set(tickers)):
        if get_levels_func is None:
            this_levels = get_bbg_history([ticker], 'PX_LAST', start_date, end_date)
        else:
            this_levels = get_levels_func([ticker], 'PX_LAST', start_date, end_date)
        this_levels = this_levels.rename(columns={'PX_LAST': ticker}).drop('ticker', axis=1)
        if levels.empty:
            levels = this_levels
        else:
            levels = levels.merge(this_levels, on='date', how=how)
            levels = levels[~levels[ticker].isna()]
    return levels


def returns_df(tickers, start_date, end_date, performance_type='start to end',
               simple_return=True, return_step=1, include_return_start_date=False, levels=None, how='left',
               get_levels_func=None):
    if levels is None:
        levels = levels_df(tickers, start_date, end_date, how=how, get_levels_func=get_levels_func)
    returns = pd.DataFrame()
    for ticker in list(set(tickers)):
        this_returns = get_returns(levels, ticker, performance_type=performance_type, simple_return=simple_return,
                                   result_as_list=False, return_step=return_step,
                                   include_return_start_date=include_return_start_date)
        if returns.empty:
            returns = this_returns
        else:
            if include_return_start_date:
                returns = returns.merge(this_returns, on=['date', 'date_start'], how=how)
            else:
                returns = returns.merge(this_returns, on='date', how=how)
    return returns


def get_returns(df, column, performance_type='start to end', simple_return=True, result_as_list=True,
                date_column_name='date', return_step=1, include_return_start_date=False):
    """
    Calculates the returns for a given column in a dataframe.
    :param df: The dataframe containing the column.
    :param column: The column to be used.
    :return: The dataframe with the dates and returns of selected column, or the returns as list of numbers.
    the date column is the end day of the return period
    if include_return_start_date is true the dataframe contains a column with return start date as well
    """
    if performance_type == 'start to end':
        perf = df[column] / df[column].shift(return_step)
    elif performance_type == 'start to min':
        perf = df[column].rolling(window=return_step + 1).min() / df[column].shift(return_step)
    else:
        raise RuntimeError(f'Unknown performance calculation type {performance_type}')

    if simple_return:
        returns = perf - 1.0
    else:
        returns = np.log(perf)
    dates = df[date_column_name]

    if result_as_list:
        return pd.DataFrame(data={date_column_name: dates[return_step:], column: returns[return_step:]})[column].values
    else:
        if include_return_start_date:
            return pd.DataFrame(data={date_column_name: dates[return_step:].values,
                                      date_column_name + '_start': dates[:-return_step].values,
                                      column: returns[return_step:].values})
        else:
            return pd.DataFrame(
                data={date_column_name: dates[return_step:].values, column: returns[return_step:].values})


def annual_simple_return(returns, annualization_factor, input_is_simple_return=True):
    """
    Given return series, calulate the annualized simple return
    @param returns: input returns series, frequency is represented by the annualization_factor
    @param annualization_factor: frequency (represented as annualization factor) of the input returns
    @param input_is_simple_return: if the input returns series are simple or log returns
    @return: the simple (annual compounding) return
    """
    if input_is_simple_return:
        total_return = np.product(1.0 + returns)
    else:
        total_return = np.product(np.exp(returns))
    n = len(returns)
    return np.power(total_return, annualization_factor / n) - 1


def annual_vol(returns, annualization_factor, demean=True, vsc=False):
    returns_array = np.array(returns)
    n = len(returns)
    if not vsc:
        if demean:
            mean = np.mean(returns)
            return np.sqrt((sum(returns_array * returns_array) / (n - 1) - mean * mean) * annualization_factor)
        else:
            return np.sqrt(sum(returns_array * returns_array) / (n - 1) * annualization_factor)
    else:
        return np.sqrt(sum(returns_array ** 2) / n * annualization_factor)


def rolling_annual_vol(returns, annualization_factor, rolling_window, demean=True):
    returns_array = np.array(returns)
    n = len(returns)
    vols = []
    for i in range(rolling_window - 1, n):
        rolling_returns = returns_array[(i - rolling_window + 1):(i + 1)]
        if demean:
            mean = np.mean(rolling_returns)
            vols.append(np.sqrt(
                (sum(rolling_returns * rolling_returns) / (rolling_window - 1) - mean * mean) * annualization_factor))
        else:
            vols.append(np.sqrt(sum(rolling_returns * rolling_returns) / (rolling_window - 1) * annualization_factor))
    return vols


def rolling_annual_vol_from_df(df, column, annualization_factor, rolling_window, demean=True, result_as_list=True,
                               date_column_name='date'):
    returns = df[column].values
    vols = rolling_annual_vol(returns, annualization_factor=annualization_factor, rolling_window=rolling_window,
                              demean=demean)
    if result_as_list:
        return vols
    else:
        dates = df[date_column_name]
        return pd.DataFrame({'date': dates[-len(vols):], column: vols})


def annual_downside_vol(returns, annualization_factor, barrier=0):
    n = len(returns)
    downside_returns = np.array(list(map(lambda x: min(barrier, x), returns)))
    var = sum(downside_returns * downside_returns / (n - 1)) * annualization_factor
    return np.sqrt(var)


def sharpe(returns, annualization_factor, input_is_simple_return=True):
    return annual_simple_return(returns, annualization_factor, input_is_simple_return) / annual_vol(returns,
                                                                                                    annualization_factor,
                                                                                                    demean=True)


def sortino(returns, annualization_factor, input_is_simple_return=True):
    return annual_simple_return(returns, annualization_factor, input_is_simple_return) / annual_downside_vol(returns,
                                                                                                             annualization_factor)


def max_drawdown(returns, input_is_simple_return=True):
    level = 1
    max_level = 1
    max_drawdown_value = 0
    for r in returns:
        if input_is_simple_return:
            level = level * (1 + r)
        else:
            level = level * np.exp(r)
        if level > max_level:
            max_level = level
        else:
            drawdown = (max_level - level) / max_level
            if drawdown > max_drawdown_value:
                max_drawdown_value = drawdown
    return max_drawdown_value


def correlations(list_of_returns):
    np_array = np.array(list_of_returns)
    return np.corrcoef(np_array)


def beta(returns, benchmark_returns, annualization_factor, benchmark_annualization_factor=None, adjusted=False):
    vol = annual_vol(returns, annualization_factor, demean=True)
    if benchmark_annualization_factor is None:
        benchmark_annualization_factor = annualization_factor
    vol_benchmark = annual_vol(benchmark_returns, benchmark_annualization_factor, demean=True)
    correl = correlations([returns, benchmark_returns])[0][1]
    cov = vol * correl * vol_benchmark
    raw_beta = cov / vol_benchmark / vol_benchmark
    if adjusted:
        return raw_beta * 2.0 / 3.0 + 1.0 / 3.0
    else:
        return raw_beta


def conditional_beta(returns, benchmark_returns, annualization_factor, adjusted=False, condition_level=0,
                     downside=True):
    if downside:
        indexes = [i for i in range(len(benchmark_returns)) if benchmark_returns[i] < condition_level]
    else:
        indexes = [i for i in range(len(benchmark_returns)) if benchmark_returns[i] > condition_level]
    conditional_returns = [returns[i] for i in indexes]
    conditional_benchmark_returns = [benchmark_returns[i] for i in indexes]
    return beta(conditional_returns, conditional_benchmark_returns, annualization_factor, adjusted=adjusted)


def upside_beta(returns, benchmark_returns, annualization_factor, adjusted=False, condition_level=0):
    return conditional_beta(returns, benchmark_returns, annualization_factor, adjusted=adjusted,
                            condition_level=condition_level, downside=False)


def downside_beta(returns, benchmark_returns, annualization_factor, adjusted=False, condition_level=0):
    return conditional_beta(returns, benchmark_returns, annualization_factor, adjusted=adjusted,
                            condition_level=condition_level, downside=True)


def connect_price_series(dfs, connection_dates, data_column_name, additional_column_names=[], date_column_name='date'):
    """
    connect a number of series at connection dates, rescaling at connection point,
    we do not look at time element but only compare the dates
    @param dfs:
    @param connection_dates: need to be in ascending order
    @return:
    """
    new_df = pd.DataFrame(dfs[-1][[date_column_name, data_column_name] + additional_column_names])
    for i in reversed(range(len(connection_dates))):
        connection_date = connection_dates[i]
        prev_df = pd.DataFrame(dfs[i][[date_column_name, data_column_name] + additional_column_names])
        after_level = new_df[new_df[date_column_name] == connection_date][data_column_name].values
        assert len(after_level) == 1
        after_level = after_level[0]
        before_level = prev_df[prev_df[date_column_name] == connection_date][data_column_name].values
        assert len(before_level) == 1
        before_level = before_level[0]
        ratio = after_level / before_level
        prev_df[data_column_name] = prev_df[data_column_name].apply(lambda x: x * ratio)
        prev_df = prev_df.drop(prev_df[prev_df[date_column_name] == connection_date].index)
        new_df = pd.concat([prev_df, new_df])
    return new_df


def cvar(sample, percentile):
    sorted_sample = np.sort(sample)
    index = int(percentile * len(sorted_sample))
    cvar = np.mean(sorted_sample[:index])
    return cvar


def var(sample, percentile):
    sorted_sample = np.sort(sample)
    index = int(percentile * len(sorted_sample))
    var = sorted_sample[index]
    return var


def conditional_joint_normal(mu_x, mu_y, cov, barrier):
    """
    Given a multivariate normal distribution, and a subset of conditional random variable (x) and
    a subset of conditioning random variable (y), the distribution of x conditional on y = barrier
    is multivariate normal as well
    This function calculate the mean and cov of the conditional distribution
    @param mu_x:
    @param mu_y:
    @param cov:
    @param barrier:
    @return:
    """
    n_x = len(mu_x)
    n_y = len(mu_y)
    cov_x = cov[:n_x, :n_x]
    cov_y = cov[n_x:, n_x:]
    cov_y_inv = np.linalg.inv(cov_y)
    cov_xy = cov[:n_x, n_x:]
    mu_c = mu_x + cov_xy.dot(cov_y_inv).dot(barrier - mu_y)
    cov_c = cov_x - cov_xy.dot(cov_y_inv).dot(cov_xy.T)
    return mu_c, cov_c


def trancated_normal_pdf(x, mu, sigma, lb, ub):
    """
    pdf of a 1-D trancated normal distribution
    @param x:
    @param mu:
    @param sigma:
    @param barrier:
    @return:
    """
    if ub == float('inf'):
        A = 1.0
    else:
        A = norm.cdf((ub - mu) / sigma)
    if lb == -float('inf'):
        B = 0.0
    else:
        B = norm.cdf((lb - mu) / sigma)
    return 1.0 / sigma * norm.pdf((x - mu) / sigma) / (A - B)


def trancated_normal_stats(mu, sigma, lb, ub):
    alpha = (lb - mu) / sigma
    beta = (ub - mu) / sigma
    z = norm.cdf(beta) - norm.cdf(alpha)
    mean = mu + (norm.pdf(alpha) - norm.pdf(beta)) / z * sigma
    var = sigma * sigma * (1 - (beta * norm.pdf(beta) - alpha * norm.pdf(alpha)) / z - np.power(
        (norm.pdf(alpha) - norm.pdf(beta)) / z, 2))
    std = np.sqrt(var)
    return mean, std


def calibrate_conditional_multivariate_normal(returns_df, benchmark, ticker1, ticker2,
                                              barrier_std_lb, barrier_std_ub,
                                              mu_ticker1, mu_ticker2, mu_benchmark,
                                              vol_ticker1, vol_ticker2, vol_benchmark,
                                              asset_annualization_factor1, asset_annualization_factor2,
                                              benchmark_annualization_factor,
                                              calibrated_benchmark_vol=None,
                                              verbose=False):
    """
    calibrate undelrying vols, benchmark vol, correlations between underlyings and benchmark and between underlyings
    using the conditional data (condition defined by barrier_std_lb/ub)
    we take all data such that benchmark return is between the two std bounds multiplied by the unconditional vol (vol_benchmark)
    we use unconditional mus in all calibration, as conditional mu is by definition biased
    the inputs vol_ticker1, vol_ticker2, vol_benchmark are used as initial guess of the calibration
    @param returns_df: un-annualized returns
    @param benchmark:
    @param ticker1:
    @param ticker2:
    @param barrier_std_lb:
    @param barrier_std_ub:
    @param mu_ticker1:
    @param mu_ticker2:
    @param mu_benchmark:
    @param vol_ticker1:
    @param vol_ticker2:
    @param vol_benchmark:
    @param annualization_factor:
    @return:
    """

    def log_likelihood(data, mu, sigma, lb, ub):
        sum = 0
        for d in data:
            pdf = trancated_normal_pdf(d, mu, sigma, lb, ub)  # max(EPSILON, trancated_normal_pdf(d, mu, sigma, lb, ub))
            sum += np.log(pdf)
        return sum

    def func(x, data, mu, lb, ub):
        return -log_likelihood(data, mu, np.exp(x[0]), lb, ub)

    # bounds are un-annualized numbers, to be consistent with returns
    barrier_absolute_lb = mu_benchmark / benchmark_annualization_factor + barrier_std_lb * vol_benchmark / np.sqrt(
        benchmark_annualization_factor)
    barrier_absolute_ub = mu_benchmark / benchmark_annualization_factor + barrier_std_ub * vol_benchmark / np.sqrt(
        benchmark_annualization_factor)
    barrier_absolute_mid = (barrier_absolute_lb + barrier_absolute_ub) / 2.0
    conditional_returns = returns_df[
        (barrier_absolute_lb <= returns_df[benchmark]) & (returns_df[benchmark] < barrier_absolute_ub)
        ]
    n = conditional_returns.shape[0]
    if verbose:
        print(f'barrier={barrier_std_lb},{barrier_std_ub}, n={n}')

    if calibrated_benchmark_vol is None:
        data = returns_df[returns_df[benchmark] < barrier_absolute_mid]

        def _prob_func(_x):
            _prob_theoretical = norm.cdf(barrier_absolute_mid, loc=mu_benchmark / benchmark_annualization_factor,
                                         scale=_x * _x / benchmark_annualization_factor)
            _prob_data = data.shape[0] / returns_df.shape[0]
            return _prob_theoretical - _prob_data

        benchmark_res = least_squares(_prob_func, (vol_benchmark / np.sqrt(benchmark_annualization_factor),),
                                      bounds=((0.0,), (float('inf'),)))
        calibrated_vol_b = benchmark_res[0]

        # calibrate conditional vol of benchmark
        # everything is based on un-annualized numbers
        # data = conditional_returns[benchmark].values
        # sol = minimize(func, np.array([np.log(vol_benchmark / np.sqrt(annualization_factor))]), args=(data, mu_benchmark / annualization_factor, barrier_absolute_lb, barrier_absolute_ub))
        # if not sol.success:
        #     for method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
        #         if verbose:
        #             print(f"Cannot find maximum likelihood estimate of conditional benchmark vol, trying {method} method")
        #         sol = minimize(func, np.array([np.log(vol_benchmark / np.sqrt(annualization_factor))]), args=(data, mu_benchmark / annualization_factor, barrier_absolute_lb, barrier_absolute_ub), method=method)
        #         if sol.success:
        #             if verbose:
        #                 print(f"Found maximum likelihood estimate of conditional benchmark vol using {method} method")
        #             break
        # # calibrated result to output is annualized
        # calibrated_vol_b = np.exp(sol.x[0]) * np.sqrt(annualization_factor)
        # if verbose:
        #     print(f'calibrated vol_benchmark={calibrated_vol_b}')
    else:
        calibrated_vol_b = calibrated_benchmark_vol

    # calibrate conditional vols (2) and corrs (3) of underlying against the sample mean (2), vol (2) and corr (1)
    # stats from conditional data are annualized for output, but they are un-annualized again in the actual calibration
    mu_c1 = np.mean(conditional_returns[ticker1]) * asset_annualization_factor1
    mu_c2 = np.mean(conditional_returns[ticker2]) * asset_annualization_factor2
    vol_c1 = np.std(conditional_returns[ticker1]) * np.sqrt(asset_annualization_factor1)
    vol_c2 = np.std(conditional_returns[ticker2]) * np.sqrt(asset_annualization_factor2)
    cov_c1_c2 = np.mean(conditional_returns[ticker1] * conditional_returns[ticker2]) - np.mean(
        conditional_returns[ticker1]) * np.mean(conditional_returns[ticker2])
    cov_c1_c2 = cov_c1_c2 * np.sqrt(asset_annualization_factor1) * np.sqrt(asset_annualization_factor2)
    rho_c1_c2 = cov_c1_c2 / vol_c1 / vol_c2
    sample_stats = {
        'range_absolute': (barrier_absolute_lb, barrier_absolute_ub),
        ticker1: {'mu': mu_c1 / asset_annualization_factor1, 'vol': vol_c1 / np.sqrt(asset_annualization_factor1)},
        ticker2: {'mu': mu_c2 / asset_annualization_factor2, 'vol': vol_c2 / np.sqrt(asset_annualization_factor2)},
        'rho': rho_c1_c2,
    }
    target_parameters = (mu_c1 / asset_annualization_factor1, mu_c2 / asset_annualization_factor2,
                         vol_c1 / np.sqrt(asset_annualization_factor1), vol_c2 / np.sqrt(asset_annualization_factor2),
                         rho_c1_c2)
    if verbose:
        print(f'mu_c1={mu_c1}, mu_c2={mu_c2}, std_c1={vol_c1}, std_c2={vol_c2}, rho_c1_c2={rho_c1_c2}')

    def _solve_func(_x):
        _cov = np.array([
            [_x[0] * _x[0], _x[4] * _x[0] * _x[1],
             _x[2] * _x[0] * calibrated_vol_b / np.sqrt(benchmark_annualization_factor)],
            [_x[4] * _x[0] * _x[1], _x[1] * _x[1],
             _x[3] * _x[1] * calibrated_vol_b / np.sqrt(benchmark_annualization_factor)],
            [_x[2] * _x[0] * calibrated_vol_b / np.sqrt(benchmark_annualization_factor),
             _x[3] * _x[1] * calibrated_vol_b / np.sqrt(benchmark_annualization_factor),
             calibrated_vol_b * calibrated_vol_b / benchmark_annualization_factor]
        ])
        _mu_c, _cov_c = conditional_joint_normal((np.array([mu_ticker1, mu_ticker2]) / np.array(
            [asset_annualization_factor1, asset_annualization_factor2])).T,
                                                 np.array([mu_benchmark]) / benchmark_annualization_factor, _cov,
                                                 np.array([barrier_absolute_mid]))
        _mu_c1 = _mu_c[0]
        _mu_c2 = _mu_c[1]
        _vol_c1 = np.sqrt(_cov_c[0, 0])
        _vol_c2 = np.sqrt(_cov_c[1, 1])
        _rho_c1_c2 = _cov_c[0, 1] / _vol_c1 / _vol_c2

        return tuple(map(lambda i, j: i / j - 1, (_mu_c1, _mu_c2, _vol_c1, _vol_c2, _rho_c1_c2), target_parameters))
        # return tuple(map(lambda i, j: i - j, (_mu_c1, _mu_c2, _vol_c1, _vol_c2, _rho_c1_c2), target_parameters))

    # calibration is all based on un-annualized numbers
    res = least_squares(
        _solve_func,
        (vol_ticker1 / np.sqrt(asset_annualization_factor1), vol_ticker2 / np.sqrt(asset_annualization_factor2), 0, 0,
         0), bounds=((0.0, 0.0, -1.0, -1.0, -1.0), (float('inf'), float('inf'), 1.0, 1.0, 1.0)),
        max_nfev=5000,
    )

    if res.cost > 0.01 and verbose:
        print(f'cost function: {res.cost}')
        print(f'solution: {res.x}')
        print(f'target parameters: {target_parameters}')
        print(f'fitted parameters: {(1 + np.array(_solve_func(res.x))) * np.array(target_parameters)}')
        # print(f'solution: {_solve_func(res.x) + target_parameters}')

    if res.success:
        # calibrated results to output are annualized
        calib_vol_u1 = res.x[0] * np.sqrt(asset_annualization_factor1)
        calib_vol_u2 = res.x[1] * np.sqrt(asset_annualization_factor2)
        calib_rho_u1_b = res.x[2]
        calib_rho_u2_b = res.x[3]
        calib_rho_u1_u2 = res.x[4]

        return calib_vol_u1, calib_vol_u2, calibrated_vol_b, calib_rho_u1_b, calib_rho_u2_b, calib_rho_u1_u2, sample_stats
    else:
        raise RuntimeError('cannot solve for the conditional vol and rho for the underlying')
