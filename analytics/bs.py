from ..dates.utils import datetime_diff
from scipy.stats import norm
import numpy as np


def BlackScholes(strike, expiration, is_call, is_american, dt, spot, fwd, vol, disc=None, TTM=None):
    if is_american:
        RuntimeError( 'BSPricer does not handle American exercise' )

    if TTM is None:
        TTM = datetime_diff(expiration, dt).days / 365
    assert TTM >= 0

    if TTM == 0:
        delta, gamma, vega, theta, vanna, volga, forward_delta, theta, rho = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        if is_call:
            price = np.maximum(spot - strike, 0)
        else:
            price = np.maximum(strike - spot, 0)
    else:
        d1 = np.log( fwd / strike) + vol * vol * TTM / 2
        d1 /= ( vol * np.sqrt(TTM) )
        d2 = d1 - vol * np.sqrt( TTM )
        N_d1, N_d2 = norm.cdf( [ d1, d2 ] )
        n_d1, n_d2 = norm.pdf([d1, d2])
        N_md1 = 1.0 - N_d1
        N_md2 = 1.0 - N_d2
        if disc is None:
            disc = spot / fwd
        discounted_fwd_factor = fwd / spot * disc
        if is_call:
            price = disc * ( fwd * N_d1 - strike * N_d2 )
            delta = discounted_fwd_factor * N_d1
            forward_delta = disc * N_d1
            theta = - spot * n_d1 * vol / np.sqrt(TTM) / 2 + np.log(disc) / TTM * strike * disc * N_d2
            rho = strike * TTM * disc * N_d2
        else:
            price = disc * ( strike * N_md2 - fwd * N_md1 )
            delta = -discounted_fwd_factor * N_md1
            forward_delta = -disc * N_md1
            theta = - spot * n_d1 * vol / np.sqrt(TTM) / 2 - np.log(disc) / TTM * strike * disc * N_md2
            rho = -strike * TTM * disc * N_md2
        gamma = discounted_fwd_factor * n_d1 / (spot * vol * np.sqrt(TTM))
        vega = discounted_fwd_factor * n_d1 * spot * np.sqrt(TTM) / 100
        vanna = -discounted_fwd_factor * n_d1 * d2 / vol
        volga = discounted_fwd_factor * spot * n_d1 * np.sqrt(TTM) * d1 * d2 / vol

    results = {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga,
        'rho': rho,
        'theta': theta,
        'forward_delta': forward_delta,
        'fwd': fwd,
        'vol': vol,
        'spot': spot,
    }

    return results