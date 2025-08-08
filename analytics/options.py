import requests
import json
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
from datetime import datetime
from ..data.utils import chunks
from ..dates.utils import vola_datetime_to_datetime, datetime_to_vola_datetime, tenor_to_datetime, \
    datetime_diff
from ..infrastructure.volatility_surface import VolatilitySurface
from ..constants.underlying_type import UnderlyingType
from scipy.optimize import fsolve
from scipy.stats import norm
import numpy as np


def root_from_ticker(ticker: str):
    return ticker.split(" ")[0]


def get_vol_with_vola(vola_surface, expiration, strike):
    as_of_date = vola_surface.asOfTime
    # if isinstance(expiration, numbers.Number):
    #     expiry = vola.DateTime(as_of_date.nanos + expiration * vola.DateTime.nanosPerYear)
    if isinstance(expiration, datetime):
        expiry = datetime_to_vola_datetime(expiration)
    else:
        expiry = datetime_to_vola_datetime(tenor_to_datetime(expiration, vola_datetime_to_datetime(as_of_date)))
    vol = vola_surface.volAtT(expiry, strike, vola.StrikeType.K)
    return vol


def get_smile_with_vola(vola_surface: vola.VolSurface if ENABLE_PYVOLAR else None, tenor: str, moneyness_strikes: list):
    smile = []
    for moneyness_strike in moneyness_strikes:
        strike = moneyness_strike * vola_surface.spot
        vol = get_vol_with_vola(vola_surface=vola_surface, expiration=tenor, strike=strike)
        smile.append(vol)
    return smile


def map_to_vola_calc_type(calc_type):
    if calc_type == 'price' or calc_type == 'bid' or calc_type == 'ask':
        return vola.PricerResultsType.VALUE
    elif calc_type == 'delta':
        return vola.PricerResultsType.DELTA
    elif calc_type == 'gamma':
        return vola.PricerResultsType.GAMMA
    elif calc_type == 'vega':
        return vola.PricerResultsType.VEGA
    elif calc_type == 'theta':
        return vola.PricerResultsType.THETA
    elif calc_type == 'vanna':
        return vola.PricerResultsType.VANNA
    elif calc_type == 'volga':
        return vola.PricerResultsType.VOLGA
    elif calc_type == 'rho':
        return vola.PricerResultsType.RHO
    else:
        raise RuntimeError('Unknown calc type ' + calc_type)


def price_option_with_vola(pricer, vola_surface, expiration, strike, is_call, is_american, vol_override=None,
                           calc_types='price'):
    if not isinstance(calc_types, list):
        calc_types_list = [calc_types]
    else:
        calc_types_list = calc_types
    vola_calc_types_list = list(map(lambda x: map_to_vola_calc_type(x), calc_types_list))

    vol_surface = VolatilitySurface.create_from_vola_surface(vola_surface, "DummyUnderlying")
    as_of_date = vol_surface.get_base_datetime()
    # if isinstance(expiration, numbers.Number):
    #     expiry = vola.DateTime(as_of_date.nanos + expiration * vola.DateTime.nanosPerYear)
    if isinstance(expiration, datetime):
        expiry = datetime_to_vola_datetime(expiration)
    else:
        expiry = datetime_to_vola_datetime(tenor_to_datetime(expiration, vola_datetime_to_datetime(as_of_date)))

    if vol_override is None:
        vol = vola_surface.get_vol(expiry, strike, vola.StrikeType.K)
    else:
        vol = vol_override

    if vol_surface.get_underlying_type() == UnderlyingType.EQUITY:
        r = vol_surface.get_discount_rate(expiry)
        q = vol_surface.get_borrow_rate(expiry)
        spot = vol_surface.get_spot(as_of_date)
        values = pricer.price(as_of_date, expiry, strike, is_call, is_american, spot, vol, r, q, vola_calc_types_list)
    elif isinstance(vola_surface, vola.VolSurfaceFutures):
        r = vol_surface.get_discount_rate(expiry)
        spot = vol_surface.get_future_price(as_of_date, expiry)
        values = pricer.price(as_of_date, expiry, strike, is_call, is_american, spot, vol, r, vola_calc_types_list)
    else:
        raise RuntimeError('Unknown vola surface type')

    if isinstance(calc_types, list):
        return list(values)
    else:
        return values[0]


def price_option_with_API(ids, ref_time, chunk_size=200):
    server_name = 'volar_surface_server'
    env_name = 'cpiceregistry'
    result_ids = []
    result_prices = []
    for chunk in chunks(ids, chunk_size):
        payload = {
            'server_name': server_name,
            'ids': chunk,
            'refTime': ref_time,
            'format': 'json',
            'column_names': 'ID,RefTime,VALUE',
        }
        result = json.loads(requests.get('http://' + env_name + ':6703/price', params=payload).content.decode("utf-8"))[0]
        result_ids = result_ids + result['ids']
        result_prices = result_prices + list(map(lambda x: x[0], result['values']))
    return dict(zip(result_ids, result_prices))


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


def Black76(strike, expiration, is_call, dt, fwd, vol, disc, TTM=None):

    if TTM is None:
        TTM = datetime_diff(expiration, dt).days / 365
    assert TTM >= 0

    if TTM == 0:
        delta, gamma, vega, theta, vanna, volga = 0, 0, 0, 0, 0, 0
        if is_call:
            price = max(fwd - strike, 0)
        else:
            price = max(strike - fwd, 0)
    else:
        d1 = np.log(fwd / strike) + vol * vol * TTM / 2
        d1 /= (vol * np.sqrt(TTM))
        d2 = d1 - vol * np.sqrt(TTM)
        N_d1, N_d2 = norm.cdf( [d1, d2])
        n_d1, n_d2 = norm.pdf([d1, d2])
        N_md1 = 1.0 - N_d1
        N_md2 = 1.0 - N_d2
        if is_call:
            price = disc * (fwd * N_d1 - strike * N_d2)
            delta = disc * N_d1
        else:
            price = disc * (strike * N_md2 - fwd * N_md1)
            delta = -disc * N_md1
        gamma = disc * n_d1 / (fwd * vol * np.sqrt(TTM))
        vega = disc * n_d1 * fwd * np.sqrt(TTM) / 100
        vanna = -disc * n_d1 * d2 / vol
        volga = disc * fwd * n_d1 * np.sqrt(TTM) * d1 * d2 / vol

    results = {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga,
        'fwd': fwd,
        'vol': vol,
    }

    return results


def get_implied_vol(strike, expiration, is_call, dt, fwd, price, disc, initial_guess=0):
    def func(vol):
        res = Black76(strike, expiration, is_call, dt, fwd, vol, disc)
        if isinstance(res["price"], list):
            err = (res["price"][0] - price)**2
        else:
            err = (res["price"] - price)**2
        return err
    sol = fsolve(func, initial_guess)
    return sol[0]


def get_strike_from_delta(delta, fwd, vol, ttm, is_call, prem_adj=False):
    phi = 1 if is_call else -1
    if prem_adj:
        def func(x):
            d1 = np.log(fwd / x) + vol * vol * ttm / 2
            d1 /= (vol * np.sqrt(ttm))
            d2 = d1 - vol * np.sqrt(ttm)
            res = x / fwd * phi * norm.cdf(phi * d2)
            return abs(res - delta)

        sol = fsolve(lambda x: func(x[0]), fwd)
        strike = sol[0]
    else:
        exponent = -phi * norm.ppf(phi * delta) * vol * np.sqrt(ttm) + 0.5 * vol**2 * ttm
        strike = fwd * np.exp(exponent)
    return strike
