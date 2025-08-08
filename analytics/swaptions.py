import math
import re

import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import norm
from ..analytics.utils import float_equal, find_bracket_bounds
from ..data.market import get_expiry_in_year
from ..dates.utils import datetime_to_tenor
from datetime import timedelta


def slice_interpolation(vol_slice, atmf, strike, extrapolate_strike=True):
    # vol slice strikes are relative to atmf

    vol_slice = [(float(k) + atmf, float(v)) for k, v in vol_slice.items()]
    vol_slice = sorted(vol_slice, key=lambda x: x[0])
    x = [item[0] for item in vol_slice]
    y = [item[1] for item in vol_slice]
    if strike > x[-1] or strike < x[0]:
        if extrapolate_strike:
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            return float(f(strike))
        else:
            raise RuntimeError('strike is outside range of data and extrapolate_strike flag is set to False')
    else:
        if len(x) >= 4:
            f = interp1d(x, y, kind='cubic')
        elif len(x) == 3:
            f = interp1d(x, y, kind='linear')
        else:
            raise RuntimeError('Cannot interpolate slice with less than 3 strikes')
        return float(f(strike))


def surface_interpolation(as_of_date, tenor, surface, atmf, expiry, strike, linear_in_vol=True, flat_expiry_extrapolation=True, extrapolate_strike=True):
    # surface vol strikes are relative to atmf

    (lb, ub) = find_bracket_bounds(sorted(list(surface.keys())), expiry)
    if lb == -float('inf') and ub == float('inf'):
        raise RuntimeError(f'Cannot find vol slice on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # print(f'Cannot find vol slice on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # return 1.0

    if ub == float('inf') and flat_expiry_extrapolation:
        ub = lb
    if lb == -float('inf') and flat_expiry_extrapolation:
        lb = ub

    vol_slice_lb = surface[lb]
    vol_slice_ub = surface[ub]
    atmf_lb = atmf[lb][tenor]
    atmf_ub = atmf[ub][tenor]
    vol_lb = slice_interpolation(vol_slice_lb, atmf_lb, strike, extrapolate_strike=extrapolate_strike)
    vol_ub = slice_interpolation(vol_slice_ub, atmf_ub, strike, extrapolate_strike=extrapolate_strike)

    if float_equal(lb, ub):
        return vol_lb
    if linear_in_vol:
        vol = vol_lb + (expiry - lb) * (vol_ub - vol_lb) / (ub - lb)
    else:
        vol = np.sqrt(vol_lb * vol_lb + (expiry - lb) * (vol_ub * vol_ub - vol_lb * vol_lb) / (ub - lb))
    return vol


def surface_interpolation_old(as_of_date, tenor, surface, atmf, expiry, strike, linear_in_vol=True, flat_expiry_extrapolation=True, extrapolate_strike=True):
    # surface vol strikes are relative to atmf

    (lb, ub) = find_bracket_bounds(sorted(list(surface.keys())), expiry)
    if lb == -float('inf') and ub == float('inf'):
        raise RuntimeError(f'Cannot find vol slice on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # print(f'Cannot find vol slice on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # return 1.0

    if ub == float('inf') and flat_expiry_extrapolation:
        ub = lb
    if lb == -float('inf') and flat_expiry_extrapolation:
        lb = ub

    vol_slice_lb = surface[lb]
    vol_slice_ub = surface[ub]
    atmf_lb = atmf[lb]
    atmf_ub = atmf[ub]
    vol_lb = slice_interpolation(vol_slice_lb, atmf_lb, strike, extrapolate_strike=extrapolate_strike)
    vol_ub = slice_interpolation(vol_slice_ub, atmf_ub, strike, extrapolate_strike=extrapolate_strike)

    if float_equal(lb, ub):
        return vol_lb
    if linear_in_vol:
        vol = vol_lb + (expiry - lb) * (vol_ub - vol_lb) / (ub - lb)
    else:
        vol = np.sqrt(vol_lb * vol_lb + (expiry - lb) * (vol_ub * vol_ub - vol_lb * vol_lb) / (ub - lb))
    return vol


def cube_interpolate(cubes, atmf, as_of_date, tenor, expiry, strike, linear_in_vol=True, flat_expiry_extrapolation=True, extrapolate_strike=True):
    if isinstance(expiry, datetime):
        expiry = datetime_to_tenor(expiry, as_of_date)

    # if as_of_date not in cubes:
    #     print(f'missing vol cube for {as_of_date}')
    #     return 1.0
    if as_of_date not in cubes:
        raise RuntimeError(f"{as_of_date} not in rate vol data")
    if tenor not in cubes[as_of_date]:
        raise RuntimeError(f"{tenor} not in rate vol data on {as_of_date}")
    surface = cubes[as_of_date][tenor]
    return surface_interpolation(as_of_date, tenor, surface, atmf[as_of_date], expiry, strike, linear_in_vol=linear_in_vol, flat_expiry_extrapolation=flat_expiry_extrapolation, extrapolate_strike=extrapolate_strike)


def cube_interpolate_old(cubes, atmf, as_of_date, tenor, expiry, strike, linear_in_vol=True, flat_expiry_extrapolation=True, extrapolate_strike=True):
    if isinstance(expiry, datetime):
        expiry = datetime_to_tenor(expiry, as_of_date)

    # if as_of_date not in cubes:
    #     print(f'missing vol cube for {as_of_date}')
    #     return 1.0
    surface = cubes[as_of_date][tenor]
    return surface_interpolation_old(as_of_date, tenor, surface, atmf[as_of_date][tenor], expiry, strike, linear_in_vol=linear_in_vol, flat_expiry_extrapolation=flat_expiry_extrapolation, extrapolate_strike=extrapolate_strike)


def atmf_yields_interpolate(atmf_yields, spot_rates, as_of_date, tenor, expiry,
                            flat_upper_expiry_extrapolation=True, flat_lower_expiry_extrapolation=True):
    if isinstance(expiry, datetime):
        expiry = datetime_to_tenor(expiry, as_of_date)

    # if as_of_date not in atmf_yields:
    #     print(f'missing atmf yields for {as_of_date}')
    #     return 1.0
    yields = atmf_yields[as_of_date]
    (lb, ub) = find_bracket_bounds(sorted(list(yields.keys())), expiry)
    if lb == -float('inf') and ub == float('inf'):
        raise RuntimeError(f'Cannot find the atmf yield on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # print(f'Cannot find the atmf yield on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # return 1.0

    if ub == float('inf') and flat_upper_expiry_extrapolation:
        return yields[lb][tenor]
    if lb == -float('inf'):
        tenor_elements = re.match(r'(\d{1,2})(Y|M|W|D|y|m|w|d)', tenor).groups()
        tenor_years = get_expiry_in_year(tenor_elements[0], tenor_elements[1])
        spot_rate = rate_interpolate(spot_rates, as_of_date, tenor_years,
                                     flat_upper_expiry_extrapolation=flat_upper_expiry_extrapolation,
                                     flat_lower_expiry_extrapolation=flat_lower_expiry_extrapolation)
        interpolated_yield = spot_rate + expiry * (yields[ub][tenor] - spot_rate) / ub
        return interpolated_yield

    if float_equal(lb, ub):
        return yields[lb][tenor]
    interpolated_yield = yields[lb][tenor] + (expiry - lb) * (yields[ub][tenor] - yields[lb][tenor]) / (ub - lb)
    return interpolated_yield


def atmf_yields_interpolate_old(atmf_yields, spot_rates, as_of_date, tenor, expiry,
                            flat_upper_expiry_extrapolation=True, flat_lower_expiry_extrapolation=True):
    if isinstance(expiry, datetime):
        expiry = datetime_to_tenor(expiry, as_of_date)

    # if as_of_date not in atmf_yields:
    #     print(f'missing atmf yields for {as_of_date}')
    #     return 1.0
    yields = atmf_yields[as_of_date][tenor]
    (lb, ub) = find_bracket_bounds(sorted(list(yields.keys())), expiry)
    if lb == -float('inf') and ub == float('inf'):
        raise RuntimeError(f'Cannot find the atmf yield on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # print(f'Cannot find the atmf yield on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)} and tenor {tenor}')
        # return 1.0

    if ub == float('inf') and flat_upper_expiry_extrapolation:
        return yields[lb]
    if lb == -float('inf'):
        tenor_elements = re.match(r'(\d{1,2})(Y|M|W|D|y|m|w|d)', tenor).groups()
        tenor_years = get_expiry_in_year(tenor_elements[0], tenor_elements[1])
        spot_rate = rate_interpolate(spot_rates, as_of_date, tenor_years,
                                     flat_upper_expiry_extrapolation=flat_upper_expiry_extrapolation,
                                     flat_lower_expiry_extrapolation=flat_lower_expiry_extrapolation)
        interpolated_yield = spot_rate + expiry * (yields[ub] - spot_rate) / ub
        return interpolated_yield

    if float_equal(lb, ub):
        return yields[lb]
    interpolated_yield = yields[lb] + (expiry - lb) * (yields[ub] - yields[lb]) / (ub - lb)
    return interpolated_yield


def rate_curve_interpolate(rate_curve, as_of_date, expiry, flat_upper_expiry_extrapolation=True, flat_lower_expiry_extrapolation=True):
    if isinstance(expiry, datetime):
        expiry = datetime_to_tenor(expiry, as_of_date)

    if rate_curve is None or len(rate_curve) == 0:
        print(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
        return 1.0
        # raise Exception(f"Missing rate curve on {as_of_date.strftime('%Y-%m-%d')} ")
    processed_exp = [
        datetime_to_tenor(key, as_of_date) if isinstance(key, datetime) else key
        for key in sorted(rate_curve.keys())
    ]
    (lb, ub) = find_bracket_bounds(processed_exp, expiry)
    if lb == -float('inf') and flat_lower_expiry_extrapolation:
        if ub in rate_curve:
            return rate_curve[ub]
        else:
            print(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
            return 1.0
    elif lb == -float('inf'):
        # raise RuntimeError(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
        print(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
        return 1.0

    if ub == float('inf') and flat_upper_expiry_extrapolation:
        if lb in rate_curve:
            return rate_curve[lb]
        else:
            print(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
            return 1.0
    elif ub == float('inf'):
        # raise RuntimeError(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
        print(f'Cannot find the rate on {as_of_date.strftime("%Y-%m-%d")} for expiry {str(expiry)}')
        return 1.0
    if float_equal(lb, ub):
        return rate_curve[lb]
    interpolated_rate = rate_curve[lb] + (expiry - lb) * (rate_curve[ub] - rate_curve[lb]) / (ub - lb)
    return interpolated_rate


def rate_interpolate(rates, as_of_date, expiry, flat_upper_expiry_extrapolation=True,
                     flat_lower_expiry_extrapolation=True, backfill_rate=False):
    if as_of_date not in rates or rates[as_of_date] is None:
        print(f'missing rate data for {as_of_date}, backfiling now')
        backfill_rate = True
    if backfill_rate:
        counter = 0

        if not any(isinstance(i, dict) for i in rates.values()):
            rates_dict = {as_of_date: rates}
            rates = rates_dict

        while counter < 5:
            if as_of_date in rates:
                counter = 6
            else:
                as_of_date = as_of_date - timedelta( days = 1 )
                counter += 1
        rate_curve = rates[as_of_date]
    else:
        rate_curve = rates[as_of_date]

    return rate_curve_interpolate(rate_curve, as_of_date, expiry, flat_upper_expiry_extrapolation,
                                  flat_lower_expiry_extrapolation)


def df_interpolate(rates, as_of_date, expiry, flat_upper_expiry_extrapolation=True, flat_lower_expiry_extrapolation=True, backfill_rate=False,
                   compounding_convention='annual', rate_bump=0.0):
    rate = rate_bump + rate_interpolate(rates, as_of_date, expiry,
                            flat_upper_expiry_extrapolation=flat_upper_expiry_extrapolation,
                            flat_lower_expiry_extrapolation=flat_lower_expiry_extrapolation,
                            backfill_rate = backfill_rate )
    if compounding_convention == 'annual':
        if isinstance(expiry, datetime):
            t = datetime_to_tenor(expiry, as_of_date)
        else:
            t = expiry
        return 1.0 / math.pow(1.0 + rate / 100.0, t)
    else:
        raise RuntimeError('Unknown compounding convention ' + compounding_convention)


def swaptioncalc(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, rps):
    """returns results for 1 dollar notional"""
    def get_optvalbps(F, K, t, vol, CP):  # CP 1 for call, -1 for PUT, input yields for F and K
        F = F * 100
        K = K * 100
        return CP * (F - K) * norm.cdf(-1 * CP * (K - F) / (math.sqrt(t) * vol)) + (math.sqrt(t) * vol) / math.sqrt(
            2 * math.pi) * math.exp(-0.5 * ((F - K) / (math.sqrt(t) * vol)) ** 2)

    if rps == 'r':
        cp = -1  # receiver put on yield
    elif rps == 'p':
        cp = 1  # payer, call on yield
    elif rps=='s':
        cp=0

    if cp==0:
        rtrresult = swaptioncalc(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, 'r')
        rtpresult = swaptioncalc(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, 'p')
        total = rtrresult
        total['pvbps']+=rtpresult['pvbps']
        total['fvbps']+=rtpresult['fvbps']
        total['pv'] += rtpresult['pv']
        total['fv']+=rtpresult['fv']
        total['deltapct']+=rtpresult['deltapct']
        total['delta']+=rtpresult['delta']
        total['gammapct']+=rtpresult['gammapct']
        total['gamma']+=rtpresult['gamma']
        total['vega']+=rtpresult['vega']
        total['theta']+=rtpresult['theta']
        return total

    iTenor = int(tenor[0:(len(tenor) - 1)])
    convPayPeriod = 4  # payments per year
    t = (exp - asofdate).days / 365.25 + 0.0001

    vol = cube_interpolate(vol_cube, fwd_rate_curve, asofdate, tenor, exp, strike, linear_in_vol=True) * 100
    atmf = atmf_yields_interpolate(fwd_rate_curve, spot_rate_curve, asofdate, tenor, exp)

    def base_pricer(bump, vol_bump=0):
        usedatmf = atmf + bump
        usedvol = vol + vol_bump
        result_dict = {}
        result_dict['df'] = df_interpolate(df_curve, asofdate, exp, compounding_convention='annual', rate_bump=bump)
        result_dict['fvbps'] = get_optvalbps(usedatmf, strike, t, usedvol, cp)  # bps running in yield, not bps of notional
        result_dict['pvbps'] = result_dict['fvbps'] * result_dict['df']
        result_dict['numeraireyr'] = (
                    (1 - (1 / ((1 + (usedatmf / 100. / convPayPeriod)) ** ((iTenor) * convPayPeriod)))) / (
                        usedatmf / 100.))  # fv
        result_dict['numerairedollars'] = result_dict['numeraireyr']/ 10000  # fv
        result_dict['fv'] = result_dict['fvbps'] * result_dict['numeraireyr'] / 10000
        result_dict['pv'] = result_dict['fv'] * result_dict['df']
        return result_dict

    mid_result = base_pricer(0.0)
    up_result = base_pricer(0.01)
    down_result = base_pricer(-0.01)

    # combine delta gamma
    deltabps = (up_result['fvbps'] - down_result['fvbps']) / 2
    deltadollars = (up_result['pv'] - down_result['pv']) / 2
    gammabps = up_result['fvbps'] + down_result['fvbps'] - 2 * mid_result['fvbps']
    gammadollars = up_result['pv'] + down_result['pv'] - 2 * mid_result['pv']

    # vega calc
    # vol_up_result = base_pricer(0, 1)
    # vol_down_result = base_pricer(0, -1)
    pvbpsupvol = get_optvalbps(atmf, strike, t, vol + 1, cp) * mid_result['df']
    pvbpsdownvol = get_optvalbps(atmf, strike, t, vol - 1, cp) * mid_result['df']
    vega = (pvbpsupvol - pvbpsdownvol) / 2 * mid_result['numerairedollars']

    # theta calc
    if t < 1 / 365.25:
        theta = 0
    else:
        tnext = ((exp - asofdate).days - 1) / 365.25 + 0.0001
        # doesnt take into account df change
        pvbpsnextday = get_optvalbps(atmf, strike, tnext, vol, cp) * mid_result['df']
        theta = (pvbpsnextday - mid_result['pvbps']) * mid_result['numerairedollars']

    # delta makes simplifying assumptions of flat curve and matching schedule
    # https://quant.stackexchange.com/questions/49582/interest-rate-swap-pv01-vs-dv01
    results = mid_result;
    results['fwd'] = atmf
    results['vol'] = vol
    results['tte'] = t
    results['deltapct'] = deltabps
    results['delta'] = deltadollars
    results['gammapct'] = gammabps
    results['gamma'] = gammadollars
    results['vega'] = vega
    results['theta'] = theta
    return results


def swaptioncalc_old(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, rps):
    """returns results for 1 dollar notional"""
    def get_optvalbps(F, K, t, vol, CP):  # CP 1 for call, -1 for PUT, input yields for F and K
        F = F * 100
        K = K * 100
        return CP * (F - K) * norm.cdf(-1 * CP * (K - F) / (math.sqrt(t) * vol)) + (math.sqrt(t) * vol) / math.sqrt(
            2 * math.pi) * math.exp(-0.5 * ((F - K) / (math.sqrt(t) * vol)) ** 2)

    if rps == 'r':
        cp = -1  # receiver put on yield
    elif rps == 'p':
        cp = 1  # payer, call on yield
    elif rps=='s':
        cp=0

    if cp==0:
        rtrresult = swaptioncalc_old(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, 'r')
        rtpresult = swaptioncalc_old(asofdate, vol_cube, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, 'p')
        total = rtrresult
        total['pvbps']+=rtpresult['pvbps']
        total['fvbps']+=rtpresult['fvbps']
        total['pv'] += rtpresult['pv']
        total['fv']+=rtpresult['fv']
        total['deltapct']+=rtpresult['deltapct']
        total['delta']+=rtpresult['delta']
        total['gammapct']+=rtpresult['gammapct']
        total['gamma']+=rtpresult['gamma']
        total['vega']+=rtpresult['vega']
        total['theta']+=rtpresult['theta']
        return total

    iTenor = int(tenor[0:(len(tenor) - 1)])
    convPayPeriod = 4  # payments per year
    t = (exp - asofdate).days / 365.25 + 0.0001

    vol = cube_interpolate_old(vol_cube, fwd_rate_curve, asofdate, tenor, exp, strike, linear_in_vol=True) * 100
    atmf = atmf_yields_interpolate_old(fwd_rate_curve, spot_rate_curve, asofdate, tenor, exp)

    def base_pricer(bump):
        usedatmf = atmf + bump
        result_dict = {}
        result_dict['df'] = df_interpolate(df_curve, asofdate, exp, compounding_convention='annual', rate_bump=bump)
        result_dict['fvbps'] = get_optvalbps(usedatmf, strike, t, vol, cp)  # bps running in yield, not bps of notional
        result_dict['pvbps'] = result_dict['fvbps'] * result_dict['df']
        result_dict['numeraireyr'] = (
                    (1 - (1 / ((1 + (usedatmf / 100. / convPayPeriod)) ** ((iTenor) * convPayPeriod)))) / (
                        usedatmf / 100.))  # fv
        result_dict['numerairedollars'] = result_dict['numeraireyr']/ 10000  # fv
        result_dict['fv'] = result_dict['fvbps'] * result_dict['numeraireyr'] / 10000
        result_dict['pv'] = result_dict['fv'] * result_dict['df']
        return result_dict

    mid_result = base_pricer(0.0)
    up_result = base_pricer(0.01)
    down_result = base_pricer(-0.01)

    # combine delta gamma
    deltabps = (up_result['fvbps'] - down_result['fvbps']) / 2
    deltadollars = (up_result['pv'] - down_result['pv']) / 2
    gammabps = up_result['fvbps'] + down_result['fvbps'] - 2 * mid_result['fvbps']
    gammadollars = up_result['pv'] + down_result['pv'] - 2 * mid_result['pv']

    # vega calc
    pvbpsupvol = get_optvalbps(atmf, strike, t, vol + 1, cp) * mid_result['df']
    pvbpsdownvol = get_optvalbps(atmf, strike, t, vol - 1, cp) * mid_result['df']
    vega = (pvbpsupvol - pvbpsdownvol) / 2 * mid_result['numerairedollars']

    # theta calc
    if t < 1 / 365.25:
        theta = 0
    else:
        tnext = ((exp - asofdate).days - 1) / 365.25 + 0.0001
        # doesnt take into account df change
        pvbpsnextday = get_optvalbps(atmf, strike, tnext, vol, cp) * mid_result['df']
        theta = (pvbpsnextday - mid_result['pvbps']) * mid_result['numerairedollars']

    # delta makes simplifying assumptions of flat curve and matching schedule
    # https://quant.stackexchange.com/questions/49582/interest-rate-swap-pv01-vs-dv01
    results = mid_result;
    results['fwd'] = atmf
    results['vol'] = vol
    results['tte'] = t
    results['deltapct'] = deltabps
    results['delta'] = deltadollars
    results['gammapct'] = gammabps
    results['gamma'] = gammadollars
    results['vega'] = vega
    results['theta'] = theta
    return results


def fwdstartswapcalc(asofdate, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, rps, atmf_override=None):
    if rps == 'r':
        cp = -1  # receiver put on yield
    elif rps == 'p':
        cp = 1  # payer, call on yield

    iTenor = int(tenor[0:(len(tenor) - 1)])
    convPayPeriod = 4  # payments per year
    t = (exp - asofdate).days / 365.25 + 0.0001

    if atmf_override is not None:
        atmf = atmf_override
    else:
        atmf = atmf_yields_interpolate(fwd_rate_curve, spot_rate_curve, asofdate, tenor, exp)

    def base_pricer(bump):
        usedatmf = atmf + bump
        result_dict = {}
        result_dict['df'] = df_interpolate(df_curve, asofdate, exp, compounding_convention='annual', rate_bump=bump)
        result_dict['fvbps'] = cp * (usedatmf - strike) * 100 # bps running in yield, not bps of notional
        result_dict['pvbps'] = result_dict['fvbps'] * result_dict['df']
        result_dict['numeraireyr'] = (
                    (1 - (1 / ((1 + (usedatmf / 100. / convPayPeriod)) ** ((iTenor) * convPayPeriod)))) / (
                        usedatmf / 100.))  # fv
        result_dict['numerairedollars'] = result_dict['numeraireyr'] / 10000  # fv
        result_dict['fv'] = result_dict['fvbps'] * result_dict['numeraireyr'] / 10000
        result_dict['pv'] = result_dict['fv'] * result_dict['df']
        return result_dict

    mid_result = base_pricer(0.0)
    up_result = base_pricer(0.01)
    down_result = base_pricer(-0.01)

    # combine delta gamma
    deltabps = (up_result['fvbps'] - down_result['fvbps']) / 2
    deltadollars = (up_result['pv'] - down_result['pv']) / 2
    gammabps = up_result['fvbps'] + down_result['fvbps'] - 2 * mid_result['fvbps']
    gammadollars = up_result['pv'] + down_result['pv'] - 2 * mid_result['pv']

    # delta makes simplifying assumptions of flat curve and matching schedule
    # https://quant.stackexchange.com/questions/49582/interest-rate-swap-pv01-vs-dv01
    results = mid_result;
    results['fwd'] = atmf
    results['tte'] = t
    results['deltapct'] = deltabps
    results['delta'] = deltadollars
    results['gammapct'] = gammabps
    results['gamma'] = gammadollars
    results['theta'] = 0
    return results


def fwdstartswapcalc_old(asofdate, fwd_rate_curve, spot_rate_curve, df_curve, ccy, exp, tenor, strike, rps, atmf_override=None):
    if rps == 'r':
        cp = -1  # receiver put on yield
    elif rps == 'p':
        cp = 1  # payer, call on yield

    iTenor = int(tenor[0:(len(tenor) - 1)])
    convPayPeriod = 4  # payments per year
    t = (exp - asofdate).days / 365.25 + 0.0001

    if atmf_override is not None:
        atmf = atmf_override
    else:
        atmf = atmf_yields_interpolate_old(fwd_rate_curve, spot_rate_curve, asofdate, tenor, exp)

    def base_pricer(bump):
        usedatmf = atmf + bump
        result_dict = {}
        result_dict['df'] = df_interpolate(df_curve, asofdate, exp, compounding_convention='annual', rate_bump=bump)
        result_dict['fvbps'] = cp * (usedatmf - strike) * 100 # bps running in yield, not bps of notional
        result_dict['pvbps'] = result_dict['fvbps'] * result_dict['df']
        result_dict['numeraireyr'] = (
                    (1 - (1 / ((1 + (usedatmf / 100. / convPayPeriod)) ** ((iTenor) * convPayPeriod)))) / (
                        usedatmf / 100.))  # fv
        result_dict['numerairedollars'] = result_dict['numeraireyr'] / 10000  # fv
        result_dict['fv'] = result_dict['fvbps'] * result_dict['numeraireyr'] / 10000
        result_dict['pv'] = result_dict['fv'] * result_dict['df']
        return result_dict

    mid_result = base_pricer(0.0)
    up_result = base_pricer(0.01)
    down_result = base_pricer(-0.01)

    # combine delta gamma
    deltabps = (up_result['fvbps'] - down_result['fvbps']) / 2
    deltadollars = (up_result['pv'] - down_result['pv']) / 2
    gammabps = up_result['fvbps'] + down_result['fvbps'] - 2 * mid_result['fvbps']
    gammadollars = up_result['pv'] + down_result['pv'] - 2 * mid_result['pv']

    # delta makes simplifying assumptions of flat curve and matching schedule
    # https://quant.stackexchange.com/questions/49582/interest-rate-swap-pv01-vs-dv01
    results = mid_result;
    results['fwd'] = atmf
    results['tte'] = t
    results['deltapct'] = deltabps
    results['delta'] = deltadollars
    results['gammapct'] = gammabps
    results['gamma'] = gammadollars
    return results
