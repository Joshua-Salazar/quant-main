from scipy.stats import norm
from scipy.optimize import minimize
from ..data.Datalake3 import Datalake
from ..dates.utils import add_tenor
from datetime import timedelta, date

import numpy as np
import pandas as pd
import pickle


def delta_strike_to_relative(disc, fwd_delta_strike, vol, TTM, is_call):
    # return relative fwd strike
    if is_call:
        d1 = norm.ppf(fwd_delta_strike / disc)
    else:
        assert fwd_delta_strike < 0
        d1 = -norm.ppf(-fwd_delta_strike / disc)
    numerat = d1 * vol * np.sqrt(TTM)
    lnF_K = numerat - (vol * vol * TTM) / 2
    return np.exp(lnF_K)


def calib_SABR_params(kLo, volLo, kMid, volMid, kHi, volHi, fwd, beta):
    # kLo, kMid, kHi are relative-Fwd strikes
    # see https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2467231 p8-9
    wLo = 1 / ((kLo - kMid) * (kLo - kHi))
    wMid = 1 / ((kMid - kLo) * (kMid - kHi))
    wHi = 1 / ((kHi - kLo) * (kHi - kMid))
    volATM = (kMid * kHi * wLo * volLo) + (kLo * kHi * wMid * volMid) + (kLo * kMid * wHi * volHi)
    slope = -((kMid + kHi) * wLo * volLo) - ((kLo + kHi) * wMid * volMid) - ((kLo + kMid) * wMid * volMid)
    curve = 2 * ((wLo * volLo) + (wMid * volMid) + (wHi * volHi))

    # see ref p7 eqn (32)
    alpha0 = volATM * np.power(fwd, 1 - beta)
    nu_sq0 = 3 * volATM * curve
    nu_sq0 -= (1 / 2) * ((1 - beta) * volATM) ** 2
    nu_sq0 += (3 / 2) * (2 * slope + (1 - beta) * volATM) ** 2
    if nu_sq0 <= 0:
        rho0 = np.sign()
        nu0 = (2 * slope + (1 - beta) * volATM) / rho0
    else:
        nu0 = np.sqrt(nu_sq0)
        rho0 = (2 * slope + (1 - beta) * volATM) / nu0

    return {'rho': rho0, 'alpha': alpha0, 'nu': nu0}


def SABR_vol(logK, fwd, alpha, beta, rho, nu):
    fwd_beta = np.power(fwd, (beta - 1))
    C = alpha * fwd_beta
    B = (rho * nu - alpha * (1 - beta) * fwd_beta) / 2
    A = (1 - beta) ** 2 * ((alpha * fwd_beta) ** 2 + nu ** 2 * (2 - 3 * rho ** 2))
    A /= (12 * alpha * fwd_beta)
    return A * logK * logK + B * logK + C


def SABR_vol_CTP(fwd, TTM, K, alpha, beta, rho, nu):
    # implementation of https://gitea.capstoneco.com/ctp/ctp/src/branch/master/ctp/models/vol/SABRFormula.cpp
    assert beta > 0
    if alpha == 0:
        return 0
    else:
        # handle beta = 0, 1 special cases
        if beta < 1e-7:
            return SABR_zeroBeta_CTP(fwd, K, TTM, alpha, rho, nu)
        elif 1 - beta < 1e-7:
            return SABR_oneBeta_CTP(fwd, K, TTM, alpha, rho, nu)
        else:
            if abs(fwd / K - 1) < 1e-7:
                return SABR_atmVol_CTP(fwd, TTM, alpha, beta, rho, nu)
            else:
                ln = np.log(fwd / K)
                beta1 = 1 - beta
                f1 = np.power(fwd * K, beta1)
                f1Sqrt = np.sqrt(f1)
                lnBetaSq = np.power(beta1 * ln, 2)
                z = nu * f1Sqrt * ln / alpha
                first = alpha / (f1Sqrt * (1 + lnBetaSq / 24 + lnBetaSq * lnBetaSq / 1920))
                second = zOverChi(rho, z)
                third = 1 + TTM * (
                            beta1 * beta1 * alpha * alpha / 24 / f1 + rho * nu * beta * alpha / 4 / f1Sqrt + nu * nu * (
                                2 - 3 * rho * rho) / 24)
                return first * second * third


def SABR_zeroBeta_CTP(fwd, K, TTM, alpha, rho, nu):
    ln = np.log(fwd / K)
    z = nu * np.sqrt(fwd * K) * ln / alpha
    zX = zOverChi(rho, z)
    return alpha * ln * zX * (1 + TTM * (alpha * alpha / fwd / K + nu * nu * (2 - 3 * rho * rho)) / 24) / (fwd - K)


def SABR_oneBeta_CTP(fwd, K, TTM, alpha, rho, nu):
    ln = np.log(fwd / K)
    z = nu * ln / alpha
    zX = zOverChi(rho, z)
    return alpha * zX * (1 + TTM * (rho * alpha * nu / 4 + nu * nu * (2 - 3 * rho * rho) / 24))


def SABR_atmVol_CTP(fwd, TTM, alpha, beta, rho, nu):
    beta1 = 1 - beta
    f1 = np.power(fwd, beta1)
    return alpha * (1. + TTM * (beta1 * beta1 * alpha * alpha / 24 / f1 / f1 + rho * alpha * beta * nu / 4 / f1 +
                                nu * nu * (2. - 3. * rho * rho) / 24)) / f1


def zOverChi(rho, z):
    if abs(z) < 1e-6:
        return 1 - rho * z / 2
    else:
        rhoStar = 1 - rho
        if abs(rhoStar) < 1e-8:
            if z > 1:
                if rhoStar == 0.0:
                    return 0.0
                else:
                    return z / (np.log(2 * (z - 1)) - np.log(rhoStar))
            elif z < 1:
                return -z / np.log(1 - z)
            else:
                return 0.0
        else:
            rhoHat = 1 + rho
            if abs(rhoHat) < 1e-8:
                if z > -1:
                    return z / np.log(1 + z)
                elif z < -1:
                    if rhoHat == 0:
                        return 0.0
                    else:
                        chi = np.log(rhoHat) - np.log(-(1 + z) / rhoStar)
                        return z / chi
                else:
                    return 0.0
            else:
                if z < -1e6:
                    arg = (rho * rho - 1) / 2 / z
                elif z > 1e8:
                    arg = 2 * (z - rho)
                else:
                    arg = np.sqrt(1 - 2 * rho * z + z * z) + z - rho
                    if arg <= 0:
                        return 0.0
                chi = np.log(arg) - np.log(rhoStar)
                return z / chi


def SSVI_total_var(nu, gamma, rho, volATM, TTM, logK):
    # eqn (4.1) in https://arxiv.org/pdf/1204.0646.pdf
    ATM_total_var = volATM * volATM * TTM
    # assume power-law function
    phi_ATM = nu / np.power(ATM_total_var, gamma)
    return (ATM_total_var / 2) * (1 + rho * phi_ATM * logK + np.sqrt((phi_ATM * logK + rho) ** 2 + (1 - rho ** 2)))


def fit_SSVI_slice(vols, logKs, TTM, volATM):
    # minimise meansq error of quoted vols and SSVI vols
    vols = np.array(vols)
    MSE = lambda x: np.mean(
        (vols - np.array([np.sqrt(SSVI_total_var(x[0], x[1], x[2], volATM, TTM, K) / TTM) for K in logKs])) ** 2)
    bnds = ((0, 3), (0, 1 / 2), (-0.75, 0.75))
    init_guess = (0.2, 0.4, -0.4)
    res = minimize(MSE, init_guess, method='Powell', tol=1e-12, bounds=bnds)
    return res.x


def interp_vol_SSVI(params1, TTM1, volATM1, params2, TTM2, volATM2, logK, TTM):
    # linearly interpolate between SSVI fits at TTM1 <= TTM <= TTM2 to give vol at log-strike logK with TTM tenor.
    nu1, gamma1, rho1 = params1
    vol1 = np.sqrt(SSVI_total_var(nu1, gamma1, rho1, volATM1, TTM1, logK) / TTM1)
    assert vol1 > 0
    assert vol1 < 10 * volATM1
    nu2, gamma2, rho2 = params2
    vol2 = np.sqrt(SSVI_total_var(nu2, gamma2, rho2, volATM2, TTM2, logK) / TTM2)
    assert vol2 > 0
    assert vol2 < 10 * volATM2
    # handle exact expiries
    if TTM1 == TTM2:
        return vol1
    else:
        # use sqrt scaling for vol interp
        return vol1 + (np.sqrt(TTM) - np.sqrt(TTM1)) * (vol1 - vol2) / (np.sqrt(TTM1) - np.sqrt(TTM2))


def pull_data(dt, und, target_tenor, is_call):
    # for a given underlier <und> pull fwd and implied vol data for inf and sup tenors of <target_tenor>

    # find closest quoted expiries
    target_exp = add_tenor(dt, target_tenor)
    fixed_tenors = ['0W', '1W', '2W', '1M', '2M', '3M', '6M', '9M']
    fixed_exp = np.array([add_tenor(dt, ten) for ten in fixed_tenors])
    lower_exp = max(fixed_exp[fixed_exp <= target_exp])
    lower_ten = fixed_tenors[np.where(fixed_exp == lower_exp)[0][0]]
    if lower_ten == '0W':
        lower_ten = 'ON'
    upper_exp = min(fixed_exp[fixed_exp >= target_exp])
    upper_ten = fixed_tenors[np.where(fixed_exp == upper_exp)[0][0]]

    # generate tickers for vol and fwd
    if is_call:
        opts = ['ATM', 'C35', 'C25', 'C10']
    else:
        opts = ['ATM', 'P35', 'P25', 'P10']
    tickers = []
    for tenor in [lower_ten, upper_ten]:
        for opt in opts:
            if opt == 'ATM':
                tickers.append('FX.IMPLIED_VOL.USD.%s.%s.%s.CITI' % (und, opt, tenor))
            else:
                tickers.append('FX.IMPLIED_VOL.USD.%s.STRIKE_%s.%s.CITI' % (und, opt, tenor))
    for tenor in [lower_ten, upper_ten]:
        tickers.append('FX.FORWARD.FWD_OUTRIGHT.USD.%s.%s.CITI' % (und, tenor))
    # handle AUD, EUR, GBP quoting convention
    if und in ['AUD', 'EUR', 'GBP']:
        to_replace = 'USD.' + und
        replace = und + '.USD'
        tickers = [t.replace(to_replace, replace) for t in tickers]

    # pull data
    all_data = pd.DataFrame()
    start = dt - timedelta(days=5)
    DL = Datalake()
    for tick in tickers:
        data = DL.getData('CITI_VELOCITY', tick, 'VALUE', start, dt, None).rename(columns={'tstamp': 'date'})
        data['date'] = data['date'].apply(lambda x: pd.Timestamp(x).date())
        data = data[data.date == dt]
        try:
            data = data.rename(columns={'VALUE': data.ticker.unique()[0]}).drop(columns=['ticker'])
        except:
            print(tick)
        if len(data) == 0:
            print('Missing data for %s on %s' % (tick, dt))
        data.set_index('date', inplace=True)
        if 'IMPLIED_VOL' in tick:
            data /= 100
        all_data = pd.concat([all_data, data], axis=1)
    # reformat columns
    cols = all_data.columns
    cols = [x.replace('.%s.' % lower_ten, '.near.') for x in cols]
    cols = [x.replace('.%s.' % upper_ten, '.far.') for x in cols]
    all_data.rename(columns=dict(zip(all_data.columns, cols)), inplace=True)
    # add TTM
    all_data['near_TTM'] = (lower_exp - dt).days / 365.2425
    all_data['far_TTM'] = (upper_exp - dt).days / 365.2425
    return all_data


def process_data(raw_data, und, near_far, is_call):
    # split <raw_data> into 'near' or 'far' then process
    assert near_far == 'near' or near_far == 'far'
    assert len(raw_data) == 1

    # drop duplicate columns: occurs when target_tenor is equal to one fo the quoted tenors.
    if raw_data.near_TTM.values.all() == raw_data.far_TTM.all():
        raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()].copy()
        # set <near_far> to 'near' by default
        near_far = 'near'

    nf_cols = [col for col in raw_data.columns if near_far in col]
    nf_data = raw_data[nf_cols]
    nf_fwd = nf_data[[col for col in nf_data.columns if 'FORWARD' in col]].values[0][0]
    nf_data = nf_data[[col for col in nf_data.columns if 'IMPLIED_VOL' in col]]
    nf_TTM = raw_data['%s_TTM' % near_far].values[0]

    deltaK = []
    vols = []
    for col in nf_data.columns:
        strike = col.replace('FX.IMPLIED_VOL', '').replace('.%s.CITI' % near_far, '').replace('USD', '').replace(und,
                                                                                                                 '').replace(
            '.', '')
        if strike == 'ATM':
            deltaK.append(0.5)
            volATM = nf_data[col].values[0]
        else:
            strike = strike.replace('STRIKE_', '')
            deltaK.append(float(strike[1:]) / 100)
        vols.append(nf_data[col].values[0])
    if not is_call:
        deltaK = [-x for x in deltaK]
    logKs = [np.log(delta_strike_to_relative(deltaK[i], vols[i], nf_TTM, is_call)) for i in np.arange(len(vols))]
    return {'fwd': nf_fwd, 'TTM': nf_TTM, 'vols': vols, 'volATM': volATM, 'logKs': logKs}


def implied_vol_SSVI(dt, und, target_tenor, rel_fwd_K, is_call):
    data = pull_data(dt, und, target_tenor, is_call)
    near_data = process_data(data, und, 'near', is_call)
    far_data = process_data(data, und, 'far', is_call)
    near_params = fit_SSVI_slice(near_data['vols'], near_data['logKs'], near_data['TTM'], near_data['volATM'])
    far_params = fit_SSVI_slice(far_data['vols'], far_data['logKs'], far_data['TTM'], far_data['volATM'])
    TTM = (add_tenor(dt, target_tenor) - dt).days / 365.2425
    return interp_vol_SSVI(near_params, near_data['TTM'], near_data['volATM'], far_params, far_data['TTM'],
                           far_data['volATM'], np.log(rel_fwd_K), TTM)


check_values = False
if check_values:
    res = {}
    dt = date(2023, 1, 17)
    for und in ['AUD', 'EUR', 'GBP', 'CAD', 'JPY', 'CHF']:
        for ten in ['2W', '1M', '2M', '3M', '6M', '9M']:
            id = und + '_' + ten
            near_data = process_data(pull_data(dt, und, ten, 1), und, 'near', 1)
            # check ATM
            ATM_vol = near_data['volATM']
            ATM_K = delta_strike_to_relative(0.5, ATM_vol, near_data['TTM'], 1)
            vol_estimate = implied_vol_SSVI(dt, und, ten, ATM_K, 1)
            res[id + '50dC'] = 100 * (vol_estimate - ATM_vol)
            # check 10dC
            vol_10dC = near_data['vols'][-1]
            C10 = delta_strike_to_relative(0.1, vol_10dC, near_data['TTM'], 1)
            vol_estimate = implied_vol_SSVI(dt, und, ten, C10, 1)
            res[id + '10dC'] = 100 * (vol_estimate - vol_10dC)
            # check 10dP
            near_data = process_data(pull_data(dt, und, ten, 0), und, 'near', 0)
            vol_10dP = near_data['vols'][-1]
            P10 = delta_strike_to_relative(-0.1, vol_10dP, near_data['TTM'], 0)
            vol_estimate = implied_vol_SSVI(dt, und, ten, P10, 0)
            res[id + '10dP'] = 100 * (vol_estimate - vol_10dP)

    print('max error: %s (%.3f vol points)' % (max(res, key=res.get), res[max(res, key=res.get)]))
    print('min error: %s (%.3f vol points)' % (min(res, key=res.get), res[min(res, key=res.get)]))

## program entry
if __name__ == '__main__':
    # check vols vs listed values
    check_values = False
    if check_values:
        res = {}
        dt = date(2023, 1, 17)
        for und in ['AUD', 'EUR', 'GBP', 'CAD', 'JPY', 'CHF']:
            for ten in ['2W', '1M', '2M', '3M', '6M', '9M']:
                id = und + '_' + ten
                near_data = process_data(pull_data(dt, und, ten, 1), und, 'near', 1)
                # check ATM
                ATM_vol = near_data['volATM']
                ATM_K = delta_strike_to_relative(0.5, ATM_vol, near_data['TTM'], 1)
                vol_estimate = implied_vol_SSVI(dt, und, ten, ATM_K, 1)
                res[id + '50dC'] = 100 * (vol_estimate - ATM_vol)
                # check 10dC
                vol_10dC = near_data['vols'][-1]
                C10 = delta_strike_to_relative(0.1, vol_10dC, near_data['TTM'], 1)
                vol_estimate = implied_vol_SSVI(dt, und, ten, C10, 1)
                res[id + '10dC'] = 100 * (vol_estimate - vol_10dC)
                # check 10dP
                near_data = process_data(pull_data(dt, und, ten, 0), und, 'near', 0)
                vol_10dP = near_data['vols'][-1]
                P10 = delta_strike_to_relative(-0.1, vol_10dP, near_data['TTM'], 0)
                vol_estimate = implied_vol_SSVI(dt, und, ten, P10, 0)
                res[id + '10dP'] = 100 * (vol_estimate - vol_10dP)

        print('max error: %s (%.3f vol points)' % (max(res, key=res.get), res[max(res, key=res.get)]))
        print('min error: %s (%.3f vol points)' % (min(res, key=res.get), res[min(res, key=res.get)]))

    check_SABR = True
    fields = 'source,location,under_id,under_pricing_id,type,dimension,term,actual_date,capture_date,fit_source,spot,forward,time_to_expiry,alpha,beta,rho,nu'
    ex_vals = 'CLOSE|LDN|SURFACE'
    if check_SABR:
        ticker_to_CTPid = {'AUDUSD': '1127503', 'EURUSD': '1127704', 'GBPUSD': '1127737', 'USDCAD': '1128035',
                           'USDCHF': '1128031', 'USDJPY': '1128060'}
        DL = Datalake()
        TTM = (add_tenor(date.today(), '1M') - date.today()).days / 365
        diff_res = {}
        for und, tick in ticker_to_CTPid.items():
            # pull params
            # see https://gitea.capstoneco.com/dcirmirakis/ctp_py_examples/src/branch/master/data_provider.py
            all_data_pull = DL.getData('CTP_DAILY_VOL_SABR', tick, fields, date(2007, 1, 1), date.today(),
                                       extra_fields='source|location|dimension', extra_values=ex_vals)
            all_data_pull['abs_TTM'] = abs(all_data_pull.time_to_expiry - TTM)
            all_data_TTM = []
            for dt in all_data_pull.tstamp.unique():
                t1 = all_data_pull[all_data_pull.tstamp == dt]
                all_data_TTM.append(t1[t1.abs_TTM == t1.abs_TTM.min()])
            all_data_pull = pd.concat(all_data_TTM)
            all_data_pull = all_data_pull[['tstamp', 'forward', 'alpha', 'beta', 'rho', 'nu']]
            all_data_pull['tstamp'] = [pd.Timestamp(x).date() for x in all_data_pull.tstamp.values]
            all_data_pull.set_index(['tstamp'], inplace=True)
            all_data_pull['1m_ATM_SABR'] = [SABR_vol_CTP(x[0], TTM, x[0], x[1], x[2], x[3], x[4]) for x in
                                            all_data_pull.to_numpy()]
            # check SABR vol vs Citi
            tick_Citi = 'FX.IMPLIED_VOL.%s.ATM.1M.CITI' % (und[:3] + '.' + und[3:])
            Citi_data = DL.getData('CITI_VELOCITY', tick_Citi, 'VALUE', all_data_pull.index[0], all_data_pull.index[-1])
            Citi_data['tstamp'] = [pd.Timestamp(x).date() for x in Citi_data.tstamp.values]
            Citi_data.set_index(['tstamp'], inplace=True)
            common_dates = set(all_data_pull.index).intersection(Citi_data.index)
            calib_vols = all_data_pull[all_data_pull.index.isin(common_dates)]['1m_ATM_SABR']
            Citi_vols = Citi_data[Citi_data.index.isin(common_dates)].VALUE
            Citi_vols /= 100
            diff_citi = calib_vols.subtract(Citi_vols, axis='index')
            diff_res[und + '_diff_Citi'] = diff_citi

            # check SABR vol vs BBG
            tick_BBG = und + " 1M ATM VOL BVOL Curncy"
            BBG_data = DL.getData('BBG_PRICE', tick_BBG, 'PX_LAST', all_data_pull.index[0], all_data_pull.index[-1])
            BBG_data['tstamp'] = [pd.Timestamp(x).date() for x in BBG_data.tstamp.values]
            BBG_data.set_index(['tstamp'], inplace=True)
            common_dates = set(all_data_pull.index).intersection(BBG_data.index)
            calib_vols = all_data_pull[all_data_pull.index.isin(common_dates)]['1m_ATM_SABR']
            BBG_vols = BBG_data[BBG_data.index.isin(common_dates)].PX_LAST
            BBG_vols /= 100
            diff_BBG = calib_vols.subtract(BBG_vols, axis='index')
            diff_res[und + '_diff_BBG'] = diff_BBG

        # refactor <diff_res> to DF
        diff_res_df = pd.DataFrame()
        for und, df in diff_res.items():
            df = df.to_frame()
            df.rename(columns={0: und}, inplace=True)
            # scale to vol points
            df *= 100
            if len(diff_res_df) == 0:
                diff_res_df = df
            else:
                diff_res_df = diff_res_df.join(df)

        with open('diff_1mATM.pkl', 'wb') as f:
            pickle.dump(diff_res, f)

    test_sample_day = False
    if test_sample_day:
        # test with AUD values 12jan23
        deltaKs = [0.5, 0.35, 0.25, 0.1]

        # 2w
        fwd2w = 0.693662
        TTM1 = 10 / 252
        vols1 = [0.130787, 0.12984, 0.129884, 0.131392]
        logKs1 = [delta_strike_to_relative(deltaKs[i], vols1[i], TTM1, 1) for i in np.arange(len(vols1))]
        volATM1 = 0.130787
        params1 = fit_SSVI_slice(vols1, logKs1, TTM1, volATM1)

        # 1m
        fwd1m = 0.694138
        TTM2 = 22 / 252
        vols2 = [0.133259, 0.131652, 0.131261, 0.131946]
        logKs2 = [delta_strike_to_relative(deltaKs[i], vols2[i], TTM2, 1) for i in np.arange(len(vols2))]
        volATM2 = 0.133259
        params2 = fit_SSVI_slice(vols2, logKs2, TTM2, volATM2)

        test_vol = interp_vol_SSVI(params1, TTM1, volATM1, params2, TTM2, volATM2, np.log(0.973), 17 / 252)
