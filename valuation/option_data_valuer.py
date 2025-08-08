from ..analytics.utils import float_equal
from ..interface.ivaluer import IValuer
from ..interface.imarket import IMarket
from ..tradable.future import Future
from ..tradable.option import Option
from ..valuation import valuer_utils
from ..dates.utils import datetime_diff
import numpy as np
from scipy.stats import norm


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


class OptionDataValuer(IValuer):
    def set_backfill_markets(self, backfill_markets):
        self.backfill_markets = backfill_markets

    def price(self, tradable: Option, market: IMarket, calc_types='price', **kwargs):
        if isinstance(tradable.underlying, Future) and tradable.underlying.root != 'VX':
            data = market.get_future_option_data(tradable)
            if data is None:
                for backfill_market in self.backfill_markets:
                    data = backfill_market.get_future_option_data(tradable)
                    if data is not None:
                        break
        else:
            data = market.get_option_data(tradable)
            if data is None:
                for backfill_market in self.backfill_markets:
                    data = backfill_market.get_option_data(tradable)
                    if data is not None:
                        break
        if data is None:
            print(f"missing option data on {market.get_base_datetime().strftime('%Y-%m-%d')} "
                  f"for trade: {tradable.name()}")
        assert data is not None
        data["revega"] = data["vega"] * data["iv"]
        return valuer_utils.return_results_based_on_dictionary(calc_types, data)


class OptionDataValuer_Zero_Px(OptionDataValuer):
    def __init__(self, otm_threshold=-0.1, itm_threshold=0.1, raise_if_zero_px=True, underlying_valuer=None,
                 pct_px_round_to_zero=0.0001, verbose=True):
        self.otm_threshold = otm_threshold
        self.itm_threshold = itm_threshold
        self.raise_if_zero_px = raise_if_zero_px
        self.underlying_valuer = underlying_valuer
        self.pct_px_round_to_zero = pct_px_round_to_zero
        self.verbose = verbose

    def set_backfill_markets(self, backfill_markets):
        self.backfill_markets = backfill_markets

    def itm_pct(self, tradable: Option, market: IMarket):
        spot = tradable.underlying.price(market, calc_types='price', valuer=self.underlying_valuer)
        if float_equal(spot, 0):
            return 0
        phi = 1 if tradable.is_call else -1
        res = phi * (spot - tradable.strike)
        res = res / spot
        return res

    def price(self, tradable: Option, market: IMarket, calc_types='price', **kwargs):
        results = super().price(tradable, market, calc_types, **kwargs)
        if not isinstance(calc_types, list):
            calc_types = [calc_types]
            results = [results]
        else:
            results = list(results)

        if 'price' in calc_types:
            price_index = calc_types.index('price')
            price = results[price_index]
            if price == 0.:
                iv = super().price(tradable, market, ['iv'], **kwargs)
                if iv > 0:
                    spot = tradable.underlying.price(market, calc_types='price', valuer=self.underlying_valuer)
                    pct_px = Black76(tradable.strike, tradable.expiration, tradable.is_call, market.get_base_datetime(),
                                 tradable.underlying.price(market, calc_types='price', valuer=self.underlying_valuer),
                                 iv, 1)['price']/spot
                    if pct_px < self.pct_px_round_to_zero:
                        if self.verbose:
                            print(f'Black model price is {pct_px: .5%} so accepted 0 price on ' +
                                  market.base_datetime.strftime('%Y-%m-%d'))
                        if len(results) == 1:
                            return results[0]
                        else:
                            return tuple(results)

                itm_pct = self.itm_pct(tradable, market)
                if itm_pct > self.itm_threshold:
                    if self.verbose:
                        print('used intrinsic for '+tradable.name()+' on '+market.base_datetime.strftime('%Y-%m-%d'))
                    results[price_index] = tradable.intrinsic_value(market, self.underlying_valuer)
                elif itm_pct < self.otm_threshold:
                    if self.otm_threshold != 0 and self.verbose:
                        print('accepted zero price for '+tradable.name()+' on '+market.base_datetime.strftime('%Y-%m-%d'))
                    results[price_index] = 0.
                elif self.raise_if_zero_px:
                    raise ValueError('zero price for '+tradable.name()+' on '+market.base_datetime.strftime('%Y-%m-%d'))
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
