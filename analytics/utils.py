from ..analytics.constants import EPSILON
from math import log, sqrt, exp
from scipy.stats import norm


def float_less(x, y):
    return x < y - EPSILON


def float_equal(x, y, threshold=EPSILON):
    return -threshold < x - y < threshold


def convert_dictionary(d, key_convert_func=lambda x: x, value_convert_func=lambda x: x):
    return dict(zip(map(key_convert_func, d.keys()), map(value_convert_func, d.values())))


def find_bracket_bounds(expiries, expiry):
    lb = -float('inf')
    ub = float('inf')
    for exp in expiries:
        if expiry >= exp:
            lb = max(lb, exp)
        else:
            break
    for exp in reversed(expiries):
        if expiry <= exp:
            ub = min(ub, exp)
        else:
            break
    return lb, ub


def interpolate_curve(curve, point, flat_extrapolate_lower=False, flat_extrapolate_upper=False):
    lb, ub = find_bracket_bounds(sorted(list(curve.keys())), point)
    if lb == -float('inf') and flat_extrapolate_lower:
        return curve[ub]
    if ub == float('inf') and flat_extrapolate_upper:
        return curve[lb]
    if float_equal(lb, ub):
        return curve[lb]
    interpolated_rate = curve[lb] + (point - lb) * (curve[ub] - curve[lb]) / (ub - lb)
    return interpolated_rate


class BSPricer:
    '''
    The Black-Scholes Model
    Calculate the price of option, delta, gamma, vega, theta, vanna, vomma, dual_delta,
    dual_gamma, forward price of underlying stock, and implied volatility
    '''
    __slots__ = ('callputflag', 'S', 'X', 'T', 'r', 'q', 'v', 'isminute', 'TinBdays','d1', 'd2', 'cdf_d1',
                 'cdf_d2', 'cdf_m_d1', 'cdf_m_d2', 'pdf_d1', 'pdf_d2')
    def __init__(self, callputflag, S, X, T, r, q, v=None, isminute=0, TinBdays=False):
        '''
        constructor
        :param callputflag: 'C' for call and 'P' for put
        :param S: spot price of underlying asset
        :param X: strike
        :param T: time to maturity in days
        :param r: risk free interest rate
        :param v: volatility
        :param q: dividend paying rate
        :param isminute: 1 if T is in minute, 0 if T is in years
        '''
        self.callputflag = callputflag
        self.S = S
        self.X = X
        self.r = r
        self.v = v
        self.q = q

        if isminute == 1:
            T = T/1440

        self.TinBdays = TinBdays
        if TinBdays:
            self.T = T/252
        else:
            self.T = T/365

        if v is not None:
            #calculate d1 and d2
            if self.T != 0 and self.v != 0:
                self.d1 = (log(S/X)+(r-q+0.5*v**2)*self.T)/(v*sqrt(self.T))

            else:
                self.d1 = log(S/X)
            self.d2 = self.d1-v*sqrt(self.T)

            #calculate cdf of d1, d2, -d1 and -d2, pdf of d1 and d2.
            self.cdf_d1 = norm.cdf(self.d1)
            self.cdf_d2 = norm.cdf(self.d2)
            self.cdf_m_d1 = norm.cdf(-self.d1)
            self.cdf_m_d2 = norm.cdf(-self.d2)
            self.pdf_d1 = norm.pdf(self.d1)
            self.pdf_d2 = norm.pdf(self.d2)

    def forward(self):
        '''
        :return: the forward price of the underlying asset
        '''
        fwd = self.S*exp((self.r-self.q)*self.T)
        return fwd

    def price(self):
        '''
        :return: the Black-Scholes price of the option
        '''
        if self.callputflag == 'C':
            if self.T == 0:
                BS = max(self.S-self.X, 0)
            else:
                BS = self.S*exp(-self.q*self.T)*self.cdf_d1-self.X*exp(-self.r*self.T)*self.cdf_d2
        else:
            if self.T == 0:
                BS = max(self.X-self.S, 0)
            else:
                BS = self.X * exp(-self.r * self.T) * self.cdf_m_d2 - self.S * exp(-self.q * self.T) * self.cdf_m_d1
        return BS

    def delta(self):
        '''
        :return: the delta of the option
        '''
        if self.T == 0:
            return 0
        if self.callputflag == 'C':
            d = exp(-self.q*self.T)*self.cdf_d1
        else:
            d = -exp(-self.q * self.T) * self.cdf_m_d1
        return d

    def gamma(self):
        '''
        :return: the gamma of the option
        '''
        #----------------------------------------------------------------------------------------------------------------????
        #g = exp(-self.q*self.T)*self.pdf_d1/(self.S*self.v*sqrt(self.T))
        if self.T == 0:
            return 0
        # g = exp(-self.q * self.T) * self.pdf_d1 / (self.S * self.v * sqrt(self.T)) * 0.01 * self.S

        g = exp(-self.q * self.T) * self.pdf_d1 / (self.S * self.v * sqrt(self.T))
        return g

    def vega(self):
        '''
        :return: the vega of the option
        '''
        if self.T == 0:
            return 0
        v = self.S*exp(-self.q*self.T)*self.pdf_d1*sqrt(self.T)*0.01
        return v

    def theta(self):
        '''
        :return: the theta of the option
        '''
        if self.T == 0:
            return 0
        if self.callputflag == 'C':
            th = -self.S*exp(-self.q*self.T)*self.pdf_d1*self.v / (2*sqrt(self.T))\
                    - self.r*self.X*exp(-self.r*self.T)*self.cdf_d2 \
                    + self.q*self.S*exp(-self.q*self.T)*self.cdf_d1
        else:
            th = -self.S*exp(-self.q*self.T)*self.pdf_d1*self.v / (2*sqrt(self.T))\
                    + self.r*self.X*exp(-self.r*self.T)*self.cdf_m_d2 \
                    - self.q*self.S*exp(-self.q*self.T)*self.cdf_m_d1

        if self.TinBdays:
            return th/252
        else:
            return th/365

    def vanna(self):
        '''
        :return: the vanna of the option
        '''
        va = -exp(-self.q*self.T)*self.pdf_d1 * self.d2 / self.v
        return va

    def vomma(self):
        '''
        :return: the vomma of the option
        '''
        vo = 0.01*self.S*exp(-self.q*self.T)*self.pdf_d1*sqrt(self.T)*self.d1*self.d2/self.v
        return vo

    def dual_delta(self):
        '''
        :return: the dual_delta of the option
        '''
        if self.callputflag == 'C':
            dual_d = -exp(-self.r*self.T)*self.cdf_d2
        else:
            dual_d = exp(-self.r * self.T) * self.cdf_m_d2
        return dual_d

    def dual_gamma(self):
        '''
        :return: the dual_gamma of the option
        '''
        dual_g = exp(-self.r*self.T)*self.pdf_d2 / (self.X*sqrt(self.T)*self.v)
        return dual_g

    def price_vega_changing_vol(self, vol):
        '''
        Given volatility, calculate price and vega.
        Used to calculate the implied volatility.
        :param vol: volatility
        :return: Black-Scholes price, vega
        '''
        try:
            d1 = (log(self.S/self.X) + ((self.r-self.q+0.5*vol**2)*self.T))/(vol*sqrt(self.T))
            d2 = d1 - vol * sqrt(self.T)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_m_d1 = norm.cdf(-d1)
            cdf_m_d2 = norm.cdf(-d2)
            if self.callputflag == 'C':
                BS = self.S * exp(-self.q * self.T) * cdf_d1 - self.X * exp(-self.r * self.T) * cdf_d2
            else:
                BS = self.X * exp(-self.r * self.T) * cdf_m_d2 - self.S * exp(-self.q * self.T) * cdf_m_d1

            pdf_d1 = norm.pdf(d1)
            vega = self.S * exp(-self.q * self.T) * pdf_d1 * sqrt(self.T) * 0.01

            return BS, vega
        except:
            print('Error where Spot =', self.S)

    def iv(self, actual_p):
        '''
        Using Newton's method to calculate the implied volatility
        :param actual_p: the actual price of the option
        :return: implied volatility
        '''
        actual_p = actual_p[0]
        tolerance = 0.01
        vol_old = 0
        vol_new = 0.5
        p, vega = self.price_vega_changing_vol(vol_new)
        p_zero_vol, temp = self.price_vega_changing_vol(0.01)
        counter = 0

        if p_zero_vol <= actual_p:
            while (abs(p-actual_p) > tolerance or abs(vol_new-vol_old) > tolerance) and counter <= 50:

                counter += 1
                vol_old = vol_new
                vol_new = max(0.01, vol_old - (p-actual_p) / (max(0.001, vega)*100))  #when the option is deep OTM, p=actual_p=0, can not update vol, will always be 0.5
                p, vega = self.price_vega_changing_vol(vol_new)
                if counter == 30:
                    vol_new = 4
                    p, vega = self.price_vega_changing_vol(vol_new)

        else:
            #print('out of bounds')
            vol_new = 0.01

        if counter > 50 and abs(p-actual_p) >= tolerance:
            #print('no convergence',self.X)
            # print(p_zero_vol)
            # print(actual_p)
            vol_new = 0.01

        return vol_new

    def get_value(self, G, *args):
        '''
        Get value according to command G
        :param G: be in {'p','d','g','v','t','vanna','vomma','dual_delta','dual_gamma','f','iv'}
        :param args: if G=='iv', there should be another input parameter
                     to indicate the actual price of the option
        :return: return the wanted value: price, delta, gamma, vega, theta, vanna,
                 vomma, dual_delta, dual_gamma, forward or implied volatility
        '''
        if G == 'p':
            return self.price()
        elif G == 'd':
            return self.delta()
        elif G == 'g':
            return self.gamma()
        elif G == 'v':
            return self.vega()
        elif G == 't':
            return self.theta()
        elif G == 'vanna':
            return self.vanna()
        elif G == 'vomma':
            return self.vomma()
        elif G == 'dual_delta':
            return self.dual_delta()
        elif G == 'dual_gamma':
            return self.dual_gamma()
        elif G == 'f':
            return self.forward()
        elif G == 'iv':
            return self.iv(args)
        else:
            print('invalid command.')
            return None