import numpy as np
from ..analytics.FX_vol_interpolation import delta_strike_to_relative, fit_SSVI_slice, interp_vol_SSVI
from ..analytics.utils import find_bracket_bounds, float_equal, interpolate_curve
from ..dates.utils import add_tenor, datetime_to_tenor


class FXVolSurface:
    def __init__(self):
        pass

    def get_expiration_pillars(self):
        pass

    def get_strike_pillars(self, expiration):
        pass

    def get_forward(self, expiration):
        pass

    def get_vol(self, expiration, strike):
        pass


class FXVolSurfaceFromQuotedVols(FXVolSurface):
    def __init__(self, fx_pair, base_date, spot, delta_strikes, forwards, vols):
        self.fx_pair = fx_pair
        self.base_date = base_date
        self.spot = spot
        self.delta_strikes = delta_strikes
        self.forwards = forwards
        self.vols = vols
        self.vol_params = {}
        self.expirations = list(map(lambda x: add_tenor(base_date, x), list(forwards.keys())))

    def get_expiration_pillars(self):
        return list(self.forwards.keys())

    def get_strike_pillars(self, expiration):
        return self.delta_strikes

    def get_forward(self, expiration):
        TTM = datetime_to_tenor(expiration, self.base_date)
        forward = interpolate_curve(self.forwards, TTM, flat_extrapolate_lower=True, flat_extrapolate_upper=True)
        return forward

    def get_vol(self, expiration, strike):
        TTM = datetime_to_tenor(expiration, self.base_date)

        forward = self.get_forward(expiration)
        disc = self.spot / forward

        # interpolate vol
        lb, ub = find_bracket_bounds(sorted(list(self.vols.keys())), TTM)
        if lb == -float('inf'):
            lb = ub
        if ub == float('inf'):
            ub = lb

        if lb in self.vol_params:
            params_lb = self.vol_params[lb]
        else:
            # TODO: change or add SABR
            logKs_lb = []
            vols_lb = []
            for deltak, vol in zip(self.delta_strikes, self.vols[lb]):
                if float_equal(deltak, 0.5):
                    volATM_lb = vol
                else:
                    logKs_lb.append( np.log( delta_strike_to_relative(disc, deltak, vol, lb, deltak > 0 ) ))
                    vols_lb.append(vol)
            params_lb = fit_SSVI_slice(vols_lb, logKs_lb, lb, volATM_lb)
            # self.vol_params[lb] = params_lb

        if ub in self.vol_params:
            params_ub = self.vol_params[ub]
        else:
            logKs_ub = []
            vols_ub = []
            for deltak, vol in zip(self.delta_strikes, self.vols[ub]):
                if float_equal(deltak, 0.5):
                    volATM_ub = vol
                else:
                    logKs_ub.append( np.log( delta_strike_to_relative(disc, deltak, vol, ub, deltak > 0 ) ))
                    vols_ub.append(vol)
            params_ub = fit_SSVI_slice(vols_ub, logKs_ub, ub, volATM_ub)
            # self.vol_params[ub] = params_ub

        # TODO: change or add SABR
        vol = interp_vol_SSVI(params_lb, lb, volATM_lb, params_ub, ub, volATM_ub, np.log(strike / forward), TTM)

        return vol
