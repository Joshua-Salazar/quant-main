
import numpy as np
from scipy.interpolate import griddata
from ..analytics.FX_vol_interpolation import fit_SSVI_slice, interp_vol_SSVI
from ..analytics.utils import find_bracket_bounds, interpolate_curve
from ..dates.utils import add_tenor


class CRVolSurface:
    def __init__(self):
        pass

    def get_expiration_pillars(self):
        pass

    def get_strike_pillars(self, expiration):
        pass

    def get_forward(self, TTM):
        pass

    def get_vol(self, TTM, strike):
        pass


class CRVolSurfaceFromQuotedVols(CRVolSurface):
    def __init__(self, underlier, base_date, abs_strikes, forwards, vols, spots, holidays = []):
        self.underlier = underlier
        self.base_date = base_date
        self.abs_strikes = abs_strikes
        self.forwards = forwards
        self.vols = vols
        self.spots = spots
        self.expirations = list(map(lambda x: add_tenor(base_date, x), list(forwards.keys())))
        self.holidays = holidays

    def get_expiration_pillars(self):
        return list(self.forwards.keys())

    def get_strike_pillars(self, expiration):
        return self.abs_strikes[ expiration ]

    def get_forward(self, TTM ):
        return interpolate_curve(self.forwards, TTM, flat_extrapolate_lower=True, flat_extrapolate_upper=True)

    def get_vol_interpolation(self, TTM, abs_strike):
        time = []
        strikes = []
        values = []
        for t, s in self.abs_strikes.items():
            time.append(t)
            strikes_t = []
            values_t = []
            for k, v in s.items():
                strikes_t.append(v)
                values_t.append(self.vols[t][k])
            strikes_t.append(1)
            values_t.append(values_t[-1])
            strikes_t.insert(0, strikes_t[0] * 10)
            values_t.insert(0, values_t[0])

            strikes.append(strikes_t)
            values.append(values_t)

        # Flatten the data for griddata
        points = np.array([(t, s) for t, strike_list in zip(time, strikes) for s in strike_list])
        values_flat = np.concatenate(values)

        # Define new time and strike points for interpolation
        new_time = [max(TTM, time[0])]
        new_strikes = [abs_strike]
        new_points = np.array([(t, s) for t in new_time for s in new_strikes])

        # Interpolate the values on the new grid
        interpolated_values = griddata(points, values_flat, new_points, method='linear')

        # Reshape the interpolated values for plotting
        interpolated_values = interpolated_values.reshape(len(new_time), len(new_strikes))

        return interpolated_values[0][0]

    def get_vol(self, TTM, abs_strike):
        forward = self.get_forward(TTM)

        # interpolate vol
        lb, ub = find_bracket_bounds(sorted(list(self.vols.keys())), TTM)
        if lb == -float('inf'):
            lb = ub
        logKs_lb = []
        vols_lb = []
        for delK, vol in self.vols[lb].items():
            if delK == 0.5:
                volATM_lb = vol
            else:
                absK = self.abs_strikes[lb][delK]
                logKs_lb.append(np.log(absK / forward))
                vols_lb.append(vol)
        params_lb = fit_SSVI_slice(vols_lb, logKs_lb, lb, volATM_lb)

        logKs_ub = []
        vols_ub = []
        for delK, vol in self.vols[ub].items():
            if delK == 0.5:
                volATM_ub = vol
            else:
                absK = self.abs_strikes[ub][delK]
                logKs_ub.append(np.log(absK / forward))
                vols_ub.append(vol)
        params_ub = fit_SSVI_slice(vols_ub, logKs_ub, ub, volATM_ub)

        vol = interp_vol_SSVI(params_lb, lb, volATM_lb, params_ub, ub, volATM_ub, np.log(abs_strike / forward), TTM)
        return vol
