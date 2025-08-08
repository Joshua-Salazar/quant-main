from enum import Enum
import numpy as np


class SABRRawParams(Enum):
    SIGMA0 = "Sigma0"   # start vol
    VOV = "Vov"         # vol of vol
    RHO = "Rho"         # spot vol correlation


class SABRPhysicalParams(Enum):
    ATMFVOL = "ATMFVol"   # vol of the forward
    NU = "Nu"           # vol ratio
    RHO = "Rho"         # spot vol correlation


def convert_raw_params_to_physical_params(sigma0, vov, rho, tte):
    nu = vov / sigma0
    fwd = 1 + sigma0**2 * tte * nu / 4 * (rho + (2 - 3 * rho ** 2) * nu / 6)
    atmf_vol = sigma0 * fwd
    return atmf_vol, nu, rho, tte


def get_vol_from_raw_params(k, fwd, tte, sigma0, vov, rho):
    if k == fwd:
        nu = vov / sigma0
        tmp = 1 + sigma0 ** 2 * tte * nu / 4 * (rho + (2 - 3 * rho ** 2) * nu / 6)
        vol = sigma0 * tmp
    else:
        z = vov / sigma0 * np.log(fwd/k)
        exponent = (np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
        chi = np.log(exponent)
        tmp = 1 + (rho * vov * sigma0 / 4 + (2 - 3 * rho**2) * vov**2 / 24) * tte
        if z == 0:
            raise
        vol = sigma0 * z / chi * tmp
    return vol


def get_vol_from_physical_params(k, fwd, atmf_vol, nu, rho):
    z = nu * np.log(fwd/k)
    exponent = (np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
    chi = np.log(exponent)
    vol = atmf_vol * z / chi
    return vol
