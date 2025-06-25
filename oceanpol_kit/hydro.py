from typing import Tuple
from functools import cache

import numpy as np
from csu_radartools.csu_fhc import csu_fhc_summer


# def hid_beta(x_arr, a, b, m):
#     """Beta function calculator"""
#     return 1.0 / (1.0 + (((x_arr - m) / a) ** 2) ** b)


# @cache
# def get_weights_hid():
#     weights = {"DZ": 1.5, "DR": 0.8, "KD": 1.0, "RH": 0.8, "T": 0.4}
#     sets = beta_functions.get_mbf_sets_summer(True, plot_flag=False, band="C")
#     one = ["DZ", "DR", "KD", "RH", "T"]
#     two = ["Zh_set", "Zdr_set", "Kdp_set", "rho_set", "T_set"]
#     coeffs = {}
#     for i in range(len(one)):
#         coeffs[one[i]] = sets[two[i]]

#     return weights, coeffs


# def compute_hid(dbz, zdr, kdp, rhohv, temperature):
#     # HID types:           Species #:
#     # -------------------------------
#     # Unclassified             0
#     # Drizzle                  1
#     # Rain                     2
#     # Ice Crystals             3
#     # Aggregates               4
#     # Wet Snow                 5
#     # Vertical Ice             6
#     # Low-Density Graupel      7
#     # High-Density Graupel     8
#     # Hail                     9
#     # Big Drops                10
#     weights, coeffs = get_weights_hid()

#     data = {}
#     data["DZ"] = dbz
#     data["DR"] = zdr
#     data["KD"] = kdp
#     data["RH"] = rhohv
#     data["T"] = temperature

#     hid = np.zeros((10, *dbz.shape))
#     for i in range(10):
#         for k, v in data.items():
#             params = coeffs[k]
#             a = params["a"][i]
#             b = params["b"][i]
#             m = params["m"][i]
#             hid[i, :, :] += weights[k] * hid_beta(v, a, b, m)

#     score = np.argmax(hid, axis=0)
#     return score.astype(np.int16)


def compute_hid(dbz, zdr, kdp, rhohv, temperature):
    hid = csu_fhc_summer(
        use_temp=True,
        method="hybrid",
        dz=dbz,
        zdr=zdr,
        ldr=None,
        kdp=kdp,
        rho=rhohv,
        T=temperature,
        verbose=False,
        plot_flag=False,
        n_types=10,
        temp_factor=1,
        band="C",
    )
    hid[np.isnan(dbz)] = 0
    return hid


def get_dsd_estimate(dbz: np.ndarray, zdr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the drop size distribution (DSD) parameters from reflectivity
    (DBZ) and differential reflectivity (ZDR).

    Parameters
    ----------
    dbz : np.ndarray
        A 2D array containing the reflectivity values.
    zdr : np.ndarray
        A 2D array containing the differential reflectivity values.

    Returns
    -------
    nw, d0: Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - nw: A 2D array of the estimated normalized intercept parameter.
        - d0: A 2D array of the estimated median volume diameter.
    """
    d0 = np.zeros_like(zdr)
    pos = (-0.5 <= zdr) & (zdr < 1.25)
    tmp = 0.0203 * zdr**4 - 0.1488 * zdr**3 + 0.2209 * zdr**2 + 0.5571 * zdr + 0.801
    d0[pos] = tmp[pos]

    pos = (1.25 <= zdr) & (zdr < 5)
    tmp = 0.0355 * zdr**3 - 0.3021 * zdr**2 + 1.0556 * zdr + 0.6844
    d0[pos] = tmp[pos]

    nw = 10 ** (dbz / 10) / (0.056 * d0**7.319)
    return np.log10(nw), d0


def get_rainfall_estimate(
    zh: np.ndarray, zdr: np.ndarray, kdp: np.ndarray, temperature: np.ndarray, southern_ocean: bool
) -> np.ndarray:
    """
    Estimate rainfall rate from reflectivity (ZH), differential reflectivity
    (ZDR), specific differential phase (KDP), and temperature.

    Parameters
    ----------
    zh : np.ndarray
        A 2D array containing the reflectivity values.
    zdr : np.ndarray
        A 2D array containing the differential reflectivity values.
    kdp : np.ndarray
        A 2D array containing the specific differential phase values.
    temperature : np.ndarray
        A 2D array containing the temperature values.
    southern_ocean : bool
        A boolean indicating if the data is from the Southern Ocean.

    Returns
    -------
    rainfall: np.ndarray
        A 2D array containing the estimated rainfall rates.
    """
    sigma_dr = 10 ** (0.1 * zdr)
    eta_h = 10 ** (0.1 * zh)
    rainfall = np.zeros_like(zh) + np.nan

    pos = (kdp <= 0.3) & (zdr <= 0.25)
    if southern_ocean:
        tmp = 0.021 * eta_h**0.72
    else:
        tmp = 0.016 * eta_h**0.846
    rainfall[pos] = tmp[pos]

    pos = (kdp <= 0.3) & (zdr > 0.25)
    if southern_ocean:
        tmp = 0.0086 * eta_h**0.91 * sigma_dr ** (-4.21)
    else:
        tmp = 0.011 * eta_h**0.825 * sigma_dr ** (-3.055)
    rainfall[pos] = tmp[pos]

    pos = (kdp > 0.3) & (zdr <= 0.25)
    if southern_ocean:
        tmp = 30.62 * kdp**0.78
    else:
        tmp = 16.171 * kdp**0.742
    rainfall[pos] = tmp[pos]

    pos = (kdp > 0.3) & (zdr > 0.25)
    if southern_ocean:
        tmp = 45.70 * kdp ** (0.88) * sigma_dr ** (-1.67)
    else:
        tmp = 24.199 * kdp ** (0.827) * sigma_dr ** (-0.488)

    rainfall[pos] = tmp[pos]
    rainfall[np.isnan(rainfall) | (temperature <= -18)] = 0
    return rainfall


def get_snowfall_estimate(dbz_clean: np.ndarray, kdp: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """
    Estimate snowfall rate from reflectivity (DBZ) and specific differential phase (KDP).

    Parameters
    ----------
    dbz_clean : np.ndarray
        A 2D array containing the cleaned reflectivity values.
    kdp : np.ndarray
        A 2D array containing the specific differential phase values.
    temps : np.ndarray
        A 2D array containing the temperature values.

    Returns
    -------
    np.ndarray
        A 2D array containing the estimated snowfall rates.
    """
    snow = 1.48 * kdp**0.615 * (10 ** (dbz_clean / 10)) ** 0.33
    snow[temps > 0] = np.nan
    return snow
