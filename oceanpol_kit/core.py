"""
This module contains functions for processing radar data, including calculations for
differential reflectivity, drop size distribution, hydrometeor classification, and more.

@title: oceanpol_kit
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology
@date: 19/03/2025

.. autosummary::
    :toctree: generated/

    resolve_fields
    area_std
    correct_zdr
    get_hydrometeor_mask
    get_velocity_mask
    get_precip_mask
    get_phidp
    speckle_filter
    write_hvar_dset
    process_oceanpol
"""

import time
from typing import List, Tuple, Optional

import h5py
import pyodim
import unravel
import numpy as np
import pandas as pd
import xarray as xr

from numba import njit
from phido import phidp_to_kdp
from . import hydro
from . import atten
from . import temperature


# Canonical field name -> ordered list of accepted ODIM aliases.
# The radar variable names changed over the OceanPOL archive (e.g. single-pol
# style "SNR"/"VRAD"/"WRAD" vs dual-pol style "SNRH"/"VRADH"/"WRADH"), so fields
# are resolved by trying each alias in order. Add new aliases here as needed.
FIELD_ALIASES = {
    "DBZH": ["DBZH"],
    "TH": ["TH"],
    "RHOHV": ["RHOHV", "RHOHVH"],
    "ZDR": ["ZDR"],
    "PHIDP": ["PHIDP", "PHIDPH"],
    "SNR": ["SNR", "SNRH"],
    "VRAD": ["VRAD", "VRADH"],
    "WRAD": ["WRAD", "WRADH"],
    "SQI": ["SQI", "SQIH"],
}

# Fields without which the file cannot be processed.
REQUIRED_FIELDS = ["DBZH", "TH", "RHOHV", "ZDR", "PHIDP", "SNR", "VRAD"]


def resolve_fields(radarlist: List["xr.Dataset"], aliases: dict = FIELD_ALIASES) -> dict:
    """
    Resolve canonical field names to the actual variable names present in the
    radar volume, accounting for naming conventions that changed over the
    archive.

    A canonical name is only resolved to an alias if that alias is present in
    *every* sweep of the volume, so the same variable name is used consistently
    across the whole file.

    Parameters
    ----------
    radarlist : list of xr.Dataset
        The list of per-sweep datasets returned by pyodim.
    aliases : dict, optional
        Mapping of canonical name -> ordered list of accepted aliases
        (default is FIELD_ALIASES).

    Returns
    -------
    dict
        Mapping of canonical name -> actual variable name (or None if no alias
        is present in all sweeps).

    Raises
    ------
    KeyError
        If any field listed in REQUIRED_FIELDS cannot be resolved.
    """
    resolved = {}
    for canonical, candidates in aliases.items():
        resolved[canonical] = next(
            (name for name in candidates if all(name in r for r in radarlist)), None
        )

    missing = [c for c in REQUIRED_FIELDS if resolved.get(c) is None]
    if missing:
        available = sorted({v for r in radarlist for v in r.data_vars})
        raise KeyError(
            f"Required field(s) {missing} not found. "
            f"Available variables: {available}. "
            f"Add the correct alias to FIELD_ALIASES."
        )

    return resolved


@njit(cache=True)
def area_std(rawphi: np.ndarray, winlen: int = 10) -> np.ndarray:
    """
    Calculate the standard deviation of a sliding window over a 2D array.

    Parameters
    ----------
    rawphi: np.ndarray
        A 2D array of shape (na, nr) containing the input data.
    winlen: int, optional
        The length of the sliding window (default is 10).

    Returns
    -------
    area_phi: np.ndarray
        A 2D array of the same shape as `rawphi` containing the standard deviation
        of each sliding window.
    """
    na, nr = rawphi.shape
    area_phi = np.zeros_like(rawphi)
    for i in range(na):
        for j in range(nr - winlen):
            window = rawphi[i, j : (j + winlen)]
            area_phi[i, j] = np.std(window)
        for j in range(nr - 1, nr - winlen - 1, -1):
            window = rawphi[i, j - winlen : j]
            area_phi[i, j] = np.std(window)

    return area_phi


def correct_zdr(zdr: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """
    Correct differential reflectivity (ZDR) from noise based on the Schuur et al.
    2003 NOAA report (p7 eq 6).

    Parameters
    ----------
    zdr : np.ndarray
        A 2D array containing the differential reflectivity values.
    snr : np.ndarray
        A 2D array containing the signal-to-noise ratio values.

    Returns
    -------
    corr_zdr: np.ndarray
        A 2D array containing the corrected differential reflectivity values.
    """
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def get_hydrometeor_mask(dbz: np.ndarray, phidp: np.ndarray, rhohv: np.ndarray) -> np.ndarray:
    """
    Generate a mask for hydrometeors based on reflectivity (DBZ), differential
    phase (PHIDP), and correlation coefficient (RHOHV).

    Parameters
    ----------
    dbz : np.ndarray
        A 2D array containing the reflectivity values.
    phidp : np.ndarray
        A 2D array containing the differential phase values.
    rhohv : np.ndarray
        A 2D array containing the correlation coefficient values.

    Returns
    -------
    np.ndarray
        A 2D array containing the hydrometeor mask.
    """
    area = area_std(phidp)
    pos = (area > 60) | (rhohv < 0.5) | (dbz < -20)
    pos[(rhohv > 0.90)] = 0
    return pos


def get_velocity_mask(
    vel: np.ndarray,
    sqi: Optional[np.ndarray] = None,
    sqi_min: float = 0.4,
    texture_max: Optional[float] = None,
    texture_win: int = 5,
) -> np.ndarray:
    """
    Build a permissive "keep" mask for Doppler velocity that separates coherent
    echo (including weak clear-air returns) from incoherent receiver noise.

    The discriminator is coherence, not amplitude: receiver noise has near-zero
    normalized coherent power (SQI) and a random, high-texture velocity field,
    whereas genuine echo - even weak clear-air Bragg/insect returns at low SNR -
    is coherent. Reflectivity, SNR and RHOHV thresholds are deliberately not
    used here so that clear-air velocity is preserved.

    Parameters
    ----------
    vel : np.ndarray
        Raw (folded) Doppler velocity.
    sqi : np.ndarray, optional
        Normalized coherent power / signal quality index. If given, gates with
        SQI < sqi_min are rejected (primary discriminator).
    sqi_min : float, optional
        Minimum SQI to keep a gate (default 0.4).
    texture_max : float, optional
        If set, also reject gates whose local velocity texture (windowed
        standard deviation, m s-1) exceeds this value. Off by default to avoid
        trimming aliasing fold lines in real echo; used automatically (6.0) if
        no SQI is available.
    texture_win : int, optional
        Range window length for the texture estimate (default 5).

    Returns
    -------
    np.ndarray
        Boolean array, True where the velocity gate should be kept.
    """
    vel = np.asarray(vel)
    good = np.isfinite(vel)

    if sqi is not None:
        good = good & (np.asarray(sqi) >= sqi_min)

    use_texture = texture_max if texture_max is not None else (6.0 if sqi is None else None)
    if use_texture is not None:
        texture = area_std(np.ascontiguousarray(vel, dtype="float64"), texture_win)
        good = good & (texture <= use_texture)

    return good


def get_precip_mask(
    rhohv: np.ndarray,
    snr: np.ndarray,
    dbz: np.ndarray,
    rho_min: float = 0.8,
    snr_min: float = 3.0,
) -> np.ndarray:
    """
    Build a strict "keep" mask of genuine precipitation gates for the phase and
    retrieval products (PHIDP/KDP, rain, snow, DSD, HID).

    Unlike the velocity mask, phase is only physical in precipitation, so this
    is intentionally conservative: high correlation coefficient, adequate
    signal-to-noise and finite reflectivity. Applying it before integrating
    PHIDP prevents KDP noise from accumulating along the ray.

    Parameters
    ----------
    rhohv : np.ndarray
        Correlation coefficient.
    snr : np.ndarray
        Signal-to-noise ratio (dB).
    dbz : np.ndarray
        (Cleaned) reflectivity; only finite values are kept.
    rho_min : float, optional
        Minimum RHOHV to keep a gate (default 0.8).
    snr_min : float, optional
        Minimum SNR in dB to keep a gate (default 3.0).

    Returns
    -------
    np.ndarray
        Boolean array, True where the gate is genuine precipitation.
    """
    return (
        (np.asarray(rhohv) >= rho_min)
        & (np.asarray(snr) >= snr_min)
        & np.isfinite(np.asarray(dbz))
    )


def get_phidp(
    r: np.ndarray,
    phidp: np.ndarray,
    refl: np.ndarray,
    temperature: np.ndarray,
    precip: Optional[np.ndarray] = None,
    window: List[int] = [3, 7],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the corrected differential phase (PHIDP) and specific
    differential phase (KDP).

    Parameters
    ----------
    r : np.ndarray
        A 1D array containing the range values.
    phidp : np.ndarray
        A 2D array containing the differential phase values.
    refl : np.ndarray
        A 2D array containing the reflectivity values.
    temperature : np.ndarray
        A 2D array containing the temperature values.
    precip : np.ndarray, optional
        Boolean precipitation mask (see get_precip_mask). Where False, KDP is
        not integrated and the outputs are set to NaN. If None, no gating is
        applied.
    window : List[int], optional
        A list containing the window sizes for KDP calculation (default is [3, 7]).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - phidp_corr: A 2D array of the corrected differential phase.
        - kdp: A 2D array of the specific differential phase.
    """
    dr = (r[1] - r[0]) / 1000
    angles = np.sort(np.mod(phidp[~phidp.mask].ravel() + 180, 360) - 180)
    factor = 2 if np.all(abs(np.diff(angles, append=angles[0])) <= 180) else 1

    kdp = phidp_to_kdp(phidp, dr, window=window, factor=factor, complex=True)
    kdp[(kdp < 0) & (refl < 40) & (temperature >= 0)] = 0

    if precip is not None:
        # Restrict phase to genuine precipitation: outside it KDP is set to 0 so
        # the cumulative integration below does not accumulate noise along the
        # ray (the cause of the radial PHIDP streaks over clear air).
        kdp = np.where(precip, kdp, 0.0)

    phidp_corr = 2 * dr * np.cumsum(kdp, 1)

    if precip is not None:
        kdp = np.where(precip, kdp, np.nan)
        phidp_corr = np.where(precip, phidp_corr, np.nan)

    return phidp_corr, kdp


@njit(cache=True)
def speckle_filter(data: np.ndarray, mask: np.ndarray, min_dbz: float = -10, min_neighbours: int = 3) -> np.ndarray:
    """
    Apply a speckle filter to radar data to remove noise.

    Parameters
    ----------
    data : np.ndarray
        A 2D array containing the radar reflectivity values.
    mask : np.ndarray
        A 2D array containing the mask values.
    min_dbz : float, optional
        The minimum reflectivity value to consider (default is -10).
    min_neighbours : int, optional
        The minimum number of neighbouring pixels above the threshold to retain a pixel (default is 3).

    Returns
    -------
    np.ndarray
        A 2D array containing the filtered radar reflectivity values.
    """
    ny, nx = data.shape
    copy = np.zeros((ny, nx)) + np.nan

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if data[y][x] <= min_dbz:
                continue

            if mask[y][x] == 1:
                count_mask = (
                    (mask[y - 1, x - 1] == 0)
                    + (mask[y - 1, x] == 0)
                    + (mask[y - 1, x + 1] == 0)
                    + (mask[y, x - 1] == 0)
                    + (mask[y, x + 1] == 0)
                    + (mask[y + 1, x - 1] == 0)
                    + (mask[y + 1, x] == 0)
                    + (mask[y + 1, x + 1] == 0)
                )
                if count_mask > min_neighbours:
                    copy[y][x] = data[y][x]
                    continue

            count = (
                (data[y - 1, x - 1] > min_dbz)
                + (data[y - 1, x] > min_dbz)
                + (data[y - 1, x + 1] > min_dbz)
                + (data[y, x - 1] > min_dbz)
                + (data[y, x + 1] > min_dbz)
                + (data[y + 1, x - 1] > min_dbz)
                + (data[y + 1, x] > min_dbz)
                + (data[y + 1, x + 1] > min_dbz)
            )

            if count >= min_neighbours:
                copy[y][x] = data[y][x]
    return copy



def write_hvar_dset(
    hfile: h5py.File,
    dataset_idx: int,
    data: np.ndarray,
    name: str,
    gain: float = 1.0,
    offset: float = 0.0,
    undetect: float = -9999.0,
    nodata: float = -9999.0,
    dtype: np.dtype = np.float32,
) -> None:
    """
    Write a dataset to an HDF5 file with specified attributes.

    Parameters
    ----------
    hfile : h5py.File
        The HDF5 file object.
    dataset_idx : int
        The index of the dataset to write to.
    data : np.ndarray
        The data array to write.
    name : str
        The name of the quantity.
    gain : float, optional
        The gain attribute (default is 1.0).
    offset : float, optional
        The offset attribute (default is 0.0).
    undetect : float, optional
        The undetect attribute (default is -9999.0).
    nodata : float, optional
        The nodata attribute (default is -9999.0).
    dtype : np.dtype, optional
        The data type for the dataset (default is np.float32).

    Returns
    -------
    None
    """
    data_count = len([k for k in hfile[f"/dataset{dataset_idx}/"].keys() if k.startswith("data")])
    data_id = f"data{data_count + 1}"
    h5_new_field = hfile[f"/dataset{dataset_idx}"].require_group(data_id)
    nrays = hfile[f"/dataset{dataset_idx}/where/"].attrs["nrays"]
    nbins = hfile[f"/dataset{dataset_idx}/where/"].attrs["nbins"]

    h5_data = h5_new_field.create_dataset("data", (nrays, nbins), dtype, compression="gzip")
    assert data.shape == (nrays, nbins), f"Problem with data /dataset{dataset_idx}/ {name}"
    h5_data[:] = data[:nrays, :nbins].astype(dtype)
    pyodim.write_odim_str_attrib(h5_data, "CLASS", "IMAGE")
    pyodim.write_odim_str_attrib(h5_data, "IMAGE_VERSION", "1.2")

    h5_what = h5_new_field.create_group("what")
    pyodim.write_odim_str_attrib(h5_what, "quantity", name)
    h5_what.attrs["gain"] = gain
    h5_what.attrs["offset"] = offset
    h5_what.attrs["undetect"] = undetect
    h5_what.attrs["nodata"] = nodata

    return None


def write_hvar_dset_hydro(hfile: h5py.File, dataset_idx: int, data: np.ndarray, name: str) -> None:
    """
    Write a dataset to an HDF5 file with specified attributes.

    Parameters
    ----------
    hfile : h5py.File
        The HDF5 file object.
    dataset_idx : int
        The index of the dataset to write to.
    data : np.ndarray
        The data array to write.
    name : str
        The name of the quantity.

    Returns
    -------
    None
    """
    data_count = len([k for k in hfile[f"/dataset{dataset_idx}/"].keys() if k.startswith("data")])
    data_id = f"data{data_count + 1}"
    h5_new_field = hfile[f"/dataset{dataset_idx}"].require_group(data_id)
    nrays = hfile[f"/dataset{dataset_idx}/where/"].attrs["nrays"]
    nbins = hfile[f"/dataset{dataset_idx}/where/"].attrs["nbins"]

    h5_data = h5_new_field.create_dataset("data", (nrays, nbins), np.int16, compression="gzip")
    assert data.shape == (nrays, nbins), f"Problem with data /dataset{dataset_idx}/ {name}"
    h5_data[:] = data[:nrays, :nbins].astype(np.int16)
    pyodim.write_odim_str_attrib(h5_data, "CLASS", "IMAGE")
    pyodim.write_odim_str_attrib(h5_data, "IMAGE_VERSION", "1.2")

    x = np.array([1, 0], dtype=np.int16)

    h5_what = h5_new_field.create_group("what")
    pyodim.write_odim_str_attrib(h5_what, "quantity", name)
    h5_what.attrs["gain"] = x[0]
    h5_what.attrs["offset"] = x[1]
    h5_what.attrs["undetect"] = x[1]
    h5_what.attrs["nodata"] = x[1]

    return None


def process_oceanpol(odim_file: str, cal_offset: float = 0.7, zdr_offset: float = 0.7) -> None:
    """
    Process an ODIM file to clean and correct radar data, and write the results to the file.

    Parameters
    ----------
    odim_file : str
        The path to the ODIM file to process.
    cal_offset : float, optional
        The calibration offset for reflectivity (default is 0.7).
    zdr_offset : float, optional
        The offset for differential reflectivity (default is 0.7).

    Returns
    -------
    None
    """
    st = time.time()
    print("Running UNRAVEL.")
    radarlist, hfile = pyodim.read_write_odim(odim_file, read_write=True, lazy_load=False)

    try:
        # Resolve field names once for the whole volume (raises if a required
        # field is missing in any sweep).
        fields = resolve_fields(radarlist)
        sqi_name = fields["SQI"]
        vel_name = fields["VRAD"]

        condition = (sqi_name, "lower", 0.3) if sqi_name else None

        # Censor incoherent (noise) velocity gates before dealiasing. UNRAVEL
        # then only processes coherent echo: far faster on near-empty volumes
        # and free of the random-noise artefacts it would otherwise dealias.
        # Coherent clear-air returns (high SQI) are preserved.
        for sweep in radarlist:
            good_vel = get_velocity_mask(
                sweep[vel_name].values,
                sweep[sqi_name].values if sqi_name else None,
            )
            sweep[vel_name].values[~good_vel] = np.nan

        unfolded_vel = unravel.unravel_3D_pyodim(
            radarlist,
            vel_name=vel_name,
            condition=condition,
            read_write=False,
            output_vel_name="VRADDH",
        )
        unravel_vel = {r.attrs["id"]: r.VRADDH for r in unfolded_vel}
        mt = time.time()
        print(f"UNRAVEL done. Processing file {odim_file}.")

        radar = radarlist[0]
        date = pd.Timestamp(radar.time.values[0])
        lon = radar.attrs["longitude"]
        lat = radar.attrs["latitude"]
        southern_ocean = lat < -40

        geo_h_profile, temp_profile = temperature.get_volume_temperature_profile(
            date, lat, lon, radarlist, fields
        )

        for radar in radarlist:
            dataset_idx = int(radar.attrs["id"].strip("dataset"))

            r = radar.range.values
            dbz = radar[fields["DBZH"]].values
            rhohv = radar[fields["RHOHV"]].values
            zdr = radar[fields["ZDR"]].values
            phidp = radar[fields["PHIDP"]].values
            snr = radar[fields["SNR"]].values
            vraddh = unravel_vel[radar.attrs["id"]]

            temps = temperature.interp_temperature(geo_h_profile, temp_profile, radar.z.values)
            mask = get_hydrometeor_mask(dbz, phidp, rhohv)

            refl = np.ma.masked_where(mask, radar[fields["TH"]].values).copy().filled(np.nan)
            dbz_clean = speckle_filter(refl.copy(), mask) + cal_offset
            dbz_clean = np.ma.masked_invalid(dbz_clean)

            phidp = np.ma.masked_where(np.isnan(dbz_clean), radar[fields["PHIDP"]])
            precip = get_precip_mask(rhohv, snr, np.ma.filled(dbz_clean, np.nan))
            phidp_corr, kdp = get_phidp(r, phidp, refl, temps, precip=precip)

            pia = atten.correct_attenuation(r, dbz_clean, phidp_corr, temps)
            dbz_clean2 = dbz_clean + pia

            zdr_clean = correct_zdr(zdr + zdr_offset, snr)
            zdr_clean[np.isnan(dbz_clean)] = np.nan

            nw, d0 = hydro.get_dsd_estimate(dbz_clean, zdr_clean, temps)
            snowfall = hydro.get_snowfall_estimate(dbz_clean, kdp, temps)
            rainfall = hydro.get_rainfall_estimate(dbz_clean, zdr_clean, kdp, temps, southern_ocean)
            scores = hydro.compute_hid(dbz_clean, zdr_clean, kdp, rhohv, temps)

            h5_kwargs = {"gain": 1.0, "offset": 0.0, "nodata": -9999.0, "dtype": np.float32}
            write_hvar_dset(hfile, dataset_idx, dbz_clean, "DBZH_CLEAN", undetect=-32.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, dbz_clean2, "DBZH_CLEAN2", undetect=-32.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, pia, "PIA", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, zdr_clean, "ZDR_CLEAN", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, vraddh, "VRADDH", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, phidp_corr, "PHIDP_PHIDO", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, kdp, "KDP_PHIDO", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, rainfall, "RAIN", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, nw, "NW", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, d0, "D0", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, snowfall, "SNOW", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(hfile, dataset_idx, temps, "TEMPERATURE", undetect=-9999.0, **h5_kwargs)
            write_hvar_dset(
                hfile,
                dataset_idx,
                scores.astype(np.int16),
                "HID",
                gain=np.int16(1),
                offset=np.int16(0),
                undetect=np.int16(0),
                nodata=np.int16(0),
                dtype=np.int16,
            )
    finally:
        # Always release the ODIM handle, even if processing raised partway
        # through (e.g. UNRAVEL crash), so the file is not left open/locked.
        hfile.close()

    et = time.time()
    print(f"{odim_file} processed in {et-st:.3f}s total (unravel: {mt-st:.3f}s).")

    return None
