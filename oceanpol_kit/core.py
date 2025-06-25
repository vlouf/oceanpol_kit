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

    area_std
    correct_zdr
    get_dsd_estimate
    get_hydrometeor_mask
    get_phidp
    get_temperature
    speckle_filter
    read_era5_temperature
    write_hvar_dset
    process_file
"""

import os
import glob
import time
from typing import List, Tuple

import h5py
import pyodim
import unravel
import numpy as np
import pandas as pd
import xarray as xr

from numba import jit
from scipy import interpolate
from phido import phidp_to_kdp
from . import hydro
from . import atten


@jit
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


def get_phidp(
    r: np.ndarray, phidp: np.ndarray, refl: np.ndarray, temperature: np.ndarray, window: List[int] = [3, 7]
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

    phidp_corr = 2 * dr * np.cumsum(kdp, 1)

    return phidp_corr, kdp


def get_temperature(date: pd.Timestamp, lat: float, lon: float, gates_z: np.ndarray) -> np.ndarray:
    """
    Retrieve the temperature profile for a given date, latitude, and longitude.

    Parameters
    ----------
    date : str
        The date for which to retrieve the temperature profile.
    lat : float
        The latitude of the location.
    lon : float
        The longitude of the location.
    gates_z : np.ndarray
        A 1D array containing the gate heights.

    Returns
    -------
    fld np.ndarray
        A 1D array containing the temperature profile at the specified gate heights.
    """
    geo_h_profile, temp_profile = read_era5_temperature(date, lon, lat)
    f_interp = interpolate.interp1d(geo_h_profile, temp_profile, bounds_error=False, fill_value=9999)
    fld = np.ma.masked_equal(f_interp(gates_z), 9999) - 273.15
    return fld


@jit
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
    return data


def read_era5_temperature(date: pd.Timestamp, longitude: float, latitude: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the temperature profile from ERA5 data for a given date, longitude, and latitude.

    Parameters
    ----------
    date : pd.Timestamp
        Date for extraction.
    longitude : float
        Radar longitude.
    latitude : float
        Radar latitude.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - z: A 1D array of heights in meters.
        - temperature: A 1D array of temperatures in Kelvin.
    """
    # Generate filename.
    era5_root = "/g/data/rt52/era5/pressure-levels/reanalysis/"
    # Build file paths
    month_str = date.month
    year_str = date.year
    era5_file = glob.glob(f"{era5_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str:02}*.nc")[0]

    if not os.path.isfile(era5_file):
        raise FileNotFoundError(f"{era5_file} not found.")

    # Get temperature
    dset = xr.open_dataset(era5_file)
    nset = dset.sel(longitude=longitude, latitude=latitude, time=date, method="nearest")
    temp_profile = nset.t.values
    level = nset.level.values
    geo_h_profile = -2494.3 / 0.218 * np.log(level / 1013.15)
    zmin = geo_h_profile.min()
    temp_ground = temp_profile[np.argmin(geo_h_profile)] + 0.0065 * zmin

    # Extrapolate sea level temperature.
    geo_h_profile = np.append(geo_h_profile, 0)
    temp_profile = np.append(temp_profile, temp_ground)

    return geo_h_profile, temp_profile


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
        The calibration offset for reflectivity (default is -0.7).
    zdr_offset : float, optional
        The offset for differential reflectivity (default is -0.7).

    Returns
    -------
    None
    """
    st = time.time()
    print("Running UNRAVEL.")
    unfolded_vel = unravel.unravel_3D_pyodim(
        odim_file, vel_name="VRAD", condition=("SQI", "lower", 0.3), read_write=False, output_vel_name="VRADDH"
    )
    unravel_vel = {r.attrs["id"]: r.VRADDH for r in unfolded_vel}

    mt = time.time()
    print(f"UNRAVEL done. Processing file {odim_file}.")
    radarlist, hfile = pyodim.read_write_odim(odim_file, read_write=True, lazy_load=False)

    radar = radarlist[0]
    date = pd.Timestamp(radar.time.values[0])
    lon = radar.attrs["longitude"]
    lat = radar.attrs["latitude"]
    southern_ocean = lat < -40

    for radar in radarlist:
        dataset_idx = int(radar.attrs["id"].strip("dataset"))

        r = radar.range.values
        dbz = radar.DBZH.values
        rhohv = radar.RHOHV.values
        zdr = radar.ZDR.values
        phidp = radar.PHIDP.values
        snr = radar.SNR.values
        vraddh = unravel_vel[radar.attrs["id"]]

        temps = get_temperature(date, lat, lon, radar.z.values)
        mask = get_hydrometeor_mask(dbz, phidp, rhohv)

        refl = np.ma.masked_where(mask, radar.TH.values).copy().filled(np.nan)
        dbz_clean = speckle_filter(refl.copy(), mask) + cal_offset
        dbz_clean = np.ma.masked_invalid(dbz_clean)

        phidp = np.ma.masked_where(np.isnan(dbz_clean), radar.PHIDP)
        phidp_corr, kdp = get_phidp(r, phidp, refl, temps)

        pia = atten.correct_attenuation(r, dbz_clean, phidp_corr, temps)
        dbz_clean2 = dbz_clean + pia

        zdr_clean = correct_zdr(zdr + zdr_offset, snr)
        zdr_clean[np.isnan(dbz_clean)] = np.nan

        nw, d0 = hydro.get_dsd_estimate(dbz_clean, zdr_clean)
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

    hfile.close()
    et = time.time()

    print(f"{odim_file} processed in {et-st:.3f}s total (unravel: {mt-st:.3f}s).")

    return None
