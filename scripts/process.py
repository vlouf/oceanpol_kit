import os
import io
import time
import argparse
import traceback

from pathlib import Path
from typing import Tuple, List
from multiprocessing import Pool

import pyproj
import numpy as np
import pandas as pd


def get_calib_offset(date: pd.Timestamp) -> Tuple[float, float]:
    """
    Get the calibration offset for a given date.

    Parameters
    ----------
    date : pd.Timestamp
        The date to get the calibration offset for.

    Returns
    -------
    Tuple[float, float]
        The calibration offset for reflectivity and differential reflectivity.
    """
    # Define the dates
    dbz_arr = """date,offset
2000-01-01,1.4
2020-07-31,1.7
2020-11-30,2.3
2021-01-31,2.5
2021-04-30,1.5
2022-05-31,0.0
2024-06-30,3.2"""
    zdr_arr = """date,offset
2000-01-01,0.5
2020-01-01,0.5
2021-01-01,0.75
2022-01-01,1.0
2023-01-01,1.0
2023-05-01,0.5
2024-06-30,0.5"""

    dbz_f = pd.read_csv(io.StringIO(dbz_arr), sep=",", parse_dates=["date"], index_col='date')
    zdr_f = pd.read_csv(io.StringIO(zdr_arr), sep=",", parse_dates=["date"], index_col='date')
    try:
        calib_dbz = dbz_f.resample("D").ffill().loc[date].values[0]
        calib_zdr = zdr_f.resample("D").ffill().loc[date].values[0]
    except KeyError:
        print(f"Date {date} not found in the data.")
        if date > dbz_f.index[-1]:            
            calib_dbz = dbz_f.offset[-1]
            calib_zdr = zdr_f.offset[-1]
        else:
            calib_dbz, calib_zdr = 0, 0

    return calib_dbz, calib_zdr


def plot_opol(opol_file: str, out_png: str) -> None:
    cartopy.config["pre_existing_data_dir"] = "/g/data/kl02/vhl548/.local/share/cartopy/"
    rsets = pyodim.read_odim(opol_file)
    radar = rsets[1].compute()

    hid_types = [
        "Unclassified",
        "Drizzle",
        "Rain",
        "Ice Crystals",
        "Aggregates",
        "Wet Snow",
        "Vertical Ice",
        "Low-Density Graupel",
        "High-Density Graupel",
        "Hail",
        "Big Drops"
    ]

    latlims = (radar.latitude.values.min(), radar.latitude.values.max())
    lonlims = (radar.longitude.values.min(), radar.longitude.values.max())
    lon0 = radar.attrs["longitude"]
    lat0 = radar.attrs["latitude"]
    date = radar.attrs["date"]
    proj = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=m")

    labels = [
        "Total Reflectivity (TH)",
        "Cleaned Reflectivity (DBZH_CLEAN)",
        "Atten. corr. Reflectivity (DBZH_CLEAN2)",
        "Doppler Velocity (VRADDH)",
        "Differential Reflectivity (ZDR_CLEAN)",
        "Correlation Coefficient (RHOHV)",
        "Differential Phase (PHIDP)",
        "Processed Differential Phase (PHIDP_PHIDO)",
        "Specific Differential Phase (KDP_PHIDO)",
        "Rainfall Rate (RAIN)",
        "Median Volume Diameter (D0)",
        "Hydrometeor Identification (HID)"
    ]

    labels_with_units = [
        "Total Reflectivity (dBZ)",
        "Cleaned Reflectivity (dBZ)",
        "Atten. corr. Reflectivity (dBZ)",
        "Doppler Velocity (m/s)",
        "Differential Reflectivity (dB)",
        "Correlation Coefficient (unitless)",
        "Differential Phase (degrees)",
        "Processed Differential Phase (degrees)",
        "Specific Differential Phase (degrees/km)",
        "Rainfall Rate (mm/h)",
        "Median Volume Diameter (mm)",
        "Hydrometeor Identification (index)"
    ]

    fig = pplt.figure()
    axs = fig.subplots(nrows=3, ncols=4, proj="aeqd", proj_kw={"lon_0": lon0, "lat_0":lat0})
    im = [None] * len(axs)

    im[0] = axs[0].pcolormesh(radar.longitude, radar.latitude, radar.TH, cmap="HomeyerRainbow", vmin=-15, vmax=65)
    im[1] = axs[1].pcolormesh(radar.longitude, radar.latitude, radar.DBZH_CLEAN, cmap="HomeyerRainbow", vmin=-15, vmax=65)
    im[2] = axs[2].pcolormesh(radar.longitude, radar.latitude, radar.DBZH_CLEAN2, cmap="HomeyerRainbow", vmin=-15, vmax=65)
    im[3] = axs[3].pcolormesh(radar.longitude, radar.latitude, radar.VRADDH, cmap="NWSVel", vmin=-32, vmax=32, levels=32)
    im[4] = axs[4].pcolormesh(radar.longitude, radar.latitude, radar.ZDR_CLEAN, cmap="RefDiff", vmin=-2, vmax=8, levels=32)
    im[5] = axs[5].pcolormesh(radar.longitude, radar.latitude, radar.RHOHV, cmap="RefDiff", vmin=0.5, vmax=1.05, levels=32)
    im[6] = axs[6].pcolormesh(radar.longitude, radar.latitude, radar.PHIDP, cmap="Wild25", vmin=-180, vmax=180, levels=32)
    im[7] = axs[7].pcolormesh(radar.longitude, radar.latitude, radar.PHIDP_PHIDO, cmap="Wild25", vmin=-180, vmax=180, levels=32)
    im[8] = axs[8].pcolormesh(radar.longitude, radar.latitude, radar.KDP_PHIDO, cmap="Theodore16", vmin=-2, vmax=5, levels=32)
    im[9] = axs[9].pcolormesh(radar.longitude, radar.latitude, radar.RAIN, cmap="RRate11", vmin=0, vmax=50, levels=25)
    im[10] = axs[10].pcolormesh(radar.longitude, radar.latitude, radar.D0, cmap="RRate11", vmin=0, vmax=3, levels=32)
    im[11] = axs[11].pcolormesh(radar.longitude, radar.latitude, np.ma.masked_invalid(radar.HID).filled(0), cmap="Theodore16", vmin=0, vmax=10, levels=10)

    for i in range(len(axs)):
        axs[i].format(title=labels[i])
        if i != 11:
            fig.colorbar(im[i], ax=axs[i], label=labels_with_units[i])
        else:
            fig.colorbar(im[i], ax=axs[i], ticklabels=hid_types)

    th = np.linspace(0, 6.28)
    for r in [50e3, 100e3, 150e3]:
        x0 = r * np.cos(th)
        y0 = r * np.sin(th)
        xlo, yla = proj(x0, y0, inverse=True)
        axs.plot(xlo, yla, "k", linewidth=.25, alpha=0.5)

    axs.format(
        abc=True,
        suptitle=date,
        coast=True,
        latlim=latlims,
        lonlim=lonlims,
        reso="hi",
        latlines=1,
        lonlines=1,
        lonlabels='b',
        latlabels='l'
    )

    pl.savefig(out_png, dpi=200)
    pl.close()
    return None


def process_file(src_file: str, out_h5: str, out_nc: str, out_png: str, date: str):
    if os.path.exists(out_h5):
        print(f"{out_h5} already exists. Doing nothing.")
        return None

    dtime = pd.Timestamp(date)
    calib_dbz, calib_zdr = get_calib_offset(dtime)
    try:
        with open(src_file, 'rb') as fid:
            memory_file = io.BytesIO(fid.read())

        oceanpol_kit.process_oceanpol(memory_file, cal_offset=calib_dbz, zdr_offset=calib_zdr)
        memory_file.seek(0)
        with open(out_h5, "wb") as fid:
            fid.write(memory_file.read())

        memory_file.seek(0)
        _ = oceanpol_kit.grid_opol(memory_file, out_nc)

        memory_file.seek(0)
        plot_opol(memory_file, out_png)
    except Exception:
        print(f"Problem with file {src_file}.")
        traceback.print_exc()

    return None


def get_flist(source_root: str, destination_root: str, date: str) -> List[str]:
    flist = []
    for src_path in Path(source_root).rglob(f"**/hdf5/{date}/*PPIVol*.hdf"):
        # Determine the relative path from the source root
        relative_path = src_path.relative_to(source_root)

        # Extract the date from the relative path
        date_subdir = relative_path.parts[-2]

        # Construct the new destination path for .h5 file
        dest_path_h5 = Path(destination_root, voyage, relative_path)
        dest_path_h5 = dest_path_h5.with_suffix('.h5')
        dest_path_h5.parent.mkdir(parents=True, exist_ok=True)

        # Construct the new destination path for .nc file in "grid" directory with date subdirectory
        dest_path_nc = Path(destination_root, voyage, "grid", date_subdir, relative_path.name).with_suffix('.nc')
        dest_path_nc.parent.mkdir(parents=True, exist_ok=True)

        # Construct the new destination path for .png file in "plots" directory with date subdirectory
        dest_path_png = Path(destination_root, voyage, "plots", date_subdir, relative_path.name).with_suffix('.png')
        dest_path_png.parent.mkdir(parents=True, exist_ok=True)
        flist.append((str(src_path), str(dest_path_h5), str(dest_path_nc), str(dest_path_png), date))

    if len(flist) == 0:
        raise FileNotFoundError(f"No file found in {src_path}")

    return flist


def main(root, destination_root, voyage, nproc):
    source_root = os.path.join(root, voyage)
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"{source_root} does not exist.")

    for date in os.listdir(os.path.join(source_root, "hdf5")):
        if not os.path.isdir(os.path.join(source_root, "hdf5", date)):
            continue
        try:
            flist = get_flist(source_root, destination_root, date)
            print(f"Found {len(flist)} files to process at date {date}.")
            st = time.time()
            # bag = db.from_sequence(flist).starmap(process_file)
            # bag.compute(nproc=nproc)
            with Pool(nproc) as pool:
                _ = pool.starmap(process_file, flist)
            delta = time.time() - st
            print(f"{date} completed in {delta:.2f}s.")
        except FileNotFoundError:
           traceback.print_exc()
           continue

    return None


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pl
    import ultraplot as pplt    
    import pyart  # Colormaps.

    import oceanpol_kit
    import cartopy
    import pyodim    

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voyage", type=str, required=True, help="RV Voyage ID", dest="voyage")
    parser.add_argument("-j", "--ncpu", type=int, required=False, default=32, help="Number of CPUs", dest="ncpu")

    args = parser.parse_args()
    voyage = args.voyage
    nproc = args.ncpu

    root = "/g/data/hj10/admin/incoming_globus/OPOL"
    destination_root = "/scratch/kl02/vhl548/OPOL"
    main(root, destination_root, voyage, nproc)
