import os
import sys
import re
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


# def plot_opol(opol_file: str, out_png: str) -> None:
#     cartopy.config["pre_existing_data_dir"] = "/g/data/kl02/vhl548/.local/share/cartopy/"
#     rsets = pyodim.read_odim(opol_file)
#     radar = rsets[1].compute()

#     hid_types = [
#         "Unclassified",
#         "Drizzle",
#         "Rain",
#         "Ice Crystals",
#         "Aggregates",
#         "Wet Snow",
#         "Vertical Ice",
#         "Low-Density Graupel",
#         "High-Density Graupel",
#         "Hail",
#         "Big Drops"
#     ]

#     latlims = (radar.latitude.values.min(), radar.latitude.values.max())
#     lonlims = (radar.longitude.values.min(), radar.longitude.values.max())
#     lon0 = radar.attrs["longitude"]
#     lat0 = radar.attrs["latitude"]
#     date = radar.attrs["date"]
#     proj = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +units=m")

#     labels = [
#         "Total Reflectivity (TH)",
#         "Cleaned Reflectivity (DBZH_CLEAN)",
#         "Atten. corr. Reflectivity (DBZH_CLEAN2)",
#         "Doppler Velocity (VRADDH)",
#         "Differential Reflectivity (ZDR_CLEAN)",
#         "Correlation Coefficient (RHOHV)",
#         "Differential Phase (PHIDP)",
#         "Processed Differential Phase (PHIDP_PHIDO)",
#         "Specific Differential Phase (KDP_PHIDO)",
#         "Rainfall Rate (RAIN)",
#         "Median Volume Diameter (D0)",
#         "Hydrometeor Identification (HID)"
#     ]

#     labels_with_units = [
#         "Total Reflectivity (dBZ)",
#         "Cleaned Reflectivity (dBZ)",
#         "Atten. corr. Reflectivity (dBZ)",
#         "Doppler Velocity (m/s)",
#         "Differential Reflectivity (dB)",
#         "Correlation Coefficient (unitless)",
#         "Differential Phase (degrees)",
#         "Processed Differential Phase (degrees)",
#         "Specific Differential Phase (degrees/km)",
#         "Rainfall Rate (mm/h)",
#         "Median Volume Diameter (mm)",
#         "Hydrometeor Identification (index)"
#     ]

#     fig = pplt.figure()
#     axs = fig.subplots(nrows=3, ncols=4, proj="aeqd", proj_kw={"central_longitude": radar.attrs["longitude"], "central_latitude": radar.attrs["latitude"]})
#     im = [None] * len(axs)

#     im[0] = axs[0].pcolormesh(radar.longitude, radar.latitude, radar.TH, cmap=cm.cm_colorblind.HomeyerRainbow, vmin=-15, vmax=65)
#     im[1] = axs[1].pcolormesh(radar.longitude, radar.latitude, radar.DBZH_CLEAN, cmap=cm.cm_colorblind.HomeyerRainbow, vmin=-15, vmax=65)
#     im[2] = axs[2].pcolormesh(radar.longitude, radar.latitude, radar.DBZH_CLEAN2, cmap=cm.cm_colorblind.HomeyerRainbow, vmin=-15, vmax=65)
#     im[3] = axs[3].pcolormesh(radar.longitude, radar.latitude, radar.VRADDH, cmap=cm.cm.NWSVel, vmin=-32, vmax=32, levels=32)
#     im[4] = axs[4].pcolormesh(radar.longitude, radar.latitude, radar.ZDR_CLEAN - 1, cmap=cm.cm.RefDiff, vmin=-2, vmax=8, levels=32)
#     im[5] = axs[5].pcolormesh(radar.longitude, radar.latitude, radar.RHOHV, cmap=cm.cm.RefDiff, vmin=0.5, vmax=1.05, levels=32)
#     im[6] = axs[6].pcolormesh(radar.longitude, radar.latitude, radar.PHIDP, cmap=cm.cm.Wild25, vmin=-180, vmax=180, levels=32)
#     im[7] = axs[7].pcolormesh(radar.longitude, radar.latitude, radar.PHIDP_PHIDO, cmap=cm.cm.Wild25, vmin=-180, vmax=180, levels=32)
#     im[8] = axs[8].pcolormesh(radar.longitude, radar.latitude, radar.KDP_PHIDO, cmap=cm.cm.Theodore16, vmin=-2, vmax=5, levels=32)
#     im[9] = axs[9].pcolormesh(radar.longitude, radar.latitude, radar.RAIN, cmap=cm.cm.RRate11, norm=LogNorm(0.01, 100), levels=25)
#     im[10] = axs[10].pcolormesh(radar.longitude, radar.latitude, radar.D0, cmap=cm.cm.RRate11, vmin=0, vmax=3, levels=32)
#     im[11] = axs[11].pcolormesh(radar.longitude, radar.latitude, np.ma.masked_invalid(radar.HID).filled(0), cmap=cm.cm.Theodore16, vmin=0, vmax=10, levels=10)

#     for i in range(len(axs)):
#         axs[i].format(title=labels[i])
#         if i != 11:
#             fig.colorbar(im[i], ax=axs[i], label=labels_with_units[i])
#         else:
#             fig.colorbar(im[i], ax=axs[i], ticklabels=hid_types)
#         th = np.linspace(0, 6.28)
#         for r in [50e3, 100e3, 150e3]:
#             x0 = r * np.cos(th)
#             y0 = r * np.sin(th)
#             xlo, yla = proj(x0, y0, inverse=True)
#             axs[i].plot(xlo, yla, "k", linewidth=.25, alpha=0.5)

#     axs.format(
#         abc=True,
#         suptitle=date,
#         coast=True,
#         latlim=latlims,
#         lonlim=lonlims,
#         reso="hi",
#         latlines=1,
#         lonlines=1,
#         lonlabels='b',
#         latlabels='l'
#     )

#     pl.savefig(out_png, dpi=200)
#     pl.close()
#     return None


def process_file(src_file: str, out_h5: str, out_nc: str, out_png: str, date: str):
    if os.path.exists(out_h5) and os.path.exists(out_nc):
        print(f"{out_nc} already exists. Doing nothing.")
        return None

    dtime = pd.Timestamp(date)
    calib_dbz, calib_zdr = get_calib_offset(dtime)    
    
#     with open(src_file, 'rb') as fid:
#         memory_file = io.BytesIO(fid.read())

#     print(f"Processing {src_file}")        
    import shutil
    shutil.copyfile(src_file, out_h5)
    oceanpol_kit.process_oceanpol(out_h5, cal_offset=calib_dbz, zdr_offset=calib_zdr)
#     memory_file.seek(0)
#     with open(out_h5, "wb") as fid:
#         fid.write(memory_file.read())

#     memory_file.seek(0)
    _ = oceanpol_kit.grid_opol(out_h5, out_nc)
        
#         memory_file.seek(0)
#         plot_opol(memory_file, out_png)    

    if os.path.exists(out_nc):
        print(f"{out_nc} written, process complete.")
    else:
        raise FileNotFoundError(f"{out_nc} not created.")

    return None
   
    
def get_flist(source_root: str, destination_root: str, voyage, date: str) -> List[str]:
    flist = []
    for hdf_file in Path(source_root).rglob(f"{date}/*PPIVol*.hdf"):
        rel_path = hdf_file.relative_to(source_root)
        filename = rel_path.name
        
        date_match = re.search(r"\d{8}", str(rel_path))
        date_dir = date_match.group() if date_match else date
        
        # Create output file names
        h5_file = Path(destination_root, voyage, "hdf5", date_dir, filename).with_suffix('.h5')
        h5_file.parent.mkdir(parents=True, exist_ok=True)

        nc_file = Path(destination_root, voyage, "grid", date_dir, filename).with_suffix('.nc')
        nc_file.parent.mkdir(parents=True, exist_ok=True)

        if h5_file.exists() and nc_file.exists():
            continue

        png_dest = Path(destination_root, voyage, "plots", date_dir, filename).with_suffix('.png')
        png_dest.parent.mkdir(parents=True, exist_ok=True)

        flist.append((str(hdf_file), str(h5_file), str(nc_file), str(png_dest), date))
        
    if len(flist) == 0:
        raise FileNotFoundError(f"No file found in {source_root}")

    return flist


def process_file_safe(src_file, out_h5, out_nc, out_png, date):
    """Wrapper that catches exceptions without killing the pool"""
    try:
        return process_file(src_file, out_h5, out_nc, out_png, date)
    except Exception:
        print(f"Problem with file {src_file}.")
        traceback.print_exc()
        return None


def main(root, destination_root, voyage, nproc):
    # source_root = os.path.join(root, voyage)
    source_root = root
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"{source_root} does not exist.")

    for date in os.listdir(os.path.join(source_root)):
        if not os.path.isdir(os.path.join(source_root, date)):
            continue
        
        flist = get_flist(source_root, destination_root, voyage, date)
        print(f"Found {len(flist)} files to process at date {date}.")
        st = time.time()
        
        with Pool(nproc) as pool:
            # Submit all tasks asynchronously
            async_results = [
                pool.apply_async(process_file_safe, args) 
                for args in flist
            ]

            # Collect results
            results = []
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=600)
                    results.append(result)
                except TimeoutError:
                    print(f"Timeout on file {flist[i][0]}")
                    results.append(None)
                except Exception as e:
                    print(f"Error collecting result for {flist[i][0]}: {e}")
                    results.append(None)

                    delta = time.time() - st
                    print(f"{date} completed in {delta:.2f}s.")
                except FileNotFoundError:
                    traceback.print_exc()
                    continue

    return None


if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as pl
#     import ultraplot as pplt    

    import oceanpol_kit
#     import cartopy
    import pyodim    
#     import cmweather as cm
#     from matplotlib.colors import LogNorm

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voyage", type=str, required=True, help="RV Voyage ID", dest="voyage")
    parser.add_argument("-j", "--ncpu", type=int, required=False, default=32, help="Number of CPUs", dest="ncpu")

    args = parser.parse_args()
    voyage = args.voyage
    nproc = args.ncpu

    # root = "/g/data/hj10/admin/opol/level_1/"
    root = f"/g/data/hj10/admin/incoming_globus/OPOL/{voyage.upper()}/hdf5"
    if not os.path.exists(root):
        print(root)
        print("Input directory does not exist")
        sys.exit(1)
    destination_root = "/scratch/kl02/vhl548/OPOL"
    main(root, destination_root, voyage, nproc)
