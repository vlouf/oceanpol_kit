"""
Process OceanPOL radar data for a given voyage.
Applies QC, attenuation correction, DSD retrieval, and gridding.

Usage:
    python process_opol.py -v IN2023_V06 [-j 16]

@author: Valentin Louf
"""
import os
import io
import sys
import glob
import time
import shutil
import logging
import argparse
import traceback

from pathlib import Path
from typing import Tuple, List, Optional
from multiprocessing import Pool

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================
SOURCE_ROOT = "/g/data/hj10/admin/incoming_globus/OPOL"
DESTINATION_ROOT = "/scratch/kl02/vhl548/OPOL"

DBZ_CALIB = """date,offset
2000-01-01,1.4
2020-07-31,1.7
2020-11-30,2.3
2021-01-31,2.5
2021-04-30,1.5
2022-05-31,0.0
2024-06-30,3.2
2026-01-01,1.7"""

ZDR_CALIB = """date,offset
2000-01-01,0.5
2020-01-01,0.5
2021-01-01,0.75
2022-01-01,1.0
2023-01-01,1.0
2023-05-01,0.5
2024-06-30,0.5"""


# =====================================================================
# Logging
# =====================================================================
def get_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("process_opol")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =====================================================================
# Calibration
# =====================================================================
def get_calib_offset(date: pd.Timestamp) -> Tuple[float, float]:
    """Return (dbz_offset, zdr_offset) for a given date."""
    dbz_f = pd.read_csv(io.StringIO(DBZ_CALIB), sep=",", parse_dates=["date"], index_col="date")
    zdr_f = pd.read_csv(io.StringIO(ZDR_CALIB), sep=",", parse_dates=["date"], index_col="date")

    try:
        calib_dbz = dbz_f.resample("D").ffill().loc[date].values[0]
        calib_zdr = zdr_f.resample("D").ffill().loc[date].values[0]
    except KeyError:
        if date > dbz_f.index[-1]:
            calib_dbz = dbz_f.offset.iloc[-1]
            calib_zdr = zdr_f.offset.iloc[-1]
        else:
            calib_dbz, calib_zdr = 0.0, 0.0

    return float(calib_dbz), float(calib_zdr)


# =====================================================================
# File list
# =====================================================================
def get_flist(voyage: str) -> List[Tuple[str, str, str]]:
    """
    Build the list of (src_hdf, dst_h5, dst_nc) tuples for all unprocessed files.
    src_hdf : original read-only file on gdata/hj10
    dst_h5  : writable copy on scratch (input to process_oceanpol)
    dst_nc  : gridded output on scratch
    """
    pattern = os.path.join(SOURCE_ROOT, voyage, "hdf5", "**", "*PPIVol*.hdf")
    src_files = sorted(glob.glob(pattern, recursive=True))

    if not src_files:
        raise FileNotFoundError(f"No source files found matching: {pattern}")

    flist = []
    for src in src_files:
        src_path = Path(src)
        date_dir = src_path.parent.name
        stem = src_path.stem

        dst_h5 = Path(DESTINATION_ROOT, voyage, "hdf5", date_dir, stem).with_suffix(".h5")
        dst_nc = Path(DESTINATION_ROOT, voyage, "grid", date_dir, stem).with_suffix(".nc")

        if dst_h5.exists() and dst_nc.exists():
            continue

        dst_h5.parent.mkdir(parents=True, exist_ok=True)
        dst_nc.parent.mkdir(parents=True, exist_ok=True)

        flist.append((str(src), str(dst_h5), str(dst_nc)))

    return flist


# =====================================================================
# Processing
# =====================================================================
def process_file(src: str, dst_h5: str, dst_nc: str) -> Optional[str]:
    """
    Copy src to dst_h5, run QC + gridding, return dst_nc on success.
    """
    if Path(dst_h5).exists() and Path(dst_nc).exists():
        return dst_nc

    date = pd.Timestamp(Path(src).stem.split("-")[2])  # e.g. 20231019 from filename
    calib_dbz, calib_zdr = get_calib_offset(date)

    shutil.copyfile(src, dst_h5)

    oceanpol_kit.process_oceanpol(dst_h5, cal_offset=calib_dbz, zdr_offset=calib_zdr)
    oceanpol_kit.grid_opol(dst_h5, dst_nc)

    if not Path(dst_nc).exists():
        raise FileNotFoundError(f"Gridded output not created: {dst_nc}")

    return dst_nc


def process_file_safe(args: Tuple[str, str, str]) -> Optional[str]:
    """Multiprocessing wrapper — catches all exceptions and logs them."""
    src, dst_h5, dst_nc = args
    try:
        result = process_file(src, dst_h5, dst_nc)
        return result
    except Exception:
        # Write traceback to a sidecar .err file next to the output
        err_file = Path(dst_h5).with_suffix(".err")
        with open(err_file, "w") as f:
            f.write(f"Source: {src}\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        return None


# =====================================================================
# Main
# =====================================================================
def main(voyage: str, nproc: int) -> None:
    log_file = os.path.join(DESTINATION_ROOT, voyage, f"process_{voyage}.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file)

    logger.info(f"Voyage: {voyage}  |  CPUs: {nproc}")
    logger.info(f"Source root : {SOURCE_ROOT}")
    logger.info(f"Output root : {DESTINATION_ROOT}")

    try:
        flist = get_flist(voyage)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not flist:
        logger.info("All files already processed. Nothing to do.")
        return

    logger.info(f"Files to process: {len(flist)}")
    t0 = time.time()

    with Pool(nproc) as pool:
        async_results = [pool.apply_async(process_file_safe, (args,)) for args in flist]

        n_ok, n_fail, n_timeout = 0, 0, 0
        for i, ar in enumerate(async_results):
            src = flist[i][0]
            try:
                result = ar.get(timeout=600)
                if result is not None:
                    n_ok += 1
                    logger.info(f"[{i+1}/{len(flist)}] OK  {Path(src).name}")
                else:
                    n_fail += 1
                    logger.warning(f"[{i+1}/{len(flist)}] FAIL {Path(src).name}")
            except TimeoutError:
                n_timeout += 1
                logger.error(f"[{i+1}/{len(flist)}] TIMEOUT {Path(src).name}")
            except Exception as e:
                n_fail += 1
                logger.error(f"[{i+1}/{len(flist)}] ERROR {Path(src).name}: {e}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s  —  ok={n_ok}  fail={n_fail}  timeout={n_timeout}")


if __name__ == "__main__":
    import oceanpol_kit

    parser = argparse.ArgumentParser(description="Process OceanPOL radar data for a voyage.")
    parser.add_argument("-v", "--voyage", type=str, required=True, help="Voyage ID e.g. IN2023_V06")
    parser.add_argument("-j", "--ncpu", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    main(args.voyage, args.ncpu)
