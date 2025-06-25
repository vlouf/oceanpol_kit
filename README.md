# oceanpol_kit

**oceanpol_kit** is a Python toolkit for processing radar data, with a focus on oceanic and polarimetric radar applications. It provides functions for calculations such as differential reflectivity correction, drop size distribution estimation, hydrometeor classification, attenuation correction, and more.

## Features

- Differential reflectivity (ZDR) correction
- Drop size distribution (DSD) estimation
- Hydrometeor classification (HID)
- Attenuation correction
- Temperature profile retrieval from ERA5 reanalysis
- Speckle filtering and noise removal
- Support for ODIM HDF5 radar data format
- Integration with external libraries: `pyodim`, `unravel`, `phido`, `h5py`, `xarray`, `numpy`, `pandas`, `scipy`, and `numba`

## Installation

Clone the repository and install the dependencies:

```sh
git clone <repository-url>
cd oceanpol_kit
pip install .
```

Or install using the provided setup.py:

```sh
pip install .
```

## Usage

The main processing function is `process_oceanpol`, which processes an ODIM HDF5 radar file and writes cleaned and derived products back to the file.

Example usage:

```python
from oceanpol_kit.core import process_oceanpol

process_oceanpol("path/to/odim_file.h5")
```

### Key Functions

- `area_std`: Sliding window standard deviation for 2D arrays.
- `correct_zdr`: Noise correction for ZDR.
- `get_hydrometeor_mask`: Hydrometeor mask generation.
- `get_phidp`: PHIDP and KDP calculation.
- `get_temperature`: Retrieve temperature profiles from ERA5.
- `speckle_filter`: Speckle noise filtering.
- `read_era5_temperature`: Read ERA5 temperature profiles.
- `write_hvar_dset`: Write datasets to ODIM HDF5 files.

See core.py for full API documentation.

## Scripts

Example processing scripts are available in the scripts directory.

## Requirements

- Python 3.7+
- h5py
- pyodim
- unravel
- numpy
- pandas
- xarray
- scipy
- numba
- phido

## Data

This toolkit expects radar data in ODIM HDF5 format and ERA5 reanalysis data for temperature profiles.

## License

See LICENSE for details.

## Author

Valentin Louf  

---
For more information, see the docstrings in core.py.