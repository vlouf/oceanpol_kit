'''
Gridding code for oceanpol ODIM files.

@title: oceanpol_kit
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology
@date: 21/03/2025

.. autosummary::
    :toctree: generated/

    compress_data
    get_var_metadata
    get_var_cct
    coverage_content_type
    correct_standard_name
    rename_field_names
    get_metadata
    to_compress
    grid_opol
'''
import uuid
import cftime
from typing import Dict, Union
from datetime import datetime

import pyart
import numpy as np
import xarray as xr
from . import echo_class


def compress_data(nset: xr.Dataset) -> xr.Dataset:
    """
    DType compression of the data. Transform the data to uint8 or uint16 and
    set the "scale_factor" and "add_offset" in the Dataset.
    """
    refl_labels = ["NW", "total_power", "total_power_v", "corrected_reflectivity", "attenuation_corrected_reflectivity", "reflectivity", "reflectivity_v"]
    for r in refl_labels:
        nset[r].values = to_compress(nset, r, 0.5, -32, -32)
        nset[r].attrs["scale_factor"] = 0.5
        nset[r].attrs["add_offset"] = -32.0
        nset[r].attrs["_FillValue"] = 0    
    for r in ["signal_to_noise_ratio"]:
        nset[r].values = to_compress(nset, r, 0.5, -0.5, -0.5)
        nset[r].attrs["scale_factor"] = 0.5
        nset[r].attrs["add_offset"] = -0.5
        nset[r].attrs["_FillValue"] = 0 
    for r in ["corrected_differential_reflectivity", "differential_reflectivity", "path_integrated_attenuation", "spectrum_width"]:
        nset[r].values = to_compress(nset, r, 1 / 12.5, -8, -8)
        nset[r].attrs["scale_factor"] = 1 / 12.5
        nset[r].attrs["add_offset"] = -8
        nset[r].attrs["_FillValue"] = 0    
    for r in ["radar_estimated_rain_rate", "radar_estimated_snow_rate"]:
        nset[r].values = to_compress(nset, r, 1 / 20, 0, 0, np.uint16)
        nset[r].attrs["scale_factor"] = 1 / 20
        nset[r].attrs["add_offset"] = 0
        nset[r].attrs["_FillValue"] = 0    
    for r in ["velocity", "corrected_velocity", "temperature"]:
        nset[r].values = to_compress(nset, r, 1 / 12.5, -300, -300, np.uint16)
        nset[r].attrs["scale_factor"] = 1 / 12.5
        nset[r].attrs["add_offset"] = -300
        nset[r].attrs["_FillValue"] = 0    
    for r in ["cross_correlation_ratio", "normalized_coherent_power"]:
        nset[r].values = to_compress(nset, r, 1 / 256, 0, 0, np.uint8)
        nset[r].attrs["scale_factor"] = 1 / 256
        nset[r].attrs["add_offset"] = 0
        nset[r].attrs["_FillValue"] = 0
    for r in ["differential_phase", "corrected_differential_phase", "D0"]:
        nset[r].values = to_compress(nset, r, 1 / 200, -180, -180, np.uint16)
        nset[r].attrs["scale_factor"] = 1 / 200
        nset[r].attrs["add_offset"] = -180
        nset[r].attrs["_FillValue"] = 0
    for r in ["corrected_specific_differential_phase"]:
        nset[r].values = to_compress(nset, r, 1 / 12.5, -8, -8, np.uint16)
        nset[r].attrs["scale_factor"] = 1 / 12.5
        nset[r].attrs["add_offset"] = -8
        nset[r].attrs["_FillValue"] = 0    
    for r in ["D0"]:
        nset[r].values = to_compress(nset, r, 1 / 4000, 0, 0, np.uint16)
        nset[r].attrs["scale_factor"] = 1 / 4000
        nset[r].attrs["add_offset"] = 0
        nset[r].attrs["_FillValue"] = 0    
    return nset


def get_var_metadata(f: str) -> Union[Dict, None]:
    """
    Get the CF-compliant metadata for a variable.
    """
    variables_metadata = {
        "lat": {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "description": "Geographical latitude of the data point.",
            "units": "degrees_north",
        },
        "lon": {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "description": "Geographical longitude of the data point.",
            "units": "degrees_east",
        },
        "cross_correlation_ratio": {
            "standard_name": "cross_correlation_ratio",
            "long_name": "Cross Correlation Ratio",
            "description": "Ratio of cross-polarized to co-polarized power.",
            "units": "1",
        },
        "radar_estimated_snow_rate": {
            "standard_name": "radar_estimated_snow_rate",
            "long_name": "Radar Estimated Snow Rate",
            "description": "Estimated rate of snowfall based on radar measurements.",
            "units": "mm h-1",
        },
        "signal_to_noise_ratio": {
            "standard_name": "signal_to_noise_ratio",
            "long_name": "Signal-to-Noise Ratio",
            "description": "Ratio of the signal power to the noise power.",
            "units": "dB",
        },
        "corrected_differential_phase": {
            "standard_name": "corrected_differential_phase",
            "long_name": "Corrected Differential Phase",
            "description": "Differential phase shift corrected for system errors.",
            "units": "degree",
        },
        "velocity": {
            "standard_name": "radial_velocity",
            "long_name": "Radial Velocity",
            "description": "Velocity of targets towards or away from the radar.",
            "units": "m s-1",
        },
        "corrected_velocity": {
            "standard_name": "corrected_radial_velocity",
            "long_name": "Corrected Radial Velocity",
            "description": "Radial velocity corrected for phase wrapping using UNRAVEL algorithm.",
            "units": "m s-1",
            "references": "Louf, V. et al. (2020): UNRAVEL: A Robust Modular Velocity Dealiasing Technique for Doppler Radar.",
            "doi": "https://doi.org/10.1175/JTECH-D-19-0020.1."
        },
        "spectrum_width": {
            "standard_name": "spectrum_width",
            "long_name": "Spectrum Width",
            "description": "Width of the Doppler velocity spectrum.",
            "units": "m s-1",
        },
        "corrected_specific_differential_phase": {
            "standard_name": "corrected_specific_differential_phase",
            "long_name": "Corrected Specific Differential Phase",
            "description": "Specific differential phase corrected for noise and artifacts.",
            "units": "degree km-1",
        },
        "radar_echo_classification": {
            "standard_name": "radar_echo_classification",
            "long_name": "Radar Echo Classification",
            "description": "0 - Unclassified, 1 - Drizzle, 2 - Rain, 3 - Ice Crystals, 4 - Aggregates, 5 - Wet Snow, 6 - Vertical Ice, 7 - Low-Density Graupel, 8 - High-Density Graupel, 9 - Hail, 10 - Big Drops",            
            "valid_min": 0,
            "valid_max": 10,
            "reference": "Thompson et al., (2014). A dual-polarization radar hydrometeor classification algorithm for winter precipitation",
        },
        "differential_reflectivity": {
            "standard_name": "differential_reflectivity",
            "long_name": "Differential Reflectivity",
            "description": "Ratio of reflectivity in horizontal and vertical polarizations.",
            "units": "dB",
        },
        "normalized_coherent_power": {
            "standard_name": "normalized_coherent_power",
            "long_name": "Normalized Coherent Power",
            "description": "Coherence of the radar signal, normalized between 0 and 1.",
            "units": "1",
        },
        "radar_estimated_rain_rate": {
            "standard_name": "radar_estimated_rain_rate",
            "long_name": "Radar Estimated Rain Rate",
            "description": "Rainfall rate estimated from radar reflectivity.",
            "units": "mm h-1",
            "reference": "Aragon et al. (2024). Characterizing precipitation and improving rainfall estimates over the Southern Ocean using ship-borne disdrometer and dual-polarimetric C-band radar.",
            "doi": "https://doi.org/10.1029/2023JD040250"
        },
        "total_power": {
            "standard_name": "total_power",
            "long_name": "Total Power",
            "description": "Total received power including noise and signal.",
            "units": "dBm",
        },
        "corrected_differential_reflectivity": {
            "standard_name": "corrected_differential_reflectivity",
            "long_name": "Corrected Differential Reflectivity",
            "description": "Differential reflectivity corrected for calibration errors.",
            "units": "dB",
        },
        "reflectivity": {
            "standard_name": "reflectivity",
            "long_name": "Reflectivity",
            "description": "Power returned to the radar from targets.",
            "units": "dBZ",
        },
        "D0": {
            "standard_name": "median_volume_diameter",
            "long_name": "Median Volume Diameter",
            "description": "Median volume diameter of hydrometeors.",
            "units": "mm",
            "reference": "Bringi et al. (2009): Using Dual-Polarized Radar and Dual-Frequency Profiler for DSD Characterization: A Case Study from Darwin, Australia.",
            "doi": "https://doi.org/10.1175/2009JTECHA1258.1."
        },
        "attenuation_corrected_reflectivity": {
            "standard_name": "attenuation_corrected_reflectivity",
            "long_name": "Attenuation Corrected Reflectivity",
            "description": "Reflectivity corrected for attenuation due to precipitation.",
            "units": "dBZ",
        },
        "path_integrated_attenuation": {
            "standard_name": "path_integrated_attenuation",
            "long_name": "Path Integrated Attenuation",
            "description": "Cumulative attenuation along the radar beam path.",
            "units": "dB",
            "reference": "Gu et al. (2011): Polarimetric Attenuation Correction in Heavy Rain at C Band.",
            "doi": "https://doi.org/10.1175/2010JAMC2258.1."
        },
        "total_power_v": {
            "standard_name": "total_power_vertical",
            "long_name": "Total Power (Vertical Polarization)",
            "description": "Total power for the vertically polarized radar signal.",
            "units": "dBm",
        },
        "reflectivity_v": {
            "standard_name": "reflectivity_vertical",
            "long_name": "Reflectivity (Vertical Polarization)",
            "description": "Reflectivity for vertically polarized radar signals.",
            "units": "dBZ",
        },
        "corrected_reflectivity": {
            "standard_name": "corrected_reflectivity",
            "long_name": "Corrected Reflectivity",
            "description": "Reflectivity corrected for noise and calibration.",
            "units": "dBZ",
        },
        "differential_phase": {
            "standard_name": "differential_phase",
            "long_name": "Differential Phase",
            "description": "Phase difference between horizontal and vertical polarizations.",
            "units": "degree",
        },
        "temperature": {
            "standard_name": "air_temperature",
            "long_name": "Temperature",
            "description": "Ambient air temperature from ERA5 reanalysis.",
            "units": "degree_C",
        },
        "NW": {
            "standard_name": "normalized_intercept_parameter",
            "long_name": "Log10 of Normalized Intercept Parameter",
            "description": "Log10 Intercept parameter of the drop size distribution.",
            "units": "mm-1 m-3",
            "reference": "Bringi et al. (2009): Using Dual-Polarized Radar and Dual-Frequency Profiler for DSD Characterization: A Case Study from Darwin, Australia.",
            "doi": "https://doi.org/10.1175/2009JTECHA1258.1."
        },
        "steiner_echo_classification": {
            "long_name": "Steiner Convective Stratiform Classification",
            "units": "index",
            "coverage_content_type": "thematicClassification",
            "description": "0 = Undefined, 1 = Stratiform, 2 = Convective",
            "comment_0": "Based on the gridded reflectivity at 1500m.",
            "valid_min": 0,
            "valid_max": 2,
            "reference": "Steiner et al. (1995): Climatological Characterization of Three-Dimensional Storm Structure from Operational Radar and Rain Gauge Data.",
            "doi": "https://doi.org/10.1175/1520-0450(1995)034<1978:CCOTDS>2.0.CO;2."
        },
    }
    try:
        return variables_metadata[f]
    except KeyError:
        return None


def get_var_cct(f: str) -> Union[Dict, None]:
    """
    Metadata compliance of coverage content types with ACDD-1.3

    Parameters:
    ===========
    f: str
        Field name
    Returns:
    ========
    cct: str
        Coverage content type.
    """
    coverage_content_types = {
        "lat": "coordinate",
        "lon": "coordinate",
        "cross_correlation_ratio": "physicalMeasurement",
        "radar_estimated_snow_rate": "modelResult",
        "signal_to_noise_ratio": "qualityInformation",
        "corrected_differential_phase": "physicalMeasurement",
        "velocity": "physicalMeasurement",
        "corrected_velocity": "physicalMeasurement",
        "spectrum_width": "physicalMeasurement",
        "corrected_specific_differential_phase": "physicalMeasurement",
        "radar_echo_classification": "thematicClassification",
        "differential_reflectivity": "physicalMeasurement",
        "normalized_coherent_power": "qualityInformation",
        "radar_estimated_rain_rate": "physicalMeasurement",
        "total_power": "physicalMeasurement",
        "corrected_differential_reflectivity": "physicalMeasurement",
        "reflectivity": "physicalMeasurement",
        "D0": "physicalMeasurement",
        "attenuation_corrected_reflectivity": "physicalMeasurement",
        "path_integrated_attenuation": "physicalMeasurement",
        "total_power_v": "physicalMeasurement",
        "reflectivity_v": "physicalMeasurement",
        "corrected_reflectivity": "physicalMeasurement",
        "differential_phase": "physicalMeasurement",
        "temperature": "auxiliaryInformation",
        "NW": "physicalMeasurement",
    }
    try:
        return coverage_content_types[f]
    except KeyError:
        return None


def coverage_content_type(radar: pyart.core.Radar) -> None:
    """
    Adding metadata for compatibility with ACDD-1.3

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    radar.range["coverage_content_type"] = "coordinate"
    radar.azimuth["coverage_content_type"] = "coordinate"
    radar.elevation["coverage_content_type"] = "coordinate"
    radar.latitude["coverage_content_type"] = "coordinate"
    radar.longitude["coverage_content_type"] = "coordinate"
    radar.altitude["coverage_content_type"] = "coordinate"

    radar.sweep_number["coverage_content_type"] = "auxiliaryInformation"
    radar.fixed_angle["coverage_content_type"] = "auxiliaryInformation"
    radar.sweep_mode["coverage_content_type"] = "auxiliaryInformation"

    for k in radar.fields.keys():
        if k == "radar_echo_classification":
            radar.fields[k]["coverage_content_type"] = "thematicClassification"
        elif k in ["normalized_coherent_power", "normalized_coherent_power_v"]:
            radar.fields[k]["coverage_content_type"] = "qualityInformation"
        else:
            radar.fields[k]["coverage_content_type"] = "physicalMeasurement"

    return None


def correct_standard_name(radar: pyart.core.Radar) -> None:
    """
    'standard_name' is a protected keyword for metadata in the CF conventions.
    To respect the CF conventions we can only use the standard_name field that
    exists in the CF table.

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    try:
        radar.range.pop("standard_name")
        radar.azimuth.pop("standard_name")
        radar.elevation.pop("standard_name")
    except Exception:
        pass

    try:
        radar.sweep_number.pop("standard_name")
        radar.fixed_angle.pop("standard_name")
        radar.sweep_mode.pop("standard_name")
    except Exception:
        pass

    good_keys = ["corrected_reflectivity", "total_power", "radar_estimated_rain_rate", "corrected_velocity"]
    for k in radar.fields.keys():
        if k not in good_keys:
            try:
                radar.fields[k].pop("standard_name")
            except Exception:
                continue

    try:
        radar.fields["velocity"]["standard_name"] = "radial_velocity_of_scatterers_away_from_instrument"
        radar.fields["velocity"]["long_name"] = "Doppler radial velocity of scatterers away from instrument"
    except KeyError:
        pass

    radar.latitude["standard_name"] = "latitude"
    radar.longitude["standard_name"] = "longitude"
    radar.altitude["standard_name"] = "altitude"

    return None


def rename_field_names(radar: pyart.core.Radar) -> pyart.core.Radar:
    """
    Rename fields from ODIM convention to a more "CF compliant" naming scheme.
    """
    FIELDS_NAMES = [
        ("VRAD", "velocity"),
        ("VRADDH", "corrected_velocity"),
        ("TH", "total_power"),
        ("TV", "total_power_v"),
        ("DBZH", "reflectivity"),
        ("DBZV", "reflectivity_v"),
        ("DBZH_CLEAN", "corrected_reflectivity"),
        ("DBZH_CLEAN2", "attenuation_corrected_reflectivity"),
        ("PIA", "path_integrated_attenuation"),
        ("RHOHV", "cross_correlation_ratio"),
        ("ZDR", "differential_reflectivity"),
        ("ZDR_CLEAN", "corrected_differential_reflectivity"),
        ("PHIDP", "differential_phase"),
        ("PHIDP_PHIDO", "corrected_differential_phase"),
        ("KDP", "specific_differential_phase"),
        ("KDP_PHIDO", "corrected_specific_differential_phase"),
        ("WRAD", "spectrum_width"),
        ("SNR", "signal_to_noise_ratio"),
        ("SQI", "normalized_coherent_power"),
        ("WRADV", "spectrum_width_v"),
        ("SNRV", "signal_to_noise_ratio_v"),
        ("SQIV", "normalized_coherent_power_v"),
        ("HID", "radar_echo_classification"),
        ("TEMPERATURE", "temperature"),
        ("SNOW", "radar_estimated_snow_rate"),
        ("RAIN", "radar_estimated_rain_rate"),
    ]
    for old_key, new_key in FIELDS_NAMES:
        try:
            radar.add_field(new_key, radar.fields.pop(old_key), replace_existing=True)
        except KeyError:
            continue
    return radar


def get_metadata(radar: pyart.core.Radar) -> Dict:
    today = datetime.now()
    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    radar_end_date = cftime.num2pydate(radar.time["data"][-1], radar.time["units"])

    # Lat/lon informations
    latitude = radar.gate_latitude["data"]
    longitude = radar.gate_longitude["data"]
    maxlon = longitude.max()
    minlon = longitude.min()
    maxlat = latitude.max()
    minlat = latitude.min()
    origin_altitude = radar.altitude["data"][0]
    origin_latitude = radar.latitude["data"][0]
    origin_longitude = radar.longitude["data"][0]

    unique_id = str(uuid.uuid4())
    metadata = {
        "Conventions": "CF-1.6, ACDD-1.3",
        "country": "Australia",
        "creator_email": "CPOL-support@bom.gov.au",
        "creator_name": "Commonwealth of Australia, Bureau of Meteorology, Science and Innovation Group, Research, Radar Science",
        "creator_url": "https://bom365.sharepoint.com/sites/SI_WEP_RSAN",
        "date_created": today.isoformat(),
        "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
        "geospatial_lat_max": f"{maxlat:0.6}",
        "geospatial_lat_min": f"{minlat:0.6}",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_max": f"{maxlon:0.6}",
        "geospatial_lon_min": f"{minlon:0.6}",
        "geospatial_lon_units": "degrees_east",
        "history": "created by Valentin Louf on gadi.nci.org.au at " + today.isoformat() + " using Py-ART",
        "id": unique_id,
        "institution": "Bureau of Meteorology",
        "instrument": "radar",
        "instrument_name": "OPOL",
        "instrument_type": "radar",
        "keywords": "R/V INVESTIGATOR, POLARIMETRIC RADAR, C-BAND RADAR",
        "keywords_vocabulary": "NASA Global Change Master Directory (GCMD) Science Keywords",
        "license": "CC BY-NC-SA 4.0",
        "naming_authority": "au.gov.bom",
        "origin_altitude": np.float32(origin_altitude),
        "origin_latitude": np.float32(origin_latitude),
        "origin_longitude": np.float32(origin_longitude),
        "platform_is_mobile": "true",
        "processing_level": "L2",
        "project": "OPOL",
        "publisher_name": "NCI",
        "publisher_url": "nci.gov.au",
        "product_version": f"v{today.year}.{today.month:02}",
        "site_name": "RV Investigator",
        "source": "radar",
        "standard_name_vocabulary": "CF Standard Name Table v71",
        "summary": "Volumetric scan from OPOL dual-polarization Doppler radar (RV Investigator)",
        "time_coverage_start": radar_start_date.isoformat(),
        "time_coverage_end": radar_end_date.isoformat(),
        "time_coverage_duration": "P06M",
        "time_coverage_resolution": "PT06M",
        "title": "radar PPI volume from OPOL",        
    }
    return metadata


def to_compress(dset: xr.Dataset, name: str, scale: float, offset: float, nodata: float, dtype=np.uint8) -> np.ndarray:
    data = dset[name].fillna(nodata).values
    data[data < nodata] = nodata
    dsave = (1 / scale * (data - offset)).astype(dtype)
    return dsave


def grid_opol(odm_file: str, out_nc: str) -> xr.Dataset:    
    radar = pyart.aux_io.read_odim_h5(odm_file, file_field_names=True)
    
    # Renaming, metadata, ADCC compliance.
    radar = rename_field_names(radar)
    radar.metadata = get_metadata(radar)
    correct_standard_name(radar)
    coverage_content_type(radar)

    grid = pyart.map.grid_from_radars(
        radar,
        (33, 301, 301),
        ((0, 16e3), (-150e3, 150e3), (-150e3, 150e3)),
        copy_field_dtypes=True
    )
    
    grid.init_point_longitude_latitude()
    sclass = echo_class.steiner_conv_strat(grid)
    dset = grid.to_xarray().drop(["ROI", "projection", "radar_name", "radar_time", "radar_altitude", "radar_longitude", "radar_latitude"])
    
    # Correct datetime type (cftime to datetime64).
    dtime = datetime(*dset.time.values[0].timetuple()[:6])
    dset = dset.assign_coords({"time": np.array([dtime], dtype=np.datetime64)})
    for k in ["x", "y", "z"]:
        dset[k].attrs["units"] = "m"
    
    # Steiner
    dset["radar_echo_classification"].values = dset["radar_echo_classification"].values.astype(np.uint8)
    dset = dset.merge({"steiner_echo_classification": (("y", "x"), sclass["data"].astype(np.uint8))})    
    
    for k in dset.variables.keys():
        cct = get_var_cct(k)
        if cct is not None:
            dset[k].attrs["coverage_content_type"] = cct
        att = get_var_metadata(k)
        if att is None:
            continue
        dset[k].attrs.update(**att)    
        if k not in ["steiner_echo_classification", "radar_echo_classification"]:
            dset[k].attrs["_FillValue"] = np.nan
        else:
            dset[k].attrs["_FillValue"] = 0
            
    dset = compress_data(dset)
    dset.to_netcdf(out_nc, encoding={k: {"zlib": True} for k, v in dset.variables.items()})
    
    return dset