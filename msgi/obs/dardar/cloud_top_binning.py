
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict 
import numpy as np
import xarray as xr
from multiprocessing import Pool, cpu_count
import subprocess
from datetime import datetime, timedelta
from os import path as os_path
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

# Class encoding
PHASE_NO_CLOUD = np.int8(0)
PHASE_LIQUID = np.int8(1)
PHASE_ICE = np.int8(2)
PHASE_MIXED = np.int8(3)

PHASE_LABELS = np.array(["no_cloud", "liquid", "ice", "mixed"], dtype=object)

# Excluded from cloud-top candidate search
EXCLUDE_CODES = np.array([-99, -2, -1, 0, 6, 8], dtype=np.int16)

# Top-code to 4-class mapping
LIQUID_CODES = np.array([3, 5, 7, 11, 12, 13, 15], dtype=np.int16)  # includes rain-only treatment
ICE_CODES = np.array([1, 2, 9, 10], dtype=np.int16)
MIXED_CODES = np.array([4, 14], dtype=np.int16)


def _detect_height_orientation(height_values: np.ndarray) -> bool:
    """
    Return True if height increases with index, False if decreases.

    Robust to NaNs by using finite differences only.
    """
    h = np.asarray(height_values, dtype=np.float64)
    finite = np.isfinite(h)
    if finite.sum() < 2:
        return True  # fallback: assume increasing if insufficient info

    h_finite = h[finite]
    dh = np.diff(h_finite)
    dh = dh[np.isfinite(dh)]
    if dh.size == 0:
        return True

    return np.nanmedian(dh) > 0.0

def _height_to_km(height_values: np.ndarray, units: str | None) -> np.ndarray:
    """
    Convert CS_TRACK_Height to km using units when available, else a heuristic.
    """
    h = np.asarray(height_values, dtype=np.float64)
    u = (units or "").strip().lower()

    if "km" in u:
        return h
    if "m" in u and "km" not in u:
        return h / 1000.0

    # Heuristic if units missing/ambiguous: if typical magnitude is large, assume meters.
    med_abs = np.nanmedian(np.abs(h)) if np.isfinite(h).any() else np.nan
    if np.isfinite(med_abs) and med_abs > 200.0:
        return h / 1000.0
    return h

def extract_cloud_top_phase4(ds: xr.Dataset, file_ind: int = 1) -> xr.Dataset:
    """
    Extract cloud-top phase in 4 classes for each along-track profile.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing:
        - DARMASK_Simplified_Categorization(time, height)
        - CS_TRACK_Height(height)
        - CLOUDSAT_UTC_Time(time)
        Optional:
        - CLOUDSAT_Latitude(time)
        - CLOUDSAT_Longitude(time)

    Returns
    -------
    xr.Dataset
        Output dataset with dimension (time,) and variables:
        - cloud_top_height_km (float)
        - cloud_top_code (int16)
        - cloud_top_phase_4class (int8: 0/1/2/3)
        - cloud_top_phase_4class_str (string)
        - qa_unmapped_top_code (bool)
        - qa_no_cloud_candidate (bool)
        - CLOUDSAT_UTC_Time
        - optional lat/lon if present
    """
    required = [
        "DARMASK_Simplified_Categorization",
        "CS_TRACK_Height",
        "CLOUDSAT_UTC_Time",
    ]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    dmask = ds["DARMASK_Simplified_Categorization"]
    if dmask.ndim != 2:
        raise ValueError("DARMASK_Simplified_Categorization must be 2D (time, height).")

    time_dim, height_dim = dmask.dims
    n_time = dmask.sizes[time_dim]
    n_height = dmask.sizes[height_dim]

    # DARMASK as float for NaN-safe validity checks, plus int16 code view for mapping.
    dmask_float = np.asarray(dmask.values, dtype=np.float64)
    valid_numeric = np.isfinite(dmask_float)

    dmask_code = np.full(dmask_float.shape, -99, dtype=np.int16)
    dmask_code[valid_numeric] = np.rint(dmask_float[valid_numeric]).astype(np.int16)

    # Candidate bins: valid and not in exclusion set.
    is_excluded = np.isin(dmask_code, EXCLUDE_CODES)
    candidate = valid_numeric & (~is_excluded)

    # Determine vertical orientation and pick highest cloud bin accordingly.
    height_var = ds["CS_TRACK_Height"]
    height_vals = np.asarray(height_var.values, dtype=np.float64)
    if height_vals.ndim != 1 or height_vals.size != n_height:
        raise ValueError("CS_TRACK_Height must be 1D and match DARMASK height dimension.")

    increasing_with_index = _detect_height_orientation(height_vals)
    idx = np.broadcast_to(np.arange(n_height, dtype=np.int32), (n_time, n_height))

    if increasing_with_index:
        top_idx = np.where(candidate, idx, -1).max(axis=1)
        has_candidate = top_idx >= 0
    else:
        top_idx = np.where(candidate, idx, n_height).min(axis=1)
        has_candidate = top_idx < n_height

    # Extract top-bin code and height.
    cloud_top_code = np.full(n_time, -99, dtype=np.int16)
    cloud_top_height_km = np.full(n_time, np.nan, dtype=np.float32)
    # cloud_top_temp = np.full(n_time, np.nan, dtype=np.float32)

    rows = np.where(has_candidate)[0]
    cols = top_idx[has_candidate]
    cloud_top_code[has_candidate] = dmask_code[rows, cols]

    height_km = _height_to_km(height_vals, height_var.attrs.get("units"))
    cloud_top_height_km[has_candidate] = height_km[cols].astype(np.float32)

    # Map top code -> 4 classes.
    phase = np.full(n_time, PHASE_NO_CLOUD, dtype=np.int8)

    is_liquid = has_candidate & np.isin(cloud_top_code, LIQUID_CODES)
    is_ice = has_candidate & np.isin(cloud_top_code, ICE_CODES)
    is_mixed = has_candidate & np.isin(cloud_top_code, MIXED_CODES)

    phase[is_liquid] = PHASE_LIQUID
    phase[is_ice] = PHASE_ICE
    phase[is_mixed] = PHASE_MIXED

    mapped = is_liquid | is_ice | is_mixed
    qa_unmapped_top_code = has_candidate & (~mapped)
    qa_no_cloud_candidate = ~has_candidate
    phase_str = PHASE_LABELS[phase]

    # Build output with required (time,) variables.
    out = xr.Dataset(
        data_vars={
            "cloud_top_height_km": (("time",), cloud_top_height_km),
            "cloud_top_code": (("time",), cloud_top_code),
            "cloud_top_phase_4class": (("time",), phase),
            "cloud_top_phase_4class_str": (("time",), phase_str),
            "qa_unmapped_top_code": (("time",), qa_unmapped_top_code.astype(bool)),
            "qa_no_cloud_candidate": (("time",), qa_no_cloud_candidate.astype(bool)),
            "CLOUDSAT_UTC_Time": (("time",), np.asarray(ds["CLOUDSAT_UTC_Time"].values)),
        },
        coords={"time": np.arange(n_time, dtype=np.int32)},
    )

    # Optional geolocation
    if "CLOUDSAT_Latitude" in ds.variables:
        out["CLOUDSAT_Latitude"] = ("time", np.asarray(ds["CLOUDSAT_Latitude"].values))
    if "CLOUDSAT_Longitude" in ds.variables:
        out["CLOUDSAT_Longitude"] = ("time", np.asarray(ds["CLOUDSAT_Longitude"].values))

    # Helpful metadata
    out["cloud_top_phase_4class"].attrs.update(
        {
            "long_name": "Cloud-top phase (4 classes)",
            "flag_values": np.array([0, 1, 2, 3], dtype=np.int8),
            "flag_meanings": "no_cloud liquid ice mixed",
        }
    )
    out["cloud_top_code"].attrs.update(
        {"long_name": "DARMASK code at detected cloud-top bin", "_FillValue": np.int16(-99)}
    )
    out["cloud_top_height_km"].attrs.update({"long_name": "Cloud-top height", "units": "km"})
    out.attrs["source"] = "Derived from DARDAR-MASK simplified categorization"
    out.attrs["height_index_increasing"] = str(bool(increasing_with_index))
    # out.to_netcdf(f"/wolke_scratch/dnikolo/MSGI/DARDAR_analysis/debug/cloud_top_observations_{file_ind}.nc")  # for debugging
    return out

def round_ns_timestamps(ts: np.ndarray, dt_min = 15) -> np.ndarray:
    """
    Round numpy datetime64[ns] array to nearest dt_min minutes.

    Parameters
    ----------
    ts : np.ndarray
        Array of dtype datetime64[ns].

    Returns
    -------
    np.ndarray
        Rounded array with dtype datetime64[ns].
    """
    ts = np.asarray(ts, dtype="datetime64[ns]")

    # 15 minutes in ns: 15 * 60 * 1e9 = 300_000_000_000
    step = np.timedelta64(dt_min, "m").astype("timedelta64[ns]").astype(np.int64)

    # Convert timestamps to integer ns since epoch
    ns = ts.astype("datetime64[ns]").astype(np.int64)

    # Half-up rounding to nearest 15-minute bucket
    rounded_ns = np.empty_like(ns)
    is_nat = np.isnat(ts)
    rounded_ns[is_nat] = 0
    rounded_ns[~is_nat] = ((ns[~is_nat] + step // 2) // step) * step
    output = np.empty_like(ts)
    output[~is_nat] = rounded_ns[~is_nat].astype("datetime64[ns]")
    output[is_nat] = np.datetime64("NaT", "ns")
    return output

def dt_arange(start: np.datetime64, end: np.datetime64, dt_min = 15) -> np.ndarray:
    """
    Generate array of datetime64[ns] from start to end with step dt_min minutes.

    Parameters
    ----------
    start : np.datetime64
        Start timestamp.
    end : np.datetime64
        End timestamp.
    dt_min : int, optional
        Step in minutes (default is 15).

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] timestamps.
    """
    start = np.asarray(start, dtype="datetime64[ns]")
    end = np.asarray(end, dtype="datetime64[ns]")
    step = np.timedelta64(dt_min, "m").astype("timedelta64[ns]")
    return np.arange(start, end + step, step)

def scatter_2d_to_3d_by_time(var2d, rounded_time2d, time_values, fill_value=np.nan):
    """
    Place var2d[i,j] into cube[t,i,j] where rounded_time2d[i,j] == time_values[t].
    Other times at that (i,j) remain fill_value.
    """
    n_t = time_values.size
    n_lat, n_lon = var2d.shape

    cube = np.full((n_t, n_lat, n_lon), fill_value, dtype=var2d.dtype)

    valid = ~np.isnat(rounded_time2d)
    if not np.any(valid) or n_t == 0:
        return cube

    lat_i, lon_i = np.where(valid)
    tvals = rounded_time2d[valid].astype("datetime64[ns]")

    # map timestamp -> index in time_values
    # time_values is regular and sorted, so searchsorted works
    tidx = np.searchsorted(time_values, tvals)

    in_range = (tidx >= 0) & (tidx < n_t) & (time_values[tidx] == tvals)
    if np.any(in_range):
        cube[tidx[in_range], lat_i[in_range], lon_i[in_range]] = var2d[lat_i[in_range], lon_i[in_range]]

    return cube

def bin_cloud_top_phase4_latlon(
    profile_ds: xr.Dataset, bin_size_deg: float = 0.05
) -> xr.Dataset:
    """
    Bin per-profile cloud-top phase into latitude/longitude bins.

    Outputs per-bin:
    - cph_mean: mean cloudy-top phase score (liquid=1, mixed=1.5, ice=2),
      set to -1 where no cloudy tops exist.
    - frac_*: class fractions among all observations in each bin
    - n_*: observation counts
    """
    if bin_size_deg <= 0:
        raise ValueError("bin_size_deg must be > 0.")

    if "CLOUDSAT_Latitude" not in profile_ds.variables or "CLOUDSAT_Longitude" not in profile_ds.variables:
        raise ValueError("Latitude/longitude are required for binning but are missing.")

    lat = np.asarray(profile_ds["CLOUDSAT_Latitude"].values, dtype=np.float64)
    lon = np.asarray(profile_ds["CLOUDSAT_Longitude"].values, dtype=np.float64)
    phase = np.asarray(profile_ds["cloud_top_phase_4class"].values, dtype=np.int8)
    time = np.asarray(profile_ds["CLOUDSAT_UTC_Time"].values)
    height = np.asarray(profile_ds["cloud_top_height_km"].values, dtype=np.float64)

    valid_geo = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid_geo):
        raise ValueError("No finite latitude/longitude values found for binning.")

    lat_valid = lat[valid_geo]
    lon_valid = lon[valid_geo]
    phase_valid = phase[valid_geo]
    height_valid = height[valid_geo]
    time_valid = time[valid_geo]

    lat_min = np.floor(np.nanmin(lat_valid) / bin_size_deg) * bin_size_deg
    lat_max = np.ceil(np.nanmax(lat_valid) / bin_size_deg) * bin_size_deg
    lon_min = np.floor(np.nanmin(lon_valid) / bin_size_deg) * bin_size_deg
    lon_max = np.ceil(np.nanmax(lon_valid) / bin_size_deg) * bin_size_deg

    lat_edges = np.arange(lat_min, lat_max + bin_size_deg, bin_size_deg, dtype=np.float64)
    lon_edges = np.arange(lon_min, lon_max + bin_size_deg, bin_size_deg, dtype=np.float64)
    if lat_edges.size < 2 or lon_edges.size < 2:
        raise ValueError("Could not construct latitude/longitude bin edges.")

    n_lat = lat_edges.size - 1
    n_lon = lon_edges.size - 1

    lat_idx = np.digitize(lat_valid, lat_edges) - 1
    lon_idx = np.digitize(lon_valid, lon_edges) - 1
    in_bounds = (lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon)

    lat_idx = lat_idx[in_bounds]
    lon_idx = lon_idx[in_bounds]
    phase_valid = phase_valid[in_bounds]
    height_valid = height_valid[in_bounds]
    time_valid = time_valid[in_bounds].astype("datetime64[ns]")

    shape = (n_lat, n_lon)
    n_obs = np.zeros(shape, dtype=np.int32)
    n_no_cloud = np.zeros(shape, dtype=np.int32)
    n_liquid = np.zeros(shape, dtype=np.int32)
    n_ice = np.zeros(shape, dtype=np.int32)
    n_mixed = np.zeros(shape, dtype=np.int32)

    phase_sum_cloudy = np.zeros(shape, dtype=np.float64)
    height_sum_cloudy = np.zeros(shape, dtype=np.float64)
    height_sq_sum_cloudy = np.zeros(shape, dtype=np.float64)
    time_sum_cloudy = np.zeros(shape, dtype="timedelta64[ns]")

    np.add.at(n_obs, (lat_idx, lon_idx), 1)
    np.add.at(n_no_cloud, (lat_idx[phase_valid == PHASE_NO_CLOUD], lon_idx[phase_valid == PHASE_NO_CLOUD]), 1)
    np.add.at(n_liquid, (lat_idx[phase_valid == PHASE_LIQUID], lon_idx[phase_valid == PHASE_LIQUID]), 1)
    np.add.at(n_ice, (lat_idx[phase_valid == PHASE_ICE], lon_idx[phase_valid == PHASE_ICE]), 1)
    np.add.at(n_mixed, (lat_idx[phase_valid == PHASE_MIXED], lon_idx[phase_valid == PHASE_MIXED]), 1)

    # Mean phase score over cloudy-only tops: liquid=0, mixed=0.5, ice=1
    score = np.full(phase_valid.shape, np.nan, dtype=np.float64)
    score[phase_valid == PHASE_LIQUID] = 1.0
    score[phase_valid == PHASE_MIXED] = 1.5
    score[phase_valid == PHASE_ICE] = 2.0
    cloudy = np.isfinite(score)

    np.add.at(phase_sum_cloudy, (lat_idx[cloudy], lon_idx[cloudy]), score[cloudy])
    np.add.at(height_sum_cloudy, (lat_idx[cloudy], lon_idx[cloudy]), height_valid[cloudy])
    np.add.at(height_sq_sum_cloudy, (lat_idx[cloudy], lon_idx[cloudy]), height_valid[cloudy] ** 2)

    # Reference datetime to safely accumulate timedeltas
    measured = n_obs > 0
    time_ref = time_valid.min()
    np.add.at(time_sum_cloudy, (lat_idx, lon_idx), time_valid - time_ref)

    n_cloudy = n_liquid + n_ice + n_mixed
    cloudy_bins = n_cloudy > 0
    obs_bins = n_obs > 0

    # Use explicit sentinels for "defined but no cloudy tops"
    cph_fill = np.float32(-1.0)
    cth_fill = np.float32(-1.0)

    mean_phase = np.full(shape, cph_fill, dtype=np.float32)
    mean_height = np.full(shape, cth_fill, dtype=np.float32)
    var_height = np.full(shape, np.nan, dtype=np.float64)
    std_height = np.full(shape, np.nan, dtype=np.float32)
    mean_time = np.full(shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    mean_phase[cloudy_bins] = (phase_sum_cloudy[cloudy_bins] / n_cloudy[cloudy_bins]).astype(np.float32)
    mean_phase[obs_bins & ~cloudy_bins] = 0.0  # defined as no cloudy tops among observations
    mean_height[cloudy_bins] = (height_sum_cloudy[cloudy_bins] / n_cloudy[cloudy_bins]).astype(np.float32)

    var_height[cloudy_bins] = (
        height_sq_sum_cloudy[cloudy_bins] / n_cloudy[cloudy_bins]
    ) - (mean_height[cloudy_bins].astype(np.float64) ** 2)
    var_height[cloudy_bins] = np.maximum(var_height[cloudy_bins], 0.0)
    std_height[cloudy_bins] = np.sqrt(var_height[cloudy_bins]).astype(np.float32)

    mean_time[obs_bins] = (
        time_sum_cloudy[obs_bins] / n_obs[obs_bins] + time_ref
    ).astype("datetime64[ns]")

    rounded_mean_time = round_ns_timestamps(mean_time, dt_min=15)
    valid_time = ~np.isnat(rounded_mean_time)
    if not np.any(valid_time):
        raise ValueError("No valid mean times found for cloudy bins, cannot construct time dimension.")
    tmin = rounded_mean_time[valid_time].min()
    tmax = rounded_mean_time[valid_time].max()
    time_dim_values = dt_arange(tmin, tmax, dt_min=15)

    frac_no_cloud = np.full(shape, np.nan, dtype=np.float32)
    frac_liquid = np.full(shape, np.nan, dtype=np.float32)
    frac_ice = np.full(shape, np.nan, dtype=np.float32)
    frac_mixed = np.full(shape, np.nan, dtype=np.float32)

    frac_no_cloud[obs_bins] = (n_no_cloud[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_liquid[obs_bins] = (n_liquid[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_ice[obs_bins] = (n_ice[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_mixed[obs_bins] = (n_mixed[obs_bins] / n_obs[obs_bins]).astype(np.float32)

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    mean_phase_3d = scatter_2d_to_3d_by_time(mean_phase, rounded_mean_time, time_dim_values, fill_value=cph_fill)
    mean_height_3d = scatter_2d_to_3d_by_time(mean_height, rounded_mean_time, time_dim_values, fill_value=cth_fill)
    std_height_3d = scatter_2d_to_3d_by_time(std_height, rounded_mean_time, time_dim_values, fill_value=np.float32(np.nan))
    frac_no_cloud_3d = scatter_2d_to_3d_by_time(frac_no_cloud, rounded_mean_time, time_dim_values, fill_value=np.float32(np.nan))
    frac_liquid_3d = scatter_2d_to_3d_by_time(frac_liquid, rounded_mean_time, time_dim_values, fill_value=np.float32(np.nan))
    frac_ice_3d = scatter_2d_to_3d_by_time(frac_ice, rounded_mean_time, time_dim_values, fill_value=np.float32(np.nan))
    frac_mixed_3d = scatter_2d_to_3d_by_time(frac_mixed, rounded_mean_time, time_dim_values, fill_value=np.float32(np.nan))
    n_obs_3d = scatter_2d_to_3d_by_time(n_obs, rounded_mean_time, time_dim_values, fill_value=np.int32(0))
    n_no_cloud_3d = scatter_2d_to_3d_by_time(n_no_cloud, rounded_mean_time, time_dim_values, fill_value=np.int32(0))
    n_liquid_3d = scatter_2d_to_3d_by_time(n_liquid, rounded_mean_time, time_dim_values, fill_value=np.int32(0))
    n_ice_3d = scatter_2d_to_3d_by_time(n_ice, rounded_mean_time, time_dim_values, fill_value=np.int32(0))
    n_mixed_3d = scatter_2d_to_3d_by_time(n_mixed, rounded_mean_time, time_dim_values, fill_value=np.int32(0))
    n_cloudy_3d = scatter_2d_to_3d_by_time(n_cloudy, rounded_mean_time, time_dim_values, fill_value=np.int32(0))

    # --- Convert to seconds since epoch for CDO compatibility ---
    epoch_units = "seconds since 1970-01-01 00:00:00"

    # time_bin coordinate (1D)
    time_bin_sec = time_dim_values.astype("datetime64[s]").astype(np.int64)
    # use int32 if range is safe (it is for 2008), otherwise keep int64
    # time_bin_sec = time_bin_sec.astype(np.int32)

    # rounded_mean_time (2D)
    rmt = rounded_mean_time.astype("datetime64[s]")
    rmt_sec = rmt.astype(np.int64)
    fill = np.int32(-1)  # int32 fill
    rmt_sec = rmt_sec.astype(np.int32)
    rmt_sec[np.isnat(rmt)] = fill

    out = xr.Dataset(
        data_vars={
            "cph_mean": (("time_bin", "lat_bin", "lon_bin"), mean_phase_3d),
            "cth_mean": (("time_bin", "lat_bin", "lon_bin"), mean_height_3d),
            "cth_std": (("time_bin", "lat_bin", "lon_bin"), std_height_3d),
            "frac_no_cloud": (("time_bin", "lat_bin", "lon_bin"), frac_no_cloud_3d),
            "frac_liquid": (("time_bin", "lat_bin", "lon_bin"), frac_liquid_3d),
            "frac_ice": (("time_bin", "lat_bin", "lon_bin"), frac_ice_3d),
            "frac_mixed": (("time_bin", "lat_bin", "lon_bin"), frac_mixed_3d),
            "n_obs": (("time_bin", "lat_bin", "lon_bin"), n_obs_3d),
            "n_no_cloud": (("time_bin", "lat_bin", "lon_bin"), n_no_cloud_3d),
            "n_liquid": (("time_bin", "lat_bin", "lon_bin"), n_liquid_3d),
            "n_ice": (("time_bin", "lat_bin", "lon_bin"), n_ice_3d),
            "n_mixed": (("time_bin", "lat_bin", "lon_bin"), n_mixed_3d),
            "n_cloudy": (("time_bin", "lat_bin", "lon_bin"), n_cloudy_3d),
            "rounded_mean_time": (("lat_bin", "lon_bin"), rmt_sec.astype(np.int32)),
        },
        coords={
            "time_bin": ("time_bin", time_bin_sec),
            "lat_bin": ("lat_bin", lat_centers.astype(np.float32)),
            "lon_bin": ("lon_bin", lon_centers.astype(np.float32)),
            "lat_bin_edge": ("lat_bin_edge", lat_edges.astype(np.float32)),
            "lon_bin_edge": ("lon_bin_edge", lon_edges.astype(np.float32)),
        },
    )

    # ---------- Coordinate metadata (CF-ish) ----------
    out["time_bin"].attrs.update({
    "standard_name": "time",
    "long_name": "time bin center",
    "units": epoch_units,
    "calendar": "standard",
    })

    out["lat_bin"].attrs.update({
        "standard_name": "latitude",
        "long_name": "latitude bin center",
        "units": "degrees_north",
        "bounds": "lat_bin_edge",
    })
    out["lon_bin"].attrs.update({
        "standard_name": "longitude",
        "long_name": "longitude bin center",
        "units": "degrees_east",
        "bounds": "lon_bin_edge",
    })
    out["lat_bin_edge"].attrs.update({
        "long_name": "latitude bin edges",
        "units": "degrees_north",
    })
    out["lon_bin_edge"].attrs.update({
        "long_name": "longitude bin edges",
        "units": "degrees_east",
    })

    coord_ref = "time_bin lat_bin lon_bin"

    # ---------- Data variable metadata ----------
    out["cph_mean"].attrs.update({
        "long_name": "Mean cloud-top phase score over cloudy tops",
        "_FillValue": np.float32(cph_fill),
        "missing_value": np.float32(cph_fill),
        "valid_min": np.float32(0.0),
        "valid_max": np.float32(2.0),
        "flag_values": np.array([-1.0, 0.0 , 1.0, 1.5, 2], dtype=np.float32),
        "flag_meanings": "no_measurement no_cloudy_tops liquid mixed ice",
        "coordinates": coord_ref,
        "comment": "Computed from cloudy samples only: liquid=1, mixed=1.5, ice=2; -1 indicates no measurements in bin.",
    })

    out["cth_mean"].attrs.update({
        "long_name": "Mean cloud-top height over cloudy tops",
        "units": "km",
        "_FillValue": np.float32(cth_fill),
        "missing_value": np.float32(cth_fill),
        "valid_min": np.float32(0.0),
        "coordinates": coord_ref,
        "comment": "-1 indicates no cloudy tops in bin.",
    })

    out["cth_std"].attrs.update({
        "long_name": "Standard deviation of cloud-top height over cloudy tops",
        "units": "km",
        "_FillValue": np.float32(np.nan),
        "missing_value": np.float32(np.nan),
        "valid_min": np.float32(0.0),
        "coordinates": coord_ref,
        "comment": "NaN indicates undefined (e.g., no cloudy tops).",
    })

    for v, ln in [
        ("frac_no_cloud", "Fraction of observations classified as no cloud"),
        ("frac_liquid", "Fraction of observations classified as liquid"),
        ("frac_ice", "Fraction of observations classified as ice"),
        ("frac_mixed", "Fraction of observations classified as mixed"),
    ]:
        out[v].attrs.update({
            "long_name": ln,
            "units": "1",
            "_FillValue": np.float32(np.nan),
            "missing_value": np.float32(np.nan),
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(1.0),
            "coordinates": coord_ref,
            "comment": "NaN indicates no observations in bin.",
        })

    for v, ln in [
        ("n_obs", "Number of observations"),
        ("n_no_cloud", "Number of no-cloud observations"),
        ("n_liquid", "Number of liquid observations"),
        ("n_ice", "Number of ice observations"),
        ("n_mixed", "Number of mixed-phase observations"),
        ("n_cloudy", "Number of cloudy observations"),
    ]:
        out[v].attrs.update({
            "long_name": ln,
            "units": "1",
            "_FillValue": np.int32(0),
            "missing_value": np.int32(0),
            "valid_min": np.int32(0),
            "coordinates": coord_ref,
            "comment": "Count variable; 0 means none in bin/time.",
        })

    out["rounded_mean_time"].attrs.update({
        "long_name": "Rounded mean cloudy-top observation time at each lat/lon bin",
        "units": epoch_units,
        "calendar": "standard",
        "_FillValue": fill,
    })
    out["rounded_mean_time"].encoding.update({
        "dtype": "int32",
    })
    # for v in out.data_vars:
    #     out[v].attrs.pop("coordinates", None)

    # ---------- Global metadata ----------
    out.attrs.update({
        "title": "Binned cloud-top phase and cloud-top height statistics",
        "summary": "Cloud-top phase/height statistics aggregated to regular lat/lon bins and mapped to time bins.",
        "Conventions": "CF-1.10",
        "featureType": "grid",
        "source": "Binned from per-profile cloud-top phase classes derived from DARDAR-MASK simplified categorization",
        "bin_size_degrees": float(bin_size_deg),
        "history": "Created by bin_cloud_top_phase4_latlon",
    })

    return out

def _print_summary(out: xr.Dataset) -> None:
    phase = np.asarray(out["cloud_top_phase_4class"].values, dtype=np.int8)
    total = phase.size

    counts: Dict[int, int] = {
        0: int(np.sum(phase == 0)),
        1: int(np.sum(phase == 1)),
        2: int(np.sum(phase == 2)),
        3: int(np.sum(phase == 3)),
    }
    labels: Dict[int, str] = {0: "no_cloud", 1: "liquid", 2: "ice", 3: "mixed"}

    print(f"total_profiles: {total}")
    for cid in [0, 1, 2, 3]:
        pct = (100.0 * counts[cid] / total) if total > 0 else 0.0
        print(f"{labels[cid]}: {counts[cid]} ({pct:.2f}%)")

    unmapped = int(np.sum(np.asarray(out["qa_unmapped_top_code"].values, dtype=bool)))
    print(f"unmapped_top_codes: {unmapped}")

def get_single_binned_cloud_top() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cloud-top phase (4 classes) from a DARDAR-MASK NetCDF file."
    )
    parser.add_argument("input", help="Input DARDAR-MASK NetCDF file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output NetCDF path (default: <input_basename>_cloud_top_phase4_binned.nc)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Latitude/longitude bin size in degrees (default: 0.05).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if args.output is None:
        out_path = in_path.with_name(f"{in_path.stem}_cloud_top_phase4_binned.nc")
    else:
        out_path = Path(args.output)

    with xr.open_dataset(in_path) as ds_in:
        out_profile = extract_cloud_top_phase4(ds_in)

        # Unit-test-like checks requested by specification.
        phase = np.asarray(out_profile["cloud_top_phase_4class"].values)
        if not np.isin(phase, [0, 1, 2, 3]).all():
            raise AssertionError("cloud_top_phase_4class contains values outside {0,1,2,3}")
        if out_profile.sizes["time"] != ds_in["DARMASK_Simplified_Categorization"].sizes[
            ds_in["DARMASK_Simplified_Categorization"].dims[0]
        ]:
            raise AssertionError("Output time length does not equal input time length")

        out_binned = bin_cloud_top_phase4_latlon(out_profile, bin_size_deg=args.bin_size)
        out_binned.to_netcdf(out_path)

    _print_summary(out_profile)
    print(f"saved: {out_path}")

def _process_file(in_path: Path, out_path: Path, bin_size_deg: float, file_ind=1) -> None:
    if in_path is None:
        raise ValueError("in_path must be provided to _process_file")
    if out_path is None:
        raise ValueError("out_path must be provided to _process_file")

    with xr.open_dataset(in_path) as ds_in:
        out_profile = extract_cloud_top_phase4(ds_in,file_ind)

        # Unit-test-like checks requested by specification.
        phase = np.asarray(out_profile["cloud_top_phase_4class"].values)
        if not np.isin(phase, [0, 1, 2, 3]).all():
            raise AssertionError("cloud_top_phase_4class contains values outside {0,1,2,3}")
        if out_profile.sizes["time"] != ds_in["DARMASK_Simplified_Categorization"].sizes[
            ds_in["DARMASK_Simplified_Categorization"].dims[0]
        ]:
            raise AssertionError("Output time length does not equal input time length")

        out_binned = bin_cloud_top_phase4_latlon(out_profile, bin_size_deg=bin_size_deg)
        # out_binned.to_netcdf(out_path)
        groups = list(out_binned.groupby("time_bin",squeeze=False))
        datasets = [g for _, g in groups]
        # dt = np.datetime64("1970-01-01T00:00:00", "s") + np.timedelta64(int(t), "s")
        epoch = np.datetime64("1970-01-01T00:00:00", "s")

        paths = []
        for t, _ in groups:
            # t is seconds since epoch (int), rebuild datetime
            dt64 = epoch + np.timedelta64(int(t), "s")
            dt_obj = dt64.astype("datetime64[m]").astype(object)  # minute precision for formatting
            paths.append(os_path.join(out_path.parent, f"DD_CT_{dt_obj:%Y%m%d_%H%M}_{file_ind}.nc"))

        # paths = [
        #     os_path.join(out_path.parent, f"DD_CT_{np.datetime64(t, 'us').astype(object):%Y%m%d_%H%M}_{file_ind}.nc")
        #     for t, _ in groups
        # ]
        xr.save_mfdataset(datasets, paths, mode="w")

    # _print_summary(out_profile)
    # print(f"saved: {out_path}")

def _process_files_in_parallel(file_pairs: list[tuple[Path, Path]], bin_size_deg: float, n_workers) -> None:
    n_workers = min(cpu_count(), len(file_pairs), n_workers)
    if n_workers < 1:
        print("No files to process, no CPU available, or n_workers set to 0. Nothing to do.")
        return
    if n_workers < 2:
        print("Not enough files to process in parallel, only one CPU available or only n_workers=1. Processing sequentially.")
        for in_path, out_path in file_pairs:
            _process_file(in_path, out_path, bin_size_deg)
        return
    print(f"Processing {len(file_pairs)} files with {n_workers} parallel workers...")
    with Pool(processes=n_workers) as pool:
        pool.starmap(_process_file, [(file_pairs[j][0], file_pairs[j][1], bin_size_deg, j) for j in range(len(file_pairs))])
        # for _, out_path in file_pairs:
        #     subprocess.run(["cdo","splittime", out_path], check=True)

def merge_ddct_suffix_files_in_dir(day_dir: Path, workers: int = 8) -> None:
    """
    Merge/rename DD_CT files in one folder.

    Input files:
      DD_CT_YYYYMMDD_HHMM_<n>.nc
    Output files:
      DD_CT_YYYYMMDD_HHMM.nc
    """
    pattern = re.compile(r"^(DD_CT_\d{8}_\d{4})_(\d+)\.nc$")

    # 1) Collect groups by timestamp key
    groups: dict[str, list[Path]] = {}
    for f in day_dir.glob("DD_CT_*_*.nc"):
        m = pattern.match(f.name)
        if not m:
            continue
        key = m.group(0,1)
        groups.setdefault(key, []).append(f)
        # print(key)
    if not groups:
        print(f"[merge-ddct] no DD_CT_*_<n>.nc files in {day_dir}")
        return

    # 2) Worker for one timestamp key
    def process_key(key: str) -> None:
        files = sorted(groups[key])
        out = day_dir / f"{key[1]}.nc"
        tmp = day_dir / f"{key[1]}.tmp.nc"
        # print(len(files))
        try:
            if len(files) == 1:
                # No duplicate: just move to suffixless name
                files[0].replace(out)
                print(f"[rename ] {files[0].name} -> {out.name}")
                return

            # Duplicates: ensmean
            if tmp.exists():
                tmp.unlink()

            cmd = ["cdo", "-L", "ensmean"] + [str(f) for f in files] + [str(tmp)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or "cdo ensmean failed")

            tmp.replace(out)  # atomic rename in same folder

            # Remove old indexed files
            for f in files:
                if f.exists():
                    f.unlink()

            print(f"[merge] {len(files)} files -> {out.name}")

        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"[ERROR] {key}: {e}")

    # 3) Run keys in parallel
    keys = sorted(groups.keys())
    n_workers = max(1, min(workers, len(keys)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(process_key, keys))

    print(f"[merge-ddct] done: {day_dir}")



def move_wrong_day_files(day_dir: Path, base_path: Path, workers: int = 8) -> None:
    """
    Move DD_CT files from `day_dir` into the correct YYYY/MM/DD folder from the filename date.
    Only merge (cdo ensmean) if a file with the same suffixless name already exists in the destination.
    """
    rx = re.compile(r"^(DD_CT_(\d{8})_\d{4})(?:_(\d+))?\.nc$")  # key, yyyymmdd, optional index
    groups: dict[str, list[Path]] = {}

    for f in day_dir.glob("DD_CT_*.nc"):
        m = rx.match(f.name)
        if m:
            groups.setdefault(m.group(1), []).append(f)

    if not groups:
        print(f"[move-wrong-day] no DD_CT_*.nc in {day_dir}")
        return

    def dst_for(key: str, yyyymmdd: str) -> Path:
        d = datetime.strptime(yyyymmdd, "%Y%m%d")
        return base_path / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / f"{key}.nc"

    def cdo_ensmean(inputs: list[Path], out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.nc")
        tmp.unlink(missing_ok=True)
        res = subprocess.run(
            ["cdo", "-L", "ensmean", *map(str, inputs), str(tmp)],
            capture_output=True, text=True
        )
        if res.returncode:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(res.stderr.strip() or "cdo ensmean failed")
        tmp.replace(out)

    def process(key: str) -> None:
        srcs = sorted(groups[key])
        m = rx.match(srcs[0].name)
        yyyymmdd = m.group(2)
        dst = dst_for(key, yyyymmdd)

        # If it's already in the correct place, skip
        if any(s.resolve() == dst.resolve() for s in srcs):
            return

        try:
            if dst.exists():
                # Only case where we merge: same name exists in destination
                cdo_ensmean([dst] + srcs, dst)
                for s in srcs:
                    s.unlink(missing_ok=True)
                print(f"[merge-from-move] {len(srcs)+1} -> {dst.name} (collision)")
                return

            # No collision: do NOT merge.
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Prefer moving a suffixless file if present; otherwise move the first and delete the rest.
            preferred = next((s for s in srcs if s.name == f"{key}.nc"), srcs[0])
            preferred.replace(dst)
            for s in srcs:
                if s != preferred:
                    s.unlink(missing_ok=True)
            print(f"[move  ] {preferred.name} -> {dst}")

        except Exception as e:
            print(f"[ERROR] {key}: {e}")

    keys = sorted(groups)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(keys)))) as pool:
        list(pool.map(process, keys))

    print(f"[move-wrong-day] done: {day_dir}")


def process_one_day(input_dir: Path,output_dir: Path, bin_size_deg=0.05, n_workers = 4) -> None:
    # Example usage for processing one day of files in parallel.
    output_dir.mkdir(parents=True, exist_ok=True)  # assuming input_dir is like "2024-01-01"

    file_pairs = []
    for in_file in input_dir.glob("DARDAR-MASK_*.nc"):
        out_file = output_dir / f"Binned_{in_file.stem}.nc"
        file_pairs.append((in_file, out_file))

    _process_files_in_parallel(file_pairs, bin_size_deg=bin_size_deg , n_workers=n_workers)
    # for file_pair in file_pairs:
    #     print(f"Processed: {file_pair[0]} -> {file_pair[1]}")
    #     subprocess.run(["bash","/wolke_scratch/dnikolo/MSGI/DARDAR_analysis/merge_problematic.sh",""])

    # Merge DD_CT_*_<n>.nc -> DD_CT_*.nc once per day
    merge_ddct_suffix_files_in_dir(output_dir, workers=8)
    move_wrong_day_files(output_dir, base_path=output_dir.parent.parent.parent, workers=8)  # move to correct day folders if needed

def process_multiple_days(base_input_dir: Path, base_output_dir: Path, start_date: str, end_date: str,n_workers=4, bin_size_deg=0.05) -> None:
    # Example usage for processing multiple days in parallel.

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    day_dirs_inp = []
    day_dirs_out = []
    current_dt = start_dt
    while current_dt <= end_dt:
        day_dirs_inp.append(current_dt.strftime("%Y/%Y_%m_%d"))
        day_dirs_out.append(current_dt.strftime("%Y/%m/%d"))
        current_dt += timedelta(days=1)

    for day_inp, day_out in zip(day_dirs_inp, day_dirs_out):
        input_dir = base_input_dir / day_inp
        output_dir = base_output_dir / day_out
        process_one_day(input_dir, output_dir,bin_size_deg=bin_size_deg,n_workers=n_workers)

if __name__ == "__main__":
    # get_single_binned_cloud_top()
    process_multiple_days(
        base_input_dir=Path("/wolke_scratch/dnikolo/DARDAR_data/DARDAR-MASK"),
        base_output_dir=Path("/wolke_scratch/dnikolo/DARDAR_data/CPH_02"),
        start_date="20060601",
        end_date="20061231",
        bin_size_deg=0.2,
        n_workers=12
    )
    
