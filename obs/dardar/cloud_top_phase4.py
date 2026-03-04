#!/usr/bin/env python3
"""
Cloud-top phase extraction and daily aggregation for DARDAR-MASK.

Core capabilities:
1) Single-file mode (backward compatible):
   - Extract cloud-top phase per profile (4 classes).
   - Build lat/lon-binned phase statistics for that file.
2) Daily mode:
   - Discover/process all DARDAR-MASK files for one day.
   - Save per-file products (binned and optional profile-level).
   - Build one daily global lat/lon grid that stores all overpass timestamps
     in each grid cell using a NetCDF ragged representation.

Timestamp ragged representation in daily output:
- `cell_timestamp_start(lat_bin, lon_bin)` gives start index in
  `overpass_time_seconds` for the cell, or -1 if empty.
- `cell_timestamp_count(lat_bin, lon_bin)` gives number of timestamps in cell.
- For any occupied cell, timestamp list is:
  `overpass_time_seconds[start : start + count]`
- `overpass_cell_index` is included as an explicit flattened cell index
  (lat-major, then lon) for each timestamp element.

Examples:
- Backward-compatible single-file usage:
    python cloud_top_phase4.py DARDAR-MASK_2008122001458_10684_V2-23.nc
- Explicit single mode:
    python cloud_top_phase4.py single input.nc --bin-size 0.05
- Daily mode by directory+date:
    python cloud_top_phase4.py daily --input-dir /wolke_scratch/dnikolo/MSGI/DARDAR_analysis --date 20081220
- Daily mode by explicit files:
    python cloud_top_phase4.py daily --files file1.nc file2.nc --date 20081220
"""

from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


LOGGER = logging.getLogger("cloud_top_phase4")

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


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _detect_height_orientation(height_values: np.ndarray) -> bool:
    """Return True if height increases with index, False if decreases."""
    h = np.asarray(height_values, dtype=np.float64)
    finite = np.isfinite(h)
    if finite.sum() < 2:
        return True

    h_finite = h[finite]
    dh = np.diff(h_finite)
    dh = dh[np.isfinite(dh)]
    if dh.size == 0:
        return True

    return np.nanmedian(dh) > 0.0


def _height_to_km(height_values: np.ndarray, units: str | None) -> np.ndarray:
    """Convert CS_TRACK_Height to km using units when available, else heuristic."""
    h = np.asarray(height_values, dtype=np.float64)
    u = (units or "").strip().lower()

    if "km" in u:
        return h
    if "m" in u and "km" not in u:
        return h / 1000.0

    med_abs = np.nanmedian(np.abs(h)) if np.isfinite(h).any() else np.nan
    if np.isfinite(med_abs) and med_abs > 200.0:
        return h / 1000.0
    return h


def extract_cloud_top_phase4(ds: xr.Dataset) -> xr.Dataset:
    """
    Extract cloud-top phase in 4 classes for each along-track profile.

    Required inputs:
    - DARMASK_Simplified_Categorization(time, height)
    - CS_TRACK_Height(height)
    - CLOUDSAT_UTC_Time(time)

    Optional propagated inputs:
    - CLOUDSAT_Latitude(time)
    - CLOUDSAT_Longitude(time)
    """
    required = ["DARMASK_Simplified_Categorization", "CS_TRACK_Height", "CLOUDSAT_UTC_Time"]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    dmask = ds["DARMASK_Simplified_Categorization"]
    if dmask.ndim != 2:
        raise ValueError("DARMASK_Simplified_Categorization must be 2D (time, height).")

    time_dim, height_dim = dmask.dims
    n_time = dmask.sizes[time_dim]
    n_height = dmask.sizes[height_dim]

    dmask_float = np.asarray(dmask.values, dtype=np.float64)
    valid_numeric = np.isfinite(dmask_float)

    dmask_code = np.full(dmask_float.shape, -99, dtype=np.int16)
    dmask_code[valid_numeric] = np.rint(dmask_float[valid_numeric]).astype(np.int16)

    is_excluded = np.isin(dmask_code, EXCLUDE_CODES)
    candidate = valid_numeric & (~is_excluded)

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

    cloud_top_code = np.full(n_time, -99, dtype=np.int16)
    cloud_top_height_km = np.full(n_time, np.nan, dtype=np.float32)

    rows = np.where(has_candidate)[0]
    cols = top_idx[has_candidate]
    cloud_top_code[has_candidate] = dmask_code[rows, cols]

    height_km = _height_to_km(height_vals, height_var.attrs.get("units"))
    cloud_top_height_km[has_candidate] = height_km[cols].astype(np.float32)

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

    if "CLOUDSAT_Latitude" in ds.variables:
        out["CLOUDSAT_Latitude"] = ("time", np.asarray(ds["CLOUDSAT_Latitude"].values))
    if "CLOUDSAT_Longitude" in ds.variables:
        out["CLOUDSAT_Longitude"] = ("time", np.asarray(ds["CLOUDSAT_Longitude"].values))

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

    return out


def _global_edges(bin_size_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build fixed global bin edges for latitude/longitude."""
    if bin_size_deg <= 0:
        raise ValueError("bin_size_deg must be > 0.")

    # Include endpoint with small epsilon to avoid floating truncation.
    lat_edges = np.arange(-90.0, 90.0 + 0.5 * bin_size_deg, bin_size_deg, dtype=np.float64)
    lon_edges = np.arange(-180.0, 180.0 + 0.5 * bin_size_deg, bin_size_deg, dtype=np.float64)

    if lat_edges.size < 2 or lon_edges.size < 2:
        raise ValueError("Invalid global grid edges; check bin_size_deg.")

    if not np.isclose(lat_edges[-1], 90.0):
        lat_edges = np.append(lat_edges, 90.0)
    if not np.isclose(lon_edges[-1], 180.0):
        lon_edges = np.append(lon_edges, 180.0)

    return lat_edges, lon_edges


def _compute_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin indices using right-inclusive upper edge handling for exact boundary values."""
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = idx.astype(np.int64, copy=False)

    # Include exact max-edge value in last bin.
    at_upper = np.isfinite(values) & np.isclose(values, edges[-1])
    idx[at_upper] = edges.size - 2
    return idx


def _profile_to_global_grid_points(
    profile_ds: xr.Dataset,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return valid binned points as (lat_idx, lon_idx, phase, utc_raw).

    Keeps only finite lat/lon and in-grid rows.
    """
    if "CLOUDSAT_Latitude" not in profile_ds.variables or "CLOUDSAT_Longitude" not in profile_ds.variables:
        raise ValueError("Latitude/longitude missing in profile dataset.")

    lat = np.asarray(profile_ds["CLOUDSAT_Latitude"].values, dtype=np.float64)
    lon = np.asarray(profile_ds["CLOUDSAT_Longitude"].values, dtype=np.float64)
    phase = np.asarray(profile_ds["cloud_top_phase_4class"].values, dtype=np.int8)
    utc_raw = np.asarray(profile_ds["CLOUDSAT_UTC_Time"].values, dtype=np.float64)

    valid_geo = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid_geo):
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int8),
            np.array([], dtype=np.float64),
        )

    lat_valid = lat[valid_geo]
    lon_valid = lon[valid_geo]
    phase_valid = phase[valid_geo]
    utc_valid = utc_raw[valid_geo]

    lat_idx = _compute_bin_indices(lat_valid, lat_edges)
    lon_idx = _compute_bin_indices(lon_valid, lon_edges)
    in_bounds = (
        (lat_idx >= 0)
        & (lat_idx < lat_edges.size - 1)
        & (lon_idx >= 0)
        & (lon_idx < lon_edges.size - 1)
    )

    return (
        lat_idx[in_bounds].astype(np.int32),
        lon_idx[in_bounds].astype(np.int32),
        phase_valid[in_bounds],
        utc_valid[in_bounds],
    )


def _to_unix_epoch_seconds(
    utc_da: xr.DataArray,
    date_yyyymmdd: Optional[str] = None,
) -> np.ndarray:
    """
    Convert CLOUDSAT_UTC_Time to seconds since Unix epoch.

    Strategy:
    1) If units contain "since", use cftime conversion.
    2) Else if values look like seconds-of-day and date is provided, anchor to day start.
    3) Else if values look like hours-of-day and date is provided, convert to seconds-of-day.
    4) Else if values look like Unix epoch seconds, pass through.
    5) Otherwise return NaNs.
    """
    values = np.asarray(utc_da.values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(values)
    if not np.any(finite):
        return out

    units = str(utc_da.attrs.get("units", "")).strip()
    calendar_name = str(utc_da.attrs.get("calendar", "standard")).strip() or "standard"

    if "since" in units.lower():
        try:
            import cftime  # type: ignore

            converted = cftime.num2date(values[finite], units=units, calendar=calendar_name)
            epoch = np.array(
                [calendar.timegm(dt.utctimetuple()) + dt.microsecond / 1e6 for dt in converted],
                dtype=np.float64,
            )
            out[finite] = epoch
            return out
        except Exception as exc:
            LOGGER.debug("Failed cftime conversion for units '%s': %s", units, exc)

    min_v = np.nanmin(values[finite])
    max_v = np.nanmax(values[finite])

    if date_yyyymmdd is not None and 0.0 <= min_v and max_v <= 172800.0:
        day0 = datetime.strptime(date_yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
        day0_epoch = float(day0.timestamp())
        out[finite] = day0_epoch + values[finite]
        return out

    if date_yyyymmdd is not None and 0.0 <= min_v and max_v <= 48.0:
        day0 = datetime.strptime(date_yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
        day0_epoch = float(day0.timestamp())
        out[finite] = day0_epoch + values[finite] * 3600.0
        return out

    # Already epoch-like in seconds/ms/us/ns.
    if 9.0e8 <= min_v <= 2.0e10 and 9.0e8 <= max_v <= 2.0e10:
        out[finite] = values[finite]
        return out

    if 9.0e11 <= min_v <= 2.0e13 and 9.0e11 <= max_v <= 2.0e13:
        out[finite] = values[finite] / 1.0e3  # ms -> s
        return out

    if 9.0e14 <= min_v <= 2.0e16 and 9.0e14 <= max_v <= 2.0e16:
        out[finite] = values[finite] / 1.0e6  # us -> s
        return out

    if 9.0e17 <= min_v <= 2.0e19 and 9.0e17 <= max_v <= 2.0e19:
        out[finite] = values[finite] / 1.0e9  # ns -> s
        return out

    LOGGER.warning(
        "Could not infer absolute UTC conversion for CLOUDSAT_UTC_Time (units='%s'); timestamps set to NaN.",
        units,
    )
    return out


def bin_cloud_top_phase4_latlon(profile_ds: xr.Dataset, bin_size_deg: float = 0.05) -> xr.Dataset:
    """
    Bin per-profile cloud-top phase into latitude/longitude bins for one pass.

    Outputs:
    - mean_cloud_top_phase_liquid0_ice1: liquid=0, mixed=0.5, ice=1, -1 if no cloudy tops
    - class fractions among all observations in the bin
    - class counts and total counts
    """
    if bin_size_deg <= 0:
        raise ValueError("bin_size_deg must be > 0.")

    if "CLOUDSAT_Latitude" not in profile_ds.variables or "CLOUDSAT_Longitude" not in profile_ds.variables:
        raise ValueError("Latitude/longitude are required for binning but are missing.")

    lat = np.asarray(profile_ds["CLOUDSAT_Latitude"].values, dtype=np.float64)
    lon = np.asarray(profile_ds["CLOUDSAT_Longitude"].values, dtype=np.float64)
    phase = np.asarray(profile_ds["cloud_top_phase_4class"].values, dtype=np.int8)

    valid_geo = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid_geo):
        raise ValueError("No finite latitude/longitude values found for binning.")

    lat_valid = lat[valid_geo]
    lon_valid = lon[valid_geo]
    phase_valid = phase[valid_geo]

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

    lat_idx = _compute_bin_indices(lat_valid, lat_edges)
    lon_idx = _compute_bin_indices(lon_valid, lon_edges)
    in_bounds = (lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon)
    lat_idx = lat_idx[in_bounds]
    lon_idx = lon_idx[in_bounds]
    phase_valid = phase_valid[in_bounds]

    shape = (n_lat, n_lon)
    n_obs = np.zeros(shape, dtype=np.int32)
    n_no_cloud = np.zeros(shape, dtype=np.int32)
    n_liquid = np.zeros(shape, dtype=np.int32)
    n_ice = np.zeros(shape, dtype=np.int32)
    n_mixed = np.zeros(shape, dtype=np.int32)
    phase_sum_cloudy = np.zeros(shape, dtype=np.float64)

    np.add.at(n_obs, (lat_idx, lon_idx), 1)
    np.add.at(n_no_cloud, (lat_idx[phase_valid == PHASE_NO_CLOUD], lon_idx[phase_valid == PHASE_NO_CLOUD]), 1)
    np.add.at(n_liquid, (lat_idx[phase_valid == PHASE_LIQUID], lon_idx[phase_valid == PHASE_LIQUID]), 1)
    np.add.at(n_ice, (lat_idx[phase_valid == PHASE_ICE], lon_idx[phase_valid == PHASE_ICE]), 1)
    np.add.at(n_mixed, (lat_idx[phase_valid == PHASE_MIXED], lon_idx[phase_valid == PHASE_MIXED]), 1)

    score = np.full(phase_valid.shape, np.nan, dtype=np.float64)
    score[phase_valid == PHASE_LIQUID] = 0.0
    score[phase_valid == PHASE_MIXED] = 0.5
    score[phase_valid == PHASE_ICE] = 1.0
    cloudy = np.isfinite(score)
    np.add.at(phase_sum_cloudy, (lat_idx[cloudy], lon_idx[cloudy]), score[cloudy])

    n_cloudy = n_liquid + n_ice + n_mixed
    mean_phase = np.full(shape, -1.0, dtype=np.float32)
    cloudy_bins = n_cloudy > 0
    mean_phase[cloudy_bins] = (phase_sum_cloudy[cloudy_bins] / n_cloudy[cloudy_bins]).astype(np.float32)

    frac_no_cloud = np.full(shape, np.nan, dtype=np.float32)
    frac_liquid = np.full(shape, np.nan, dtype=np.float32)
    frac_ice = np.full(shape, np.nan, dtype=np.float32)
    frac_mixed = np.full(shape, np.nan, dtype=np.float32)
    obs_bins = n_obs > 0
    frac_no_cloud[obs_bins] = (n_no_cloud[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_liquid[obs_bins] = (n_liquid[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_ice[obs_bins] = (n_ice[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_mixed[obs_bins] = (n_mixed[obs_bins] / n_obs[obs_bins]).astype(np.float32)

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    out = xr.Dataset(
        data_vars={
            "mean_cloud_top_phase_liquid0_ice1": (("lat_bin", "lon_bin"), mean_phase),
            "frac_no_cloud": (("lat_bin", "lon_bin"), frac_no_cloud),
            "frac_liquid": (("lat_bin", "lon_bin"), frac_liquid),
            "frac_ice": (("lat_bin", "lon_bin"), frac_ice),
            "frac_mixed": (("lat_bin", "lon_bin"), frac_mixed),
            "n_obs": (("lat_bin", "lon_bin"), n_obs),
            "n_no_cloud": (("lat_bin", "lon_bin"), n_no_cloud),
            "n_liquid": (("lat_bin", "lon_bin"), n_liquid),
            "n_ice": (("lat_bin", "lon_bin"), n_ice),
            "n_mixed": (("lat_bin", "lon_bin"), n_mixed),
            "n_cloudy": (("lat_bin", "lon_bin"), n_cloudy),
        },
        coords={
            "lat_bin": lat_centers.astype(np.float32),
            "lon_bin": lon_centers.astype(np.float32),
            "lat_bin_edge": ("lat_bin_edge", lat_edges.astype(np.float32)),
            "lon_bin_edge": ("lon_bin_edge", lon_edges.astype(np.float32)),
        },
    )

    out["mean_cloud_top_phase_liquid0_ice1"].attrs.update(
        {
            "long_name": "Mean cloud-top phase score in bin (liquid=0, mixed=0.5, ice=1)",
            "no_cloud_or_no_cloudy_tops_value": -1.0,
            "valid_min": -1.0,
            "valid_max": 1.0,
        }
    )
    out["frac_no_cloud"].attrs["long_name"] = "Fraction of observations in bin classified as no_cloud"
    out["frac_liquid"].attrs["long_name"] = "Fraction of observations in bin classified as liquid"
    out["frac_ice"].attrs["long_name"] = "Fraction of observations in bin classified as ice"
    out["frac_mixed"].attrs["long_name"] = "Fraction of observations in bin classified as mixed"
    out["lat_bin"].attrs["units"] = "degrees_north"
    out["lon_bin"].attrs["units"] = "degrees_east"
    out["lat_bin_edge"].attrs["units"] = "degrees_north"
    out["lon_bin_edge"].attrs["units"] = "degrees_east"
    out.attrs["bin_size_degrees"] = float(bin_size_deg)
    out.attrs["source"] = "Binned from per-profile cloud-top phase classes"

    return out


def _assert_phase_output(profile_ds: xr.Dataset, expected_time: int) -> None:
    phase = np.asarray(profile_ds["cloud_top_phase_4class"].values, dtype=np.int8)
    if not np.isin(phase, [0, 1, 2, 3]).all():
        raise AssertionError("cloud_top_phase_4class contains values outside {0,1,2,3}")
    if profile_ds.sizes["time"] != expected_time:
        raise AssertionError("Output time length does not equal input time length")


def _print_profile_summary(out: xr.Dataset) -> None:
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


def _parse_yyyymmdd(value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}', expected YYYYMMDD") from exc
    return value

# EDIT: This inference is wrong 
def _infer_date_from_files(files: Sequence[Path]) -> Optional[str]:
    dates = set()
    for path in files:
        m = re.search(r"(\d{8})", path.name)
        if m is not None:
            dates.add(m.group(1))
    if len(dates) == 1:
        return next(iter(dates))
    return None


def _discover_daily_files(input_dir: Path, date_yyyymmdd: str) -> List[Path]:
    dt = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    y_str = dt.strftime("%Y")
    d_dash = dt.strftime("%Y-%m-%d")
    d_us = dt.strftime("%Y_%m_%d")

    candidate_dirs = [
        input_dir / y_str / d_dash,  # e.g. 2008/2008-12-20
        input_dir / y_str / d_us,    # e.g. 2008/2008_12_20
        input_dir / d_dash,          # optional flat style
        input_dir / d_us,            # optional flat style
        input_dir,                   # if files are directly here
    ]

    files: List[Path] = []
    for d in candidate_dirs:
        if d.exists():
            files.extend(sorted(d.glob(f"DARDAR-MASK_*.nc")))

    if not files:
        # Fallback recursive search from input_dir
        files = sorted(input_dir.rglob(f"DARDAR-MASK_*.nc"))

    # Exclude derived outputs to keep only source DARDAR-MASK files.
    files = [f for f in files if "_cloud_top_phase4" not in f.name]

    # De-duplicate while preserving sort
    files = sorted(set(files))
    
    return files

def build_daily_timestamp_grid(
    n_obs: np.ndarray,
    n_no_cloud: np.ndarray,
    n_liquid: np.ndarray,
    n_ice: np.ndarray,
    n_mixed: np.ndarray,
    phase_sum_cloudy: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    cell_ids: np.ndarray,
    timestamps_epoch_s: np.ndarray,
    bin_size_deg: float,
    date_yyyymmdd: str,
) -> xr.Dataset:
    """Build daily gridded dataset with ragged timestamp storage."""
    n_cloudy = n_liquid + n_ice + n_mixed

    mean_phase = np.full(n_obs.shape, -1.0, dtype=np.float32)
    cloudy_bins = n_cloudy > 0
    mean_phase[cloudy_bins] = (phase_sum_cloudy[cloudy_bins] / n_cloudy[cloudy_bins]).astype(np.float32)

    frac_no_cloud = np.full(n_obs.shape, np.nan, dtype=np.float32)
    frac_liquid = np.full(n_obs.shape, np.nan, dtype=np.float32)
    frac_ice = np.full(n_obs.shape, np.nan, dtype=np.float32)
    frac_mixed = np.full(n_obs.shape, np.nan, dtype=np.float32)
    obs_bins = n_obs > 0
    frac_no_cloud[obs_bins] = (n_no_cloud[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_liquid[obs_bins] = (n_liquid[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_ice[obs_bins] = (n_ice[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_mixed[obs_bins] = (n_mixed[obs_bins] / n_obs[obs_bins]).astype(np.float32)

    n_cells = n_obs.size
    flat_counts = n_obs.ravel().astype(np.int64)

    starts_flat = np.full(n_cells, -1, dtype=np.int64)
    nz = flat_counts > 0
    if np.any(nz):
        cumulative = np.cumsum(flat_counts, dtype=np.int64)
        starts_flat[nz] = cumulative[nz] - flat_counts[nz]

    if cell_ids.size == 0:
        sorted_cell_ids = np.array([], dtype=np.int64)
        sorted_ts = np.array([], dtype=np.float64)
    else:
        order = np.argsort(cell_ids, kind="stable")
        sorted_cell_ids = cell_ids[order]
        sorted_ts = timestamps_epoch_s[order]

    # Sanity check: ragged mapping consistency.
    if sorted_cell_ids.size != int(flat_counts.sum()):
        raise AssertionError("Ragged timestamp length does not match gridded observation counts")

    if sorted_cell_ids.size > 0:
        nz_idx = np.where(nz)[0]
        starts_nz = starts_flat[nz_idx]
        ends_nz = starts_nz + flat_counts[nz_idx] - 1
        if not np.all(sorted_cell_ids[starts_nz] == nz_idx):
            raise AssertionError("Ragged start indices are inconsistent")
        if not np.all(sorted_cell_ids[ends_nz] == nz_idx):
            raise AssertionError("Ragged end indices are inconsistent")

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    starts_2d = starts_flat.reshape(n_obs.shape)
    counts_2d = flat_counts.reshape(n_obs.shape).astype(np.int32)

    out = xr.Dataset(
        data_vars={
            "obs_count": (("lat_bin", "lon_bin"), n_obs.astype(np.int32)),
            "n_no_cloud": (("lat_bin", "lon_bin"), n_no_cloud.astype(np.int32)),
            "n_liquid": (("lat_bin", "lon_bin"), n_liquid.astype(np.int32)),
            "n_ice": (("lat_bin", "lon_bin"), n_ice.astype(np.int32)),
            "n_mixed": (("lat_bin", "lon_bin"), n_mixed.astype(np.int32)),
            "n_cloudy": (("lat_bin", "lon_bin"), n_cloudy.astype(np.int32)),
            "mean_cloud_top_phase_liquid0_ice1": (("lat_bin", "lon_bin"), mean_phase),
            "frac_no_cloud": (("lat_bin", "lon_bin"), frac_no_cloud),
            "frac_liquid": (("lat_bin", "lon_bin"), frac_liquid),
            "frac_ice": (("lat_bin", "lon_bin"), frac_ice),
            "frac_mixed": (("lat_bin", "lon_bin"), frac_mixed),
            "cell_timestamp_start": (("lat_bin", "lon_bin"), starts_2d),
            "cell_timestamp_count": (("lat_bin", "lon_bin"), counts_2d),
            "overpass_time_seconds": (("obs",), sorted_ts.astype(np.float64)),
            "overpass_cell_index": (("obs",), sorted_cell_ids.astype(np.int32)),
        },
        coords={
            "lat_bin": lat_centers.astype(np.float32),
            "lon_bin": lon_centers.astype(np.float32),
            "lat_bin_edge": ("lat_bin_edge", lat_edges.astype(np.float32)),
            "lon_bin_edge": ("lon_bin_edge", lon_edges.astype(np.float32)),
            "obs": np.arange(sorted_ts.size, dtype=np.int64),
        },
    )

    out["mean_cloud_top_phase_liquid0_ice1"].attrs.update(
        {
            "long_name": "Mean cloud-top phase score in bin (liquid=0, mixed=0.5, ice=1)",
            "no_cloud_or_no_cloudy_tops_value": -1.0,
            "valid_min": -1.0,
            "valid_max": 1.0,
        }
    )
    out["overpass_time_seconds"].attrs.update(
        {
            "long_name": "All overpass timestamps for occupied cells",
            "units": "seconds since 1970-01-01 00:00:00 UTC",
            "calendar": "standard",
        }
    )
    out["overpass_cell_index"].attrs.update(
        {
            "long_name": "Flattened cell index (lat-major then lon) for each overpass_time_seconds element",
            "comment": "cell_index = lat_index * n_lon + lon_index",
        }
    )
    out["cell_timestamp_start"].attrs.update(
        {
            "long_name": "Start index into overpass_time_seconds for this cell",
            "empty_cell_value": -1,
        }
    )
    out["cell_timestamp_count"].attrs.update(
        {
            "long_name": "Number of timestamp entries for this cell",
        }
    )

    out.attrs.update(
        {
            "title": "Daily DARDAR-MASK cloud-top phase grid with ragged overpass timestamps",
            "date_yyyymmdd": date_yyyymmdd,
            "bin_size_degrees": float(bin_size_deg),
            "ragged_timestamp_encoding": (
                "For each cell: start=cell_timestamp_start(lat,lon), "
                "count=cell_timestamp_count(lat,lon), "
                "timestamps=overpass_time_seconds[start:start+count]."
            ),
        }
    )

    return out


def process_day(
    files: Sequence[Path],
    date_yyyymmdd: str,
    out_dir: Path,
    bin_size_deg: float = 0.05,
    write_per_file_binned: bool = True,
    write_per_file_profile: bool = False,
) -> Dict[str, object]:
    """Process all daily files and write daily timestamp-grid NetCDF."""
    if len(files) == 0:
        raise ValueError("No input files provided for daily processing.")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = out_dir / "per_file"
    if write_per_file_binned or write_per_file_profile:
        per_file_dir.mkdir(parents=True, exist_ok=True)

    lat_edges, lon_edges = _global_edges(bin_size_deg)
    n_lat = lat_edges.size - 1
    n_lon = lon_edges.size - 1
    shape = (n_lat, n_lon)

    n_obs = np.zeros(shape, dtype=np.int64)
    n_no_cloud = np.zeros(shape, dtype=np.int64)
    n_liquid = np.zeros(shape, dtype=np.int64)
    n_ice = np.zeros(shape, dtype=np.int64)
    n_mixed = np.zeros(shape, dtype=np.int64)
    phase_sum_cloudy = np.zeros(shape, dtype=np.float64)

    cell_id_chunks: List[np.ndarray] = []
    ts_chunks: List[np.ndarray] = []

    processed = 0
    failed = 0
    total_profiles = 0

    for file_path in files:
        try:
            with xr.open_dataset(file_path) as ds_in:
                out_profile = extract_cloud_top_phase4(ds_in)
                expected_time = ds_in["DARMASK_Simplified_Categorization"].sizes[
                    ds_in["DARMASK_Simplified_Categorization"].dims[0]
                ]
                _assert_phase_output(out_profile, expected_time)

                total_profiles += out_profile.sizes["time"]

                if write_per_file_profile:
                    profile_path = per_file_dir / f"{file_path.stem}_cloud_top_phase4_profile.nc"
                    out_profile.to_netcdf(profile_path)

                if write_per_file_binned:
                    if "CLOUDSAT_Latitude" in out_profile.variables and "CLOUDSAT_Longitude" in out_profile.variables:
                        try:
                            out_binned = bin_cloud_top_phase4_latlon(out_profile, bin_size_deg=bin_size_deg)
                            binned_path = per_file_dir / f"{file_path.stem}_cloud_top_phase4_binned.nc"
                            out_binned.to_netcdf(binned_path)
                        except Exception as exc:
                            LOGGER.warning("Per-file binned output skipped for %s: %s", file_path.name, exc)
                    else:
                        LOGGER.warning("Per-file binned output skipped for %s: missing lat/lon", file_path.name)

                if "CLOUDSAT_Latitude" not in out_profile.variables or "CLOUDSAT_Longitude" not in out_profile.variables:
                    LOGGER.warning("Daily timestamp-grid contribution skipped for %s: missing lat/lon", file_path.name)
                    processed += 1
                    continue

                lat_idx, lon_idx, phase, utc_raw = _profile_to_global_grid_points(out_profile, lat_edges, lon_edges)
                if lat_idx.size == 0:
                    LOGGER.warning("No valid geo samples in-grid for %s", file_path.name)
                    processed += 1
                    continue

                utc_da = out_profile["CLOUDSAT_UTC_Time"]
                utc_full_epoch = _to_unix_epoch_seconds(utc_da, date_yyyymmdd=date_yyyymmdd)
                # Match masks used in _profile_to_global_grid_points
                lat_full = np.asarray(out_profile["CLOUDSAT_Latitude"].values, dtype=np.float64)
                lon_full = np.asarray(out_profile["CLOUDSAT_Longitude"].values, dtype=np.float64)
                valid_geo = np.isfinite(lat_full) & np.isfinite(lon_full)

                if np.any(valid_geo):
                    lat_valid = lat_full[valid_geo]
                    lon_valid = lon_full[valid_geo]
                    lat_valid_idx = _compute_bin_indices(lat_valid, lat_edges)
                    lon_valid_idx = _compute_bin_indices(lon_valid, lon_edges)
                    in_bounds = (
                        (lat_valid_idx >= 0)
                        & (lat_valid_idx < n_lat)
                        & (lon_valid_idx >= 0)
                        & (lon_valid_idx < n_lon)
                    )
                    utc_epoch = utc_full_epoch[valid_geo][in_bounds]
                else:
                    utc_epoch = np.array([], dtype=np.float64)

                if utc_epoch.size != lat_idx.size:
                    raise AssertionError("Timestamp vector does not match binned profile vector length")

                finite_time = np.isfinite(utc_epoch)
                if not np.all(finite_time):
                    dropped = int(np.sum(~finite_time))
                    LOGGER.warning("Dropped %d rows with invalid timestamp in %s", dropped, file_path.name)

                lat_idx = lat_idx[finite_time]
                lon_idx = lon_idx[finite_time]
                phase = phase[finite_time]
                utc_epoch = utc_epoch[finite_time]

                if lat_idx.size == 0:
                    processed += 1
                    continue

                np.add.at(n_obs, (lat_idx, lon_idx), 1)
                np.add.at(n_no_cloud, (lat_idx[phase == PHASE_NO_CLOUD], lon_idx[phase == PHASE_NO_CLOUD]), 1)
                np.add.at(n_liquid, (lat_idx[phase == PHASE_LIQUID], lon_idx[phase == PHASE_LIQUID]), 1)
                np.add.at(n_ice, (lat_idx[phase == PHASE_ICE], lon_idx[phase == PHASE_ICE]), 1)
                np.add.at(n_mixed, (lat_idx[phase == PHASE_MIXED], lon_idx[phase == PHASE_MIXED]), 1)

                score = np.full(phase.shape, np.nan, dtype=np.float64)
                score[phase == PHASE_LIQUID] = 0.0
                score[phase == PHASE_MIXED] = 0.5
                score[phase == PHASE_ICE] = 1.0
                cloudy = np.isfinite(score)
                np.add.at(phase_sum_cloudy, (lat_idx[cloudy], lon_idx[cloudy]), score[cloudy])

                cell_id = lat_idx.astype(np.int64) * n_lon + lon_idx.astype(np.int64)
                cell_id_chunks.append(cell_id)
                ts_chunks.append(utc_epoch.astype(np.float64))

                processed += 1

        except Exception as exc:
            failed += 1
            LOGGER.warning("Failed file %s: %s", file_path, exc)
            continue

    all_cell_ids = np.concatenate(cell_id_chunks) if cell_id_chunks else np.array([], dtype=np.int64)
    all_ts = np.concatenate(ts_chunks) if ts_chunks else np.array([], dtype=np.float64)

    daily_ds = build_daily_timestamp_grid(
        n_obs=n_obs,
        n_no_cloud=n_no_cloud,
        n_liquid=n_liquid,
        n_ice=n_ice,
        n_mixed=n_mixed,
        phase_sum_cloudy=phase_sum_cloudy,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
        cell_ids=all_cell_ids,
        timestamps_epoch_s=all_ts,
        bin_size_deg=bin_size_deg,
        date_yyyymmdd=date_yyyymmdd,
    )

    daily_path = out_dir / f"cloud_top_phase4_daily_{date_yyyymmdd}_timestamps_grid.nc"
    daily_ds.to_netcdf(daily_path)

    occupied_cells = int(np.sum(np.asarray(daily_ds["obs_count"].values) > 0))

    summary = {
        "files_discovered": len(files),
        "files_processed": processed,
        "files_failed": failed,
        "total_profiles": int(total_profiles),
        "occupied_cells": occupied_cells,
        "daily_output": str(daily_path),
        "per_file_dir": str(per_file_dir) if (write_per_file_binned or write_per_file_profile) else "",
    }
    return summary


def run_single_file(input_path: Path, output_path: Optional[Path], bin_size: float) -> Path:
    """Run single-file workflow (backward compatible)."""
    if output_path is None:
        out_path = input_path.with_name(f"{input_path.stem}_cloud_top_phase4_binned.nc")
    else:
        out_path = output_path

    with xr.open_dataset(input_path) as ds_in:
        out_profile = extract_cloud_top_phase4(ds_in)
        expected_time = ds_in["DARMASK_Simplified_Categorization"].sizes[
            ds_in["DARMASK_Simplified_Categorization"].dims[0]
        ]
        _assert_phase_output(out_profile, expected_time)
        out_binned = bin_cloud_top_phase4_latlon(out_profile, bin_size_deg=bin_size)
        out_binned.to_netcdf(out_path)

    _print_profile_summary(out_profile)
    print(f"saved: {out_path}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cloud-top phase extraction and daily aggregation for DARDAR-MASK."
    )
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("single", help="Process one DARDAR-MASK file")
    single.add_argument("input", help="Input DARDAR-MASK NetCDF file")
    single.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output NetCDF path (default: <input_basename>_cloud_top_phase4_binned.nc)",
    )
    single.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Latitude/longitude bin size in degrees (default: 0.05)",
    )
    single.add_argument("--verbose", action="store_true", help="Enable debug logging")

    daily = subparsers.add_parser("daily", help="Process a full day and build daily timestamp grid")
    daily.add_argument(
        "--input-dir",
        default="/wolke_scratch/dnikolo/MSGI/DARDAR_analysis",
        help="Base directory containing YYYY/YYYY-MM-DD subfolders",
    )
    daily.add_argument("--date", default=None, help="Date as YYYYMMDD")
    daily.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit list of DARDAR-MASK files for one day",
    )
    daily.add_argument("--out-dir", default=".", help="Output directory for daily and per-file products")
    daily.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Latitude/longitude bin size in degrees (default: 0.05)",
    )
    daily.add_argument(
        "--no-per-file-binned",
        action="store_true",
        help="Disable writing per-file binned products",
    )
    daily.add_argument(
        "--write-per-file-profile",
        action="store_true",
        help="Also write per-file profile-level outputs",
    )
    daily.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args_in = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility:
    # python cloud_top_phase4.py input.nc -> interpreted as single mode.
    if args_in and args_in[0] not in {"single", "daily", "-h", "--help"}:
        args_in = ["single"] + args_in

    parser = _build_parser()
    args = parser.parse_args(args_in)

    if args.command is None:
        parser.print_help()
        return

    _setup_logging(verbose=bool(getattr(args, "verbose", False)))

    if args.command == "single":
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output is not None else None
        run_single_file(in_path, out_path, bin_size=float(args.bin_size))
        return

    if args.command == "daily":
        if args.files:
            files = [Path(f) for f in args.files]
            date_yyyymmdd = args.date
            if date_yyyymmdd is None:
                inferred = _infer_date_from_files(files)
                if inferred is None:
                    raise ValueError("--date is required when using --files unless one unique YYYYMMDD is inferable")
                date_yyyymmdd = inferred
        else:
            if args.date is None:
                raise ValueError("--date is required when discovering files from --input-dir")
            date_yyyymmdd = args.date
            files = _discover_daily_files(Path(args.input_dir), _parse_yyyymmdd(date_yyyymmdd))
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No DARDAR-MASK files found for date={date_yyyymmdd} under input-dir={args.input_dir}. "
                    "Expected names like DARDAR-MASK_YYYYMMDD*.nc"
                )

        date_yyyymmdd = _parse_yyyymmdd(date_yyyymmdd)

        if len(files) == 0:
            raise FileNotFoundError("No DARDAR-MASK files found for requested day")

        summary = process_day(
            files=files,
            date_yyyymmdd=date_yyyymmdd,
            out_dir=Path(args.out_dir),
            bin_size_deg=float(args.bin_size),
            write_per_file_binned=not bool(args.no_per_file_binned),
            write_per_file_profile=bool(args.write_per_file_profile),
        )

        print(f"files_discovered: {summary['files_discovered']}")
        print(f"files_processed: {summary['files_processed']}")
        print(f"files_failed: {summary['files_failed']}")
        print(f"total_profiles: {summary['total_profiles']}")
        print(f"occupied_grid_cells: {summary['occupied_cells']}")
        print(f"daily_output: {summary['daily_output']}")
        if summary["per_file_dir"]:
            print(f"per_file_output_dir: {summary['per_file_dir']}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
