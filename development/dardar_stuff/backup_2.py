#!/usr/bin/env python3
"""
Extract cloud-top phase (4 classes) from a DARDAR-MASK NetCDF file.

This script identifies the highest valid cloud bin in each along-track profile
from `DARMASK_Simplified_Categorization(time, height)`, then maps the top-bin
DARMASK code to 4 cloud-top phase classes:

- 0: no_cloud
- 1: liquid
- 2: ice
- 3: mixed

Mapping rationale:
- Candidate cloud-top search excludes non-cloud / unusable codes
  (missing, clear sky, aerosols, etc.).
- The top candidate code is then mapped to liquid/ice/mixed using the requested
  code groups.
- If no candidate exists, class is `no_cloud`.
- If a candidate exists but the code is unmapped, class is set to `no_cloud`
  and a QA flag is raised.

Usage:
    python script.py input.nc [-o output.nc] [--bin-size 0.05]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr


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


def extract_cloud_top_phase4(ds: xr.Dataset) -> xr.Dataset:
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

    return out


def bin_cloud_top_phase4_latlon(
    profile_ds: xr.Dataset, bin_size_deg: float = 0.05
) -> xr.Dataset:
    """
    Bin per-profile cloud-top phase into latitude/longitude bins.

    Outputs per-bin:
    - mean_cloud_top_phase_liquid0_ice1:
      liquid=0, mixed=0.5, ice=1; set to -1 where no cloudy tops exist.
    - class fractions among all observations in each bin:
      frac_no_cloud, frac_liquid, frac_ice, frac_mixed
    - observation counts:
      n_obs, n_no_cloud, n_liquid, n_ice, n_mixed, n_cloudy
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

    lat_idx = np.digitize(lat_valid, lat_edges) - 1
    lon_idx = np.digitize(lon_valid, lon_edges) - 1
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

    # Mean phase score over cloudy-only tops: liquid=0, mixed=0.5, ice=1.
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


def main() -> None:
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


if __name__ == "__main__":
    main()
