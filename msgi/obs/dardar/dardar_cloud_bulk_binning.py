from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from os import path as os_path
from pathlib import Path

import numpy as np
import xarray as xr


PHASE_NO_CLOUD = np.int8(0)
PHASE_LIQUID = np.int8(1)
PHASE_ICE = np.int8(2)
PHASE_MIXED = np.int8(3)

PHASE_LABELS = np.array(["no_cloud", "liquid", "ice", "mixed"], dtype=object)

EXCLUDE_CODES = np.array([-99, -9, -2, -1, 0, 6, 8], dtype=np.int16)
LIQUID_CODES = np.array([3, 5, 7, 11, 12, 13, 15], dtype=np.int16)
ICE_CODES = np.array([1, 2, 9, 10], dtype=np.int16)
MIXED_CODES = np.array([4, 14], dtype=np.int16)

DARDAR_NHEIGHTS = 436
DARDAR_DZ_M = 60.0
DARDAR_HEIGHT_TOP_M = 25080.0
HEIGHT_RANGES_M = ((-240.0, -1200.0), (-1380.0, -2100.0))


def build_bulk_level_offsets(height_ranges_m: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Build cloud-bulk layer offset bounds from cloud-top-relative height ranges."""
    layers = []
    layer_depth_m = DARDAR_DZ_M * 2.0
    for upper_m, lower_m in height_ranges_m:
        edges = np.arange(upper_m, lower_m - DARDAR_DZ_M, -layer_depth_m, dtype=np.float32)
        layers.append(np.column_stack((edges[:-1], edges[1:])))
    return np.concatenate(layers, axis=0).astype(np.float32)


BULK_LEVEL_OFFSETS_M = build_bulk_level_offsets(HEIGHT_RANGES_M)


def reconstruct_height(n_levels: int = DARDAR_NHEIGHTS) -> np.ndarray:
    """Reconstruct the nominal DARDAR-CLOUD height axis in meters."""
    return np.arange(DARDAR_HEIGHT_TOP_M, DARDAR_HEIGHT_TOP_M - n_levels * DARDAR_DZ_M, -DARDAR_DZ_M)


def _detect_height_orientation(height_values: np.ndarray) -> bool:
    """Return True if height increases with index, False if it decreases."""
    height = np.asarray(height_values, dtype=np.float64)
    finite = np.isfinite(height)
    if finite.sum() < 2:
        return False

    diffs = np.diff(height[finite])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return False
    return np.nanmedian(diffs) > 0.0


def _prepare_height_m(height_var: xr.DataArray, n_height: int) -> np.ndarray:
    """Convert height to meters, reconstructing the nominal axis if needed."""
    height_raw = np.asarray(height_var.values, dtype=np.float64)
    units = (height_var.attrs.get("units") or "").strip().lower()

    if "km" in units:
        height_m = height_raw * 1000.0
    else:
        height_m = height_raw.copy()

    if height_m.ndim != 1 or height_m.size != n_height:
        raise ValueError("height must be a 1D coordinate matching the DARDAR height dimension")

    finite = np.isfinite(height_m)
    if finite.sum() < n_height:
        fallback = reconstruct_height(n_height)
        height_m = np.where(finite, height_m, fallback)

    return height_m


def _phase_score_from_code(code: np.ndarray) -> np.ndarray:
    """Map top-cloud DARMASK codes to DD_CT-style phase scores."""
    score = np.full(code.shape, np.nan, dtype=np.float64)
    score[np.isin(code, LIQUID_CODES)] = 1.0
    score[np.isin(code, MIXED_CODES)] = 1.5
    score[np.isin(code, ICE_CODES)] = 2.0
    return score


def _phase_class_from_code(code: np.ndarray) -> np.ndarray:
    """Map top-cloud DARMASK codes to the 4 DD_CT phase classes."""
    phase = np.full(code.shape, PHASE_NO_CLOUD, dtype=np.int8)
    phase[np.isin(code, LIQUID_CODES)] = PHASE_LIQUID
    phase[np.isin(code, ICE_CODES)] = PHASE_ICE
    phase[np.isin(code, MIXED_CODES)] = PHASE_MIXED
    return phase


def _to_datetime64_ns(time_var: xr.DataArray) -> np.ndarray:
    """Return observation time as numpy datetime64[ns]."""
    values = np.asarray(time_var.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")

    units = time_var.attrs.get("units", "")
    match = re.search(r"since\s+(\d{4})\s+(\d{2})\s+(\d{2})", units)
    if not match:
        raise ValueError(f"Could not decode time units: {units!r}")

    year, month, day = map(int, match.groups())
    base = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T00:00:00", "ns")
    seconds = np.asarray(values, dtype=np.float64)
    out = np.full(seconds.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = np.isfinite(seconds)
    out[valid] = base + np.rint(seconds[valid] * 1e9).astype("timedelta64[ns]")
    return out
    


def round_ns_timestamps(ts: np.ndarray, dt_min: int = 15) -> np.ndarray:
    """Round numpy datetime64[ns] values to the nearest dt_min minutes."""
    ts = np.asarray(ts, dtype="datetime64[ns]")
    step = np.timedelta64(dt_min, "m").astype("timedelta64[ns]").astype(np.int64)
    ns = ts.astype("datetime64[ns]").astype(np.int64)

    rounded_ns = np.empty_like(ns)
    is_nat = np.isnat(ts)
    rounded_ns[is_nat] = 0
    rounded_ns[~is_nat] = ((ns[~is_nat] + step // 2) // step) * step

    out = np.empty_like(ts)
    out[is_nat] = np.datetime64("NaT", "ns")
    out[~is_nat] = rounded_ns[~is_nat].astype("datetime64[ns]")
    return out


def dt_arange(start: np.datetime64, end: np.datetime64, dt_min: int = 15) -> np.ndarray:
    """Build a regular datetime64[ns] range including both endpoints."""
    start = np.asarray(start, dtype="datetime64[ns]")
    end = np.asarray(end, dtype="datetime64[ns]")
    step = np.timedelta64(dt_min, "m").astype("timedelta64[ns]")
    return np.arange(start, end + step, step)


def build_regular_edges(coord_min: float, coord_max: float, delta: float) -> np.ndarray:
    """Build regular bin edges while preserving the requested max bound."""
    edges = np.arange(coord_min, coord_max + 0.5 * delta, delta, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Could not construct coordinate edges")

    tol = max(1e-10, abs(delta) * 1e-8)
    if edges[-1] < coord_max - tol:
        edges = np.append(edges, coord_max)
    else:
        edges[-1] = coord_max
    edges[0] = coord_min
    return edges


def scatter_2d_to_3d_by_time(var2d: np.ndarray, rounded_time2d: np.ndarray, time_values: np.ndarray, fill_value) -> np.ndarray:
    """Scatter a 2D field into a 3D time cube using rounded_time2d."""
    n_time = time_values.size
    n_lat, n_lon = var2d.shape
    cube = np.full((n_time, n_lat, n_lon), fill_value, dtype=var2d.dtype)

    valid = ~np.isnat(rounded_time2d)
    if n_time == 0 or not np.any(valid):
        return cube

    lat_idx, lon_idx = np.where(valid)
    tvals = rounded_time2d[valid].astype("datetime64[ns]")
    tidx = np.searchsorted(time_values, tvals)
    in_range = (tidx >= 0) & (tidx < n_time) & (time_values[tidx] == tvals)
    cube[tidx[in_range], lat_idx[in_range], lon_idx[in_range]] = var2d[lat_idx[in_range], lon_idx[in_range]]
    return cube


def scatter_3d_to_4d_by_time(
    var3d: np.ndarray,
    rounded_time2d: np.ndarray,
    time_values: np.ndarray,
    fill_value,
) -> np.ndarray:
    """Scatter a (bulk_level, lat, lon) field into (time, bulk_level, lat, lon)."""
    n_level, n_lat, n_lon = var3d.shape
    n_time = time_values.size
    cube = np.full((n_time, n_level, n_lat, n_lon), fill_value, dtype=var3d.dtype)

    for level in range(n_level):
        cube[:, level, :, :] = scatter_2d_to_3d_by_time(
            var3d[level], rounded_time2d, time_values, fill_value=fill_value
        )

    return cube


def extract_cloud_bulk_profiles(ds: xr.Dataset) -> xr.Dataset:
    """Extract cloud-top and cloud-bulk profile quantities from a DARDAR-CLOUD file."""
    required = [
        "DARMASK_Simplified_Categorization",
        "height",
        "time",
        "latitude",
        "longitude",
        "iwc",
        "temperature",
    ]
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    dmask = ds["DARMASK_Simplified_Categorization"]
    iwc_var = ds["iwc"]
    temp_var = ds["temperature"]
    if dmask.ndim != 2 or iwc_var.ndim != 2 or temp_var.ndim != 2:
        raise ValueError("DARMASK_Simplified_Categorization, iwc, and temperature must all be 2D")

    time_dim, height_dim = dmask.dims
    n_time = dmask.sizes[time_dim]
    n_height = dmask.sizes[height_dim]
    n_bulk = BULK_LEVEL_OFFSETS_M.shape[0]

    height_m = _prepare_height_m(ds["height"], n_height)
    height_increasing = _detect_height_orientation(height_m)
    lower_step = -1 if height_increasing else 1

    dmask_float = np.asarray(dmask.values, dtype=np.float64)
    valid_dmask = np.isfinite(dmask_float)
    dmask_code = np.full(dmask_float.shape, -99, dtype=np.int16)
    dmask_code[valid_dmask] = np.rint(dmask_float[valid_dmask]).astype(np.int16)
    cloud_candidate = valid_dmask & (~np.isin(dmask_code, EXCLUDE_CODES))

    idx = np.broadcast_to(np.arange(n_height, dtype=np.int32), (n_time, n_height))
    if height_increasing:
        top_idx = np.where(cloud_candidate, idx, -1).max(axis=1)
        has_top = top_idx >= 0
    else:
        top_idx = np.where(cloud_candidate, idx, n_height).min(axis=1)
        has_top = top_idx < n_height

    top_code = np.full(n_time, -99, dtype=np.int16)
    valid_rows = np.where(has_top)[0]
    top_code[has_top] = dmask_code[valid_rows, top_idx[has_top]]
    top_phase = _phase_class_from_code(top_code)
    top_phase_score = _phase_score_from_code(top_code).astype(np.float32)

    iwc = np.asarray(iwc_var.values, dtype=np.float64)
    temp = np.asarray(temp_var.values, dtype=np.float64)
    iwc_fill = iwc_var.attrs.get("_FillValue", iwc_var.attrs.get("missing_value", -999.0))
    temp_fill = temp_var.attrs.get("_FillValue", temp_var.attrs.get("missing_value", -999.0))

    if np.isfinite(iwc_fill):
        iwc[np.isclose(iwc, iwc_fill)] = np.nan
    iwc[~np.isfinite(iwc)] = np.nan
    iwc[iwc <= 0.0] = np.nan

    if np.isfinite(temp_fill):
        temp[np.isclose(temp, temp_fill)] = np.nan
    temp[~np.isfinite(temp)] = np.nan

    top_iwc = np.full(n_time, np.nan, dtype=np.float32)
    top_height = np.full(n_time, np.nan, dtype=np.float32)
    top_temp = np.full(n_time, np.nan, dtype=np.float32)
    top_n_valid = np.zeros(n_time, dtype=np.int16)

    bulk_iwc = np.full((n_time, n_bulk), np.nan, dtype=np.float32)
    bulk_height = np.full((n_time, n_bulk), np.nan, dtype=np.float32)
    bulk_temp = np.full((n_time, n_bulk), np.nan, dtype=np.float32)
    bulk_n_valid = np.zeros((n_time, n_bulk), dtype=np.int16)

    for row in valid_rows:
        top = int(top_idx[row])

        top_pair = [top, top + lower_step]
        valid_top_idx = [col for col in top_pair if 0 <= col < n_height]
        if valid_top_idx:
            top_iwc_vals = iwc[row, valid_top_idx]
            top_height_vals = height_m[valid_top_idx]
            top_temp_vals = temp[row, valid_top_idx]

            top_valid = np.isfinite(top_iwc_vals)
            top_n_valid[row] = np.int16(np.count_nonzero(top_valid))
            if top_n_valid[row] > 0:
                top_iwc[row] = np.float32(np.nanmean(top_iwc_vals[top_valid]))
                top_height[row] = np.float32(np.nanmean(top_height_vals[top_valid]) / 1000.0)
                top_temp[row] = np.float32(np.nanmean(top_temp_vals[top_valid]))
        for level, (top_offset_m, bottom_offset_m) in enumerate(BULK_LEVEL_OFFSETS_M):
            start = int(round(abs(top_offset_m) / DARDAR_DZ_M))
            stop = int(round(abs(bottom_offset_m) / DARDAR_DZ_M))

            pair = [
                top + lower_step * idx
                for idx in range(start, stop)
                if 0 <= top + lower_step * idx < n_height
            ]
            if (not pair) or (len(pair)!=2):
                continue
            valid_pair_idx=pair

            iwc_vals = iwc[row, valid_pair_idx]
            finite = np.isfinite(iwc_vals)
            bulk_n_valid[row, level] = np.int16(np.count_nonzero(finite))
            if bulk_n_valid[row, level] == 0:
                continue

            bulk_iwc[row, level] = np.float32(np.nanmean(iwc_vals[finite]))
            bulk_height[row, level] = np.float32(np.nanmean(height_m[valid_pair_idx][finite]) / 1000.0)
            bulk_temp[row, level] = np.float32(np.nanmean(temp[row, valid_pair_idx][finite]))

    out = xr.Dataset(
        data_vars={
            "cloud_top_code": (("time",), top_code),
            "cloud_top_phase_4class": (("time",), top_phase),
            "cloud_top_phase_4class_str": (("time",), PHASE_LABELS[top_phase]),
            "cb_top_cph_score": (("time",), top_phase_score),
            "cb_top_iwc": (("time",), top_iwc),
            "cb_top_height_km": (("time",), top_height),
            "cb_top_temp_k": (("time",), top_temp),
            "cb_top_n_valid_levels": (("time",), top_n_valid),
            "cb_iwc": (("time", "bulk_level"), bulk_iwc),
            "cb_height_km": (("time", "bulk_level"), bulk_height),
            "cb_temp_k": (("time", "bulk_level"), bulk_temp),
            "cb_n_valid_levels": (("time", "bulk_level"), bulk_n_valid),
            "bulk_level_offset_top": (("bulk_level",), BULK_LEVEL_OFFSETS_M[:, 0]),
            "bulk_level_offset_bottom": (("bulk_level",), BULK_LEVEL_OFFSETS_M[:, 1]),
            "time": (("time",), _to_datetime64_ns(ds["time"])),
            "latitude": (("time",), np.asarray(ds["latitude"].values, dtype=np.float64)),
            "longitude": (("time",), np.asarray(ds["longitude"].values, dtype=np.float64)),
        },
        coords={
            "time_index": ("time", np.arange(n_time, dtype=np.int32)),
            "bulk_level": ("bulk_level", np.arange(n_bulk, dtype=np.int16)),
        },
    )

    out["cloud_top_phase_4class"].attrs.update(
        {
            "long_name": "Cloud-top phase (4 classes)",
            "flag_values": np.array([0, 1, 2, 3], dtype=np.int8),
            "flag_meanings": "no_cloud liquid ice mixed",
        }
    )
    out["cb_top_cph_score"].attrs.update(
        {
            "long_name": "Cloud-top phase score",
            "valid_min": np.float32(1.0),
            "valid_max": np.float32(2.0),
            "comment": "Computed for cloudy tops only: liquid=1, mixed=1.5, ice=2.",
        }
    )
    out["cb_top_iwc"].attrs.update({"long_name": "Cloud-top IWC", "units": "kg m-3"})
    out["cb_top_height_km"].attrs.update({"long_name": "Cloud-top layer height", "units": "km"})
    out["cb_top_temp_k"].attrs.update({"long_name": "Cloud-top layer temperature", "units": "K"})
    out["bulk_level"].attrs.update(
        {
            "long_name": "Cloud-bulk layer index below cloud top",
            "comment": f"0 is the shallowest bulk layer, {n_bulk - 1} is the deepest bulk layer.",
        }
    )
    out["bulk_level_offset_top"].attrs.update(
        {
            "long_name": "Upper bound of cloud-bulk layer relative to detected cloud-top cell",
            "units": "m",
        }
    )
    out["bulk_level_offset_bottom"].attrs.update(
        {
            "long_name": "Lower bound of cloud-bulk layer relative to detected cloud-top cell",
            "units": "m",
        }
    )
    out["cb_iwc"].attrs.update({"long_name": "Cloud-bulk IWC", "units": "kg m-3"})
    out["cb_height_km"].attrs.update({"long_name": "Cloud-bulk layer height", "units": "km"})
    out["cb_temp_k"].attrs.update({"long_name": "Cloud-bulk layer temperature", "units": "K"})
    out["cb_top_n_valid_levels"].attrs.update({"long_name": "Number of valid source levels used for cloud-top averaging"})
    out["cb_n_valid_levels"].attrs.update({"long_name": "Number of valid source levels used for cloud-bulk averaging"})
    out.attrs["source"] = "Derived from DARDAR-CLOUD iwc/temperature and DARMASK_Simplified_Categorization"
    out.attrs["height_index_increasing"] = str(bool(height_increasing))
    return out


def _stats_from_sum(sum_arr: np.ndarray, sq_sum_arr: np.ndarray, count_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and std arrays from sum, squared sum, and counts."""
    mean = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    std = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    valid = count_arr > 0
    if np.any(valid):
        mean64 = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        mean64[valid] = sum_arr[valid] / count_arr[valid]
        mean[valid] = mean64[valid].astype(np.float32)
        var = np.full(sum_arr.shape, np.nan, dtype=np.float64)
        var[valid] = sq_sum_arr[valid] / count_arr[valid] - mean64[valid] ** 2
        var[valid] = np.maximum(var[valid], 0.0)
        std[valid] = np.sqrt(var[valid]).astype(np.float32)
    return mean, std


def bin_cloud_bulk_latlon(
    profile_ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
) -> xr.Dataset:
    """Aggregate cloud-top and cloud-bulk statistics on a regular lat-lon grid."""
    if dlat <= 0.0 or dlon <= 0.0:
        raise ValueError("dlat and dlon must both be > 0")
    if lat_max <= lat_min or lon_max <= lon_min:
        raise ValueError("Grid bounds must satisfy min < max")

    lat_edges = build_regular_edges(lat_min, lat_max, dlat)
    lon_edges = build_regular_edges(lon_min, lon_max, dlon)
    if lat_edges.size < 2 or lon_edges.size < 2:
        raise ValueError("Could not construct latitude/longitude bin edges")

    lat = np.asarray(profile_ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(profile_ds["longitude"].values, dtype=np.float64)
    time = np.asarray(profile_ds["time"].values, dtype="datetime64[ns]")
    top_phase = np.asarray(profile_ds["cb_top_cph_score"].values, dtype=np.float64)
    top_iwc = np.asarray(profile_ds["cb_top_iwc"].values, dtype=np.float64)
    top_height = np.asarray(profile_ds["cb_top_height_km"].values, dtype=np.float64)
    top_temp = np.asarray(profile_ds["cb_top_temp_k"].values, dtype=np.float64)
    top_n_valid = np.asarray(profile_ds["cb_top_n_valid_levels"].values, dtype=np.int16)
    bulk_iwc = np.asarray(profile_ds["cb_iwc"].values, dtype=np.float64)
    bulk_height = np.asarray(profile_ds["cb_height_km"].values, dtype=np.float64)
    bulk_temp = np.asarray(profile_ds["cb_temp_k"].values, dtype=np.float64)
    bulk_n_valid = np.asarray(profile_ds["cb_n_valid_levels"].values, dtype=np.int16)

    valid_geo = np.isfinite(lat) & np.isfinite(lon) & (~np.isnat(time))
    if not np.any(valid_geo):
        raise ValueError("No finite latitude/longitude/time values found for binning")

    lat = lat[valid_geo]
    lon = lon[valid_geo]
    time = time[valid_geo]
    top_phase = top_phase[valid_geo]
    top_iwc = top_iwc[valid_geo]
    top_height = top_height[valid_geo]
    top_temp = top_temp[valid_geo]
    top_n_valid = top_n_valid[valid_geo]
    bulk_iwc = bulk_iwc[valid_geo]
    bulk_height = bulk_height[valid_geo]
    bulk_temp = bulk_temp[valid_geo]
    bulk_n_valid = bulk_n_valid[valid_geo]

    n_lat = lat_edges.size - 1
    n_lon = lon_edges.size - 1
    n_bulk = bulk_iwc.shape[1]

    lat_idx = np.digitize(lat, lat_edges) - 1
    lon_idx = np.digitize(lon, lon_edges) - 1
    in_bounds = (lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon)
    if not np.any(in_bounds):
        raise ValueError("No observations fall inside the requested output grid")

    lat_idx = lat_idx[in_bounds]
    lon_idx = lon_idx[in_bounds]
    time = time[in_bounds]
    top_phase = top_phase[in_bounds]
    top_iwc = top_iwc[in_bounds]
    top_height = top_height[in_bounds]
    top_temp = top_temp[in_bounds]
    top_n_valid = top_n_valid[in_bounds]
    bulk_iwc = bulk_iwc[in_bounds]
    bulk_height = bulk_height[in_bounds]
    bulk_temp = bulk_temp[in_bounds]
    bulk_n_valid = bulk_n_valid[in_bounds]

    shape2d = (n_lat, n_lon)
    shape3d = (n_bulk, n_lat, n_lon)

    obs_count = np.zeros(shape2d, dtype=np.int32)
    time_sum = np.zeros(shape2d, dtype="timedelta64[ns]")

    np.add.at(obs_count, (lat_idx, lon_idx), 1)
    time_ref = time.min()
    np.add.at(time_sum, (lat_idx, lon_idx), time - time_ref)

    mean_time = np.full(shape2d, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    has_obs = obs_count > 0
    mean_time[has_obs] = (time_sum[has_obs] / obs_count[has_obs] + time_ref).astype("datetime64[ns]")
    rounded_mean_time = round_ns_timestamps(mean_time, dt_min=15)

    top_valid = np.isfinite(top_iwc)
    top_phase_valid = np.isfinite(top_phase)
    top_sum_iwc = np.zeros(shape2d, dtype=np.float64)
    top_sq_iwc = np.zeros(shape2d, dtype=np.float64)
    top_sum_height = np.zeros(shape2d, dtype=np.float64)
    top_sq_height = np.zeros(shape2d, dtype=np.float64)
    top_sum_temp = np.zeros(shape2d, dtype=np.float64)
    top_sq_temp = np.zeros(shape2d, dtype=np.float64)
    top_sum_phase = np.zeros(shape2d, dtype=np.float64)
    top_n_obs = np.zeros(shape2d, dtype=np.int32)
    top_n_phase = np.zeros(shape2d, dtype=np.int32)
    top_n_valid_levels_sum = np.zeros(shape2d, dtype=np.float64)

    if np.any(top_valid):
        np.add.at(top_sum_iwc, (lat_idx[top_valid], lon_idx[top_valid]), top_iwc[top_valid])
        np.add.at(top_sq_iwc, (lat_idx[top_valid], lon_idx[top_valid]), top_iwc[top_valid] ** 2)
        np.add.at(top_sum_height, (lat_idx[top_valid], lon_idx[top_valid]), top_height[top_valid])
        np.add.at(top_sq_height, (lat_idx[top_valid], lon_idx[top_valid]), top_height[top_valid] ** 2)
        np.add.at(top_sum_temp, (lat_idx[top_valid], lon_idx[top_valid]), top_temp[top_valid])
        np.add.at(top_sq_temp, (lat_idx[top_valid], lon_idx[top_valid]), top_temp[top_valid] ** 2)
        np.add.at(top_n_obs, (lat_idx[top_valid], lon_idx[top_valid]), 1)
        np.add.at(
            top_n_valid_levels_sum,
            (lat_idx[top_valid], lon_idx[top_valid]),
            top_n_valid[top_valid].astype(np.float64),
        )

    if np.any(top_phase_valid):
        np.add.at(top_sum_phase, (lat_idx[top_phase_valid], lon_idx[top_phase_valid]), top_phase[top_phase_valid])
        np.add.at(top_n_phase, (lat_idx[top_phase_valid], lon_idx[top_phase_valid]), 1)

    bulk_sum_iwc = np.zeros(shape3d, dtype=np.float64)
    bulk_sq_iwc = np.zeros(shape3d, dtype=np.float64)
    bulk_sum_height = np.zeros(shape3d, dtype=np.float64)
    bulk_sq_height = np.zeros(shape3d, dtype=np.float64)
    bulk_sum_temp = np.zeros(shape3d, dtype=np.float64)
    bulk_sq_temp = np.zeros(shape3d, dtype=np.float64)
    bulk_n_obs = np.zeros(shape3d, dtype=np.int32)
    bulk_n_valid_levels_sum = np.zeros(shape3d, dtype=np.float64)

    for level in range(n_bulk):
        level_valid = np.isfinite(bulk_iwc[:, level])
        if not np.any(level_valid):
            continue

        li = lat_idx[level_valid]
        lo = lon_idx[level_valid]
        np.add.at(bulk_sum_iwc[level], (li, lo), bulk_iwc[level_valid, level])
        np.add.at(bulk_sq_iwc[level], (li, lo), bulk_iwc[level_valid, level] ** 2)
        np.add.at(bulk_sum_height[level], (li, lo), bulk_height[level_valid, level])
        np.add.at(bulk_sq_height[level], (li, lo), bulk_height[level_valid, level] ** 2)
        np.add.at(bulk_sum_temp[level], (li, lo), bulk_temp[level_valid, level])
        np.add.at(bulk_sq_temp[level], (li, lo), bulk_temp[level_valid, level] ** 2)
        np.add.at(bulk_n_obs[level], (li, lo), 1)
        np.add.at(
            bulk_n_valid_levels_sum[level],
            (li, lo),
            bulk_n_valid[level_valid, level].astype(np.float64),
        )

    top_iwc_mean, top_iwc_std = _stats_from_sum(top_sum_iwc, top_sq_iwc, top_n_obs)
    top_height_mean, top_height_std = _stats_from_sum(top_sum_height, top_sq_height, top_n_obs)
    top_temp_mean, top_temp_std = _stats_from_sum(top_sum_temp, top_sq_temp, top_n_obs)
    top_n_valid_levels_mean = np.full(shape2d, np.nan, dtype=np.float32)
    top_has = top_n_obs > 0
    top_n_valid_levels_mean[top_has] = (
        top_n_valid_levels_sum[top_has] / top_n_obs[top_has]
    ).astype(np.float32)

    top_cph_fill = np.float32(-1.0)
    top_cph_mean = np.full(shape2d, top_cph_fill, dtype=np.float32)
    top_has_phase = top_n_phase > 0
    top_cph_mean[top_has_phase] = (top_sum_phase[top_has_phase] / top_n_phase[top_has_phase]).astype(np.float32)

    bulk_iwc_mean = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_iwc_std = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_height_mean = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_height_std = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_temp_mean = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_temp_std = np.full(shape3d, np.nan, dtype=np.float32)
    bulk_n_valid_levels_mean = np.full(shape3d, np.nan, dtype=np.float32)

    for level in range(n_bulk):
        bulk_iwc_mean[level], bulk_iwc_std[level] = _stats_from_sum(
            bulk_sum_iwc[level], bulk_sq_iwc[level], bulk_n_obs[level]
        )
        bulk_height_mean[level], bulk_height_std[level] = _stats_from_sum(
            bulk_sum_height[level], bulk_sq_height[level], bulk_n_obs[level]
        )
        bulk_temp_mean[level], bulk_temp_std[level] = _stats_from_sum(
            bulk_sum_temp[level], bulk_sq_temp[level], bulk_n_obs[level]
        )
        valid = bulk_n_obs[level] > 0
        bulk_n_valid_levels_mean[level, valid] = (
            bulk_n_valid_levels_sum[level, valid] / bulk_n_obs[level, valid]
        ).astype(np.float32)

    valid_time = ~np.isnat(rounded_mean_time)
    if not np.any(valid_time):
        raise ValueError("No valid rounded mean times found, cannot construct time dimension")

    time_values = dt_arange(rounded_mean_time[valid_time].min(), rounded_mean_time[valid_time].max(), dt_min=15)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    top_iwc_mean_3d = scatter_2d_to_3d_by_time(top_iwc_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_iwc_std_3d = scatter_2d_to_3d_by_time(top_iwc_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_height_mean_3d = scatter_2d_to_3d_by_time(top_height_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_height_std_3d = scatter_2d_to_3d_by_time(top_height_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_temp_mean_3d = scatter_2d_to_3d_by_time(top_temp_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_temp_std_3d = scatter_2d_to_3d_by_time(top_temp_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan))
    top_cph_mean_3d = scatter_2d_to_3d_by_time(top_cph_mean, rounded_mean_time, time_values, fill_value=top_cph_fill)
    top_n_obs_3d = scatter_2d_to_3d_by_time(top_n_obs, rounded_mean_time, time_values, fill_value=np.int32(0))
    top_n_valid_levels_3d = scatter_2d_to_3d_by_time(
        top_n_valid_levels_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )

    bulk_iwc_mean_4d = scatter_3d_to_4d_by_time(
        bulk_iwc_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_iwc_std_4d = scatter_3d_to_4d_by_time(
        bulk_iwc_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_height_mean_4d = scatter_3d_to_4d_by_time(
        bulk_height_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_height_std_4d = scatter_3d_to_4d_by_time(
        bulk_height_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_temp_mean_4d = scatter_3d_to_4d_by_time(
        bulk_temp_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_temp_std_4d = scatter_3d_to_4d_by_time(
        bulk_temp_std, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )
    bulk_n_obs_4d = scatter_3d_to_4d_by_time(
        bulk_n_obs.astype(np.int32), rounded_mean_time, time_values, fill_value=np.int32(0)
    )
    bulk_n_valid_levels_4d = scatter_3d_to_4d_by_time(
        bulk_n_valid_levels_mean, rounded_mean_time, time_values, fill_value=np.float32(np.nan)
    )

    epoch_units = "seconds since 1970-01-01 00:00:00"
    time_bin_sec = time_values.astype("datetime64[s]").astype(np.int64)
    rounded_mean_time_sec = rounded_mean_time.astype("datetime64[s]").astype(np.int64)
    rounded_mean_time_fill = np.int32(-1)
    rounded_mean_time_sec = rounded_mean_time_sec.astype(np.int32)
    rounded_mean_time_sec[np.isnat(rounded_mean_time)] = rounded_mean_time_fill

    out = xr.Dataset(
        data_vars={
            "cb_top_iwc_mean": (("time_bin", "lat_bin", "lon_bin"), top_iwc_mean_3d),
            "cb_top_iwc_std": (("time_bin", "lat_bin", "lon_bin"), top_iwc_std_3d),
            "cb_top_height_mean": (("time_bin", "lat_bin", "lon_bin"), top_height_mean_3d),
            "cb_top_height_std": (("time_bin", "lat_bin", "lon_bin"), top_height_std_3d),
            "cb_top_temp_mean": (("time_bin", "lat_bin", "lon_bin"), top_temp_mean_3d),
            "cb_top_temp_std": (("time_bin", "lat_bin", "lon_bin"), top_temp_std_3d),
            "cb_top_cph_mean": (("time_bin", "lat_bin", "lon_bin"), top_cph_mean_3d),
            "cb_top_n_obs": (("time_bin", "lat_bin", "lon_bin"), top_n_obs_3d),
            "cb_top_n_valid_levels": (("time_bin", "lat_bin", "lon_bin"), top_n_valid_levels_3d),
            "cb_iwc_mean": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_iwc_mean_4d),
            "cb_iwc_std": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_iwc_std_4d),
            "cb_height_mean": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_height_mean_4d),
            "cb_height_std": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_height_std_4d),
            "cb_temp_mean": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_temp_mean_4d),
            "cb_temp_std": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_temp_std_4d),
            "cb_n_obs": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_n_obs_4d),
            "cb_n_valid_levels": (("time_bin", "bulk_level", "lat_bin", "lon_bin"), bulk_n_valid_levels_4d),
            "rounded_mean_time": (("lat_bin", "lon_bin"), rounded_mean_time_sec.astype(np.int32)),
            "bulk_level_offset_top": (("bulk_level",), BULK_LEVEL_OFFSETS_M[:, 0]),
            "bulk_level_offset_bottom": (("bulk_level",), BULK_LEVEL_OFFSETS_M[:, 1]),
        },
        coords={
            "time_bin": ("time_bin", time_bin_sec),
            "bulk_level": ("bulk_level", np.arange(n_bulk, dtype=np.int16)),
            "lat_bin": ("lat_bin", lat_centers.astype(np.float32)),
            "lon_bin": ("lon_bin", lon_centers.astype(np.float32)),
            "lat_bin_edge": ("lat_bin_edge", lat_edges.astype(np.float32)),
            "lon_bin_edge": ("lon_bin_edge", lon_edges.astype(np.float32)),
        },
    )

    out["time_bin"].attrs.update(
        {
            "standard_name": "time",
            "long_name": "time bin center",
            "units": epoch_units,
            "calendar": "standard",
        }
    )
    out["bulk_level"].attrs.update(
        {
            "long_name": "Cloud-bulk layer index below cloud top",
            "comment": f"0 is the shallowest bulk layer, {n_bulk - 1} is the deepest bulk layer.",
        }
    )
    out["lat_bin"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": "latitude bin center",
            "units": "degrees_north",
            "bounds": "lat_bin_edge",
        }
    )
    out["lon_bin"].attrs.update(
        {
            "standard_name": "longitude",
            "long_name": "longitude bin center",
            "units": "degrees_east",
            "bounds": "lon_bin_edge",
        }
    )
    out["lat_bin_edge"].attrs.update({"long_name": "latitude bin edges", "units": "degrees_north"})
    out["lon_bin_edge"].attrs.update({"long_name": "longitude bin edges", "units": "degrees_east"})
    out["bulk_level_offset_top"].attrs.update(
        {
            "long_name": "Upper bound of cloud-bulk layer relative to detected cloud-top cell",
            "units": "m",
        }
    )
    out["bulk_level_offset_bottom"].attrs.update(
        {
            "long_name": "Lower bound of cloud-bulk layer relative to detected cloud-top cell",
            "units": "m",
        }
    )

    coord_ref_3d = "time_bin lat_bin lon_bin"
    coord_ref_4d = "time_bin bulk_level lat_bin lon_bin"

    for name, long_name, units in [
        ("cb_top_iwc_mean", "Mean cloud-top IWC", "kg m-3"),
        ("cb_top_iwc_std", "Standard deviation of cloud-top IWC", "kg m-3"),
        ("cb_top_height_mean", "Mean cloud-top layer height", "km"),
        ("cb_top_height_std", "Standard deviation of cloud-top layer height", "km"),
        ("cb_top_temp_mean", "Mean cloud-top layer temperature", "K"),
        ("cb_top_temp_std", "Standard deviation of cloud-top layer temperature", "K"),
    ]:
        out[name].attrs.update(
            {
                "long_name": long_name,
                "units": units,
                "_FillValue": np.float32(np.nan),
                "missing_value": np.float32(np.nan),
                "coordinates": coord_ref_3d,
            }
        )

    out["cb_top_cph_mean"].attrs.update(
        {
            "long_name": "Mean cloud-top phase score over cloudy tops",
            "_FillValue": np.float32(top_cph_fill),
            "missing_value": np.float32(top_cph_fill),
            "valid_min": np.float32(1.0),
            "valid_max": np.float32(2.0),
            "flag_values": np.array([-1.0, 1.0, 1.5, 2.0], dtype=np.float32),
            "flag_meanings": "no_cloudy_tops liquid mixed ice",
            "coordinates": coord_ref_3d,
            "comment": "Computed from cloudy samples only: liquid=1, mixed=1.5, ice=2.",
        }
    )
    out["cb_top_n_obs"].attrs.update(
        {
            "long_name": "Number of cloudy cloud-top observations",
            "units": "1",
            "_FillValue": np.int32(0),
            "missing_value": np.int32(0),
            "coordinates": coord_ref_3d,
        }
    )
    out["cb_top_n_valid_levels"].attrs.update(
        {
            "long_name": "Mean number of valid source levels used for cloud-top averaging",
            "units": "1",
            "_FillValue": np.float32(np.nan),
            "missing_value": np.float32(np.nan),
            "valid_min": np.float32(1.0),
            "valid_max": np.float32(2.0),
            "coordinates": coord_ref_3d,
        }
    )

    for name, long_name, units in [
        ("cb_iwc_mean", "Mean cloud-bulk IWC", "kg m-3"),
        ("cb_iwc_std", "Standard deviation of cloud-bulk IWC", "kg m-3"),
        ("cb_height_mean", "Mean cloud-bulk layer height", "km"),
        ("cb_height_std", "Standard deviation of cloud-bulk layer height", "km"),
        ("cb_temp_mean", "Mean cloud-bulk layer temperature", "K"),
        ("cb_temp_std", "Standard deviation of cloud-bulk layer temperature", "K"),
    ]:
        out[name].attrs.update(
            {
                "long_name": long_name,
                "units": units,
                "_FillValue": np.float32(np.nan),
                "missing_value": np.float32(np.nan),
                "coordinates": coord_ref_4d,
            }
        )

    out["cb_n_obs"].attrs.update(
        {
            "long_name": "Number of cloudy cloud-bulk observations",
            "units": "1",
            "_FillValue": np.int32(0),
            "missing_value": np.int32(0),
            "coordinates": coord_ref_4d,
        }
    )
    out["cb_n_valid_levels"].attrs.update(
        {
            "long_name": "Mean number of valid source levels used for cloud-bulk averaging",
            "units": "1",
            "_FillValue": np.float32(np.nan),
            "missing_value": np.float32(np.nan),
            "valid_min": np.float32(1.0),
            "valid_max": np.float32(2.0),
            "coordinates": coord_ref_4d,
        }
    )

    out["rounded_mean_time"].attrs.update(
        {
            "long_name": "Rounded mean observation time at each lat/lon bin",
            "units": epoch_units,
            "calendar": "standard",
            "_FillValue": rounded_mean_time_fill,
        }
    )
    out["rounded_mean_time"].encoding.update({"dtype": "int32"})

    out.attrs.update(
        {
            "title": "Binned cloud-bulk IWC, height, temperature, and cloud-top phase statistics",
            "summary": "DARDAR-CLOUD cloud-top and cloud-bulk statistics aggregated to regular lat/lon bins and mapped to 15-minute time bins.",
            "Conventions": "CF-1.10",
            "featureType": "grid",
            "source": "Binned from DARDAR-CLOUD iwc/temperature and DARMASK_Simplified_Categorization",
            "history": "Created by dardar_cloud_bulk_binning.py",
            "lat_spacing_degrees": float(dlat),
            "lon_spacing_degrees": float(dlon),
        }
    )
    return out


def _epoch_seconds_to_datetime(time_value: int) -> datetime:
    """Convert epoch seconds to a naive UTC datetime for file naming."""
    epoch = np.datetime64("1970-01-01T00:00:00", "s")
    dt64 = epoch + np.timedelta64(int(time_value), "s")
    return dt64.astype("datetime64[m]").astype(object)


def _write_time_split_outputs(out_binned: xr.Dataset, output_dir: Path, prefix: str, file_ind: int) -> None:
    """Write one NetCDF file per time bin with an indexed temporary suffix."""
    groups = list(out_binned.groupby("time_bin", squeeze=False))
    datasets = [group for _, group in groups]
    paths = [
        os_path.join(output_dir, f"{prefix}_{_epoch_seconds_to_datetime(t):%Y%m%d_%H%M}_{file_ind}.nc")
        for t, _ in groups
    ]
    xr.save_mfdataset(datasets, paths, mode="w")


def _process_file(
    in_path: Path,
    output_dir: Path,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    file_ind: int = 1,
) -> None:
    with xr.open_dataset(in_path) as ds_in:
        # print(f"Processing {in_path}")
        profiles = extract_cloud_bulk_profiles(ds_in)
        out_binned = bin_cloud_bulk_latlon(
            profiles,
            lat_min=lat_min,
            lat_max=lat_max,
            dlat=dlat,
            lon_min=lon_min,
            lon_max=lon_max,
            dlon=dlon,
        )
        _write_time_split_outputs(out_binned, output_dir=output_dir, prefix="DD_CB", file_ind=file_ind)


def _process_files_in_parallel(
    input_files: list[Path],
    output_dir: Path,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    n_workers: int,
) -> None:
    n_workers = min(cpu_count(), len(input_files), n_workers)
    if n_workers < 1:
        print("No files to process.")
        return
    if n_workers < 2:
        for file_ind, in_file in enumerate(input_files):
            _process_file(in_file, output_dir, lat_min, lat_max, dlat, lon_min, lon_max, dlon, file_ind=file_ind)
        return

    tasks = [
        (in_file, output_dir, lat_min, lat_max, dlat, lon_min, lon_max, dlon, file_ind)
        for file_ind, in_file in enumerate(input_files)
    ]
    print(f"Processing {len(tasks)} files with {n_workers} parallel workers...")
    with Pool(processes=n_workers) as pool:
        pool.starmap(_process_file, tasks)


def merge_ddcb_suffix_files_in_dir(day_dir: Path, workers: int = 8) -> None:
    """Merge or rename `DD_CB_YYYYMMDD_HHMM_<n>.nc` files within one directory."""
    pattern = re.compile(r"^(DD_CB_\d{8}_\d{4})_(\d+)\.nc$")
    groups: dict[str, list[Path]] = {}
    for file_path in day_dir.glob("DD_CB_*_*.nc"):
        match = pattern.match(file_path.name)
        if match:
            groups.setdefault(match.group(1), []).append(file_path)

    if not groups:
        print(f"[merge-ddcb] no DD_CB_*_<n>.nc files in {day_dir}")
        return

    def process_key(key: str) -> None:
        files = sorted(groups[key])
        out_path = day_dir / f"{key}.nc"
        tmp_path = day_dir / f"{key}.tmp.nc"
        try:
            if len(files) == 1:
                files[0].replace(out_path)
                print(f"[rename ] {files[0].name} -> {out_path.name}")
                return

            tmp_path.unlink(missing_ok=True)
            result = subprocess.run(
                ["cdo", "-L", "ensmean", *map(str, files), str(tmp_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "cdo ensmean failed")

            tmp_path.replace(out_path)
            for file_path in files:
                file_path.unlink(missing_ok=True)
            print(f"[merge] {len(files)} files -> {out_path.name}")
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            print(f"[ERROR] {key}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(groups)))) as pool:
        list(pool.map(process_key, sorted(groups)))
    print(f"[merge-ddcb] done: {day_dir}")


def move_wrong_day_files(day_dir: Path, base_path: Path, workers: int = 8) -> None:
    """Move DD_CB files into the correct YYYY/MM/DD directory using the filename date."""
    rx = re.compile(r"^(DD_CB_(\d{8})_\d{4})(?:_(\d+))?\.nc$")
    groups: dict[str, list[Path]] = {}

    for file_path in day_dir.glob("DD_CB_*.nc"):
        match = rx.match(file_path.name)
        if match:
            groups.setdefault(match.group(1), []).append(file_path)

    if not groups:
        print(f"[move-wrong-day] no DD_CB_*.nc in {day_dir}")
        return

    def dst_for(key: str, yyyymmdd: str) -> Path:
        day = datetime.strptime(yyyymmdd, "%Y%m%d")
        return base_path / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / f"{key}.nc"

    def cdo_ensmean(inputs: list[Path], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.nc")
        tmp_path.unlink(missing_ok=True)
        result = subprocess.run(
            ["cdo", "-L", "ensmean", *map(str, inputs), str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(result.stderr.strip() or "cdo ensmean failed")
        tmp_path.replace(out_path)

    def process(key: str) -> None:
        srcs = sorted(groups[key])
        match = rx.match(srcs[0].name)
        assert match is not None
        dst = dst_for(key, match.group(2))

        if any(src.resolve() == dst.resolve() for src in srcs):
            return

        try:
            if dst.exists():
                cdo_ensmean([dst] + srcs, dst)
                for src in srcs:
                    src.unlink(missing_ok=True)
                print(f"[merge-from-move] {len(srcs)+1} -> {dst.name} (collision)")
                return

            dst.parent.mkdir(parents=True, exist_ok=True)
            preferred = next((src for src in srcs if src.name == f"{key}.nc"), srcs[0])
            preferred.replace(dst)
            for src in srcs:
                if src != preferred:
                    src.unlink(missing_ok=True)
            print(f"[move  ] {preferred.name} -> {dst}")
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(groups)))) as pool:
        list(pool.map(process, sorted(groups)))
    print(f"[move-wrong-day] done: {day_dir}")


def process_one_day(
    input_dir: Path,
    output_dir: Path,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    n_workers: int,
) -> None:
    """Process all DARDAR-CLOUD files found for one input day."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(input_dir.glob("DARDAR-CLOUD_*.nc"))
    if not input_files:
        print(f"[skip] no DARDAR-CLOUD files in {input_dir}")
        return

    _process_files_in_parallel(
        input_files,
        output_dir=output_dir,
        lat_min=lat_min,
        lat_max=lat_max,
        dlat=dlat,
        lon_min=lon_min,
        lon_max=lon_max,
        dlon=dlon,
        n_workers=n_workers,
    )
    merge_ddcb_suffix_files_in_dir(output_dir, workers=8)
    move_wrong_day_files(output_dir, base_path=output_dir.parent.parent.parent, workers=8)


def process_multiple_days(
    base_input_dir: Path,
    base_output_dir: Path,
    start_date: str,
    end_date: str,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    n_workers: int,
) -> None:
    """Process a contiguous range of day directories."""
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    current = start_dt
    while current <= end_dt:
        input_dir = base_input_dir / current.strftime("%Y/%Y_%m_%d")
        output_dir = base_output_dir / current.strftime("%Y/%m/%d")
        process_one_day(
            input_dir=input_dir,
            output_dir=output_dir,
            lat_min=lat_min,
            lat_max=lat_max,
            dlat=dlat,
            lon_min=lon_min,
            lon_max=lon_max,
            dlon=dlon,
            n_workers=n_workers,
        )
        current += timedelta(days=1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bin DARDAR-CLOUD cloud-top and cloud-bulk IWC, height, and temperature "
            "statistics onto a regular lon-lat grid and write DD_CB_YYYYMMDD_HHMM.nc outputs."
        )
    )
    parser.add_argument("--single-file", type=Path, default=None, help="Path to one DARDAR-CLOUD file to process.")
    parser.add_argument("--input-root", type=Path, default=None, help="Base DARDAR-CLOUD input root for multi-day mode.")
    parser.add_argument("--output-root", type=Path, required=True, help="Base output directory.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date in YYYYMMDD format for multi-day mode.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date in YYYYMMDD format for multi-day mode.")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum latitude edge.")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum latitude edge.")
    parser.add_argument("--dlat", type=float, default=0.3, help="Latitude grid spacing in degrees.")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum longitude edge.")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum longitude edge.")
    parser.add_argument("--dlon", type=float, default=0.3, help="Longitude grid spacing in degrees.")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of worker processes for per-file parallelism.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.single_file is not None:
        output_dir = Path(args.output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        _process_file(
            in_path=args.single_file,
            output_dir=output_dir,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            dlat=args.dlat,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            dlon=args.dlon,
            file_ind=0,
        )
        merge_ddcb_suffix_files_in_dir(output_dir, workers=1)
        move_wrong_day_files(output_dir, base_path=output_dir, workers=1)
        return

    if args.input_root is None or args.start_date is None or args.end_date is None:
        parser.error("Multi-day mode requires --input-root, --start-date, and --end-date when --single-file is not used.")

    process_multiple_days(
        base_input_dir=Path(args.input_root),
        base_output_dir=Path(args.output_root),
        start_date=args.start_date,
        end_date=args.end_date,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        dlat=args.dlat,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        dlon=args.dlon,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    # main()
    _process_file(
            in_path="/wolke_scratch/dnikolo/DARDAR_data/DARDAR-CLOUD/2007/2007_10_12/DARDAR-CLOUD_2007285011553_07743_V3-00.nc",
            output_dir="/wolke_scratch/dnikolo/dump/dardar-cloud_cb",
            lat_min=-81,
            lat_max=81,
            dlat=0.3,
            lon_min=-81,
            lon_max=81,
            dlon=0.3,
            file_ind=0,
        )
