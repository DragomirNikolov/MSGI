from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import xarray as xr


EPOCH_UNITS = "seconds since 1970-01-01 00:00:00"
PHASE_INVALID = -1
PHASE_NO_CLOUD = 0
PHASE_WATER = 1
PHASE_ICE = 2
PHASE_MIXED = 3


"""
"python /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py \
    --input /wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua/ \
    --output-root /wolke_scratch/dnikolo/MODIS/CPH_data \
        --lon-min -90 --lon-max 90 --dlat 0.3 --dlon 0.3 --lat-min -81 --lat-max 81"
"""

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MODIS CLDPROP L2 granule(s) and bin cloud-top phase/height/temperature "
            "to a regular lon-lat grid, writing MOD_CT_YYYYMMDD_HHMM.nc."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CLDPROP_L2_MODIS*.nc file or base directory (e.g., .../CLDPROP_L2_MODIS_Aqua/)",
    )
    parser.add_argument("--output-root", required=True, help="Root output directory")
    parser.add_argument("--lat-min", type=float, required=True, help="Latitude lower edge")
    parser.add_argument("--lat-max", type=float, required=True, help="Latitude upper edge")
    parser.add_argument("--dlat", type=float, required=True, help="Latitude step (deg)")
    parser.add_argument("--lon-min", type=float, required=True, help="Longitude lower edge")
    parser.add_argument("--lon-max", type=float, required=True, help="Longitude upper edge")
    parser.add_argument("--dlon", type=float, required=True, help="Longitude step (deg)")
    parser.add_argument(
        "--skip-non-quarter-hour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files where StartTime minute is not 00/15/30/45 (default: true)",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Period mode only: start datetime in YYYYMMDD_HHMM (inclusive)",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Period mode only: end datetime in YYYYMMDD_HHMM (inclusive)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing MOD_CT outputs (default: true)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Period mode only: number of multiprocessing workers (default: 8)",
    )
    return parser.parse_args()


def _build_edges(vmin: float, vmax: float, dv: float, name: str) -> np.ndarray:
    if dv <= 0:
        raise ValueError(f"{name}: step must be > 0")
    if vmax <= vmin:
        raise ValueError(f"{name}: max must be > min")

    n = (vmax - vmin) / dv
    n_round = int(round(n))
    if not np.isclose(n, n_round, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"{name}: (max-min)/d is not an integer within tolerance; got {(vmax-vmin)/dv}"
        )
    if n_round < 1:
        raise ValueError(f"{name}: must contain at least one bin")

    return vmin + np.arange(n_round + 1, dtype=np.float64) * dv


def _bin_index(values: np.ndarray, vmin: float, dv: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.floor((values - vmin) / dv).astype(np.int64)
    upper_edge = vmin + n * dv
    idx[np.isclose(values, upper_edge, rtol=0.0, atol=1e-10)] = n - 1
    in_bounds = (idx >= 0) & (idx < n)
    return idx, in_bounds


def _parse_start_time(ds_root: xr.Dataset) -> datetime:
    start_str = ds_root.attrs.get("StartTime")
    if not start_str:
        raise ValueError("Missing global attribute StartTime")

    try:
        return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"Unsupported StartTime format: {start_str}") from exc


def _quarter_hour(dt: datetime) -> bool:
    return dt.minute in (0, 15, 30, 45)


def parse_time_yyyymmdd_hhmm(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M")
    except ValueError as exc:
        raise ValueError(f"Invalid time '{value}', expected YYYYMMDD_HHMM") from exc


def discover_modis_files(base_input_dir: Path) -> list[Path]:
    if not base_input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_input_dir}")
    if not base_input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {base_input_dir}")
    return sorted(base_input_dir.rglob("*.nc"))


def extract_datetime_from_filename(path: Path) -> datetime | None:
    m = re.search(r"\.A(\d{7})\.(\d{4})\.", path.name)
    if m is None:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y%j %H%M")
    except ValueError:
        return None


def _output_path_for_datetime(output_root: Path, dt: datetime) -> Path:
    day_dir = output_root / f"{dt:%Y}" / f"{dt:%m}" / f"{dt:%d}"
    return day_dir / f"MOD_CT_{dt:%Y%m%d_%H%M}.nc"


def _period_worker(task: tuple) -> tuple[Path, Path | None, str | None]:
    in_file = task[0]
    try:
        out_file = process_one_file(*task)
        return in_file, out_file, None
    except Exception as exc:  # keep period processing robust to per-file failures
        return in_file, None, str(exc)


def _read_modis_fields(input_path: Path) -> tuple[datetime, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(input_path) as ds_root:
        start_dt = _parse_start_time(ds_root)

    with xr.open_dataset(input_path, group="geolocation_data") as ds_geo:
        lat = np.asarray(ds_geo["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds_geo["longitude"].values, dtype=np.float64)

    with xr.open_dataset(input_path, group="geophysical_data") as ds_gph:
        phase_raw = np.asarray(ds_gph["Cloud_Phase_Cloud_Top_Properties"].values)
        cth_raw_m = np.asarray(ds_gph["Cloud_Top_Height"].values, dtype=np.float64)
        ctt_raw_k = np.asarray(ds_gph["Cloud_Top_Temperature"].values, dtype=np.float64)

    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude shapes differ")
    if phase_raw.shape != lat.shape or cth_raw_m.shape != lat.shape or ctt_raw_k.shape != lat.shape:
        raise ValueError("Geophysical/geolocation variable shapes do not match")

    phase = np.full(lat.shape, PHASE_INVALID, dtype=np.int8)
    phase_f = np.asarray(phase_raw, dtype=np.float64)
    finite_phase = np.isfinite(phase_f)
    phase_i = np.full(lat.shape, PHASE_INVALID, dtype=np.int16)
    phase_i[finite_phase] = np.rint(phase_f[finite_phase]).astype(np.int16)

    # 0 no cloud, 1 water, 2 ice, 3 mixed are valid classes.
    # 6 undetermined is treated as invalid and excluded from all stats.
    valid_codes = np.isin(phase_i, [0, 1, 2, 3])
    phase[valid_codes] = phase_i[valid_codes].astype(np.int8)

    cth_km = cth_raw_m / 1000.0
    ctt_k = ctt_raw_k

    return start_dt, lat, lon, phase, cth_km, ctt_k


def _aggregate_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    phase: np.ndarray,
    cth_km: np.ndarray,
    ctt_k: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    time_sec: np.int64,
) -> xr.Dataset:
    lat_min = float(lat_edges[0])
    lon_min = float(lon_edges[0])
    dlat = float(lat_edges[1] - lat_edges[0])
    dlon = float(lon_edges[1] - lon_edges[0])
    n_lat = lat_edges.size - 1
    n_lon = lon_edges.size - 1

    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    phase_flat = phase.ravel()
    cth_flat = cth_km.ravel()
    ctt_flat = ctt_k.ravel()

    geo_ok = np.isfinite(lat_flat) & np.isfinite(lon_flat)
    lat_idx, lat_in = _bin_index(lat_flat, lat_min, dlat, n_lat)
    lon_idx, lon_in = _bin_index(lon_flat, lon_min, dlon, n_lon)

    in_grid = geo_ok & lat_in & lon_in
    lat_idx = lat_idx[in_grid]
    lon_idx = lon_idx[in_grid]
    phase_flat = phase_flat[in_grid]
    cth_flat = cth_flat[in_grid]
    ctt_flat = ctt_flat[in_grid]

    shape = (n_lat, n_lon)
    n_obs = np.zeros(shape, dtype=np.int32)
    n_no_cloud = np.zeros(shape, dtype=np.int32)
    n_liquid = np.zeros(shape, dtype=np.int32)
    n_ice = np.zeros(shape, dtype=np.int32)
    n_mixed = np.zeros(shape, dtype=np.int32)

    valid_phase = phase_flat >= 0
    if np.any(valid_phase):
        li = lat_idx[valid_phase]
        lj = lon_idx[valid_phase]
        pv = phase_flat[valid_phase]

        np.add.at(n_obs, (li, lj), 1)
        np.add.at(n_no_cloud, (li[pv == PHASE_NO_CLOUD], lj[pv == PHASE_NO_CLOUD]), 1)
        np.add.at(n_liquid, (li[pv == PHASE_WATER], lj[pv == PHASE_WATER]), 1)
        np.add.at(n_ice, (li[pv == PHASE_ICE], lj[pv == PHASE_ICE]), 1)
        np.add.at(n_mixed, (li[pv == PHASE_MIXED], lj[pv == PHASE_MIXED]), 1)

    n_cloudy = n_liquid + n_ice + n_mixed

    # cph_mean over cloudy-only valid phase values: water=1, mixed=1.5, ice=2
    phase_score = np.full(phase_flat.shape, np.nan, dtype=np.float64)
    phase_score[phase_flat == PHASE_WATER] = 1.0
    phase_score[phase_flat == PHASE_MIXED] = 1.5
    phase_score[phase_flat == PHASE_ICE] = 2.0

    phase_sum = np.zeros(shape, dtype=np.float64)
    cloudy_for_phase = np.isfinite(phase_score)
    if np.any(cloudy_for_phase):
        np.add.at(
            phase_sum,
            (lat_idx[cloudy_for_phase], lon_idx[cloudy_for_phase]),
            phase_score[cloudy_for_phase],
        )

    cph_fill = np.float32(-1.0)
    cth_fill = np.float32(-1.0)
    ctt_fill = np.float32(-1.0)

    cph_mean = np.full(shape, cph_fill, dtype=np.float32)
    obs_bins = n_obs > 0
    cloudy_bins = n_cloudy > 0
    cph_mean[cloudy_bins] = (phase_sum[cloudy_bins] / n_cloudy[cloudy_bins]).astype(np.float32)
    cph_mean[obs_bins & ~cloudy_bins] = np.float32(0.0)

    # Height stats over cloudy-only samples with finite Cloud_Top_Height
    cth_sum = np.zeros(shape, dtype=np.float64)
    cth_sq_sum = np.zeros(shape, dtype=np.float64)
    n_cth = np.zeros(shape, dtype=np.int32)
    cloudy_h = (phase_flat == PHASE_WATER) | (phase_flat == PHASE_MIXED) | (phase_flat == PHASE_ICE)
    valid_h = cloudy_h & np.isfinite(cth_flat)
    if np.any(valid_h):
        ih = lat_idx[valid_h]
        jh = lon_idx[valid_h]
        hv = cth_flat[valid_h]
        np.add.at(cth_sum, (ih, jh), hv)
        np.add.at(cth_sq_sum, (ih, jh), hv * hv)
        np.add.at(n_cth, (ih, jh), 1)

    cth_mean = np.full(shape, cth_fill, dtype=np.float32)
    cth_std = np.full(shape, np.nan, dtype=np.float32)
    hbins = n_cth > 0
    cth_mean[hbins] = (cth_sum[hbins] / n_cth[hbins]).astype(np.float32)
    cth_var = np.zeros(shape, dtype=np.float64)
    cth_var[hbins] = cth_sq_sum[hbins] / n_cth[hbins] - cth_mean[hbins].astype(np.float64) ** 2
    cth_var[hbins] = np.maximum(cth_var[hbins], 0.0)
    cth_std[hbins] = np.sqrt(cth_var[hbins]).astype(np.float32)

    # Temperature stats over cloudy-only samples with finite Cloud_Top_Temperature
    ctt_sum = np.zeros(shape, dtype=np.float64)
    ctt_sq_sum = np.zeros(shape, dtype=np.float64)
    n_ctt = np.zeros(shape, dtype=np.int32)
    valid_t = cloudy_h & np.isfinite(ctt_flat)
    if np.any(valid_t):
        it = lat_idx[valid_t]
        jt = lon_idx[valid_t]
        tv = ctt_flat[valid_t]
        np.add.at(ctt_sum, (it, jt), tv)
        np.add.at(ctt_sq_sum, (it, jt), tv * tv)
        np.add.at(n_ctt, (it, jt), 1)

    ctt_mean = np.full(shape, ctt_fill, dtype=np.float32)
    ctt_std = np.full(shape, np.nan, dtype=np.float32)
    tbins = n_ctt > 0
    ctt_mean[tbins] = (ctt_sum[tbins] / n_ctt[tbins]).astype(np.float32)
    ctt_var = np.zeros(shape, dtype=np.float64)
    ctt_var[tbins] = ctt_sq_sum[tbins] / n_ctt[tbins] - ctt_mean[tbins].astype(np.float64) ** 2
    ctt_var[tbins] = np.maximum(ctt_var[tbins], 0.0)
    ctt_std[tbins] = np.sqrt(ctt_var[tbins]).astype(np.float32)

    frac_no_cloud = np.full(shape, np.nan, dtype=np.float32)
    frac_liquid = np.full(shape, np.nan, dtype=np.float32)
    frac_ice = np.full(shape, np.nan, dtype=np.float32)
    frac_mixed = np.full(shape, np.nan, dtype=np.float32)
    frac_no_cloud[obs_bins] = (n_no_cloud[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_liquid[obs_bins] = (n_liquid[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_ice[obs_bins] = (n_ice[obs_bins] / n_obs[obs_bins]).astype(np.float32)
    frac_mixed[obs_bins] = (n_mixed[obs_bins] / n_obs[obs_bins]).astype(np.float32)

    lat_centers = ((lat_edges[:-1] + lat_edges[1:]) / 2.0).astype(np.float32)
    lon_centers = ((lon_edges[:-1] + lon_edges[1:]) / 2.0).astype(np.float32)

    # DD_CT-like 2D rounded_mean_time field
    rounded_mean_time = np.full(shape, np.int32(-1), dtype=np.int32)
    rounded_mean_time[obs_bins] = np.int32(time_sec)

    def _to3d(arr2d: np.ndarray) -> np.ndarray:
        return arr2d[np.newaxis, :, :]

    out = xr.Dataset(
        data_vars={
            "cph_mean": (("time_bin", "lat_bin", "lon_bin"), _to3d(cph_mean)),
            "cth_mean": (("time_bin", "lat_bin", "lon_bin"), _to3d(cth_mean)),
            "cth_std": (("time_bin", "lat_bin", "lon_bin"), _to3d(cth_std)),
            "ctt_mean": (("time_bin", "lat_bin", "lon_bin"), _to3d(ctt_mean)),
            "ctt_std": (("time_bin", "lat_bin", "lon_bin"), _to3d(ctt_std)),
            "frac_no_cloud": (("time_bin", "lat_bin", "lon_bin"), _to3d(frac_no_cloud)),
            "frac_liquid": (("time_bin", "lat_bin", "lon_bin"), _to3d(frac_liquid)),
            "frac_ice": (("time_bin", "lat_bin", "lon_bin"), _to3d(frac_ice)),
            "frac_mixed": (("time_bin", "lat_bin", "lon_bin"), _to3d(frac_mixed)),
            "n_obs": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_obs)),
            "n_no_cloud": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_no_cloud)),
            "n_liquid": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_liquid)),
            "n_ice": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_ice)),
            "n_mixed": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_mixed)),
            "n_cloudy": (("time_bin", "lat_bin", "lon_bin"), _to3d(n_cloudy)),
            "rounded_mean_time": (("lat_bin", "lon_bin"), rounded_mean_time),
        },
        coords={
            "time_bin": ("time_bin", np.array([time_sec], dtype=np.int64)),
            "lat_bin": ("lat_bin", lat_centers),
            "lon_bin": ("lon_bin", lon_centers),
            "lat_bin_edge": ("lat_bin_edge", lat_edges.astype(np.float32)),
            "lon_bin_edge": ("lon_bin_edge", lon_edges.astype(np.float32)),
        },
    )

    out["time_bin"].attrs.update(
        {
            "standard_name": "time",
            "long_name": "time bin center",
            "units": EPOCH_UNITS,
            "calendar": "standard",
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
    out["lat_bin_edge"].attrs.update(
        {
            "long_name": "latitude bin edges",
            "units": "degrees_north",
        }
    )
    out["lon_bin_edge"].attrs.update(
        {
            "long_name": "longitude bin edges",
            "units": "degrees_east",
        }
    )

    coord_ref = "time_bin lat_bin lon_bin"

    out["cph_mean"].attrs.update(
        {
            "long_name": "Mean cloud-top phase score over cloudy tops",
            "_FillValue": np.float32(cph_fill),
            "missing_value": np.float32(cph_fill),
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(2.0),
            "flag_values": np.array([-1.0, 0.0, 1.0, 1.5, 2.0], dtype=np.float32),
            "flag_meanings": "no_measurement no_cloudy_tops liquid mixed ice",
            "coordinates": coord_ref,
            "comment": "Computed from cloudy samples only:  liquid=1, mixed=1.5, ice=2; -1 indicates no measurements in bin.",
        }
    )

    out["cth_mean"].attrs.update(
        {
            "long_name": "Mean cloud-top height over cloudy tops",
            "units": "km",
            "_FillValue": np.float32(cth_fill),
            "missing_value": np.float32(cth_fill),
            "valid_min": np.float32(0.0),
            "coordinates": coord_ref,
            "comment": "-1 indicates no cloudy tops in bin.",
        }
    )
    out["cth_std"].attrs.update(
        {
            "long_name": "Standard deviation of cloud-top height over cloudy tops",
            "units": "km",
            "_FillValue": np.float32(np.nan),
            "missing_value": np.float32(np.nan),
            "valid_min": np.float32(0.0),
            "coordinates": coord_ref,
            "comment": "NaN indicates undefined (e.g., no cloudy tops).",
        }
    )

    out["ctt_mean"].attrs.update(
        {
            "long_name": "Mean cloud-top temperature over cloudy tops",
            "units": "K",
            "_FillValue": np.float32(ctt_fill),
            "missing_value": np.float32(ctt_fill),
            "valid_min": np.float32(0.0),
            "coordinates": coord_ref,
            "comment": "-1 indicates no cloudy tops in bin.",
        }
    )
    out["ctt_std"].attrs.update(
        {
            "long_name": "Standard deviation of cloud-top temperature over cloudy tops",
            "units": "K",
            "_FillValue": np.float32(np.nan),
            "missing_value": np.float32(np.nan),
            "valid_min": np.float32(0.0),
            "coordinates": coord_ref,
            "comment": "NaN indicates undefined (e.g., no cloudy tops).",
        }
    )

    for var_name, long_name in [
        ("frac_no_cloud", "Fraction of observations classified as no cloud"),
        ("frac_liquid", "Fraction of observations classified as liquid"),
        ("frac_ice", "Fraction of observations classified as ice"),
        ("frac_mixed", "Fraction of observations classified as mixed"),
    ]:
        out[var_name].attrs.update(
            {
                "long_name": long_name,
                "units": "1",
                "_FillValue": np.float32(np.nan),
                "missing_value": np.float32(np.nan),
                "valid_min": np.float32(0.0),
                "valid_max": np.float32(1.0),
                "coordinates": coord_ref,
                "comment": "NaN indicates no observations in bin.",
            }
        )

    for var_name, long_name in [
        ("n_obs", "Number of valid classified observations"),
        ("n_no_cloud", "Number of no-cloud observations"),
        ("n_liquid", "Number of liquid observations"),
        ("n_ice", "Number of ice observations"),
        ("n_mixed", "Number of mixed-phase observations"),
        ("n_cloudy", "Number of cloudy observations"),
    ]:
        out[var_name].attrs.update(
            {
                "long_name": long_name,
                "units": "1",
                "_FillValue": np.int32(0),
                "missing_value": np.int32(0),
                "valid_min": np.int32(0),
                "coordinates": coord_ref,
                "comment": "Count variable; 0 means none in bin/time.",
            }
        )

    out["rounded_mean_time"].attrs.update(
        {
            "long_name": "Rounded mean cloudy-top observation time at each lat/lon bin",
            "units": EPOCH_UNITS,
            "calendar": "standard",
            "_FillValue": np.int32(-1),
        }
    )

    out.attrs.update(
        {
            "title": "Binned cloud-top phase, height, and temperature statistics",
            "summary": "MODIS cloud-top phase/height/temperature statistics aggregated to regular lat/lon bins.",
            "Conventions": "CF-1.10",
            "featureType": "grid",
            "source": "Binned from MODIS CLDPROP_L2 Cloud_Phase_Cloud_Top_Properties, Cloud_Top_Height, and Cloud_Top_Temperature",
            "history": "Created by modis_cloud_top_binning.py",
        }
    )

    return out


def process_one_file(
    input_path: Path,
    output_root: Path,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    skip_non_quarter_hour: bool = True,
    overwrite: bool = True,
) -> Path | None:
    lat_edges = _build_edges(lat_min, lat_max, dlat, "latitude")
    lon_edges = _build_edges(lon_min, lon_max, dlon, "longitude")

    start_dt, lat, lon, phase, cth_km, ctt_k = _read_modis_fields(input_path)
    out_path = _output_path_for_datetime(output_root, start_dt)

    if skip_non_quarter_hour and not _quarter_hour(start_dt):
        # print(
        #     f"Skipping {input_path.name}: StartTime={start_dt:%Y-%m-%d %H:%M:%S} is not at 00/15/30/45"
        # )
        return None
    if out_path.exists():
        if overwrite:
            print(f"Overwriting existing output: {out_path}")
            out_path.unlink()
        else:
            print(f"Skipping existing output (overwrite disabled): {out_path}")
            return None

    time_sec = np.int64(np.datetime64(start_dt, "s").astype(np.int64))
    out_ds = _aggregate_to_grid(
        lat=lat,
        lon=lon,
        phase=phase,
        cth_km=cth_km,
        ctt_k=ctt_k,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
        time_sec=time_sec,
    )

    day_dir = out_path.parent
    day_dir.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path)
    return out_path


def process_time_period(
    base_input_dir: Path,
    output_root: Path,
    start_time: str,
    end_time: str,
    lat_min: float,
    lat_max: float,
    dlat: float,
    lon_min: float,
    lon_max: float,
    dlon: float,
    skip_non_quarter_hour: bool = True,
    overwrite: bool = True,
    n_workers: int = 8,
) -> dict:
    start_dt = parse_time_yyyymmdd_hhmm(start_time)
    end_dt = parse_time_yyyymmdd_hhmm(end_time)
    if end_dt < start_dt:
        raise ValueError("end_time must be >= start_time")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    summary = {
        "seen": 0,
        "in_range": 0,
        "processed": 0,
        "skipped_non_quarter_hour": 0,
        "overwritten": 0,
        "failed": 0,
    }

    files = discover_modis_files(base_input_dir)
    dated_files: list[tuple[datetime, Path]] = []
    for f in files:
        dt = extract_datetime_from_filename(f)
        if dt is None:
            continue
        dated_files.append((dt, f))
    dated_files.sort(key=lambda x: x[0])
    summary["seen"] = len(dated_files)

    jobs: list[Path] = []
    for file_dt, in_path in dated_files:
        if file_dt < start_dt or file_dt > end_dt:
            continue
        summary["in_range"] += 1

        if skip_non_quarter_hour and not _quarter_hour(file_dt):
            summary["skipped_non_quarter_hour"] += 1
            print(f"Skipping {in_path.name}: filename time {file_dt:%Y-%m-%d %H:%M} is not quarter-hour")
            continue

        expected_out = _output_path_for_datetime(output_root, file_dt)
        if expected_out.exists() and overwrite:
            summary["overwritten"] += 1
            print(f"Overwriting existing output: {expected_out}")
            expected_out.unlink()
        elif expected_out.exists() and not overwrite:
            print(f"Skipping existing output (overwrite disabled): {expected_out}")
            continue
        jobs.append(in_path)

    if not jobs:
        return summary

    worker_count = min(n_workers, len(jobs), cpu_count())

    task_args = [
        (
            in_path,
            output_root,
            lat_min,
            lat_max,
            dlat,
            lon_min,
            lon_max,
            dlon,
            skip_non_quarter_hour,
            False,  # overwrite already handled deterministically in parent
        )
        for in_path in jobs
    ]

    if worker_count == 1:
        results = [_period_worker(t) for t in task_args]
    else:
        with Pool(processes=worker_count) as pool:
            results = pool.map(_period_worker, task_args)

    for in_file, out_file, err in results:
        if err is not None:
            summary["failed"] += 1
            print(f"Failed {in_file}: {err}")
            continue
        if out_file is None:
            # Defensive fallback; quarter-hour filtering should have been done in parent.
            summary["skipped_non_quarter_hour"] += 1
            continue
        summary["processed"] += 1
        # print(f"Processed: {in_file.name} -> {out_file}")

    return summary


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)

    if input_path.is_dir():
        if args.start_time is None or args.end_time is None:
            raise ValueError("Directory input requires --start-time and --end-time in YYYYMMDD_HHMM")
        summary = process_time_period(
            base_input_dir=input_path,
            output_root=Path(args.output_root),
            start_time=args.start_time,
            end_time=args.end_time,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            dlat=args.dlat,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            dlon=args.dlon,
            skip_non_quarter_hour=args.skip_non_quarter_hour,
            overwrite=args.overwrite,
            n_workers=args.n_workers,
        )
        print("Summary:", summary)
        return

    output = process_one_file(
        input_path=input_path,
        output_root=Path(args.output_root),
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        dlat=args.dlat,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        dlon=args.dlon,
        skip_non_quarter_hour=args.skip_non_quarter_hour,
        overwrite=args.overwrite,
    )

    if output is None:
        print("No output written.")
    else:
        print(f"Wrote: {output}")


if __name__ == "__main__":
    process_time_period(
        base_input_dir = Path("/wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua"),
        output_root = Path("/wolke_scratch/dnikolo/MODIS/CPH_data"),
        start_time = "20070601_0000",
        end_time = "20071231_2345",
        lat_min = -81,
        lat_max=81,
        dlat = 0.3,
        lon_min = -81,
        lon_max = 81,
        dlon = 0.3)
    # main()
