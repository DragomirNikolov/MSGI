from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from msgi.obs.dardar.dardar_cloud_bulk_binning import merge_ddcb_suffix_files_in_dir, _process_file
import re
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from netCDF4 import Dataset
from tqdm import tqdm
import os
from glaciation_time_estimator.auxiliary_func.config_reader import read_config

config = read_config()
ANALYSIS_YEAR=config["start_time"].year
OUTPUT_FOLDER=config["postprocessing_output_dir"]

TRACKING_PARQUET = Path(os.path.join(OUTPUT_FOLDER,"Final_results", f"{ANALYSIS_YEAR}_glac_04.parquet"))
DARDAR_CLOUD_ROOT = Path("/wolke_scratch/dnikolo/DARDAR_data/DARDAR-CLOUD/V3_1")
OUTPUT_DIR = Path(os.path.join("/wolke_scratch/dnikolo/DARDAR_data/CB_data",f"{ANALYSIS_YEAR}"))
ALLOWED_CTH_DEV_M = 1000.0
LAT_MIN = -81.0
LAT_MAX = 81.0
DLON = 0.3
DLAT = 0.3
LON_MIN = -81.0
LON_MAX = 81.0
N_WORKERS = 6

ALLOWED_CTH_DEV=1000


def _check_val_seviri_coincidence(row):
    points_of_interest = row["val_cth_hist"] > 0
    if type(row["cth_hist"]) == float:
            N+=1
            return np.nan 
    cth_deviations = np.abs(row["cth_hist"][points_of_interest] - row["val_cth_hist"][points_of_interest])
    if cth_deviations.size == 0:
        return False
    mean_height_dev = cth_deviations.mean()  
    if mean_height_dev > ALLOWED_CTH_DEV:
        return False
    return True

def _get_val_seviri_coincidence(row):
    if "val_seviri_coincidence" not in row.keys():
        return _check_val_seviri_coincidence(row)
    else:
        return row["val_seviri_coincidence"]

def _identify_times_to_process(row):
    val_if_hist = row.val_ice_frac_hist
    idx = np.where((val_if_hist >= -0.01) & (val_if_hist <= 1.01))[0]
    return row.track_start_time+idx*pd.Timedelta(minutes=15)

def load_relevant_tracking_times(
    tracking_parquet: Path = TRACKING_PARQUET,
    allowed_cth_dev_m: float = ALLOWED_CTH_DEV_M,
) -> pd.DatetimeIndex:
    """Reproduce the tracked-cloud relevant-time selection from development/3d_dardar_tracking.py."""
    cloud_data = pd.read_parquet(tracking_parquet)
    cloud_data["val_seviri_coincidence"] = cloud_data.apply(_get_val_seviri_coincidence,axis=1)
    valid_clouds = cloud_data[cloud_data["val_seviri_coincidence"]==True]
    return pd.DatetimeIndex(
        np.concatenate(valid_clouds.apply(_identify_times_to_process, axis=1).values)
    ).unique().sort_values() 


def _decode_time_units_to_base_datetime64(units: str) -> np.datetime64:
    """Decode a DARDAR-like `seconds since YYYY MM DD ...` units string."""
    match = re.search(r"since\s+(\d{4})\s+(\d{2})\s+(\d{2})", units)
    if not match:
        raise ValueError(f"Could not decode time units: {units!r}")

    year, month, day = map(int, match.groups())
    return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T00:00:00", "ns")


def _read_raw_time_seconds_and_units(in_path: Path) -> tuple[np.ndarray, str]:
    """Read raw time values without netCDF valid-range masking."""
    with Dataset(in_path) as ds:
        time_var = ds.variables["time"]
        time_var.set_auto_mask(False)
        values = np.asarray(time_var[:], dtype=np.float64)
        units = getattr(time_var, "units", "")
    return values, units

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

def read_dardar_file_times_ns(in_path: Path) -> np.ndarray:
    """Return finite DARDAR file times as datetime64[ns] without valid-range masking side effects."""
    values, units = _read_raw_time_seconds_and_units(in_path)
    base = _decode_time_units_to_base_datetime64(units)
    valid = np.isfinite(values)
    out = np.full(values.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    out[valid] = base + np.rint(values[valid] * 1e9).astype("timedelta64[ns]")
    return out


def find_relevant_dardar_files(
    relevant_times: pd.DatetimeIndex,
    dardar_root: Path = DARDAR_CLOUD_ROOT,
) -> list[tuple[Path, np.ndarray]]:
    """Find DARDAR files whose rounded scan times intersect the requested tracking times."""
    relevant_time_bins_ns = relevant_times.to_numpy(dtype="datetime64[ns]")
    relevant_days = sorted({ts.strftime("%Y_%m_%d") for ts in relevant_times})
    selected: list[tuple[Path, np.ndarray]] = []

    for day_key in tqdm(relevant_days, desc="Scanning DARDAR days"):
        day = pd.Timestamp(day_key.replace("_", "-"))
        day_dir = dardar_root / f"{day:%Y}" / day_key
        if not day_dir.exists():
            print(f"[skip] no DARDAR day directory for {day_key}: {day_dir}")
            continue

        # for in_path in sorted(day_dir.glob("DARDAR-CLOUD_*.nc")):
        files = sorted(day_dir.glob("DARDAR-CLOUD_*.nc"))
        for in_path in tqdm(files, desc=f"Checking files {day_key}", leave=False):
            file_times_ns = read_dardar_file_times_ns(in_path)
            rounded_file_times_ns = round_ns_timestamps(file_times_ns, dt_min=15)
            matched_time_bins_ns = np.sort(np.unique(rounded_file_times_ns[np.isin(rounded_file_times_ns, relevant_time_bins_ns)]))
            if matched_time_bins_ns.size == 0:
                continue
            selected.append((in_path, matched_time_bins_ns))

    return selected

def _process_file_star(task):
    return _process_file(*task)

def process_tracking_selected_dardar_bulk(
    tracking_parquet: Path = TRACKING_PARQUET,
    dardar_root: Path = DARDAR_CLOUD_ROOT,
    output_dir: Path = OUTPUT_DIR,
    n_workers: int = N_WORKERS,
) -> None:
    """Run cloud-bulk analysis only for DARDAR files matching tracked relevant times."""
    relevant_times = load_relevant_tracking_times(tracking_parquet)
    print(f"Relevant times: {relevant_times}")
    if relevant_times.empty:
        print(f"[skip] no relevant tracking times found in {tracking_parquet}")
        return

    selected_files = find_relevant_dardar_files(relevant_times, dardar_root=dardar_root)
    if not selected_files:
        print("[skip] no DARDAR-CLOUD files matched the relevant tracking times")
        return
    print(f"Selected files: {selected_files}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir = output_dir / "_tracking_selected_tmp"
    shutil.rmtree(temp_output_dir, ignore_errors=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] processing {len(selected_files)} DARDAR files for {len(relevant_times)} relevant 15-minute bins "
        f"from {relevant_times[0]} to {relevant_times[-1]}"
    )

    tasks = [
        (
            in_file,
            temp_output_dir,
            LAT_MIN,
            LAT_MAX,
            DLAT,
            LON_MIN,
            LON_MAX,
            DLON,
            file_ind,
        )
        for file_ind, (in_file, matched_time_bins_ns) in enumerate(selected_files)
    ]

    worker_count = min(cpu_count(), len(tasks), n_workers)
    if worker_count < 2:
        for task in tqdm(tasks, desc="Processing selected files"):
            _process_file(*task)
    else:
        print(f"[info] processing selected files with {worker_count} workers")
        with Pool(processes=worker_count) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_file_star, tasks),
                total=len(tasks),
                desc="Processing selected files",
            ):
                pass
    merge_ddcb_suffix_files_in_dir(temp_output_dir, workers=min(8, max(1, len(tasks))))
    for merged_file in sorted(temp_output_dir.glob("DD_CB_*.nc")):
        merged_file.replace(output_dir / merged_file.name)
    shutil.rmtree(temp_output_dir, ignore_errors=True)

    

if __name__=="__main__":
    process_tracking_selected_dardar_bulk()
