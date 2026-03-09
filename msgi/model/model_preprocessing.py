import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
from glaciation_time_estimator.auxiliary_func.config_reader import read_config as read_gte_config
from msgi.aux import read_config as read_msgi_config


#--------------------------------------------------------------
#Helper functions 
#--------------------------------------------------------------

def time_filter(data):
    winter_month = list([1,2,11,12])
    time_months = data['time'].dt.month
    time_mask = time_months.isin(winter_month)
    data = data.where(time_mask, drop=True)
    return data

def lat_lon_filter(data):
    lat_mask = ((data['lat'] >= 30) & (data['lat'] <= 80)) | ((data['lat'] >= -80) & (data['lat'] <= -30))
    lon_mask = (data['lon'] >= -80) | (data['lon'] <= 80)
    data = data.where(lat_mask & lon_mask, drop=True)
    return data

def float_time_to_datetime(t):
    """
    Convert 'day as %Y%m%d.%f' float (e.g. 20080905.010417)
    to a Python datetime.
    """
    t = float(t)
    date_int = int(t)
    frac = t - date_int  # fraction of day

    year  = date_int // 10000
    month = (date_int % 10000) // 100
    day   = date_int % 100

    # fraction of day -> seconds (round to nearest second)
    seconds = int(round(frac * 24 * 3600))

    return datetime(year, month, day) + timedelta(seconds=seconds)

def convert_time_coord(da):
    """
    Convert float time coordinate (YYYYMMDD + fractional day)
    to datetime64[m], matching your existing logic.
    """
    t = da["time"].values.astype(float)
    ymd = np.floor(t).astype(int)
    frac = t - ymd
    base = pd.to_datetime(ymd.astype(str), format="%Y%m%d")
    dt = (
        (base + pd.to_timedelta(frac, unit="D")).to_numpy(dtype="datetime64[ns]")
        + np.timedelta64(30, "s")
    ).astype("datetime64[m]").astype("datetime64[ns]")
    return da.assign_coords(time=("time", dt))

def prepare_split_datasets(ds):
    """
    Split dataset into one dataset per timestep and assign coord attrs/encoding.
    """
    times, dataset_list = zip(*ds.groupby("time", squeeze=False))
    new_data_list = []

    for data, dt in zip(dataset_list, times):
        data = data.assign_coords(time=("time", [dt]))

        data["time"] = data["time"].assign_attrs({
            "standard_name": "time",
            "axis": "T"
        })
        data.time.encoding = {
            "units": "seconds since 2008-08-30 00:00:00",
            "calendar": "proleptic_gregorian",
        }

        data["lat"] = data["lat"].assign_attrs({
            "_FillValue": np.nan,
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y"
        })

        data["lon"] = data["lon"].assign_attrs({
            "_FillValue": np.nan,
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X"
        })

        new_data_list.append(data)

    return list(times), new_data_list

def split_dir_files(gte_config, msgi_config):
    # ---------------------------------------------------------------
    # Temperature range loop
    # ---------------------------------------------------------------
    T_ranges = [(i, i + 6) for i in range(273 - 6, 273 - 38, -6)]
    

    folder_path = msgi_config["resample_output_dir"]
    base_dir = msgi_config.get("preprocessing_output_dir",None)
    if base_dir is None:
        base_dir=gte_config["CLAAS_fp"]
        print(f"preprocessing_output_dir is not included in msgi_config. Fallback to gte_config[CLAAS_fp]={base_dir}")
    filtered_base_dir = os.path.join(base_dir, "Filtered_Data","np")
    resampled_base_dir = os.path.join(base_dir, "Resampled_Data","np")

    # ---- variables you want in the mixed-range output ----
    # Replace these with the actual variable names from your dataset
    mixed_var_names = msgi_config["mixed_vars"]
    twc_threshold = msgi_config["twc_threshold"]

    # Mixed temperature bounds across all ranges
    T_min_mixed = min(x for t in T_ranges for x in t)
    T_max_mixed = max(x for t in T_ranges for x in t)

    for file in os.listdir(folder_path):
        if file.startswith("remap_laf_whole_ryan_nudged_test"):
            file_path = os.path.join(folder_path, file)
        else:
            continue

        data = xr.open_dataset(file_path, chunks={"time": 1})
        data = lat_lon_filter(data)

        # Load variables
        CPH = data["cph"]
        CTT = data["ctt"]

        # Extra variables for the mixed-range files
        mixed_vars = [data[v] for v in mixed_var_names]

        print(f"Processing file: {file} with shape: {CPH.shape}")

        # ---------------------------------------------------------------
        # 1) Existing per-temperature-range output (unchanged behavior)
        # ---------------------------------------------------------------
        CPH_results = []

        for T_range in T_ranges:
            T_min, T_max = T_range
            CTT_mask = (CTT >= T_min) & (CTT < T_max)

            CPH_results.append(CPH.where(CTT_mask, other=0))

        # ---------------------------------------------------------------
        # 2) New mixed-range output (single mask, one file per time)
        # ---------------------------------------------------------------
        mixed_mask = (CTT >= T_min_mixed) & (CTT < T_max_mixed)
        if "twc" in data.data_vars:
            mixed_mask = mixed_mask & (data["twc"] >= twc_threshold)
        mixed_results = [var.where(mixed_mask, other=0) for var in mixed_vars]
        # If yiu want celcius
        # for i in range(len(mixed_vars)):
        #     if (mixed_var_names[i]).lower() != "ctt":
        #         mixed_results.append(mixed_vars[i].where(mixed_mask, other=0))
        #     else:
        #         mixed_results.append(mixed_vars[i].where(mixed_mask-273.15, other=np.nan))

        with ProgressBar(dt=15.0):
             CPH_output, mixed_output = dask.compute(
                CPH_results,
                mixed_results
            )

        # ---------------------------------------------------------------
        # Write existing filtered CPH files (same as before)
        # ---------------------------------------------------------------
        for i, cph_temp_range in enumerate(CPH_output):
            cph_temp_range = cph_temp_range.assign_attrs({
                "name": "cph",
                "long_name": "Cloud Phase",
                "units": "unitless"
            }).rename("cph_filtered")

            cph_temp_range = convert_time_coord(cph_temp_range)

            cph_ds = cph_temp_range.to_dataset()
            times, new_data_list = prepare_split_datasets(cph_ds)

            paths = []
            for dt in times:
                ts = pd.Timestamp(dt)
                subdir = ts.strftime("%Y/%m/%d")
                filename = (
                    f"R_Agg_30_T_{(273 - T_ranges[i][0]):02}_{(273 - T_ranges[i][1]):02}_"
                    f"{ts.strftime('%Y%m%d%H%M%S')}.nc"
                )
                full_path = os.path.join(filtered_base_dir, subdir, filename)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                paths.append(full_path)

            xr.save_mfdataset(new_data_list, paths)

        # ---------------------------------------------------------------
        # Write new mixed-range files (one file per timestep)
        # ---------------------------------------------------------------
        mixed_ds = xr.Dataset()

        for var_name, var_da in zip(mixed_var_names, mixed_output):
            # optional: rename to indicate filtering
            out_name = f"{var_name}_filtered"

            mixed_ds[out_name] = convert_time_coord(var_da).rename(out_name)

        times, new_data_list = prepare_split_datasets(mixed_ds)

        paths = []
        for dt in times:
            ts = pd.Timestamp(dt)
            subdir = ts.strftime("%Y/%m/%d")
            filename = f"R_Agg_30_{ts.strftime('%Y%m%d%H%M%S')}.nc"
            full_path = os.path.join(resampled_base_dir, subdir, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            paths.append(full_path)

        xr.save_mfdataset(new_data_list, paths)

def resample_dir_files(gte_config,msgi_config):

    input_dir = msgi_config["resample_input_dir"]
    output_dir = msgi_config["resample_output_dir"]
    experiment_name = msgi_config.get("experiment_name", "")

    aux_dir = msgi_config.get("cdo_aux_dir", "/wolke_scratch/dnikolo/ICON_Data/aux_tools")
    grid_file = msgi_config.get("cdo_grid_file", os.path.join(aux_dir, "grid_1.5x1.5.txt"))

    remap_methods = msgi_config.get("resampling_methods", ["laf", "dis"])
    remap_methods = [m for m in remap_methods if m in {"laf", "dis"}]
    if not remap_methods:
        remap_methods = ["laf", "dis"]

    cdo_threads = int(msgi_config.get("cdo_threads", 8))
    workers = int(msgi_config.get("resample_workers", max(1, (os.cpu_count() or 2) // 2)))

    input_files = sorted(glob.glob(os.path.join(input_dir, "*_ctp_2d*")))
    if experiment_name:
        input_files = [f for f in input_files if os.path.basename(f).startswith(experiment_name)]

    if not input_files:
        print(f"No CTP 2D files found in {input_dir} for experiment prefix '{experiment_name}'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for in_file in input_files:
        base = os.path.basename(in_file)
        for method in remap_methods:
            weights_file = msgi_config.get(
                f"{method}_weights_file",
                os.path.join(aux_dir, f"{method}_2d_1.5x1.5_weights.nc"),
            )
            out_file = os.path.join(output_dir, f"remap_{method}_whole_{base}")
            tasks.append((in_file, out_file, weights_file, method))

    def _run_one(task):
        in_file, out_file, weights_file, method = task
        cmd = [
            "cdo",
            "-P",
            str(cdo_threads),
            f"remap,{grid_file},{weights_file}",
            "-setctomiss,0",
            in_file,
            out_file,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return method, out_file

    max_workers = max(1, min(workers, len(tasks)))
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for fut in as_completed(futures):
            try:
                _, out_file = fut.result()
                print(f"Finished {out_file}")
            except subprocess.CalledProcessError as exc:
                failures.append(exc.stderr.strip() if exc.stderr else str(exc))

    if failures:
        raise RuntimeError("CDO remapping failed:\n" + "\n".join(failures))

def preprocess_dir(gte_config,msgi_config):
    resample_dir_files(gte_config,msgi_config)
    split_dir_files(gte_config,msgi_config)

if __name__=="__main__":

    gte_config = read_gte_config()
    # gte_config = None
    msgi_config = read_msgi_config()
    preprocess_dir(gte_config, msgi_config)
