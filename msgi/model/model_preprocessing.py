import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import dask
import re
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
    Convert time coordinate to datetime64[ns].

    Handles:
    1. ICON-style float time: YYYYMMDD + fractional day
    2. CESM/cftime time: cftime.DatetimeNoLeap objects already decoded by xarray
    3. Already-valid numpy/pandas datetime values
    """
    time_vals = da["time"].values

    # Case 1: already datetime64
    if np.issubdtype(time_vals.dtype, np.datetime64):
        dt = time_vals.astype("datetime64[ns]")

    # Case 2: numeric ICON-style YYYYMMDD.fraction
    elif np.issubdtype(time_vals.dtype, np.number):
        t = time_vals.astype(float)
        ymd = np.floor(t).astype(int)
        frac = t - ymd

        base = pd.to_datetime(ymd.astype(str), format="%Y%m%d")
        dt = (
            (base + pd.to_timedelta(frac, unit="D")).to_numpy(dtype="datetime64[ns]")
            + np.timedelta64(30, "s")
        ).astype("datetime64[m]").astype("datetime64[ns]")

    # Case 3: cftime objects, e.g. CESM no-leap calendar
    else:
        dt = np.array(
            [
                np.datetime64(
                    f"{t.year:04d}-{t.month:02d}-{t.day:02d}T"
                    f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                )
                for t in time_vals
            ],
            dtype="datetime64[ns]",
        )

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
    grid_type = msgi_config.get("grid_type","")
    poles = msgi_config.get("poles", ["np","sp"])
    experiment_name = msgi_config["experiment_name"]
    model = msgi_config.get("model", "ICON")
    var_name_dict = {"ACTP": "ctp","ACTT": "ctt"}
    if base_dir is None:
        base_dir=gte_config["CLAAS_fp"]
        print(f"preprocessing_output_dir is not included in msgi_config. Fallback to gte_config[CLAAS_fp]={base_dir}")
    if grid_type=="latlon":
        filtered_base_dir = os.path.join(base_dir, "Filtered_Data","np")
        resampled_base_dir = os.path.join(base_dir, "Resampled_Data","np")
    if grid_type=="geostationary":
        filtered_base_dir = os.path.join(base_dir, "Filtered_Data")
        resampled_base_dir = os.path.join(base_dir, "Resampled_Data")

    # ---- variables you want in the mixed-temp-range output ----
    # Replace these with the actual variable names from your dataset
    mixed_var_names = msgi_config["additional_vars"]
    twc_threshold = msgi_config["twc_threshold"]
    tag = f"Agg_{gte_config['agg_fact']:02}"

    # Mixed temperature bounds across all ranges
    T_min_mixed = min(x for t in T_ranges for x in t)
    T_max_mixed = max(x for t in T_ranges for x in t)

    start_date = gte_config["start_time"].replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = gte_config["end_time"].replace(hour=0, minute=0, second=0, microsecond=0)

    if model == "ICON":
        def extract_file_date(filepath):
            """Extract datetime from filename."""
            basename = os.path.basename(filepath)
            match = re.search(r"(\d{8}T\d{6}Z)", basename)
            if not match:
                return None
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    if model == "CESM":
        def extract_file_date(filepath):
            """Extract datetime from CESM filename."""
            basename = os.path.basename(filepath)

            match = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$", basename)
            if not match:
                return None

            year, month, day, seconds = match.groups()
            return datetime(
                int(year),
                int(month),
                int(day),
            ) + timedelta(seconds=int(seconds))

    for file in os.listdir(folder_path):
        if grid_type=="latlon":
            if file.startswith(f"remap_laf_whole_{experiment_name}"):
                date = extract_file_date(file)
                if date is None or date < start_date or date > end_date:
                    continue
                file_path = os.path.join(folder_path, file)
            else:
                continue
        elif grid_type=="geostationary":
            file_path=None
            for pole in poles:
                if file.startswith(f"remap_laf_{pole}_{experiment_name}"):
                    date = extract_file_date(file)
                    if date is None or date < start_date or date > end_date:
                        continue
                    current_pole=pole
                    file_path = os.path.join(folder_path, file)
            if file_path is None:
                continue
        else:
            raise ValueError("grid_type should be either latlon or geostationary")
        data = xr.open_dataset(file_path, chunks={"time": 1})
        if grid_type=="latlon":
            data = lat_lon_filter(data)

        # Load variables
        if model == "ICON":
            CPH = data["cph"]
            CTT = data["ctt"]
        
        if model == "CESM":
            CPH = data["ACT_ITMF"]
            # CPH = data["ACT_ITF"]
            CTT = data["ACTT"]

        
        

        # Extra variables for the mixed-temp-range files
        additional_vars = [data[v] for v in mixed_var_names]

        print(f"Processing file: {file} with shape: {CPH.shape}")



        # ---------------------------------------------------------------
        # 1) Existing per-temperature-range output (unchanged behavior)
        # ---------------------------------------------------------------
        CPH_results = []

        for T_range in T_ranges:
            T_min, T_max = T_range
            CTT_mask = (CTT >= T_min) & (CTT < T_max)
            if model == "ICON": 
                CPH_results.append(CPH.where(CTT_mask, other=0))
            if model =="CESM":
                CPH_results.append((CPH+1).where(CTT_mask, other=0))

        # ---------------------------------------------------------------
        # 2) New mixed-temp-range output (single mask, one file per time)
        # ---------------------------------------------------------------
        mixed_mask = (CTT >= T_min_mixed) & (CTT < T_max_mixed)
        if "twc" in data.data_vars:
            mixed_mask = mixed_mask & (data["twc"] >= twc_threshold)
        mixed_results = [var.where(mixed_mask, other=0) for var in additional_vars]
        # If yiu want celcius
        # for i in range(len(additional_vars)):
        #     if (mixed_var_names[i]).lower() != "ctt":
        #         mixed_results.append(additional_vars[i].where(mixed_mask, other=0))
        #     else:
        #         mixed_results.append(additional_vars[i].where(mixed_mask-273.15, other=np.nan))

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
                if grid_type=="latlon":
                    subdir = ts.strftime(f"%Y/%m/%d")
                    filename = (
                    f"R_{tag}_T_{(273 - T_ranges[i][0]):02}_{(273 - T_ranges[i][1]):02}_"
                    f"{ts.strftime('%Y%m%d%H%M%S')}.nc"
                    )
                elif grid_type=="geostationary":
                    subdir = ts.strftime(f"{current_pole}/%Y/%m/%d")
                    filename = (
                    f"{tag}_T_{(273 - T_ranges[i][0]):02}_{(273 - T_ranges[i][1]):02}_"
                    f"{ts.strftime('%Y%m%d%H%M%S')}.nc"
                    )
                full_path = os.path.join(filtered_base_dir, subdir, filename)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                paths.append(full_path)

            xr.save_mfdataset(new_data_list, paths)

        # ---------------------------------------------------------------
        # Write new mixed-temp-range files (one file per timestep)
        # ---------------------------------------------------------------
        mixed_ds = xr.Dataset()

        for var_name, var_da in zip(mixed_var_names, mixed_output):
            # optional: rename to indicate filtering
            if model == "ICON":
                out_name = f"{var_name}_resampled"
            if model == "CESM":
                out_name = f"{var_name_dict[var_name]}_resampled"

            mixed_ds[out_name] = convert_time_coord(var_da).rename(out_name)

        times, new_data_list = prepare_split_datasets(mixed_ds)

        paths = []
        for dt in times:
            ts = pd.Timestamp(dt)
            if grid_type=="latlon":
                subdir = ts.strftime(f"%Y/%m/%d")
                filename = f"R_{tag}_{ts.strftime('%Y%m%d%H%M%S')}.nc"
            elif grid_type=="geostationary":
                subdir = ts.strftime(f"{current_pole}/%Y/%m/%d")
                filename = f"{tag}_{ts.strftime('%Y%m%d%H%M%S')}.nc"
            full_path = os.path.join(resampled_base_dir, subdir, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            paths.append(full_path)

        xr.save_mfdataset(new_data_list, paths)

def resample_dir_files(gte_config,msgi_config):

    input_dir = msgi_config["resample_input_dir"]
    output_dir = msgi_config["resample_output_dir"]
    experiment_name = msgi_config.get("experiment_name", "")
    grid_type = msgi_config.get("grid_type","")
    aux_dir = msgi_config.get("cdo_aux_dir", "/wolke_scratch/dnikolo/ICON_Data/aux_tools")
    poles = msgi_config.get("poles", ["np","sp"])
    model = msgi_config.get("model", "ICON")
    if grid_type=="latlon":
        grid_file = msgi_config.get("cdo_grid_file", os.path.join(aux_dir, "grid_1.5x1.5.txt"))
    elif grid_type=="geostationary":
        grid_files = msgi_config.get("cdo_grid_file", {"np":os.path.join(aux_dir, "grid_CLAAS_Agg_3_np.txt"),"sp":os.path.join(aux_dir, "grid_CLAAS_Agg_3_sp.txt") })
    else:
        raise ValueError("grid_type not supported or not specified in msgi auxiliary file")
    
    remap_methods = msgi_config.get("resampling_methods", ["laf", "dis"])
    remap_methods = [m for m in remap_methods if m in {"laf", "dis"}]
    if not remap_methods:
        remap_methods = ["laf", "dis"]

    cdo_threads = int(msgi_config.get("cdo_threads", 8))
    workers = int(msgi_config.get("resample_workers", max(1, (os.cpu_count() or 2) // 2)))
    if model == "ICON":
        input_files = sorted(glob.glob(os.path.join(input_dir, "*_ctp_2d*")))
    if model == "CESM":
        input_files = sorted(glob.glob(os.path.join(input_dir, "*cam.h1*")))
    if experiment_name:
        input_files = [
            f for f in input_files
            if os.path.basename(f).startswith(experiment_name) and f.endswith(".nc")
        ]
    if model == "ICON":
        def extract_file_date(filepath):
            """Extract datetime from filename."""
            basename = os.path.basename(filepath)
            match = re.search(r"(\d{8}T\d{6}Z)", basename)
            if not match:
                return None
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    if model == "CESM":
        def extract_file_date(filepath):
            """Extract datetime from CESM filename."""
            basename = os.path.basename(filepath)

            match = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$", basename)
            if not match:
                return None

            year, month, day, seconds = match.groups()
            return datetime(
                int(year),
                int(month),
                int(day),
            ) + timedelta(seconds=int(seconds))
    start_date = gte_config["start_time"].replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = gte_config["end_time"].replace(hour=0, minute=0, second=0, microsecond=0)

    if start_date or end_date:
        filtered_files = []
        for f in input_files:
            file_dt = extract_file_date(f)
            if file_dt is None:
                continue

            if start_date and file_dt < start_date:
                continue
            if end_date and file_dt > end_date:
                continue

            filtered_files.append(f)

        input_files = filtered_files

    if not input_files:
        print(
            f"No CTP 2D files found in {input_dir} "
            f"for experiment prefix '{experiment_name}' "
            f"within date range {start_date} to {end_date}."
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for in_file in input_files:
        base = os.path.basename(in_file)
        if grid_type=="latlon":
            setgrid_file=None
            for method in remap_methods:
                weights_file = msgi_config.get(
                    f"{method}_weights_file",
                    os.path.join(aux_dir, f"{method}_2d_1.5x1.5_weights.nc"),
                )
                out_file = os.path.join(output_dir, f"remap_{method}_whole_{base}")
                tasks.append((in_file, out_file, setgrid_file, weights_file,grid_file, method))
        elif grid_type=="geostationary":
            setgrid_files = {"np":"/nfs/ch4/wolke_scratch/dnikolo/Glaciation_time_estimator/grids/grid_np_agg.txt",
                             "sp":"/nfs/ch4/wolke_scratch/dnikolo/Glaciation_time_estimator/grids/grid_sp_agg.txt"}
            for method in remap_methods:
                weights_files = msgi_config.get(
                        f"{method}_weights_file",
                        {"np": os.path.join(aux_dir, f"CLAAS_{method}_weights_np.nc"),
                        "sp": os.path.join(aux_dir, f"CLAAS_{method}_weights_sp.nc")})
                for pole in poles:
                    weights_file = weights_files[pole]
                    grid_file=grid_files[pole]
                    out_file = os.path.join(output_dir, f"remap_{method}_{pole}_{base}")
                    setgrid_file= setgrid_files[pole]
                    tasks.append((in_file, out_file,setgrid_file, weights_file,grid_file, method))
    if grid_type=="latlon":
        def _run_one(task):
            in_file, out_file, setgrid_file, weights_file,grid_file, method = task
            cmd = [
                "cdo",
                "-P",
                str(cdo_threads),f"-remap,{grid_file},{weights_file}",
                # "-setctomiss,0",
                in_file,
                out_file,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return method, out_file
    elif grid_type=="geostationary":
        if model == "ICON":
            def _run_one(task):
                in_file, out_file, setgrid_file, weights_file,grid_file, method = task
                cmd = [
                    "cdo",
                    "-P",
                    str(cdo_threads),
                    f"-setgrid,{setgrid_file}",f"-remap,{grid_file},{weights_file}",
                    # "-setctomiss,0",
                    in_file,
                    out_file,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                return method, out_file
        if model == "CESM":
            def _run_one(task):
                in_file, out_file, setgrid_file, weights_file,grid_file, method = task
                cmd = [
                    "cdo",
                    "-P",
                    str(cdo_threads),
                    f"-setgrid,{setgrid_file}",f"-remap,{grid_file},{weights_file}",
                    "-selvar,ACT_ITMF,ACTT,ACTP",
                    # "-selvar,ACT_ITF,ACTT,ACTP",
                    # "-setctomiss,0",
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

