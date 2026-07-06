from pathlib import Path
import re
import numpy as np
import pandas as pd
import scipy.io
import os
from glaciation_time_estimator.auxiliary_func.config_reader import read_config
from glaciation_time_estimator.data_postprocessing.Tracking_result_analysis import save_single_temp_range_results
from glaciation_time_estimator.data_postprocessing.Single_cloud_analysis import Cloud
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
# You need this import from your project
# from glaciation_time_estimator.data_postprocessing.Single_cloud_analysis import Cloud


BASE_TRAJ_DIR = Path(
    "/wolke_scratch/dnikolo/Satellite/Leipzig_tracking/"
    "one_month/claas-3_full-disk_trajectories"
)


def find_date_folder(day_dir: Path, day: int) -> Path | None:
    """
    Given a month directory like .../2015/07 and a day number,
    find the subfolder containing that day, e.g. 03-05.
    """
    for folder in day_dir.iterdir():
        if not folder.is_dir():
            continue

        match = re.fullmatch(r"(\d{2})-(\d{2})", folder.name)
        if match is None:
            continue

        start_day = int(match.group(1))
        end_day = int(match.group(2))

        if start_day <= day <= end_day:
            return folder

    return None


def analyse_traj_file(traj_data):
    n_trajs = traj_data["trajectories"].shape[1]
    n_clouds = 0
    for traj_nr in range(n_trajs):
            n_clouds+=traj_data["trajectories"][0,traj_nr]["clouds"][0,0].shape[0]
    cloud_arr = np.empty((n_clouds),dtype=Cloud)
    idx=0
    for traj_nr in range(n_trajs):
        traj_time = traj_data["trajectories"][0,traj_nr]["times"][0,0][0]
        flat_time = np.concatenate(traj_time)
        times = pd.to_datetime(flat_time, format="%Y%m%d%H%M")
        traj_cloud_count = traj_data["trajectories"][0,traj_nr]["clouds"][0,0].shape[0]
        for cloud_nr in range(traj_cloud_count):
            cloud_arr[idx] = Cloud(idx, False,1)
            data = traj_data["trajectories"][0,traj_nr]["clouds"][0,0][cloud_nr,0]["data"]
            cloud_values = data["cph"][0]
            cot = data["cot"][0]
            ctp = data["ctp"][0]
            ctt = data["ctt"][0]
            pix_area = data["pixel_area"][0]
            for i in range(pix_area.shape[0]):
                    # print(i)
                    lat = np.full(ctt[i].shape,fill_value=30)
                    lon = np.full(ctt[i].shape,fill_value=30)
                    if pix_area[i].sum()<1000:
                            cloud_arr[idx]=None
                            break
                    mean_ctt = np.average(ctt[i], weights=pix_area[i].T)
                    if (mean_ctt>=273.15) or (mean_ctt<=273.15-36):
                            cloud_arr[idx]=None
                            break
                    cloud_arr[idx].update_status( times[i], cloud_values[i],cot[i],ctp[i],ctt[i],cloud_lat=lat,cloud_lon=lon,pixel_area_non_agg=pix_area[i].T,pixel_area_agg=pix_area[i].T)
            idx+=1
    for i, cloud in enumerate(cloud_arr):
        if cloud == None:
                continue
        # Check if the cloud is in the mixed phase
        temp_check = ((np.array(cloud.mean_ctt_list)>273.15-38) & (np.array(cloud.mean_ctt_list)<273.15)).all()
        # Large enough for comparison
        size_check = (np.array(cloud.cloud_size_km_list)>1000).all()
        # Lived long enough
        time_check = len(cloud.cloud_size_km_list)>=4
        if not (temp_check and size_check and time_check):
                    cloud_arr[i]=None
    cloud_arr = cloud_arr[cloud_arr!=None]
    return cloud_arr


def analyse_single_file_worker(traj_file):
    """
    Worker function.

    Each process loads and analyses exactly one trajectory file.
    This function must be top-level so that multiprocessing can pickle it.
    """
    traj_file = Path(traj_file)

    try:
        print(f"Loading {traj_file}", flush=True)

        traj_data = scipy.io.loadmat(traj_file)
        cloud_arr = analyse_traj_file(traj_data)

        return {
            "file": str(traj_file),
            "success": True,
            "clouds": cloud_arr,
            "n_clouds": len(cloud_arr),
            "error": None,
            "traceback": None,
        }

    except Exception as e:
        return {
            "file": str(traj_file),
            "success": False,
            "clouds": np.array([], dtype=object),
            "n_clouds": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def collect_traj_files(start_date, end_date, base_dir=BASE_TRAJ_DIR):
    """
    Collect all trajectory files between start_date and end_date inclusive.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    all_traj_files = []

    for date in pd.date_range(start_date, end_date, freq="D"):
        print(date)

        year = f"{date.year:04d}"
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"

        month_dir = Path(base_dir) / year / month
        date_folder = find_date_folder(month_dir, date.day)

        if date_folder is None:
            print(f"WARNING: no folder found for {date.date()} in {month_dir}")
            continue

        pattern = f"trajectories_day{day}_part*.mat"
        traj_files = sorted(date_folder.glob(pattern))

        if len(traj_files) == 0:
            print(f"WARNING: no trajectory files found for {date.date()} in {date_folder}")
            continue

        print(f"Date {date.date()}: found {len(traj_files)} files")

        all_traj_files.extend(traj_files)

    return all_traj_files


def load_and_analyse_traj_range(
    start_date,
    end_date,
    base_dir=BASE_TRAJ_DIR,
    n_workers=None,
):
    """
    Load all trajectory files between start_date and end_date inclusive.

    Files are processed in parallel.
    Each file is loaded and analysed inside a single worker process.
    """

    traj_files = collect_traj_files(start_date, end_date, base_dir=base_dir)

    if len(traj_files) == 0:
        print("No trajectory files found.")
        return np.array([], dtype=object)

    if n_workers is None:
        # Conservative default for HPC / shared filesystem
        n_workers = min(os.cpu_count() or 1, len(traj_files), 8)

    print(f"\nProcessing {len(traj_files)} files with {n_workers} workers\n")

    all_clouds = []
    failed_files = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyse_single_file_worker, traj_file): traj_file
            for traj_file in traj_files
        }

        for future in as_completed(futures):
            result = future.result()

            if result["success"]:
                print(
                    f"DONE {result['file']} | valid clouds: {result['n_clouds']}",
                    flush=True,
                )

                if result["n_clouds"] > 0:
                    all_clouds.append(result["clouds"])

            else:
                print(
                    f"ERROR while processing {result['file']}: {result['error']}",
                    flush=True,
                )
                failed_files.append(result)

    print(f"\nFinished processing.")
    print(f"Successful files with clouds: {len(all_clouds)}")
    print(f"Failed files: {len(failed_files)}")

    if len(failed_files) > 0:
        print("\nFailed file list:")
        for fail in failed_files:
            print(f"  {fail['file']}")
            print(f"    {fail['error']}")

    if len(all_clouds) == 0:
        return np.array([], dtype=object)

    return np.concatenate(all_clouds)


if __name__ == "__main__":
    config = read_config()

    n_workers = config.get("n_workers", None)

    all_clouds = load_and_analyse_traj_range(
        config["start_time"],
        config["end_time"],
        n_workers=n_workers,
    )

    output_dir = os.path.join(
        config["postprocessing_output_dir"],
        "Leibzig_data",
        config["time_folder_name"],
    )

    save_single_temp_range_results(all_clouds, output_dir, config)