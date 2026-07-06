from pathlib import Path
import re
import numpy as np
import pandas as pd
import scipy.io
import os
from glaciation_time_estimator.auxiliary_func.config_reader import read_config
from glaciation_time_estimator.data_postprocessing.Tracking_result_analysis import save_single_temp_range_results
from glaciation_time_estimator.data_postprocessing.Single_cloud_analysis import Cloud
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
    print(n_trajs)
    print(n_clouds)
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


def load_and_analyse_traj_range(start_date, end_date, base_dir=BASE_TRAJ_DIR):
    """
    Load all trajectory files between start_date and end_date inclusive.

    Parameters
    ----------
    start_date, end_date : str or pandas-compatible datetime
        Example: "2015-07-03", "2015-07-05"

    base_dir : Path
        Base directory containing year/month folders.

    Returns
    -------
    all_clouds : np.ndarray
        Array of valid Cloud objects from all trajectory files.
    """

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    all_clouds = []

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

        print(f"\nDate {date.date()}: found {len(traj_files)} files")

        for traj_file in traj_files:
            print(f"Loading {traj_file}")

            try:
                traj_data = scipy.io.loadmat(traj_file)
                cloud_arr = analyse_traj_file(traj_data)

                if len(cloud_arr) > 0:
                    all_clouds.append(cloud_arr)

            except Exception as e:
                print(f"ERROR while processing {traj_file}: {e}")
                continue

    if len(all_clouds) == 0:
        return np.array([], dtype=object)

    return np.concatenate(all_clouds)

if __name__=="__main__":
    config=read_config()
    all_clouds = load_and_analyse_traj_range(config["start_time"],config["end_time"])
    output_dir= os.path.join(
            config['postprocessing_output_dir'], "Leibzig_data",
            config['time_folder_name']
        )
    save_single_temp_range_results(all_clouds,output_dir,config)