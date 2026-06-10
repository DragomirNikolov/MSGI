import xarray as xr
import pandas as pd
import numpy as np
import os
from glaciation_time_estimator.auxiliary_func.config_reader import read_config
from glaciation_time_estimator.auxiliary_func.chunking_data import ChunkLoader

ALLOWED_CTH_DEV=1000
config = read_config()
ANALYSIS_YEAR=config["start_time"].year
OUTPUT_FOLDER=config["postprocessing_output_dir"]
# CB_data_path = f"/nfs/ch4/wolke_scratch/dnikolo/DARDAR_data/CB_data/{ANALYSIS_YEAR}"
CB_data_path = "/nfs/ch4/wolke_scratch/dnikolo/dump/dardar-cloud_cb"


def _check_val_seviri_coincidence(row):
    points_of_interest = row["val_cth_hist"] > 0
    if type(row["cth_hist"]) == float:
            # N+=1
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

# cloud_data = pd.read_parquet("/nfs/ch4/wolke_scratch/dnikolo/GTE_Validation/DARDAR/Test_bulk/Final_results/2007_glac_04.parquet")
# cloud_data = pd.read_parquet(os.path.join(OUTPUT_FOLDER,"Final_results", f"{ANALYSIS_YEAR}_glac_04.parquet"))
loader = ChunkLoader([ANALYSIS_YEAR],config=config)
cloud_data = loader.glac_chunk
cloud_data["val_seviri_coincidence"] = cloud_data.apply(_get_val_seviri_coincidence,axis=1)
valid_clouds = cloud_data[cloud_data["val_seviri_coincidence"]==True]
valid_clouds["colocation_time"]=valid_clouds.apply(_identify_times_to_process, axis=1)

row_temp = valid_clouds.iloc[1]
# for col_time in row["colocation_time"]:
col_time = row_temp["colocation_time"][0]
dt = pd.Timestamp(col_time)
time_str = dt.strftime("%Y%m%d_%H%M")

path = os.path.join(CB_data_path,f"DD_CB_{time_str}.nc")
print(path)
ds = xr.open_dataset(path)

lat_bins = ds.lat_bin.values
lon_bins = ds.lon_bin.values

lat0 = lat_bins[0]
lon0 = lon_bins[0]

dlat = lat_bins[1] - lat_bins[0]
dlon = lon_bins[1] - lon_bins[0]

nlat = len(lat_bins)
nlon = len(lon_bins)


def get_colocation_ind(row):
    lat_idx_list = []
    lon_idx_list = []
    for lat, lon in zip(row.val_intersec_lat,row.val_intersec_lon):
        lat_idx = np.rint((lat - lat0) / dlat).astype(np.int64)
        lon_idx = np.rint((lon - lon0) / dlon).astype(np.int64)

        valid = (
            (lat_idx >= 0) & (lat_idx < nlat) &
            (lon_idx >= 0) & (lon_idx < nlon)
        )
        lat_idx_list.append(lat_idx[valid])
        lon_idx_list.append(lon_idx[valid])
    return lat_idx_list,lon_idx_list
# col_time = row["colocation_time"][0]
def get_bulk_values(row):
    cb_iwc_means  = []
    ct_iwc_means = []
    for i in range(len(row["colocation_time"])):
        col_time = row["colocation_time"][i]
        dt = pd.Timestamp(col_time)
        time_str = dt.strftime("%Y%m%d_%H%M")

        # path = f"/nfs/ch4/wolke_scratch/dnikolo/dump/dardar-cloud_cb/"
        path = os.path.join(CB_data_path,f"DD_CB_{time_str}.nc")
        print(path)
        ds = xr.open_dataset(path)
        cb_iwc_values = ds["cb_iwc_mean"].isel(
            lat_bin=xr.DataArray(row["lat_bin_idx"][i], dims="pixel"),
            lon_bin=xr.DataArray(row["lon_bin_idx"][i], dims="pixel"),
            )
        ct_iwc_values = ds["cb_top_iwc_mean"].isel(
            lat_bin=xr.DataArray(row["lat_bin_idx"][i], dims="pixel"),
            lon_bin=xr.DataArray(row["lon_bin_idx"][i], dims="pixel"),
        )
        cb_iwc_means.append(cb_iwc_values.mean(axis=2)[0])
        ct_iwc_means.append(ct_iwc_values.mean(axis=1)[0])
    return cb_iwc_means, ct_iwc_means


def extract_iwc_relative_to_peak(row):
    records = []

    track_start = pd.Timestamp(row["track_start_time"])
    peak_ind = int(row["Glac_peak_ind"])

    for i, iwc_profile in enumerate(row["iwc_bulk_means"]):
        col_time = pd.Timestamp(row["colocation_time"][i])
        iwc_ct = row["iwc_top_means"]

        # Index of the colocation time within the cloud track
        col_ind = int(round((col_time - track_start).total_seconds() / (15 * 60)))

        # Time index relative to the ice-fraction peak
        rel_ind = col_ind - peak_ind

        records.append({
            "Cloud_ID": row["Cloud_ID"],
            "rel_ind": rel_ind,
            "rel_time_min": rel_ind * 15,
            "iwc_profile": np.asarray(iwc_profile),
            "iwc_ct": iwc_ct
        })

    return records


    
valid_clouds[["lat_bin_idx", "lon_bin_idx"]] = (
    valid_clouds
    .apply(get_colocation_ind, axis=1, result_type="expand")
)

valid_clouds[["iwc_bulk_means","iwc_top_means"]] = (valid_clouds.apply(get_bulk_values, axis=1, result_type="expand"))


all_records = []

for _, row in valid_clouds.iterrows():
    all_records.extend(extract_iwc_relative_to_peak(row))

iwc_rel_df = pd.DataFrame(all_records)
iwc_stats_by_rel_time = (
    iwc_rel_df
    .groupby("rel_ind")["iwc_profile"]
    .agg(
        n_hits="count",
        mean_iwc_profile=lambda profiles: np.nanmean(np.stack(profiles), axis=0),
    )
    .reset_index()
)
iwc_stats_by_rel_time["rel_time_min"] = iwc_stats_by_rel_time["rel_ind"] * 15

iwc_rel_df.to_parquet(f"iwc_rel_df_V3_1_{ANALYSIS_YEAR}.parquet")
iwc_stats_by_rel_time.to_parquet(f"mean_iwc_by_height_data_V3_1_{ANALYSIS_YEAR}.parquet")
