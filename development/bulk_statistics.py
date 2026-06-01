import xarray as xr
import pandas as pd
import numpy as np


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

cloud_data = pd.read_parquet("/nfs/ch4/wolke_scratch/dnikolo/GTE_Validation/DARDAR/Test_bulk/Final_results/2007_glac_04.parquet")
cloud_data["val_seviri_coincidence"] = cloud_data.apply(_get_val_seviri_coincidence,axis=1)
valid_clouds = cloud_data[cloud_data["val_seviri_coincidence"]==True]
valid_clouds["colocation_time"]=valid_clouds.apply(_identify_times_to_process, axis=1)

row_temp = valid_clouds.iloc[1]
# for col_time in row["colocation_time"]:
col_time = row_temp["colocation_time"][0]
dt = pd.Timestamp(col_time)
time_str = dt.strftime("%Y%m%d_%H%M")

path = f"/nfs/ch4/wolke_scratch/dnikolo/dump/dardar-cloud_cb/DD_CB_{time_str}.nc"
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
    iwc_means  = []
    for i in range(len(row["colocation_time"])):
        col_time = row["colocation_time"][i]
        dt = pd.Timestamp(col_time)
        time_str = dt.strftime("%Y%m%d_%H%M")

        path = f"/nfs/ch4/wolke_scratch/dnikolo/dump/dardar-cloud_cb/DD_CB_{time_str}.nc"
        print(path)
        ds = xr.open_dataset(path)
        cb_iwc_values = ds["cb_iwc_mean"].isel(
            lat_bin=xr.DataArray(row["lat_bin_idx"][i], dims="pixel"),
            lon_bin=xr.DataArray(row["lon_bin_idx"][i], dims="pixel"),
            )
        iwc_means.append(cb_iwc_values.mean(axis=2)[0])
    return iwc_means


def extract_iwc_relative_to_peak(row):
    records = []

    track_start = pd.Timestamp(row["track_start_time"])
    peak_ind = int(row["Glac_peak_ind"])

    for i, iwc_profile in enumerate(row["iwc_bulk_means"]):
        col_time = pd.Timestamp(row["colocation_time"][i])

        # Index of the colocation time within the cloud track
        col_ind = int(round((col_time - track_start).total_seconds() / (15 * 60)))

        # Time index relative to the ice-fraction peak
        rel_ind = col_ind - peak_ind

        records.append({
            "Cloud_ID": row["Cloud_ID"],
            "rel_ind": rel_ind,
            "rel_time_min": rel_ind * 15,
            "iwc_profile": np.asarray(iwc_profile)
        })

    return records


    
valid_clouds[["lat_bin_idx", "lon_bin_idx"]] = (
    valid_clouds
    .apply(get_colocation_ind, axis=1, result_type="expand")
)

valid_clouds["iwc_bulk_means"] = valid_clouds.apply(get_bulk_values, axis=1)



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

iwc_rel_df.to_parquet("iwc_rel_df.parquet")
iwc_stats_by_rel_time.to_parquet("mean_iwc_by_height_data.parquet")
