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
CB_data_path = f"/wolke_scratch/dnikolo/DARDAR_data/CB_02/{ANALYSIS_YEAR}"
# CB_data_path = "/wolke_scratch/dnikolo/dump/dardar-cloud_cb"
CONVERT_LINEAR_Z_TO_DBZE = False

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
valid_clouds = cloud_data.loc[cloud_data["val_seviri_coincidence"] == True].copy()
valid_clouds["colocation_time"] = valid_clouds.apply(_identify_times_to_process, axis=1)

# These are the columns/information belonging to the original valid cloud.
# Save this before adding bulk-derived columns like iwc_bulk_means, lat_bin_idx, etc.
ORIGINAL_VALID_CLOUD_COLUMNS = valid_clouds.columns.tolist()

i=0
row_temp = valid_clouds.iloc[0]
while row_temp["colocation_time"].size==0:
    row_temp=valid_clouds.iloc[ind]
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




def _to_dBZe(linear_z):
    positive_z = linear_z.where(linear_z > 0.0)
    return 10.0 * np.log10(positive_z)


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
    cb_iwc_means = []
    ct_iwc_means = []

    cb_height_means = []
    ct_height_means = []
    ct_temp_means = []

    cb_temp_means = []
    cb_Z_means = []
    cb_ext_means = []

    bulk_offset_top_m = None
    bulk_offset_bottom_m = None
    bulk_offset_mid_m = None

    for i in range(len(row["colocation_time"])):
        col_time = row["colocation_time"][i]
        dt = pd.Timestamp(col_time)
        time_str = dt.strftime("%Y%m%d_%H%M")

        path = os.path.join(CB_data_path, f"DD_CB_{time_str}.nc")
        print(path)
        if not os.path.exists(path):
            print(f"Error: skipping {path}")
            if len(cb_iwc_means)>=1:
                missing_profile = np.full(cb_iwc_means[-1].shape, np.nan)
            else:
                missing_profile = np.full((18,), np.nan)
            cb_iwc_means.append(missing_profile.copy())
            cb_height_means.append(missing_profile.copy())
            cb_ext_means.append(missing_profile.copy())
            cb_temp_means.append(missing_profile.copy())
            cb_Z_means.append(missing_profile.copy())

            ct_iwc_means.append(np.nan)
            ct_height_means.append(np.nan)
            ct_temp_means.append(np.nan)
            if bulk_offset_top_m is None:
                bulk_offset_top_m = missing_profile.copy()
                bulk_offset_bottom_m = missing_profile.copy()
                bulk_offset_mid_m = 0.5 * (bulk_offset_top_m + bulk_offset_bottom_m)
            continue
        with xr.open_dataset(path,decode_times=False) as ds:
            if (bulk_offset_top_m is None):
                bulk_offset_top_m = ds["bulk_level_offset_top"].values.astype(float)
                bulk_offset_bottom_m = ds["bulk_level_offset_bottom"].values.astype(float)
                bulk_offset_mid_m = 0.5 * (bulk_offset_top_m + bulk_offset_bottom_m)
            elif np.isnan(bulk_offset_top_m).all():
                bulk_offset_top_m = ds["bulk_level_offset_top"].values.astype(float)
                bulk_offset_bottom_m = ds["bulk_level_offset_bottom"].values.astype(float)
                bulk_offset_mid_m = 0.5 * (bulk_offset_top_m + bulk_offset_bottom_m)
            lat_idx = xr.DataArray(row["lat_bin_idx"][i], dims="pixel")
            lon_idx = xr.DataArray(row["lon_bin_idx"][i], dims="pixel")

            cb_iwc_values = ds["cb_iwc_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            ct_iwc_values = ds["cb_top_iwc_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            cb_height_values = ds["cb_height_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            ct_height_values = ds["cb_top_height_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            ct_temp_values = ds["cb_top_temp_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            cb_temp_values = ds["cb_temp_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            cb_ext_values = ds["cb_extinction_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )
            cb_Z_values = ds["cb_Z_mean"].isel(
                lat_bin=lat_idx,
                lon_bin=lon_idx,
            )

            if CONVERT_LINEAR_Z_TO_DBZE:
                cb_Z_values = _to_dBZe(cb_Z_values)

            cb_iwc_profile = cb_iwc_values.mean(dim="pixel").isel(time_bin=0).values
            cb_height_profile = cb_height_values.mean(dim="pixel").isel(time_bin=0).values
            cb_ext_profile = cb_ext_values.mean(dim="pixel").isel(time_bin=0).values
            cb_Z_profile = (
                cb_Z_values
                .mean(dim="pixel", skipna=True)
                .isel(time_bin=0)
                .values
            )
            cb_temp_profile = cb_temp_values.mean(dim="pixel").isel(time_bin=0).values

            ct_iwc_mean = float(ct_iwc_values.mean(dim="pixel").isel(time_bin=0).values)
            ct_height_mean = float(ct_height_values.mean(dim="pixel").isel(time_bin=0).values)
            ct_temp_mean = float(ct_temp_values.mean(dim="pixel").isel(time_bin=0).values)

            cb_iwc_means.append(cb_iwc_profile)
            cb_height_means.append(cb_height_profile)
            cb_ext_means.append(cb_ext_profile)
            cb_temp_means.append(cb_temp_profile)
            cb_Z_means.append(cb_Z_profile)

            ct_iwc_means.append(ct_iwc_mean)
            ct_height_means.append(ct_height_mean)
            ct_temp_means.append(ct_temp_mean)

    return (
        cb_iwc_means,
        ct_iwc_means,
        cb_height_means,
        ct_height_means,
        ct_temp_means,
        cb_ext_means,
        cb_temp_means,
        cb_Z_means,
        bulk_offset_top_m,
        bulk_offset_bottom_m,
        bulk_offset_mid_m,
    )

def make_parquet_safe(value):
    """
    Convert values that often cause issues in parquet object columns
    into safer Python-native representations.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.DatetimeIndex):
        return value.astype(str).tolist()

    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, list):
        return [
            v.tolist() if isinstance(v, np.ndarray) else v
            for v in value
        ]

    return value

def extract_iwc_relative_to_peak(row):
    records = []

    track_start = pd.Timestamp(row["track_start_time"])
    peak_ind = int(row["Glac_peak_ind"])

    # Save all original valid_clouds information for this cloud
    original_cloud_info = {
        col: make_parquet_safe(row[col])
        for col in ORIGINAL_VALID_CLOUD_COLUMNS
    }

    for i, iwc_profile in enumerate(row["iwc_bulk_means"]):
        col_time = pd.Timestamp(row["colocation_time"][i])

        col_ind = int(round((col_time - track_start).total_seconds() / (15 * 60)))
        rel_ind = col_ind - peak_ind

        record = {
            "Cloud_ID": row["Cloud_ID"],
            "rel_ind": rel_ind,
            "rel_time_min": rel_ind * 15,
            "absolute_colocation_time": col_time,

            "iwc_profile": np.asarray(iwc_profile),
            "iwc_ct": row["iwc_top_means"][i],

            "bulk_height_profile_m": np.asarray(row["bulk_height_means_m"][i]),
            "cloud_top_height_m": row["ddc_cloud_top_height_means_m"][i],
            "cloud_top_temp_K": row["ddc_cloud_top_temp_means_K"][i],

            "bulk_level_offset_top_m": np.asarray(row["bulk_level_offset_top_m"]),
            "bulk_level_offset_bottom_m": np.asarray(row["bulk_level_offset_bottom_m"]),
            "bulk_level_offset_mid_m": np.asarray(row["bulk_level_offset_mid_m"]),
        }

        # Add the full original cloud information to this individual record
        record.update(original_cloud_info)

        records.append(record)

    return records

    
valid_clouds[["lat_bin_idx", "lon_bin_idx"]] = (
    valid_clouds
    .apply(get_colocation_ind, axis=1, result_type="expand")
)

valid_clouds[
    [
        "iwc_bulk_means",
        "iwc_top_means",
        "bulk_height_means_m",
        "ddc_cloud_top_height_means_m",
        "ddc_cloud_top_temp_means_K",
        "ext_bulk_means",
        "temp_bulk_means",
        "Z_bulk_means",
        "bulk_level_offset_top_m",
        "bulk_level_offset_bottom_m",
        "bulk_level_offset_mid_m",
    ]
] = valid_clouds.apply(get_bulk_values, axis=1, result_type="expand")

all_records = []

for _, row in valid_clouds.iterrows():
    all_records.extend(extract_iwc_relative_to_peak(row))

iwc_rel_df = pd.DataFrame(all_records)
iwc_stats_by_rel_time = (
    iwc_rel_df
    .groupby("rel_ind")
    .agg(
        n_hits=("iwc_profile", "count"),

        mean_iwc_profile=(
            "iwc_profile",
            lambda profiles: np.nanmean(np.stack(profiles), axis=0),
        ),

        mean_bulk_height_profile_m=(
            "bulk_height_profile_m",
            lambda profiles: np.nanmean(np.stack(profiles), axis=0),
        ),

        bulk_offset_top_m=(
            "bulk_level_offset_top_m",
            lambda profiles: np.nanmean(np.stack(profiles), axis=0),
        ),

        bulk_offset_bottom_m=(
            "bulk_level_offset_bottom_m",
            lambda profiles: np.nanmean(np.stack(profiles), axis=0),
        ),

        dd_cth=(
            "cloud_top_height_m",
            "mean",
        ),

        sev_cth=(
            "avg_cth",
            "mean",
        ),

        dd_ctt=(
            "cloud_top_temp_K",
            "mean",
        ),

        mean_iwc_ct=(
            "iwc_ct",
            "mean",
        ),

        sev_ctt=(
            "avg_ctt",
            "mean",
        ),
        
        mean_area_km=(
            "avg_size[km]",
            "mean",
        ),
        
    )
    .reset_index()
)
iwc_stats_by_rel_time["rel_time_min"] = iwc_stats_by_rel_time["rel_ind"] * 15

iwc_stats_by_rel_time["rel_time_min"] = iwc_stats_by_rel_time["rel_ind"] * 15
stats_out_fp = os.path.join(OUTPUT_FOLDER, "Bulk_statistics")
os.makedirs(stats_out_fp, exist_ok=True)

iwc_rel_df.to_parquet(
    os.path.join(stats_out_fp, f"iwc_rel_df_V3_1_{ANALYSIS_YEAR}.parquet")
)

iwc_stats_by_rel_time.to_parquet(
    os.path.join(stats_out_fp, f"mean_iwc_by_height_data_V3_1_{ANALYSIS_YEAR}.parquet")
)

valid_clouds.to_parquet(
    os.path.join(stats_out_fp, f"glaciations_with_bulk_data_V3_1_{ANALYSIS_YEAR}.parquet")
)
