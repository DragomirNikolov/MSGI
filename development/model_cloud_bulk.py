import xarray as xr
import numpy as np

# ============================================================
# User settings
# ============================================================

max_depth_m = 3000.0
dz_m = 120.0

# Change these if your variable names are different
zg_name = "zg"
ze_name = "cloudsat_ze_mean"
iwc_name = "cli"


atm_3d = xr.open_dataset("/wolke_scratch/dnikolo/ICON_Data/cosp_dardar_runs/R2B4/wbf_v3/raw_data/sedim_r2b4_wbfv3_atm_3d_cloud_ml_20071221T000000Z.nc")
ct = xr.open_dataset("/wolke_scratch/dnikolo/ICON_Data/cosp_dardar_runs/R2B4/wbf_v3/raw_data/sedim_r2b4_wbfv3_ctp_2d_ml_20071221T000000Z.nc")



ctmask_3d = (
    ct["ctmask"]
    .expand_dims(height=atm_3d.height)
    .broadcast_like(atm_3d["cli"])
    .transpose("time", "height", "ncells")
)
CPH = xr.where(((ctmask_3d>1e-5) & ((atm_3d["cli"]+atm_3d["clw"])>=1e-5)),(atm_3d["cli"]/(atm_3d["cli"]+atm_3d["clw"])+1),0)

# ctlev is probably 1-based because height is 1..47
# Convert model level number to zero-based positional index
ctlev_idx = ct["ctlev"].astype(float)

rel_level = xr.DataArray(
    np.arange(4),
    dims="rel_level",
    coords={"rel_level": np.arange(4)}
)

# cloud top + 1/2/3 levels below
target_idx = ctlev_idx + rel_level
nlev = atm_3d.sizes["height"]

valid = (
    np.isfinite(target_idx) &
    (target_idx >= 0) &
    (target_idx < nlev)
)

target_idx_safe = (
    target_idx
    .where(valid, 0)
    .astype(int)
)

CPH_ct_profile = (
    CPH
    .isel(height=target_idx_safe)
    .where(valid, other=0)
    .transpose("time", "rel_level", "ncells")
)




# If ctlev ranges from 1..47, set this to True.
# If ctlev ranges from 0..46, set this to False.
ctlev_is_one_based = False


# ============================================================
# Target cloud-top-relative vertical grid
# ============================================================

# Bin centers: 60, 180, ..., 2100 m
depth_bins = xr.DataArray(
    np.arange(dz_m / 2, max_depth_m, dz_m),
    dims="depth_bin",
    coords={"depth_bin": np.arange(dz_m / 2, max_depth_m, dz_m)},
    name="depth_below_ct_m",
)

depth_bins.attrs["long_name"] = "Depth below cloud top"
depth_bins.attrs["units"] = "m"


# ============================================================
# Cloud-top level handling
# ============================================================

nlev = atm_3d.sizes["height"]

ctlev_raw = ct["ctlev"].astype(float)

if ctlev_is_one_based:
    ctlev_idx = ctlev_raw - 1
else:
    ctlev_idx = ctlev_raw

valid_ctlev = (
    np.isfinite(ctlev_idx) &
    (ctlev_idx >= 0) &
    (ctlev_idx < nlev) &
    (ct["ctmask"] > 1e-5)
)

ctlev_idx_safe = (
    ctlev_idx
    .where(valid_ctlev, 0)
    .astype(int)
)


# ============================================================
# Get cloud-top geometric height
# ============================================================

zg = (
    atm_3d[zg_name]
    .broadcast_like(atm_3d["cli"])
    .transpose("time", "height", "ncells")
)

zg_ct = zg.isel(height=ctlev_idx_safe)

# Positive depth means below cloud top
depth_below_ct = zg_ct - zg

depth_below_ct = depth_below_ct.transpose("time", "height", "ncells")
depth_below_ct.name = "depth_below_ct"


# ============================================================
# Interpolation helper
# ============================================================

def interp_to_depth_bins(depth_col, var_col, target_depths):
    """
    Interpolate one vertical column to fixed cloud-top-relative depths.

    Parameters
    ----------
    depth_col : array, shape (height,)
        Depth below cloud top in meters. Positive values are below cloud top.

    var_col : array, shape (height,)
        Variable profile to interpolate.

    target_depths : array, shape (depth_bin,)
        Target depths below cloud top.

    Returns
    -------
    out : array, shape (depth_bin,)
        Interpolated profile.
    """

    depth_col = np.asarray(depth_col)
    var_col = np.asarray(var_col)
    target_depths = np.asarray(target_depths)

    valid = (
        np.isfinite(depth_col) &
        np.isfinite(var_col) &
        (depth_col >= 0.0) &
        (depth_col <= max_depth_m)
    )

    if valid.sum() < 2:
        return np.full(target_depths.shape, np.nan, dtype=float)

    x = depth_col[valid]
    y = var_col[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate depth values
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if len(x_unique) < 2:
        return np.full(target_depths.shape, np.nan, dtype=float)

    return np.interp(
        target_depths,
        x_unique,
        y_unique,
        left=np.nan,
        right=np.nan,
    )


# ============================================================
# Generic function for one variable
# ============================================================

def make_ct_relative_profile(var):
    """
    Convert a 3D variable with dims (time, height, ncells)
    into cloud-top-relative depth coordinates.

    Output dims: (time, depth_bin, ncells)
    """

    var = var.transpose("time", "height", "ncells")

    out = xr.apply_ufunc(
        interp_to_depth_bins,
        depth_below_ct,
        var,
        depth_bins,
        input_core_dims=[["height"], ["height"], ["depth_bin"]],
        output_core_dims=[["depth_bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    out = out.transpose("time", "depth_bin", "ncells")

    # Mask invalid cloud-top pixels
    out = out.where(valid_ctlev)

    return out


# ============================================================
# Prepare variables
# ============================================================

ze = atm_3d[ze_name]
iwc = atm_3d[iwc_name]


# ============================================================
# Option A: interpolate Ze directly
# ============================================================

ze_depth = make_ct_relative_profile(ze)
iwc_depth = make_ct_relative_profile(iwc)


# ============================================================
# Option B: better for reflectivity statistics
# Convert dBZ -> linear Z, interpolate, then back to dBZ.
# Use this if ze is in dBZ.
# ============================================================

# Z_linear = 10.0 ** (ze / 10.0)
# Z_linear_depth = make_ct_relative_profile(Z_linear)
# ze_depth = 10.0 * np.log10(Z_linear_depth)


# ============================================================
# Optional: also compute CPH and interpolate it
# ============================================================

ctmask_3d = (
    ct["ctmask"]
    .expand_dims(height=atm_3d.height)
    .broadcast_like(atm_3d["cli"])
    .transpose("time", "height", "ncells")
)

condensate = atm_3d["cli"] + atm_3d["clw"]

CPH = xr.where(
    ((ctmask_3d > 1e-5) & (condensate >= 1e-5)),
    atm_3d["cli"] / condensate + 1.0,
    0.0,
)

CPH_depth = make_ct_relative_profile(CPH)


# ============================================================
# Final compact dataset
# ============================================================

ct_relative_ds = xr.Dataset(
    data_vars={
        "ze": ze_depth,
        "iwc": iwc_depth,
        "cph": CPH_depth,
    },
    coords={
        "time": atm_3d["time"],
        "depth_bin": depth_bins,
        "ncells": atm_3d["ncells"],
        "clon": atm_3d["clon"],
        "clat": atm_3d["clat"],
    },
)

ct_relative_ds["ze"].attrs = atm_3d[ze_name].attrs
ct_relative_ds["iwc"].attrs = atm_3d[iwc_name].attrs
ct_relative_ds["cph"].attrs["long_name"] = "Cloud phase interpolated to cloud-top-relative depth"
ct_relative_ds["cph"].attrs["units"] = "1 liquid, 2 ice"

ct_relative_ds["depth_bin"].attrs["long_name"] = "Depth below cloud top"
ct_relative_ds["depth_bin"].attrs["units"] = "m"


# ============================================================
# Check result
# ============================================================
print(ct_relative_ds)

print("ze shape:", ct_relative_ds["ze"].shape)
print("iwc shape:", ct_relative_ds["iwc"].shape)
print("cph shape:", ct_relative_ds["cph"].shape)


output_path = (
    "/wolke_scratch/dnikolo/ICON_Data/cosp_dardar_runs/"
    "R2B4/wbf_v3/bulk_data/test.nc"
)

compression = {
    var_name: {
        "zlib": True,
        "complevel": 9,
        "shuffle": True,
    }
    for var_name in ct_relative_ds.data_vars
}

ct_relative_ds.to_netcdf(
    output_path,
    # engine="netcdf4",
    format="NETCDF4",
    encoding=compression,
)

print(np.isnan(atm_3d["cloudsat_ze_mean"]).mean())
print(np.isnan(atm_3d["cli"]).mean())
print(np.isnan(ct_relative_ds.iwc).all())
print(np.isnan(ct_relative_ds["ze"]).all())