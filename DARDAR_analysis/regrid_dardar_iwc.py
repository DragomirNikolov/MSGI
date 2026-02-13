"""
Regrid DARDAR-CLOUD IWC (vertically averaged) onto a 2x2 degree lat/lon grid.

Usage:
    python regrid_dardar_iwc.py DARDAR-CLOUD_2006164050952_00668_V3-10.nc

Output:
    DARDAR-CLOUD_2006164050952_00668_V3-10_iwc_2x2.nc
"""

import sys
import numpy as np
import h5py
import netCDF4 as nc

# ── Configuration ──────────────────────────────────────────────────────────
GRID_RES = 2.0  # degrees
FILL_VALUE = -999.0
IWC_FILL = -999.0

# DARDAR-CLOUD fixed height grid: 436 levels, 60m spacing, 25080m down to -1020m
DARDAR_NHEIGHTS = 436
DARDAR_DZ = 60.0  # m
DARDAR_HEIGHT_TOP = 25080.0  # m


def reconstruct_height(n=DARDAR_NHEIGHTS):
    """Reconstruct the DARDAR height axis (fixed across all files)."""
    return np.arange(DARDAR_HEIGHT_TOP, DARDAR_HEIGHT_TOP - n * DARDAR_DZ, -DARDAR_DZ)


def read_dardar(filepath):
    """
    Read latitude, longitude, height, and IWC from a DARDAR-CLOUD file.

    Tries netCDF4 first, falls back to h5py with reconstructed height if needed.
    """
    # Try netCDF4 first (preferred — handles masking/fill values)
    try:
        ds = nc.Dataset(filepath)
        ds.set_auto_mask(False)
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        height = ds.variables["height"][:]
        iwc = ds.variables["iwc"][:]
        ds.close()
        return lat, lon, height, iwc
    except RuntimeError:
        pass

    # Fallback: h5py for 2D vars, reconstruct height
    print("WARNING: Some coordinate variables are unreadable (possible file corruption).")
    print("         Reconstructing height from DARDAR specification.")
    print("         Attempting h5py fallback for lat/lon...")

    f = h5py.File(filepath, "r")
    n_time = f["iwc"].shape[0]
    n_height = f["iwc"].shape[1]

    # Try reading lat/lon (may fail if file is truncated)
    try:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    except OSError:
        f.close()
        raise RuntimeError(
            "Cannot read latitude/longitude — file appears corrupted. "
            "Please re-download the file."
        )

    # Reconstruct height
    height = reconstruct_height(n_height)

    # Read IWC row by row (works with chunked 2D data)
    iwc = np.empty((n_time, n_height), dtype=np.float32)
    for i in range(n_time):
        iwc[i, :] = f["iwc"][i, :]

    f.close()
    return lat, lon, height, iwc


def vertical_mean_iwc(iwc, fill_value=IWC_FILL):
    """
    Compute column-mean IWC for each profile, ignoring fill/invalid values.

    Only averages over height levels where IWC > 0 (i.e., where ice cloud exists).
    Returns NaN for profiles with no valid IWC data.
    """
    iwc_masked = np.where((iwc > 0) & (iwc != fill_value), iwc, np.nan)
    with np.errstate(all="ignore"):
        col_mean = np.nanmean(iwc_masked, axis=1)
    return col_mean


def regrid_to_2x2(lat, lon, values, grid_res=GRID_RES):
    """
    Bin irregularly-spaced satellite track data onto a regular lat/lon grid.

    Uses simple bin-averaging: for each grid cell, average all track points
    that fall within it.

    Returns:
        lat_grid (1D), lon_grid (1D), gridded_mean (2D), gridded_count (2D)
    """
    # Define grid edges and centers
    lat_edges = np.arange(-90, 90 + grid_res, grid_res)
    lon_edges = np.arange(-180, 180 + grid_res, grid_res)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    nlat = len(lat_centers)
    nlon = len(lon_centers)

    # Accumulator arrays
    grid_sum = np.zeros((nlat, nlon), dtype=np.float64)
    grid_count = np.zeros((nlat, nlon), dtype=np.int64)

    # Filter valid data points
    valid = np.isfinite(values) & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
    lat_v = lat[valid]
    lon_v = lon[valid]
    val_v = values[valid]

    # Compute bin indices
    ilat = np.clip(((lat_v - (-90)) / grid_res).astype(int), 0, nlat - 1)
    ilon = np.clip(((lon_v - (-180)) / grid_res).astype(int), 0, nlon - 1)

    # Accumulate using np.add.at for speed
    np.add.at(grid_sum, (ilat, ilon), val_v)
    np.add.at(grid_count, (ilat, ilon), 1)

    # Compute mean (NaN where no data)
    with np.errstate(invalid="ignore"):
        grid_mean = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    return lat_centers, lon_centers, grid_mean, grid_count


def save_gridded(outpath, lat_grid, lon_grid, iwc_grid, count_grid, source_file):
    """Save the regridded IWC to a CF-compliant NetCDF file."""
    ds = nc.Dataset(outpath, "w", format="NETCDF4")

    # Dimensions
    ds.createDimension("lat", len(lat_grid))
    ds.createDimension("lon", len(lon_grid))

    # Coordinate variables
    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlat.units = "degrees_north"
    vlat.long_name = "latitude"
    vlat.standard_name = "latitude"
    vlat[:] = lat_grid

    vlon = ds.createVariable("lon", "f4", ("lon",))
    vlon.units = "degrees_east"
    vlon.long_name = "longitude"
    vlon.standard_name = "longitude"
    vlon[:] = lon_grid

    # IWC variable
    viwc = ds.createVariable("iwc_column_mean", "f4", ("lat", "lon"),
                             fill_value=np.float32(np.nan))
    viwc.units = "kg m-3"
    viwc.long_name = "Column-mean Ice Water Content on 2x2 degree grid"
    viwc.comment = ("Vertical mean of IWC over height levels with ice cloud present, "
                    "then bin-averaged onto a 2x2 degree grid.")
    viwc[:] = iwc_grid

    # Count variable (number of satellite profiles per grid cell)
    vcnt = ds.createVariable("profile_count", "i4", ("lat", "lon"))
    vcnt.long_name = "Number of satellite profiles per grid cell"
    vcnt[:] = count_grid

    # Global attributes
    ds.Conventions = "CF-1.7"
    ds.title = "DARDAR-CLOUD IWC regridded to 2x2 degree grid"
    ds.source = f"Regridded from {source_file}"
    ds.grid_resolution = f"{GRID_RES} degrees"
    ds.history = "Vertical mean of IWC (ice-cloud levels only), bin-averaged to 2x2 deg grid"

    ds.close()
    print(f"Saved: {outpath}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <DARDAR-CLOUD_file.nc>")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = infile.replace(".nc", "_iwc_2x2.nc")

    print(f"Reading {infile} ...")
    lat, lon, height, iwc = read_dardar(infile)
    print(f"  Profiles: {len(lat)}, Height levels: {len(height)}")
    print(f"  Lat range: {lat[lat > -900].min():.1f} to {lat[lat > -900].max():.1f}")
    print(f"  Lon range: {lon[lon > -900].min():.1f} to {lon[lon > -900].max():.1f}")
    print(f"  Height range: {height.min():.0f} to {height.max():.0f} m")

    print("Computing vertical mean IWC ...")
    iwc_col = vertical_mean_iwc(iwc)
    n_valid = np.sum(np.isfinite(iwc_col))
    print(f"  Profiles with ice cloud: {n_valid} / {len(iwc_col)}")

    print(f"Regridding to {GRID_RES}x{GRID_RES} degree grid ...")
    lat_grid, lon_grid, iwc_grid, count_grid = regrid_to_2x2(lat, lon, iwc_col)
    n_cells = np.sum(count_grid > 0)
    print(f"  Grid cells with data: {n_cells} / {len(lat_grid)*len(lon_grid)}")

    save_gridded(outfile, lat_grid, lon_grid, iwc_grid, count_grid,
                 source_file=infile.split("/")[-1])


if __name__ == "__main__":
    main()
