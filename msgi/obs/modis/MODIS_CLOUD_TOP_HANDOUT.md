# MODIS Cloud-Top Binning Handout

## What this script does
`modis_cloud_top_binning.py` bins MODIS CLDPROP L2 cloud-top properties onto a regular lon-lat grid.

It supports:
- single-file processing (`--input` is one `.nc`)
- period processing (`--input` is base folder like `/wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua/`)
  with multiprocessing workers (`--n-workers`, default `8`)

Outputs are NetCDF files named:
- `MOD_CT_YYYYMMDD_HHMM.nc`

inside:
- `OUTPUT_ROOT/YYYY/MM/DD/`

## Input data used
From MODIS CLDPROP L2:
- `geolocation_data/latitude`
- `geolocation_data/longitude`
- `geophysical_data/Cloud_Phase_Cloud_Top_Properties`
- `geophysical_data/Cloud_Top_Height`
- `geophysical_data/Cloud_Top_Temperature`
- global attribute `StartTime` for output timestamp

## Phase mapping used
- no cloud (`0`) -> class `0`
- water (`1`) -> class value `1`
- mixed (`3`) -> class value `0.5`
- ice (`2`) -> class value `2`
- undetermined (`6`) -> invalid (excluded from all stats/counts)
- fill/other invalid values -> excluded from all stats/counts

## Important processing rules
- By default, only quarter-hour granules are processed:
  - `HH:00`, `HH:15`, `HH:30`, `HH:45`
- Non-quarter-hour files are skipped unless `--no-skip-non-quarter-hour` is passed.
- In period mode, files are first filtered by filename datetime range (`A%Y%j.%H%M`) and then quarter-hour filtering is applied.
- In period mode, selected files are processed in parallel via `multiprocessing.Pool`.
- `cph_mean` is computed from cloudy-only samples (water/mixed/ice), excluding no-cloud.
- `cth_*` and `ctt_*` are also computed from cloudy-only samples.
- Overwrite behavior is enabled by default (`--overwrite`), with explicit overwrite messages.
- Period mode prints a final summary with:
  - `seen`, `in_range`, `processed`, `skipped_non_quarter_hour`, `overwritten`, `failed`

## Main output variables
- `cph_mean`
- `cth_mean`, `cth_std`
- `ctt_mean`, `ctt_std`
- `frac_no_cloud`, `frac_liquid`, `frac_mixed`, `frac_ice`
- `n_obs`, `n_no_cloud`, `n_liquid`, `n_mixed`, `n_ice`, `n_cloudy`
- `rounded_mean_time`

The file layout and metadata are DD_CT-like (`time_bin`, `lat_bin`, `lon_bin`, edges, CF-style attrs).

## Key CLI options
- Required always:
  - `--input`
  - `--output-root`
  - `--lat-min --lat-max --dlat`
  - `--lon-min --lon-max --dlon`
- Optional:
  - `--skip-non-quarter-hour` / `--no-skip-non-quarter-hour`
  - `--overwrite` / `--no-overwrite`
  - `--n-workers` (period mode, default `8`)
- Required for directory/period mode:
  - `--start-time YYYYMMDD_HHMM`
  - `--end-time YYYYMMDD_HHMM`

## Run environment
Use:
- `/wolke_scratch/dnikolo/flex_trkr_env/bin/python`

Script location:
- `/wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py`
