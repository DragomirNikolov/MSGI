# Instructions: How to use `modis_cloud_top_binning.py`

## 1) Activate Python interpreter
Use the project environment Python directly:

```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python --version
```

## 2) Run on one MODIS granule (single-file mode)
Example:

```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python \
  /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py \
  --input /wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua/2007/002/CLDPROP_L2_MODIS_Aqua.A2007002.0015.011.2021143102716.nc \
  --output-root /wolke_scratch/dnikolo/dump/modis_cloud_top \
  --lat-min -90 --lat-max 90 --dlat 0.3 \
  --lon-min -180 --lon-max 180 --dlon 0.3
```

Expected output for the example above:

```text
/wolke_scratch/dnikolo/dump/modis_cloud_top/2007/01/02/MOD_CT_20070102_0015.nc
```

## 3) Run over a selected time period (directory mode)
Use the MODIS base directory as `--input` and provide a datetime range.
By default, this runs in parallel with `8` workers:

```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python \
  /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py \
  --input /wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua \
  --output-root /wolke_scratch/dnikolo/dump/modis_cloud_top_period \
  --lat-min -90 --lat-max 90 --dlat 0.3 \
  --lon-min -180 --lon-max 180 --dlon 0.3 \
  --start-time 20070102_0000 \
  --end-time 20070102_0030 \
  --n-workers 8
```

This scans `YYYY/DDD/*.nc`, filters by filename time in `[start, end]`, and writes one `MOD_CT_...nc` per accepted granule.
It also prints a summary dictionary at the end (`seen`, `in_range`, `processed`, `skipped_non_quarter_hour`, `overwritten`, `failed`).
All outputs are computed only from observations where `sensor_zenith < 18.0°`.

## 4) Run period mode sequentially (for comparison/debug)
```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python \
  /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py \
  --input /wolke_scratch/dnikolo/MODIS/CLDPROP_L2_MODIS_Aqua \
  --output-root /wolke_scratch/dnikolo/dump/modis_cloud_top_period \
  --lat-min -90 --lat-max 90 --dlat 0.3 \
  --lon-min -180 --lon-max 180 --dlon 0.3 \
  --start-time 20070102_0000 \
  --end-time 20070102_0030 \
  --n-workers 1
```

## 5) Quarter-hour behavior
Default behavior skips files not starting at minute `00/15/30/45`.

If you want to force processing of any minute:

```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python \
  /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py \
  --input <input_file.nc> \
  --output-root <output_root> \
  --lat-min <lat_min> --lat-max <lat_max> --dlat <dlat> \
  --lon-min <lon_min> --lon-max <lon_max> --dlon <dlon> \
  --no-skip-non-quarter-hour
```

## 6) Overwrite behavior
Default is overwrite enabled. If output already exists, it is replaced and logged.

Disable overwrite with:

```bash
--no-overwrite
```

## 7) Required arguments in directory mode
If `--input` is a directory, both are required:
- `--start-time YYYYMMDD_HHMM`
- `--end-time YYYYMMDD_HHMM`

Without them, the script raises:
- `ValueError: Directory input requires --start-time and --end-time in YYYYMMDD_HHMM`

## 8) Check output header quickly
```bash
ncdump -h /wolke_scratch/dnikolo/dump/modis_cloud_top/2007/01/02/MOD_CT_20070102_0015.nc | sed -n '1,220p'
```

## 9) Show CLI help
```bash
/wolke_scratch/dnikolo/flex_trkr_env/bin/python \
  /wolke_scratch/dnikolo/MSGI/msgi/obs/modis/modis_cloud_top_binning.py --help
```

## Notes
- Phase code `6` (undetermined) is treated as invalid and excluded from all stats/counts.
- `n_obs` counts only valid classified observations (`0/1/2/3`).
- `cph_mean` excludes no-cloud samples and uses: water `1`, mixed `0.5`, ice `2`.
- All counts/fractions/means/stds use only `sensor_zenith < 18.0°` observations.
