# DARDAR cloud-top and bulk pipeline

[Back to MSGI](../README.md) · [Installation and GTE](gte-pipeline.md) · [Model comparison](model-pipeline.md)

This workflow first validates **GTE-tracked CLAAS cloud tops** using DARDAR-MASK, then samples DARDAR-CLOUD vertical properties at the collocated track locations and times. Top-only validation stops after step 2; steps 3–5 add bulk statistics. Satellite overpasses provide sparse observations, not continuous cloud histories.

## 1. Bin DARDAR-MASK cloud tops

Organize DARDAR-MASK files as:

```text
/data/DARDAR-MASK/YYYY/YYYY_MM_DD/DARDAR-MASK_*.nc
```

Each file must expose `DARMASK_Simplified_Categorization(time,height)`, `CS_TRACK_Height(height)`, `CLOUDSAT_UTC_Time(time)`, `CLOUDSAT_Latitude(time)`, and `CLOUDSAT_Longitude(time)`. Latitude/longitude are required for spatial binning. The extractor handles either vertical orientation and interprets height units to obtain kilometers.

The module's `__main__` uses fixed paths/dates; call the public function instead:

```bash
python - <<'PY'
from pathlib import Path
from msgi.obs.dardar.cloud_top_binning import process_multiple_days

process_multiple_days(
    base_input_dir=Path('/data/DARDAR-MASK'),
    base_output_dir=Path('/data/dardar/top_02'),
    start_date='20070101',
    end_date='20071231',  # Inclusive.
    bin_size_deg=0.2,
    n_workers=4,
)
PY
```

For one day, use `process_one_day(input_dir, output_dir, bin_size_deg=0.2, n_workers=4)`, with `output_dir` ending in `YYYY/MM/DD` so midnight spillover is moved to the correct day. A one-file inspection is available through `extract_cloud_top_phase4(ds)` and `bin_cloud_top_phase4_latlon(profile, bin_size_deg=0.2)`; save that dataset with xarray. The function `get_single_binned_cloud_top` defines a CLI parser but is not the active module entry point.

The extractor finds the highest valid cloud-mask bin, excluding `-99, -2, -1, 0, 6, 8`. Top codes map as follows:

| Class | Codes |
| --- | --- |
| Liquid | 3, 5, 7, 11, 12, 13, 15 |
| Ice | 1, 2, 9, 10 |
| Mixed | 4, 14 |

Unmapped/no-candidate profiles carry QA flags at the profile stage. The binner computes a cloudy-only phase score (liquid=1, mixed=1.5, ice=2), height mean/std, class fractions among all observations, and observation counts. It assigns each spatial bin to the nearest **15-minute mean cloudy-observation time**. Top spatial extents are derived from each input granule, rather than fixed global limits.

```text
/data/dardar/top_02/YYYY/MM/DD/DD_CT_YYYYMMDD_HHMM.nc
```

These files contain `cph_mean`, `cth_mean`, `cth_std`, phase fractions and counts, with `time_bin`, `lat_bin`, and `lon_bin`. Heights are in **km**. A phase mean of `-1` marks no cloudy tops; missing observations are not liquid clouds. xarray can decode count fill values of zero to NaN, so use `mask_and_scale=False` if you need raw counts.

Per-granule files initially have a numeric suffix. The daily helper merges duplicate timestamps using CDO `ensmean`, removes the suffix inputs, and moves midnight spillover to its correct day. This is an ensemble mean of the files, **not a pooled, observation-count-weighted statistic**. Review merge logs for `[ERROR]`; use a clean output root for an independent rerun, because existing merged files and new inputs can otherwise interact.

## 2. Track CLAAS and validate with DARDAR tops

Prepare a complete GTE YAML as described in the [integration guide](gte-pipeline.md), with these changes:

```yaml
validation_mode: DARDAR
val_CPH_fp: /data/dardar/top_02
collect_additional_properties: true
Resample: false
agg_fact: 3
t_deltas: [6]
glac_threshold: 0.4
```

`CLAAS_fp` continues to point to the original CLAAS CPP/CTX data root. `val_CPH_fp` is the DARDAR top root above; replace obsolete `DARDAR_CPH_fp` in older examples. Keep CLAAS auxiliary geometry/pixel-area files and dedicated tracking/results directories in the configuration.

Run CLAAS preprocessing, tracking, postprocessing, and detection following the [GTE commands](gte-pipeline.md#run-tracking-and-postprocessing). GTE reads `DD_CT` files during postprocessing, reindexes their regular grid to tracked pixels, and records validation histories. Preserve the cloud tables as well as the glaciation tables.

For annual bulk analysis, complete the annual combination/detection stage to obtain:

```text
postprocessing_output_dir/Final_results/2007_all.parquet
postprocessing_output_dir/Final_results/2007_glac_04.parquet
```

The year comes from GTE `start_time.year`. Confirm glaciation rows contain `track_start_time`, `Glac_peak_ind`, `val_ice_frac_hist`, `val_cth_hist`, `cth_hist`, `val_intersec_lat`, and `val_intersec_lon`. Nonzero/valid validation histories establish the overpass coincidences used below. If the requested period has no collocations, the downstream scripts do not produce a meaningful bulk sample.

## 3. Generate DARDAR-CLOUD bulk files

DARDAR-CLOUD inputs are a separate product:

```text
/data/DARDAR-CLOUD/V3_1/YYYY/YYYY_MM_DD/DARDAR-CLOUD_*.nc
```

Required variables are `DARMASK_Simplified_Categorization`, `height`, `time`, `latitude`, `longitude`, `iwc`, `temperature`, `extinction`, and `Z`. Mask/IWC/temperature/extinction/reflectivity must be 2D and share the profile/height layout. The bulk extractor handles either height orientation and reconstructs nonfinite heights from the nominal DARDAR axis (436 levels, 60 m spacing, 25,080 m top). Its excluded cloud-mask codes also include `-9`.

Bulk layers default to 25 two-bin layers: 0 to −120 m, −120 to −240 m, …, −2,880 to −3,000 m relative to the identified cloud top. Top IWC/height/temperature use the top pair of native bins. Nonpositive IWC and negative extinction are masked; `Z` retains finite source values. Phase scores use the same liquid/mixed/ice convention as top analysis.

Choose either all-days processing or the track-selected route below. Keep the grid spacing and bounds consistent throughout a bulk run.

### All files in a date range

```bash
python -m msgi.obs.dardar.dardar_cloud_bulk_binning \
  --input-root /data/DARDAR-CLOUD/V3_1 \
  --output-root /data/dardar/bulk_daily \
  --start-date 20070101 --end-date 20071231 \
  --lat-min -81 --lat-max 81 --dlat 0.2 \
  --lon-min -81 --lon-max 81 --dlon 0.2 \
  --n-workers 4
```

Both dates are inclusive. For a single granule, replace the input/date arguments with `--single-file /path/to/DARDAR-CLOUD_....nc`; keep `--output-root` and the spatial bounds. Single-file mode also moves results into dated subdirectories. The CLI defaults to 0.3°; the example deliberately uses the 0.2° grid of the track-selected workflow.

Daily outputs are:

```text
/data/dardar/bulk_daily/YYYY/MM/DD/DD_CB_YYYYMMDD_HHMM.nc
```

These helpers merge numeric-suffix duplicates with CDO `ensmean` and move wrong-day files, as in the top workflow. Do not merge different grids/resolutions together. The bulk-statistics consumer expects a **flat directory** instead of this daily tree; create a separate flat view of a single year's files if using this route:

```python
from pathlib import Path
source = Path('/data/dardar/bulk_daily/2007')
flat = Path('/data/dardar/bulk_flat/2007')
flat.mkdir(parents=True, exist_ok=True)
for src in source.glob('*/*/DD_CB_*.nc'):
    dst = flat / src.name
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.symlink_to(src.resolve())
```

### Files selected by tracked-cloud coincidences

[`3d_dardar_tracking.py`](../msgi/obs/dardar/3d_dardar_tracking.py) reads annual validated glaciations, selects valid 15-minute history times, scans raw DARDAR-CLOUD files for overlap, and processes only the matching **files**. It currently processes all bins in each selected file, not just the matched timestamps. It writes merged files directly into a flat output directory, suitable for bulk statistics.

The script defaults to site-specific raw/output paths and a 0.2° grid spanning −81 to 81° in both coordinates. To choose local paths without editing the script, load its functions without invoking its `__main__` block:

```bash
export GTE_CONFIG=/path/to/gte_dardar.yaml
export DARDAR_CLOUD_ROOT=/data/DARDAR-CLOUD/V3_1
export DARDAR_BULK_DIR=/data/dardar/bulk_flat/2007
python - "$GTE_CONFIG" <<'PY'
import os
import runpy
import sys
from pathlib import Path

config_path = sys.argv[1]
sys.argv = ['dardar_bulk', '-cf', config_path]
ns = runpy.run_path('msgi/obs/dardar/3d_dardar_tracking.py', run_name='dardar_bulk')
ns['process_tracking_selected_dardar_bulk'](
    tracking_parquet=ns['TRACKING_PARQUET'],
    dardar_root=Path(os.environ['DARDAR_CLOUD_ROOT']),
    output_dir=Path(os.environ['DARDAR_BULK_DIR']),
    n_workers=4,
)
PY
```

This uses `Final_results/YYYY_glac_04.parquet` under the configured postprocessing root. For another threshold or input table, pass its explicit path as `tracking_parquet`. The grid bounds/spacing remain the module constants; the all-days CLI offers direct control over them. The selected workflow clears its own `_tracking_selected_tmp` directory and replaces same-named final files, so reserve its output directory for this run.

The selection currently checks for available validation heights/phase, rather than enforcing the declared 1,000 m height-deviation threshold. The helper has an invalid-height branch referencing an undefined `N`; use validated rows with array-valued histories and inspect malformed rows before invoking it. Do not assume a height-agreement filter was applied.

## 4. Understand and inspect the bulk product

| Fields | Meaning |
| --- | --- |
| `cb_top_iwc_mean`, `cb_top_height_mean`, `cb_top_temp_mean`, `cb_top_cph_mean` | Cloud-top properties at each spatial/time bin |
| `cb_iwc_mean`, `cb_height_mean`, `cb_temp_mean` | Vertical profiles; IWC units follow source metadata, height in km, temperature in K |
| `cb_Z_mean`, `cb_extinction_mean`, `cb_cph_mean` | Reflectivity, extinction, and phase-score profiles |
| Corresponding `*_std`, `*_n_obs`, `*_n_valid_levels` | Variability and sampling information where provided |
| `bulk_level_offset_top`, `bulk_level_offset_bottom` | Layer bounds in meters relative to cloud top |

Bulk fields use dimensions `(time_bin, bulk_level, lat_bin, lon_bin)`; top fields omit `bulk_level`. Bulk timestamps are derived from rounded profile times. Files use NetCDF compression.

```python
import xarray as xr
ds = xr.open_dataset('/data/dardar/bulk_flat/2007/DD_CB_20070101_1200.nc')
print(ds.sizes)
print(ds[['cb_iwc_mean', 'cb_Z_mean', 'bulk_level_offset_top']])
```

Verify the source `Z` units before converting: the binner does not automatically transform dBZ to linear reflectivity or vice versa. The statistics script defaults to `CONVERT_LINEAR_Z_TO_DBZE = False`. If explicitly enabled, it applies `10*log10(Z)` to positive per-pixel values before spatial averaging. This differs from taking the logarithm of a linear spatial mean.

## 5. Collocate bulk profiles and calculate event-relative statistics

[`bulk_statistics.py`](../msgi/obs/dardar/bulk_statistics.py) uses GTE `ChunkLoader` to load annual glaciations, samples `DD_CB` bins at `val_intersec_lat`/`val_intersec_lon`, and averages across collocated pixels. It aligns samples to `Glac_peak_ind`; `rel_time_min = 15 × (sample_index − peak_index)`, so zero is the peak cloud-top ice fraction.

It reads GTE configuration at import time and defaults to `/wolke_scratch/dnikolo/DARDAR_data/CB_02/YYYY`. For that layout the direct command is:

```bash
python msgi/obs/dardar/bulk_statistics.py -cf "$GTE_CONFIG"
```

For the local flat output directory above, this wrapper substitutes only that path assignment **in memory**, without changing the library file:

```bash
python - "$GTE_CONFIG" <<'PY'
import os
import sys
from pathlib import Path

config_path = sys.argv[1]
sys.argv = ['bulk_statistics', '-cf', config_path]
script = Path('msgi/obs/dardar/bulk_statistics.py')
source = script.read_text()
original = 'CB_data_path = f"/wolke_scratch/dnikolo/DARDAR_data/CB_02/{ANALYSIS_YEAR}"'
assert source.count(original) == 1, 'Check the path assignment for this MSGI revision'
source = source.replace(original, 'CB_data_path = os.environ["DARDAR_BULK_DIR"]', 1)
exec(compile(source, str(script), 'exec'), {'__name__': '__main__', 'os': os})
PY
```

Set `DARDAR_BULK_DIR` to the flat directory, including the year, and retain the GTE config from step 2. The script expects at least one valid coincident cloud and an existing bulk file for the first selected sample. Later missing files produce NaN profiles and log messages. Confirm file coverage before running; missing observations must remain missing in the analysis.

Results are written under `postprocessing_output_dir/Bulk_statistics/`:

```text
glaciations_with_bulk_data_V3_1_2007.parquet
iwc_rel_df_V3_1_2007.parquet
mean_iwc_by_height_data_V3_1_2007.parquet
```

The first table retains cloud properties with collocated profiles. The second has one record per collocation with event-relative time and original cloud information. The third groups profiles by relative timestep, storing means and `n_hits`. Height histories are converted to meters for these tables. The current relative-time aggregate focuses on IWC, height, and phase; reflectivity/extinction histories are collected in the enriched cloud table for further analysis.

Use [DARDAR validation plots](../plotting/Colocation/daradar_validation_plots.ipynb), [bulk plots](../plotting/Bulk/bulk_plots.ipynb), and [publication cloud-bulk analysis](../plotting/Publication_figures/cloud_bulk_figure.ipynb), updating their paths and selections. Compare sampling counts alongside means and align temperature, region, timing, and vertical offsets with the [model results](model-pipeline.md).
