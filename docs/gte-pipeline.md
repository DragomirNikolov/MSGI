# Installation and GTE integration

[Back to MSGI](../README.md) · [Model guide](model-pipeline.md) · [DARDAR guide](dardar-pipeline.md)

## Environment

Use a Python ≥3.10 environment with GTE and, for tracking, a compatible PyFLEXTRKR installation. GTE is a separate project; obtain its source and installation instructions from its repository. The integration below follows the `runscripts/` and `src/glaciation_time_estimator/` interfaces in the accompanying GTE checkout; check these interfaces when using another revision.

```bash
# Run from the MSGI repository root, in your activated environment.
export MSGI_DIR="$PWD"
export GTE_DIR=/path/to/Glaciation_time_estimator
python -m pip install -e "$GTE_DIR"
python -m pip install -e "$MSGI_DIR"
python -m pip install PyYAML netCDF4 cftime 'dask[distributed]' tqdm h5py
```

MSGI declares NumPy, SciPy, xarray, pandas, PyArrow, Dask, Matplotlib, and Cartopy in `pyproject.toml`. The extra packages above cover imports used by the processing scripts that are not all declared there. Install PyFLEXTRKR following its own environment instructions. Install the **CDO executable** for remapping and DARDAR duplicate merging, and **rsync** for GTE tracking file transfers; these are separate system dependencies.

```bash
python -c 'import msgi, glaciation_time_estimator, netCDF4, yaml'
cdo -V
rsync --version
```

Input datasets, CDO weights/grid descriptions, satellite auxiliary grids, and model cell-index maps are external assets. Installation does not download or generate them.

## Two configurations with different jobs

| Configuration | CLI argument | Owns |
| --- | --- | --- |
| GTE YAML | `-cf` / `--configuration_filepath` | Analysis dates, hemispheres, aggregation tag, tracking/postprocessing paths, auxiliary geometry, glaciation detection, validation |
| MSGI YAML | `-mcf` / `--msgi_configuration_filepath` | Model, raw/remapped/preprocessed locations, CDO grids/weights, exported variables, temperature mode, ICON bulk settings |
| PyFLEXTRKR base YAML | `-bc` when tracking | Feature variable (`cph_filtered`), spatial resolution, object size/lifetime criteria, tracking switches |

Use `-cf` explicitly. Some shell wrappers use `-c` as an abbreviation; that is not the MSGI config flag.

Start from the **installed GTE checkout's complete config**, preserving its required keys. The [MSGI DARDAR example](../msgi/obs/dardar/test_gte_config.yaml) shows the general structure but contains old paths and the obsolete `DARDAR_CPH_fp` key. The current GTE reader requires `val_CPH_fp` for DARDAR/MODIS and rejects missing or redundant keys. Do not add MSGI settings to GTE YAML.

Configure these fields before any processing:

| GTE field | Model run | CLAAS + DARDAR run |
| --- | --- | --- |
| `start_time`, `end_time` | Strings `YYYYMMDD_HHMM`; cover available model timesteps | Same format; cover CLAAS and DARDAR overlap |
| `CLAAS_fp` | Model preprocessing root, equal to MSGI `preprocessing_output_dir` | Original CLAAS root, with `np/YYYY/MM/DD/CPP*.nc` and `CTX*.nc` (and `sp` if used) |
| `Resample` | `false` for geostationary outputs; `true` for regular-grid `R_` outputs | Normally `false` |
| `agg_fact` | Must match output tag and target-grid geometry | Normally `3` |
| `t_deltas` | `[6]` for MSGI mixed-phase bins | Usually `[6]` |
| `pole_split`, `pole_folders` | `true`, `[np, sp]` for geostationary; regular-grid MSGI output is under `np` only | Set to actual CLAAS coverage |
| `aux_fps`, `aux_fps_agg`, `grid_fps` | Geometry, pixel areas and grids consistent with the selected target grid | CLAAS original/aggregated geometry and grids |
| `job_output_fp` | Dedicated tracking-output directory | Dedicated tracking-output directory |
| `postprocessing_output_dir` | Dedicated model-results directory | Dedicated validated-results directory |
| `collect_additional_properties` | `true` for model cell histories/bulk enrichment | `true` for collocation properties |
| `validation_mode` | `model` for model-aware postprocessing | `DARDAR` (reader normalizes case) |
| `val_CPH_fp` | Omit for model mode with the current reader | Root containing `YYYY/MM/DD/DD_CT_YYYYMMDD_HHMM.nc` |
| `Analyze_year` | `false` for an initial period; `true` to combine completed yearly jobs | Same; yearly results are expected by the bulk-statistics scripts |
| `glac_threshold` | `0.4` produces suffix `04` for annual glaciation tables | Keep `0.4` for the default bulk consumers |

Also retain the reader's preprocessing/postprocessing worker settings, `struct_boundary_date`, `time_folder_format`, `Global_sqrt_mse`, CSV/fast-mode settings, and annual splitting keys when enabled. Use plain local paths when running locally; example remote prefixes are site-specific.

Validate configurations without processing data:

```bash
python - "$GTE_CONFIG" <<'PY'
import sys
from glaciation_time_estimator.auxiliary_func.config_reader import read_config
config = read_config(sys.argv[1])
print(config['start_time'], config['end_time'], config['min_temp_arr'])
PY
# For a model configuration:
python -m msgi.aux.config_reader -mcf "$MSGI_CONFIG"
```

Set `GTE_CONFIG` and (for models) `MSGI_CONFIG` to your actual YAML paths first. MSGI model preprocessing truncates the GTE start/end timestamps to midnight and selects **whole input files by filename time**, inclusively. It does not trim their internal timesteps. A file dated later than midnight on the end date can therefore be skipped; choose the file coverage accordingly.

## Run tracking and postprocessing

First generate `Filtered_Data` and `Resampled_Data`: use [MSGI preprocessing](model-pipeline.md) for models, or GTE's `runscripts/Preprocessing_pipeline.py -cf "$GTE_CONFIG"` for CLAAS. DARDAR `DD_CT` files are validation inputs and do not replace CLAAS preprocessing.

Set a PyFLEXTRKR base YAML compatible with GTE's generic tracking driver and `cph_filtered`. Its grid spacing/pixel radius must match the target grid, taking into account that GTE multiplies the base `pixel_radius` by `agg_fact`. GTE uses **`TMPDIR`**, despite older documentation mentioning `TMP_DIR`.

The following runs the six mixed-phase bins for each selected hemisphere. Use a fresh scratch directory for every invocation: the driver stages inputs under `Data/` and writes to `Result/` there.

```bash
export PYFLEX_BASE=/path/to/generic_tracking_base.yml
export RUN_SCRATCH=/path/to/writable/scratch
mkdir -p "$RUN_SCRATCH"
for pole in np sp; do  # Use only np for MSGI regular-grid output.
  for cold in 6 12 18 24 30 36; do
    warm=$((cold - 6))
    export TMPDIR
    TMPDIR=$(mktemp -d "$RUN_SCRATCH/msgi_${pole}_${cold}.XXXXXX")
    python "$GTE_DIR/runscripts/Tracking_pipeline.py" \
      -cf "$GTE_CONFIG" -bc "$PYFLEX_BASE" \
      -p "$pole" -t "-$cold" "-$warm" -rd "$TMPDIR/Result/"
  done
done
```

The `-rd` path is used by GTE's final result-transfer step. Confirm tracking outputs reached `job_output_fp` before removing scratch directories. Then run:

```bash
python "$GTE_DIR/runscripts/Postprocessing_pipeline.py" -cf "$GTE_CONFIG"
python "$GTE_DIR/runscripts/Glaciation_detection.py" -cf "$GTE_CONFIG"
```

For models, complete the auxiliary-variable and cell-map compatibility steps in the [model guide](model-pipeline.md#gte-handoff) **before postprocessing**. The inspected GTE postprocessor opens `MODEL_NCELL_LOC` maps even outside model mode; ensure those site-specific paths are valid or supply local maps in your GTE integration.

For `Analyze_year: false`, glaciation detection writes `Results/[R_]Agg_XX_glaciations_<period>.parquet`. For an annual run, first complete tracking/postprocessing for all period configs in `yearly_config_folder`; then run detection with `Analyze_year: true`, the matching `n_month_parts`, and the analysis year as `start_time.year`. It combines those periods into:

```text
postprocessing_output_dir/
  Final_results/
    YYYY_all.parquet
    YYYY_glac_04.parquet       # glac_threshold = 0.4
```

A successful short-period run does not automatically create these annual files. DARDAR bulk selection/statistics expect annual results and validation-history columns. The model bulk extractor can accept a period table directly if it has its required columns.

## Inspect results

```python
import pandas as pd
clouds = pd.read_parquet('/path/to/results/Final_results/2007_glac_04.parquet')
print(clouds.shape)
print(clouds.columns.tolist())
```

Check that histories cover the intended dates and that the validation or model-cell columns needed downstream are present. An empty glaciation selection can raise an error during GTE detection; start with a representative period before launching a full year.
