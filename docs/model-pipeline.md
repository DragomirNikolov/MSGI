# Model processing pipeline

[Back to MSGI](../README.md) · [Installation and GTE](gte-pipeline.md) · [DARDAR comparison](dardar-pipeline.md)

The cloud-top pipeline supports `model: ICON` and `model: CESM`. The bulk-generation and track-enrichment pipeline currently targets ICON. Run commands from the MSGI repository root after installation.

## 1. Prepare input diagnostics and CDO assets

MSGI expects cloud-top diagnostics to exist already; `model_preprocessing.py` does not derive them from arbitrary 3D model output.

| Model | Input discovery and filename date | Required cloud-top fields |
| --- | --- | --- |
| ICON | `*_ctp_2d*`, normally `<experiment>_ctp_2d_ml_YYYYMMDDTHHMMSSZ.nc` | `cph`, `ctt` (K), and each requested `additional_vars` field |
| CESM | `*cam.h1*`, ending `YYYY-MM-DD-SSSSS.nc` (seconds since midnight) | `ACT_ITMF`, `ACTT` (K), and requested `ACTP`/`ACTT` |

`experiment_name` must match the start of input filenames. ICON phase values are retained; CESM `ACT_ITMF` is shifted by `+1` for the GTE phase-score convention. Check the meaning of the source diagnostic before comparing models. Files need `time`, `lat`, and `lon` compatible with the selected target grid.

Provide CDO target-grid descriptions and precomputed weights for the **actual source grid**, target grid, and method. `resample_dir_files` calls CDO `remap` with these files; it does not generate weights. `laf` means largest-area fraction and `dis` distance weighting. If generating weights separately, use CDO's `genlaf`/`gendis` with a representative input carrying the correct source-grid metadata. For unstructured ICON data, that metadata must identify the original mesh.

## 2. Configure MSGI

A regular-grid ICON example, saved as your own `model.yaml`:

```yaml
model: ICON
mode: mixed_phase
experiment_name: my_icon_experiment
resample_input_dir: /data/icon/raw_data
resample_output_dir: /data/icon/remapped
preprocessing_output_dir: /data/icon/gte_input

grid_type: latlon
cdo_aux_dir: /data/icon/aux
cdo_grid_file: /data/icon/aux/grid_1.5x1.5.txt
laf_weights_file: /data/icon/aux/laf_1.5x1.5.nc
resampling_methods: [laf]
cdo_threads: 4
resample_workers: 2

twc_threshold: 1.0e-5
additional_vars: [ctt, ctp, cot]
```

The config reader requires `model`, `resample_input_dir`, `twc_threshold`, a nonempty `additional_vars` list, and `grid_type`. A complete cloud-top run also needs the remapping/output/experiment settings above. `preprocessing_output_dir` defaults to GTE `CLAAS_fp` if omitted. Use `additional_vars`, replacing the old `mixed_vars` key in historical configurations. Choose only variables that actually exist in the input.

For CLAAS-compatible geostationary output, replace the grid/weight settings with pole mappings:

```yaml
grid_type: geostationary
poles: [np, sp]
cdo_grid_file:
  np: /data/aux/grid_CLAAS_Agg_3_np.txt
  sp: /data/aux/grid_CLAAS_Agg_3_sp.txt
laf_weights_file:
  np: /data/aux/ICON_to_CLAAS_laf_np.nc
  sp: /data/aux/ICON_to_CLAAS_laf_sp.nc
```

The geostationary implementation also applies `setgrid` using **fixed paths** under `/nfs/ch4/wolke_scratch/dnikolo/Glaciation_time_estimator/grids/grid_{np,sp}_agg.txt`. Those descriptions must be accessible and match the remapped geometry; `cdo_grid_file` does not override them. For other installations, use the regular-grid route or adapt the remapping stage externally. You can then call `split_dir_files` directly on correctly named/georeferenced `remap_laf_<pole>_<original-name>.nc` files.

For CESM, set `model: CESM`, the CAM experiment prefix, and:

```yaml
additional_vars: [ACTT, ACTP]
```

Use weights for the CESM source grid. The CESM geostationary CDO command selects only `ACT_ITMF,ACTT,ACTP`; the output rename map supports `ACTT → ctt_resampled` and `ACTP → ctp_resampled`. Arbitrary CESM extra variables are not currently supported by this path. [GTE handoff](#gte-handoff) explains the additional optical-thickness requirement.

## 3. Remap and split into temperature bins

```bash
export GTE_CONFIG=/path/to/gte_model.yaml
export MSGI_CONFIG=/path/to/model.yaml
python -m msgi.model.model_preprocessing \
  -cf "$GTE_CONFIG" -mcf "$MSGI_CONFIG"
```

This calls `resample_dir_files` followed by `split_dir_files`. Both are also available as Python functions:

```python
from glaciation_time_estimator.auxiliary_func.config_reader import read_config as read_gte
from msgi.aux import read_config as read_msgi
from msgi.model.model_preprocessing import resample_dir_files, split_dir_files

gte = read_gte('/path/to/gte_model.yaml')
msgi = read_msgi('/path/to/model.yaml')
resample_dir_files(gte, msgi)  # Omit if remap_laf_* files already exist.
split_dir_files(gte, msgi)
```

Remapping can produce both `laf` and `dis` outputs, but **splitting only reads `remap_laf_*`**. Include `laf` for a full pipeline. The implementation processes matching files again on reruns. Budget CPU for `resample_workers × cdo_threads`; each remapping worker launches CDO.

| `mode` | Bin bounds in K, warmest first | Filename labels |
| --- | --- | --- |
| `mixed_phase` (default) | [267,273), [261,267), …, [237,243) | `06_00`, `12_06`, …, `36_30` |
| `cirrus` | [225,235), [215,225), …, [175,185) | `48_38`, `58_48`, …, `98_88` |

The code uses **273 K**, rather than 273.15 K, for temperature labels. Lower bounds are inclusive and upper bounds exclusive. Outside a bin, `cph_filtered` is zero. Combined-range additional variables are also zero outside the selected temperature range. If `twc` exists, the combined mask additionally requires `twc >= twc_threshold`; this threshold is not applied to the individual phase-bin images. If `twc` is absent, there is no TWC filtering.

Regular-grid splitting retains 30–80° latitude in both hemispheres and stores the result under `np`. Its current longitude condition is `(lon >= -80) | (lon <= 80)`, which does not restrict finite longitudes. The `time_filter` winter-month helper is not called. Numeric ICON times (`YYYYMMDD.fraction`) are rounded to minutes; CESM cftime fields are converted to Gregorian datetime components, so check calendar compatibility.

For `agg_fact: 3`, outputs are:

```text
preprocessing_output_dir/
  Filtered_Data/np/YYYY/MM/DD/
    Agg_03_T_06_00_YYYYMMDDHHMMSS.nc       # geostationary
    R_Agg_03_T_06_00_YYYYMMDDHHMMSS.nc     # latlon
  Resampled_Data/np/YYYY/MM/DD/
    Agg_03_YYYYMMDDHHMMSS.nc              # geostationary
    R_Agg_03_YYYYMMDDHHMMSS.nc            # latlon
```

Geostationary runs also write `sp` when requested. Filtered files contain `cph_filtered`; combined files contain `<variable>_resampled` (with the CESM renaming above). The tag records GTE `agg_fact`; MSGI does not perform an additional `agg_fact × agg_fact` aggregation. Use **separate preprocessing roots for mixed-phase and cirrus**: combined filenames are shared and would overwrite each other.

## GTE handoff

Set GTE `CLAAS_fp` to the preprocessing root, match `Resample` to the filename prefix, and follow [tracking and detection](gte-pipeline.md#run-tracking-and-postprocessing).

For geostationary model postprocessing, use `validation_mode: model`, `collect_additional_properties: true`, and target-grid auxiliary latitude/longitude/pixel-area arrays consistent with the remapped data. GTE's `MODEL_NCELL_LOC` mapping currently points to fixed ICON R2B4 files. It must map each target pixel back to the correct zero-based original model cell; R2B5 and CESM require their own maps. Supply these in your local GTE integration before extracting `model_grid_ncell`. Regular-grid tracking uses the `R_` path, but the inspected GTE postprocessor's cell-history extraction is implemented in its non-resampled model branch; verify or adapt that branch before expecting bulk enrichment from regular-grid tracks.

The inspected GTE `extract_resampled_vars` reads all three fields `ctp_resampled`, `ctt_resampled`, and `cot_resampled`. For ICON, include `ctp`, `ctt`, and `cot` when available. If your diagnostic uses `modis_optical_thickness_total`, it is exported as `modis_optical_thickness_total_resampled`; provide a `cot_resampled` alias in a derived copy of the combined files or adapt the GTE reader. CESM preprocessing exports only CTT/CTP, so its postprocessor needs an explicit COT policy as well. The inspected GTE code replaces COT with ones after loading it; do not report its resulting COT statistics as model optical thickness.

For a local geostationary run, this explicit postprocessing adapter supplies the cell maps and handles the COT reader contract without changing installed source files. Replace the map paths with maps for your actual source mesh. It uses unit COT because the inspected GTE model branch already substitutes ones; it is suitable for phase/temperature/bulk analysis, not optical-thickness analysis.

```bash
python - "$GTE_CONFIG" <<'PYCODE'
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from glaciation_time_estimator.auxiliary_func.config_reader import read_config
from glaciation_time_estimator.data_postprocessing import Tracking_result_analysis as analysis

config = read_config(sys.argv[1])
analysis.MODEL_NCELL_LOC = {
    'np': '/data/aux/model_grid_ncell_np.nc',
    'sp': '/data/aux/model_grid_ncell_sp.nc',
}

def read_model_vars(time, pole, config):
    tag = f"Agg_{config['agg_fact']:02}"
    filename = time.strftime(f'%Y/%m/%d/{tag}_%Y%m%d%H%M%S.nc')
    path = Path(config['CLAAS_fp']) / 'Resampled_Data' / pole / filename
    with xr.open_dataset(path) as ds:
        ctp = ds['ctp_resampled'].values
        ctt = ds['ctt_resampled'].values
    return ctp, ctt, None, np.ones_like(ctp)

analysis.extract_resampled_vars = read_model_vars
analysis.analyze_tracked_clouds(config)
PYCODE
```

This replaces the standard postprocessing command for this run; execute GTE glaciation detection afterward. For CLAAS/DARDAR runs that only need local cell-map paths, use the same `MODEL_NCELL_LOC` override and call `analyze_tracked_clouds`, retaining the original variable reader. Runtime overrides must be visible in worker processes; this recipe targets the Linux/fork environment used by these workflows.

For phase-only tracking, the filtered images suffice. A complete run with additional properties needs these compatibility steps; the example YAML alone does not supply missing diagnostics or cell maps.

**Cirrus:** MSGI preprocessing supports the six cirrus bins, but the inspected GTE `generate_temp_range` still generates mixed-phase bins from `t_deltas`. Setting `t_deltas: [10]` does not select cirrus. Configure/adapt the GTE integration to use bounds `(-48,-38), …, (-98,-88)` consistently for tracking, file discovery, and postprocessing. Keep cirrus outputs and result directories separate. Do not apply the mixed-phase tracking loop unchanged.

## 4. Generate ICON cloud-bulk files

The bulk generator uses the **original mesh**, independently of cloud-top remapping. Supply matching files in `resample_input_dir`:

```text
<experiment>_atm_3d_cloud_ml_YYYYMMDDTHHMMSSZ.nc
<experiment>_ctp_2d_ml_YYYYMMDDTHHMMSSZ.nc
```

Pairs are matched by filename timestamp. Required 3D fields are `zg`, `cloudsat_ze_mean`, `cli`, and `clw`; the 2D file supplies `ctmask` and normally `ctlev`. Condensate/reflectivity dimensions are `(time, height, ncells)`; `zg` supports `(height, ncells)` or `(time, height, ncells)`. Cloud-top level is a **one-based positional index**, independent of height-coordinate labels.

```bash
python -m msgi.model.model_cloud_bulk_v2 \
  -cf "$GTE_CONFIG" -mcf "$MSGI_CONFIG"
```

Config-driven output goes to `resample_input_dir/../bulk_data/`, replacing `_atm_3d_cloud_ml_` with `_bulk_cloud_ml_` in filenames. It uses the same midnight-normalized date selection. Optional MSGI keys:

| Keys | Defaults / meaning |
| --- | --- |
| `bulk_max_depth_m`, `bulk_dz_m` | `3000`, `120`: 25 layers below cloud top |
| `bulk_condensate_threshold`, `bulk_ctmask_threshold` | Both `1.0e-5` |
| `bulk_derive_cloud_top` | `false`; `true` derives the highest cloudy condensate level, still requiring `ctmask` |
| `bulk_zg_name`, `bulk_ze_name`, `bulk_cli_name`, `bulk_clw_name`, `bulk_ctlev_name`, `bulk_ctmask_name` | Override source variable names listed above |

A single pair can be processed without the two YAML files:

```bash
python -m msgi.model.model_cloud_bulk_v2 \
  --atm-3d /data/icon/raw_data/my_experiment_atm_3d_cloud_ml_20070101T000000Z.nc \
  --cloud-top /data/icon/raw_data/my_experiment_ctp_2d_ml_20070101T000000Z.nc \
  --output /data/icon/bulk_data/my_experiment_bulk_cloud_ml_20070101T000000Z.nc \
  --max-depth-m 3000 --dz-m 120
```

Use `--derive-cloud-top` to derive the top; `--help` lists the variable-name/threshold overrides. Outputs include `ze`, `cph`, valid-source-level counts, cloud-top index/height, and negative layer bounds/midpoints in meters. `cph` is based on `1 + cli/(cli+clw)`. Reflectivity retains the source metadata/scale; boundary interpolation uses a `-30` reflectivity target and a `0` phase target where needed. Inspect counts and masks before interpreting profiles.

## 5. Attach bulk reflectivity to model tracks

[`bulk_data_extractor.py`](../msgi/model/bulk_data_extractor.py) reads each daily file once and adds `bulk_values`, shaped `(track timesteps, 25 layers)`, containing a NaN-aware mean of `ze` over the tracked original cells. It currently extracts reflectivity only.

Requirements are strict: 96 quarter-hour slots per daily file; numeric ICON `YYYYMMDD.fraction` time; exactly 25 bulk levels; consecutive zero-based `ncells`; and track columns `tracknumber`, `track_start_time`, `track_length`, `model_grid_ncell`. Each cell-history entry is a list of cell-index arrays, one per timestep, with length `track_length / 15 min + 1`.

The module's CLI defaults and filename builder are experiment-specific. Use its Python API and override the filename builder for another experiment without editing library files:

```python
from pathlib import Path
import msgi.model.bulk_data_extractor as bulk

bulk.timestamp_to_filename = lambda timestamp, bulk_data_dir: (
    Path(bulk_data_dir) / f'my_experiment_bulk_cloud_ml_{timestamp:%Y%m%dT000000}Z.nc'
)
bulk.run_batch(
    input_path=Path('/data/icon/results/Final_results/2007_glac_04.parquet'),
    output_path=Path('/data/icon/results/Final_results/2007_glac_04_bulk.parquet'),
    bulk_data_dir=Path('/data/icon/bulk_data'),
)
```

Create the output parent directory beforehand. `save_parquet_atomically` preserves the original columns and writes nested bulk arrays through PyArrow, replacing the destination after a successful write. Files with another cadence, time encoding, mesh index convention, or layer count require adaptation before this extractor can consume them.

Inspect the enriched Parquet with pandas and use the [model bulk notebooks](../plotting/Bulk/model_bulk_weighted_means.ipynb) for subsequent averaging/figures. Align model and DARDAR selections by cloud-top temperature, location, event timing, layer offsets, and reflectivity units before comparing profiles.
