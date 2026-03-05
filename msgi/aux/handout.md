# MSGI Aux Config Reader Handout

## Overview
A new YAML configuration reader was added for MSGI under `msgi/aux`.
It mirrors the GTE reader style by returning a Python dictionary.

## Files Added
- `msgi/aux/config_reader.py`
- `msgi/aux/__init__.py`
- `msgi/aux/msgi_config_example.yaml`
- `msgi/aux/handout.md`

## Reader API
Use:

```python
from msgi.aux.config_reader import read_config

cfg = read_config("/path/to/msgi_config.yaml")
```

Or from command line behavior (when no argument is passed to `read_config()`), it expects:

- `-cf` / `--configuration_filepath`

## Required YAML Keys
The reader validates and requires:

- `model_data_dir`: string path to model data root directory
- `twc_threshold`: numeric threshold for TWC filtering
- `mixed_vars`: non-empty list of variable names to export in mixed-range output

## Integration in model_preprocessing.py
`msgi/model/model_preprocessing.py` now imports:

```python
from msgi.aux.config_reader import read_config as read_msgi_config
```

And uses config values as follows:

- `model_data_dir` sets `base_dir`
- `mixed_vars` controls variables included in mixed outputs
- `twc_threshold` is applied to mixed mask when `twc` exists in dataset

## Example Config
`msgi/aux/msgi_config_example.yaml`

```yaml
model_data_dir: /wolke_scratch/dnikolo/ICON_Data/Nudged_run_cph_03_03
twc_threshold: 0.0
mixed_vars:
  - ctt
  - ctp
```

## Notes
- The reader returns a plain dictionary after validation.
- Validation errors are explicit for missing keys or wrong types.
