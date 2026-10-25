from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from msgi.aux.config_reader import read_config
from msgi.model.model_preprocessing import split_dir_files


def _config():
    return {
        "model": "ICON",
        "grid_type": "latlon",
        "resample_input_dir": "input",
        "twc_threshold": 1.0e-5,
        "additional_vars": ["ctt"],
    }


@pytest.mark.parametrize("mode", [None, "mixed_phase", "cirrus"])
def test_read_config_defaults_and_validates_mode(tmp_path, mode):
    config = _config()
    if mode is not None:
        config["mode"] = mode
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert read_config(str(path))["mode"] == (mode or "mixed_phase")


@pytest.mark.parametrize("mode", ["mixed phase", "invalid", "", None, 42, [], {}])
def test_read_config_rejects_invalid_mode(tmp_path, mode):
    config = {**_config(), "mode": mode}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="'mode'.*mixed_phase.*cirrus"):
        read_config(str(path))


def test_example_config_is_valid():
    path = Path(__file__).resolve().parents[1] / "msgi/aux/msgi_config_example.yaml"
    assert read_config(str(path))["mode"] == "mixed_phase"


@pytest.mark.parametrize("mode", [None, "mixed_phase", "cirrus"])
@pytest.mark.parametrize("model", ["ICON", "CESM"])
@pytest.mark.parametrize("grid_type", ["latlon", "geostationary"])
def test_preprocessing_temperature_masks_and_filenames(
    tmp_path, monkeypatch, mode, model, grid_type
):
    # Include every boundary, nearby temperatures, out-of-range values, and NaN.
    if mode == "cirrus":
        edges = np.array([175, 185, 195, 205, 215, 225, 235])
    else:
        edges = np.array([237, 243, 249, 255, 261, 267, 273])
    temperatures = np.concatenate([edges - 0.1, edges, edges + 0.1, [np.nan]])
    values = temperatures.reshape(1, 1, -1)
    phase = np.full_like(values, 2 if model == "ICON" else 0.5)
    twc = np.full_like(values, 2.0e-5)
    # One in-range point fails TWC only in the combined output.
    twc[0, 0, 1] = 0
    temperature_name = "ctt" if model == "ICON" else "ACTT"
    phase_name = "cph" if model == "ICON" else "ACT_ITMF"
    pressure_name = "ctp" if model == "ICON" else "ACTP"
    dims = ("time", "lat", "lon")
    data = xr.Dataset(
        {
            temperature_name: (dims, values),
            phase_name: (dims, phase),
            pressure_name: (dims, np.full_like(values, 300)),
            "twc": (dims, twc),
        },
        coords={
            "time": [np.datetime64("2008-09-05T00:00:00", "ns")],
            "lat": [45.0],
            "lon": np.linspace(-70, 70, len(temperatures)),
        },
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    date_suffix = "20080905T000000Z.nc" if model == "ICON" else "2008-09-05-00000.nc"
    region = "whole" if grid_type == "latlon" else "np"
    (input_dir / f"remap_laf_{region}_experiment_{date_suffix}").touch()
    config = {
        **_config(),
        "model": model,
        "grid_type": grid_type,
        "resample_output_dir": str(input_dir),
        "preprocessing_output_dir": str(tmp_path / "output"),
        "experiment_name": "experiment",
        "poles": ["np"],
        "additional_vars": [temperature_name, pressure_name],
    }
    if mode is not None:
        config["mode"] = mode
    gte_config = {
        "agg_fact": 3,
        "start_time": datetime(2008, 9, 5),
        "end_time": datetime(2008, 9, 5),
    }
    saved = {}
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: data)

    def capture_save(datasets, paths):
        for dataset, path in zip(datasets, paths):
            saved[Path(path)] = dataset

    monkeypatch.setattr(xr, "save_mfdataset", capture_save)
    split_dir_files(gte_config, config)

    assert len(saved) == 7
    prefix = "R_" if grid_type == "latlon" else ""
    date_dir = Path("np/2008/09/05")
    for lower, upper in reversed(list(zip(edges[:-1], edges[1:]))):
        filename = f"{prefix}Agg_03_T_{273 - lower:02}_{273 - upper:02}_20080905000000.nc"
        path = tmp_path / "output/Filtered_Data" / date_dir / filename
        expected = np.where((values >= lower) & (values < upper), phase if model == "ICON" else phase + 1, 0)
        np.testing.assert_array_equal(saved[path]["cph_filtered"].values, expected)

    path = tmp_path / "output/Resampled_Data" / date_dir / f"{prefix}Agg_03_20080905000000.nc"
    combined_mask = (values >= edges[0]) & (values < edges[-1]) & (twc >= config["twc_threshold"])
    np.testing.assert_array_equal(saved[path]["ctt_resampled"].values, np.where(combined_mask, values, 0))
    np.testing.assert_array_equal(saved[path]["ctp_resampled"].values, np.where(combined_mask, 300, 0))
