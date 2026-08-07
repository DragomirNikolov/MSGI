from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


MODULE_PATH = Path(__file__).with_name("model_bulk_level_v2_xr.py")
SPEC = importlib.util.spec_from_file_location("model_bulk_level_v2_xr", MODULE_PATH)
bulk = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bulk
SPEC.loader.exec_module(bulk)


def make_dataset(
    zg_profile: list[float],
    ze_profile: list[float],
    cli_profile: list[float],
    clw_profile: list[float] | None = None,
    *,
    ctlev: float = 1.0,
    ctmask: float = 1.0,
) -> tuple[xr.Dataset, xr.Dataset]:
    zg_values = np.asarray(zg_profile, dtype=np.float32)
    ze_values = np.asarray(ze_profile, dtype=np.float32)
    cli_values = np.asarray(cli_profile, dtype=np.float32)
    if clw_profile is None:
        clw_values = np.zeros_like(cli_values)
    else:
        clw_values = np.asarray(clw_profile, dtype=np.float32)

    time = np.array([0], dtype=np.int32)
    height = np.arange(zg_values.size, dtype=np.int32)
    ncells = np.array([0], dtype=np.int32)

    zg_2d = zg_values[:, None]
    atm = xr.Dataset(
        data_vars={
            "zg": (("height", "ncells"), zg_2d),
            "cloudsat_ze_mean": (("time", "height", "ncells"), ze_values[None, :, None]),
            "cli": (("time", "height", "ncells"), cli_values[None, :, None]),
            "clw": (("time", "height", "ncells"), clw_values[None, :, None]),
        },
        coords={"time": time, "height": height, "ncells": ncells},
    )
    ct = xr.Dataset(
        data_vars={
            "ctlev": (("time", "ncells"), np.array([[ctlev]], dtype=np.float32)),
            "ctmask": (("time", "ncells"), np.array([[ctmask]], dtype=np.float32)),
        },
        coords={"time": time, "ncells": ncells},
    )
    return atm, ct


def scalar(ds: xr.Dataset, name: str, bulk_level: int = 0) -> float:
    return float(ds[name].isel(time=0, bulk_level=bulk_level, ncells=0).compute())


def build(atm: xr.Dataset, ct: xr.Dataset, *, max_depth_m: float = 300.0, dz_m: float = 100.0) -> xr.Dataset:
    return bulk.build_output_dataset(
        atm,
        ct,
        bulk.VariableNames(),
        max_depth_m=max_depth_m,
        dz_m=dz_m,
        condensate_threshold=1.0e-5,
    )


def test_linear_interpolation_between_two_cloudy_levels() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm, ct)

    assert scalar(out, "ze") == pytest.approx(15.0)
    assert scalar(out, "cli") == pytest.approx(3.0e-5)
    assert int(out["ze_n_valid_levels"].isel(time=0, bulk_level=0, ncells=0)) == 2


def test_linear_reduction_to_noncloudy_anchors() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 0.0, 6.0e-5],
    )

    out = build(atm, ct)

    assert scalar(out, "ze") == pytest.approx(-10.0)
    assert scalar(out, "cli") == pytest.approx(1.0e-5)


def test_nan_when_nearest_model_level_is_noncloudy() -> None:
    atm, ct = make_dataset(
        [1000.0, 960.0, 900.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 0.0, 6.0e-5],
    )

    out = build(atm, ct)

    assert np.isnan(scalar(out, "ze"))
    assert np.isnan(scalar(out, "cli"))
    assert int(out["ze_n_valid_levels"].isel(time=0, bulk_level=0, ncells=0)) == 0


def test_equal_distance_to_cloudy_and_noncloudy_is_retained() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 0.0, 6.0e-5],
    )

    out = build(atm, ct)

    assert scalar(out, "ze") == pytest.approx(-10.0)
    assert scalar(out, "cli") == pytest.approx(1.0e-5)


def test_exact_model_level_matches_are_kept() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm, ct, max_depth_m=200.0, dz_m=200.0)

    assert scalar(out, "ze") == pytest.approx(20.0)
    assert scalar(out, "cli") == pytest.approx(4.0e-5)
    assert int(out["ze_n_valid_levels"].isel(time=0, bulk_level=0, ncells=0)) == 1


def test_invalid_cloud_top_returns_nan() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
        ctmask=0.0,
    )

    out = build(atm, ct)

    assert np.isnan(scalar(out, "ze"))
    assert np.isnan(scalar(out, "cli"))
    assert int(out["cloud_top_level_index"].isel(time=0, ncells=0)) == -1


def test_missing_source_values_remove_cloudy_anchor() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, np.nan, np.nan],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm, ct, max_depth_m=200.0, dz_m=200.0)

    assert scalar(out, "cli") == pytest.approx(4.0e-5)
    assert np.isnan(scalar(out, "ze"))


def test_no_extrapolation() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [np.nan, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm, ct)

    assert np.isnan(scalar(out, "ze"))


def test_ascending_vertical_index_order() -> None:
    atm, ct = make_dataset(
        [800.0, 900.0, 1000.0],
        [30.0, 20.0, 10.0],
        [6.0e-5, 4.0e-5, 2.0e-5],
        ctlev=3.0,
    )

    out = build(atm, ct)

    assert scalar(out, "ze") == pytest.approx(15.0)
    assert out.attrs["zg_decreases_with_height_index"] == "False"


def test_descending_vertical_index_order() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm, ct)

    assert scalar(out, "ze") == pytest.approx(15.0)
    assert out.attrs["zg_decreases_with_height_index"] == "True"


def test_chunked_ctlev_indexing_is_dask_safe() -> None:
    atm, ct = make_dataset(
        [1000.0, 900.0, 800.0],
        [10.0, 20.0, 30.0],
        [2.0e-5, 4.0e-5, 6.0e-5],
    )

    out = build(atm.chunk({"height": 2}), ct.chunk())

    assert type(out["ze"].data).__name__ == "Array"
    assert scalar(out, "ze") == pytest.approx(15.0)
