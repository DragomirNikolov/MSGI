from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from msgi.model.bulk_data_extractor import (
    N_BULK_LEVELS,
    TIMESTEPS_PER_DAY,
    extract_bulk_values_by_day,
    run_batch,
    timestamp_to_filename,
)


def _write_day(directory: Path, day: str, offset: float) -> np.ndarray:
    timestamp = pd.Timestamp(day)
    values = np.arange(TIMESTEPS_PER_DAY * N_BULK_LEVELS * 5, dtype=np.float32).reshape(
        TIMESTEPS_PER_DAY, N_BULK_LEVELS, 5
    )
    values += offset
    values[0, 3, :] = np.nan
    values[1, 4, 1] = np.nan

    dataset = xr.Dataset(
        {"ze": (("time", "bulk_level", "ncells"), values)},
        coords={
            "time": int(timestamp.strftime("%Y%m%d"))
            + np.arange(TIMESTEPS_PER_DAY) / TIMESTEPS_PER_DAY,
            "bulk_level": np.arange(N_BULK_LEVELS),
            "ncells": np.arange(5),
        },
    )
    dataset.to_netcdf(timestamp_to_filename(timestamp, directory))
    return values


@pytest.fixture
def two_days(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    first = _write_day(tmp_path, "2007-01-01", 0)
    second = _write_day(tmp_path, "2007-01-02", 100_000)
    return first, second


def _clouds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tracknumber": [10.0, 11.0, np.nan],
            "track_start_time": pd.to_datetime(
                ["2007-01-01 23:45", "2007-01-02 00:00", None]
            ),
            "track_length": pd.to_timedelta(["30min", "15min", None]),
            "model_grid_ncell": [
                [np.array([0, 1]), np.array([2]), np.array([], dtype=int)],
                [np.array([1, 3]), np.array([4])],
                None,
            ],
        },
        index=[8, 3, 12],
    )


def test_extracts_shared_and_cross_midnight_tracks(
    tmp_path: Path, two_days: tuple[np.ndarray, np.ndarray]
) -> None:
    first, second = two_days
    clouds = _clouds()

    result = extract_bulk_values_by_day(clouds, tmp_path, show_progress=False)

    assert result is clouds
    assert result.index.tolist() == [8, 3, 12]
    assert result.iloc[0].bulk_values.dtype == np.float32
    np.testing.assert_allclose(
        result.iloc[0].bulk_values[0], np.nanmean(first[95, :, [0, 1]], axis=0)
    )
    np.testing.assert_allclose(
        result.iloc[0].bulk_values[1], second[0, :, 2], equal_nan=True
    )
    assert np.isnan(result.iloc[0].bulk_values[2]).all()
    np.testing.assert_allclose(
        result.iloc[1].bulk_values[0],
        np.nanmean(second[0, :, [1, 3]], axis=0),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.iloc[1].bulk_values[1], second[1, :, 4], equal_nan=True
    )
    assert result.iloc[2].bulk_values is None


def test_run_batch_preserves_source_and_round_trips_parquet(
    tmp_path: Path, two_days: tuple[np.ndarray, np.ndarray]
) -> None:
    input_path = tmp_path / "clouds.parquet"
    output_path = tmp_path / "clouds_bulk.parquet"
    source = _clouds()
    source.to_parquet(input_path)
    original_bytes = input_path.read_bytes()

    run_batch(input_path, output_path, tmp_path)

    assert input_path.read_bytes() == original_bytes
    saved = pd.read_parquet(output_path)
    assert saved.index.tolist() == source.index.tolist()
    saved_values = np.stack(saved.bulk_values.iloc[0]).astype(np.float32)
    assert saved_values.shape == (3, N_BULK_LEVELS)
    assert saved_values.dtype == np.float32
    assert saved.bulk_values.iloc[2] is None
    assert not (tmp_path / ".clouds_bulk.parquet.tmp").exists()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("track_start_time", pd.Timestamp("2007-01-01 00:01"), "aligned"),
        ("track_length", pd.Timedelta(minutes=16), "multiples"),
        ("track_length", pd.Timedelta(minutes=30), "disagree"),
    ],
)
def test_rejects_invalid_track_metadata(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    clouds = pd.DataFrame(
        {
            "tracknumber": [1],
            "track_start_time": [pd.Timestamp("2007-01-01")],
            "track_length": [pd.Timedelta(minutes=15)],
            "model_grid_ncell": [[np.array([0]), np.array([1])]],
        }
    )
    clouds.at[0, column] = value

    with pytest.raises(ValueError, match=message):
        extract_bulk_values_by_day(clouds, tmp_path, show_progress=False)


def test_rejects_out_of_range_cells(
    tmp_path: Path, two_days: tuple[np.ndarray, np.ndarray]
) -> None:
    clouds = pd.DataFrame(
        {
            "tracknumber": [1],
            "track_start_time": [pd.Timestamp("2007-01-01")],
            "track_length": [pd.Timedelta(0)],
            "model_grid_ncell": [[np.array([5])]],
        }
    )

    with pytest.raises(IndexError, match="between 0 and 4"):
        extract_bulk_values_by_day(clouds, tmp_path, show_progress=False)


def test_missing_daily_file_has_clear_error(tmp_path: Path) -> None:
    clouds = pd.DataFrame(
        {
            "tracknumber": [1],
            "track_start_time": [pd.Timestamp("2007-01-01")],
            "track_length": [pd.Timedelta(0)],
            "model_grid_ncell": [[np.array([0])]],
        }
    )

    with pytest.raises(FileNotFoundError, match="Bulk-data file does not exist"):
        extract_bulk_values_by_day(clouds, tmp_path, show_progress=False)
