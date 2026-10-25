from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


INPUT_PARQUET = Path(
    "/wolke_scratch/dnikolo/ICON_Data/sedim_search/R2B4/p3_lite/"
    "postprocessing_output/Final_results/2007_glac_04.parquet"
)
OUTPUT_PARQUET = INPUT_PARQUET.with_name(f"{INPUT_PARQUET.stem}_bulk.parquet")
BULK_DATA_DIR = Path(
    "/wolke_scratch/dnikolo/ICON_Data/sedim_search/" "R2B4/p3_lite/bulk_data"
)

TIMESTEP = pd.Timedelta(minutes=15)
TIMESTEP_NS = TIMESTEP.value
TIMESTEPS_PER_DAY = 96
N_BULK_LEVELS = 25

REQUIRED_COLUMNS = {
    "tracknumber",
    "track_start_time",
    "track_length",
    "model_grid_ncell",
}


def timestamp_to_filename(
    timestamp: pd.Timestamp, bulk_data_dir: Path = BULK_DATA_DIR
) -> Path:
    timestamp = pd.Timestamp(timestamp)
    filename = f"sedim_r2b4_p3_bulk_cloud_ml_{timestamp:%Y%m%dT000000}Z.nc"
    return bulk_data_dir / filename


def _validate_tracks(
    clouds: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing_columns = REQUIRED_COLUMNS.difference(clouds.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {names}")

    valid_positions = np.flatnonzero(clouds["tracknumber"].notna().to_numpy())
    starts = pd.to_datetime(clouds["track_start_time"], errors="coerce")
    start_ns = starts.array.asi8

    if np.any(start_ns[valid_positions] == pd.NaT.value):
        raise ValueError("Valid tracks must have a track_start_time")
    if np.any(start_ns[valid_positions] % TIMESTEP_NS):
        raise ValueError(
            "track_start_time values must be aligned to 15-minute intervals"
        )

    cell_sequences = clouds["model_grid_ncell"].to_numpy(copy=False)
    step_counts = np.zeros(len(clouds), dtype=np.int64)
    for position in valid_positions:
        sequence = cell_sequences[position]
        if sequence is None:
            raise ValueError(
                f"Track at row position {position} has no model_grid_ncell data"
            )
        step_counts[position] = len(sequence)

    if np.any(step_counts[valid_positions] == 0):
        raise ValueError("Valid tracks must contain at least one model_grid_ncell step")

    lengths = pd.to_timedelta(clouds["track_length"], errors="coerce")
    length_ns = lengths.array.asi8
    if np.any(length_ns[valid_positions] == pd.NaT.value):
        raise ValueError("Valid tracks must have a track_length")
    if np.any(length_ns[valid_positions] < 0):
        raise ValueError("track_length cannot be negative")
    if np.any(length_ns[valid_positions] % TIMESTEP_NS):
        raise ValueError("track_length values must be multiples of 15 minutes")

    expected_counts = length_ns[valid_positions] // TIMESTEP_NS + 1
    actual_counts = step_counts[valid_positions]
    mismatched = valid_positions[expected_counts != actual_counts]
    if len(mismatched):
        preview = ", ".join(map(str, mismatched[:5]))
        raise ValueError(
            "track_length and model_grid_ncell disagree at row positions " f"{preview}"
        )

    return valid_positions, start_ns, step_counts, cell_sequences


def _load_daily_ze(filename: Path, day: pd.Timestamp) -> np.ndarray:
    if not filename.is_file():
        raise FileNotFoundError(f"Bulk-data file does not exist: {filename}")

    with xr.open_dataset(filename) as dataset:
        if "ze" not in dataset:
            raise ValueError(f"Dataset does not contain a 'ze' variable: {filename}")

        ze_data = dataset["ze"]
        required_dims = {"time", "bulk_level", "ncells"}
        if set(ze_data.dims) != required_dims:
            raise ValueError(f"Unexpected ze dimensions in {filename}: {ze_data.dims}")

        ze_data = ze_data.transpose("time", "bulk_level", "ncells")
        if ze_data.sizes["time"] != TIMESTEPS_PER_DAY:
            raise ValueError(
                f"Expected {TIMESTEPS_PER_DAY} time slots in {filename}, "
                f"found {ze_data.sizes['time']}"
            )
        if ze_data.sizes["bulk_level"] != N_BULK_LEVELS:
            raise ValueError(
                f"Expected {N_BULK_LEVELS} bulk levels in {filename}, "
                f"found {ze_data.sizes['bulk_level']}"
            )

        expected_times = int(day.strftime("%Y%m%d")) + (
            np.arange(TIMESTEPS_PER_DAY) / TIMESTEPS_PER_DAY
        )
        if not np.allclose(
            dataset["time"].values,
            expected_times,
            rtol=0,
            atol=1e-7,
        ):
            raise ValueError(
                f"time coordinates do not match 15-minute slots for "
                f"{day:%Y-%m-%d} in {filename}"
            )

        expected_cells = np.arange(ze_data.sizes["ncells"])
        if not np.array_equal(dataset["ncells"].values, expected_cells):
            raise ValueError(
                f"ncells must be consecutive zero-based indices in {filename}"
            )

        # Loading the whole variable once avoids repeatedly decompressing the
        # large NetCDF chunks for individual cloud-cell selections.
        return np.asarray(ze_data.values, dtype=np.float32)


def _aggregate_time_slot(
    ze_at_time: np.ndarray,
    requests: list[tuple[int, int, np.ndarray]],
    bulk_values: np.ndarray,
) -> None:
    nonempty_requests = [request for request in requests if request[2].size]
    if not nonempty_requests:
        return

    lengths = np.fromiter(
        (request[2].size for request in nonempty_requests),
        dtype=np.int64,
        count=len(nonempty_requests),
    )
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    cell_indices = np.concatenate([request[2] for request in nonempty_requests])

    n_cells = ze_at_time.shape[1]
    if np.any(cell_indices < 0) or np.any(cell_indices >= n_cells):
        raise IndexError(
            f"model_grid_ncell indices must be between 0 and {n_cells - 1}"
        )

    selected = np.take(ze_at_time, cell_indices, axis=1).T
    valid = ~np.isnan(selected)
    sums = np.add.reduceat(
        np.where(valid, selected, np.float32(0)), offsets[:-1], axis=0
    )
    counts = np.add.reduceat(valid, offsets[:-1], axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float32),
        where=counts > 0,
    )

    for mean, (row_position, track_step, _) in zip(
        means, nonempty_requests, strict=True
    ):
        bulk_values[row_position][track_step] = mean


def extract_bulk_values_by_day(
    clouds: pd.DataFrame,
    bulk_data_dir: Path = BULK_DATA_DIR,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Add bulk values while reading each required daily dataset only once."""
    bulk_data_dir = Path(bulk_data_dir)
    valid_positions, start_ns, step_counts, cell_sequences = _validate_tracks(clouds)

    bulk_values = np.empty(len(clouds), dtype=object)
    bulk_values[:] = None
    for position in valid_positions:
        bulk_values[position] = np.full(
            (step_counts[position], N_BULK_LEVELS), np.nan, dtype=np.float32
        )

    if not len(valid_positions):
        clouds["bulk_values"] = bulk_values
        return clouds

    end_ns = start_ns.copy()
    end_ns[valid_positions] += (step_counts[valid_positions] - 1) * TIMESTEP_NS
    first_day = pd.Timestamp(start_ns[valid_positions].min()).normalize()
    last_day = pd.Timestamp(end_ns[valid_positions].max()).normalize()
    days = pd.date_range(first_day, last_day, freq="D")

    overall_start = time.perf_counter()
    for day_number, day in enumerate(days, start=1):
        day_start = day.value
        next_day = day_start + pd.Timedelta(days=1).value
        active = valid_positions[
            (start_ns[valid_positions] < next_day)
            & (end_ns[valid_positions] >= day_start)
        ]
        if not len(active):
            continue

        day_start_time = time.perf_counter()
        filename = timestamp_to_filename(day, bulk_data_dir)
        ze = _load_daily_ze(filename, day)
        requests_by_slot: list[list[tuple[int, int, np.ndarray]]] = [
            [] for _ in range(TIMESTEPS_PER_DAY)
        ]

        for position in active:
            first_step = max(0, (day_start - start_ns[position]) // TIMESTEP_NS)
            stop_step = min(
                step_counts[position],
                (next_day - start_ns[position]) // TIMESTEP_NS,
            )
            first_slot = (
                start_ns[position] + first_step * TIMESTEP_NS - day_start
            ) // TIMESTEP_NS

            for track_step in range(first_step, stop_step):
                cells = np.asarray(cell_sequences[position][track_step], dtype=np.intp)
                if cells.ndim != 1:
                    raise ValueError(
                        "Each model_grid_ncell timestep must be one-dimensional; "
                        f"row position {position}, timestep {track_step} has "
                        f"shape {cells.shape}"
                    )
                slot = first_slot + track_step - first_step
                requests_by_slot[slot].append((position, track_step, cells))

        request_count = 0
        for slot, requests in enumerate(requests_by_slot):
            request_count += len(requests)
            if requests:
                _aggregate_time_slot(ze[slot], requests, bulk_values)

        del ze
        if show_progress:
            elapsed = time.perf_counter() - day_start_time
            print(
                f"[{day_number}/{len(days)}] {day:%Y-%m-%d}: "
                f"{request_count:,} requests in {elapsed:.2f}s",
                flush=True,
            )

    clouds["bulk_values"] = bulk_values
    if show_progress:
        elapsed = time.perf_counter() - overall_start
        print(f"Extraction completed in {elapsed:.1f}s", flush=True)
    return clouds


def _bulk_values_to_arrow(values: np.ndarray) -> pa.ListArray:
    offsets = np.zeros(len(values) + 1, dtype=np.int32)
    null_mask = np.zeros(len(values), dtype=bool)
    arrays: list[np.ndarray] = []

    for position, value in enumerate(values):
        if value is None:
            null_mask[position] = True
            offsets[position + 1] = offsets[position]
            continue

        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != N_BULK_LEVELS:
            raise ValueError(
                "bulk_values entries must have shape (timesteps, "
                f"{N_BULK_LEVELS}); row-group position {position} has "
                f"shape {array.shape}"
            )
        arrays.append(array)
        offsets[position + 1] = offsets[position] + array.shape[0]

    if arrays:
        flat_values = np.concatenate(arrays, axis=0).reshape(-1)
    else:
        flat_values = np.empty(0, dtype=np.float32)

    primitive_values = pa.array(flat_values, type=pa.float32())
    timestep_values = pa.FixedSizeListArray.from_arrays(primitive_values, N_BULK_LEVELS)
    return pa.ListArray.from_arrays(
        pa.array(offsets),
        timestep_values,
        mask=pa.array(null_mask),
    )


def save_parquet_atomically(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    row_group_size: int = 5_000,
) -> None:
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if "bulk_values" not in dataframe:
        raise ValueError("DataFrame does not contain a bulk_values column")

    output_path = Path(output_path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        base = dataframe.drop(columns="bulk_values")
        base_schema = pa.Schema.from_pandas(base, preserve_index=True)
        stop_positions = range(row_group_size, max(len(dataframe), 1), row_group_size)
        boundaries = [0, *stop_positions, len(dataframe)]

        for start, stop in zip(boundaries, boundaries[1:]):
            base_table = pa.Table.from_pandas(
                base.iloc[start:stop],
                schema=base_schema,
                preserve_index=True,
            )
            arrow_bulk_values = _bulk_values_to_arrow(
                dataframe["bulk_values"].iloc[start:stop].to_numpy(copy=False)
            )
            table = base_table.append_column("bulk_values", arrow_bulk_values)

            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema)
            writer.write_table(table)

        if writer is None:
            raise RuntimeError("Failed to initialize Parquet writer")
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise


def run_batch(
    input_path: Path = INPUT_PARQUET,
    output_path: Path = OUTPUT_PARQUET,
    bulk_data_dir: Path = BULK_DATA_DIR,
) -> None:
    clouds = pd.read_parquet(input_path)
    extract_bulk_values_by_day(clouds, bulk_data_dir)
    save_parquet_atomically(clouds, output_path)
    print(f"Saved {len(clouds):,} rows to {output_path}", flush=True)


def main() -> None:
    run_batch()


if __name__ == "__main__":
    main()
