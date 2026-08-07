from __future__ import annotations

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import dask
from glaciation_time_estimator.auxiliary_func.config_reader import read_config as read_gte_config
from msgi.aux import read_config as read_msgi_config


DEFAULT_CONDENSATE_THRESHOLD = 1.0e-5
DEFAULT_CTMASK_THRESHOLD = 1.0e-5
ZE_NONCLOUDY_ANCHOR = np.float32(-30.0)
CLI_NONCLOUDY_ANCHOR = np.float32(0.0)


@dataclass(frozen=True)
class VariableNames:
    zg: str = "zg"
    ze: str = "cloudsat_ze_mean"
    cli: str = "cli"
    clw: str = "clw"
    ctlev: str = "ctlev"
    ctmask: str = "ctmask"


@dataclass(frozen=True)
class CloudTopIndex:
    index: xr.DataArray
    valid: xr.DataArray


def build_bulk_level_offsets(max_depth_m: float, dz_m: float) -> np.ndarray:
    """Return DARDAR-style layer bounds below cloud top as negative offsets."""
    if max_depth_m <= 0.0:
        raise ValueError("--max-depth-m must be > 0")
    if dz_m <= 0.0:
        raise ValueError("--dz-m must be > 0")

    edges = np.arange(0.0, max_depth_m + 0.5 * dz_m, dz_m, dtype=np.float32)
    if edges.size < 2:
        raise ValueError("Could not construct at least one bulk layer")
    edges[-1] = np.float32(max_depth_m)
    return np.column_stack((-edges[:-1], -edges[1:])).astype(np.float32)


def require_variables(ds: xr.Dataset, names: list[str], dataset_label: str) -> None:
    missing = [name for name in names if name not in ds.variables]
    if missing:
        raise ValueError(f"{dataset_label} is missing required variables: {missing}")


def require_dims(var: xr.DataArray, dims: set[str], name: str) -> None:
    if not dims.issubset(var.dims):
        raise ValueError(f"{name!r} must have dimensions including {sorted(dims)}")


def as_time_height_ncells(var: xr.DataArray, name: str) -> xr.DataArray:
    require_dims(var, {"time", "height", "ncells"}, name)
    return var.transpose("time", "height", "ncells").astype(np.float32)


def as_time_ncells(var: xr.DataArray, name: str) -> xr.DataArray:
    require_dims(var, {"time", "ncells"}, name)
    return var.transpose("time", "ncells")


def as_zg_da(zg: xr.DataArray) -> xr.DataArray:
    if {"height", "ncells"}.issubset(zg.dims):
        return zg.transpose("height", "ncells").astype(np.float32)
    raise ValueError("'zg' must have dimensions (height, ncells)")


def single_height_chunk(var: xr.DataArray) -> xr.DataArray:
    if var.chunks is None or "height" not in var.dims:
        return var
    return var.chunk({"height": -1})


def validate_shapes(ze: xr.DataArray, cli: xr.DataArray, clw: xr.DataArray, zg: xr.DataArray, ctmask: xr.DataArray) -> None:
    if ze.sizes != cli.sizes or ze.sizes != clw.sizes:
        raise ValueError("ze, cli, and clw must have the same (time, height, ncells) shape")
    if ctmask.sizes["time"] != ze.sizes["time"] or ctmask.sizes["ncells"] != ze.sizes["ncells"]:
        raise ValueError("ctmask must have shape (time, ncells) matching atm_3d")
    if zg.sizes["height"] != ze.sizes["height"] or zg.sizes["ncells"] != ze.sizes["ncells"]:
        raise ValueError("zg must have shape (height, ncells)")


def height_decreases_with_index(zg: xr.DataArray) -> bool:
    """Return True when geometric height decreases as the height index increases."""
    profile = zg.median(dim=[dim for dim in zg.dims if dim != "height"], skipna=True)
    diffs = profile.diff("height")
    finite = diffs.where(np.isfinite(diffs))
    if int(np.isfinite(finite).sum().compute()) == 0:
        raise ValueError("Could not determine vertical orientation from finite zg values")
    finite = finite.compute()
    return bool(finite.median(skipna=True) < 0.0)


def rounded_integer_index(values: xr.DataArray, label: str) -> xr.DataArray:
    rounded = values.round()
    bad = np.isfinite(values) & (abs(values - rounded) > 1.0e-4)
    if bool(bad.any().compute()):
        raise ValueError(f"{label} contains non-integer-like cloud-top levels")
    return rounded


def compute_cloudy(cli: xr.DataArray, clw: xr.DataArray, valid_cloud_top: xr.DataArray, condensate_threshold: float) -> xr.DataArray:
    condensate = cli + clw
    return (
        valid_cloud_top
        & np.isfinite(cli)
        & np.isfinite(clw)
        & np.isfinite(condensate)
        & (condensate >= condensate_threshold)
    )


def compute_noncloudy(
    cli: xr.DataArray,
    clw: xr.DataArray,
    valid_cloud_top: xr.DataArray,
    condensate_threshold: float,
) -> xr.DataArray:
    condensate = cli + clw
    return valid_cloud_top & np.isfinite(condensate) & (condensate < condensate_threshold)


def cloud_top_index_from_one_based_ctlev(ctlev: xr.DataArray, ctmask: xr.DataArray, n_height: int, ctmask_threshold: float) -> CloudTopIndex:
    """Convert one-based ICON cloud-top model levels to zero-based positional indices."""
    index_candidate = rounded_integer_index(ctlev.astype(np.float64), "ctlev") - 1.0
    valid = (
        np.isfinite(index_candidate)
        & (index_candidate >= 0)
        & (index_candidate < n_height)
        & np.isfinite(ctmask)
        & (ctmask > ctmask_threshold)
    )
    index = index_candidate.where(valid, -1).astype(np.int16)
    return CloudTopIndex(index=index, valid=valid)


def derive_cloud_top_index_from_condensate(
    cli: xr.DataArray,
    clw: xr.DataArray,
    ctmask: xr.DataArray,
    zg: xr.DataArray,
    condensate_threshold: float,
    ctmask_threshold: float,
) -> CloudTopIndex:
    """Derive cloud top as the highest geometric level with enough condensate."""
    valid_column = np.isfinite(ctmask) & (ctmask > ctmask_threshold)
    cloudy = compute_cloudy(cli, clw, valid_column, condensate_threshold)
    zg_for_cloudy = zg.broadcast_like(cli)
    valid = (cloudy & np.isfinite(zg_for_cloudy)).any("height")
    candidate_heights = zg_for_cloudy.where(cloudy & np.isfinite(zg_for_cloudy), -np.inf)
    index = candidate_heights.argmax("height").where(valid, -1).astype(np.int16)
    return CloudTopIndex(index=index, valid=valid)


def cloud_top_height(zg: xr.DataArray, cloud_top: CloudTopIndex) -> xr.DataArray:
    safe_index = cloud_top.index.where(cloud_top.valid, 0).astype(np.int64).compute()
    height = zg.isel(height=safe_index).where(cloud_top.valid).astype(np.float32)
    return height.drop_vars("height", errors="ignore")


def bulk_midpoint_depths(bulk_offsets_m: np.ndarray) -> xr.DataArray:
    midpoints = np.abs(0.5 * (bulk_offsets_m[:, 0] + bulk_offsets_m[:, 1])).astype(np.float32)
    return xr.DataArray(midpoints, dims=("bulk_level",), coords={"bulk_level": np.arange(midpoints.size, dtype=np.int16)})


def _interp_time_block(
    values: np.ndarray,
    depth: np.ndarray,
    cloudy: np.ndarray,
    noncloudy: np.ndarray,
    target_depths: np.ndarray,
    noncloudy_anchor: float,
) -> tuple[np.ndarray, np.ndarray]:
    leading_shape = values.shape[:-2]
    height_count, cell_count = values.shape[-2:]
    bulk_count = target_depths.size
    values_3d = values.reshape((-1, height_count, cell_count))
    depth_3d_all = depth.reshape((-1, height_count, cell_count))
    cloudy_3d_all = cloudy.reshape((-1, height_count, cell_count))
    noncloudy_3d_all = noncloudy.reshape((-1, height_count, cell_count))
    output = np.full((values_3d.shape[0], bulk_count, cell_count), np.nan, dtype=np.float32)
    n_valid = np.zeros((values_3d.shape[0], bulk_count, cell_count), dtype=np.int16)
    target = target_depths.astype(np.float32)[None, None, :]

    for block in range(values_3d.shape[0]):
        block_values = values_3d[block]
        block_depth = depth_3d_all[block]
        block_cloudy = cloudy_3d_all[block]
        block_noncloudy = noncloudy_3d_all[block]
        depth_3d = block_depth[:, :, None]
        cloudy_3d = block_cloudy[:, :, None]
        noncloudy_3d = block_noncloudy[:, :, None]
        finite_values = np.isfinite(block_values)[:, :, None]
        anchor_valid = np.isfinite(depth_3d) & ((cloudy_3d & finite_values) | noncloudy_3d)
        anchor_values = np.where(block_cloudy, block_values, np.float32(noncloudy_anchor))[:, :, None]

        with np.errstate(all="ignore"):
            lower_depth = np.nanmax(np.where(anchor_valid & (depth_3d <= target), depth_3d, np.nan), axis=0)
            upper_depth = np.nanmin(np.where(anchor_valid & (depth_3d >= target), depth_3d, np.nan), axis=0)
            lower_value = np.nanmax(np.where(np.abs(depth_3d - lower_depth[None, :, :]) <= 1.0e-5, anchor_values, np.nan), axis=0)
            upper_value = np.nanmax(np.where(np.abs(depth_3d - upper_depth[None, :, :]) <= 1.0e-5, anchor_values, np.nan), axis=0)
            nearest_cloudy = np.nanmin(np.where(cloudy_3d, np.abs(depth_3d - target), np.nan), axis=0)
            nearest_noncloudy = np.nanmin(np.where(noncloudy_3d, np.abs(depth_3d - target), np.nan), axis=0)
            interpolated = lower_value + (target_depths[None, :] - lower_depth) * (upper_value - lower_value) / (upper_depth - lower_depth)

        same_depth = np.abs(lower_depth - upper_depth) <= 1.0e-5
        interpolated = np.where(same_depth, lower_value, interpolated)
        exact_cloudy = np.any((np.abs(depth_3d - target) <= 1.0e-5) & cloudy_3d & finite_values, axis=0)
        valid = (
            np.any(block_cloudy, axis=0)[:, None]
            & np.isfinite(lower_depth)
            & np.isfinite(upper_depth)
            & np.isfinite(interpolated)
            & ~(nearest_noncloudy < nearest_cloudy)
        )
        output[block] = np.where(valid, interpolated, np.nan).T.astype(np.float32)
        n_valid[block] = np.where(valid, np.where(exact_cloudy, 1, 2), 0).T.astype(np.int16)

    return output.reshape(leading_shape + (bulk_count, cell_count)), n_valid.reshape(leading_shape + (bulk_count, cell_count))


def interpolate_to_bulk_midpoints(
    values: xr.DataArray,
    depth: xr.DataArray,
    cloudy: xr.DataArray,
    noncloudy: xr.DataArray,
    target_depths: xr.DataArray,
    noncloudy_anchor: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate profile-specific anchors to bulk-layer midpoint depths."""
    interpolated, n_valid = xr.apply_ufunc(
        _interp_time_block,
        values,
        depth,
        cloudy,
        noncloudy,
        target_depths,
        input_core_dims=[["height", "ncells"], ["height", "ncells"], ["height", "ncells"], ["height", "ncells"], ["bulk_level"]],
        output_core_dims=[["bulk_level", "ncells"], ["bulk_level", "ncells"]],
        dask="parallelized",
        output_dtypes=[np.float32, np.int16],
        kwargs={"noncloudy_anchor": float(noncloudy_anchor)},
        dask_gufunc_kwargs={
            "output_sizes": {"bulk_level": target_depths.sizes["bulk_level"]},
            "allow_rechunk": True,
        },
    )
    return interpolated.transpose("time", "bulk_level", "ncells"), n_valid.transpose("time", "bulk_level", "ncells")


def build_output_dataset(
    atm_3d: xr.Dataset,
    ct: xr.Dataset,
    names: VariableNames,
    max_depth_m: float,
    dz_m: float,
    condensate_threshold: float = DEFAULT_CONDENSATE_THRESHOLD,
    ctmask_threshold: float = DEFAULT_CTMASK_THRESHOLD,
    derive_cloud_top: bool = False,
) -> xr.Dataset:
    require_variables(atm_3d, [names.zg, names.ze, names.cli, names.clw, "time"], "atm_3d")
    required_ct_vars = [names.ctmask] if derive_cloud_top else [names.ctlev, names.ctmask]
    require_variables(ct, required_ct_vars, "cloud-top dataset")

    ze = as_time_height_ncells(atm_3d[names.ze], names.ze)
    cli = as_time_height_ncells(atm_3d[names.cli], names.cli)
    clw = as_time_height_ncells(atm_3d[names.clw], names.clw)
    zg = as_zg_da(atm_3d[names.zg])
    ctmask = as_time_ncells(ct[names.ctmask], names.ctmask).astype(np.float32)
    validate_shapes(ze, cli, clw, zg, ctmask)
    ze = single_height_chunk(ze)
    cli = single_height_chunk(cli)
    clw = single_height_chunk(clw)
    zg = single_height_chunk(zg)

    bulk_offsets_m = build_bulk_level_offsets(max_depth_m, dz_m)
    target_depths = bulk_midpoint_depths(bulk_offsets_m)

    if derive_cloud_top:
        cloud_top = derive_cloud_top_index_from_condensate(
            cli=cli,
            clw=clw,
            ctmask=ctmask,
            zg=zg,
            condensate_threshold=condensate_threshold,
            ctmask_threshold=ctmask_threshold,
        )
    else:
        ctlev = as_time_ncells(ct[names.ctlev], names.ctlev)
        if ctlev.sizes != ctmask.sizes:
            raise ValueError("ctlev must have shape (time, ncells) matching atm_3d")
        cloud_top = cloud_top_index_from_one_based_ctlev(ctlev, ctmask, ze.sizes["height"], ctmask_threshold)

    cloud_top_height_m = cloud_top_height(zg, cloud_top)
    depth = (cloud_top_height_m - zg).transpose("time", "height", "ncells")
    in_requested_depth = np.isfinite(depth) & (depth >= 0.0) & (depth <= max_depth_m)
    cloudy = compute_cloudy(cli, clw, cloud_top.valid, condensate_threshold) & in_requested_depth
    noncloudy = compute_noncloudy(cli, clw, cloud_top.valid, condensate_threshold) & in_requested_depth

    ze_interp, ze_n_valid = interpolate_to_bulk_midpoints(
        ze,
        depth,
        cloudy,
        noncloudy,
        target_depths,
        ZE_NONCLOUDY_ANCHOR,
    )
    cli_interp, cli_n_valid = interpolate_to_bulk_midpoints(
        cli,
        depth,
        cloudy,
        noncloudy,
        target_depths,
        CLI_NONCLOUDY_ANCHOR,
    )

    coords: dict[str, xr.DataArray | tuple[str, np.ndarray]] = {
        "time": atm_3d["time"],
        "bulk_level": target_depths["bulk_level"],
    }
    coords["ncells"] = atm_3d["ncells"] if "ncells" in atm_3d.coords else ("ncells", np.arange(ze.sizes["ncells"], dtype=np.int32))

    out = xr.Dataset(
        data_vars={
            "ze": ze_interp,
            "cli": cli_interp,
            "ze_n_valid_levels": ze_n_valid,
            "cli_n_valid_levels": cli_n_valid,
            "cloud_top_level_index": cloud_top.index,
            "cloud_top_height_m": cloud_top_height_m,
            "bulk_level_offset_top": (("bulk_level",), bulk_offsets_m[:, 0]),
            "bulk_level_offset_bottom": (("bulk_level",), bulk_offsets_m[:, 1]),
            "bulk_level_offset_mid": (("bulk_level",), 0.5 * (bulk_offsets_m[:, 0] + bulk_offsets_m[:, 1])),
        },
        coords=coords,
    )
    for coord_name in ("clon", "clat"):
        if coord_name in atm_3d:
            out[coord_name] = atm_3d[coord_name]

    out["ze"].attrs.update(atm_3d[names.ze].attrs)
    out["ze"].attrs.setdefault("long_name", "Cloud-bulk radar reflectivity")
    out["ze"].attrs["comment"] = "Linearly interpolated to bulk-layer midpoints below cloud top; non-cloudy anchors use -30 dBZ."
    out["cli"].attrs.update(atm_3d[names.cli].attrs)
    out["cli"].attrs.setdefault("long_name", "Cloud-bulk ice water content")
    out["cli"].attrs["comment"] = "Linearly interpolated to bulk-layer midpoints below cloud top; non-cloudy anchors use 0."
    out["ze_n_valid_levels"].attrs["long_name"] = "Number of source anchors used for ze midpoint interpolation"
    out["cli_n_valid_levels"].attrs["long_name"] = "Number of source anchors used for cli midpoint interpolation"
    out["cloud_top_level_index"].attrs.update(
        {"long_name": "Zero-based positional cloud-top level index", "comment": "-1 marks invalid cloud tops."}
    )
    out["cloud_top_height_m"].attrs.update({"long_name": "Cloud-top geometric height", "units": "m"})
    for name, long_name in [
        ("bulk_level_offset_top", "Upper bound of cloud-bulk layer relative to cloud top"),
        ("bulk_level_offset_bottom", "Lower bound of cloud-bulk layer relative to cloud top"),
        ("bulk_level_offset_mid", "Midpoint of cloud-bulk layer relative to cloud top"),
    ]:
        out[name].attrs.update({"long_name": long_name, "units": "m"})
    out["bulk_level"].attrs.update(
        {
            "long_name": "Cloud-bulk layer index below cloud top",
            "comment": f"0 is the shallowest layer, {bulk_offsets_m.shape[0] - 1} is the deepest layer.",
        }
    )
    out.attrs.update(
        {
            "source": "Derived from ICON atm_3d and cloud-top files",
            "derive_cloud_top": str(bool(derive_cloud_top)),
            "zg_decreases_with_height_index": str(height_decreases_with_index(zg)),
            "max_depth_m": float(max_depth_m),
            "dz_m": float(dz_m),
            "ctmask_threshold": float(ctmask_threshold),
            "condensate_threshold": float(condensate_threshold),
            "bulk_interpolation": "midpoint_depth_linear_anchors",
        }
    )
    return out


def netcdf_encoding(ds: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name in ds.data_vars:
        if ds[name].dtype.kind in {"f", "i", "u"}:
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
    return encoding


def print_diagnostics(ds: xr.Dataset) -> None:
    valid_ct = ds["cloud_top_level_index"] >= 0
    print(ds)
    print(f"zg decreases with height index: {ds.attrs.get('zg_decreases_with_height_index')}")
    print(f"valid cloud-top fraction: {float(valid_ct.mean().compute()):.6f}")
    for name in ("ze", "cli"):
        finite = np.isfinite(ds[name])
        print(f"{name} finite fraction: {float(finite.mean().compute()):.6f}")
        print(f"{name} all NaN: {bool((~finite).all().compute())}")


def write_output(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with dask.config.set(scheduler="single-threaded"):
        ds.to_netcdf(output_path, format="NETCDF4", encoding=netcdf_encoding(ds))


def extract_icon_file_date(filepath: str | Path) -> datetime | None:
    basename = os.path.basename(filepath)
    match = re.search(r"(\d{8}T\d{6}Z)", basename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")


def output_path_for_atm_file(atm_file: Path, output_dir: Path) -> Path:
    name = atm_file.name
    if "_atm_3d_cloud_ml_" in name:
        name = name.replace("_atm_3d_cloud_ml_", "_bulk_cloud_ml_")
    else:
        name = f"bulk_{name}"
    return output_dir / name


def collect_icon_file_pairs(
    input_dir: Path,
    output_dir: Path,
    experiment_name: str,
    start_date: datetime,
    end_date: datetime,
) -> list[tuple[Path, Path, Path]]:
    atm_files = sorted(glob.glob(str(input_dir / "*_atm_3d_cloud_ml_*.nc")))
    ctp_files = sorted(glob.glob(str(input_dir / "*_ctp_2d_ml_*.nc")))
    ctp_by_date = {extract_icon_file_date(path): Path(path) for path in ctp_files}

    tasks: list[tuple[Path, Path, Path]] = []
    for atm_path_raw in atm_files:
        atm_path = Path(atm_path_raw)
        if experiment_name and not atm_path.name.startswith(experiment_name):
            continue
        file_dt = extract_icon_file_date(atm_path)
        if file_dt is None or file_dt < start_date or file_dt > end_date:
            continue
        ctp_path = ctp_by_date.get(file_dt)
        if ctp_path is None:
            print(f"Skipping {atm_path.name}: no matching ctp_2d file for {file_dt:%Y%m%dT%H%M%SZ}")
            continue
        tasks.append((atm_path, ctp_path, output_path_for_atm_file(atm_path, output_dir)))
    return tasks


def _process_one_file(task: tuple[Path, Path, Path, VariableNames, float, float, float, float, bool]) -> Path:
    (
        atm_path,
        ct_path,
        output_path,
        names,
        max_depth_m,
        dz_m,
        condensate_threshold,
        ctmask_threshold,
        derive_cloud_top,
    ) = task
    with xr.open_dataset(atm_path, chunks={}) as atm_3d, xr.open_dataset(ct_path, chunks={}) as ct:
        out = build_output_dataset(
            atm_3d,
            ct,
            names,
            max_depth_m=max_depth_m,
            dz_m=dz_m,
            condensate_threshold=condensate_threshold,
            ctmask_threshold=ctmask_threshold,
            derive_cloud_top=derive_cloud_top,
        )
        write_output(out, output_path)
    return output_path


def process_dir_from_configs(gte_config: dict, msgi_config: dict) -> None:
    if msgi_config.get("model", "ICON") != "ICON":
        raise ValueError("model_bulk_level_v2_xr.py currently supports only ICON inputs")

    input_dir = Path(msgi_config["resample_input_dir"])
    output_dir = input_dir.parent / "bulk_data"
    experiment_name = msgi_config.get("experiment_name", "")
    start_date = gte_config["start_time"].replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = gte_config["end_time"].replace(hour=0, minute=0, second=0, microsecond=0)

    file_pairs = collect_icon_file_pairs(
        input_dir=input_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        start_date=start_date,
        end_date=end_date,
    )
    if not file_pairs:
        print(
            f"No ICON atm_3d_cloud files found in {input_dir} "
            f"for experiment prefix '{experiment_name}' "
            f"within date range {start_date} to {end_date}."
        )
        return

    names = VariableNames(
        zg=msgi_config.get("bulk_zg_name", VariableNames.zg),
        ze=msgi_config.get("bulk_ze_name", VariableNames.ze),
        cli=msgi_config.get("bulk_cli_name", VariableNames.cli),
        clw=msgi_config.get("bulk_clw_name", VariableNames.clw),
        ctlev=msgi_config.get("bulk_ctlev_name", VariableNames.ctlev),
        ctmask=msgi_config.get("bulk_ctmask_name", VariableNames.ctmask),
    )
    max_depth_m = float(msgi_config.get("bulk_max_depth_m", 3000.0))
    dz_m = float(msgi_config.get("bulk_dz_m", 120.0))
    condensate_threshold = float(msgi_config.get("bulk_condensate_threshold", DEFAULT_CONDENSATE_THRESHOLD))
    ctmask_threshold = float(msgi_config.get("bulk_ctmask_threshold", DEFAULT_CTMASK_THRESHOLD))
    derive_cloud_top = bool(msgi_config.get("bulk_derive_cloud_top", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            atm_path,
            ct_path,
            out_path,
            names,
            max_depth_m,
            dz_m,
            condensate_threshold,
            ctmask_threshold,
            derive_cloud_top,
        )
        for atm_path, ct_path, out_path in file_pairs
    ]
    workers = min(
        int(gte_config.get("n_preproc_cores")) * int(gte_config.get("n_preproc_threads")),
        16,
        len(tasks),
        os.cpu_count() or 1,
    )
    print(f"Processing {len(tasks)} ICON files with {workers} workers...")
    print(f"Writing bulk outputs to {output_dir}")

    failures: list[str] = []
    if workers == 1:
        for task in tasks:
            try:
                out_path = _process_one_file(task)
                print(f"Finished {out_path}")
            except Exception as exc:
                failures.append(f"{task[0]}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one_file, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    out_path = fut.result()
                    print(f"Finished {out_path}")
                except Exception as exc:
                    failures.append(f"{task[0]}: {exc}")

    if failures:
        raise RuntimeError("Bulk file generation failed:\n" + "\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ICON cloud-top-relative bulk midpoint interpolation for Ze and CLI. "
            "Pass --atm-3d/--cloud-top/--output for one file, or pass configured GTE/MSGI arguments "
            "to process a config-selected directory."
        )
    )
    parser.add_argument("--atm-3d", type=Path, help="ICON atm_3d NetCDF input file")
    parser.add_argument("--cloud-top", type=Path, help="Cloud-top/ctp_2d NetCDF input file")
    parser.add_argument("--output", type=Path, help="Output NetCDF file")
    parser.add_argument("--max-depth-m", type=float, default=3000.0)
    parser.add_argument("--dz-m", type=float, default=120.0)
    parser.add_argument("--zg-name", default=VariableNames.zg)
    parser.add_argument("--ze-name", default=VariableNames.ze)
    parser.add_argument("--cli-name", default=VariableNames.cli)
    parser.add_argument("--clw-name", default=VariableNames.clw)
    parser.add_argument("--ctlev-name", default=VariableNames.ctlev)
    parser.add_argument("--ctmask-name", default=VariableNames.ctmask)
    parser.add_argument("--condensate-threshold", type=float, default=DEFAULT_CONDENSATE_THRESHOLD)
    parser.add_argument("--ctmask-threshold", type=float, default=DEFAULT_CTMASK_THRESHOLD)
    parser.add_argument("--derive-cloud-top", action="store_true", help="Derive cloud top from the highest cli+clw level.")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    single_file_args = [args.atm_3d, args.cloud_top, args.output]
    if all(value is None for value in single_file_args):
        process_dir_from_configs(read_gte_config(), read_msgi_config())
        return

    missing = [name for name in ("atm_3d", "cloud_top", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required single-file arguments: {missing}")

    names = VariableNames(
        zg=args.zg_name,
        ze=args.ze_name,
        cli=args.cli_name,
        clw=args.clw_name,
        ctlev=args.ctlev_name,
        ctmask=args.ctmask_name,
    )
    with xr.open_dataset(args.atm_3d, chunks={}) as atm_3d, xr.open_dataset(args.cloud_top, chunks={}) as ct:
        out = build_output_dataset(
            atm_3d,
            ct,
            names,
            max_depth_m=args.max_depth_m,
            dz_m=args.dz_m,
            condensate_threshold=args.condensate_threshold,
            ctmask_threshold=args.ctmask_threshold,
            derive_cloud_top=args.derive_cloud_top,
        )
        write_output(out, args.output)
    print_diagnostics(out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
