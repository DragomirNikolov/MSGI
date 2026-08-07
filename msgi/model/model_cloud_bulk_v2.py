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
import yaml
from glaciation_time_estimator.auxiliary_func.config_reader import parse_cmd_args as parse_gte_config_path
from glaciation_time_estimator.auxiliary_func.config_reader import read_config as read_gte_config
from msgi.aux import read_config as read_msgi_config
from msgi.aux.config_reader import parse_cmd_args as parse_msgi_config_path


DEFAULT_CONDENSATE_THRESHOLD = 1.0e-5
DEFAULT_CTMASK_THRESHOLD = 1.0e-5


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
    index: np.ndarray
    valid: np.ndarray


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


def as_time_height_ncells(var: xr.DataArray, name: str) -> np.ndarray:
    required_dims = {"time", "height", "ncells"}
    if not required_dims.issubset(var.dims):
        raise ValueError(f"{name!r} must have dimensions including {sorted(required_dims)}")
    return np.asarray(var.transpose("time", "height", "ncells").values, dtype=np.float32)


def as_time_ncells(var: xr.DataArray, name: str) -> np.ndarray:
    required_dims = {"time", "ncells"}
    if not required_dims.issubset(var.dims):
        raise ValueError(f"{name!r} must have dimensions including {sorted(required_dims)}")
    return np.asarray(var.transpose("time", "ncells").values)


def as_zg_array(zg: xr.DataArray) -> np.ndarray:
    if {"height", "ncells"}.issubset(zg.dims) and "time" in zg.dims:
        return np.asarray(zg.transpose("time", "height", "ncells").values, dtype=np.float32)
    if {"height", "ncells"}.issubset(zg.dims):
        return np.asarray(zg.transpose("height", "ncells").values, dtype=np.float32)
    raise ValueError("'zg' must have dimensions (height, ncells) or (time, height, ncells)")


def height_decreases_with_index(zg: np.ndarray) -> bool:
    """Return True when geometric height decreases as the height index increases."""
    if zg.ndim == 3:
        profile = np.nanmedian(zg, axis=(0, 2))
    else:
        profile = np.nanmedian(zg, axis=1)

    diffs = np.diff(profile[np.isfinite(profile)])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("Could not determine vertical orientation from finite zg values")
    return bool(np.nanmedian(diffs) < 0.0)


def _round_if_integer_like(values: np.ndarray, label: str) -> np.ndarray:
    rounded = np.rint(values)
    finite = np.isfinite(values)
    if np.any(finite) and not np.allclose(values[finite], rounded[finite], atol=1.0e-4):
        raise ValueError(f"{label} contains non-integer-like cloud-top levels")
    return rounded.astype(np.float64)


def _zg_at_index(zg: np.ndarray, index: np.ndarray) -> np.ndarray:
    safe = np.where(np.isfinite(index), index, 0).astype(np.int64)
    time_count, cell_count = safe.shape
    cell_idx = np.arange(cell_count, dtype=np.int64)[None, :]

    if zg.ndim == 2:
        return zg[safe, cell_idx]

    time_idx = np.arange(time_count, dtype=np.int64)[:, None]
    return zg[time_idx, safe, cell_idx]


def _depth_from_cloud_top(zg: np.ndarray, cloud_top_height_m: np.ndarray) -> np.ndarray:
    if zg.ndim == 2:
        return cloud_top_height_m[:, None, :] - zg[None, :, :]
    return cloud_top_height_m[:, None, :] - zg


def cloud_top_index_from_one_based_ctlev(
    ctlev: np.ndarray,
    ctmask: np.ndarray,
    source_finite: np.ndarray,
    ctmask_threshold: float,
) -> CloudTopIndex:
    """Convert one-based ICON cloud-top model levels to zero-based positional indices."""
    n_height = source_finite.shape[1]
    ctmask_valid = np.isfinite(ctmask) & (ctmask > ctmask_threshold)
    index_candidate = _round_if_integer_like(np.asarray(ctlev, dtype=np.float64), "ctlev") - 1.0
    valid = (
        np.isfinite(index_candidate)
        & (index_candidate >= 0)
        & (index_candidate < n_height)
        & ctmask_valid
    )
    index = np.where(valid, index_candidate, -1).astype(np.int32)
    return CloudTopIndex(
        index=index,
        valid=valid,
    )


def derive_cloud_top_index_from_condensate(
    cli: np.ndarray,
    clw: np.ndarray,
    ctmask: np.ndarray,
    zg: np.ndarray,
    condensate_threshold: float,
    ctmask_threshold: float,
) -> CloudTopIndex:
    """Derive cloud top as the highest geometric level with enough condensate."""
    condensate = cli + clw
    cloudy = compute_cloudy(cli,clw,ctmask,condensate_threshold, ctmask_threshold)
    valid = np.any(cloudy, axis=1)
    if zg.ndim == 2:
        zg_for_argmax = np.broadcast_to(zg[None, :, :], cloudy.shape)
    else:
        zg_for_argmax = zg
    candidate_heights = np.where(cloudy & np.isfinite(zg_for_argmax), zg_for_argmax, -np.inf)
    index = np.argmax(candidate_heights, axis=1).astype(np.int32)
    index[~valid] = -1

    return CloudTopIndex(
        index=index,
        valid=valid,
    )

def compute_cloudy(
        cli: np.ndarray,
        clw: np.ndarray, 
        ctmask: np.ndarray, 
        condensate_threshold: float,
        ctmask_threshold: float
    ):
    condensate = cli + clw
    return (
        np.isfinite(cli)
        & np.isfinite(clw)
        & np.isfinite(condensate)
        & (condensate >= condensate_threshold)
        & np.isfinite(ctmask)[:, None, :]
        & (ctmask[:, None, :] > ctmask_threshold)
    )

def compute_cph(
    cli: np.ndarray,
    clw: np.ndarray,
    ctmask: np.ndarray,
    condensate_threshold: float,
    ctmask_threshold: float,
) -> np.ndarray:
    condensate = cli + clw
    valid = compute_cloudy(cli,clw,ctmask,condensate_threshold, ctmask_threshold)
    cph = np.full(cli.shape, np.nan, dtype=np.float32)
    cph[valid] = (1.0 + cli[valid] / condensate[valid]).astype(np.float32)
    return cph, valid


# def compute_layer_mean(
#     var: np.ndarray,
#     zg: np.ndarray,
#     cloud_top_height_m: np.ndarray,
#     valid_cloud_top: np.ndarray,
#     bulk_offsets_m: np.ndarray,
#     cli: np.ndarray,
#     clw: np.ndarray,
#     condensate_threshold: float = DEFAULT_CONDENSATE_THRESHOLD
# ) -> tuple[np.ndarray, np.ndarray]:
#     time_count, _, cell_count = var.shape
#     layer_count = bulk_offsets_m.shape[0]
#     mean = np.full((time_count, layer_count, cell_count), np.nan, dtype=np.float32)
#     is_cloudy = np.full((time_count, layer_count, cell_count), np.nan, dtype=np.float32)
#     n_valid = np.zeros((time_count, layer_count, cell_count), dtype=np.int16)
#     depth = _depth_from_cloud_top(zg, cloud_top_height_m)

#     cloudy_mask = (
#         valid_cloud_top[:, None, :] &
#         np.isfinite(cli) &
#         np.isfinite(clw) &
#         ((cli+clw)>=condensate_threshold))
#     for layer, (top_offset_m, bottom_offset_m) in enumerate(bulk_offsets_m):
#         upper_depth_m = abs(float(top_offset_m))
#         lower_depth_m = abs(float(bottom_offset_m))
#         layer_mask = (
#             valid_cloud_top[:, None, :]
#             & np.isfinite(var)
#             & np.isfinite(depth)
#             & (depth >= upper_depth_m)
#             & (depth < lower_depth_m)
#         )
#         count = np.count_nonzero(layer_mask, axis=1).astype(np.int16)
#         total = np.where(layer_mask, var, 0.0).sum(axis=1, dtype=np.float64)
#         valid = count > 0
#         mean[:, layer, :][valid] = (total[valid] / count[valid]).astype(np.float32)
#         n_valid[:, layer, :] = count
    
#     cloudy_mask 
#     return mean, n_valid
def compute_interpolated_layers(
    var: np.ndarray,
    zg: np.ndarray,
    cloud_top_height_m: np.ndarray,
    valid_cloud_top: np.ndarray,
    bulk_offsets_m: np.ndarray,
    is_cloudy: np.ndarray,
    null_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_count, _, cell_count = var.shape
    layer_count = bulk_offsets_m.shape[0]
    
    mean = np.full((time_count, layer_count, cell_count), np.nan, dtype=np.float32)
    n_valid = np.zeros((time_count, layer_count, cell_count), dtype=np.int16)
    depth = _depth_from_cloud_top(zg, cloud_top_height_m)

    
    # cloudy_mask = (
    #     valid_cloud_top[:, None, :] &
    #     np.isfinite(cli) &
    #     np.isfinite(clw) &
    #     ((cli+clw)>=condensate_threshold))
    
    top_depth_out = abs(bulk_offsets_m[:,0])
    bottom_depth_out = abs(bulk_offsets_m[:,1])
    mid_depth_out = (top_depth_out + bottom_depth_out)/2
    max_depth = bottom_depth_out.max()
    
    upper_layer_mask = layer_mask = (
            valid_cloud_top
            & np.isfinite(var[:,0,:])
            & np.isfinite(depth[:,0,:])
            & (depth[:,0,:] >=0)
            & (depth[:,0,:] <=max_depth)
            & is_cloudy[:,0,:]
        )
    for model_layer in range(1,zg.shape[0]):
        layer_mask = (
            valid_cloud_top
            & np.isfinite(var[:,model_layer,:])
            & np.isfinite(depth[:,model_layer,:])
            & (depth[:,model_layer,:] >= 0)
            & is_cloudy[:,model_layer,:]
        )
        
        two_valid_lev_mask = upper_layer_mask & layer_mask
        only_upper_mask= upper_layer_mask & (~layer_mask)
        only_lower_mask = (~upper_layer_mask) & layer_mask
        any_layer_mask = upper_layer_mask | layer_mask
        upper_depth =  depth[:,model_layer-1,:]
        current_depth = depth[:,model_layer,:]
        # lower_depth = depth[:,model_layer+1,:]
        # layer_top_depth = current_depth - (current_depth - upper_depth)/2
        # layer_bottom_depth = current_depth + (lower_depth - current_depth)/2
        corresponding_ind = (mid_depth_out[None,:,None] >= upper_depth[:,None,:]) & (mid_depth_out[None,:,None] < current_depth[:,None,:])
        # print(corresponding_ind.shape)
        # print(depth.shape,
        #   var.shape,
        #   mid_depth_out.shape,
        #   layer_top_depth.shape,
        #   layer_mask.shape)
        current_level_val = var[:, model_layer, :]
        upper_level_val = var[:, model_layer-1, :]
        lev_diff=np.where(only_upper_mask,2*(null_target-upper_level_val),np.nan)
        lev_diff=np.where(only_lower_mask,2*(current_level_val-null_target),lev_diff)
        lev_diff = np.where(two_valid_lev_mask, current_level_val-upper_level_val,lev_diff)
        
        bulk_lev_multiplier = np.where(corresponding_ind, (mid_depth_out[None, :, None]- upper_depth[:,None,:])/(current_depth[:,None,:]-upper_depth[:,None,:]),np.nan)
        bulk_lev_multiplier = np.where(only_upper_mask[:,None,:] & (bulk_lev_multiplier>0.5),np.nan,bulk_lev_multiplier)
        bulk_lev_multiplier = np.where(only_lower_mask[:,None,:] & (bulk_lev_multiplier<0.5),np.nan,bulk_lev_multiplier)
        mean = np.where(corresponding_ind & upper_layer_mask[:,None,:],upper_level_val[:,None,:] + bulk_lev_multiplier*lev_diff[:,None,:],mean)
        mean = np.where(corresponding_ind & only_lower_mask[:,None,:], current_level_val[:,None,:] - (1-bulk_lev_multiplier)*lev_diff[:,None,:],mean)

        both_valid_out = (
            corresponding_ind
            & two_valid_lev_mask[:, None, :]
        )

        upper_only_out = (
            corresponding_ind
            & only_upper_mask[:, None, :]
            & (bulk_lev_multiplier <= 0.5)
        )

        lower_only_out = (
            corresponding_ind
            & only_lower_mask[:, None, :]
            & (bulk_lev_multiplier >= 0.5)
        )

        n_valid = np.where(
            both_valid_out,
            2,
            n_valid,
        )

        n_valid = np.where(
            upper_only_out | lower_only_out,
            1,
            n_valid,
        )

        upper_layer_mask = layer_mask 
                            # (depth[:,model_layer,:] <=max_depth))
        
        

    return mean, n_valid





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
    zg = as_zg_array(atm_3d[names.zg])
    ctmask = as_time_ncells(ct[names.ctmask], names.ctmask).astype(np.float32)
    ctlev = None if derive_cloud_top else as_time_ncells(ct[names.ctlev], names.ctlev)

    if ze.shape != cli.shape or ze.shape != clw.shape:
        raise ValueError("ze, cli, and clw must have the same (time, height, ncells) shape")
    if ctlev is not None and ctlev.shape != (ze.shape[0], ze.shape[2]):
        raise ValueError("ctlev must have shape (time, ncells) matching atm_3d")
    if ctmask.shape != (ze.shape[0], ze.shape[2]):
        raise ValueError("ctmask must have shape (time, ncells) matching atm_3d")
    if zg.ndim == 2 and zg.shape != (ze.shape[1], ze.shape[2]):
        raise ValueError("2D zg must have shape (height, ncells)")
    if zg.ndim == 3 and zg.shape != ze.shape:
        raise ValueError("3D zg must have shape (time, height, ncells)")

    cph, is_cloudy = compute_cph(cli, clw, ctmask, condensate_threshold, ctmask_threshold)
    source_finite = np.isfinite(ze) | np.isfinite(cph)
    bulk_offsets_m = build_bulk_level_offsets(max_depth_m, dz_m)

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
        if ctlev is None:
            raise RuntimeError("Internal error: ctlev is required when derive_cloud_top is disabled")
        cloud_top = cloud_top_index_from_one_based_ctlev(
            ctlev=ctlev,
            ctmask=ctmask,
            source_finite=source_finite,
            ctmask_threshold=ctmask_threshold,
        )
    safe_index = np.where(cloud_top.valid, cloud_top.index, 0)
    cloud_top_height_m = _zg_at_index(zg, safe_index).astype(np.float32)
    cloud_top_height_m[~cloud_top.valid] = np.nan

    ze_mean, ze_n_valid = compute_interpolated_layers(
        ze,
        zg,
        cloud_top_height_m,
        cloud_top.valid,
        bulk_offsets_m,
        is_cloudy,
        null_target=-30,
    )
    cph_mean, cph_n_valid = compute_interpolated_layers(
        cph,
        zg,
        cloud_top_height_m,
        cloud_top.valid,
        bulk_offsets_m,
        is_cloudy,
        null_target=0,
    )

    coords: dict[str, xr.DataArray | tuple[str, np.ndarray]] = {
        "time": atm_3d["time"],
        "bulk_level": ("bulk_level", np.arange(bulk_offsets_m.shape[0], dtype=np.int16)),
    }
    if "ncells" in atm_3d.coords:
        coords["ncells"] = atm_3d["ncells"]
    else:
        coords["ncells"] = ("ncells", np.arange(ze.shape[2], dtype=np.int32))

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray] | xr.DataArray] = {
        "ze": (("time", "bulk_level", "ncells"), ze_mean),
        "cph": (("time", "bulk_level", "ncells"), cph_mean),
        "ze_n_valid_levels": (("time", "bulk_level", "ncells"), ze_n_valid),
        "cph_n_valid_levels": (("time", "bulk_level", "ncells"), cph_n_valid),
        "cloud_top_level_index": (("time", "ncells"), cloud_top.index.astype(np.int16)),
        "cloud_top_height_m": (("time", "ncells"), cloud_top_height_m),
        "bulk_level_offset_top": (("bulk_level",), bulk_offsets_m[:, 0]),
        "bulk_level_offset_bottom": (("bulk_level",), bulk_offsets_m[:, 1]),
        "bulk_level_offset_mid": (("bulk_level",), 0.5 * (bulk_offsets_m[:, 0] + bulk_offsets_m[:, 1])),
    }
    for coord_name in ("clon", "clat"):
        if coord_name in atm_3d:
            data_vars[coord_name] = atm_3d[coord_name]

    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out["ze"].attrs.update(atm_3d[names.ze].attrs)
    out["ze"].attrs.setdefault("long_name", "Cloud-bulk radar reflectivity")
    out["cph"].attrs.update(
        {
            "long_name": "Cloud-bulk phase score",
            "valid_min": np.float32(1.0),
            "valid_max": np.float32(2.0),
            "comment": "Layer mean of 1 + cli / (cli + clw), computed only where condensate and ctmask are valid.",
        }
    )
    out["ze_n_valid_levels"].attrs["long_name"] = "Number of finite source levels used for ze layer mean"
    out["cph_n_valid_levels"].attrs["long_name"] = "Number of finite source levels used for cph layer mean"
    out["cloud_top_level_index"].attrs.update(
        {
            "long_name": "Zero-based positional cloud-top level index",
            "comment": "-1 marks invalid cloud tops.",
        }
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
    valid_ct = np.asarray(ds["cloud_top_level_index"].values) >= 0
    print(ds)
    print(f"zg decreases with height index: {ds.attrs.get('zg_decreases_with_height_index')}")
    print(f"valid cloud-top fraction: {valid_ct.mean():.6f}")
    for name in ("ze", "cph"):
        values = np.asarray(ds[name].values)
        print(f"{name} finite fraction: {np.isfinite(values).mean():.6f}")
        print(f"{name} all NaN: {bool(np.isnan(values).all())}")


def write_output(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        if file_dt is None:
            continue
        if file_dt < start_date or file_dt > end_date:
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
    with xr.open_dataset(atm_path) as atm_3d, xr.open_dataset(ct_path) as ct:
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
        raise ValueError("model_cloud_bulk_fixed.py currently supports only ICON inputs")

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
    workers = min(int(gte_config.get("n_preproc_cores"))*int(gte_config.get("n_preproc_threads")),16, len(tasks), os.cpu_count())
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


def _synthetic_datasets(ctlev_values: np.ndarray, height_coord: np.ndarray) -> tuple[xr.Dataset, xr.Dataset]:
    time = np.arange(ctlev_values.shape[0], dtype=np.int32)
    ncells = np.arange(ctlev_values.shape[1], dtype=np.int32)
    n_height = height_coord.size

    zg_profile = np.array([500.0, 400.0, 300.0, 200.0, 100.0, 0.0], dtype=np.float32)
    if n_height != zg_profile.size:
        raise ValueError("Synthetic test expects exactly 6 height levels")
    zg = np.broadcast_to(zg_profile[:, None], (n_height, ncells.size)).copy()

    base = np.arange(time.size * n_height * ncells.size, dtype=np.float32).reshape(time.size, n_height, ncells.size)
    ze = base + 1.0
    cli = np.full_like(ze, 2.0e-5)
    clw = np.full_like(ze, 2.0e-5)
    ctmask = np.ones((time.size, ncells.size), dtype=np.float32)
    ctmask[-1, -1] = 0.0

    atm = xr.Dataset(
        data_vars={
            "zg": (("height", "ncells"), zg),
            "cloudsat_ze_mean": (("time", "height", "ncells"), ze),
            "cli": (("time", "height", "ncells"), cli),
            "clw": (("time", "height", "ncells"), clw),
        },
        coords={
            "time": ("time", time),
            "height": ("height", height_coord.astype(np.float64)),
            "ncells": ("ncells", ncells),
            "clon": ("ncells", np.linspace(0.0, 1.0, ncells.size, dtype=np.float32)),
            "clat": ("ncells", np.linspace(45.0, 46.0, ncells.size, dtype=np.float32)),
        },
    )
    ct = xr.Dataset(
        data_vars={
            "ctlev": (("time", "ncells"), ctlev_values.astype(np.float32)),
            "ctmask": (("time", "ncells"), ctmask),
        },
        coords={"time": ("time", time), "ncells": ("ncells", ncells)},
    )
    return atm, ct


def run_self_test() -> None:
    atm, ct = _synthetic_datasets(
        np.array([[1, 3, 5], [3, 4, 5]], dtype=np.float32),
        np.arange(10, 70, 10),
    )
    out = build_output_dataset(atm, ct, VariableNames(), max_depth_m=300.0, dz_m=100.0)
    if bool(np.isnan(out["ze"].isel(time=0, bulk_level=0, ncells=1))):
        raise AssertionError("Expected a finite ze layer mean for a valid cloudy synthetic column")
    if bool(np.isfinite(out["cph"].isel(time=-1, ncells=-1)).any()):
        raise AssertionError("Expected invalid ctmask column to remain NaN for cph")
    if int(out["ze_n_valid_levels"].isel(time=-1, bulk_level=0, ncells=-1)) != 0:
        raise AssertionError("Expected invalid ctmask column to have zero valid ze levels")

    atm, ct = _synthetic_datasets(
        np.array([[1, 3, 5], [3, 4, 5]], dtype=np.float32),
        np.arange(10, 70, 10),
    )
    out = build_output_dataset(atm, ct, VariableNames(), max_depth_m=350.0, dz_m=50.0)
    if not bool(np.isnan(out["ze"].isel(time=0, bulk_level=1, ncells=1))):
        raise AssertionError("Expected a layer with no source levels to remain NaN")

    atm, ct = _synthetic_datasets(
        np.zeros((2, 3), dtype=np.float32),
        np.arange(10, 70, 10),
    )
    cli = atm["cli"].values.copy()
    clw = atm["clw"].values.copy()
    cli[:] = 0.0
    clw[:] = 0.0
    cli[:, 2:5, :] = 2.0e-5
    clw[:, 2:5, :] = 2.0e-5
    atm["cli"] = (atm["cli"].dims, cli)
    atm["clw"] = (atm["clw"].dims, clw)
    out = build_output_dataset(
        atm,
        ct,
        VariableNames(),
        max_depth_m=300.0,
        dz_m=100.0,
        derive_cloud_top=True,
    )
    if not np.all(out["cloud_top_level_index"].isel(time=0).values[:2] == 2):
        raise AssertionError("Expected derived cloud top at first cloudy model level")
    if not bool(np.isfinite(out["ze"].isel(time=0, bulk_level=0, ncells=0))):
        raise AssertionError("Expected finite ze from derived cloud-top path")

    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ICON cloud-top-relative bulk layer means for Ze and CPH. "
            "Pass --atm-3d/--cloud-top/--output for one file, or pass -cf and -mcf "
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
    parser.add_argument(
        "--derive-cloud-top",
        action="store_true",
        help="Derive cloud top from the highest cli+clw level instead of using ctlev.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic tests and exit")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    single_file_args = [args.atm_3d, args.cloud_top, args.output]
    if all(value is None for value in single_file_args):
        gte_config = read_gte_config()
        msgi_config = read_msgi_config()
        process_dir_from_configs(gte_config, msgi_config)
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
    with xr.open_dataset(args.atm_3d) as atm_3d, xr.open_dataset(args.cloud_top) as ct:
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
