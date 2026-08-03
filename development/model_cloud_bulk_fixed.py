from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


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
    convention: str
    score: int


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


def _candidate_from_height_labels(ctlev: np.ndarray, height_coord: np.ndarray) -> np.ndarray | None:
    finite_height = np.isfinite(height_coord)
    if not np.any(finite_height):
        return None

    labels = height_coord[finite_height]
    positions = np.flatnonzero(finite_height).astype(np.int32)
    order = np.argsort(labels)
    labels = labels[order]
    positions = positions[order]

    finite_ctlev = np.isfinite(ctlev)
    out = np.full(ctlev.shape, np.nan, dtype=np.float64)
    if not np.any(finite_ctlev):
        return out

    raw = ctlev[finite_ctlev]
    insertion = np.searchsorted(labels, raw)
    matched = insertion < labels.size
    if np.any(matched):
        raw_matched = raw[matched]
        label_matched = labels[insertion[matched]]
        matched[matched] = np.isclose(raw_matched, label_matched, atol=1.0e-4, rtol=0.0)

    flat_out = out[finite_ctlev]
    flat_out[matched] = positions[insertion[matched]]
    out[finite_ctlev] = flat_out
    return out


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


def _score_cloud_top_candidate(
    index_candidate: np.ndarray,
    ctmask_valid: np.ndarray,
    zg: np.ndarray,
    source_finite: np.ndarray,
    max_depth_m: float,
) -> tuple[int, np.ndarray]:
    n_height = source_finite.shape[1]
    in_bounds = (
        np.isfinite(index_candidate)
        & (index_candidate >= 0)
        & (index_candidate < n_height)
        & ctmask_valid
    )
    safe_index = np.where(in_bounds, index_candidate, 0).astype(np.int64)
    cloud_top_height_m = _zg_at_index(zg, safe_index)
    depth = _depth_from_cloud_top(zg, cloud_top_height_m)
    usable = (
        in_bounds[:, None, :]
        & source_finite
        & np.isfinite(depth)
        & (depth >= 0.0)
        & (depth < max_depth_m)
    )
    return int(np.count_nonzero(usable)), in_bounds


def detect_cloud_top_index(
    ctlev: np.ndarray,
    ctmask: np.ndarray,
    height_coord: np.ndarray,
    zg: np.ndarray,
    source_finite: np.ndarray,
    max_depth_m: float,
    ctmask_threshold: float,
) -> CloudTopIndex:
    """Choose the cloud-top index convention that leaves the most usable levels."""
    n_height = source_finite.shape[1]
    ctmask_valid = np.isfinite(ctmask) & (ctmask > ctmask_threshold)
    ctlev_float = np.asarray(ctlev, dtype=np.float64)
    rounded_ctlev = _round_if_integer_like(ctlev_float, "ctlev")

    candidates: list[tuple[str, np.ndarray]] = []
    label_candidate = _candidate_from_height_labels(ctlev_float, np.asarray(height_coord, dtype=np.float64))
    if label_candidate is not None:
        candidates.append(("height_coordinate_labels", label_candidate))
    candidates.extend(
        [
            ("zero_based_positional_index", rounded_ctlev),
            ("one_based_model_level", rounded_ctlev - 1.0),
        ]
    )

    scored: list[CloudTopIndex] = []
    seen: set[bytes] = set()
    for convention, candidate in candidates:
        normalized = np.where(np.isfinite(candidate), candidate, np.nan)
        key = np.nan_to_num(normalized, nan=-999999.0).astype(np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)

        score, valid = _score_cloud_top_candidate(
            normalized,
            ctmask_valid,
            zg,
            source_finite,
            max_depth_m,
        )
        index = np.where(valid, normalized, -1).astype(np.int32)
        scored.append(CloudTopIndex(index=index, valid=valid, convention=convention, score=score))

    best = max(scored, key=lambda item: item.score)
    if best.score <= 0:
        details = ", ".join(f"{item.convention}={item.score}" for item in scored)
        raise ValueError(
            "No cloud-top convention produced usable finite levels below cloud top. "
            f"Candidate scores: {details}"
        )

    if np.any((best.index[best.valid] < 0) | (best.index[best.valid] >= n_height)):
        raise RuntimeError("Internal error: selected cloud-top indices are out of bounds")
    return best


def compute_cph(
    cli: np.ndarray,
    clw: np.ndarray,
    ctmask: np.ndarray,
    condensate_threshold: float,
    ctmask_threshold: float,
) -> np.ndarray:
    condensate = cli + clw
    valid = (
        np.isfinite(cli)
        & np.isfinite(clw)
        & np.isfinite(condensate)
        & (condensate >= condensate_threshold)
        & np.isfinite(ctmask)[:, None, :]
        & (ctmask[:, None, :] > ctmask_threshold)
    )
    cph = np.full(cli.shape, np.nan, dtype=np.float32)
    cph[valid] = (1.0 + cli[valid] / condensate[valid]).astype(np.float32)
    return cph


def compute_layer_mean(
    var: np.ndarray,
    zg: np.ndarray,
    cloud_top_height_m: np.ndarray,
    valid_cloud_top: np.ndarray,
    bulk_offsets_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_count, _, cell_count = var.shape
    layer_count = bulk_offsets_m.shape[0]
    mean = np.full((time_count, layer_count, cell_count), np.nan, dtype=np.float32)
    n_valid = np.zeros((time_count, layer_count, cell_count), dtype=np.int16)
    depth = _depth_from_cloud_top(zg, cloud_top_height_m)

    for layer, (top_offset_m, bottom_offset_m) in enumerate(bulk_offsets_m):
        upper_depth_m = abs(float(top_offset_m))
        lower_depth_m = abs(float(bottom_offset_m))
        layer_mask = (
            valid_cloud_top[:, None, :]
            & np.isfinite(var)
            & np.isfinite(depth)
            & (depth >= upper_depth_m)
            & (depth < lower_depth_m)
        )
        count = np.count_nonzero(layer_mask, axis=1).astype(np.int16)
        total = np.where(layer_mask, var, 0.0).sum(axis=1, dtype=np.float64)
        valid = count > 0
        mean[:, layer, :][valid] = (total[valid] / count[valid]).astype(np.float32)
        n_valid[:, layer, :] = count

    return mean, n_valid


def build_output_dataset(
    atm_3d: xr.Dataset,
    ct: xr.Dataset,
    names: VariableNames,
    max_depth_m: float,
    dz_m: float,
    condensate_threshold: float = DEFAULT_CONDENSATE_THRESHOLD,
    ctmask_threshold: float = DEFAULT_CTMASK_THRESHOLD,
) -> xr.Dataset:
    require_variables(atm_3d, [names.zg, names.ze, names.cli, names.clw, "time"], "atm_3d")
    require_variables(ct, [names.ctlev, names.ctmask], "cloud-top dataset")

    ze = as_time_height_ncells(atm_3d[names.ze], names.ze)
    cli = as_time_height_ncells(atm_3d[names.cli], names.cli)
    clw = as_time_height_ncells(atm_3d[names.clw], names.clw)
    zg = as_zg_array(atm_3d[names.zg])
    ctlev = as_time_ncells(ct[names.ctlev], names.ctlev)
    ctmask = as_time_ncells(ct[names.ctmask], names.ctmask).astype(np.float32)

    if ze.shape != cli.shape or ze.shape != clw.shape:
        raise ValueError("ze, cli, and clw must have the same (time, height, ncells) shape")
    if ctlev.shape != (ze.shape[0], ze.shape[2]):
        raise ValueError("ctlev must have shape (time, ncells) matching atm_3d")
    if ctmask.shape != ctlev.shape:
        raise ValueError("ctmask must have shape (time, ncells) matching ctlev")
    if zg.ndim == 2 and zg.shape != (ze.shape[1], ze.shape[2]):
        raise ValueError("2D zg must have shape (height, ncells)")
    if zg.ndim == 3 and zg.shape != ze.shape:
        raise ValueError("3D zg must have shape (time, height, ncells)")

    cph = compute_cph(cli, clw, ctmask, condensate_threshold, ctmask_threshold)
    source_finite = np.isfinite(ze) | np.isfinite(cph)
    bulk_offsets_m = build_bulk_level_offsets(max_depth_m, dz_m)
    height_coord = np.asarray(atm_3d["height"].values, dtype=np.float64)

    cloud_top = detect_cloud_top_index(
        ctlev=ctlev,
        ctmask=ctmask,
        height_coord=height_coord,
        zg=zg,
        source_finite=source_finite,
        max_depth_m=max_depth_m,
        ctmask_threshold=ctmask_threshold,
    )
    safe_index = np.where(cloud_top.valid, cloud_top.index, 0)
    cloud_top_height_m = _zg_at_index(zg, safe_index).astype(np.float32)
    cloud_top_height_m[~cloud_top.valid] = np.nan

    ze_mean, ze_n_valid = compute_layer_mean(
        ze,
        zg,
        cloud_top_height_m,
        cloud_top.valid,
        bulk_offsets_m,
    )
    cph_mean, cph_n_valid = compute_layer_mean(
        cph,
        zg,
        cloud_top_height_m,
        cloud_top.valid,
        bulk_offsets_m,
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
            "comment": f"Derived from {names.ctlev} using convention {cloud_top.convention}. -1 marks invalid cloud tops.",
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
            "ctlev_convention": cloud_top.convention,
            "ctlev_candidate_score": str(cloud_top.score),
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
    print(f"ctlev convention: {ds.attrs.get('ctlev_convention')}")
    print(f"zg decreases with height index: {ds.attrs.get('zg_decreases_with_height_index')}")
    print(f"valid cloud-top fraction: {valid_ct.mean():.6f}")
    for name in ("ze", "cph"):
        values = np.asarray(ds[name].values)
        print(f"{name} finite fraction: {np.isfinite(values).mean():.6f}")
        print(f"{name} all NaN: {bool(np.isnan(values).all())}")


def write_output(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, format="NETCDF4", encoding=netcdf_encoding(ds))


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
    cases = [
        ("zero_based_positional_index", np.array([[0, 2, 4], [2, 3, 4]], dtype=np.float32), np.arange(10, 70, 10)),
        ("one_based_model_level", np.array([[1, 3, 6], [3, 4, 6]], dtype=np.float32), np.arange(10, 70, 10)),
        ("height_coordinate_labels", np.array([[10, 30, 60], [30, 40, 60]], dtype=np.float32), np.arange(10, 70, 10)),
    ]

    for expected_convention, ctlev, height_coord in cases:
        atm, ct = _synthetic_datasets(ctlev, height_coord)
        out = build_output_dataset(atm, ct, VariableNames(), max_depth_m=300.0, dz_m=100.0)
        convention = out.attrs["ctlev_convention"]
        if convention != expected_convention:
            raise AssertionError(f"Expected {expected_convention}, got {convention}")
        if bool(np.isnan(out["ze"].isel(time=0, bulk_level=0, ncells=1))):
            raise AssertionError("Expected a finite ze layer mean for a valid cloudy synthetic column")
        if bool(np.isfinite(out["cph"].isel(time=-1, ncells=-1)).any()):
            raise AssertionError("Expected invalid ctmask column to remain NaN for cph")
        if int(out["ze_n_valid_levels"].isel(time=-1, bulk_level=0, ncells=-1)) != 0:
            raise AssertionError("Expected invalid ctmask column to have zero valid ze levels")

    atm, ct = _synthetic_datasets(
        np.array([[0, 2, 4], [2, 3, 4]], dtype=np.float32),
        np.arange(10, 70, 10),
    )
    out = build_output_dataset(atm, ct, VariableNames(), max_depth_m=350.0, dz_m=50.0)
    if not bool(np.isnan(out["ze"].isel(time=0, bulk_level=1, ncells=1))):
        raise AssertionError("Expected a layer with no source levels to remain NaN")

    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ICON cloud-top-relative bulk layer means for Ze and CPH."
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
    parser.add_argument("--self-test", action="store_true", help="Run synthetic tests and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    missing = [name for name in ("atm_3d", "cloud_top", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments unless --self-test is used: {missing}")

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
        )

    write_output(out, args.output)
    print_diagnostics(out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
