import argparse
import os
from typing import Any

import yaml


REQUIRED_KEYS = {
    "resample_input_dir",
    "twc_threshold",
    "additional_vars",
    "grid_type",
    "model"
}

VALID_RESAMPLING_METHODS = {"laf", "dis"}
VALID_MODES = {"mixed_phase", "cirrus"}


def parse_cmd_args() -> str:
    parser = argparse.ArgumentParser(description="Read MSGI YAML configuration.")
    parser.add_argument(
        "-mcf",
        "--msgi_configuration_filepath",
        required=True,
        help="Path to MSGI YAML config file.",
    )
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.msgi_configuration_filepath):
        raise FileNotFoundError(
            f"Configuration file does not exist: {args.msgi_configuration_filepath}"
        )
    return args.msgi_configuration_filepath


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("MSGI config must be a YAML mapping (dictionary).")

    missing = REQUIRED_KEYS.difference(config.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required MSGI config key(s): {missing_str}")

    mode = config.get("mode", "mixed_phase")
    if not isinstance(mode, str) or mode not in VALID_MODES:
        raise ValueError("'mode' must be one of: mixed_phase, cirrus.")
    config["mode"] = mode

    resample_input_dir = config["resample_input_dir"]
    if not isinstance(resample_input_dir, str) or not resample_input_dir.strip():
        raise ValueError("'resample_input_dir' must be a non-empty string.")

    twc_threshold = config["twc_threshold"]
    if not isinstance(twc_threshold, (int, float)):
        raise ValueError("'twc_threshold' must be a numeric value.")

    additional_vars = config["additional_vars"]
    if not isinstance(additional_vars, list) or not additional_vars:
        raise ValueError("'additional_vars' must be a non-empty list of variable names.")
    if not all(isinstance(v, str) and v.strip() for v in additional_vars):
        raise ValueError("All entries in 'additional_vars' must be non-empty strings.")
    
    grid_type = config["grid_type"]
    allowed_types = ("latlon","geostationary")
    if not (grid_type in allowed_types) or not additional_vars:
        raise ValueError(f"'grid_type' must be one of the following {allowed_types}")

    optional_string_keys = [
        "experiment_name",
        "preprocessing_output_dir",
        "resample_input_dir",
        "resample_output_dir",
        "cdo_aux_dir",
        "dis_weights_file",
    ]
    if grid_type!="geostationary":
        optional_string_keys.extend("cdo_grid_file")
    for key in optional_string_keys:
        if key in config:
            value = config[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{key}' must be a non-empty string.")

    if "resampling_methods" in config:
        methods = config["resampling_methods"]
        if not isinstance(methods, list) or not methods:
            raise ValueError("'resampling_methods' must be a non-empty list.")
        if not all(isinstance(m, str) and m.strip() for m in methods):
            raise ValueError("All entries in 'resampling_methods' must be non-empty strings.")
        invalid = [m for m in methods if m not in VALID_RESAMPLING_METHODS]
        if invalid:
            valid_str = ", ".join(sorted(VALID_RESAMPLING_METHODS))
            invalid_str = ", ".join(invalid)
            raise ValueError(
                f"Invalid 'resampling_methods': {invalid_str}. Allowed values: {valid_str}."
            )

    optional_int_keys = ["cdo_threads", "resample_workers"]
    for key in optional_int_keys:
        if key in config:
            value = config[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'{key}' must be a positive integer.")

    # Keep return type aligned with GTE reader behavior: plain dictionary.
    return config


def read_config(config_fp: str | None = None) -> dict[str, Any]:
    if config_fp is None:
        config_fp = parse_cmd_args()
        print(f"Reading MSGI config at {config_fp}")

    with open(config_fp, "r", encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in MSGI config: {exc}") from exc

    return _validate_config(config)


if __name__ == "__main__":
    print(read_config())
