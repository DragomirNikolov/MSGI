#!/usr/bin/env bash

print_usage() {
    printf 'Usage: %s -c GTE_CONFIG_DIR\n' "$0" >&2
}

GTE_CONFIG_DIR=""

while getopts ':c:' flag; do
    case "$flag" in
        c) GTE_CONFIG_DIR="$OPTARG" ;;
        :)
            printf 'Error: Option -%s requires a value.\n' "$OPTARG" >&2
            print_usage
            exit 1
            ;;
        *)
            printf 'Error: Invalid option: -%s\n' "$OPTARG" >&2
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$GTE_CONFIG_DIR" ]]; then
    printf 'Error: GTE_CONFIG_DIR is required. Use -c GTE_CONFIG_DIR.\n' >&2
    print_usage
    exit 1
fi

python /wolke_scratch/dnikolo/MSGI/msgi/obs/dardar/3d_dardar_tracking.py \
    -c "$GTE_CONFIG_DIR"

python /wolke_scratch/dnikolo/MSGI/msgi/obs/dardar/bulk_statistics.py \
    -c "$GTE_CONFIG_DIR"
