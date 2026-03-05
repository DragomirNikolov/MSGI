#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage:
#   ./merge_ddct.sh <input_dir> [output_dir] [threads]
#
# Example:
#   ./merge_ddct.sh /wolke_scratch/dnikolo/MSGI/DARDAR_analysis/daily_output/2008/05/01 merged_ddct 8

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_dir> [output_dir] [threads]"
  exit 1
fi

INDIR="$1"
OUTDIR="${2:-merged_ddct}"
P="${3:-8}"

# Validate input dir
if [[ ! -d "$INDIR" ]]; then
  echo "Error: input directory does not exist: $INDIR"
  exit 1
fi

# Canonicalize paths
INDIR="$(cd "$INDIR" && pwd)"
if [[ "$OUTDIR" = /* ]]; then
  mkdir -p "$OUTDIR"
else
  mkdir -p "$INDIR/$OUTDIR"
  OUTDIR="$INDIR/$OUTDIR"
fi

# Collect unique keys from files in INDIR:
# DD_CT_YYYYMMDD_HHMM_<n>.nc  -> key DD_CT_YYYYMMDD_HHMM
keys=$(
  printf "%s\n" "$INDIR"/DD_CT_*.nc \
  | xargs -r -n1 basename \
  | sed -nE 's@^(DD_CT_[0-9]{8}_[0-9]{4})_[0-9]+\.nc$@\1@p' \
  | sort -u
)

if [[ -z "${keys:-}" ]]; then
  echo "No matching files found in $INDIR (expected DD_CT_YYYYMMDD_HHMM_<n>.nc)"
  exit 0
fi

while IFS= read -r key; do
  [[ -z "$key" ]] && continue

  files=( "$INDIR"/${key}_[0-9]*.nc )
  n=${#files[@]}
  out="$OUTDIR/${key}.nc"

  if (( n == 0 )); then
    continue
  elif (( n == 1 )); then
    cp -f "${files[0]}" "$out"
    echo "[copy ] $(basename "${files[0]}") -> $out"
  else
    # Average duplicates cell-by-cell, variable-by-variable, ignoring missings
    cdo -L ensmean "${files[@]}" "$out"
    echo "[merge] $n files -> $out"
    
  fi
done <<< "$keys"

echo "Done. Outputs in: $OUTDIR"