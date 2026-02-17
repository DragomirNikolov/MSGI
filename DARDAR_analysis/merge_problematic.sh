#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Threads for CDO
P=8

# Optional: keep outputs separate
OUTDIR="merged_ddct"
mkdir -p "$OUTDIR"

# Collect unique timestamp keys: DD_CT_YYYYMMDD_HHMM
# Example filename: DD_CT_20080501_0330_2.nc
keys=$(printf "%s\n" DD_CT_*.nc \
  | sed -E 's@^(DD_CT_[0-9]{8}_[0-9]{4})_[0-9]+\.nc$@\1@' \
  | sort -u)

while IFS= read -r key; do
  # all files for this timestamp
  files=( ${key}_[0-9]*.nc )
  n=${#files[@]}
  out="${OUTDIR}/${key}.nc"

  if (( n == 0 )); then
    continue
  elif (( n == 1 )); then
    # No duplicate: just copy/rename without suffix
    cp -f "${files[0]}" "$out"
    echo "[copy ] ${files[0]} -> $out"
  else
    # Duplicates: average per grid cell and variable, ignoring missings
    # (exactly your fill/val logic)
    cdo -L ensmean "${files[@]}" "$out"
    echo "[merge] $n files -> $out"
  fi
done <<< "$keys"

echo "Done. Outputs in: $OUTDIR"