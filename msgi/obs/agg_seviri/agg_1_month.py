import os
import re
import glob
from pathlib import Path
from collections import defaultdict

import cis  # Community Intercomparison Suit
# os.environ["CIS_PLUGIN_HOME"] = "/wolke_scratch/dnikolo/MSGI"
# # --------- CONFIG ----------
# search_root = "/wolke_scratch/dnikolo/CLAAS_Data/np/2008/01"
# dump_dir = Path("/wolke_scratch/dnikolo/dump/regrided_data/out_folder")
# dump_dir.mkdir(parents=True, exist_ok=True)

# # Recursively find all January 2008 inputs
# pattern = str(Path(search_root) / "**" / "CPPin200801*.nc")
# files = sorted(glob.glob(pattern, recursive=True))

# # Regex to capture the timestamp after 'CPPin'
# # Example filename: CPPin20080101000000405SVMSG01MD.nc
# # We'll grab the first 14 digits as YYYYMMDDHHMMSS and derive YYYYMMDDHH (the hour)
# ts_re = re.compile(r"CPPin(\d{14})")

# # Group files by hour key 'YYYYMMDDHH'
# by_hour = defaultdict(list)
# for f in files:
#     m = ts_re.search(os.path.basename(f))
#     if not m:
#         # Skip any file that doesn't match the expected pattern
#         continue
#     yyyymmddhhmmss = m.group(1)
#     hour_key = yyyymmddhhmmss[:10]  # YYYYMMDDHH
#     by_hour[hour_key].append(f)

# print(f"Found {len(files)} files; {len(by_hour)} hourly groups.")

# def save_agg(agg, out_path: Path):
#     """
#     Try a few common save methods depending on CIS/xarray version.
#     """
#     # 1) xarray-like
#     if hasattr(agg, "to_netcdf"):
#         agg.to_netcdf(out_path)
#         return
#     # 2) CIS writer
#     try:
#         from cis.io import write_data
#         write_data(agg, str(out_path))
#         return
#     except Exception:
#         pass
#     # 3) Generic .save()
#     if hasattr(agg, "save"):
#         agg.save(str(out_path))
#         return
#     raise RuntimeError(
#         "Unable to save aggregated object (no to_netcdf / cis.io.write_data / .save available)."
#     )

# # Process each hour
# for hour_key, hour_files in sorted(by_hour.items()):
#     # Compose a readable output name per hour, preserving date+hour
#     # e.g., AGG_CPP_20080101_00.nc
#     out_name = f"AGG_CPP_{hour_key[:8]}_{hour_key[8:10]}.nc"
#     out_path = dump_dir / out_name

#     # Skip if already done
#     if out_path.exists():
#         print(f"Skip existing: {out_path}")
#         continue

#     try:
#         print(f"Aggregating {len(hour_files)} files for hour {hour_key} -> {out_name}")

#         # Read all files for the hour
#         data = cis.read_data(hour_files, 'cph', product="SEVIRI_L2_CLAAS")

#         # Aggregate to the target grid
#         agg = data.aggregate(
#             x=slice(-80, 80, 1),
#             y=slice(-80, 80, 1),
#             how='mean'
#         )

#         # Save
#         save_agg(agg, out_path)
#         print(f"Wrote: {out_path}")

#     except Exception as e:
#         print(f"Failed on hour {hour_key} ({len(hour_files)} files): {e}")
