import glob
import os
import subprocess

# --- User settings ---
in_dir = "/wolke_scratch/rvella/shared/dragomir/test01"   # folder with input 3D files
out_dir = "/wolke_scratch/dnikolo/ICON_Data/3d_sample"                                         # you can change this if you want
pattern = "test01_atm_3d_ml_*.nc"                        # pattern for 3D files

grid_file = "/wolke_scratch/dnikolo/ICON_Data/grid_1x1.txt"
weights_file = "/wolke_scratch/dnikolo/ICON_Data/weights_3d.nc"
# ---------------------

# Build full glob pattern
search_pattern = os.path.join(in_dir, pattern)

input_files = sorted(glob.glob(search_pattern))

if not input_files:
    print(f"No files found matching {search_pattern}")
else:
    print(f"Found {len(input_files)} files.")

for in_file in input_files:
    base = os.path.basename(in_file)
    stem, ext = os.path.splitext(base)

    # Example output name: test01_atm_3d_ml_20080905T000000Z_remap.nc
    out_file = os.path.join(out_dir, f"{stem}_remap.nc")

    cmd = [
        "cdo",
        f"-remap,{grid_file},{weights_file}",
        in_file,
        out_file,
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Finished {out_file}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {in_file}: {e}")
