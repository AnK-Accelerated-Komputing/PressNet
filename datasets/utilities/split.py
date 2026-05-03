import h5py
import json
import os

base_dir = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/Splitdata/Coarse"

# Input files
h5_in = os.path.join(base_dir, "coarse_1500_test.h5")
json_in = os.path.join(base_dir, "coarse_1500_test_meta.json")

# Output files - Explicit selection
h5_explicit = os.path.join(base_dir, "coarse_1500_explicit.h5")
json_explicit = os.path.join(base_dir, "coarse_1500_explicit_meta.json")

# Output files - Range selection
h5_ranges = os.path.join(base_dir, "coarse_1500_ranges.h5")
json_ranges = os.path.join(base_dir, "coarse_1500_ranges_meta.json")
# Define indices
explicit_indices = {17, 29, 37, 40, 59, 69, 70, 87, 96, 109, 114, 125, 144}
range_indices = set(range(0, 10)) | set(range(130, 140))

# Map to group names
explicit_groups = {f"group_{i}" for i in explicit_indices}
range_groups = {f"group_{i}" for i in range_indices}

with open(json_in, "r") as f:
    meta = json.load(f)

meta_explicit = {}
meta_ranges = {}

print("🚀 Extracting specific groups...")

with h5py.File(h5_in, "r") as h5_src, \
     h5py.File(h5_explicit, "w") as h5_exp_out, \
     h5py.File(h5_ranges, "w") as h5_rng_out:

    for group_name in h5_src.keys():
        # Check for Explicit matches
        if group_name in explicit_groups:
            h5_src.copy(group_name, h5_exp_out)
            if group_name in meta:
                meta_explicit[group_name] = meta[group_name]
        
        # Check for Range matches
        elif group_name in range_groups:
            h5_src.copy(group_name, h5_rng_out)
            if group_name in meta:
                meta_ranges[group_name] = meta[group_name]

# Save JSON results
with open(json_explicit, "w") as f:
    json.dump(meta_explicit, f, indent=4)

with open(json_ranges, "w") as f:
    json.dump(meta_ranges, f, indent=4)

print("✔ Extraction complete")
print(f"--- Saved to Explicit: {len(meta_explicit)} groups")
print(f"--- Saved to Ranges:   {len(meta_ranges)} groups")