import os
import h5py
import json
import re

def get_highest_group_index(h5_file):
    """Find the highest numbered group index in the given HDF5 file."""
    max_index = -1
    for group in h5_file.keys():
        match = re.match(r"group_(\d+)", group)  # Match 'group_N' pattern
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index

def merge_multiple_h5_datasets(h5_files, json_files, merged_h5, merged_json):
    with h5py.File(merged_h5, 'w') as merged_file:
        merged_meta = {}
        group_mapping = {}
        start_index = 0  # Initial group index
        
        for h5_file, json_file in zip(h5_files, json_files):
            with h5py.File(h5_file, 'r') as input_file:
                # Copy groups with updated indices
                for group in input_file:
                    new_group_name = f"group_{start_index}"
                    input_file.copy(group, merged_file, name=new_group_name)
                    group_mapping[new_group_name] = group
                    start_index += 1  # Increment index
            
            # Merge JSON metadata
            with open(json_file, 'r') as f:
                file_meta = json.load(f)

                # file_meta is like {"group_0": {...}, "group_1": {...}}
                for _, gdata in file_meta.items():
                    merged_meta[f"group_{len(merged_meta)}"] = gdata

        
        # Add group mapping to metadata
        merged_meta["group_mapping"] = group_mapping
        
    # Save merged metadata to JSON
    with open(merged_json, 'w') as f_out:
        json.dump(merged_meta, f_out, indent=4)
    
    print(f"Merged datasets saved in {merged_h5}")
    print(f"Merged metadata saved in {merged_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge several per-shape .h5 files (and their meta .json) into one dataset")
    parser.add_argument("--h5", nargs="+", required=True, help="Input .h5 files, in the order they should be merged")
    parser.add_argument("--json", nargs="+", required=True, help="Matching meta .json files (same order)")
    parser.add_argument("--out_h5", required=True, help="Merged .h5 path")
    parser.add_argument("--out_json", default=None, help="Merged meta path (default: <out_h5>_meta.json)")
    args = parser.parse_args()
    if len(args.h5) != len(args.json):
        parser.error("--h5 and --json must list the same number of files")

    out_json = args.out_json or os.path.splitext(args.out_h5)[0] + "_meta.json"
    merge_multiple_h5_datasets(args.h5, args.json, args.out_h5, out_json)
