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

# Define file paths for 15 datasets
h5_files = [
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/channel_rect_medium.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Channel_u_1500_medium_press_dataset.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/channel_v_medium_u1.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/conical_press_dataset.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/cylinder_1500_medium_1.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/DATA_PERPENDICULAR_QTR_press_dataset.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/DATA_W_press_dataset.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/l_bend_double_medium_1500.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/l_bend_single_medium_1500.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/p_full_1500_medium_1.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/s_full_medium_1500.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/s_quarter_medium.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Semicircular_medium_u1.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/sphere_medium_u1.h5",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Wind_shield_1500_medium_press_dataset.h5"
]

json_files = [
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/channel_rect_meta_medium.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Channel_u_1500_medium_meta.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/channel_v_medium_u1.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/conical_meta.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/cylinder_1500_medium_1.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/DATA_PERPENDICULAR_QTR_meta.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/DATA_W_meta.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/l_bend_double_meta_medium.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/l_bend_single_meta_medium_1500.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/p_full_1500_medium_1.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/s_full_meta_medium_1500.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/s_quarter_meta_medium.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Semicircular_medium_u1.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/sphere_medium_u1.json",
    "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium/Wind_shield_1500_medium_meta.json"
]



print('processing-------------------------------------')
merged_h5 =     "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium//merged_1500_fine1.h5"
merged_json = "/home/ank-server5/ANK/PressNet/_NASMOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/H5_File_indiv/Medium//merged_1500_fine_meta1.json"

# Merge datasets
merge_multiple_h5_datasets(h5_files, json_files, merged_h5, merged_json)
