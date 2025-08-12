import os
import h5py
import torch
from utilities.trajectory_solid185_quarter import generate_trajectory_h5 as generate_trajectory
import json
import re

def list_subfolders(directory):
    return [f.name for f in os.scandir(directory) if f.is_dir()]

def save_multiple_trajectories_to_h5(raw_folder_path, output_folder_path, device):
    basename = os.path.basename(raw_folder_path)
    h5_path = os.path.join(output_folder_path, f"{basename}.h5")
    meta_json_path = os.path.join(output_folder_path, f"{basename}.json")
    with h5py.File(h5_path, 'w') as f:

        group_to_folder_map = {}  # Initialize dictionary for the mapping
        group_counter = 0  # Initialize the counter for groups
        
        # Iterate over each data folder in the train folder
        die_shape_folders = list_subfolders(raw_folder_path)
        for die_shape_folder in die_shape_folders:
            die_shape_folder_path = os.path.join(raw_folder_path, die_shape_folder)
            print(f"Processing folder: {die_shape_folder}")
            data_folders = list_subfolders(die_shape_folder_path)
            for data_folder in data_folders:
                print(f"Processing folder: {data_folder}")
                data_folder_path = os.path.join(die_shape_folder_path, data_folder)
                
                # Extract the trajectory
                trajectory, time_step = generate_trajectory(data_folder_path, device)
                
                # Skip folders that don't generate valid trajectories
                if len(trajectory) == 0:
                    continue
                
                # Create a unique group name based on the counter
                group_name = f"group_{group_counter}"
                
                # Create the group in the HDF5 file
                traj_group = f.create_group(group_name)
                
                # Add shared data (mesh_pos, node_type, cells) to the group once
                traj_group.create_dataset("mesh_pos", data=trajectory[0]['mesh_pos'].cpu().numpy())
                traj_group.create_dataset("node_type", data=trajectory[0]['node_type'].cpu().numpy())
                traj_group.create_dataset("cells", data=trajectory[0]['cells'].cpu().numpy())
                
                # Add the trajectory data to this group for each time step
                for t, step in enumerate(trajectory):
                    step_group = traj_group.create_group(f"step_{t+1}")
                    
                    for key, value in step.items():
                        if key not in ["mesh_pos", "node_type", "cells"]:
                            step_group.create_dataset(key, data=value.cpu().numpy())

                # Gather additional information
                num_nodes = trajectory[0]['mesh_pos'].shape[0]  # The number of nodes (rows in mesh_pos)
                num_cells = trajectory[0]['cells'].shape[0]  # The number of cells (rows in cells)
                simulation_time = 15
                number_of_steps = 400

                # Add additional info to the dictionary
                group_to_folder_map[group_name] = {
                    'data type': data_folder,
                    'number of nodes': num_nodes,
                    'number of cells': num_cells,
                    'del time': time_step,
                    'number of steps': number_of_steps,
                    'simulation time': simulation_time
                    }
                
                print(f"Processed and added trajectory for {data_folder} as {group_name}")
                
                group_counter += 1  # Increment the counter for the next group

        with open(meta_json_path, 'w') as json_file:
            json.dump(group_to_folder_map, json_file, indent=4)
        print(f"Group-to-folder mapping saved to {meta_json_path}")


def main():
    device = torch.device('cuda')
    raw_folder_path = "/home/gd_user1/AnK/project_PINN/PressNet/datasets/raw_data/15x10_400steps_coarse_data/15x10_400steps_coarse_data"
    output_folder_path = "/home/gd_user1/AnK/project_PINN/PressNet/datasets/extracted_data"
    os.makedirs(ouput_folder_path, exist_ok=True)
    save_multiple_trajectories_to_h5(raw_folder_path, output_folder_path, device)

if __name__ == '__main__':
    main()
