import os
import h5py
import torch
import generic_logic_dual_eblock as frame
import json
import re
import numpy as np
import math
import glob
import time
import pathlib as Path

def find_dat_file(folder_path):
    """Finds the first .dat file in a given folder"""
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    dat_file_path = os.path.join(folder_path, dat_files[0])
    return dat_file_path


def generate_trajectory(single_traj_folder_path, radius, device, stage=None):
    
    print(f"Extracting trajectory from {single_traj_folder_path}")
    folder_base_name = os.path.basename(single_traj_folder_path).replace("_data", "")
    dat_file_name = f"{folder_base_name}.dat"
    dat_file_path = find_dat_file(single_traj_folder_path)   
    
    # Check if the .dat file exists
    if not os.path.isfile(dat_file_path):
        print(f"Warning: {dat_file_path} does not exist. Skipping this folder.")
        return [], None

    try:
        deltim_pattern = r'deltim,\s*([0-9e\-\.]+)'
        time_step = None
        with open(dat_file_path, 'r') as file:
            # Read through each line of the file
            for line in file:
                # Search for a line containing 'deltim' and extract the value
                match = re.search(deltim_pattern, line)
                if match:
                    time_step = float(match.group(1))  # Convert to float
                    break

        if time_step is None:
            print("Warning: No 'deltim' value found in the .dat file.")
            
    except Exception as e:
        print(f"Error reading {dat_file_path}: {e}")
        return [], None
    
    # Define paths
    x_deform_path = os.path.join(single_traj_folder_path, "Directional_Deformation_X")
    y_deform_path = os.path.join(single_traj_folder_path, "Directional_Deformation_Y")
    z_deform_path = os.path.join(single_traj_folder_path, "Directional_Deformation_Z")
    stres_path = os.path.join(single_traj_folder_path, "Equivalent_Stress")

    # Read mesh data
    nodes, cells_detail, cells = frame.read_dat_file(dat_file_path)
    print("Cells and nodes extracted")
    mesh_pos_list = [list(node[1:4]) for node in nodes]
    mesh_pos = torch.tensor(mesh_pos_list, dtype=torch.float32).to(device)
    
    # Read positions at timestep 1 (file named "Directional Deformation X 1.txt")
    print("Reading positions at timestep 1...")
    x_t1 = frame.read_result_file(os.path.join(x_deform_path, 'Directional Deformation X 2.txt'))
    y_t1 = frame.read_result_file(os.path.join(y_deform_path, 'Directional Deformation Y 2.txt'))
    z_t1 = frame.read_result_file(os.path.join(z_deform_path, 'Directional Deformation Z 2.txt'))
    pos_t1 = frame.find_global_pos(mesh_pos_list, x_t1, y_t1, z_t1)
    
    # Read positions at timestep 1000 (file named "Directional Deformation X 1000.txt")
    print("Reading positions at timestep 1000...")
    x_t1000 = frame.read_result_file(os.path.join(x_deform_path, 'Directional Deformation X 1000.txt'))
    y_t1000 = frame.read_result_file(os.path.join(y_deform_path, 'Directional Deformation Y 1000.txt'))
    z_t1000 = frame.read_result_file(os.path.join(z_deform_path, 'Directional Deformation Z 1000.txt'))
    pos_t1000 = frame.find_global_pos(mesh_pos_list, x_t1000, y_t1000, z_t1000)
    
    # Assign node types using the new generic function
    print("Assigning node types based on deformation...")
    node_type = frame.assign_body_id_to_nodes_generic(
        nodes, 
        cells_detail, 
        pos_t1, 
        pos_t1000,
        tolerance_y=1e-8,
        tolerance_deform=1e-1
    )
    
    cells = torch.tensor(cells, dtype=torch.float32).to(device)
    cells = cells - 1
    node_type = torch.tensor(node_type, dtype=torch.float32).view(-1, 1).to(device)
    print("Node type extracted")
    print(f"Node type distribution: {torch.unique(node_type, return_counts=True)}")

    trajectory = []
    start = 1  # default
    end = 1499 # default
    
    if stage == 1 or stage is None:
        # For first step (timestep 1)
        stress_t1 = frame.read_result_file(os.path.join(stres_path, 'Equivalent Stress 2.txt'))
        target_stress = frame.read_result_file(os.path.join(stres_path, 'Equivalent Stress 3.txt'))
        
        stress_t1 = torch.tensor(stress_t1, dtype=torch.float32).view(-1, 1).to(device)
        target_stress = torch.tensor(target_stress, dtype=torch.float32).view(-1, 1).to(device)
        
        # Read target positions for timestep 2
        target_x = frame.read_result_file(os.path.join(x_deform_path, 'Directional Deformation X 3.txt'))
        target_y = frame.read_result_file(os.path.join(y_deform_path, 'Directional Deformation Y 3.txt'))
        target_z = frame.read_result_file(os.path.join(z_deform_path, 'Directional Deformation Z 3.txt'))
        target_world_pos = frame.find_global_pos(mesh_pos_list, target_x, target_y, target_z)
        target_world_pos = torch.tensor(target_world_pos, dtype=torch.float32).to(device)
        
        # Use pos_t1 as world_pos for first timestep
        world_pos_t1 = torch.tensor(pos_t1, dtype=torch.float32).to(device)
        
        first_step = {
            'stress': stress_t1,
            'target|stress': target_stress,
            'node_type': node_type,
            'world_pos': world_pos_t1,
            'target|world_pos': target_world_pos,
            'cells': cells,
            'mesh_pos': mesh_pos
        }
        trajectory.append(first_step)
        print("First frame/time step done")

    # Process timesteps 2 to 1499
    for i in range(start, end):
        stress = frame.read_result_file(os.path.join(stres_path, f'Equivalent Stress {i + 1}.txt'))
        target_stress = frame.read_result_file(os.path.join(stres_path, f'Equivalent Stress {i + 2}.txt'))
        x = frame.read_result_file(os.path.join(x_deform_path, f'Directional Deformation X {i + 1}.txt'))
        target_x = frame.read_result_file(os.path.join(x_deform_path, f'Directional Deformation X {i + 2}.txt'))
        y = frame.read_result_file(os.path.join(y_deform_path, f'Directional Deformation Y {i + 1}.txt'))
        target_y = frame.read_result_file(os.path.join(y_deform_path, f'Directional Deformation Y {i + 2}.txt'))
        z = frame.read_result_file(os.path.join(z_deform_path, f'Directional Deformation Z {i + 1}.txt'))
        target_z = frame.read_result_file(os.path.join(z_deform_path, f'Directional Deformation Z {i + 2}.txt'))
        world_pos = frame.find_global_pos(mesh_pos_list, x, y, z)
        target_world_pos = frame.find_global_pos(mesh_pos_list, target_x, target_y, target_z)

        stress = torch.tensor(stress, dtype=torch.float32).view(-1, 1).to(device)
        target_stress = torch.tensor(target_stress, dtype=torch.float32).view(-1, 1).to(device)
        world_pos = torch.tensor(world_pos, dtype=torch.float32).to(device)
        target_world_pos = torch.tensor(target_world_pos, dtype=torch.float32).to(device)

        step = {
            'stress': stress,
            'target|stress': target_stress,
            'node_type': node_type,
            'world_pos': world_pos,
            'target|world_pos': target_world_pos,
            'cells': cells,
            'mesh_pos': mesh_pos
        }
        trajectory.append(step)
        print(i + 1, end='\r')

    return trajectory, time_step


def extract_numbers(folder_name):
    match = re.match(r'([a-zA-Z]+)_(\d+)x(\d+)_mm_plate_solid_185_data', folder_name)
    if match:
        prefix, length, width = match.groups()
        return (prefix, int(length), int(width))  
    return (folder_name, float('inf'), float('inf'))  # Default for non-matching names


def save_multiple_trajectories_to_h5(train_folder_path, folders, output_file_path, json_output_path, device):
    with h5py.File(output_file_path, 'w') as f:

        group_to_folder_map = {}  # Initialize dictionary for the mapping
        group_counter = 0  # Initialize the counter for groups

        simulation_time = [
            "1 h", "11m 1s", "10m 49s", "11m 54s", "12m",
            "11m 22s", "12m 34s", "19m 24s", "12m 6s", "14m 40s"
        ]

        radius = [
            100, 100, 100, 100, 99,
            99, 99, 99, 98, 98
        ]

        distance_H = [
            50, 51, 52, 53, 54,
            49, 50, 51, 52, 53
        ]

        for data_folder in folders:
            data_folder_path = os.path.join(train_folder_path, data_folder)
            D = radius[group_counter] if group_counter < len(radius) else 100
            sim_time = simulation_time[group_counter] if group_counter < len(simulation_time) else "Unknown"
            dis = distance_H[group_counter] if group_counter < len(distance_H) else 50
         
            r = D

            print(f'\n{"="*80}')
            print(f'Processing folder: {data_folder}')
            print(f'r = {r}')
            print(f'{"="*80}\n')
            
            # Skip if it's not a directory
            if not os.path.isdir(data_folder_path):
                continue
            
            # Extract the trajectory (radius parameter is no longer used in the new logic)
            trajectory, time_step = generate_trajectory(data_folder_path, r, device)
            
            # Skip folders that don't generate valid trajectories
            if len(trajectory) == 0 or time_step is None:
                print(f"Skipping {data_folder} - invalid trajectory")
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
            number_of_steps = len(trajectory)

            # Add additional info to the dictionary
            group_to_folder_map[group_name] = { 
                'data type': data_folder,
                'hollow portion length in die': D,
                'distance between base and top of die': dis,
                'thickness of plate': 10,
                'simulation_time': sim_time,
                'number of nodes': num_nodes,
                'number of cells': num_cells,
                'del time': time_step,
                'number of steps': number_of_steps,
            }
            
            print(f"\n{'='*80}")
            print(f"Successfully processed and added trajectory for {data_folder} as {group_name}")
            print(f"Number of nodes: {num_nodes}, Number of cells: {num_cells}, Number of steps: {number_of_steps}")
            print(f"{'='*80}\n")
            
            group_counter += 1  # Increment the counter for the next group

        with open(json_output_path, 'w') as json_file:
            json.dump(group_to_folder_map, json_file, indent=4)
        print(f"\nGroup-to-folder mapping saved to {json_output_path}")
        print(f"Total groups processed: {group_counter}")


def main():
    start_time = time.time()
    device = torch.device('cpu')
    
    # Path to the 'train' folder containing data subdirectories
    train_folder_path = r"J:\Files\03_Work\10_AnK\3_PressNetData\DATA_P_FULL_1500_Medium\p_full"
    
    # Output path for the HDF5 file
    output_file_path = r"J:\Files\03_Work\10_AnK\3_PressNetData\Data_From_Notion\1500_Medium\h5_medium\p_full_1500_medium5.h5"
    json_output_path = r"J:\Files\03_Work\10_AnK\3_PressNetData\Data_From_Notion\1500_Medium\h5_medium\p_full_1500_medium5.json"

    folders = sorted(os.listdir(train_folder_path), key=extract_numbers)
    print(f"Found {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {folder}")
    print()
    
    # Save multiple trajectories from different folders into the HDF5 file
    save_multiple_trajectories_to_h5(train_folder_path, folders, output_file_path, json_output_path, device)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal time taken for node handling and data conversion: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == '__main__':
    main()