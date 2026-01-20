import csv
import os
import numpy as np
import torch


def read_dat_file(dat_file_path):
    in_nodes = False
    in_elements = False
    compact_mode = False
    current_body_id = None

    nodes = []
    cells_detail = []

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        # Node parsing
        if line.lower().startswith("nblock"):
            in_nodes = True
            continue

        if in_nodes and line == "-1":
            in_nodes = False
            continue

        if in_nodes:
            parts = line.split()
            if len(parts) == 4:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes.append((node_id, x, y, z))
            continue

        # Detect element block and material assignment
        if line.upper().startswith("MAT"):
            # The format is: MAT,1 $ TYPE,1 $ REAL,1 $ SECNUM,1 $ ESYS,0
            # Split by comma first
            parts = line.split(",")
            if len(parts) > 1:
                try:
            # Take the first number after MAT, ignore any trailing characters
                    current_body_id = int(parts[1].split()[0])
            
                except ValueError:
                    current_body_id = None
            continue

        if "eblock" in line.lower():
            compact_mode = "compact" in line.lower()
            in_elements = True
            continue

        if in_elements and line == "-1":
            in_elements = False
            compact_mode = False
            current_body_id = None
            continue

        # Elements parsing
        if in_elements:
            parts = line.split()
            if not parts:
                continue

            # For Compact format having 9 columns
            if compact_mode:
                if len(parts) >= 9:  # element id + 8 nodes
                    element_id = int(parts[0])
                    node_idsz = []
                    for i in range(1, 9):
                        node_idsz.append(int(parts[i]))

                    body_id = [current_body_id] if current_body_id is not None else [0]

                    cells_detail.append([element_id] + body_id + node_idsz)
                continue

             # For SOLID FORMAT having 19 columns
            num_nodes = 8
            if len(parts) > 14:
                num_nodes = int(parts[8])
                less = 0
                if num_nodes < 8:
                    less = 8 - num_nodes

                node_idsz = []
                element_id = [int(parts[10])]
                body_id = [int(parts[0])]

                for i in range(11, 19 - less):
                    node_idsz.append(int(parts[i]))

                if num_nodes <= 8:
                    cells_detail.append(element_id + body_id + node_idsz)

            if len(parts) == num_nodes - 8 and num_nodes > 8:
                for i in range(0, num_nodes - 8):
                    node_idsz.append(int(parts[i]))
                cells_detail.append(element_id + body_id + node_idsz)

    nodes.sort(key=lambda x: x[0])
    cells_detail.sort(key=lambda x: x[0])
    cells_detail = [list(item) for item in dict.fromkeys(tuple(cell) for cell in cells_detail)]

    # Optional: simplified cell connectivity
    cells = [[cell[2], cell[3], cell[4], cell[6]] for cell in cells_detail]

    return nodes, cells_detail, cells



def assign_body_id_to_nodes_generic(nodes, elements, pos_t1, pos_t1000, tolerance_y=1e-4, tolerance_deform=1e-4):
    """
    Generic function to assign node types based on deformation between timesteps.
    
    Parameters:
    - nodes: List of nodes with (node_id, x, y, z)
    - elements: List of elements with body_id information
    - pos_t1: Node positions at timestep 1 (list of [x, y, z])
    - pos_t1000: Node positions at timestep 1000 (list of [x, y, z])
    - tolerance_y: Tolerance for identifying surface nodes (y ~= 0)
    - tolerance_deform: Tolerance for identifying non-deforming nodes
    
    Returns:
    - node_type_mgn: List of node types (0, 1, or 3)
    """
    
    # Create a dictionary to store the body_id for each node
    node_body_mapping = {}
    
    # Loop through each element and assign the body_id to the corresponding nodes
    for element in elements:
        body_id = element[1]  # The second element is the body_id
        node_ids = element[2:]  # Get all node IDs from the element
        
        for node_id in node_ids:
            # Find the node with the given node_id in the nodes list
            for node in nodes:
                if node[0] == node_id:  # If node_id matches
                    if node_id not in node_body_mapping:
                        # Add node with body_id
                        node_body_mapping[node_id] = [node[0], body_id]
                    break
    
    # Convert the dictionary back to a list of nodes with their body_ids
    node_type = list(node_body_mapping.values())
    node_type.sort(key=lambda x: x[0])
    
    # Convert positions to numpy arrays for easier computation
    pos_t1_array = np.array(pos_t1, dtype=np.float32)
    pos_t1000_array = np.array(pos_t1000, dtype=np.float32)
    
    # Calculate deformation magnitude for each node
    deformation = np.linalg.norm(pos_t1000_array - pos_t1_array, axis=1)
    
    print(f"Processing {len(node_type)} nodes for node type assignment...")
    print(f"Tolerance for surface (y): {tolerance_y}")
    print(f"Tolerance for deformation: {tolerance_deform}")
    
    deforming_count = 0
    non_deforming_count = 0
    die_count = 0
    plate_non_surface_count = 0
    
    # Assign node types based on the logic
    for i, node in enumerate(node_type):
        node_id, body_id = node[0], node[1]
        
        # Die nodes keep body_id value
        if body_id == 2 or body_id == 1:
            # Die remains as type 1
            die_count += 1
            node[1] = 1   # Keep as die type
            continue
        
        # For plate nodes (change body id)
        if body_id == 3:
            # Get the y-coordinate from original node position
            original_node = next((n for n in nodes if n[0] == node_id), None)
            if original_node is None:
                continue
                
            y_coord = original_node[2]  # y is the third element
            
            # Check if node is on the surface (y ~= 0)
            if abs(y_coord) <= tolerance_y:
                # Node is on the surface, check deformation
                if deformation[i] > tolerance_deform:
                    # Node deforms significantly -> type 3
                    node[1] = 0   # plate deforming position
                    deforming_count += 1
                else:
                    # Node doesn't deform significantly -> type 0 (handle)
                    node[1] = 3   # plate non-deforming position
                    non_deforming_count += 1
            else:
                # Node is not on surface -> type 0
                node[1] = 0
                plate_non_surface_count += 1
    
    print(f"Die nodes (type 1): {die_count}")
    print(f"Surface nodes - Deforming (type 0): {deforming_count}")
    print(f"Surface nodes - Non-deforming (type 3): {non_deforming_count}")
    print(f"Plate non-surface nodes (type 0): {plate_non_surface_count}")
    
    # Generate node_type_mgn based on body_id
    node_type_mgn = [node[1] for node in node_type]
    
    return node_type_mgn


def read_result_file(result_file_path):
    parameters = []
    
    # Open the CSV file
    with open(result_file_path, mode='r') as file:
        if result_file_path.endswith(".csv"):
            # Create a CSV reader object
            reader = csv.reader(file)
            
            # Skip the header
            next(reader)
            
            # Loop through each row in the CSV file
            for row in reader:
                # The stress is in the last column (index -1)
                parameter = float(row[-1])  # Convert stress to float for numerical processing
                parameters.append(parameter)
        elif result_file_path.endswith(".txt"):
            for line in file:
                # Skip the header line
                if "Node Number" in line:
                    continue
                
                # Split the line into columns
                columns = line.split()
                                
                try:
                    # Extract the total deformation (4th column)
                    parameters.append(float(columns[-1]))
                except ValueError:
                    # Skip lines that do not have valid numerical data
                    continue
        
    return parameters


def find_global_pos(mesh_pos, disp_x, disp_y, disp_z):
    """Calculate global positions from mesh positions and displacements"""
    global_pos = []
    for i, pos in enumerate(mesh_pos):
        global_pos.append([pos[0]+float(disp_x[i]),
                        pos[1]+float(disp_y[i]),
                        pos[2]+float(disp_z[i])
                        ])
    return global_pos


def generate_frame(dat_file_path, stress_file_path, x_disp_file_path, y_disp_file_path,
                   z_disp_file_path):
    
    nodes, _, cells = read_dat_file(dat_file_path)

    stress = read_result_file(stress_file_path)
    x_disp = read_result_file(x_disp_file_path)
    y_disp = read_result_file(y_disp_file_path)
    z_disp = read_result_file(z_disp_file_path)
    mesh_pos = [list(node[1:4]) for node in nodes]
    world_pos = find_global_pos(mesh_pos, x_disp, y_disp, z_disp)
    
    frame = {
        'stress': torch.tensor(stress),
        'world_pos': torch.tensor(world_pos),
        'cells': torch.tensor(cells),
        'mesh_pos': torch.tensor(mesh_pos)
    }
    return frame


def main():
    base_folder = '/home/user/AnK_MeshGraphNets/data/raw_ANSYS_data/Static_sample/viscoelastic_glass/hexahedral'
    dat_file_path = os.path.join(base_folder, 'soln info.dat')
    stres_file_path = os.path.join(base_folder, 'stress.csv')
    x_disp_file_path = os.path.join(base_folder, 'disp_x.csv')
    y_disp_file_path = os.path.join(base_folder, 'disp_y.csv')
    z_disp_file_path = os.path.join(base_folder, 'disp_z.csv')

    
    frame = generate_frame(dat_file_path, stres_file_path,
                           x_disp_file_path, y_disp_file_path, z_disp_file_path)
    print()
    print("========================================================================")
    print("=======  Following information are found in the generated frame  =======")
    print()
    for key in frame:
        print("            :-", key, "has value of shape", frame[key].shape)
    print()
    print("=========================== Used ANSYS data ============================")
    print("========================================================================")
    print()
    return


if __name__ == '__main__':
    main()
