import csv
import os
import numpy as np
import torch
import re

def read_dat_file(dat_file_path):
    # Flags to track node section
    in_nodes = False
    in_elements = False

    #Initialize storage
    nodes = []
    cells_detail =[]
    # Read the .dat file
    deltim_pattern = r'deltim,\s*([0-9e\-\.]+)'

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        match = re.search(deltim_pattern, line)
        if match:
            time_step = float(match.group(1))  # Convert to float
                    
        line = line.strip()

        # Detect the start of the node block
        if line.lower().startswith("nblock"):
            in_nodes = True
            continue

        # Detect the end of the node block (marked by '-1')
        if in_nodes and line.strip() == "-1":
            in_nodes = False
            continue

        # Detect the start of the element block
        if "Body" in line:
            in_elements = True
            continue

        # Detect the end of the element block (marked by '-1')
        if in_elements and line.strip() == "-1":
            in_elements = False
            continue

        # Parse the node data
        if in_nodes:
            # Split by spaces and filter out empty entries
            parts = line.split()
            if len(parts) == 4:  # Ensure the line has 4 parts: ID, X, Y, Z
                node_id = int(parts[0])    # Node ID
                x = float(parts[1])        # X-coordinate
                y = float(parts[2])        # Y-coordinate
                z = float(parts[3])        # Z-coordinate
                # print(node_id, x, y, z)
                nodes.append((node_id, x, y, z))

        # Parse the element data
        if in_elements:
            parts = line.split()
            num_nodes =8
            if len(parts) > 14:
                 # Ensure the line has 4 parts: ID, X, Y, Z
                num_nodes = int(parts[8])
                less = 0
                if num_nodes < 8:
                    less = 8-num_nodes
                node_idsz = []
                # print(parts)
                element_id = [int(parts[10])]  # Element ID
                body_id = [int(parts[0])]
                for i in range(11, 19-less):
                    # print(i)
                    node_idsz.append(int(parts[i]))   # Node IDs
                if num_nodes<=8:
                    cells_detail.append(element_id+body_id+node_idsz)
            if len(parts) == num_nodes-8 and num_nodes>8:  # Ensure the line has 4 parts: ID, X, Y, Z    
                for i in range(0,num_nodes-8 ):
                    node_idsz.append(int(parts[i])) 
                # print(element_id, node_idsz)
                cells_detail.append(element_id+body_id+node_idsz)

    nodes.sort(key=lambda x:x[0])
    cells_detail.sort(key=lambda x:x[0])
    # print(cells)
    cells_detail = [list(item) for item in dict.fromkeys(tuple(cell) for cell in cells_detail)]
    cells = [[cell[2],cell[3],cell[4],cell[6]] for cell in cells_detail]
    return nodes,cells_detail,cells,time_step

def assign_body_id_to_nodes(nodes, elements):
    # Create a dictionary to store the body_id for each node
    node_body_mapping = {}

    # Loop through each element and assign the body_id to the corresponding nodes
    for element in elements:
        body_id = element[1]  # The second element is the body_id
        node_ids = element[2:7]  # The remaining elements are the node ids
        
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
    # # Assign body ID 4 based on coordinate conditions
    # for node in node_type:
    #     # Find the corresponding node in the original 'nodes' list
    #     for original_node in nodes:
    #         if node[0] == original_node[0] and node[1] == 1:  # Match node ID
    #             x, y = original_node[1], original_node[2]  # Extract x and y coordinates
    #             if  100 > x >= -100 and y == 0:
    #                 node[1] = 4  # Assign body ID to 4
    #             break
    # print(node_type)
    # print(len(node_type))
    node_type_bid = [node[1] for node in node_type] 
    node_type_mgn = [0 if num == 1 else 3 if num == 4 else 1 for num in node_type_bid]
    return node_type_mgn


def assign_body_id_to_nodes_quarter(nodes, elements):
    # Create a dictionary to store the body_id for each node
    node_body_mapping = {}

    # Loop through each element and assign the body_id to the corresponding nodes
    for element in elements:
        body_id = element[1]  # The second element is the body_id
        node_ids = element[2:7]  # The remaining elements are the node ids
        
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

     # Assign body ID 4 based on coordinate conditions
    for node in node_type:
        # Find the corresponding node in the original 'nodes' list
        for original_node in nodes:
            if node[0] == original_node[0] and node[1] == 1:  # Match node ID
                x, y = original_node[1], original_node[2]  # Extract x and y coordinates
                if x == 0 or (0 > x >= -100 and y == 0):
                    node[1] = 4  # Assign body ID to 4
                break
    # print(node_type)
    # print(len(node_type))
    node_type_bid = [node[1] for node in node_type] 
    node_type_mgn = [0 if num == 1 else 3 if num == 4 else 1 for num in node_type_bid]
    return node_type_mgn

def assign_body_id_to_nodes_thermal(nodes, elements):
    # Create a dictionary to store the body_id for each node
    node_body_mapping = {}

    # Loop through each element and assign the body_id to the corresponding nodes
    for element in elements:
        body_id = element[1]  # The second element is the body_id
        node_ids = element[2:7]  # The remaining elements are the node ids
        
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

    x_coords = [node[1] for node in nodes]
    y_coords = [node[2] for node in nodes]
    z_coords = [node[3] for node in nodes]

    x_max, x_min = max(x_coords), min(x_coords)
    y_max, y_min = max(y_coords), min(y_coords)
    z_max, z_min = max(z_coords), min(z_coords)

     # Assign body ID 4 based on coordinate conditions
    for node in node_type:
        # Find the corresponding node in the original 'nodes' list
        for original_node in nodes:
            if node[0] == original_node[0] and node[1] == 1:  # Match node ID
                x, y, z = original_node[1], original_node[2], original_node[3]  # Extract x and y coordinates
                if x == x_max or x==x_min or y == y_max or y==y_min or z == z_max or z==z_min:
                    node[1] = 2  # Assign body ID to 4
                break
    # print(node_type)
    # print(len(node_type))
    node_type_bid = [node[1] for node in node_type] 
    node_type_mgn = [0 if num == 1 else 6 if num == 2 else 9 for num in node_type_bid]
    # print(node_type_mgn)
    return node_type_mgn

def assign_body_id_to_nodes_thermal_single_body(nodes, elements):
    

    x_coords = [node[1] for node in nodes]
    y_coords = [node[2] for node in nodes]
    z_coords = [node[3] for node in nodes]

    x_max, x_min = max(x_coords), min(x_coords)
    y_max, y_min = max(y_coords), min(y_coords)
    z_max, z_min = max(z_coords), min(z_coords)

     # Assign node type based on position
    for i,node in enumerate(nodes):
        if isinstance(nodes[i], tuple):
            nodes[i] = list(nodes[i])  # Convert the tuple to a list
        x, y,z = node[1], node[2], node[3]  # Extract x and y coordinates
        if x == x_max or x==x_min or y == y_max or y==y_min or z == z_max or z==z_min:
            nodes[i].append(6)   # Assign node ID to 6 (wall boundary)
        else:
            nodes[i].append(0)   # Assign node ID to 0 (notmal)
    # print(nodes)
    node_type_mgn = [node[4] for node in nodes]
    # print(node_type_mgn, end=' ')

    return node_type_mgn




def read_result_file(result_file_path,encoding='utf-8'):
    parameters = []
    
    # Open the CSV file
    with open(result_file_path, mode='r',encoding=encoding) as file:
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

def find_global_pos(mesh_pos,disp_x,disp_y,disp_z):
    #initialize storage
    global_pos = []

    for i, pos in enumerate(mesh_pos):
        # print(i,pos)
        global_pos.append([pos[0]+float(disp_x[i]),
                           pos[1]+float(disp_y[i]),
                           pos[2]+float(disp_z[i])
                           ])
    return global_pos


def generate_frame(dat_file_path,stress_file_path,x_disp_file_path,y_disp_file_path,
                   z_disp_file_path):
    
    nodes, _,cells = read_dat_file(dat_file_path)
    node_type = assign_body_id_to_nodes(nodes,cells)

    stress = read_result_file(stress_file_path)
    x_disp = read_result_file(x_disp_file_path)
    y_disp = read_result_file(y_disp_file_path)
    z_disp = read_result_file(z_disp_file_path)
    # print(nodes)
    mesh_pos = [list(node[1:4]) for node in nodes]
    # print(mesh_pos)
    world_pos = find_global_pos(mesh_pos,x_disp,y_disp,z_disp)
    
    frame = {
        'stress' : torch.tensor(stress),
        'node_type': torch.tensor(node_type),
        'world_pos': torch.tensor(world_pos),
        'cells': torch.tensor(cells),
        'mesh_pos': torch.tensor(mesh_pos)
    }
    return frame

def main():
    base_folder = '/home/user/AnK_MeshGraphNets/data/raw_ANSYS_data/Static_sample/viscoelastic_glass/hexahedral'
    dat_file_path = os.path.join(base_folder,'soln info.dat')
    stres_file_path = os.path.join(base_folder,'stress.csv')
    x_disp_file_path = os.path.join(base_folder,'disp_x.csv')
    y_disp_file_path = os.path.join(base_folder,'disp_y.csv')
    z_disp_file_path = os.path.join(base_folder,'disp_z.csv')
    
    frame = generate_frame(dat_file_path,stres_file_path,
                           x_disp_file_path,y_disp_file_path,z_disp_file_path)
    print()
    print("========================================================================")
    print("=======  Following information are found in the generated frame  =======")
    print()
    for key in frame:
        print("            :-",key,"has value of shape",frame[key].shape)
    print()
    print("=========================== Used ANSYS data ============================")
    print("========================================================================")
    print()
    
    return


if __name__ == '__main__':
    main()
