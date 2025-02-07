''' This is to extract trajectory combining all franes/time steps of a single simulation'''
import os
import utilities.frame_solid185 as frame
import torch


def generate_trajectory(single_traj_folder_path,device,stage=None):
    print(f"extracting trajectory from {single_traj_folder_path}")
    folder_base_name = os.path.basename(single_traj_folder_path).replace("_data","")
    dat_file_path = os.path.join(single_traj_folder_path,f"{folder_base_name}.dat")
    x_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_X")
    y_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_Y")
    z_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_Z")
    stres_path = os.path.join(single_traj_folder_path,"Equivalent_Stress")

    nodes,cells_detail,cells =frame.read_dat_file(dat_file_path)
    print("cells and nodes extracted")
    # print(cells)
    mesh_pos_list =  [list(node[1:4]) for node in nodes]
    mesh_pos = torch.tensor(mesh_pos_list,dtype=torch.float32).to(device)
    node_type = frame.assign_body_id_to_nodes_quarter(nodes,cells_detail)
    cells = torch.tensor(cells,dtype=torch.float32).to(device)
    cells = cells-1
    node_type = torch.tensor(node_type,dtype=torch.float32).view(-1,1).to(device)
    print("node_type extracted")

    trajectory = []
    start=1 #default
    end=299 #default
    if stage ==1 or stage == None:
        #for first step
        target_stress = frame.read_result_file(os.path.join(stres_path,'Equivalent Stress 2.txt'))
        target_stress = torch.tensor(target_stress,dtype=torch.float32).view(-1,1).to(device)
        target_x = frame.read_result_file(os.path.join(x_deform_path,'Directional Deformation X 2.txt'))
        target_y = frame.read_result_file(os.path.join(y_deform_path,'Directional Deformation Y 2.txt'))
        target_z = frame.read_result_file(os.path.join(z_deform_path,'Directional Deformation Z 2.txt'))
        target_world_pos = frame.find_global_pos(mesh_pos_list,target_x,target_y,target_z)
        target_world_pos = torch.tensor(target_world_pos,dtype=torch.float32).to(device)
        first_step = {
            'stress' : torch.zeros_like(node_type).to(device),
            'target|stress' : target_stress,
            'node_type' : node_type,
            'world_pos' : mesh_pos,
            'target|world_pos' : target_world_pos,
            'cells' : cells,
            'mesh_pos' : mesh_pos
        }
        trajectory.append(first_step)
        print("first frame/time step done")
    if stage ==1:
        end=99
    elif stage ==2:
        start=100
        end=199
    elif stage ==3:
        start=200
    for i in range(start,end):
        stress = frame.read_result_file(os.path.join(stres_path,f'Equivalent Stress {i+1}.txt'))
        target_stress = frame.read_result_file(os.path.join(stres_path,f'Equivalent Stress {i+2}.txt'))
        x = frame.read_result_file(os.path.join(x_deform_path,f'Directional Deformation X {i+1}.txt'))
        target_x = frame.read_result_file(os.path.join(x_deform_path,f'Directional Deformation X {i+2}.txt'))
        y = frame.read_result_file(os.path.join(y_deform_path,f'Directional Deformation Y {i+1}.txt'))
        target_y = frame.read_result_file(os.path.join(y_deform_path,f'Directional Deformation Y {i+2}.txt'))
        z = frame.read_result_file(os.path.join(z_deform_path,f'Directional Deformation Z {i+1}.txt'))
        target_z = frame.read_result_file(os.path.join(z_deform_path,f'Directional Deformation Z {i+2}.txt'))

        world_pos = frame.find_global_pos(mesh_pos_list,x,y,z)
        target_world_pos = frame.find_global_pos(mesh_pos_list,target_x,target_y,target_z)

        stress = torch.tensor(stress,dtype=torch.float32).view(-1,1).to(device)
        target_stress = torch.tensor(target_stress,dtype=torch.float32).view(-1,1).to(device)
        world_pos = torch.tensor(world_pos,dtype=torch.float32).to(device)
        target_world_pos = torch.tensor(target_world_pos,dtype=torch.float32).to(device)


        step = {
            'stress' : stress,
            'target|stress' : target_stress,
            'node_type' : node_type,
            'world_pos' : world_pos,
            'target|world_pos' : target_world_pos,
            'cells' : cells,
            'mesh_pos' : mesh_pos
        }
        trajectory.append(step)
        print(i+1,end='\r')
        continue

    return trajectory



def generate_trajectory_h5(single_traj_folder_path,device):
    print(f"extracting trajectory from {single_traj_folder_path}")
    folder_base_name = os.path.basename(single_traj_folder_path).replace("_data","")
    dat_file_path = os.path.join(single_traj_folder_path,f"{folder_base_name}.dat")
    x_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_X")
    y_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_Y")
    z_deform_path = os.path.join(single_traj_folder_path,"Directional_Deformation_Z")
    stres_path = os.path.join(single_traj_folder_path,"Equivalent_Stress")

    nodes,cells_detail,cells,time_step =frame.read_dat_file(dat_file_path)
    print("cells and nodes extracted")
    # print(cells)
    mesh_pos_list =  [list(node[1:4]) for node in nodes]
    mesh_pos = torch.tensor(mesh_pos_list,dtype=torch.float32).to(device)
    node_type = frame.assign_body_id_to_nodes(nodes,cells_detail)
    cells = torch.tensor(cells,dtype=torch.float32).to(device)
    cells = cells-1
    node_type = torch.tensor(node_type,dtype=torch.float32).view(-1,1).to(device)
    print("node_type extracted")

    trajectory = []
    start=1 #default
    end=400 #default
    #for first step
    first_step = {
        'curr_stress' : torch.zeros_like(node_type).to(device),
        'node_type' : node_type,
        'curr_pos' : mesh_pos,
        'cells' : cells,
        'mesh_pos' : mesh_pos
    }
    trajectory.append(first_step)
    print("first frame/time step done")

    for i in range(start,end):
        stress = frame.read_result_file(os.path.join(stres_path,f'Equivalent Stress {i+1}.txt'))
        x = frame.read_result_file(os.path.join(x_deform_path,f'Directional Deformation X {i+1}.txt'))
        y = frame.read_result_file(os.path.join(y_deform_path,f'Directional Deformation Y {i+1}.txt'))
        z = frame.read_result_file(os.path.join(z_deform_path,f'Directional Deformation Z {i+1}.txt'))

        world_pos = frame.find_global_pos(mesh_pos_list,x,y,z)

        stress = torch.tensor(stress,dtype=torch.float32).view(-1,1).to(device)
        world_pos = torch.tensor(world_pos,dtype=torch.float32).to(device)


        step = {
            'curr_stress' : stress,
            'node_type' : node_type,
            'curr_pos' : world_pos,
            'cells' : cells,
            'mesh_pos' : mesh_pos
        }
        trajectory.append(step)
        print(i+1,end='\r')
        continue
    print("trajectory done, frames",i+1)
    return trajectory,time_step


def main():
    device = torch.device('cuda')
    generated_trajectory = generate_trajectory("/home/user/AnK_MeshGraphNets/raw_data/solid185/train/3mm_plate_data",device)
    print()
    print("========================================================================")
    print()
    print(f"================ Total steps in this trajectory: {len(generated_trajectory)} ==================")
    # print(len(generated_trajectory))
    # print(generated_trajectory)
    print()
    print("========================================================================")
    print("=======  Following information are found in the generated frame  =======")
    print()
    for key in generated_trajectory[0]:
        print("            :-",key,"has value of shape",generated_trajectory[0][key].shape)
        # if key == 'node_type':
        #     print(generated_trajectory[0][key])
    print()
    print("=========================== Used ANSYS data ============================")
    print("========================================================================")
    print()
    return
   

if __name__ == '__main__':
    main()