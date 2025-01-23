import torch



def convert_to_evaluation_trajectory(original_trajectory):
    restructured_trajectory = {
        'stress': [],
        'target|stress': [],
        'node_type': [],
        'world_pos': [],
        'target|world_pos': [],
        'cells': [],
        'mesh_pos': []
    }

    for frame in original_trajectory:
        # Append each tensor from the dictionary to the corresponding key in the new dictionary
        restructured_trajectory['stress'].append(frame['stress'])
        restructured_trajectory['target|stress'].append(frame['target|stress'])
        restructured_trajectory['node_type'].append(frame['node_type'])
        restructured_trajectory['world_pos'].append(frame['world_pos'])
        restructured_trajectory['target|world_pos'].append(frame['target|world_pos'])
        restructured_trajectory['cells'].append(frame['cells'])
        restructured_trajectory['mesh_pos'].append(frame['mesh_pos'])

    # Now stack the lists along the first axis to create tensors with the desired shape
    restructured_trajectory['stress'] = torch.stack(restructured_trajectory['stress'], dim=0)  # Shape: [399, 2016, 1]
    restructured_trajectory['target|stress'] = torch.stack(restructured_trajectory['target|stress'], dim=0)  # Shape: [399, 2016, 1]
    restructured_trajectory['node_type'] = torch.stack(restructured_trajectory['node_type'], dim=0)  # Shape: [399, 2016, 1]
    restructured_trajectory['world_pos'] = torch.stack(restructured_trajectory['world_pos'], dim=0)  # Shape: [399, 2016, 3]
    restructured_trajectory['target|world_pos'] = torch.stack(restructured_trajectory['target|world_pos'], dim=0)  # Shape: [399, 2016, 3]
    restructured_trajectory['cells'] = torch.stack(restructured_trajectory['cells'], dim=0)  # Shape: [399, 5264, 4]
    restructured_trajectory['mesh_pos'] = torch.stack(restructured_trajectory['mesh_pos'], dim=0)  # Shape: [399, 2016, 3]


    return restructured_trajectory



def generate_evaluation_trajectory(data_folder_path, device,key="quarter",stage=None):
    if key == "quarter":
        from utilities.trajectory_solid185_quarter import generate_trajectory
    
    train_style_trajectory = generate_trajectory(data_folder_path,device,stage)
    evaluation_style_trajectory = convert_to_evaluation_trajectory(train_style_trajectory)

    return evaluation_style_trajectory

def main():
    device = torch.device('cuda')
    frame = generate_evaluation_trajectory('/home/user/AnK_MeshGraphNets/raw_data/solid185/val/4mm_plate_data',device)
    print()
    print("========================================================================")
    print()
    # print(f"================ Total steps in this trajectory: {len(generated_trajectory)} ==================")
    # print(len(generated_trajectory))
    # print(generated_trajectory)
    print()
    print("========================================================================")
    print("=======  Following information are found in the generated frame  =======")
    print()
    for key in frame:
        print("            :-",key,"has value of shape",frame[key].shape)
        # if key == 'node_type':
        #     print(generated_trajectory[0][key])
    print()
    print("=========================== Used ANSYS data ============================")
    print("========================================================================")
    print()
    return

if __name__ == '__main__':
    main()