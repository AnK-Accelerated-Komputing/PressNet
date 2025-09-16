## inference for the distributed model architecture
## need to change a lot, just copying old code for now

from models import press_model
from utilities import press_eval
from utilities.dataset import TrajectoryDataset
from torch.utils.data import DataLoader


import os
import torch
import pickle
import time
import json
import copy

device = torch.device('cuda')

PARAMETERS = {'press': dict(noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=4, batch=2, model=press_model, evaluator=press_eval, loss_type='deform',
                  stochastic_message_passing_used='False')}

def load_model(inference_config,params,checkpoint_dir):
    
    
    
    # create or load model
    model1 = params['model'].Model(params, inference_config['core_model'], inference_config['message_passing_aggregator'],
                                        inference_config['message_passing_steps'], inference_config['attention'])
    model2 = params['model'].Model(params, inference_config['core_model'], inference_config['message_passing_aggregator'],
                                        inference_config['message_passing_steps'], inference_config['attention'])
    model3 = params['model'].Model(params, inference_config['core_model'], inference_config['message_passing_aggregator'],
                                        inference_config['message_passing_steps'], inference_config['attention'])


    model1_checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_1') #'/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/final_checkpoints/checkpoint_stress_000_133'
    model2_checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_2') #'/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/final_checkpoints/checkpoint_stress_133-266'
    model3_checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_3') #'/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/final_checkpoints/checkpoint_stress_266_399'

    model1.load_model(os.path.join(model1_checkpoint_dir, 'model_checkpoint'))
    model2.load_model(os.path.join(model2_checkpoint_dir, 'model_checkpoint'))
    model3.load_model(os.path.join(model3_checkpoint_dir, 'model_checkpoint'))
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model1.eval()
    model2.eval()
    model3.eval()
    return model1,model2,model3


def obtain_infer_traj(inference_config,checkpoint_dir, val_dir,continuous):
    params = PARAMETERS[inference_config['model']]
    model1,model2,model3 = load_model(inference_config,params,checkpoint_dir)

    if continuous:
        #for continuous rollout
        val_traj_folders = [
            os.path.join(val_dir, name)
            for name in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, name))
            ] 
        val_dataset = TrajectoryDataset(val_dir, split='val', stage=1)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        info = {
            'index': [],
            'file_name' : [],
            'inference_time' : [],

        }
        trajectories1 = []
        trajectories2 = []
        trajectories3 = []
        count=0
        for folder_path in val_traj_folders: #change
            info['index'].append(count)
            info['file_name'].append(os.path.basename(folder_path))
            count+=1
            start_time = time.time()
            print("Evaluating trajectory " + str(count))
            # trajectory = next(ds_iterator)
            trajectory1 = generate_evaluation_trajectory(folder_path, device,stage=1)
            _, prediction_trajectory1 = params['evaluator'].evaluate(model1, trajectory1)
            trajectory2 = generate_evaluation_trajectory(folder_path, device,stage=2)
            trajectory2_temp = copy.deepcopy(trajectory2)
            trajectory2_temp['world_pos'][0] = prediction_trajectory1['pred_pos'][-1]
            # pred_pos tensor size: torch.Size([399, 445, 3])
            _, prediction_trajectory2 = params['evaluator'].evaluate(model2, trajectory2_temp)
            trajectory3 = generate_evaluation_trajectory(folder_path, device,stage=3)
            trajectory3_temp = copy.deepcopy(trajectory3)
            trajectory3_temp['world_pos'][0] = prediction_trajectory2['pred_pos'][-1]
            _, prediction_trajectory3 = params['evaluator'].evaluate(model3, trajectory3_temp)
            trajectories1.append(prediction_trajectory1)
            trajectories2.append(prediction_trajectory2)
            trajectories3.append(prediction_trajectory3)
            end_time = time.time()
            info['inference_time'].append(end_time - start_time)
    else:
        ## for discontinuous rollout
        val_traj_folders = [
            os.path.join(val_dir, name)
            for name in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, name))
            ] 

        trajectories1 = []
        trajectories2 = []
        trajectories3 = []
        info = {
            'index': [],
            'file_name' : [],
            'inference_time' : [],

        }
        count=0
        for folder_path in val_traj_folders: #change
            info['index'].append(count)
            info['file_name'].append(os.path.basename(folder_path))
            count+=1
            start_time = time.time()
            print("Evaluating trajectory " + str(count))
            # trajectory = next(ds_iterator)
            trajectory1 = generate_evaluation_trajectory(folder_path, device,stage=1)
            trajectory2 = generate_evaluation_trajectory(folder_path, device,stage=2)
            trajectory3 = generate_evaluation_trajectory(folder_path, device,stage=3)
            _, prediction_trajectory1 = params['evaluator'].evaluate(model1, trajectory1)
            _, prediction_trajectory2 = params['evaluator'].evaluate(model2, trajectory2)
            _, prediction_trajectory3 = params['evaluator'].evaluate(model3, trajectory3)
            trajectories1.append(prediction_trajectory1)
            trajectories2.append(prediction_trajectory2)
            trajectories3.append(prediction_trajectory3)
            end_time = time.time()
            info['inference_time'].append(end_time - start_time)
    return trajectories1,trajectories2,trajectories3,info

def save_rollout(rollout_path,trajectories,disp=""):
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
        print(f" rollout {disp} saved to {rollout_path}")
    return

def concat_trajectories(trajectories1,trajectories2,trajectories3):

    data_dicts = []
    data_dicts.append(trajectories1)
    data_dicts.append(trajectories2)
    data_dicts.append(trajectories3)    

    num_dicts = len(data_dicts[0])
    # Create an empty list for the concatenated dictionaries
    concatenated_dicts = [{} for _ in range(num_dicts)]

    # Iterate over each dictionary index
    for i in range(num_dicts):
        # Get the list of keys from the first dictionary
        keys = data_dicts[0][i].keys()
        for key in keys:
            # Concatenate the tensors for the current key across all files
            concatenated_tensors = torch.cat([d[i][key] for d in data_dicts], dim=0)
            # Add the concatenated tensor to the new dictionary
            concatenated_dicts[i][key] = concatenated_tensors

    return concatenated_dicts

def inference(checkpoint_dir, val_dir,rollout_dir,continuous=True,animation=False,animation_html=False,evaluate=False):
    inference_config = {'model': 'press', 'mode': 'all', 'rollout_split': 'valid',
                    'dataset': 'deforming_plate', 'epochs': 2000, 'trajectories': 7,
                    'num_rollouts': 3, 'core_model': 'encode_process_decode',
                    'message_passing_aggregator': 'sum',
                    'message_passing_steps': 16, 'attention': False,
                    # 'dataset_dir': dataset_dir,
                    'last_run_dir': None}

    
    trajectories1,trajectories2,trajectories3,info = obtain_infer_traj(inference_config,checkpoint_dir, val_dir,continuous)
    concatenated_trajectories = concat_trajectories(trajectories1,trajectories2,trajectories3)

    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f,indent=4)
    
    save_rollout(os.path.join(rollout_dir, 'rollout_000-133.pkl'),trajectories1,disp="000-133")
    save_rollout(os.path.join(rollout_dir, 'rollout_133-266.pkl'),trajectories2,disp="133-266")
    save_rollout(os.path.join(rollout_dir, 'rollout_266-399.pkl'),trajectories3,disp="266-399")
    output_file = os.path.join(rollout_dir, 'concatenated_rollout_all.pkl')
    save_rollout(output_file,concatenated_trajectories,disp="concatenated")

    

    animation_save_path = os.path.join(rollout_dir, 'visualization')
    os.makedirs(animation_save_path, exist_ok=True)

    if evaluate:
        from utilities.evaluation import evaluate_rollout
        evaluate_dir = os.path.join(rollout_dir,'rollout_evaluation')
        evaluate_rollout(output_file, evaluate_dir,plot=True)

    if animation:
        from postprocessing.animate_compare import animate_rollout
        for i in range(len(concatenated_trajectories)):
            animation_save_path = os.path.join(rollout_dir, 'visualization',f"{i}")
            os.makedirs(animation_save_path, exist_ok=True)
            print("created folder:",animation_save_path)
            animate_rollout(output_file, animation_save_path,i=i,key='stress')


    if animation_html:
        from postprocessing.animate_compare_html import animate_rollout as animate_rollout_html
        animate_rollout_html(output_file, animation_save_path)
    
    return

def main():
    checkpoint_dir = '/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/final_checkpoints/final_checkpoints_combined'
    val_dir = '/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/raw_data/data_combined/val'
    rollout_dir = '/home/gd_user1/AnK/project_PINN/AnK_MeshGraphNets/test_results/combined'

    inference(checkpoint_dir, val_dir,rollout_dir,continuous=True,animation=False,animation_html=False,evaluate=True)

    return


if __name__ == '__main__':
    main()