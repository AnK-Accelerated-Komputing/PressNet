import os
import torch
import pickle
import time
import json

import numpy as np

from model import press_model, thermal_model, del_thermal_model
from utilities import press_eval, common 

Thermal = True
two_D = False 
Quarter = False
if Thermal:
    from utilities.evaluation_thermal import evaluate_rollout

    if two_D:
        from postprocessing.animate_compare_thermal2d import animate_rollout

        from preprocessing.trajectory_thermal2d import generate_trajectory
        from utilities import thermal2d_eval as thermal_eval
    else:
        from postprocessing.animate_compare_thermal import animate_rollout

        from utilities import thermal_eval
        from preprocessing.trajectory_thermal import generate_trajectory
    from preprocessing.evaluation_trajectory_thermal import generate_evaluation_trajectory
else:    
    from utilities.evaluation import evaluate_rollout
    from postprocessing.animate_compare import animate_rollout
   
    if Quarter:
        from preprocessing.trajectory_solid185_quarter import generate_trajectory
    else:
        from preprocessing.trajectory_solid185 import generate_trajectory
    from preprocessing.evalutation_trajectory import generate_evaluation_trajectory

device = torch.device('cuda')

PARAMETERS = {
    'press': dict(noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=4, batch=2, model=press_model, evaluator=press_eval, loss_type='deform',
                  stochastic_message_passing_used='False'),
    'thermal': dict(noise=0.02, gamma=1.0, field='temp', history=False,
                size=1, batch=2, model=thermal_model, evaluator=thermal_eval, loss_type='thermal',
                stochastic_message_passing_used='False'),
    'del_thermal': dict(noise=0.02, gamma=1.0, field='temp', history=False,
                size=1, batch=2, model=del_thermal_model, evaluator=thermal_eval, loss_type='del_thermal',
                stochastic_message_passing_used='False')
    }

def load_model(inference_config,params,checkpoint_dir):
    
    
    
    # create or load model
    model = params['model'].Model(params, inference_config['core_model'], inference_config['message_passing_aggregator'],
                                        inference_config['message_passing_steps'], inference_config['attention'])


    model.load_model(os.path.join(checkpoint_dir,'model_checkpoint'))
    model.to(device)
    model.eval()
    return model

def obtain_infer_traj(inference_config,checkpoint_dir, val_dir,continuous):
    params = PARAMETERS[inference_config['model']]
    model= load_model(inference_config,params,checkpoint_dir)

    val_traj_folders = [
        os.path.join(val_dir, name)
        for name in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, name))
        ] 
        
    info = {
        'index': [],
        'file_name' : [],
        'inference_time' : [],

    }
    trajectories = []
    count=0
    for folder_path in val_traj_folders: #change
        info['index'].append(count)
        info['file_name'].append(os.path.basename(folder_path))
        count+=1
        start_time = time.time()
        print("Evaluating trajectory " + str(count))
        # trajectory = next(ds_iterator)
        trajectory = generate_evaluation_trajectory(folder_path, device,key="thermal") #stage optional
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        trajectories.append(prediction_trajectory)
        end_time = time.time()
        info['inference_time'].append(end_time - start_time)
    return trajectories, info

def save_rollout(rollout_path,trajectories,disp=""):
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
        print(f" rollout {disp} saved to {rollout_path}")
    return

def inference(checkpoint_dir, val_dir,rollout_dir,continuous=True,animation=False,animation_html=False,evaluate=False):
    inference_config = {'model': 'del_thermal', 'mode': 'all', 'rollout_split': 'valid',
                    'dataset': 'deforming_plate', 'epochs': 2000, 'trajectories': 7,
                    'num_rollouts': 3, 'core_model': 'encode_process_decode',
                    'message_passing_aggregator': 'sum',
                    'message_passing_steps': 16, 'attention': False,
                    # 'dataset_dir': dataset_dir,
                    'last_run_dir': None}

    
    trajectories, info = obtain_infer_traj(inference_config,checkpoint_dir, val_dir,continuous)

    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f,indent=4)
    output_file = os.path.join(rollout_dir, 'rollout.pkl')
    save_rollout(output_file,trajectories,disp="theraml2d") #disp for print

    

    animation_save_path = os.path.join(rollout_dir, 'visualization')
    os.makedirs(animation_save_path, exist_ok=True)

    if evaluate:
        evaluate_dir = os.path.join(rollout_dir,'rollout_evaluation')
        evaluate_rollout(output_file, evaluate_dir,plot=True)

    if animation:
        for i in range(len(trajectories)):
            animation_save_path = os.path.join(rollout_dir, 'visualization',f"{i}")
            os.makedirs(animation_save_path, exist_ok=True)
            print("created folder:",animation_save_path)
            animate_rollout(output_file, animation_save_path,i=i,key='stress') #use key for different fields (not necessary for thermal


    if animation_html:
        # not for 2d currently
        from postprocessing.animate_compare_html import animate_rollout as animate_rollout_html
        animate_rollout_html(output_file, animation_save_path)
    
    return

def main():
    checkpoint_dir = '/home/user/AnK_MeshGraphNets/output/press_3d/Wed-Jan-15-05-13-20-2025/1/checkpoint'
    val_dir = '/home/user/AnK_MeshGraphNets/raw_data/DATA_COMBINED/train'
    rollout_dir = '/home/user/AnK_MeshGraphNets/test_results/combined_thermal_train'

    inference(checkpoint_dir, val_dir,rollout_dir,continuous=True,animation=False,animation_html=False,evaluate=False)

    return


if __name__ == '__main__':
    main()