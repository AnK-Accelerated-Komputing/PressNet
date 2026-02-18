import os
import torch
import pickle
import time
import json
import copy
from torch.utils.data import DataLoader

# Import your custom modules
from models import press_model
from utilities import press_eval, common
# Use the Dataset class for loading .h5/.pt files directly
from utilities.dataset import TrajectoryDataset 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [UPDATE]: Added k_neighbor and dilated_k_sample required for specific architectures
PARAMETERS = {
    'press': dict(
        noise=0.003, 
        gamma=1.0, 
        field='world_pos', 
        history=False,
        size=4, 
        batch=1, 
        model=press_model, 
        evaluator=press_eval, 
        loss_type='deform',
        stochastic_message_passing_used='False',
        k_neighbor=10,       
        dilated_k_sample=20  
    )
}

def load_model(inference_config, params, checkpoint_dir):
    """
    Loads three separate models for Stage 1, Stage 2, and Stage 3.
    """
    # 1. Initialize the model architecture
    core_name = inference_config.get('core_model', 'transolver')
    
    model1 = params['model'].Model(params, core_model_name=core_name)
    model2 = params['model'].Model(params, core_model_name=core_name)
    model3 = params['model'].Model(params, core_model_name=core_name)

    # 2. Define paths for each stage's checkpoint
    model1_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_1') 
    model2_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_2') 
    model3_dir = os.path.join(checkpoint_dir, 'checkpoint_stress_stage_3') 

    # 3. Helper function to check files before loading
    def safe_load(model, folder_path, stage_name):
        best_prefix = os.path.join(folder_path, 'best_model_checkpoint')
        final_prefix = os.path.join(folder_path, 'model_checkpoint')

        # Try loading BEST first
        if os.path.exists(best_prefix + "_learned_model.pth"):
            print(f"[{stage_name}] Found BEST checkpoint. Loading...")
            try:
                model.load_model(best_prefix)
                print(f"[{stage_name}] Success: Loaded Best model.")
                return
            except Exception as e:
                print(f"[{stage_name}] Warning: Error loading Best model ({e}).")

        # Fallback to FINAL
        if os.path.exists(final_prefix + "_learned_model.pth"):
            print(f"[{stage_name}] Loading FINAL checkpoint...")
            try:
                model.load_model(final_prefix)
                print(f"[{stage_name}] Success: Loaded Final model.")
                return
            except Exception as e:
                print(f"[{stage_name}] Error loading Final model ({e}).")
        
        print(f"CRITICAL ERROR: No checkpoint found in {folder_path}")

    # 4. Load weights
    print("\n--- Loading Models ---")
    safe_load(model1, model1_dir, "Stage 1")
    safe_load(model2, model2_dir, "Stage 2")
    safe_load(model3, model3_dir, "Stage 3")
    print("----------------------\n")

    # 5. Apply "Monkey Patch" to fix 'is_training' crash
    def apply_patch(wrapper_model):
        original_forward = wrapper_model.learned_model.forward
        
        def patched_forward(*args, **kwargs):
            # Silently remove 'is_training' argument if present
            kwargs.pop('is_training', None)
            return original_forward(*args, **kwargs)
        
        wrapper_model.learned_model.forward = patched_forward

    apply_patch(model1)
    apply_patch(model2)
    apply_patch(model3)

    # 6. Move to GPU
    model1.to(device).eval()
    model2.to(device).eval()
    model3.to(device).eval()
    
    return model1, model2, model3

def squeeze_data(data):
    """
    Removes the batch dimension added by DataLoader.
    DataLoader shape: [1, Time, Node, Feature] -> Model expects: [Time, Node, Feature]
    """
    return {k: v.squeeze(0).to(device) for k, v in data.items()}

def obtain_infer_traj(inference_config, checkpoint_dir, val_file_path, continuous, split_type='val'):
    """
    Runs inference by loading data from a single .h5/.pt file, splitting it into 3 stages.
    Accepts 'split_type' to pass correctly to the Dataset class.
    """
    params = PARAMETERS[inference_config['model']]
    model1, model2, model3 = load_model(inference_config, params, checkpoint_dir)

    print(f"\nPreparing DataLoaders for file: {val_file_path}")
    print(f"Mode: {split_type.upper()}")
    
    # Create 3 datasets pointing to the SAME file, but different 'stage' slices
    # Pass 'split=split_type' so TrajectoryDataset knows if it's 'val' or 'test'
    ds1 = TrajectoryDataset(val_file_path, split=split_type, stage=1)
    ds2 = TrajectoryDataset(val_file_path, split=split_type, stage=2)
    ds3 = TrajectoryDataset(val_file_path, split=split_type, stage=3)

    # shuffle=False is MANDATORY to align trajectories across stages
    loader1 = DataLoader(ds1, batch_size=1, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=1, shuffle=False)
    loader3 = DataLoader(ds3, batch_size=1, shuffle=False)

    trajectories1, trajectories2, trajectories3 = [], [], []
    info = {'index': [], 'inference_time': []}
    
    print(f"Starting inference on {len(loader1)} trajectories...")
    count = 0
    
    # Iterate through all 3 stages simultaneously using zip
    for batch_stage1, batch_stage2, batch_stage3 in zip(loader1, loader2, loader3):
        start_time = time.time()
        
        # 1. Unpack Batch (Remove [1, ...] dimension)
        traj1_input = squeeze_data(batch_stage1)
        traj2_input = squeeze_data(batch_stage2)
        traj3_input = squeeze_data(batch_stage3)
        
        info['index'].append(count)
        print(f"Processing {split_type} Trajectory {count}...")

        # --- RUN STAGE 1 ---
        _, pred_traj1 = params['evaluator'].evaluate(model1, traj1_input)

        # --- RUN STAGE 2 ---
        if continuous:
            # Stitching: Overwrite start position with Stage 1's end position
            traj2_input_stitched = copy.deepcopy(traj2_input)
            traj2_input_stitched['curr_pos'][0] = pred_traj1['pred_pos'][-1]
            _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input_stitched)
        else:
            _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input)

        # --- RUN STAGE 3 ---
        if continuous:
            # Stitching: Overwrite start position with Stage 2's end position
            traj3_input_stitched = copy.deepcopy(traj3_input)
            traj3_input_stitched['curr_pos'][0] = pred_traj2['pred_pos'][-1]
            _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input_stitched)
        else:
            _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input)

        # Store results
        trajectories1.append(pred_traj1)
        trajectories2.append(pred_traj2)
        trajectories3.append(pred_traj3)

        info['inference_time'].append(time.time() - start_time)
        count += 1

    return trajectories1, trajectories2, trajectories3, info

def save_rollout(rollout_path, trajectories, disp=""):
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
        print(f"Saved {disp} rollout to: {rollout_path}")

def concat_trajectories(trajectories1, trajectories2, trajectories3):
    """
    Merges the list of trajectory dictionaries from 3 stages into one.
    """
    num_samples = len(trajectories1)
    concatenated_results = []

    for i in range(num_samples):
        sample_dict = {}
        keys = trajectories1[i].keys()
        
        for key in keys:
            t1 = trajectories1[i][key]
            t2 = trajectories2[i][key]
            t3 = trajectories3[i][key]
            # Concatenate along time dimension (dim 0)
            sample_dict[key] = torch.cat([t1, t2, t3], dim=0)
            
        concatenated_results.append(sample_dict)

    return concatenated_results

def inference(checkpoint_dir, val_file_path, rollout_dir, continuous=True, evaluate=False, animation=False, animation_html=False, split_type='val'):
    
    # CONFIGURATION
    inference_config = {
        'model': 'press', 
        'mode': 'all', 
        'core_model': 'transolver', 
        'aggregator': 'sum',
        'steps': 16, 
        'attention': False
    }

    # 1. Run the Multi-Stage Inference
    # Pass split_type down so Dataset knows what to load
    trajs1, trajs2, trajs3, info = obtain_infer_traj(
        inference_config, checkpoint_dir, val_file_path, continuous, split_type
    )
    
    # 2. Stitch the stages together
    combined_trajs = concat_trajectories(trajs1, trajs2, trajs3)

    # 3. Save Results
    os.makedirs(rollout_dir, exist_ok=True)
    
    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_1.pkl'), trajs1, disp="Stage 1")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_2.pkl'), trajs2, disp="Stage 2")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_3.pkl'), trajs3, disp="Stage 3")
    
    output_file = os.path.join(rollout_dir, 'concatenated_rollout_all.pkl')
    save_rollout(output_file, combined_trajs, disp="Concatenated All")

    # 4. Evaluation (Calculate Errors)
    if evaluate:
        try:
            from utilities.evaluation_stress import evaluate_rollout
            evaluate_dir = os.path.join(rollout_dir, 'rollout_evaluation')
            print(f"\nRunning Evaluation for {split_type}...")
            evaluate_rollout(output_file, evaluate_dir, plot=True)
        except ImportError:
            print("Evaluation module not found, skipping.")

    # 5. Video Animation
    if animation:
        try:
            from utilities.animate_compare import animate_rollout
            vis_root = os.path.join(rollout_dir, 'visualization')
            os.makedirs(vis_root, exist_ok=True)
            print(f"\nGenerating Video Animations for {split_type}...")
            
            for i in range(len(combined_trajs)):
                traj_vis_path = os.path.join(vis_root, f"{i}")
                os.makedirs(traj_vis_path, exist_ok=True)
                animate_rollout(output_file, traj_vis_path, i=i, key='stress')
        except ImportError:
            print("Animation module not found, skipping.")

    # 6. HTML Animation
    if animation_html:
        try:
            from utilities.animate_compare_html import animate_rollout as animate_rollout_html
            vis_root = os.path.join(rollout_dir, 'visualization')
            os.makedirs(vis_root, exist_ok=True)
            print(f"\nGenerating HTML Visualization for {split_type}...")
            animate_rollout_html(output_file, vis_root)
        except ImportError:
            print("HTML Animation module not found, skipping.")

    print(f"\nInference Complete for {split_type}.")
    return

def main():
    # --- USER SETTINGS ---
    
    # 1. Checkpoint Directory
    checkpoint_dir = '/home/ank-server6/PINN/PressNet/datasets/final_checkpoints/transolver'
    
    # 2. Base Output Directory
    base_rollout_dir = '/home/ank-server6/PINN/PressNet/datasets/final_checkpoints/transolver/h5_inference'

    # 3. Define Tasks: (Split Name, File Path)
    # Ensure these paths are correct on your server
    tasks = [
        {
            'split': 'val', 
            'file': '/home/ank-server6/PINN/PressNet/datasets/data/coarse_1500_train_val.h5'
        },
        {
            'split': 'test', 
            'file': '/home/ank-server6/PINN/PressNet/datasets/data/coarse_1500_test.h5'
        }
    ]

    # --- EXECUTION LOOP ---
    for task in tasks:
        split_name = task['split']
        file_path = task['file']
        
        # Simple check to avoid crashing if file is missing
        if not os.path.exists(file_path):
            print(f"\n[SKIP] File not found: {file_path}")
            continue

        # Create subfolder: e.g. .../h5_inference/val or .../h5_inference/test
        current_rollout_dir = os.path.join(base_rollout_dir, split_name)
        
        
        print(f" STARTING TASK: {split_name.upper()}")
        print(f" Input File: {file_path}")
        print(f" Output Dir: {current_rollout_dir}")
        

        inference(
            checkpoint_dir=checkpoint_dir, 
            val_file_path=file_path, 
            rollout_dir=current_rollout_dir, 
            continuous=True,      # True = Feed output of Stage 1 to Stage 2
            evaluate=True,        # Calculate RMSE/Loss
            animation=False,      # Generate MP4/GIFs
            animation_html=False, # Generate Interactive HTML
            split_type='test' # Pass 'val' or 'test'
        )

if __name__ == '__main__':
    main()