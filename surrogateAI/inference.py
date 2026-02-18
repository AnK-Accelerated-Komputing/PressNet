import os
import time
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import press_model
from utilities import press_eval
from utilities.dataset import TrajectoryDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. FIXED: Params matched to training code ---
PARAMETERS = {
    'press': dict(
        field='world_pos',
        size=4,             # Changed from 3 to 4 to match training input
        model=press_model,
        evaluator=press_eval,
        k_neighbor=50,      # Added: Required for graph construction
        dilated_k_sample=100, # Added: Required for graph construction
        message_passing_steps=10,
        world_radius=30,
        slice=64,
        edge_dim=8
    )
}

def save_rollout(rollout_path, trajectories, disp=""):
    """Save predicted trajectories to a pickle file."""
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Rollout {disp} saved to {rollout_path}")

def load_model(inference_config, params, checkpoint_dir):
    """Load the model from a checkpoint."""
    
    # --- 2. FIXED: Initialization signature matches training ---
    model = params['model'].Model(
        params, 
        core_model_name=inference_config['core_model']
    )
    
    # Check if we are pointing to the parent folder or the specific best_checkpoint folder
    # Based on your previous code, the file is inside "best_checkpoint"
    if "best_checkpoint" not in checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "best_checkpoint")):
        load_path = os.path.join(checkpoint_dir, "best_checkpoint", 'best_model_checkpoint')
    else:
        load_path = os.path.join(checkpoint_dir, 'best_model_checkpoint')

    print(f"Loading weights from: {load_path}")
    model.load_model(load_path)
    
    model.to(device)
    model.eval()
    return model

def get_meta_data(data_path):
    """Load metadata JSON corresponding to the dataset."""
    base_folder = os.path.dirname(data_path)
    data_name = os.path.basename(data_path).split('.')[0]
    # Handle cases where meta file might not match exact data name if needed
    meta_file = os.path.join(base_folder, f'{data_name}_meta.json')
    
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
        return meta_data
    else:
        print(f"Warning: Meta file {meta_file} not found.")
        return {}

def inference(checkpoint_dir, val_dir, rollout_dir):
    """Run inference on dataset and save rollout."""
    os.makedirs(rollout_dir, exist_ok=True)

    # Inference configuration
    inference_config = {
        "model": "press",
        "core_model": "transolver", # Ensure this matches the trained model type
    }

    # Load dataset
    # Note: Ensure your dataset supports split='test'. If not, change to 'val'.
    val_dataset = TrajectoryDataset(val_dir, split='test', stage=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    params = PARAMETERS[inference_config['model']]
    
    # Load Model
    model = load_model(inference_config, params, checkpoint_dir)

    meta_data = get_meta_data(val_dir)
    groups = sorted(list(meta_data.keys())) if meta_data else []
    
    info = {
        'index': [],
        'file_name' : [],
        'inference_time' : [],
    }

    # Run inference
    trajectories = []
    print(f"Starting inference on {len(val_loader)} samples...")
    
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            info['index'].append(idx)
            
            # Safe handling if metadata is missing
            if idx < len(groups):
                group_name = groups[idx]
                fname = meta_data[group_name].get('data type', 'unknown')
                info['file_name'].append(fname)
            else:
                info['file_name'].append(f"sample_{idx}")
            
            start_time = time.time()
            _, prediction_trajectory = press_eval.evaluate(model, data)
            end_time = time.time()

            trajectories.append(prediction_trajectory)
            info['inference_time'].append(end_time - start_time)

            print(f"Processed sample {idx} in {end_time - start_time:.2f}s")

    # Save predictions
    rollout_file = Path(rollout_dir) / "infer_rollout.pkl"
    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    save_rollout(str(rollout_file), trajectories)

def main():
    # Define directories
    # Ensure this points to the folder containing 'best_checkpoint' or the checkpoint itself
    checkpoint_dir = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/Transolver_C_ST1_Slice64/checkpoint"
    
    val_dir = r"/home/sushil/PressNet/datasets/data/coarse/coarse_1500_train_val.h5"

    rollout_dir = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/infer_rollout"
    # Run inference
    inference(checkpoint_dir, val_dir, rollout_dir)

if __name__ == "__main__":
    main()