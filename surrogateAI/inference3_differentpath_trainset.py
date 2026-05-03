import os
import torch
import pickle
import time
import copy
from torch.utils.data import Dataset, DataLoader

# Import your custom modules
from models import press_model
from utilities import press_eval, common

# =====================================================================
# THE ULTIMATE MONKEY PATCH
# Intercepts _rollout to guarantee node_type is 2D right before line 14
# =====================================================================
original_rollout = press_eval._rollout

def safe_rollout(model, initial_state, num_steps, target_positions):
    # If node_type has been flattened to 1D [Nodes], force it back to [Nodes, 1]
    if 'node_type' in initial_state and initial_state['node_type'].dim() == 1:
        initial_state['node_type'] = initial_state['node_type'].unsqueeze(-1)
        
    return original_rollout(model, initial_state, num_steps, target_positions)

# Override the internal function with our safe version
press_eval._rollout = safe_rollout
# =====================================================================

device = torch.device('cuda')

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
        slice=64,
        stochastic_message_passing_used='False',
        k_neighbor=80,       
        dilated_k_sample=20  
    )
}

# ---------------------------------------------------------
# 1. CUSTOM DATASET TO BYPASS H5 AND DIRECTLY LOAD .PT
# ---------------------------------------------------------
class DirectPTDataset(Dataset):
    def __init__(self, pt_file_path):
        """Directly loads a processed .pt file, bypassing HDF5 entirely."""
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"Processed .pt file not found: {pt_file_path}")
        
        print(f"Loading data directly from: {pt_file_path}")
        self.data = torch.load(pt_file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------
# 2. DIMENSION-AWARE SQUEEZE FUNCTION
# ---------------------------------------------------------
def squeeze_data(data):
    """
    Removes the batch dimension and guarantees the [Steps, Nodes, Features] 
    shape required by the evaluator.
    """
    processed = {}
    for k, v in data.items():
        # Remove the DataLoader batch dimension: [1, ...] -> [...]
        tensor = v.squeeze(0)
        
        # FIX: If node_type is 1D [Nodes], make it 2D [Nodes, 1]
        if k == 'node_type' and tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
            
        # FIX: Simulate the 'Time/Steps' dimension for training data
        # Val data is [Steps, Nodes, Features]. Train is just [Nodes, Features].
        if tensor.dim() == 2 and k in ['curr_pos', 'next_pos', 'next_stress', 'mesh_pos', 'node_type']:
            tensor = tensor.unsqueeze(0)
            
        # For cells [Num_cells, 3], make it [1, Num_cells, 3]
        if k == 'cells' and tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        processed[k] = tensor.to(device)
    return processed

# ---------------------------------------------------------
# 3. CORE INFERENCE LOGIC 
# ---------------------------------------------------------
def load_model(inference_config, params, checkpoint_paths):
    core_name = inference_config.get('core_model', 'transolver')
    models = [params['model'].Model(params, core_model_name=core_name) for _ in range(3)]

    for stage_idx, (model, path) in enumerate(zip(models, checkpoint_paths)):
        prefix = path.replace('_learned_model.pth', '')
        if os.path.isdir(path):
            prefix = os.path.join(path, 'best_checkpoint', 'best_model_checkpoint')
        elif not os.path.exists(prefix + "_learned_model.pth"):
            # Fallback if pointing directly to a directory without best_checkpoint logic
            prefix = os.path.join(path, 'best_model_checkpoint')
        
        try:
            model.load_model(prefix)
            print(f"[Stage {stage_idx+1}] ✓ Loaded model from {prefix}")
        except Exception as e:
            print(f"[Stage {stage_idx+1}] ✗ Failed to load from {prefix}: {e}")

        # Monkey patch model's forward to ignore 'is_training'
        original_forward = model.learned_model.forward
        def patched_forward(*args, **kwargs):
            kwargs.pop('is_training', None)
            return original_forward(*args, **kwargs)
        model.learned_model.forward = patched_forward
        model.to(device).eval()

    return models

def concat_trajectories(trajectories1, trajectories2, trajectories3):
    concatenated_results = []
    for t1, t2, t3 in zip(trajectories1, trajectories2, trajectories3):
        sample_dict = {}
        for key in t1.keys():
            v1, v2, v3 = t1[key], t2[key], t3[key]
            if isinstance(v1, torch.Tensor):
                sample_dict[key] = torch.cat([v1, v2, v3], dim=0)
            else:
                sample_dict[key] = v1
        concatenated_results.append(sample_dict)
    return concatenated_results

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("="*70)
    print("TRAIN DATASET INFERENCE (DIRECT .PT LOAD)")
    print("="*70)

    # 1. Define Model Checkpoints
    checkpoint_paths = [
        '/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E2_F/transolver/fine_1500_train_val/Thu-Apr--2-04-54-02-2026/checkpoint',
        '/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E2_F/transolver/fine_1500_train_val/Wed-Apr--1-03-14-14-2026/checkpoint',
        '/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E2_F/transolver/fine_1500_train_val/Wed-Apr--1-17-31-25-2026/checkpoint'
    ]

    # 2. Define Output Directory
    output_dir = '/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E2_F/Combined_Inference'
    os.makedirs(output_dir, exist_ok=True)

    # 3. Define the exact paths to your existing TRAIN .pt files
    base_data_dir = '/home/sushil/PressNet/datasets/data/Fine'
    pt_files = [
        os.path.join(base_data_dir, 'fine_1500_train_val_train_1.pt'),
        os.path.join(base_data_dir, 'fine_1500_train_val_train_2.pt'),
        os.path.join(base_data_dir, 'fine_1500_train_val_train_3.pt')
    ]

    # Load Models
    inference_config = {'core_model': 'transolver'}
    params = PARAMETERS['press']
    models = load_model(inference_config, params, checkpoint_paths)

    # Prepare DataLoaders using the custom direct-load dataset
    loaders = [DataLoader(DirectPTDataset(pt_file), batch_size=1, shuffle=False) for pt_file in pt_files]

    trajectories = [[], [], []]
    
    print("\nStarting continuous inference over stitched stages...")
    
    # Run Inference
    for count, (batch1, batch2, batch3) in enumerate(zip(*loaders)):
        start_time = time.time()
        print(f"Processing Train Trajectory {count}...", end=" ", flush=True)

        inputs = [squeeze_data(batch1), squeeze_data(batch2), squeeze_data(batch3)]
        preds = []

        # Stage 1
        _, pred1 = params['evaluator'].evaluate(models[0], inputs[0])
        preds.append(pred1)

        # Stage 2 (Stitched)
        in2 = copy.deepcopy(inputs[1])
        in2['curr_pos'][0] = pred1['pred_pos'][-1]
        _, pred2 = params['evaluator'].evaluate(models[1], in2)
        pred2['gt_pos'][0] = inputs[1]['curr_pos'][0] # Restore GT
        preds.append(pred2)

        # Stage 3 (Stitched)
        in3 = copy.deepcopy(inputs[2])
        in3['curr_pos'][0] = pred2['pred_pos'][-1]
        _, pred3 = params['evaluator'].evaluate(models[2], in3)
        pred3['gt_pos'][0] = inputs[2]['curr_pos'][0] # Restore GT
        preds.append(pred3)

        # Save results to lists
        for i in range(3):
            trajectories[i].append(preds[i])

        print(f"✓ ({time.time() - start_time:.2f}s)")

    print(f"\nConcatenating {len(trajectories[0])} trajectories...")
    combined = concat_trajectories(trajectories[0], trajectories[1], trajectories[2])

    print(f"Saving concatenated rollout to {output_dir}...")
    output_file = os.path.join(output_dir, 'concatenated_rollout_train.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(combined, f)

    print(f"✓ Saved to {output_file}")
    print("="*70)
    print("INFERENCE COMPLETE.")
    print("="*70)

if __name__ == '__main__':
    main()