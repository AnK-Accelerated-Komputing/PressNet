import os
import torch
import pickle
import time
import json
import copy
import argparse
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
        slice=32,
        stochastic_message_passing_used='False',
        k_neighbor=10,       
        dilated_k_sample=20  
    )
}

def load_model(inference_config, params, checkpoint_dir):
    """
    Loads three separate models for Stage 1, Stage 2, and Stage 3.
    
    Expected directory structure:
        checkpoint_dir/
            ├── checkpoint_stress_stage_1/
            │   └── best_model_checkpoint_learned_model.pth
            ├── checkpoint_stress_stage_2/
            │   └── best_model_checkpoint_learned_model.pth
            └── checkpoint_stress_stage_3/
                └── best_model_checkpoint_learned_model.pth
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
        """Safely load model with fallback logic."""
        best_prefix = os.path.join(folder_path, 'best_model_checkpoint')
        final_prefix = os.path.join(folder_path, 'model_checkpoint')

        # Try loading BEST first
        if os.path.exists(best_prefix + "_learned_model.pth"):
            print(f"[{stage_name}] Found BEST checkpoint. Loading...")
            try:
                model.load_model(best_prefix)
                print(f"[{stage_name}] ✓ Success: Loaded Best model.")
                return True
            except Exception as e:
                print(f"[{stage_name}] ✗ Warning: Error loading Best model ({e}).")

        # Fallback to FINAL
        if os.path.exists(final_prefix + "_learned_model.pth"):
            print(f"[{stage_name}] Loading FINAL checkpoint...")
            try:
                model.load_model(final_prefix)
                print(f"[{stage_name}] ✓ Success: Loaded Final model.")
                return True
            except Exception as e:
                print(f"[{stage_name}] ✗ Error loading Final model ({e}).")
        
        print(f"✗ CRITICAL ERROR: No checkpoint found in {folder_path}")
        return False

    # 4. Load weights
    print("\n" + "="*60)
    print("--- Loading Models ---")
    print("="*60)
    success1 = safe_load(model1, model1_dir, "Stage 1 (Press)")
    success2 = safe_load(model2, model2_dir, "Stage 2 (Dwell)")
    success3 = safe_load(model3, model3_dir, "Stage 3 (Release)")
    
    if not (success1 and success2 and success3):
        raise RuntimeError(f"Failed to load all 3 models. Check checkpoint structure:\n"
                         f"  Stage 1: {model1_dir}\n"
                         f"  Stage 2: {model2_dir}\n"
                         f"  Stage 3: {model3_dir}")
    
    print("="*60 + "\n")

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

def validate_stage_alignment(loader1, loader2, loader3, num_samples=3):
    """
    Validate that stages load the same trajectories by checking tensor shapes.
    Tests first 'num_samples' batches to catch alignment issues early.
    """
    print("\n[VALIDATION] Checking stage data alignment...")
    
    for i, (batch1, batch2, batch3) in enumerate(zip(loader1, loader2, loader3)):
        if i >= num_samples:
            break
            
        # Extract shapes from key tensors
        cells_shape_1 = batch1['cells'].shape
        cells_shape_2 = batch2['cells'].shape
        cells_shape_3 = batch3['cells'].shape
        
        if not (cells_shape_1 == cells_shape_2 == cells_shape_3):
            raise ValueError(
                f"Stage misalignment detected at batch {i}!\n"
                f"  Stage 1 cells shape: {cells_shape_1}\n"
                f"  Stage 2 cells shape: {cells_shape_2}\n"
                f"  Stage 3 cells shape: {cells_shape_3}\n"
                f"Possible causes:\n"
                f"  1. Cache files (*_val_1.pt, *_val_2.pt, *_val_3.pt) are out of sync\n"
                f"  2. Different dataset versions were used\n"
                f"  3. Data processing was interrupted"
            )
        
        print(f"  Batch {i}: ✓ Aligned (shape: {cells_shape_1})")
    
    print("✓ Stage alignment validation PASSED\n")

def validate_stage_continuity(pred_traj1, pred_traj2, pred_traj3, traj2_input_stitched, traj3_input_stitched):
    """
    Check for discontinuities at stage boundaries.
    Returns warnings if large gaps are detected.
    """
    warnings = []
    
    # Stage 1 -> Stage 2 boundary
    stage1_end_pos = pred_traj1['pred_pos'][-1]  # [num_nodes, 3]
    stage2_start_pos = traj2_input_stitched['curr_pos'][0]  # [num_nodes, 3]
    pos_gap_1_2 = torch.norm(stage1_end_pos - stage2_start_pos).item()
    
    if pos_gap_1_2 > 1e-2:
        warnings.append(f"⚠️  Stage 1→2 position gap: {pos_gap_1_2:.6f}")
    
    # Stage 2 -> Stage 3 boundary
    stage2_end_pos = pred_traj2['pred_pos'][-1]
    stage3_start_pos = traj3_input_stitched['curr_pos'][0]
    pos_gap_2_3 = torch.norm(stage2_end_pos - stage3_start_pos).item()
    
    if pos_gap_2_3 > 1e-2:
        warnings.append(f"⚠️  Stage 2→3 position gap: {pos_gap_2_3:.6f}")
    
    # Check stress continuity (if available)
    if 'pred_stress' in pred_traj1 and 'pred_stress' in pred_traj2:
        stage1_end_stress = pred_traj1['pred_stress'][-1]
        stage2_start_stress = pred_traj2['pred_stress'][0]
        stress_gap = torch.norm(stage1_end_stress - stage2_start_stress).item()
        
        if stress_gap > 0.1:
            warnings.append(f"⚠️  Stage 1→2 stress jump: {stress_gap:.6f}")
    
    return warnings

def obtain_infer_traj(inference_config, checkpoint_dir, val_file_path, continuous, split_type='val'):
    """
    Runs inference by loading data from a single .h5/.pt file, splitting it into 3 stages.
    Accepts 'split_type' to pass correctly to the Dataset class.
    
    Args:
        inference_config: Configuration dict with model parameters
        checkpoint_dir: Path to directory containing checkpoint_stress_stage_1/2/3
        val_file_path: Path to input .h5 or .pt file
        continuous: Boolean - if True, stitch outputs between stages
        split_type: 'val' or 'test' - which split to use
    
    Returns:
        trajectories1, trajectories2, trajectories3: Lists of prediction dicts
        info: Dictionary with metadata (index, inference_time, etc.)
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
    
    # Validate alignment before proceeding
    validate_stage_alignment(loader1, loader2, loader3, num_samples=min(3, len(loader1)))

    trajectories1, trajectories2, trajectories3 = [], [], []
    info = {
        'index': [], 
        'inference_time': [],
        'stage_1_time': [],
        'stage_2_time': [],
        'stage_3_time': [],
        'continuity_warnings': []
    }
    
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
        print(f"Processing {split_type} Trajectory {count}...", end=" ")

        # --- RUN STAGE 1 (PRESS) ---
        stage1_start = time.time()
        _, pred_traj1 = params['evaluator'].evaluate(model1, traj1_input)
        stage1_time = time.time() - stage1_start
        info['stage_1_time'].append(stage1_time)

        # --- RUN STAGE 2 (DWELL) ---
        stage2_start = time.time()
        if continuous:
            # Stitching: Overwrite start position with Stage 1's end position
            traj2_input_stitched = copy.deepcopy(traj2_input)
            traj2_input_stitched['curr_pos'][0] = pred_traj1['pred_pos'][-1]
            _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input_stitched)
        else:
            _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input)
            traj2_input_stitched = traj2_input
        stage2_time = time.time() - stage2_start
        info['stage_2_time'].append(stage2_time)

        # --- RUN STAGE 3 (RELEASE) ---
        stage3_start = time.time()
        if continuous:
            # Stitching: Overwrite start position with Stage 2's end position
            traj3_input_stitched = copy.deepcopy(traj3_input)
            traj3_input_stitched['curr_pos'][0] = pred_traj2['pred_pos'][-1]
            _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input_stitched)
        else:
            _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input)
            traj3_input_stitched = traj3_input
        stage3_time = time.time() - stage3_start
        info['stage_3_time'].append(stage3_time)

        # Check continuity
        warnings = validate_stage_continuity(pred_traj1, pred_traj2, pred_traj3, 
                                            traj2_input_stitched, traj3_input_stitched)
        if warnings:
            for w in warnings:
                print(f"\n  {w}")
            info['continuity_warnings'].extend(warnings)
        
        # Store results
        trajectories1.append(pred_traj1)
        trajectories2.append(pred_traj2)
        trajectories3.append(pred_traj3)

        total_time = time.time() - start_time
        info['inference_time'].append(total_time)
        print(f"✓ ({total_time:.2f}s)")
        count += 1

    return trajectories1, trajectories2, trajectories3, info

def save_rollout(rollout_path, trajectories, disp=""):
    """Save predicted trajectories to pickle file."""
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
        print(f"  Saved {disp} rollout to: {rollout_path}")

def concat_trajectories(trajectories1, trajectories2, trajectories3):
    """
    Merges the list of trajectory dictionaries from 3 stages into one.
    
    Validates:
    - All 3 stages have equal number of samples
    - All required keys present
    - Tensor shapes are compatible for concatenation
    
    Returns:
        List of concatenated trajectory dictionaries
    """
    num_samples_1 = len(trajectories1)
    num_samples_2 = len(trajectories2)
    num_samples_3 = len(trajectories3)
    
    if not (num_samples_1 == num_samples_2 == num_samples_3):
        raise ValueError(
            f"Stage sample count mismatch:\n"
            f"  Stage 1: {num_samples_1} samples\n"
            f"  Stage 2: {num_samples_2} samples\n"
            f"  Stage 3: {num_samples_3} samples"
        )
    
    concatenated_results = []
    
    for i in range(num_samples_1):
        sample_dict = {}
        keys = trajectories1[i].keys()
        
        for key in keys:
            t1 = trajectories1[i][key]
            t2 = trajectories2[i][key]
            t3 = trajectories3[i][key]
            
            # Skip concatenation for non-tensor types
            if not isinstance(t1, torch.Tensor):
                sample_dict[key] = t1
                continue
            
            try:
                # Concatenate along time dimension (dim 0)
                sample_dict[key] = torch.cat([t1, t2, t3], dim=0)
            except RuntimeError as e:
                print(f"⚠️  Warning: Could not concatenate key '{key}' for sample {i}")
                print(f"     Shapes: Stage1={t1.shape}, Stage2={t2.shape}, Stage3={t3.shape}")
                print(f"     Error: {e}")
                sample_dict[key] = t1  # Fallback to first stage
        
        concatenated_results.append(sample_dict)
    
    return concatenated_results

def inference(checkpoint_dir, val_file_path, rollout_dir, continuous=True, evaluate=False, 
              animation=False, animation_html=False, split_type='val'):
    """
    Complete inference pipeline for 3-stage press simulation.
    
    Args:
        checkpoint_dir: Path containing checkpoint_stress_stage_1/2/3 folders
        val_file_path: Path to .h5 input file
        rollout_dir: Output directory for results
        continuous: Whether to stitch stage outputs
        evaluate: Whether to run evaluation metrics
        animation: Whether to generate MP4 animations
        animation_html: Whether to generate HTML visualizations
        split_type: 'val' or 'test'
    """
    
    # CONFIGURATION
    inference_config = {
        'model': 'press', 
        'mode': 'all', 
        'core_model': 'transolver', 
        'aggregator': 'sum',
        'steps': 16, 
        'attention': False
    }

    print("\n" + "="*70)
    print(f"STARTING 3-STAGE INFERENCE PIPELINE")
    print(f"Input: {val_file_path}")
    print(f"Split: {split_type.upper()}")
    print(f"Continuous Mode: {'ON (stages stitched)' if continuous else 'OFF (independent)'}")
    print("="*70)

    # 1. Run the Multi-Stage Inference
    # Pass split_type down so Dataset knows what to load
    trajs1, trajs2, trajs3, info = obtain_infer_traj(
        inference_config, checkpoint_dir, val_file_path, continuous, split_type
    )
    
    # 2. Stitch the stages together
    print(f"\nConcatenating {len(trajs1)} trajectories from 3 stages...")
    combined_trajs = concat_trajectories(trajs1, trajs2, trajs3)
    print(f"✓ Concatenation complete. Output shape: {combined_trajs[0]['pred_pos'].shape}")

    # 3. Save Results
    os.makedirs(rollout_dir, exist_ok=True)
    
    print(f"\nSaving results to: {rollout_dir}")
    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        # Convert lists to serializable format
        info_serializable = {
            'index': info['index'],
            'inference_time': [float(x) for x in info['inference_time']],
            'stage_1_time': [float(x) for x in info['stage_1_time']],
            'stage_2_time': [float(x) for x in info['stage_2_time']],
            'stage_3_time': [float(x) for x in info['stage_3_time']],
            'continuity_warnings': info['continuity_warnings']
        }
        json.dump(info_serializable, f, indent=4)
    
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_1.pkl'), trajs1, disp="Stage 1 (Press)")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_2.pkl'), trajs2, disp="Stage 2 (Dwell)")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_3.pkl'), trajs3, disp="Stage 3 (Release)")
    
    output_file = os.path.join(rollout_dir, 'concatenated_rollout_all.pkl')
    save_rollout(output_file, combined_trajs, disp="Concatenated All Stages")

    # Print timing statistics
    print(f"\nTiming Statistics:")
    print(f"  Stage 1 (Press):   {sum(info['stage_1_time']):.2f}s avg: {sum(info['stage_1_time'])/len(info['stage_1_time']):.4f}s")
    print(f"  Stage 2 (Dwell):   {sum(info['stage_2_time']):.2f}s avg: {sum(info['stage_2_time'])/len(info['stage_2_time']):.4f}s")
    print(f"  Stage 3 (Release): {sum(info['stage_3_time']):.2f}s avg: {sum(info['stage_3_time'])/len(info['stage_3_time']):.4f}s")
    print(f"  Total Inference:   {sum(info['inference_time']):.2f}s avg: {sum(info['inference_time'])/len(info['inference_time']):.4f}s")

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

    print(f"\n" + "="*70)
    print(f"✓ INFERENCE COMPLETE for {split_type.upper()}")
    print("="*70 + "\n")
    return

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run 3-stage inference on test/val/both datasets')
    parser.add_argument('--split', type=str, default='both', choices=['test', 'val', 'both'],
                        help='Which dataset split to run inference on (default: both)')
    args = parser.parse_args()

    # --- USER SETTINGS ---
    
    # 1. Checkpoint Directory - Should contain checkpoint_stress_stage_1/2/3
    checkpoint_dir = '/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/final_checkpoints/transolver'
    
    # 2. Base Output Directory
    base_rollout_dir = '/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/final_checkpoints/transolver/h5_inference_2'

    # 3. Define Tasks: (Split Name, File Path)
    # Ensure these paths are correct on your server
    all_tasks = [
        {
            'split': 'val', 
            'file': '/home/sushil/PressNet/datasets/data/coarse/coarse_1500_train_val.h5'
        },
        {
            'split': 'test', 
            'file': '/home/sushil/PressNet/datasets/data/coarse/coarse_1500_test.h5'
        }
    ]
    
    # Filter tasks based on --split argument
    if args.split == 'test':
        tasks = [t for t in all_tasks if t['split'] == 'test']
    elif args.split == 'val':
        tasks = [t for t in all_tasks if t['split'] == 'val']
    else:  # 'both'
        tasks = all_tasks
    
    print(f"\n{'='*70}")
    print(f"Running inference on: {args.split.upper()}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"{'='*70}\n")

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
        
        
        print(f"\n{'='*70}")
        print(f" STARTING TASK: {split_name.upper()}")
        print(f" Input File: {file_path}")
        print(f" Output Dir: {current_rollout_dir}")
        print(f"{'='*70}")
        

        inference(
            checkpoint_dir=checkpoint_dir, 
            val_file_path=file_path, 
            rollout_dir=current_rollout_dir, 
            continuous=True,      # True = Feed output of Stage 1 to Stage 2
            evaluate=True,        # Calculate RMSE/Loss
            animation=False,      # Generate MP4/GIFs
            animation_html=False, # Generate Interactive HTML
            split_type=split_name # Pass 'val' or 'test'
        )

if __name__ == '__main__':
    main()
