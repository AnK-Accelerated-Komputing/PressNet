"""Stage-wise (PressNet++) inference.

Three independently trained structural networks (press / dwell / release) are
assembled at inference time. Each advances its own time window autoregressively
and hands its final mesh state to the next stage. Outputs one pickle per stage
plus the concatenated 0-15 s trajectory (``concatenated_rollout_all.pkl``).

Example:
    python inference.py --config ../configs/inference/transolver_coarse.json \
        --data ../datasets/data/coarse/coarse_1500_train_val.h5 --split val --out ../outputs/infer_val
"""
import os
import torch
import pickle
import time
import json
import copy
import argparse
import gc  # Added for garbage collection (System RAM management)
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
        slice=64,
        message_passing_steps=20,
        stochastic_message_passing_used='False',
        k_neighbor=20,       
        dilated_k_sample=200,
        edge_dim=8  
    )
}

def _resolve_core_models(inference_config):
    """
    Resolve `core_model` from inference_config into a list of 3 core_model_names,
    one per stage. Accepts:
      - str: same architecture for all 3 stages (legacy behavior)
      - list/tuple of 3: per-stage names, e.g. ['transolver', 'regDGCNN_seg', 'dilated_dgcnn']
      - dict: {'stage_1': '...', 'stage_2': '...', 'stage_3': '...'}
    """
    core_cfg = inference_config.get('core_model', 'encode_process_decode')

    if isinstance(core_cfg, str):
        return [core_cfg, core_cfg, core_cfg]

    if isinstance(core_cfg, (list, tuple)):
        if len(core_cfg) != 3:
            raise ValueError(f"core_model list must have exactly 3 entries, got {len(core_cfg)}")
        return list(core_cfg)

    if isinstance(core_cfg, dict):
        try:
            return [core_cfg['stage_1'], core_cfg['stage_2'], core_cfg['stage_3']]
        except KeyError as e:
            raise ValueError(f"core_model dict missing key {e}; expected stage_1/stage_2/stage_3")

    raise TypeError(f"Unsupported core_model type: {type(core_cfg).__name__}")


def load_model(inference_config, params, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3):
    """
    Loads three separate models for Stage 1, Stage 2, and Stage 3 using individual paths.
    Each stage may use a different architecture (see _resolve_core_models).
    """
    # 1. Initialize the model architecture (potentially different per stage)
    core_names = _resolve_core_models(inference_config)
    core_name_1, core_name_2, core_name_3 = core_names
    print(f"[INIT] Stage 1 architecture: {core_name_1}")
    print(f"[INIT] Stage 2 architecture: {core_name_2}")
    print(f"[INIT] Stage 3 architecture: {core_name_3}")

    model1 = params['model'].Model(params, core_model_name=core_name_1)
    model2 = params['model'].Model(params, core_model_name=core_name_2)
    model3 = params['model'].Model(params, core_model_name=core_name_3)

    # 2. Helper function to check files before loading based on prefix priority
    def safe_load(model, target_path, stage_name):
        """
        Safely load model from a single provided directory path.
        Priority 1: best_model_checkpoint
        Priority 2: epoch_model_checkpoint
        """
        possible_prefixes = []
        
        if os.path.isdir(target_path):
            # Priority 1: Check for 'best' checkpoint directly in the provided folder
            possible_prefixes.append(os.path.join(target_path, 'best_model_checkpoint'))
            
            # Priority 2: Check for 'epoch' checkpoint directly in the provided folder
            possible_prefixes.append(os.path.join(target_path, 'epoch_model_checkpoint'))
            
            # Legacy fallback: generic 'model_checkpoint' just in case
            possible_prefixes.append(os.path.join(target_path, 'model_checkpoint'))
        else:
            # Fallback if you accidentally passed the direct .pth file path instead of the folder
            possible_prefixes.append(target_path.replace('_learned_model.pth', ''))

        # Iterate through the prefixes and load the first one that actually exists
        for prefix in possible_prefixes:
            test_file = prefix + "_learned_model.pth"
            if os.path.exists(test_file):
                print(f"[{stage_name}] Found checkpoint at {test_file}. Loading...")
                try:
                    model.load_model(prefix)
                    print(f"[{stage_name}] ✓ Success: Loaded model.")
                    return True
                except Exception as e:
                    print(f"[{stage_name}] ✗ Warning: Error loading from {prefix} ({e}). Trying next fallback...")

        print(f"[{stage_name}] ✗ CRITICAL ERROR: No valid checkpoint found in {target_path}")
        return False

    # 3. Load weights
    print("\n" + "="*60)
    print("--- Loading Models ---")
    print("="*60)
    success1 = safe_load(model1, checkpoint_path_1, "Stage 1 (Press)")
    success2 = safe_load(model2, checkpoint_path_2, "Stage 2 (Dwell)")
    success3 = safe_load(model3, checkpoint_path_3, "Stage 3 (Release)")
    
    if not (success1 and success2 and success3):
        raise RuntimeError(f"Failed to load all 3 models. Check checkpoint paths:\n"
                           f"  Stage 1: {checkpoint_path_1}\n"
                           f"  Stage 2: {checkpoint_path_2}\n"
                           f"  Stage 3: {checkpoint_path_3}")
    
    print("="*60 + "\n")

    # 4. Apply "Monkey Patch" per-stage, ONLY for stages whose model is transolver
    def apply_patch(wrapper_model):
        original_forward = wrapper_model.learned_model.forward

        def patched_forward(*args, **kwargs):
            kwargs.pop('is_training', None)
            return original_forward(*args, **kwargs)

        wrapper_model.learned_model.forward = patched_forward

    for stage_model, stage_core in zip([model1, model2, model3], core_names):
        if stage_core == 'transolver':
            apply_patch(stage_model)

    # 5. Move to GPU
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

def obtain_infer_traj(inference_config, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3, val_file_path, continuous, split_type='val', teacher_forcing=False):
    """
    Runs inference by loading data from a single .h5/.pt file, splitting it into 3 stages.
    Accepts 'split_type' to pass correctly to the Dataset class.
    """
    params = PARAMETERS[inference_config['model']]
    model1, model2, model3 = load_model(inference_config, params, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3)

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
    info = {
        'index': [], 
        'inference_time': [],
        'stage_1_time': [],
        'stage_2_time': [],
        'stage_3_time': [],
        'stage_1_steps': [],
        'stage_2_steps': [],
        'stage_3_steps': [],
        'continuity_warnings': []
    }
    
    print(f"Starting inference on {len(loader1)} trajectories...")
    count = 0
    
    # --- HELPER FUNCTION FOR GPU MEMORY MANAGEMENT ---
    def move_to_cpu(trajectory_dict):
        """Moves all tensors in a dictionary to CPU and detaches them to free GPU VRAM."""
        cpu_dict = {}
        for k, v in trajectory_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.detach().cpu()
            else:
                cpu_dict[k] = v
        return cpu_dict

    # --- NO_GRAD CONTEXT TO PREVENT VRAM BLOAT ---
    with torch.no_grad():
        # Iterate through all 3 stages simultaneously using zip
        for batch_stage1, batch_stage2, batch_stage3 in zip(loader1, loader2, loader3):
            
            # 1. Unpack Batch (Remove [1, ...] dimension)
            traj1_input = squeeze_data(batch_stage1)
            traj2_input = squeeze_data(batch_stage2)
            traj3_input = squeeze_data(batch_stage3)
            
            info['index'].append(count)
            print(f"Processing {split_type} Trajectory {count}...", end=" ")

            # Synchronize GPU before overall timer begins
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()

            # --- RUN STAGE 1 (PRESS) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            stage1_start = time.perf_counter()
            _, pred_traj1 = params['evaluator'].evaluate(model1, traj1_input, teacher_forcing=teacher_forcing)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            stage1_time = time.perf_counter() - stage1_start
            info['stage_1_time'].append(stage1_time)
            info['stage_1_steps'].append(int(pred_traj1['pred_pos'].shape[0]))

            # --- RUN STAGE 2 (DWELL) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            stage2_start = time.perf_counter()
            if continuous:
                # Stitching: Clone tensor data directly instead of deepcopy for performance
                traj2_input_stitched = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in traj2_input.items()}
                traj2_input_stitched['curr_pos'][0] = pred_traj1['pred_pos'][-1]
                _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input_stitched, teacher_forcing=teacher_forcing)

                # restored the gt_pos[0] to original 
                pred_traj2['gt_pos'][0] = traj2_input['curr_pos'][0]

            else:
                _, pred_traj2 = params['evaluator'].evaluate(model2, traj2_input, teacher_forcing=teacher_forcing)
                traj2_input_stitched = traj2_input
            
            if device.type == 'cuda': torch.cuda.synchronize()
            stage2_time = time.perf_counter() - stage2_start
            info['stage_2_time'].append(stage2_time)
            info['stage_2_steps'].append(int(pred_traj2['pred_pos'].shape[0]))

            # --- RUN STAGE 3 (RELEASE) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            stage3_start = time.perf_counter()
            if continuous:
                # Stitching: Clone tensor data directly instead of deepcopy for performance
                traj3_input_stitched = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in traj3_input.items()}
                traj3_input_stitched['curr_pos'][0] = pred_traj2['pred_pos'][-1]
                _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input_stitched, teacher_forcing=teacher_forcing)
                
                # restored the gt_pos[0] to original
                pred_traj3['gt_pos'][0] = traj3_input['curr_pos'][0]
            else:
                _, pred_traj3 = params['evaluator'].evaluate(model3, traj3_input, teacher_forcing=teacher_forcing)
                traj3_input_stitched = traj3_input
            
            if device.type == 'cuda': torch.cuda.synchronize()
            stage3_time = time.perf_counter() - stage3_start
            info['stage_3_time'].append(stage3_time)
            info['stage_3_steps'].append(int(pred_traj3['pred_pos'].shape[0]))

            # Final synchronization for total time
            if device.type == 'cuda': torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time

            # Check continuity
            warnings = validate_stage_continuity(pred_traj1, pred_traj2, pred_traj3, 
                                                traj2_input_stitched, traj3_input_stitched)
            if warnings:
                for w in warnings:
                    print(f"\n  {w}")
                info['continuity_warnings'].extend(warnings)
            
            # --- STORE RESULTS ON CPU TO PREVENT OOM ---
            trajectories1.append(move_to_cpu(pred_traj1))
            trajectories2.append(move_to_cpu(pred_traj2))
            trajectories3.append(move_to_cpu(pred_traj3))

            info['inference_time'].append(total_time)
            print(f"✓ ({total_time*1000:.2f} ms)")
            count += 1

    return trajectories1, trajectories2, trajectories3, info

def save_rollout(rollout_path, trajectories, disp=""):
    """Save predicted trajectories using torch.save for memory efficiency."""
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
    #torch.save(trajectories, rollout_path)
    print(f"  Saved {disp} rollout to: {rollout_path}")

def concat_trajectories(trajectories1, trajectories2, trajectories3):
    """
    Merges the list of trajectory dictionaries from 3 stages into one.
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

def inference(checkpoint_path_1, checkpoint_path_2, checkpoint_path_3, val_file_path, rollout_dir, continuous=True,
              evaluate=False, animation=False, split_type='val', core_model='encode_process_decode',
              teacher_forcing=False):
    """
    Complete inference pipeline for 3-stage press simulation.

    `core_model` can be:
      - a string (same architecture for all 3 stages), or
      - a list/tuple of 3 names (per-stage architectures), or
      - a dict {'stage_1': ..., 'stage_2': ..., 'stage_3': ...}
    Valid names: encode_process_decode, gcn, regDGCNN_seg, regpointnet_seg,
                 transolver, dilated_dgcnn.

    `teacher_forcing=True` switches from the rolled-out protocol to the one-step protocol.
    """

    inference_config = {
        'model': 'press',
        'mode': 'all',
        'core_model': core_model,
        'aggregator': 'sum',
        'steps': 16,
        'attention': False
    }

    print("\n" + "=" * 70)
    print("STARTING 3-STAGE INFERENCE PIPELINE")
    print(f"Input: {val_file_path}")
    print(f"Split: {split_type.upper()}")
    print(f"Protocol: {'one-step (ground truth fed every step)' if teacher_forcing else 'rolled-out (autoregressive)'}")
    print(f"Continuous Mode: {'ON (stages stitched)' if continuous else 'OFF (independent)'}")
    print("=" * 70)

    # 1. Run the multi-stage inference
    trajs1, trajs2, trajs3, info = obtain_infer_traj(
        inference_config, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3,
        val_file_path, continuous, split_type, teacher_forcing
    )

    # 2. Save individual results first
    os.makedirs(rollout_dir, exist_ok=True)

    print(f"\nSaving individual stages to: {rollout_dir}")
    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        info_serializable = {
            'index': info['index'],
            'inference_time': [float(x) for x in info['inference_time']],
            'stage_1_time': [float(x) for x in info['stage_1_time']],
            'stage_2_time': [float(x) for x in info['stage_2_time']],
            'stage_3_time': [float(x) for x in info['stage_3_time']],
            'stage_1_steps': info['stage_1_steps'],
            'stage_2_steps': info['stage_2_steps'],
            'stage_3_steps': info['stage_3_steps'],
            'continuity_warnings': info['continuity_warnings']
        }
        json.dump(info_serializable, f, indent=4)

    save_rollout(os.path.join(rollout_dir, 'rollout_stage_1.pkl'), trajs1, disp="Stage 1 (Press)")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_2.pkl'), trajs2, disp="Stage 2 (Dwell)")
    save_rollout(os.path.join(rollout_dir, 'rollout_stage_3.pkl'), trajs3, disp="Stage 3 (Release)")

    # 3. Stitch the stages together
    print(f"\nConcatenating {len(trajs1)} trajectories from 3 stages...")
    combined_trajs = concat_trajectories(trajs1, trajs2, trajs3)
    print(f"Concatenation complete. Output shape: {combined_trajs[0]['pred_pos'].shape}")

    # Free host RAM before pickling the combined trajectory
    del trajs1
    del trajs2
    del trajs3
    gc.collect()

    # 4. Save the combined trajectory
    output_file = os.path.join(rollout_dir, 'concatenated_rollout_all.pkl')
    save_rollout(output_file, combined_trajs, disp="Concatenated All Stages")

    # Timing statistics
    print("\nTiming Statistics:")
    for name, key in [("Stage 1 (Press)", 'stage_1_time'), ("Stage 2 (Dwell)", 'stage_2_time'),
                      ("Stage 3 (Release)", 'stage_3_time'), ("Total Inference", 'inference_time')]:
        total = sum(info[key])
        print(f"  {name:<18} {total:.2f}s avg: {total / len(info[key]):.4f}s")

    # 5. Evaluation (nMAE / nRMSE as defined in the paper)
    if evaluate:
        from utilities.evaluate_metrics import evaluate_and_generate_report
        evaluate_dir = os.path.join(rollout_dir, 'evaluation')
        print(f"\nRunning evaluation for {split_type}...")
        evaluate_and_generate_report(output_file, evaluate_dir)

    # 6. Animation
    if animation:
        from utilities.animate_compare import animate_rollout
        vis_root = os.path.join(rollout_dir, 'visualization')
        os.makedirs(vis_root, exist_ok=True)
        print(f"\nGenerating animations for {split_type}...")
        for i in range(len(combined_trajs)):
            traj_vis_path = os.path.join(vis_root, f"{i}")
            os.makedirs(traj_vis_path, exist_ok=True)
            animate_rollout(output_file, traj_vis_path, i=i, key='stress')

    print("\n" + "=" * 70)
    print(f"INFERENCE COMPLETE for {split_type.upper()}")
    print("=" * 70 + "\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description='Stage-wise (press -> dwell -> release) PressNet++ inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with "core_model" (str, 3-list or stage dict) and "checkpoints" '
                             '({"stage_1": dir, "stage_2": dir, "stage_3": dir}). CLI flags override it.')
    parser.add_argument('--data', type=str, required=True,
                        help='HDF5 file (its *_meta.json must sit next to it) holding the trajectories to roll out')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help="'val': last 30%% of the groups in --data (in-distribution validation set). "
                             "'test': every group in --data (use for the extrapolation / unseen-shape files)")
    parser.add_argument('--out', type=str, required=True, help='Output directory for rollouts and metrics')
    parser.add_argument('--core_model', type=str, nargs='+', default=None,
                        help='One architecture for all stages, or three (press dwell release). '
                             'Choices: gcn encode_process_decode regDGCNN_seg transolver dilated_dgcnn')
    parser.add_argument('--stage1_ckpt', type=str, default=None, help='Checkpoint dir of the press network')
    parser.add_argument('--stage2_ckpt', type=str, default=None, help='Checkpoint dir of the dwell network')
    parser.add_argument('--stage3_ckpt', type=str, default=None, help='Checkpoint dir of the release network')
    parser.add_argument('--one_step', action='store_true',
                        help='One-step protocol: feed ground truth at every step (no error accumulation, no stage hand-off)')
    parser.add_argument('--no_continuous', action='store_true',
                        help='Do not hand the predicted end state of a stage to the next one (score stages independently)')
    parser.add_argument('--evaluate', action='store_true', help='Compute nMAE / nRMSE summaries after inference')
    parser.add_argument('--animation', action='store_true', help='Render GIF comparisons (needs plotly + kaleido)')
    return parser


def main():
    args = build_parser().parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    ckpts = cfg.get('checkpoints', {})
    ckpt_1 = args.stage1_ckpt or ckpts.get('stage_1')
    ckpt_2 = args.stage2_ckpt or ckpts.get('stage_2')
    ckpt_3 = args.stage3_ckpt or ckpts.get('stage_3')
    if not (ckpt_1 and ckpt_2 and ckpt_3):
        raise SystemExit('Need three stage checkpoints: pass --stage{1,2,3}_ckpt or a --config with "checkpoints".')

    core_model = cfg.get('core_model', 'encode_process_decode')
    if args.core_model:
        core_model = args.core_model[0] if len(args.core_model) == 1 else args.core_model

    if not os.path.exists(args.data):
        raise SystemExit(f'Data file not found: {args.data}')

    inference(
        checkpoint_path_1=ckpt_1,
        checkpoint_path_2=ckpt_2,
        checkpoint_path_3=ckpt_3,
        val_file_path=args.data,
        rollout_dir=args.out,
        continuous=not (args.no_continuous or args.one_step),
        evaluate=args.evaluate,
        animation=args.animation,
        split_type=args.split,
        core_model=core_model,
        teacher_forcing=args.one_step,
    )


if __name__ == '__main__':
    main()
