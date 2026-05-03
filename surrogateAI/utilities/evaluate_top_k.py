import torch
import pickle
import os
import json
import numpy as np

def load_pkl_file(file_path):
    
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # If standard pickle fails (often due to PyTorch's specific serialization format)
        print(f"Standard pickle load failed for {file_path} (Error: {e}). Trying torch.load...")
        return torch.load(file_path, map_location='cpu', weights_only=False)


def compute_top_k_metrics(gt_vals, pred_vals, top_percentile=0.90):
    """
    Computes RMSE and Mean Absolute Error (MAE) only on the top (1 - top_percentile) 
    fraction of the ground truth values.
    """
    if gt_vals.numel() == 0:
        return None, None
        
    # Find the threshold value for the top K%
    threshold = torch.quantile(gt_vals, top_percentile)
    
    # Create a mask for values above the threshold
    top_mask = gt_vals >= threshold
    
    # Filter ground truth and predictions
    top_gt = gt_vals[top_mask]
    top_pred = pred_vals[top_mask]
    
    # Calculate errors
    error = torch.abs(top_pred - top_gt)
    rmse = torch.sqrt(torch.mean(error ** 2)).item()
    mae = torch.mean(error).item()
    
    return rmse, mae

def evaluate_top_k_rollout(rollout_pth, output_dir, top_percentile=0.90):
    # Fix the rounding issue
    top_k_pct_int = int(round((1 - top_percentile) * 100))
    metric_key = f"top_{top_k_pct_int}pct_rmse"
    
    print(f"Evaluating Top {top_k_pct_int}% Metrics for: {rollout_pth}")
    
    # Check if file exists before trying to load
    if not os.path.exists(rollout_pth):
        print(f"  [ERROR] File not found: {rollout_pth}")
        return None
        
    rollout = load_pkl_file(rollout_pth)
    os.makedirs(output_dir, exist_ok=True)
    
    # Restructure JSON without the model name
    final_output = {
        "evaluation_percentile": top_k_pct_int,
        "trajectories": []
    }

    for i, single_trajectory in enumerate(rollout):
        # 1. Extract node type and create a 1D material mask (Shape: [Time * Nodes])
        node_type = single_trajectory['node_type'].to('cpu').flatten()
        material_mask = (node_type == 0)
        
        traj_result = {
            "trajectory_index": i,
            "metrics": []
        }
        
        # --- STRESS ---
        if 'gt_stress' in single_trajectory and 'pred_stress' in single_trajectory:
            gt_stress = single_trajectory['gt_stress'].to('cpu').flatten()[material_mask]
            pred_stress = single_trajectory['pred_stress'].to('cpu').flatten()[material_mask]
            
            stress_rmse, stress_mae = compute_top_k_metrics(gt_stress, pred_stress, top_percentile)
            
            if stress_rmse is not None:
                traj_result["metrics"].append({
                    "quantity": "stress",
                    metric_key: stress_rmse  # Uses specific metric name instead of "value"
                })
                
        # --- Y-DEFORMATION ---
        # Pos has 3 values (X, Y, Z). We extract Y (index 1) FIRST, then flatten and mask.
        if 'gt_pos' in single_trajectory and 'pred_pos' in single_trajectory:
            gt_pos = single_trajectory['gt_pos'].to('cpu')
            pred_pos = single_trajectory['pred_pos'].to('cpu')
            mesh_pos = single_trajectory['mesh_pos'].to('cpu')
            
            # Extract absolute Y displacement (Shape: [Time, Nodes])
            gt_y_full = torch.abs(gt_pos[..., 1] - mesh_pos[..., 1])
            pred_y_full = torch.abs(pred_pos[..., 1] - mesh_pos[..., 1])
            
            # Flatten to 1D and apply the material mask
            gt_y_deform = gt_y_full.flatten()[material_mask]
            pred_y_deform = pred_y_full.flatten()[material_mask]
            
            deform_rmse, deform_mae = compute_top_k_metrics(gt_y_deform, pred_y_deform, top_percentile)
            
            if deform_rmse is not None:
                traj_result["metrics"].append({
                    "quantity": "y_displacement",
                    metric_key: deform_rmse  # Uses specific metric name instead of "value"
                })
        
        # Only append if we successfully collected metrics for this trajectory
        if traj_result["metrics"]:
            final_output["trajectories"].append(traj_result)

    # Save to JSON
    json_filename = f"top_{top_k_pct_int}pct_evaluation_summary.json"
    output_json_path = os.path.join(output_dir, json_filename)
    
    with open(output_json_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"  --> Saved Top {top_k_pct_int}% summary to {output_json_path}\n")
    return output_json_path


if __name__ == "__main__":
    # ---------------------------------------------------------
    # HARDCODED CONFIGURATION FOR MULTIPLE FILES
    # ---------------------------------------------------------
    top_percentile = 0.90  # 0.90 = Top 10%, 0.95 = Top 5%
    
    # Simple list of paths without model names
    rollout_paths = [
        
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Unseen_test/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Validationset/val/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_Extrapolation_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_OOD_Unseen/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Unseen_test_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/Inference_Output/val_real/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/Combined_Inference_K20_D100/val_real/concatenated_rollout_all.pkl",
    ]
    
    # ---------------------------------------------------------
    # RUN EVALUATION IN A LOOP
    # ---------------------------------------------------------
    top_k_pct_int = int(round((1 - top_percentile) * 100))
    print("=" * 70)
    print(f"STARTING TOP {top_k_pct_int}% EVALUATION PIPELINE")
    print("=" * 70)

    for rollout_path in rollout_paths:
        
        # Skip if path is a placeholder
        if not os.path.exists(rollout_path):
            print(f"Skipping: File not found ({rollout_path})")
            continue
            
        # Calculate the parent directory of the folder containing the pkl file
        # Output will be stored securely one step back in `evaluation_top_Xpct_metrics`
        parent_dir = os.path.dirname(os.path.dirname(rollout_path))
        output_dir = os.path.join(parent_dir, f"evaluation_top_{top_k_pct_int}pct_metrics")
        
        # Execute the evaluation for this path
        evaluate_top_k_rollout(rollout_path, output_dir, top_percentile)
    
    print("=" * 70)
    print("All evaluations complete!")
    print("=" * 70)