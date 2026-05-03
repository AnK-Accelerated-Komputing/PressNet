import torch
import pickle
import os
import json
import pandas as pd
import time
import numpy as np

# --------------------------------------------------
# IO
# --------------------------------------------------
def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------
# Helper: Robustly get top value and its 2D coordinates
# --------------------------------------------------
def get_top_value(tensor_2d, mask):
    """
    Finds the maximum value in a 2D tensor within the masked region.
    Args:
        tensor_2d: 2D tensor [Time, Nodes]
        mask: 2D boolean mask [Time, Nodes]
    """
    # 1. Safety check: Ensure inputs are 2D
    if tensor_2d.dim() > 2:
        tensor_2d = tensor_2d.view(tensor_2d.shape[0], -1)
    if mask.dim() > 2:
        mask = mask.view(mask.shape[0], -1)
        
    # 2. Handle Masking
    masked_tensor = tensor_2d.clone()
    masked_tensor[~mask] = -float('inf')
    
    # 3. Find Max Index
    flat_idx = torch.argmax(masked_tensor).item()
    
    # 4. Convert Flat Index to (Time, Node)
    coords = np.unravel_index(flat_idx, tensor_2d.shape)
    t, n = int(coords[0]), int(coords[1])
    
    # 5. Extract Value
    value = float(tensor_2d[t, n].item())
    
    return (t, n, value)

# --------------------------------------------------
# Trajectory comprehensive analysis
# --------------------------------------------------
def trajectory_max_error_summary(single_trajectory, trajectory_index):
    # Load raw tensors
    mesh_pos = single_trajectory["mesh_pos"].cpu()
    gt_pos = single_trajectory["gt_pos"].cpu()
    pred_pos = single_trajectory["pred_pos"].cpu()
    gt_stress = single_trajectory["gt_stress"].cpu()
    pred_stress = single_trajectory["pred_stress"].cpu()
    node_type = single_trajectory["node_type"].cpu()

    # -------------------------------------------------------
    # SHAPE NORMALIZATION & DIMENSION DETECTION
    # -------------------------------------------------------
    # Detect Number of Nodes (N) and Spatial Dimension (D)
    # mesh_pos is usually [1, N, D] or [T, N, D]
    if mesh_pos.dim() >= 2:
        num_nodes = mesh_pos.shape[-2]
        spatial_dim = mesh_pos.shape[-1] # Detects 2 or 3 dynamically
    else:
        # Fallback if mesh_pos is flattened (unlikely but safe)
        # We assume D=2 if we can't tell, or try to infer
        spatial_dim = 2 
        # Heuristic: try to guess D=3 if size is divisible by 3 but not 2
        if mesh_pos.numel() % 3 == 0 and mesh_pos.numel() % 2 != 0:
            spatial_dim = 3
        num_nodes = mesh_pos.numel() // spatial_dim

    # Helper to reshape scalar fields (Stress, Type) -> [Time, Nodes]
    def reshape_scalar(tensor, N):
        flat = tensor.reshape(-1)
        num_timesteps = flat.shape[0] // N
        return flat.view(num_timesteps, N)

    # Helper to reshape vector fields (Pos) -> [Time, Nodes, Dim]
    def reshape_pos(tensor, N, dim):
        # We reshape to (-1, dim) first.
        # This handles the case where input is [T*N*D] or [T, N, D]
        flat = tensor.reshape(-1, dim)
        num_timesteps = flat.shape[0] // N
        return flat.view(num_timesteps, N, dim)

    # Apply Reshaping
    try:
        gt_stress = reshape_scalar(gt_stress, num_nodes)
        pred_stress = reshape_scalar(pred_stress, num_nodes)
        node_type = reshape_scalar(node_type, num_nodes)
        
        gt_pos_reshaped = reshape_pos(gt_pos, num_nodes, spatial_dim)
        pred_pos_reshaped = reshape_pos(pred_pos, num_nodes, spatial_dim)
        mesh_pos_reshaped = reshape_pos(mesh_pos, num_nodes, spatial_dim)
    except RuntimeError as e:
        print(f"  [Error] Reshape failed for Traj {trajectory_index}. "
              f"Nodes={num_nodes}, Dim={spatial_dim}. Error: {e}")
        # Stop this trajectory but don't crash the script
        return []

    # Handle domain mask
    domain_mask = (node_type == 0)
    if domain_mask.shape[0] == 1 and gt_stress.shape[0] > 1:
        domain_mask = domain_mask.expand(gt_stress.shape[0], -1)

    # -------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------
    results = []

    # === 1. STRESS ANALYSIS ===
    stress_error = torch.abs(pred_stress - gt_stress)
    _, _, max_gt_stress = get_top_value(gt_stress, domain_mask)
    
    # A. Metric: Max Error
    t, n, error_val = get_top_value(stress_error, domain_mask)
    gt_val = float(gt_stress[t % gt_stress.shape[0], n].item())
    pred_val = float(pred_stress[t % pred_stress.shape[0], n].item())
    
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")
    rel_err_max = (error_val / abs(max_gt_stress) * 100) if max_gt_stress != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "stress",
        "metric_type": "max_error",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "relative_error_to_max_groundtruth(s)": rel_err_max,
        "time_index": t,
    })

    # B. Metric: Max Groundtruth
    t, n, gt_val = get_top_value(gt_stress, domain_mask)
    pred_val = float(pred_stress[t % pred_stress.shape[0], n].item())
    error_val = abs(pred_val - gt_val)
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "stress",
        "metric_type": "max_groundtruth",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "time_index": t,
    })

    # C. Metric: Max Prediction
    t, n, pred_val = get_top_value(pred_stress, domain_mask)
    gt_val = float(gt_stress[t % gt_stress.shape[0], n].item())
    error_val = abs(pred_val - gt_val)
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "stress",
        "metric_type": "max_prediction",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "time_index": t,
    })

    # === 2. Y-DISPLACEMENT ANALYSIS ===
    # Calculate Y displacement (index 1 is Y for both 2D and 3D)
    gt_y = torch.abs(gt_pos_reshaped[:, :, 1] - mesh_pos_reshaped[:, :, 1])
    pred_y = torch.abs(pred_pos_reshaped[:, :, 1] - mesh_pos_reshaped[:, :, 1])
    deform_error = torch.abs(pred_y - gt_y)

    _, _, max_gt_disp = get_top_value(gt_y, domain_mask)

    # A. Metric: Max Error
    t, n, error_val = get_top_value(deform_error, domain_mask)
    gt_val = float(gt_y[t % gt_y.shape[0], n].item())
    pred_val = float(pred_y[t % pred_y.shape[0], n].item())
    
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")
    rel_err_max = (error_val / abs(max_gt_disp) * 100) if max_gt_disp != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "y_displacement",
        "metric_type": "max_error",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "relative_error_to_max_groundtruth(y)": rel_err_max,
        "time_index": t,
    })

    # B. Metric: Max Groundtruth
    t, n, gt_val = get_top_value(gt_y, domain_mask)
    pred_val = float(pred_y[t % pred_y.shape[0], n].item())
    error_val = abs(pred_val - gt_val)
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "y_displacement",
        "metric_type": "max_groundtruth",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "time_index": t,
    })

    # C. Metric: Max Prediction
    t, n, pred_val = get_top_value(pred_y, domain_mask)
    gt_val = float(gt_y[t % gt_y.shape[0], n].item())
    error_val = abs(pred_val - gt_val)
    pct_err = (error_val / abs(gt_val) * 100) if gt_val != 0 else float("inf")

    results.append({
        "trajectory_index": trajectory_index,
        "quantity": "y_displacement",
        "metric_type": "max_prediction",
        "value": error_val,
        "groundtruth": gt_val,
        "prediction": pred_val,
        "percent_error": pct_err,
        "time_index": t,
    })

    return results


# --------------------------------------------------
# Rollout evaluation
# --------------------------------------------------
def evaluate_rollout(rollout_pth, output_dir):
    print(f"Evaluating rollout: {rollout_pth}")
    if not os.path.exists(rollout_pth):
        print(f"Error: File not found at {rollout_pth}")
        return

    rollout = load_pkl_file(rollout_pth)
    os.makedirs(output_dir, exist_ok=True)
    combined_rows = []

    for i, single_trajectory in enumerate(rollout):
        start = time.time()
        print(f"\n{'='*80}")
        print(f"Processing trajectory {i}")
        print(f"{'='*80}")

        try:
            summary = trajectory_max_error_summary(single_trajectory, i)
            if summary:
                combined_rows.extend(summary)

                # Display logical summary
                summary_df = pd.DataFrame(summary)
                for quantity in summary_df['quantity'].unique():
                    qty_data = summary_df[summary_df['quantity'] == quantity]
                    print(f"\n--- {quantity.upper()} ---")
                    
                    for m_type in ['max_error', 'max_groundtruth', 'max_prediction']:
                        row = qty_data[qty_data['metric_type'] == m_type].iloc[0]
                        
                        suffix = "(s)" if quantity == "stress" else "(y)"
                        rel_col = f"relative_error_to_max_groundtruth{suffix}"
                        rel_val = row.get(rel_col, None)
                        rel_str = f"{rel_val:.2f}%" if pd.notnull(rel_val) else "N/A"

                        print(f"{m_type.upper()}: Val={row['value']:.6f} | "
                              f"GT={row['groundtruth']:.6f} | "
                              f"Pred={row['prediction']:.6f} | "
                              f"Err={abs(row['groundtruth']-row['prediction']):.6f} | "
                              f"RelMaxGT={rel_str} | "
                              f"T={row['time_index']}")
            else:
                print("Skipped due to data error.")

        except Exception as e:
            print(f"Failed to process trajectory {i}: {e}")
            # Use this to debug hard crashes if needed
            # import traceback
            # traceback.print_exc()

        print(f"Time: {time.time() - start:.2f} sec")

    # Save Results
    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_csv = os.path.join(output_dir, "combined_detailed_summary.csv")
        combined_df.to_csv(combined_csv, index=False)
        
        combined_json = os.path.join(output_dir, "combined_detailed_summary.json")
        with open(combined_json, "w") as f:
            json.dump(combined_rows, f, indent=4)

        print("\n" + "="*80)
        print("✔ Rollout evaluation completed")
        print(f"✔ CSV: {combined_csv}")
        
        # Print Summary Max Stats
        print("\n" + "="*80)
        print("GLOBAL MAXIMUM RELATIVE ERRORS")
        print("="*80)
        
        for q, suffix in [("stress", "(s)"), ("y_displacement", "(y)")]:
            mask = (combined_df['quantity'] == q) & (combined_df['metric_type'] == 'max_error')
            df_sub = combined_df[mask]
            col = f"relative_error_to_max_groundtruth{suffix}"
            
            if not df_sub.empty and col in df_sub.columns:
                max_row = df_sub.loc[df_sub[col].idxmax()]
                print(f"\n{q.upper()}:")
                print(f"  Max Relative Error: {max_row[col]:.4f}%")
                print(f"  Trajectory Index: {int(max_row['trajectory_index'])}")
    else:
        print("\nNo results generated.")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    base_dir = [
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Combined_Inference_OneStepPrediction/test",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/Combined_Inference_K20_D100_onestepprediction/test",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/Inference_Output_OnestepPrediction/test",
    ]

    for base in base_dir:
        if not os.path.exists(base):
            print(f"Error: Base directory not found at {base}")
            continue

        rollout_pth = os.path.join(base, "concatenated_rollout_all.pkl")
        output_dir = os.path.join(base, "rollout_evaluation_max_error")

        evaluate_rollout(rollout_pth, output_dir)

if __name__ == "__main__":
    main()