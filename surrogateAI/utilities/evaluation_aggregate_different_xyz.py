import torch
import pickle
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import time

def load_pkl_file(file_path):
    """
    Attempts to load a file using standard pickle. 
    If it fails, falls back to torch.load.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # If standard pickle fails (often due to PyTorch's specific serialization format)
        print(f"Standard pickle load failed for {file_path} (Error: {e}). Trying torch.load...")
        return torch.load(file_path, map_location='cpu', weights_only=False)

def process_single_trajectory(single_trajectory):
    """
    Computes BOTH the Global Pooled metrics (for JSON) and Step-wise metrics (for Plots)
    for a single trajectory across X, Y, and Z dimensions.
    """
    def get_tensor(key):
        t = single_trajectory[key]
        if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.size(0) == 1:
            return t.squeeze(0).to('cpu')
        return t.to('cpu')

    node_type = get_tensor('node_type')
    gt_stress = get_tensor('gt_stress')
    pred_stress = get_tensor('pred_stress')
    gt_pos = get_tensor('gt_pos')
    pred_pos = get_tensor('pred_pos')
    mesh_pos = get_tensor('mesh_pos')

    num_steps = node_type.size(0)

    # --- Variables for Pooled JSON Metrics ---
    total_valid_nodes = 0
    stress_abs_err_sum, stress_sq_err_sum = 0.0, 0.0
    
    deform_x_abs_err_sum, deform_x_sq_err_sum = 0.0, 0.0
    deform_y_abs_err_sum, deform_y_sq_err_sum = 0.0, 0.0
    deform_z_abs_err_sum, deform_z_sq_err_sum = 0.0, 0.0
    
    global_stress_sum = 0.0
    global_deform_x_sum, global_deform_y_sum, global_deform_z_sum = 0.0, 0.0, 0.0
    
    global_stress_var_sum = 0.0
    global_deform_x_var_sum, global_deform_y_var_sum, global_deform_z_var_sum = 0.0, 0.0, 0.0
    
    global_stress_max, global_stress_min = -float('inf'), float('inf')
    global_deform_x_max, global_deform_x_min = -float('inf'), float('inf')
    global_deform_y_max, global_deform_y_min = -float('inf'), float('inf')
    global_deform_z_max, global_deform_z_min = -float('inf'), float('inf')

    # --- Variables for Step-by-Step Plots ---
    step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'rmse_deform_x': [], 'mae_deform_x': [], 'nrmse_deform_x': [], 'r2_deform_x': [],
        'rmse_deform_y': [], 'mae_deform_y': [], 'nrmse_deform_y': [], 'r2_deform_y': [],
        'rmse_deform_z': [], 'mae_deform_z': [], 'nrmse_deform_z': [], 'r2_deform_z': []
    }

    # 1st Pass: Find Global Ranges & Means for Normalization and Global R2
    for i in range(num_steps):
        if i == 1000 or i == 2000: continue # Skip restart frames if applicable
        mask = node_type[i].flatten() == 0
        if mask.sum() == 0: continue

        gt_s = gt_stress[i].flatten()[mask]
        gt_p = gt_pos[i][mask]
        m_p = mesh_pos[i][mask]
        
        gt_x_d = torch.abs((gt_p - m_p)[:, 0].flatten())
        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())
        gt_z_d = torch.abs((gt_p - m_p)[:, 2].flatten())

        global_stress_max = max(global_stress_max, torch.max(gt_s).item())
        global_stress_min = min(global_stress_min, torch.min(gt_s).item())
        
        global_deform_x_max = max(global_deform_x_max, torch.max(gt_x_d).item())
        global_deform_x_min = min(global_deform_x_min, torch.min(gt_x_d).item())
        global_deform_y_max = max(global_deform_y_max, torch.max(gt_y_d).item())
        global_deform_y_min = min(global_deform_y_min, torch.min(gt_y_d).item())
        global_deform_z_max = max(global_deform_z_max, torch.max(gt_z_d).item())
        global_deform_z_min = min(global_deform_z_min, torch.min(gt_z_d).item())

        global_stress_sum += torch.sum(gt_s).item()
        global_deform_x_sum += torch.sum(gt_x_d).item()
        global_deform_y_sum += torch.sum(gt_y_d).item()
        global_deform_z_sum += torch.sum(gt_z_d).item()
        total_valid_nodes += gt_s.numel()

    stress_range = max(global_stress_max - global_stress_min, 1e-6)
    deform_x_range = max(global_deform_x_max - global_deform_x_min, 1e-6)
    deform_y_range = max(global_deform_y_max - global_deform_y_min, 1e-6)
    deform_z_range = max(global_deform_z_max - global_deform_z_min, 1e-6)
    
    global_stress_mean = global_stress_sum / total_valid_nodes if total_valid_nodes > 0 else 0
    global_deform_x_mean = global_deform_x_sum / total_valid_nodes if total_valid_nodes > 0 else 0
    global_deform_y_mean = global_deform_y_sum / total_valid_nodes if total_valid_nodes > 0 else 0
    global_deform_z_mean = global_deform_z_sum / total_valid_nodes if total_valid_nodes > 0 else 0

    # 2nd Pass: Accumulate Errors
    for i in range(num_steps):
        if i == 1000 or i == 2000:
            for k in step_metrics.keys():
                if len(step_metrics[k]) > 0: step_metrics[k].append(step_metrics[k][-1])
            continue

        mask = node_type[i].flatten() == 0
        if mask.sum() == 0: continue

        gt_s = gt_stress[i].flatten()[mask]
        pred_s = pred_stress[i].flatten()[mask]
        gt_p = gt_pos[i][mask]
        pred_p = pred_pos[i][mask]
        m_p = mesh_pos[i][mask]

        gt_x_d = torch.abs((gt_p - m_p)[:, 0].flatten())
        pred_x_d = torch.abs((pred_p - m_p)[:, 0].flatten())
        
        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())
        pred_y_d = torch.abs((pred_p - m_p)[:, 1].flatten())
        
        gt_z_d = torch.abs((gt_p - m_p)[:, 2].flatten())
        pred_z_d = torch.abs((pred_p - m_p)[:, 2].flatten())

        # Accumulate for Pooled Metrics (Global Error)
        stress_abs_err_sum += torch.sum(torch.abs(gt_s - pred_s)).item()
        stress_sq_err_sum += torch.sum((gt_s - pred_s)**2).item()
        global_stress_var_sum += torch.sum((gt_s - global_stress_mean)**2).item()

        deform_x_abs_err_sum += torch.sum(torch.abs(gt_x_d - pred_x_d)).item()
        deform_x_sq_err_sum += torch.sum((gt_x_d - pred_x_d)**2).item()
        global_deform_x_var_sum += torch.sum((gt_x_d - global_deform_x_mean)**2).item()

        deform_y_abs_err_sum += torch.sum(torch.abs(gt_y_d - pred_y_d)).item()
        deform_y_sq_err_sum += torch.sum((gt_y_d - pred_y_d)**2).item()
        global_deform_y_var_sum += torch.sum((gt_y_d - global_deform_y_mean)**2).item()

        deform_z_abs_err_sum += torch.sum(torch.abs(gt_z_d - pred_z_d)).item()
        deform_z_sq_err_sum += torch.sum((gt_z_d - pred_z_d)**2).item()
        global_deform_z_var_sum += torch.sum((gt_z_d - global_deform_z_mean)**2).item()

        # --- Helper for Step-by-Step Metrics ---
        def calc_step_metrics(gt, pred, var_val):
            rmse = torch.sqrt(torch.mean((gt - pred) ** 2)).item()
            mae = torch.mean(torch.abs(gt - pred)).item()
            var = torch.sum((gt - torch.mean(gt))**2).item()
            r2 = 1.0 - (torch.sum((gt - pred)**2).item() / var) if var > 1e-6 else 0.0
            return rmse, mae, r2

        s_rmse, s_mae, s_r2 = calc_step_metrics(gt_s, pred_s, global_stress_var_sum)
        dx_rmse, dx_mae, dx_r2 = calc_step_metrics(gt_x_d, pred_x_d, global_deform_x_var_sum)
        dy_rmse, dy_mae, dy_r2 = calc_step_metrics(gt_y_d, pred_y_d, global_deform_y_var_sum)
        dz_rmse, dz_mae, dz_r2 = calc_step_metrics(gt_z_d, pred_z_d, global_deform_z_var_sum)

        # Append Stress
        step_metrics['rmse_stress'].append(s_rmse)
        step_metrics['mae_stress'].append(s_mae)
        step_metrics['nrmse_stress'].append(s_rmse / stress_range)
        step_metrics['r2_stress'].append(s_r2)

        # Append Deform X
        step_metrics['rmse_deform_x'].append(dx_rmse)
        step_metrics['mae_deform_x'].append(dx_mae)
        step_metrics['nrmse_deform_x'].append(dx_rmse / deform_x_range)
        step_metrics['r2_deform_x'].append(dx_r2)

        # Append Deform Y
        step_metrics['rmse_deform_y'].append(dy_rmse)
        step_metrics['mae_deform_y'].append(dy_mae)
        step_metrics['nrmse_deform_y'].append(dy_rmse / deform_y_range)
        step_metrics['r2_deform_y'].append(dy_r2)
        
        # Append Deform Z
        step_metrics['rmse_deform_z'].append(dz_rmse)
        step_metrics['mae_deform_z'].append(dz_mae)
        step_metrics['nrmse_deform_z'].append(dz_rmse / deform_z_range)
        step_metrics['r2_deform_z'].append(dz_r2)

    # Final Pooled Metrics
    pooled_metrics = {
        'mae_stress': stress_abs_err_sum / total_valid_nodes,
        'rmse_stress': (stress_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_stress': (stress_abs_err_sum / total_valid_nodes) / stress_range,
        'nrmse_stress': ((stress_sq_err_sum / total_valid_nodes) ** 0.5) / stress_range,
        'r2_stress': 1.0 - (stress_sq_err_sum / global_stress_var_sum) if global_stress_var_sum > 0 else 0.0,

        'mae_deform_x': deform_x_abs_err_sum / total_valid_nodes,
        'rmse_deform_x': (deform_x_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_deform_x': (deform_x_abs_err_sum / total_valid_nodes) / deform_x_range,
        'nrmse_deform_x': ((deform_x_sq_err_sum / total_valid_nodes) ** 0.5) / deform_x_range,
        'r2_deform_x': 1.0 - (deform_x_sq_err_sum / global_deform_x_var_sum) if global_deform_x_var_sum > 0 else 0.0,

        'mae_deform_y': deform_y_abs_err_sum / total_valid_nodes,
        'rmse_deform_y': (deform_y_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_deform_y': (deform_y_abs_err_sum / total_valid_nodes) / deform_y_range,
        'nrmse_deform_y': ((deform_y_sq_err_sum / total_valid_nodes) ** 0.5) / deform_y_range,
        'r2_deform_y': 1.0 - (deform_y_sq_err_sum / global_deform_y_var_sum) if global_deform_y_var_sum > 0 else 0.0,
        
        'mae_deform_z': deform_z_abs_err_sum / total_valid_nodes,
        'rmse_deform_z': (deform_z_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_deform_z': (deform_z_abs_err_sum / total_valid_nodes) / deform_z_range,
        'nrmse_deform_z': ((deform_z_sq_err_sum / total_valid_nodes) ** 0.5) / deform_z_range,
        'r2_deform_z': 1.0 - (deform_z_sq_err_sum / global_deform_z_var_sum) if global_deform_z_var_sum > 0 else 0.0
    }

    return pooled_metrics, step_metrics

def export_and_plot_time_series(aggregated_step_metrics, output_dir, skip=3, show_transitions=True):
    """
    Exports plot data to CSV and generates the final layered graphs.
    """
    plot_dir = os.path.join(output_dir, "aggregated_plots")
    csv_dir = os.path.join(output_dir, "plot_data_csvs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    for metric_name, error_matrix in aggregated_step_metrics.items():
        error_matrix = np.array(error_matrix) # Shape: [num_trajectories, num_timesteps]
        num_steps = error_matrix.shape[1]
        real_time_steps = [s * skip for s in range(1, num_steps + 1)]

        # Calculate Statistics
        mean_err = np.mean(error_matrix, axis=0)
        std_err = np.std(error_matrix, axis=0)
        min_err = np.min(error_matrix, axis=0)
        max_err = np.max(error_matrix, axis=0)

        # 1. EXPORT DATA TO CSV
        df = pd.DataFrame({
            'Timestep': real_time_steps,
            'Mean': mean_err,
            'Std_Dev': std_err,
            'Min': min_err,
            'Max': max_err
        })
        csv_path = os.path.join(csv_dir, f"{metric_name}_plot_data.csv")
        df.to_csv(csv_path, index=False)

        # 2. GENERATE THE GRAPH
        plt.figure(figsize=(10, 6))

        if 'r2' in metric_name:
            color_theme = 'green'
            lower_std = mean_err - std_err
            upper_std = np.minimum(mean_err + std_err, 1.0)
            plt.ylim(max(0.0, np.min(min_err)*0.9), 1.05)
            y_label = 'Spatial R² Score'
        else:
            color_theme = 'blue'
            lower_std = np.maximum(mean_err - std_err, 0.0)
            upper_std = mean_err + std_err
            plt.ylim(0, np.max(max_err) * 1.1)
            y_label = 'Relative Error' if 'nrmse' in metric_name else 'Absolute Error'

        # Outer Shaded Region: Min / Max
        plt.fill_between(real_time_steps, min_err, max_err, color='gray', alpha=0.15, label='Absolute Min/Max')
        # Inner Shaded Region: Mean +/- 1 Std Dev
        plt.fill_between(real_time_steps, lower_std, upper_std, color=color_theme, alpha=0.3, label='±1 Std Dev')
        # Central Line: Mean
        plt.plot(real_time_steps, mean_err, color=color_theme, linewidth=2.5, label=f'Mean {metric_name.upper()}')

        # Vertical Transition Lines
        if show_transitions:
            trans1 = (num_steps // 3) * skip
            trans2 = 2 * (num_steps // 3) * skip
            plt.axvline(x=trans1, color='black', linestyle='--', alpha=0.8, label='Stage Transition')
            plt.axvline(x=trans2, color='black', linestyle='--', alpha=0.8)

        # Formatting
        plt.title(f'Aggregated Dynamic Performance: {metric_name.replace("_", " ").title()}')
        plt.xlabel('Simulation Timestep')
        plt.ylabel(y_label)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='best')
        plt.xlim(0, real_time_steps[-1] + skip)

        plot_path = os.path.join(plot_dir, f"{metric_name}_time_series.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

def evaluate_and_generate_report(rollout_pth, output_dir):
    print(f"\nEvaluating Rollout: {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    os.makedirs(output_dir, exist_ok=True)

    # Collections
    collected_pooled_metrics = {
        'mae_stress': [], 'rmse_stress': [], 'nmae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'mae_deform_x': [], 'rmse_deform_x': [], 'nmae_deform_x': [], 'nrmse_deform_x': [], 'r2_deform_x': [],
        'mae_deform_y': [], 'rmse_deform_y': [], 'nmae_deform_y': [], 'nrmse_deform_y': [], 'r2_deform_y': [],
        'mae_deform_z': [], 'rmse_deform_z': [], 'nmae_deform_z': [], 'nrmse_deform_z': [], 'r2_deform_z': []
    }

    aggregated_step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'rmse_deform_x': [], 'mae_deform_x': [], 'nrmse_deform_x': [], 'r2_deform_x': [],
        'rmse_deform_y': [], 'mae_deform_y': [], 'nrmse_deform_y': [], 'r2_deform_y': [],
        'rmse_deform_z': [], 'mae_deform_z': [], 'nrmse_deform_z': [], 'r2_deform_z': []
    }

    start_time = time.time()
    for i in range(len(rollout)):
        print(f"  Processing Trajectory {i+1}/{len(rollout)}...", end='\r')
        pooled_metrics, step_metrics = process_single_trajectory(rollout[i])
        
        for key in collected_pooled_metrics.keys():
            collected_pooled_metrics[key].append(pooled_metrics[key])

        for key in aggregated_step_metrics.keys():
            aggregated_step_metrics[key].append(step_metrics[key])

    print("\n  Processing complete. Exporting CSVs and Plots...")
    export_and_plot_time_series(aggregated_step_metrics, output_dir, skip=3)

    def format_metric(data_list, is_percentage=False):
        mean_val = np.mean(data_list)
        std_val = np.std(data_list)
        if is_percentage:
            return f"{(mean_val * 100):.3f}% ± {(std_val * 100):.3f}%"
        else:
            return f"{mean_val:.5f} ± {std_val:.5f}"

    final_summary = {
        "metadata": {
            "total_trajectories_evaluated": len(rollout),
            "note": "Reported as (Mean ± 1 Standard Deviation) across all trajectories. Global R2 explains variance across full spatial-temporal domain."
        },
        "stress_metrics": {
            "R2_Score_(Global)": format_metric(collected_pooled_metrics['r2_stress'], is_percentage=False),
            "MAE_raw": format_metric(collected_pooled_metrics['mae_stress'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_stress'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_stress'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_stress'], is_percentage=True)
        },
        "deformation_x_metrics": {
            "R2_Score_(Global)": format_metric(collected_pooled_metrics['r2_deform_x'], is_percentage=False),
            "MAE_raw": format_metric(collected_pooled_metrics['mae_deform_x'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_deform_x'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_deform_x'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_deform_x'], is_percentage=True)
        },
        "deformation_y_metrics": {
            "R2_Score_(Global)": format_metric(collected_pooled_metrics['r2_deform_y'], is_percentage=False),
            "MAE_raw": format_metric(collected_pooled_metrics['mae_deform_y'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_deform_y'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_deform_y'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_deform_y'], is_percentage=True)
        },
        "deformation_z_metrics": {
            "R2_Score_(Global)": format_metric(collected_pooled_metrics['r2_deform_z'], is_percentage=False),
            "MAE_raw": format_metric(collected_pooled_metrics['mae_deform_z'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_deform_z'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_deform_z'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_deform_z'], is_percentage=True)
        }
    }

    json_path = os.path.join(output_dir, "journal_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"  Results saved to: {output_dir}")
    print(f"  Time taken: {time.time() - start_time:.2f}s")


def main():
    # Provide the full paths to your .pkl rollout files here
    rollout_files = [
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/dilated_dgcnn/coarse_1500_train_val/Dilated_C_ST2_K20_D100/rollout/rollout_epoch_240.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/dilated_dgcnn/coarse_1500_train_val/Dilated_C_ST3_K20_D100/rollout/rollout_epoch_80.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/DGCNN_C_ST3_80/rollout/rollout_epoch_20.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dgcnn/regDGCNN_seg/coarse_1500_train_val/DGCNN_C_ST2_K80/rollout/rollout_epoch_60.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E2_F/encode_process_decode/coarse_1500_train_val/Thu-Apr-23-06-00-27-2026/rollout/rollout_epoch_20.pkl",
        #"/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/encode_process_decode/coarse_1500_train_val/Wed-Apr-22-02-37-10-2026/rollout/rollout_epoch_60.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver_runs_braj/transolver/coarse_1500_train_val/Transolver_C_ST3_64/rollout/rollout_epoch_760.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/dilated_dgcnn/coarse_1500_train_val/Sat-Apr-25-12-36-39-2026/rollout/rollout_epoch_20.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/dilated_dgcnn/coarse_1500_train_val/Sat-Apr-25-12-36-39-2026/rollout/rollout_epoch_320.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Single_stage/Transolver_C_single_stage/rollout/rollout_epoch_220.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_Monolithic/infer_unseen_rollout_180/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/dilated_dgcnn/coarse_1500_train_val/Sat-Apr-25-12-36-39-2026/infer_extrapolated/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/dilated_dgcnn/coarse_1500_train_val/Sat-Apr-25-12-36-39-2026/infer_unseen_rollout/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_Monolithic/infer_extrapolated/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Reg_DGCNN_Monolithic/checkpoint/infer_extrapolated/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Reg_DGCNN_Monolithic/checkpoint/infer_unseen_rollout/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Single_stage/Transolver_C_single_stage/infer_extrapolation/infer_rollout.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Single_stage/Transolver_C_single_stage/infer_unseen_rollout/infer_rollout.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Extrapolation/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Unseen_test/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Validationset/val/concatenated_rollout_all.pkl",
        #"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_Extrapolation_1step/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/gcn/Inference_new/GCN_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Inference_new/DGCNN_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_Extrapolation/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_OOD_Unseen/test/concatenated_rollout_all.pkl",
       # "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/New_Combined/Inference_Retrain_OOD_Unseen_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/Combined_Inference/Infer_Unseen_test_1step/test/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/Inference_Output/val_real/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real/concatenated_rollout_all.pkl",
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/Combined_Inference_K20_D100/val_real/concatenated_rollout_all.pkl",


    ]
    
    print(f"Found {len(rollout_files)} rollout files to process.")

    for rollout_pth in rollout_files:
        if os.path.exists(rollout_pth):
            rollout_dir = os.path.dirname(rollout_pth)
            # Target output folder
            output_dir = os.path.join(rollout_dir, "journal_evaluation_results_xyz")
            
            evaluate_and_generate_report(rollout_pth, output_dir)
        else:
            print(f"Error: File not found -> {rollout_pth}")

if __name__ == "__main__":
    main()