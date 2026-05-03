import torch
import pickle
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import time

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def process_single_trajectory(single_trajectory):
    """
    Computes BOTH the Global Pooled metrics (for JSON) and Step-wise metrics (for Plots)
    for a single trajectory.
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
    deform_abs_err_sum, deform_sq_err_sum = 0.0, 0.0
    
    global_stress_sum, global_deform_sum = 0.0, 0.0
    global_stress_var_sum, global_deform_var_sum = 0.0, 0.0
    global_stress_max, global_stress_min = -float('inf'), float('inf')
    global_deform_max, global_deform_min = -float('inf'), float('inf')

    # --- Variables for Step-by-Step Plots ---
    step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'rmse_deform': [], 'mae_deform': [], 'nrmse_deform': [], 'r2_deform': []
    }

    # 1st Pass: Find Global Ranges & Means for Normalization and Global R2
    for i in range(num_steps):
        if i == 1000 or i == 2000: continue # Skip restart frames if applicable
        mask = node_type[i].flatten() == 0
        if mask.sum() == 0: continue

        gt_s = gt_stress[i].flatten()[mask]
        gt_p = gt_pos[i][mask]
        m_p = mesh_pos[i][mask]
        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())

        global_stress_max = max(global_stress_max, torch.max(gt_s).item())
        global_stress_min = min(global_stress_min, torch.min(gt_s).item())
        global_deform_max = max(global_deform_max, torch.max(gt_y_d).item())
        global_deform_min = min(global_deform_min, torch.min(gt_y_d).item())

        global_stress_sum += torch.sum(gt_s).item()
        global_deform_sum += torch.sum(gt_y_d).item()
        total_valid_nodes += gt_s.numel()

    stress_range = max(global_stress_max - global_stress_min, 1e-6)
    deform_range = max(global_deform_max - global_deform_min, 1e-6)
    global_stress_mean = global_stress_sum / total_valid_nodes if total_valid_nodes > 0 else 0
    global_deform_mean = global_deform_sum / total_valid_nodes if total_valid_nodes > 0 else 0

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

        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())
        pred_y_d = torch.abs((pred_p - m_p)[:, 1].flatten())

        # Accumulate for Pooled Metrics (Global Error)
        stress_abs_err_sum += torch.sum(torch.abs(gt_s - pred_s)).item()
        deform_abs_err_sum += torch.sum(torch.abs(gt_y_d - pred_y_d)).item()
        stress_sq_err_sum += torch.sum((gt_s - pred_s)**2).item()
        deform_sq_err_sum += torch.sum((gt_y_d - pred_y_d)**2).item()

        global_stress_var_sum += torch.sum((gt_s - global_stress_mean)**2).item()
        global_deform_var_sum += torch.sum((gt_y_d - global_deform_mean)**2).item()

        # --- Step-by-Step Metrics ---
        s_rmse = torch.sqrt(torch.mean((gt_s - pred_s) ** 2)).item()
        s_mae = torch.mean(torch.abs(gt_s - pred_s)).item()
        s_var = torch.sum((gt_s - torch.mean(gt_s))**2).item()
        s_r2 = 1.0 - (torch.sum((gt_s - pred_s)**2).item() / s_var) if s_var > 1e-6 else 0.0
        
        d_rmse = torch.sqrt(torch.mean((gt_y_d - pred_y_d) ** 2)).item()
        d_mae = torch.mean(torch.abs(gt_y_d - pred_y_d)).item()
        d_var = torch.sum((gt_y_d - torch.mean(gt_y_d))**2).item()
        d_r2 = 1.0 - (torch.sum((gt_y_d - pred_y_d)**2).item() / d_var) if d_var > 1e-6 else 0.0

        step_metrics['rmse_stress'].append(s_rmse)
        step_metrics['mae_stress'].append(s_mae)
        step_metrics['nrmse_stress'].append(s_rmse / stress_range)
        step_metrics['r2_stress'].append(s_r2)

        step_metrics['rmse_deform'].append(d_rmse)
        step_metrics['mae_deform'].append(d_mae)
        step_metrics['nrmse_deform'].append(d_rmse / deform_range)
        step_metrics['r2_deform'].append(d_r2)

    # Final Pooled Metrics
    pooled_metrics = {
        'mae_stress': stress_abs_err_sum / total_valid_nodes,
        'rmse_stress': (stress_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_stress': (stress_abs_err_sum / total_valid_nodes) / stress_range,
        'nrmse_stress': ((stress_sq_err_sum / total_valid_nodes) ** 0.5) / stress_range,
        'r2_stress': 1.0 - (stress_sq_err_sum / global_stress_var_sum) if global_stress_var_sum > 0 else 0.0,

        'mae_deform': deform_abs_err_sum / total_valid_nodes,
        'rmse_deform': (deform_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_deform': (deform_abs_err_sum / total_valid_nodes) / deform_range,
        'nrmse_deform': ((deform_sq_err_sum / total_valid_nodes) ** 0.5) / deform_range,
        'r2_deform': 1.0 - (deform_sq_err_sum / global_deform_var_sum) if global_deform_var_sum > 0 else 0.0
    }

    return pooled_metrics, step_metrics

def export_and_plot_time_series(aggregated_step_metrics, output_dir, skip=3):
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

        # 1. EXPORT DATA TO CSV (For future multi-model comparison)
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

        # R2 Formatting logic vs Error Formatting logic
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
            plt.ylim(0, np.max(upper_std) * 1.25) # Cap Y-axis slightly above the 1 Std Dev mark so huge outliers don't crush the mean line
            y_label = 'Relative Error' if 'nrmse' in metric_name else 'Absolute Error'

        # Outer Shaded Region: Min / Max
        plt.fill_between(real_time_steps, min_err, max_err, color='gray', alpha=0.15, label='Absolute Min/Max')

        # Inner Shaded Region: Mean +/- 1 Std Dev
        plt.fill_between(real_time_steps, lower_std, upper_std, color=color_theme, alpha=0.3, label='±1 Std Dev')

        # Central Line: Mean
        plt.plot(real_time_steps, mean_err, color=color_theme, linewidth=2.5, label=f'Mean {metric_name.upper()}')

        # Vertical Transition Lines
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
    print(f"Loading Rollout: {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    os.makedirs(output_dir, exist_ok=True)

    # Collections
    collected_pooled_metrics = {
        'mae_stress': [], 'rmse_stress': [], 'nmae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'mae_deform': [], 'rmse_deform': [], 'nmae_deform': [], 'nrmse_deform': [], 'r2_deform': []
    }

    aggregated_step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [], 'r2_stress': [],
        'rmse_deform': [], 'mae_deform': [], 'nrmse_deform': [], 'r2_deform': []
    }

    start_time = time.time()
    for i in range(len(rollout)):
        print(f"Processing Trajectory {i+1}/{len(rollout)}...", end='\r')
        pooled_metrics, step_metrics = process_single_trajectory(rollout[i])
        
        for key in collected_pooled_metrics.keys():
            collected_pooled_metrics[key].append(pooled_metrics[key])

        for key in aggregated_step_metrics.keys():
            aggregated_step_metrics[key].append(step_metrics[key])

    print("\n\nProcessing complete. Exporting CSVs and Plots...")
    export_and_plot_time_series(aggregated_step_metrics, output_dir, skip=3)

    print("Generating JSON Summary...")
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
        "deformation_y_metrics": {
            "R2_Score_(Global)": format_metric(collected_pooled_metrics['r2_deform'], is_percentage=False),
            "MAE_raw": format_metric(collected_pooled_metrics['mae_deform'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_deform'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_deform'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_deform'], is_percentage=True)
        }
    }

    json_path = os.path.join(output_dir, "journal_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"\nEvaluation Complete! Total Time: {time.time() - start_time:.2f}s")
    print(f"Results saved to: {output_dir}")

def main():
    # Update these to match the path you want to process
    base_dir = "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real"
    rollout_pth = os.path.join(base_dir, 'concatenated_rollout_all.pkl')
    output_dir = os.path.join(base_dir, 'journal_evaluation_results')
    
    if os.path.exists(rollout_pth):
        evaluate_and_generate_report(rollout_pth, output_dir)
    else:
        print(f"Error: File not found at {rollout_pth}")

if __name__ == "__main__":
    main()