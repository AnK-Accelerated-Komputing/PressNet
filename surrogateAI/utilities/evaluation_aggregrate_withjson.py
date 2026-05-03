import torch
import pickle
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import time

def load_pkl_file(file_path):
    """ Loads the pkl file. """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def process_single_trajectory(single_trajectory):
    """
    Processes a single trajectory to compute BOTH:
    1. The single pooled metrics for the JSON table (MAE, RMSE, nMAE, nRMSE).
    2. The step-by-step metrics for the time-series plots.
    """
    def get_tensor(key):
        t = single_trajectory[key]
        if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.size(0) == 1:
            return t.squeeze(0)
        return t

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
    global_stress_max, global_stress_min = -float('inf'), float('inf')
    global_deform_max, global_deform_min = -float('inf'), float('inf')

    # --- Variables for Step-by-Step Plots ---
    step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [],
        'rmse_deform': [], 'mae_deform': [], 'nrmse_deform': []
    }

    # 1st Pass: Find Global Ranges (Needed to normalize step-by-step properly)
    for i in range(num_steps):
        if i == 1000 or i == 2000: continue
        mask = node_type[i].to('cpu').flatten() == 0
        if mask.sum() == 0: continue

        gt_s = gt_stress[i].to('cpu').flatten()[mask]
        gt_p = gt_pos[i].to('cpu')[mask]
        m_p = mesh_pos[i].to('cpu')[mask]
        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())

        global_stress_max = max(global_stress_max, torch.max(gt_s).item())
        global_stress_min = min(global_stress_min, torch.min(gt_s).item())
        global_deform_max = max(global_deform_max, torch.max(gt_y_d).item())
        global_deform_min = min(global_deform_min, torch.min(gt_y_d).item())

    stress_range = max(global_stress_max - global_stress_min, 1e-6)
    deform_range = max(global_deform_max - global_deform_min, 1e-6)

    # 2nd Pass: Accumulate Pooled Errors & Record Step Errors
    for i in range(num_steps):
        if i == 1000 or i == 2000:
            # Repeat last step for restart points
            for k in step_metrics.keys():
                if len(step_metrics[k]) > 0:
                    step_metrics[k].append(step_metrics[k][-1])
            continue

        mask = node_type[i].to('cpu').flatten() == 0
        if mask.sum() == 0: continue

        # Extract step data
        gt_s = gt_stress[i].to('cpu').flatten()[mask]
        pred_s = pred_stress[i].to('cpu').flatten()[mask]
        gt_p = gt_pos[i].to('cpu')[mask]
        pred_p = pred_pos[i].to('cpu')[mask]
        m_p = mesh_pos[i].to('cpu')[mask]

        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())
        pred_y_d = torch.abs((pred_p - m_p)[:, 1].flatten())

        # Accumulate for Pooled Metrics
        stress_abs_err_sum += torch.sum(torch.abs(gt_s - pred_s)).item()
        deform_abs_err_sum += torch.sum(torch.abs(gt_y_d - pred_y_d)).item()
        stress_sq_err_sum += torch.sum((gt_s - pred_s)**2).item()
        deform_sq_err_sum += torch.sum((gt_y_d - pred_y_d)**2).item()
        total_valid_nodes += gt_s.numel()

        # Calculate Step-by-Step Metrics
        s_rmse = torch.sqrt(torch.mean((gt_s - pred_s) ** 2)).item()
        s_mae = torch.mean(torch.abs(gt_s - pred_s)).item()
        
        d_rmse = torch.sqrt(torch.mean((gt_y_d - pred_y_d) ** 2)).item()
        d_mae = torch.mean(torch.abs(gt_y_d - pred_y_d)).item()

        step_metrics['rmse_stress'].append(s_rmse)
        step_metrics['mae_stress'].append(s_mae)
        step_metrics['nrmse_stress'].append(s_rmse / stress_range)

        step_metrics['rmse_deform'].append(d_rmse)
        step_metrics['mae_deform'].append(d_mae)
        step_metrics['nrmse_deform'].append(d_rmse / deform_range)

    # Calculate final pooled metrics
    pooled_metrics = {
        'mae_stress': stress_abs_err_sum / total_valid_nodes,
        'rmse_stress': (stress_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_stress': (stress_abs_err_sum / total_valid_nodes) / stress_range,
        'nrmse_stress': ((stress_sq_err_sum / total_valid_nodes) ** 0.5) / stress_range,
        'mae_deform': deform_abs_err_sum / total_valid_nodes,
        'rmse_deform': (deform_sq_err_sum / total_valid_nodes) ** 0.5,
        'nmae_deform': (deform_abs_err_sum / total_valid_nodes) / deform_range,
        'nrmse_deform': ((deform_sq_err_sum / total_valid_nodes) ** 0.5) / deform_range
    }

    return pooled_metrics, step_metrics

def plot_aggregated_time_series(aggregated_step_metrics, output_dir, skip=3):
    """
    Plots Mean Error ± 1 Standard Deviation across all trajectories over time.
    """
    plot_dir = os.path.join(output_dir, "aggregated_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for metric_name, error_matrix in aggregated_step_metrics.items():
        error_matrix = np.array(error_matrix) # Shape: [num_trajectories, num_timesteps]
        num_steps = error_matrix.shape[1]
        real_time_steps = [s * skip for s in range(1, num_steps + 1)]

        # Calculate Mean and Standard Deviation across trajectories
        mean_err = np.mean(error_matrix, axis=0)
        std_err = np.std(error_matrix, axis=0)

        plt.figure(figsize=(10, 6))
        
        # Plot Mean
        plt.plot(real_time_steps, mean_err, color='blue', linewidth=2, label=f'Mean {metric_name.upper()}')

        # Plot Shaded ±1 Std Dev Region
        lower_bound = np.maximum(mean_err - std_err, 0) # Error can't be negative
        upper_bound = mean_err + std_err
        plt.fill_between(real_time_steps, lower_bound, upper_bound, color='lightblue', alpha=0.5, label='±1 Std Dev')

        # Formatting
        plt.title(f'Aggregated {metric_name.replace("_", " ").title()} Over Time')
        plt.xlabel('Simulation Step')
        
        y_label = 'Relative Error (%)' if 'nrmse' in metric_name else 'Absolute Error'
        plt.ylabel(y_label)
        
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper left')
        plt.xlim(0, real_time_steps[-1] + skip)

        filename = os.path.join(plot_dir, f"{metric_name}_time_series.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Saved plot: {filename}")

def evaluate_and_generate_report(rollout_pth, output_dir):
    print(f"Loading Rollout: {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    os.makedirs(output_dir, exist_ok=True)

    # Collections for JSON
    collected_pooled_metrics = {
        'mae_stress': [], 'rmse_stress': [], 'nmae_stress': [], 'nrmse_stress': [],
        'mae_deform': [], 'rmse_deform': [], 'nmae_deform': [], 'nrmse_deform': []
    }

    # Collections for Plots
    aggregated_step_metrics = {
        'rmse_stress': [], 'mae_stress': [], 'nrmse_stress': [],
        'rmse_deform': [], 'mae_deform': [], 'nrmse_deform': []
    }

    start_time = time.time()
    for i in range(len(rollout)):
        print(f"Processing Trajectory {i+1}/{len(rollout)}...", end='\r')
        single_trajectory = rollout[i]
        
        # Get both pooled (JSON) and step-wise (Plots) metrics in one pass
        pooled_metrics, step_metrics = process_single_trajectory(single_trajectory)
        
        # Append to JSON collections
        for key in collected_pooled_metrics.keys():
            collected_pooled_metrics[key].append(pooled_metrics[key])

        # Append to Plot collections
        for key in aggregated_step_metrics.keys():
            aggregated_step_metrics[key].append(step_metrics[key])

    print("\n\nProcessing complete. Generating Plots...")
    plot_aggregated_time_series(aggregated_step_metrics, output_dir, skip=3)

    print("Generating JSON Summary...")
    # Helper to format Mean ± Std Dev
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
            "note": "Relative metrics (nMAE, nRMSE) are normalized by the global trajectory range (Max - Min)."
        },
        "stress_metrics": {
            "MAE_raw": format_metric(collected_pooled_metrics['mae_stress'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_stress'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_stress'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_stress'], is_percentage=True)
        },
        "deformation_y_metrics": {
            "MAE_raw": format_metric(collected_pooled_metrics['mae_deform'], is_percentage=False),
            "RMSE_raw": format_metric(collected_pooled_metrics['rmse_deform'], is_percentage=False),
            "Relative_MAE_(nMAE)": format_metric(collected_pooled_metrics['nmae_deform'], is_percentage=True),
            "Relative_RMSE_(nRMSE)": format_metric(collected_pooled_metrics['nrmse_deform'], is_percentage=True)
        }
    }

    json_path = os.path.join(output_dir, "journal_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"\nSaved final metrics to: {json_path}")
    print(f"Total Evaluation Time: {time.time() - start_time:.2f}s")


def main():
    # Example paths
    base_dir = "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real"
    rollout_pth = os.path.join(base_dir, 'concatenated_rollout_all.pkl')
    output_dir = os.path.join(base_dir, 'evaluation_aggregate_withjson')
    
    if os.path.exists(rollout_pth):
        evaluate_and_generate_report(rollout_pth, output_dir)
    else:
        print(f"File not found: {rollout_pth}")

if __name__ == "__main__":
    main()