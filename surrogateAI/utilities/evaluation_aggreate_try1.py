import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import mlflow 

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def obtain_step_loss_for_trajectory(single_trajectory):
    """
    Calculates step-wise loss for a single trajectory.
    Returns dictionaries of arrays for different metrics.
    """
    def get_tensor(key):
        t = single_trajectory[key]
        if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.size(0) == 1:
            return t.squeeze(0)
        return t

    node_type = get_tensor('node_type')
    gt_stress = get_tensor('gt_stress')
    gt_pos = get_tensor('gt_pos')
    mesh_pos = get_tensor('mesh_pos')
    pred_stress = get_tensor('pred_stress')
    pred_pos = get_tensor('pred_pos')

    num_steps = node_type.size(dim=0) 

    # --- Pre-compute global constants for this trajectory ---
    global_stress_max, global_stress_min = -float('inf'), float('inf')
    global_deform_max, global_deform_min = -float('inf'), float('inf')

    for i in range(num_steps):
        # Handle skipped frames / restart points if needed
        if i == 1000 or i == 2000: 
            continue
            
        mask = node_type[i].to('cpu').flatten() == 0
        gt_s = gt_stress[i].to('cpu').flatten()[mask]
        gt_p = gt_pos[i].to('cpu')[mask]
        m_p = mesh_pos[i].to('cpu')[mask]
        gt_y_d = torch.abs((gt_p - m_p)[:, 1].flatten())
        
        if gt_s.numel() > 0:
            global_stress_max = max(global_stress_max, torch.max(gt_s).item())
            global_stress_min = min(global_stress_min, torch.min(gt_s).item())
            global_deform_max = max(global_deform_max, torch.max(gt_y_d).item())
            global_deform_min = min(global_deform_min, torch.min(gt_y_d).item())

    global_stress_range = max(global_stress_max - global_stress_min, 1e-6)
    global_deform_range = max(global_deform_max - global_deform_min, 1e-6)

    # Dictionaries to store step-by-step metrics
    traj_metrics = {
        'rmse_stress': [], 'nrmse_stress': [], 'mae_stress': [],
        'rmse_deform': [], 'nrmse_deform': [], 'mae_deform': []
    }

    # --- Main Timestep Loop ---
    for i in range(num_steps):
        # Repeat previous values for restart indices
        if i == 1000 or i == 2000:
            for k in traj_metrics.keys():
                if len(traj_metrics[k]) > 0:
                    traj_metrics[k].append(traj_metrics[k][-1])
            continue

        mask = node_type[i].to('cpu').flatten() == 0

        gt_s_step = gt_stress[i].to('cpu').flatten()[mask]
        pred_s_step = pred_stress[i].to('cpu').flatten()[mask]
        
        gt_p_step = gt_pos[i].to('cpu')[mask]
        pred_p_step = pred_pos[i].to('cpu')[mask]
        m_p_step = mesh_pos[i].to('cpu')[mask]

        gt_y_deform = torch.abs((gt_p_step - m_p_step)[:, 1].flatten())
        pred_y_deform = torch.abs((pred_p_step - m_p_step)[:, 1].flatten())

        # Metric Calculation: Stress
        rmse_stress = torch.sqrt(torch.mean((gt_s_step - pred_s_step) ** 2)).item()
        mae_stress = torch.mean(torch.abs(gt_s_step - pred_s_step)).item()
        nrmse_stress = rmse_stress / global_stress_range
        
        # Metric Calculation: Deformation
        rmse_deform = torch.sqrt(torch.mean((gt_y_deform - pred_y_deform) ** 2)).item()
        mae_deform = torch.mean(torch.abs(gt_y_deform - pred_y_deform)).item()
        nrmse_deform = rmse_deform / global_deform_range

        # Append to lists
        traj_metrics['rmse_stress'].append(rmse_stress)
        traj_metrics['nrmse_stress'].append(nrmse_stress)
        traj_metrics['mae_stress'].append(mae_stress)

        traj_metrics['rmse_deform'].append(rmse_deform)
        traj_metrics['nrmse_deform'].append(nrmse_deform)
        traj_metrics['mae_deform'].append(mae_deform)

    return traj_metrics

def plot_aggregated_metrics(aggregated_metrics, output_dir, skip=3):
    """
    Plots the aggregated mean error over timesteps with a shaded region 
    representing Min/Max and +/- 1 Standard Deviation.
    """
    os.makedirs(os.path.join(output_dir, "aggregated_plots"), exist_ok=True)

    for metric_name, error_matrix in aggregated_metrics.items():
        # error_matrix shape: [num_trajectories, num_timesteps]
        # Convert to numpy for easy row-wise operations
        error_matrix = np.array(error_matrix)
        
        num_steps = error_matrix.shape[1]
        real_time_steps = [s * skip for s in range(1, num_steps + 1)]

        # Calculate Mean, Min, Max, and Std Dev across all trajectories
        mean_err = np.mean(error_matrix, axis=0)
        min_err = np.min(error_matrix, axis=0)
        max_err = np.max(error_matrix, axis=0)
        std_err = np.std(error_matrix, axis=0)

        plt.figure(figsize=(10, 6))
        
        # Plot Mean
        plt.plot(real_time_steps, mean_err, color='blue', linewidth=2, label=f'Mean {metric_name}')

        # 1. Shaded Region for Min/Max
        plt.fill_between(real_time_steps, min_err, max_err, color='lightblue', alpha=0.3, label='Min-Max Range')
        
        # 2. (Recommended) Shaded Region for Standard Deviation
        # Ensures lower bound doesn't drop below 0
        lower_std = np.maximum(mean_err - std_err, 0)
        upper_std = mean_err + std_err
        plt.fill_between(real_time_steps, lower_std, upper_std, color='darkblue', alpha=0.3, label='±1 Std Dev')

        # Add Transition Lines (if applicable)
        trans1 = (num_steps // 3) * skip
        trans2 = 2 * (num_steps // 3) * skip
        plt.axvline(x=trans1, color='black', linestyle='--', alpha=0.6, label='Stage Transition')
        plt.axvline(x=trans2, color='black', linestyle='--', alpha=0.6)

        plt.title(f'Aggregated {metric_name.replace("_", " ").title()} Over Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Error Value')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper left')
        plt.xlim(0, real_time_steps[-1] + skip)
        plt.ylim(0, np.max(max_err) * 1.1)

        filename = os.path.join(output_dir, "aggregated_plots", f"{metric_name}_over_time.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300) # High DPI for Journal
        plt.close()
        
        print(f"Saved aggregated plot: {filename}")

        if mlflow.active_run():
            mlflow.log_artifact(filename, artifact_path="evaluation_plots/aggregated")


def evaluate_aggregated_rollout(rollout_pth, base_output_dir):
    print(f"Rollout Evaluation for {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    os.makedirs(base_output_dir, exist_ok=True)

    # Dictionary to hold lists of lists (Trajectories x Timesteps)
    aggregated_metrics = {
        'rmse_stress': [], 'nrmse_stress': [], 'mae_stress': [],
        'rmse_deform': [], 'nrmse_deform': [], 'mae_deform': []
    }

    start_time = time.time()
    for i in range(len(rollout)):
        print(f"Extracting metrics from Trajectory {i} / {len(rollout)-1}...")
        single_trajectory = rollout[i]
        
        traj_metrics = obtain_step_loss_for_trajectory(single_trajectory)
        
        # Append each metric's time-series to the aggregated dictionary
        for k in aggregated_metrics.keys():
            aggregated_metrics[k].append(traj_metrics[k])

    print("Generating Aggregated Plots...")
    # Generate the combined shaded plots
    plot_aggregated_metrics(aggregated_metrics, base_output_dir, skip=3)

    end_time = time.time()
    print(f"Total Evaluation Time: {end_time - start_time:.2f}s")


def main():
    base_directories = [
        # List your paths here
        "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real",
    ]
    
    for base_dir in base_directories:
        if not os.path.exists(base_dir):
            continue

        rollout_pth = os.path.join(base_dir, 'concatenated_rollout_all.pkl')
        output_dir = os.path.join(base_dir, 'rollout_evaluation_aggregated')
        
        if os.path.exists(rollout_pth):
            print(f"\n{'='*40}")
            print(f"PROCESSING: {rollout_pth}")
            print(f"{'='*40}\n")
            
            evaluate_aggregated_rollout(rollout_pth, output_dir)
        else:
            print(f"File not found: {rollout_pth}")

if __name__ == "__main__":
    main()