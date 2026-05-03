import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from PIL import Image
import json
import mlflow  # --- ADDED MLFLOW IMPORT ---

def load_pkl_file(file_path):
    """ Load the pkl file and return its contents. """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_metrics(pred, gt):
    """
    Compute element-wise metrics for domain-wise analysis.
    """
    error = pred - gt
    l2 = torch.norm(error, p=2)
    rmse = torch.sqrt(torch.mean(error ** 2))
    max_rmse = torch.max(torch.abs(pred - gt)) # True max absolute error
    return l2.item(), rmse.item(), max_rmse.item()

def obtain_step_loss(single_trajectory):
    """
    Calculates step-wise loss including Relative Error and True Max Error.
    """
    step_loss = {
        'stress': [],               # RMSE
        'max_stress': [],           # Peak Difference
        'max_abs_error_stress': [], # True Max Error
        'rel_rmse_stress': [],      # Relative RMSE
        
        'deform_y': [],
        'max_deform_y': [],
        'max_abs_error_deform_y': [],
        'rel_rmse_deform_y': []
    }
    
    gt = {'stress': [], 'max_stress': [], 'deform_y': [], 'max_deform_y': []}
    pred = {'stress': [], 'max_stress': [], 'deform_y': [], 'max_deform_y': []}

    for i in range(single_trajectory['node_type'].size(dim=0)):
        node_type = single_trajectory['node_type'][i].to('cpu').flatten()
        mask = node_type == 0

        # --- Extract Data ---
        gt_stress = single_trajectory['gt_stress'][i].to('cpu').flatten()[mask]
        pred_stress = single_trajectory['pred_stress'][i].to('cpu').flatten()[mask]
        
        gt_pos = single_trajectory['gt_pos'][i].to('cpu')[mask]
        pred_pos = single_trajectory['pred_pos'][i].to('cpu')[mask]
        mesh_pos = single_trajectory['mesh_pos'][i].to('cpu')[mask]

        gt_y_deform = torch.abs((gt_pos - mesh_pos)[:, 1].flatten())
        pred_y_deform = torch.abs((pred_pos - mesh_pos)[:, 1].flatten())

        # --- Calculate Ranges for Relative Error ---
        stress_range = torch.max(gt_stress) - torch.min(gt_stress)
        if stress_range == 0: stress_range = 1e-6
        
        deform_range = torch.max(gt_y_deform) - torch.min(gt_y_deform)
        if deform_range == 0: deform_range = 1e-6

        # --- Metric Calculation: Stress ---
        loss_stress_rmse = torch.sqrt(torch.mean((gt_stress - pred_stress) ** 2))
        loss_stress_peak_diff = torch.abs(torch.max(gt_stress) - torch.max(pred_stress))
        loss_stress_max_abs = torch.max(torch.abs(gt_stress - pred_stress)) # True Max
        rel_rmse_stress = loss_stress_rmse / stress_range

        # --- Metric Calculation: Deformation ---
        loss_y_rmse = torch.sqrt(torch.mean((gt_y_deform - pred_y_deform) ** 2))
        loss_y_peak_diff = torch.abs(torch.max(gt_y_deform) - torch.max(pred_y_deform))
        loss_y_max_abs = torch.max(torch.abs(gt_y_deform - pred_y_deform)) # True Max
        rel_rmse_y = loss_y_rmse / deform_range

        # --- Append to Lists (Handling Skip Logic for Restart Points) ---
        if i == 199 or i == 299:
            # Repeat last value for continuity at restart points
            for d in [gt, pred, step_loss]:
                for k in d.keys():
                    if len(d[k]) > 0:
                        d[k].append(d[k][-1])
        else:
            gt["deform_y"].append(torch.mean(gt_y_deform))
            gt["max_deform_y"].append(torch.max(gt_y_deform))
            pred["deform_y"].append(torch.mean(pred_y_deform))
            pred["max_deform_y"].append(torch.max(pred_y_deform))
            
            step_loss["deform_y"].append(loss_y_rmse)
            step_loss["max_deform_y"].append(loss_y_peak_diff)
            step_loss["max_abs_error_deform_y"].append(loss_y_max_abs)
            step_loss["rel_rmse_deform_y"].append(rel_rmse_y)

            # Append Stress Metrics
            gt['stress'].append(torch.mean(gt_stress))
            gt['max_stress'].append(torch.max(gt_stress))
            pred['stress'].append(torch.mean(pred_stress))
            pred['max_stress'].append(torch.max(pred_stress))
            
            step_loss['stress'].append(loss_stress_rmse)
            step_loss['max_stress'].append(loss_stress_peak_diff)
            step_loss['max_abs_error_stress'].append(loss_stress_max_abs)
            step_loss['rel_rmse_stress'].append(rel_rmse_stress)

    return step_loss, gt, pred

def obtain_domain_wise_loss(single_trajectory):
    """
    Calculates spatial (domain-wise) loss for BOTH Stress and Deformation (Y).
    """
    mesh_pos = single_trajectory['mesh_pos'].to('cpu')
    pred_stress = single_trajectory['pred_stress'].to('cpu')
    gt_stress = single_trajectory['gt_stress'].to('cpu')
    
    # Position data for deformation
    pred_pos = single_trajectory['pred_pos'].to('cpu')
    gt_pos = single_trajectory['gt_pos'].to('cpu')
    
    node_type = single_trajectory['node_type'].to('cpu')

    x_ranges = []
    x_max = mesh_pos[:,:,0].max()
    x_min = mesh_pos[:,:,0].min()
    discrete_size = 20
    x = x_min
    while x < x_max:
        if x + discrete_size < x_max:
            x_ranges.append((x+0, x+discrete_size))
        else:
            x_ranges.append((x+0, x_max))
        x += discrete_size

    # Initialize nested results structure
    results = {
        'range': [],
        'stress': {
            'l2': [], 'rmse': [], 'max_rmse': [],
            'gt': [], 'max_gt': [], 'pred': [], 'max_pred': []
        },
        'deform_y': {
            'l2': [], 'rmse': [], 'max_rmse': [],
            'gt': [], 'max_gt': [], 'pred': [], 'max_pred': []
        }
    }

    for x_min, x_max in x_ranges:
        x_coords = mesh_pos[:, :, 0]
        node_type_squeezed = node_type.squeeze(-1)
        mask = (x_coords <= x_max) & (x_coords > x_min) & (node_type_squeezed == 0)

        # 1. Process Stress
        filtered_pred_stress = pred_stress[mask]
        filtered_gt_stress = gt_stress[mask]

        # 2. Process Deformation
        filtered_pred_pos = pred_pos[mask]
        filtered_gt_pos = gt_pos[mask]
        filtered_mesh_pos = mesh_pos[mask]

        if filtered_pred_stress.numel() == 0:
            continue

        # Calculate Y-Deformation (Absolute displacement from mesh pos)
        filtered_gt_deform_y = torch.abs((filtered_gt_pos - filtered_mesh_pos)[:, 1])
        filtered_pred_deform_y = torch.abs((filtered_pred_pos - filtered_mesh_pos)[:, 1])

        # Compute Metrics: Stress
        l2_s, rmse_s, max_rmse_s = compute_metrics(filtered_pred_stress, filtered_gt_stress)
        
        # Compute Metrics: Deformation
        l2_d, rmse_d, max_rmse_d = compute_metrics(filtered_pred_deform_y, filtered_gt_deform_y)

        results['range'].append((x_min, x_max))

        # Store Stress Results
        results['stress']['l2'].append(l2_s)
        results['stress']['rmse'].append(rmse_s)
        results['stress']['max_rmse'].append(max_rmse_s)
        results['stress']['gt'].append(torch.mean(filtered_gt_stress).item())
        results['stress']['max_gt'].append(torch.max(filtered_gt_stress).item())
        results['stress']['pred'].append(torch.mean(filtered_pred_stress).item())
        results['stress']['max_pred'].append(torch.max(filtered_pred_stress).item())

        # Store Deformation Results
        results['deform_y']['l2'].append(l2_d)
        results['deform_y']['rmse'].append(rmse_d)
        results['deform_y']['max_rmse'].append(max_rmse_d)
        results['deform_y']['gt'].append(torch.mean(filtered_gt_deform_y).item())
        results['deform_y']['max_gt'].append(torch.max(filtered_gt_deform_y).item())
        results['deform_y']['pred'].append(torch.mean(filtered_pred_deform_y).item())
        results['deform_y']['max_pred'].append(torch.max(filtered_pred_deform_y).item())
        
    return x_ranges, results

def plot_step_loss_gif(step_loss, gt, pred, output_dir, key='all'):
    steps = list(range(1, len(step_loss['deform_y']) + 1))
    if key == 'all':
        keys = ['deform_y', 'max_deform_y', 'stress', 'max_stress']
    else:
        keys = [key]

    for key in keys:
        try:
            val_loss = [float(v) for v in step_loss[key]]
            val_gt = [float(v) for v in gt[key]]
            val_pred = [float(v) for v in pred[key]]
            y_max = max(max(val_loss), max(val_gt), max(val_pred))
        except:
            continue

        for step in steps:
            plt.figure(figsize=(8, 5))
            plt.plot(steps[:step], val_gt[:step], linestyle='-', color='g', label=f'gt_{key}')
            plt.plot(steps[:step], val_pred[:step], linestyle='-', color='b', label=f'pred_{key}')
            plt.plot(steps[:step], val_loss[:step], linestyle='-', color='r', label=f'loss_{key}')
            
            plt.title('Error over Steps')
            plt.xlabel('Step')
            plt.ylabel('Error')
            plt.grid(True)
            plt.legend()
            plt.xlim(0, steps[-1] + 1)
            plt.ylim(0, y_max * 1.1)
            
            filename = os.path.join(output_dir, key, "steps", f"frame_{step:03d}.png")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            
            if step == steps[-1]:
                final_filename = os.path.join(output_dir, key, f"frame_{step:03d}.png")
                plt.savefig(final_filename)
                
                # --- MLFLOW ARTIFACT LOGGING: Save final graph ---
                if mlflow.active_run():
                    traj_id = os.path.basename(output_dir)
                    mlflow.log_artifact(final_filename, artifact_path=f"evaluation_plots/traj_{traj_id}/step_loss/{key}")

            plt.close()
        
        # Create GIF
        frames = []
        for step in steps:
            filename = os.path.join(output_dir, key, "steps", f"frame_{step:03d}.png")
            if os.path.exists(filename):
                frames.append(Image.open(filename))

        if frames:
            gif_dir = os.path.join(output_dir, key)
            gif_filename = os.path.join(gif_dir, f"{key}_error_over_steps.gif")
            fps = 50
            frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
            print(f"GIF saved as {gif_filename}")

            # --- MLFLOW ARTIFACT LOGGING: Save generated GIF ---
            if mlflow.active_run():
                traj_id = os.path.basename(output_dir)
                mlflow.log_artifact(gif_filename, artifact_path=f"evaluation_plots/traj_{traj_id}/step_loss/{key}")

def plot_domain_wise_loss(x_ranges, results, output_dir, key='all'):
    # Calculate midpoints for plotting on a continuous axis
    x_range_labels = [(results['range'][p][0] + results['range'][p][1])/2 for p in range(len(results['range']))]
    
    if key == 'all':
        plot_keys = ['stress', 'max_stress', 'deform_y', 'max_deform_y']
    else:
        plot_keys = [key]
    
    for p_key in plot_keys:
        # Determine Category (stress vs deform) and Metric (avg vs max)
        if 'stress' in p_key:
            category = 'stress'
        elif 'deform' in p_key:
            category = 'deform_y'
        else:
            continue
            
        if 'max' in p_key:
            metric_err = 'max_rmse'
            metric_gt = 'max_gt'
            metric_pred = 'max_pred'
        else:
            metric_err = 'rmse'
            metric_gt = 'gt'
            metric_pred = 'pred'

        # Extract data from the nested results dict
        try:
            errors = results[category][metric_err]
            gt_vals = results[category][metric_gt]
            pred_vals = results[category][metric_pred]
        except KeyError:
            print(f"Warning: Could not find data for {p_key}")
            continue

        if not errors: continue

        max_error = max(errors)
        min_error = min(errors)
        
        # Normalize errors for bar colors
        denom = max_error - min_error if max_error != min_error else 1.0
        normalized_errors = [(error - min_error) / denom for error in errors]
        
        colors = plt.cm.coolwarm(normalized_errors)

        plt.figure(figsize=(10, 6))
        
        # 1. Plot GT and Pred Lines
        plt.plot(x_range_labels, gt_vals, color='g', marker='o', linewidth=2, label=f'GT {p_key}')
        plt.plot(x_range_labels, pred_vals, color='b', marker='o', linewidth=2, label=f'Pred {p_key}')
        
        # 2. Plot Error Bars (Histogram)
        bars = plt.bar(x_range_labels, errors, width=20, color=colors, edgecolor='black', align='center', alpha=0.7, label='Error Mag')
        
        # 3. Plot Error Trend Line
        plt.plot(x_range_labels, errors, color='r', marker='o', linewidth=2, label='Error Trend')

        plt.xlabel('X-coordinate', fontsize=14)
        plt.ylabel('Value / Error', fontsize=14)
        plt.title(f'{p_key} across Domain', fontsize=16)
        
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        
        # Dynamic Y-Limits
        all_values = gt_vals + pred_vals + errors
        y_min = min(all_values)
        y_max = max(all_values)
        
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        plt.ylim(y_min - padding, y_max + padding)
        
        if x_range_labels:
            plt.xlim(min(x_range_labels) - 30, max(x_range_labels) + 30)
        
        plt.legend()
        
        filename = os.path.join(output_dir, p_key, 'domain_wise_error.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved at {filename}")

        # --- MLFLOW ARTIFACT LOGGING: Save Domain Wise Plot ---
        if mlflow.active_run():
            traj_id = os.path.basename(output_dir)
            mlflow.log_artifact(filename, artifact_path=f"evaluation_plots/traj_{traj_id}/domain_wise/{p_key}")

def evaluate_rollout(rollout_pth, base_output_dir, plot=True):
    print(f"Rollout Evaluation for {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    
    # Store trajectory-wise summaries
    average_deformation_error = []
    max_deformation_error = []

    for i in range(len(rollout)):
        start_time = time.time()
        single_trajectory = rollout[i]
        
        output_dir = os.path.join(base_output_dir, str(i))
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Trajectory {i} ---")

        # 1. Step Loss (Time-series)
        print("Calculating step loss...")
        step_loss, gt, pred = obtain_step_loss(single_trajectory)
        
        # Calculate Aggregates
        step_stress_rmse = torch.mean(torch.tensor(step_loss['stress'])).item()
        step_stress_peak_diff = torch.max(torch.tensor(step_loss['max_stress'])).item()
        step_stress_max_abs = torch.max(torch.tensor(step_loss['max_abs_error_stress'])).item()
        step_stress_rel_error = torch.mean(torch.tensor(step_loss['rel_rmse_stress'])).item()

        step_y_rmse = torch.mean(torch.tensor(step_loss['deform_y'])).item()
        step_y_peak_diff = torch.max(torch.tensor(step_loss['max_deform_y'])).item()
        step_y_max_abs = torch.max(torch.tensor(step_loss['max_abs_error_deform_y'])).item()
        step_y_rel_error = torch.mean(torch.tensor(step_loss['rel_rmse_deform_y'])).item()

        # Save simple pkl summaries
        average_deformation_error.append(step_y_rmse)
        max_deformation_error.append(step_y_peak_diff)
        
        with open(os.path.join(output_dir, 'Average_deformation_error.pkl'), 'wb') as f:
            pickle.dump(average_deformation_error, f)
        with open(os.path.join(output_dir, 'Max_deformation_error.pkl'), 'wb') as f:
            pickle.dump(max_deformation_error, f)

        if plot:
            plot_step_loss_gif(step_loss, gt, pred, output_dir)

        # 2. Domain Loss (Spatial) - Updated for Deform support
        print("Calculating domain loss...")
        x_ranges, domain_loss = obtain_domain_wise_loss(single_trajectory)
        
        # Extract Stress Metrics
        domain_stress_rmse_avg = torch.mean(torch.tensor(domain_loss['stress']['rmse'])).item()
        domain_stress_max_error = torch.max(torch.tensor(domain_loss['stress']['max_rmse'])).item()
        
        # Extract Deformation Metrics
        domain_deform_rmse_avg = torch.mean(torch.tensor(domain_loss['deform_y']['rmse'])).item()
        domain_deform_max_error = torch.max(torch.tensor(domain_loss['deform_y']['max_rmse'])).item()
        
        if plot:
            plot_domain_wise_loss(x_ranges, domain_loss, output_dir)

        # 3. Compile JSON Data
        losses = {
            # --- Stress Metrics (Step) ---
            'step_stress_rmse': step_stress_rmse,
            'step_stress_peak_diff': step_stress_peak_diff,
            'step_stress_true_max_error': step_stress_max_abs,
            'step_stress_rel_error': step_stress_rel_error,
            
            # --- Deformation Metrics (Step) ---
            'step_y_deform_rmse': step_y_rmse,
            'step_y_deform_peak_diff': step_y_peak_diff,
            'step_y_deform_true_max_error': step_y_max_abs,
            'step_y_deform_rel_error': step_y_rel_error,
            
            # --- Domain Metrics (Spatial) ---
            'domain_stress_rmse_avg': domain_stress_rmse_avg,
            'domain_stress_max_error': domain_stress_max_error,
            'domain_deform_rmse_avg': domain_deform_rmse_avg,
            'domain_deform_max_error': domain_deform_max_error
        }

        # 4. Save to JSON
        json_file_path = os.path.join(os.path.dirname(base_output_dir), 'rollout_info.json')
        
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as file:
                    info = json.load(file)
            else:
                info = {'losses': []}
        except Exception as e:
            print(f"JSON Read Error: {e}, creating new.")
            info = {'losses': []}

        if 'losses' not in info or not isinstance(info['losses'], list):
            info['losses'] = []
            
        info['losses'].append(losses)

        with open(json_file_path, "w") as file:
            json.dump(info, file, indent=4)
            
        print("Appended metrics to JSON.")
        
        # --- MLFLOW ARTIFACT LOGGING: Save the JSON file containing all metadata info ---
        if mlflow.active_run():
            mlflow.log_artifact(json_file_path, artifact_path="evaluation_metrics")
            
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

    return

def main():
    # Update these paths to your actual file locations
    base_dir = "/home/sushil/PressNet/datasetsoutput/encode_process_decode/coarse_1500_train_val/MGN_C_ST1_MP3_WR20/rollout"
    rollout_pth = os.path.join(base_dir, 'rollout_epoch_171.pkl')
    output_dir = os.path.join(base_dir, 'rollout_evaluation')
    
    # --- MLFLOW SETUP FOR STANDALONE RUNS ---
    # Setup MLflow here so that standalone execution also successfully logs metrics
    mlflow.set_tracking_uri("home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/mlruns")
    mlflow.set_experiment("Standalone_Evaluation")
    
    # Start the MLflow run context
    with mlflow.start_run(run_name="MGN_C_ST1_MP3_WR20_Eval_Epoch_171"):
        # Set plot=True to generate GIFs and graphs
        evaluate_rollout(rollout_pth, output_dir, plot=True)

if __name__ == "__main__":
    main()