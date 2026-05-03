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
    # Use torch.load instead of pickle.load for files saved with torch.save
    # map_location='cpu' ensures we don't accidentally fill up GPU VRAM during evaluation
    
    return torch.load(file_path, map_location='cpu', weights_only=False)
    #with open(file_path, 'rb') as f:
       # return pickle.load(f)

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
    Calculates step-wise loss including Relative Error (nRMSE), R2, and Trajectory metrics.
    """
    step_loss = {
        'stress': [], 'max_stress': [], 'max_abs_error_stress': [], 'rel_rmse_stress': [], 'r2_stress': [],
        'deform_y': [], 'max_deform_y': [], 'max_abs_error_deform_y': [], 'rel_rmse_deform_y': [], 'r2_deform_y': []
    }
    gt = {'stress': [], 'max_stress': [], 'deform_y': [], 'max_deform_y': []}
    pred = {'stress': [], 'max_stress': [], 'deform_y': [], 'max_deform_y': []}

    # --- THE MAGIC FIX: Strip the sneaky batch dimension ---
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

    # Now this correctly counts the timesteps!
    num_steps = node_type.size(dim=0) 

    # --- 1. PRE-COMPUTE GLOBAL CONSTANTS FOR TRAJECTORY METRICS ---
    global_stress_max, global_stress_min = -float('inf'), float('inf')
    global_deform_max, global_deform_min = -float('inf'), float('inf')
    global_stress_sum, global_deform_sum = 0.0, 0.0
    total_valid_nodes = 0

    for i in range(num_steps):
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
            global_stress_sum += torch.sum(gt_s).item()
            
            global_deform_max = max(global_deform_max, torch.max(gt_y_d).item())
            global_deform_min = min(global_deform_min, torch.min(gt_y_d).item())
            global_deform_sum += torch.sum(gt_y_d).item()
            
            total_valid_nodes += gt_s.numel()

    global_stress_range = max(global_stress_max - global_stress_min, 1e-6)
    global_deform_range = max(global_deform_max - global_deform_min, 1e-6)
    global_stress_mean = global_stress_sum / total_valid_nodes if total_valid_nodes > 0 else 0.0
    global_deform_mean = global_deform_sum / total_valid_nodes if total_valid_nodes > 0 else 0.0

    traj_stress_se_sum, traj_stress_var_sum = 0.0, 0.0
    traj_deform_se_sum, traj_deform_var_sum = 0.0, 0.0

    # --- 2. MAIN TIMESTEP LOOP ---
    for i in range(num_steps):
        mask = node_type[i].to('cpu').flatten() == 0

        # Extract Data (Your original clean logic!)
        gt_s_step = gt_stress[i].to('cpu').flatten()[mask]
        pred_s_step = pred_stress[i].to('cpu').flatten()[mask]
        
        gt_p_step = gt_pos[i].to('cpu')[mask]
        pred_p_step = pred_pos[i].to('cpu')[mask]
        m_p_step = mesh_pos[i].to('cpu')[mask]

        gt_y_deform = torch.abs((gt_p_step - m_p_step)[:, 1].flatten())
        pred_y_deform = torch.abs((pred_p_step - m_p_step)[:, 1].flatten())

        # Metric Calculation: Stress
        step_stress_se = torch.sum((gt_s_step - pred_s_step) ** 2)
        step_stress_var = torch.sum((gt_s_step - torch.mean(gt_s_step)) ** 2)
        loss_stress_rmse = torch.sqrt(torch.mean((gt_s_step - pred_s_step) ** 2))
        loss_stress_peak_diff = torch.abs(torch.max(gt_s_step) - torch.max(pred_s_step))
        loss_stress_max_abs = torch.max(torch.abs(gt_s_step - pred_s_step)) 
        rel_rmse_stress = (loss_stress_rmse / global_stress_range).item() 
        r2_stress = (1.0 - (step_stress_se / step_stress_var)).item() if step_stress_var > 1e-6 else 0.0

        # Metric Calculation: Deformation
        step_deform_se = torch.sum((gt_y_deform - pred_y_deform) ** 2)
        step_deform_var = torch.sum((gt_y_deform - torch.mean(gt_y_deform)) ** 2)
        loss_y_rmse = torch.sqrt(torch.mean((gt_y_deform - pred_y_deform) ** 2))
        loss_y_peak_diff = torch.abs(torch.max(gt_y_deform) - torch.max(pred_y_deform))
        loss_y_max_abs = torch.max(torch.abs(gt_y_deform - pred_y_deform)) 
        rel_rmse_y = (loss_y_rmse / global_deform_range).item() 
        r2_y = (1.0 - (step_deform_se / step_deform_var)).item() if step_deform_var > 1e-6 else 0.0

        # Append to Lists
        if i == 1000 or i == 2000:
            for d in [gt, pred, step_loss]:
                for k in d.keys():
                    if len(d[k]) > 0:
                        d[k].append(d[k][-1])
        else:
            traj_stress_se_sum += step_stress_se.item()
            traj_stress_var_sum += torch.sum((gt_s_step - global_stress_mean)**2).item()
            traj_deform_se_sum += step_deform_se.item()
            traj_deform_var_sum += torch.sum((gt_y_deform - global_deform_mean)**2).item()

            gt["deform_y"].append(torch.mean(gt_y_deform).item())
            gt["max_deform_y"].append(torch.max(gt_y_deform).item())
            pred["deform_y"].append(torch.mean(pred_y_deform).item())
            pred["max_deform_y"].append(torch.max(pred_y_deform).item())
            
            step_loss["deform_y"].append(loss_y_rmse.item())
            step_loss["max_deform_y"].append(loss_y_peak_diff.item())
            step_loss["max_abs_error_deform_y"].append(loss_y_max_abs.item())
            step_loss["rel_rmse_deform_y"].append(rel_rmse_y)
            step_loss["r2_deform_y"].append(r2_y)

            gt['stress'].append(torch.mean(gt_s_step).item())
            gt['max_stress'].append(torch.max(gt_s_step).item())
            pred['stress'].append(torch.mean(pred_s_step).item())
            pred['max_stress'].append(torch.max(pred_s_step).item())
            
            step_loss['stress'].append(loss_stress_rmse.item())
            step_loss['max_stress'].append(loss_stress_peak_diff.item())
            step_loss['max_abs_error_stress'].append(loss_stress_max_abs.item())
            step_loss['rel_rmse_stress'].append(rel_rmse_stress)
            step_loss['r2_stress'].append(r2_stress)

    # --- 3. CALCULATE FINAL TRAJECTORY-WISE METRICS ---
    traj_metrics = {
        'traj_nrmse_stress': ((traj_stress_se_sum / total_valid_nodes)**0.5) / global_stress_range,
        'traj_r2_stress': 1.0 - (traj_stress_se_sum / traj_stress_var_sum) if traj_stress_var_sum > 0 else 0.0,
        'traj_nrmse_deform': ((traj_deform_se_sum / total_valid_nodes)**0.5) / global_deform_range,
        'traj_r2_deform': 1.0 - (traj_deform_se_sum / traj_deform_var_sum) if traj_deform_var_sum > 0 else 0.0
    }

    return step_loss, gt, pred, traj_metrics


def obtain_domain_wise_loss(single_trajectory):
    """
    Calculates spatial (domain-wise) loss for BOTH Stress and Deformation (Y).
    """
    # --- Strip batch dimensions here too ---
    def get_tensor(key):
        t = single_trajectory[key].to('cpu')
        if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.size(0) == 1:
            return t.squeeze(0)
        return t
        
    mesh_pos = get_tensor('mesh_pos')
    pred_stress = get_tensor('pred_stress')
    gt_stress = get_tensor('gt_stress')
    pred_pos = get_tensor('pred_pos')
    gt_pos = get_tensor('gt_pos')
    node_type = get_tensor('node_type')

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

    results = {
        'range': [],
        'stress': {'l2': [], 'rmse': [], 'max_rmse': [], 'gt': [], 'max_gt': [], 'pred': [], 'max_pred': []},
        'deform_y': {'l2': [], 'rmse': [], 'max_rmse': [], 'gt': [], 'max_gt': [], 'pred': [], 'max_pred': []}
    }

    for x_min, x_max in x_ranges:
        x_coords = mesh_pos[:, :, 0]
        node_type_squeezed = node_type.squeeze(-1)
        mask = (x_coords <= x_max) & (x_coords > x_min) & (node_type_squeezed == 0)

        filtered_pred_stress = pred_stress[mask]
        filtered_gt_stress = gt_stress[mask]

        filtered_pred_pos = pred_pos[mask]
        filtered_gt_pos = gt_pos[mask]
        filtered_mesh_pos = mesh_pos[mask]

        if filtered_pred_stress.numel() == 0:
            continue

        filtered_gt_deform_y = torch.abs((filtered_gt_pos - filtered_mesh_pos)[:, 1])
        filtered_pred_deform_y = torch.abs((filtered_pred_pos - filtered_mesh_pos)[:, 1])

        l2_s, rmse_s, max_rmse_s = compute_metrics(filtered_pred_stress, filtered_gt_stress)
        l2_d, rmse_d, max_rmse_d = compute_metrics(filtered_pred_deform_y, filtered_gt_deform_y)

        results['range'].append((x_min, x_max))

        results['stress']['l2'].append(l2_s)
        results['stress']['rmse'].append(rmse_s)
        results['stress']['max_rmse'].append(max_rmse_s)
        results['stress']['gt'].append(torch.mean(filtered_gt_stress).item())
        results['stress']['max_gt'].append(torch.max(filtered_gt_stress).item())
        results['stress']['pred'].append(torch.mean(filtered_pred_stress).item())
        results['stress']['max_pred'].append(torch.max(filtered_pred_stress).item())

        results['deform_y']['l2'].append(l2_d)
        results['deform_y']['rmse'].append(rmse_d)
        results['deform_y']['max_rmse'].append(max_rmse_d)
        results['deform_y']['gt'].append(torch.mean(filtered_gt_deform_y).item())
        results['deform_y']['max_gt'].append(torch.max(filtered_gt_deform_y).item())
        results['deform_y']['pred'].append(torch.mean(filtered_pred_deform_y).item())
        results['deform_y']['max_pred'].append(torch.max(filtered_pred_deform_y).item())
        
    return x_ranges, results

def plot_step_loss_gif(step_loss, gt, pred, output_dir, key='all', skip=3):

    total_evaluated_steps = len(step_loss['deform_y'])
    steps = list(range(1, total_evaluated_steps + 1))
    
    # 2. CREATE THE REAL TIMELINE: Multiply the tensor count by the skip (e.g., [3, 6, 9... 1500])
    real_time_steps = [s * skip for s in steps]
    
    # 3. CALCULATE TRANSITIONS: Divide total steps by 3, then multiply by skip
    # For 500 evaluated steps with skip=3, trans1 = 500 and trans2 = 1000
    trans1 = (total_evaluated_steps // 3) * skip
    trans2 = 2 * (total_evaluated_steps // 3) * skip

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
            plt.plot(real_time_steps[:step], val_gt[:step], linestyle='-', color='g', label=f'gt_{key}')
            plt.plot(real_time_steps[:step], val_pred[:step], linestyle='-', color='b', label=f'pred_{key}')
            plt.plot(real_time_steps[:step], val_loss[:step], linestyle='-', color='r', label=f'loss_{key}')
            
            #Adding Vertical Lines for Transitions
            plt.axvline(x=trans1, color='black', linestyle='--', alpha=0.6)
            plt.axvline(x=trans2, color='black', linestyle='--', alpha=0.6)


            plt.title('Error over Steps')
            plt.xlabel('Step')
            plt.ylabel('Error')
            plt.grid(True)
            plt.legend()
            plt.xlim(0, real_time_steps[-1] + skip)
            plt.ylim(0, y_max * 1.1)
            
            filename = os.path.join(output_dir, key, "steps", f"frame_{step:03d}.png")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            
            if step == steps[-1]:
                final_filename = os.path.join(output_dir, key, f"frame_{step:03d}.png")
                plt.savefig(final_filename)
                
                if mlflow.active_run():
                    traj_id = os.path.basename(output_dir)
                    mlflow.log_artifact(final_filename, artifact_path=f"evaluation_plots/traj_{traj_id}/step_loss/{key}")

            plt.close()
        
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

            if mlflow.active_run():
                traj_id = os.path.basename(output_dir)
                mlflow.log_artifact(gif_filename, artifact_path=f"evaluation_plots/traj_{traj_id}/step_loss/{key}")

def plot_domain_wise_loss(x_ranges, results, output_dir, key='all'):
    x_range_labels = [(results['range'][p][0] + results['range'][p][1])/2 for p in range(len(results['range']))]
    
    if key == 'all':
        plot_keys = ['stress', 'max_stress', 'deform_y', 'max_deform_y']
    else:
        plot_keys = [key]
    
    for p_key in plot_keys:
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
        
        denom = max_error - min_error if max_error != min_error else 1.0
        normalized_errors = [(error - min_error) / denom for error in errors]
        
        colors = plt.cm.coolwarm(normalized_errors)

        plt.figure(figsize=(10, 6))
        
        plt.plot(x_range_labels, gt_vals, color='g', marker='o', linewidth=2, label=f'GT {p_key}')
        plt.plot(x_range_labels, pred_vals, color='b', marker='o', linewidth=2, label=f'Pred {p_key}')
        
        bars = plt.bar(x_range_labels, errors, width=20, color=colors, edgecolor='black', align='center', alpha=0.7, label='Error Mag')
        
        plt.plot(x_range_labels, errors, color='r', marker='o', linewidth=2, label='Error Trend')

        plt.xlabel('X-coordinate', fontsize=14)
        plt.ylabel('Value / Error', fontsize=14)
        plt.title(f'{p_key} across Domain', fontsize=16)
        
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        
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

        if mlflow.active_run():
            traj_id = os.path.basename(output_dir)
            mlflow.log_artifact(filename, artifact_path=f"evaluation_plots/traj_{traj_id}/domain_wise/{p_key}")

def evaluate_rollout(rollout_pth, base_output_dir, plot=True):
    print(f"Rollout Evaluation for {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    
    # --- DYNAMIC JSON FILE NAMING ---
    timestamp_folder = os.path.basename(os.path.dirname(os.path.dirname(rollout_pth)))
    epoch_name = os.path.splitext(os.path.basename(rollout_pth))[0]
    custom_json_filename = f"{timestamp_folder}_{epoch_name}_3_metric.json"
    
    json_file_path = os.path.join(os.path.dirname(base_output_dir), custom_json_filename)

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
        step_loss, gt, pred, traj_metrics = obtain_step_loss(single_trajectory)
        
        step_stress_rmse = torch.mean(torch.tensor(step_loss['stress'])).item()
        step_stress_peak_diff = torch.max(torch.tensor(step_loss['max_stress'])).item()
        step_stress_max_abs = torch.max(torch.tensor(step_loss['max_abs_error_stress'])).item()
        
        step_stress_rel_error = torch.mean(torch.tensor(step_loss['rel_rmse_stress'])).item()
        step_stress_r2 = torch.mean(torch.tensor(step_loss['r2_stress'])).item()

        step_y_rmse = torch.mean(torch.tensor(step_loss['deform_y'])).item()
        step_y_peak_diff = torch.max(torch.tensor(step_loss['max_deform_y'])).item()
        step_y_max_abs = torch.max(torch.tensor(step_loss['max_abs_error_deform_y'])).item()
        
        step_y_rel_error = torch.mean(torch.tensor(step_loss['rel_rmse_deform_y'])).item()
        step_y_r2 = torch.mean(torch.tensor(step_loss['r2_deform_y'])).item()

        average_deformation_error.append(step_y_rmse)
        max_deformation_error.append(step_y_peak_diff)
        
        with open(os.path.join(output_dir, 'Average_deformation_error.pkl'), 'wb') as f:
            pickle.dump(average_deformation_error, f)
        with open(os.path.join(output_dir, 'Max_deformation_error.pkl'), 'wb') as f:
            pickle.dump(max_deformation_error, f)

        if plot:
            plot_step_loss_gif(step_loss, gt, pred, output_dir)

        # 2. Domain Loss (Spatial)
        print("Calculating domain loss...")
        x_ranges, domain_loss = obtain_domain_wise_loss(single_trajectory)
        
        domain_stress_rmse_avg = torch.mean(torch.tensor(domain_loss['stress']['rmse'])).item()
        domain_stress_max_error = torch.max(torch.tensor(domain_loss['stress']['max_rmse'])).item()
        
        domain_deform_rmse_avg = torch.mean(torch.tensor(domain_loss['deform_y']['rmse'])).item()
        domain_deform_max_error = torch.max(torch.tensor(domain_loss['deform_y']['max_rmse'])).item()
        
        if plot:
            plot_domain_wise_loss(x_ranges, domain_loss, output_dir)

        # 3. Compile JSON Data
        losses = {
            'traj_deform_nrmse': traj_metrics['traj_nrmse_deform'],
            'traj_deform_r2': traj_metrics['traj_r2_deform'],
            'traj_stress_nrmse': traj_metrics['traj_nrmse_stress'],
            'traj_stress_r2': traj_metrics['traj_r2_stress'],

            'step_y_deform_nrmse_avg': step_y_rel_error,
            'step_y_deform_r2_avg': step_y_r2,
            'step_y_deform_rmse': step_y_rmse,
            'step_y_deform_peak_diff': step_y_peak_diff,
            'step_y_deform_true_max_error': step_y_max_abs,
            
            'step_stress_nrmse_avg': step_stress_rel_error, 
            'step_stress_r2_avg': step_stress_r2,
            'step_stress_rmse': step_stress_rmse,
            'step_stress_peak_diff': step_stress_peak_diff,
            'step_stress_true_max_error': step_stress_max_abs,
            
            # --- Raw Step-Wise Arrays ---
            #'step_nrmse_deform_y': step_loss['rel_rmse_deform_y'],
            #'step_nrmse_stress': step_loss['rel_rmse_stress'],
            
            'domain_stress_rmse_avg': domain_stress_rmse_avg,
            'domain_stress_max_error': domain_stress_max_error,
            'domain_deform_rmse_avg': domain_deform_rmse_avg,
            'domain_deform_max_error': domain_deform_max_error
        }

        # 4. Save to JSON using the dynamic path
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
            
        print(f"Appended metrics to {custom_json_filename}")
        
        if mlflow.active_run():
            mlflow.log_artifact(json_file_path, artifact_path="evaluation_metrics")
            
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

    return

def main():
    # --- LIST YOUR BASE DIRECTORIES HERE ---
    base_directories = [
       "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_Extrapolation/test/concatenated_rollout_all.pkl",
       "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_OOD_Unseen/test/concatenated_rollout_all.pkl",
        # "/another/path/to/a/different/folder",
        # "/yet/another/path/to/evaluate",
    ]
    
    for base_dir in base_directories:
        if not os.path.exists(base_dir):
            print(f"Skipping: Directory {base_dir} does not exist.")
            continue

        # Automatically finds 'rollout_stage_3.pkl' in the current base_dir
        #rollout_pth = os.path.join(base_dir, 'concatenated_rollout_all.pkl')
        
        # Automatically sets output to 'rollout_evaluation_Stage_3' in the current base_dir
        #output_dir = os.path.join(base_dir, 'rollout_evaluation_plot')
        output_dir = os.path.join(os.path.dirname(base_dir), 'rollout_evaluation_plot')
        
        if os.path.exists(base_dir):
            print(f"\n{'='*30}")
            print(f"PROCESSING: {base_dir}")
            print(f"{'='*30}\n")
            
            # Executing evaluation (plot=False as per your original main)
            evaluate_rollout(base_dir, output_dir, plot=False)
        else:
            print(f"File not found: {base_dir}")

if __name__ == "__main__":
    main()
    