import plotly.graph_objects as go
import os
import torch
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import pandas as pd

def save_rollout_frames(rollout_data, save_directory, i=0, key="stress"):
    skip = 4
    
    # --- 1. Load and Preprocess Trajectory ---
    single_trajectory = rollout_data[i].copy() 
    
    # Robust dimension handling
    for k in single_trajectory.keys():
        if isinstance(single_trajectory[k], torch.Tensor):
             if single_trajectory[k].dim() > 1 and single_trajectory[k].shape[0] == 1:
                single_trajectory[k] = single_trajectory[k].squeeze(0)
    
    num_steps = single_trajectory['gt_pos'].shape[0]
    num_frames = num_steps // skip
    print(f"Shape {i}: Processing {num_frames} frames (Steps: {num_steps}, Skip: {skip})")

    # --- 2. Pre-calculate Global Max Error (For Fixed Legend) ---
    all_gt = single_trajectory['gt_pos'].to('cpu')
    all_pred = single_trajectory['pred_pos'].to('cpu')
    all_mesh_pos = single_trajectory['mesh_pos'].to('cpu')
    
    # Calculate Spatial Error (Euclidean Distance)
    # Shape: (Time, Nodes)
    all_pos_errors = torch.norm(all_pred - all_gt, dim=2)
    
    # Get the absolute maximum error to fix the legend scale
    global_max_error = all_pos_errors.max().item()
    global_mean_error = all_pos_errors.mean().item()
    
    gt_y_disps = torch.abs((all_gt - all_mesh_pos)[:, :, 1])
    gt_y_disp_min = gt_y_disps.min().item()
    gt_y_disp_max = gt_y_disps.max().item()

    print(f"  -> Global Max Error: {global_max_error:.6f} m")

    # --- 3. Setup Static Plotting Data ---
    bb_min = all_gt.numpy().min(axis=(0, 1))
    bb_max = all_gt.numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    # [FIX] Ensure faces are Integers
    faces = single_trajectory['faces'][0].to('cpu').numpy().astype(int)
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # [OPTIMIZATION] Pre-calculate edge connectivity indices ONCE
    # Create an array of edge pairs: (N_edges, 2)
    edge_nodes = np.vstack([
        faces[:, [0, 1]], 
        faces[:, [1, 2]], 
        faces[:, [2, 0]]
    ])

    # Helper for vectorized wireframe generation
    def get_wireframe_coords(coords, edges_idx):
        """
        Input: coords (N_nodes,), edges_idx (N_edges, 2)
        Output: (3 * N_edges,) array -> [x0, x1, NaN, x2, x3, NaN, ...]
        """
        # Create an array (N_edges, 3) filled with NaNs
        lines = np.full((edges_idx.shape[0], 3), np.nan)
        lines[:, 0] = coords[edges_idx[:, 0]]
        lines[:, 1] = coords[edges_idx[:, 1]]
        return lines.flatten()

    # --- 4. Frame Generation Loop ---
    for m in range(num_frames):
        p = m * skip
        
        # Get data for current frame
        pred_xyz = all_pred[p].numpy()
        gt_xyz = all_gt[p].numpy()
        
        pred_x, pred_y, pred_z = pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2]
        gt_x, gt_y, gt_z = gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2]
        
        # Coloring Fields
        pred_y_disp = torch.abs((all_pred[p] - all_mesh_pos[p])[:, 1]).numpy()
        gt_y_disp = torch.abs((all_gt[p] - all_mesh_pos[p])[:, 1]).numpy()
        current_error = all_pos_errors[p].numpy()
        
        # Masking
        node_type = single_trajectory['node_type'][p].to('cpu').flatten().numpy()
        mask_plate = (node_type != 1)
        mask_die = (node_type == 1)

        pred_mask_x = np.where(mask_plate, pred_x, np.nan)
        pred_mask_y = np.where(mask_plate, pred_y, np.nan)
        pred_mask_z = np.where(mask_plate, pred_z, np.nan)
        
        pred_die_x = np.where(mask_die, pred_x, np.nan)
        pred_die_y = np.where(mask_die, pred_y, np.nan)
        pred_die_z = np.where(mask_die, pred_z, np.nan)

        gt_mask_x = np.where(mask_plate, gt_x, np.nan)
        gt_mask_y = np.where(mask_plate, gt_y, np.nan)
        gt_mask_z = np.where(mask_plate, gt_z, np.nan)
        
        gt_die_x = np.where(mask_die, gt_x, np.nan)
        gt_die_y = np.where(mask_die, gt_y, np.nan)
        gt_die_z = np.where(mask_die, gt_z, np.nan)

        # [OPTIMIZATION] Generate Edges Vectorized (Instant)
        pred_edges_x = get_wireframe_coords(pred_x, edge_nodes)
        pred_edges_y = get_wireframe_coords(pred_y, edge_nodes)
        pred_edges_z = get_wireframe_coords(pred_z, edge_nodes)
        
        gt_edges_x = get_wireframe_coords(gt_x, edge_nodes)
        gt_edges_y = get_wireframe_coords(gt_y, edge_nodes)
        gt_edges_z = get_wireframe_coords(gt_z, edge_nodes)

        # --- Plotting ---
        frame_fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Ground Truth', 'Prediction', 'Error') # Title simplified
        )

        # Col 1: Ground Truth
        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x, y=gt_mask_y, z=gt_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=gt_y_disp, showscale=False, opacity=1, flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1),
            colorscale='jet', cmin=gt_y_disp_min, cmax=gt_y_disp_max
        ), row=1, col=1)
        frame_fig.add_trace(go.Mesh3d(
            x=gt_die_x, y=gt_die_y, z=gt_die_z, i=i_f, j=j_f, k=k_f,
            color="#C4A484", opacity=1, flatshading=True, showscale=False,
            lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1)
        ), row=1, col=1)
        frame_fig.add_trace(go.Scatter3d(
            x=gt_edges_x, y=gt_edges_y, z=gt_edges_z, 
            mode='lines', line=dict(color='black', width=0.4), showlegend=False
        ), row=1, col=1)

        # Col 2: Prediction
        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=pred_y_disp, showscale=True, opacity=1, flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1),
            colorscale='jet', cmin=gt_y_disp_min, cmax=gt_y_disp_max,
            colorbar=dict(title="Y Disp", x=0.64)
        ), row=1, col=2)
        frame_fig.add_trace(go.Mesh3d(
            x=pred_die_x, y=pred_die_y, z=pred_die_z, i=i_f, j=j_f, k=k_f,
            color="#C4A484", opacity=1, flatshading=True, showscale=False,
            lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1)
        ), row=1, col=2)
        frame_fig.add_trace(go.Scatter3d(
            x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, 
            mode='lines', line=dict(color='black', width=0.4), showlegend=False
        ), row=1, col=2)

        # Col 3: Error (FIXED SCALE)
        frame_fig.add_trace(go.Mesh3d(
            x=pred_x, y=pred_y, z=pred_z, i=i_f, j=j_f, k=k_f,
            intensity=current_error, 
            showscale=True, 
            colorscale=[[0, 'blue'], [1, 'red']], 
            cmin=0, 
            cmax=global_max_error, # Fixed scale
            opacity=1, flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1),
            colorbar=dict(title="Pos Error", x=1.00)
        ), row=1, col=3)
        frame_fig.add_trace(go.Scatter3d(
            x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, 
            mode='lines', line=dict(color='black', width=0.1), showlegend=False
        ), row=1, col=3)

        # Layout
        common_scene = dict(
            camera=dict(eye=dict(x=0.5, y=0.5, z=3.5), up=dict(x=0, y=1, z=0)),
            xaxis=dict(range=[bb_min[0], bb_max[0]]),
            yaxis=dict(range=[bb_min[1], bb_max[1]]),
            zaxis=dict(range=[bb_min[2], bb_max[2]]),
            aspectmode='manual',
            aspectratio=dict(x=x_range/x_range*2, y=y_range/x_range*2, z=z_range/x_range*2)
        )

        frame_fig.update_layout(
            scene=common_scene,
            scene2=common_scene,
            scene3=common_scene,
            width=1800,
            height=600,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_{p}.png")
        frame_fig.write_image(file_save_path)
        print(f"Frame {m+1}/{num_frames} done", end="\r")

    print(f"\nShape {i} Animation Frames Done.")
    
    return {
        "shape_id": i,
        "max_error": global_max_error,
        "mean_error": global_mean_error
    }

def generate_gif(save_directory, fps=5, loop=0):
    directory = Path(save_directory)
    png_files = []

    for file in directory.rglob('predicted_step*.png'):
        try:
            n = int(file.stem.split('_')[-1])
            png_files.append((file, n))
        except ValueError:
            continue

    png_files.sort(key=lambda x: x[1])
    sorted_png_files = [file_path for file_path, _ in png_files]
    
    basename = os.path.basename(save_directory)
    gif_outfile = os.path.join(save_directory, f'rollout_{basename}_fps{fps}.gif')
    
    if not sorted_png_files:
        print(f"No PNGs found in {save_directory}")
        return

    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"GIF saved at {gif_outfile}")
    return 'animated gif saved'
      
def animate_rollout(data_path, save_directory, key="stress"):
    print(f"Loading rollout data from {data_path}...")
    with open(data_path, 'rb') as fp:
        rollout_data = pickle.load(fp)
    
    total_groups = len(rollout_data)
    print(f"Found {total_groups} groups in data.")

    summary_metrics = []

    for i in range(total_groups):
        print(f"--- Processing Group {i+1}/{total_groups} ---")
        group_save_dir = os.path.join(save_directory, f"rollout_group_{i}")
        os.makedirs(group_save_dir, exist_ok=True)
        
        metrics = save_rollout_frames(rollout_data, group_save_dir, i=i, key=key)
        summary_metrics.append(metrics)
        
        generate_gif(group_save_dir, fps=8)
    
    # Generate Summary Table and JSON
    if summary_metrics:
        df = pd.DataFrame(summary_metrics)
        df.set_index("shape_id", inplace=True)
        
        print("\n" + "="*60)
        print(" SUMMARY OF MAXIMUM ERRORS PER SHAPE")
        print("="*60)
        print(df.to_string(float_format="{:.6f}".format))
        print("="*60)

        json_path = os.path.join(save_directory, "max_error_summary.json")
        csv_path = os.path.join(save_directory, "max_error_summary.csv")
        
        df.to_json(json_path, orient="index", indent=4)
        df.to_csv(csv_path)
        
        print(f"Summary metrics saved to:\n {json_path}\n {csv_path}")
        
    return 

def main():
    data_path = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/Fri-Jan-23-02-31-42-2026/rollout/rollout_epoch_97.pkl"
    save_directory = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/Fri-Jan-23-02-31-42-2026/animation/epoch_97_1"
    os.makedirs(save_directory, exist_ok=True)
    animate_rollout(data_path, save_directory)

    return

if __name__ == '__main__':  
    main()