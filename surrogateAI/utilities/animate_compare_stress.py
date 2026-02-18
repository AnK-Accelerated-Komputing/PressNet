import plotly.graph_objects as go
import os
import torch
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import numpy as np
import pickle

def save_rollout_frames(rollout_data, save_directory, i=0, key="stress"):
    skip = 4
    
    # --- Handle Dimensions and Squeeze ---
    single_trajectory = rollout_data[i].copy() 
    
    if single_trajectory['gt_pos'].dim() == 4: # Assuming (1, Steps, Nodes, 3)
         num_steps = single_trajectory['gt_pos'].squeeze(0).shape[0]
    else:
         num_steps = single_trajectory['gt_pos'].shape[0]

    print("num_steps", num_steps)
    num_frames = 1 * num_steps // skip
    print("num_frames", num_frames)
    print(f"Number of frames: {num_frames}")
    
    for keys in single_trajectory.keys():
        print(keys, single_trajectory[keys].shape)
        if single_trajectory[keys].dim() > 1 and single_trajectory[keys].shape[0] == 1:
            single_trajectory[keys] = single_trajectory[keys].squeeze(0)
        print("                                ", keys, single_trajectory[keys].shape)

    # --- Pre-calculate Bounds (Spatial) ---
    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]
    i_f, j_f, k_f = ([int(w) for w in i_f], [int(w) for w in j_f], [int(w) for w in k_f])

    # --- UPDATED: Pre-calculate Bounds (Value/Stress) ---
    # Instead of calculating displacement, we load the Ground Truth value for the key
    gt_val_all = single_trajectory[f'gt_{key}'].to('cpu')
    
    val_min = gt_val_all.numpy().min()
    val_max = gt_val_all.numpy().max()
    
    # These determine the static colorbar range
    cmin_val = float(val_min)
    cmax_val = float(val_max)

    for m in range(num_frames):
        p = m * skip
        
        # --- Data Prep ---
        # Note: We still calculate displacement just for the error masking if needed, 
        # but we don't use it for the coloring anymore.
        pred_displacement = (single_trajectory['pred_pos'][p].to('cpu') - single_trajectory['mesh_pos'][p].to('cpu'))
        pred_x_disp, pred_y_disp, pred_z_disp = pred_displacement[:, 0].numpy(), pred_displacement[:, 1].numpy(), pred_displacement[:, 2].numpy()
        
        # --- UPDATED: Fetch the Stress/Value Data ---
        # We retrieve the specific key (e.g., 'gt_stress' and 'pred_stress')
        gt_val = single_trajectory[f'gt_{key}'][p].to('cpu')
        pred_val = single_trajectory[f'pred_{key}'][p].to('cpu')

        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask = (node_type != 1) 
        mask_2 = (node_type == 1)

        # Flatten for plotting intensity
        pred_intensity = pred_val.flatten()
        gt_intensity = gt_val.flatten()

        # Positions
        pred_xyz_pos = single_trajectory['pred_pos'][p].to('cpu')
        pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()

        gt_xyz_pos = single_trajectory['gt_pos'][p].to('cpu')
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()

        # --- Error Calculation ---
        # L2 norm of position difference
        current_error = torch.norm(pred_xyz_pos - gt_xyz_pos, dim=1).numpy()
        
        # Masking
        pred_mask_x = np.where(mask, pred_x, np.nan)
        pred_mask_y = np.where(mask, pred_y, np.nan)
        pred_mask_z = np.where(mask, pred_z, np.nan)
        
        pred_mask_x_2 = np.where(mask_2, pred_x, np.nan)
        pred_mask_y_2 = np.where(mask_2, pred_y, np.nan)
        pred_mask_z_2 = np.where(mask_2, pred_z, np.nan)

        gt_mask_x = np.where(mask, gt_x, np.nan)
        gt_mask_y = np.where(mask, gt_y, np.nan)
        gt_mask_z = np.where(mask, gt_z, np.nan)

        gt_mask_x_2 = np.where(mask_2, gt_x, np.nan)
        gt_mask_y_2 = np.where(mask_2, gt_y, np.nan)
        gt_mask_z_2 = np.where(mask_2, gt_z, np.nan)

        # Edges
        pred_edges_x, pred_edges_y, pred_edges_z = [], [], []
        gt_edges_x, gt_edges_y, gt_edges_z = [], [], []
        
        edges = [(i_f[idx], j_f[idx], k_f[idx]) for idx in range(len(i_f))]

        for edge in edges:
            for start, end in [(edge[0], edge[1]), (edge[1], edge[2]), (edge[2], edge[0])]:
                pred_edges_x.extend([pred_x[start], pred_x[end], None])
                pred_edges_y.extend([pred_y[start], pred_y[end], None])
                pred_edges_z.extend([pred_z[start], pred_z[end], None])

        for edge in edges:
            for start, end in [(edge[0], edge[1]), (edge[1], edge[2]), (edge[2], edge[0])]:
                gt_edges_x.extend([gt_x[start], gt_x[end], None])
                gt_edges_y.extend([gt_y[start], gt_y[end], None])
                gt_edges_z.extend([gt_z[start], gt_z[end], None])

        # --- Plotting ---
        frame_fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Ground Truth', 'Prediction', 'Error (Positional)')
        )

        # 1. Ground Truth (Col 1)
        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x, y=gt_mask_y, z=gt_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=gt_intensity, showscale=False, coloraxis="coloraxis", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0)
        ), row=1, col=1)
        
        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x_2, y=gt_mask_y_2, z=gt_mask_z_2, i=i_f, j=j_f, k=k_f,
            showscale=False, color="#C4A484", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
        ), row=1, col=1)

        frame_fig.add_trace(go.Scatter3d(
            x=gt_x, y=gt_y, z=gt_z, mode="markers", marker=dict(size=0.4, color='black'), showlegend=False
        ), row=1, col=1)
        
        frame_fig.add_trace(go.Scatter3d(
            x=gt_edges_x, y=gt_edges_y, z=gt_edges_z, mode='lines', line=dict(color='black', width=0.4), showlegend=False
        ), row=1, col=1)

        # 2. Prediction (Col 2)
        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=pred_intensity, showscale=True, coloraxis="coloraxis", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0)
        ), row=1, col=2)

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x_2, y=pred_mask_y_2, z=pred_mask_z_2, i=i_f, j=j_f, k=k_f,
            showscale=True, color="#C4A484", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
        ), row=1, col=2)

        frame_fig.add_trace(go.Scatter3d(
            x=pred_x, y=pred_y, z=pred_z, mode="markers", marker=dict(size=0.2, color='black'), showlegend=False
        ), row=1, col=2)

        frame_fig.add_trace(go.Scatter3d(
            x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.2), showlegend=False
        ), row=1, col=2)

        # 3. Error (Col 3)
        current_frame_max_error = np.max(current_error)
        if current_frame_max_error == 0: current_frame_max_error = 1e-6 

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=current_error, 
            showscale=True, 
            colorscale=[[0, 'blue'], [1, 'red']], 
            cmin=0,
            cmax=current_frame_max_error, 
            opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Pos Error", x=1.05) 
        ), row=1, col=3)

        frame_fig.add_trace(go.Scatter3d(
            x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.1), showlegend=False
        ), row=1, col=3)

        # Layout
        common_scene = dict(
            camera=dict(eye=dict(x=0.5, y=0.5, z=3.5), up=dict(x=0, y=1, z=0)),
            xaxis=dict(range=[bb_min[0], bb_max[0]]),
            yaxis=dict(range=[bb_min[1], bb_max[1]]),
            zaxis=dict(range=[bb_min[2], bb_max[2]]),
            aspectmode='manual',
            aspectratio=dict(x=x_range / x_range * 2, y=y_range / x_range * 2, z=z_range / x_range * 2)
        )

        frame_fig.update_layout(
            scene=common_scene,
            scene2=common_scene,
            scene3=common_scene,
            width=1800,
            height=800,
            legend=dict(x=0, y=1),
            margin=dict(l=0, r=0, t=30, b=30),
            coloraxis=dict(
                cmin=cmin_val,
                cmax=cmax_val,
                colorscale="jet",
                colorbar=dict(title=key, x=0.64) # Updated title to match the key
            )
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_{p}.png")
        frame_fig.write_image(file_save_path)
        print(f"frame ---{m}--- out of {num_frames} done", end="\r")
        
    print(f"All frames for rollout {i} done")
    return "frames saved"

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
    with open(data_path, 'rb') as fp:
        rollout_data = pickle.load(fp)
    
    total_groups = len(rollout_data)
    print(f"Found {total_groups} groups in data.")

    for i in range(total_groups):
        print(f"--- Processing Group {i+1}/{total_groups} ---")
        group_save_dir = os.path.join(save_directory, f"rollout_group_{i}")
        os.makedirs(group_save_dir, exist_ok=True)
        
        save_rollout_frames(rollout_data, group_save_dir, i=i, key=key)
        generate_gif(group_save_dir, fps=8)
        
    return 

def main():
    data_path = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/Wed-Jan-21-18-49-39-2026/rollout/rollout_epoch_385.pkl"
    save_directory = r"/home/sushil/PressNet/datasetsoutput/transolver/coarse_1500_train_val/Wed-Jan-21-18-49-39-2026/animation/epoch_385_stress"
    os.makedirs(save_directory, exist_ok=True)
    
    # Passing "stress" to ensure it plots stress data
    animate_rollout(data_path, save_directory, key="stress")

    return

if __name__ == '__main__':  
    main()