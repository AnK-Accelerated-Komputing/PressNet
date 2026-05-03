import plotly.graph_objects as go
import os
import torch
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import numpy as np
import pickle

def save_rollout_frames(rollout_data, save_directory, i=0, key="stress"):
    skip = 6
    
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

    # --- Pre-calculate Bounds ---
    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]
    i_f, j_f, k_f = ([int(w) for w in i_f], [int(w) for w in j_f], [int(w) for w in k_f])

    gt_y_displacement_all = abs((single_trajectory['gt_pos'] - single_trajectory['mesh_pos']).to('cpu'))
    displacement_min = gt_y_displacement_all.numpy().min(axis=(0, 1))
    displacement_max = gt_y_displacement_all.numpy().max(axis=(0, 1))
    
    gt_y_displacement_min = displacement_min[1]
    gt_y_displacement_max = displacement_max[1]

    # ADDED: Pre-calculate maximum error across all frames for this shape
    all_errors = []
    for frame_idx in range(num_frames):
        p = frame_idx * skip
        pred_xyz_pos = single_trajectory['pred_pos'][p].to('cpu')
        gt_xyz_pos = single_trajectory['gt_pos'][p].to('cpu')
        frame_error = torch.norm(pred_xyz_pos - gt_xyz_pos, dim=1).numpy()
        all_errors.extend(frame_error)
    
    max_error_all_frames = np.max(all_errors) if all_errors else 1e-6
    if max_error_all_frames == 0:
        max_error_all_frames = 1e-6

    for m in range(num_frames):
        p = m * skip
        
        # --- Data Prep ---
        pred_displacement = (single_trajectory['pred_pos'][p].to('cpu') - single_trajectory['mesh_pos'][p].to('cpu'))
        pred_x_disp, pred_y_disp, pred_z_disp = pred_displacement[:, 0].numpy(), pred_displacement[:, 1].numpy(), pred_displacement[:, 2].numpy()
        
        gt_y_displacement = abs((single_trajectory['gt_pos'][p] - single_trajectory['mesh_pos'][p])[:, 1].to('cpu'))
        pred_y_displacement = abs((single_trajectory['pred_pos'][p].to('cpu') - single_trajectory['mesh_pos'][p].to('cpu'))[:, 1].to('cpu'))

        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask = (node_type != 1) 
        mask_2 = (node_type == 1)

        pred_stress = pred_y_displacement.flatten()
        gt_stress = gt_y_displacement.flatten()

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
            intensity=gt_stress, showscale=False, coloraxis="coloraxis", opacity=1, flatshading=True,
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
            intensity=pred_stress, showscale=True, coloraxis="coloraxis", opacity=1, flatshading=True,
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

        # 3. Error (Col 3) - FIXED LEGEND
        # Use pre-calculated maximum error across all frames for consistent legend
        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=current_error, 
            showscale=True, 
            # Blue (0 error) -> Red (Max error across all frames)
            colorscale=[[0, 'blue'], [1, 'red']], 
            cmin=0,
            # FIXED: Uses maximum error across all frames for this shape
            cmax=max_error_all_frames, 
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
                cmin=float(gt_y_displacement_min),
                cmax=float(gt_y_displacement_max),
                colorscale="jet",
                colorbar=dict(title="y_disp", x=0.64)
            )
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_{p}.png")
        frame_fig.write_image(file_save_path)
        print(f"frame ---{m}--- out of {num_frames} done", end="\r")
        
    print(f"All frames for rollout {i} done")
    return "frames saved"

def save_rollout_frames_stress(rollout_data, save_directory, i=0):
    """
    Animation for STRESS error similar to position error.
    Compares GT stress vs Predicted stress and shows error with fixed legend.
    """
    skip = 6
    
    # --- Handle Dimensions and Squeeze ---
    single_trajectory = rollout_data[i].copy() 
    
    if single_trajectory['gt_pos'].dim() == 4:
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

    # --- Pre-calculate Bounds ---
    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]
    i_f, j_f, k_f = ([int(w) for w in i_f], [int(w) for w in j_f], [int(w) for w in k_f])

    # --- Pre-calculate maximum stress error across all frames ---
    # ✅ CORRECTED: Use actual stress values from pickle, not Y-displacement
    all_stress_errors = []
    for frame_idx in range(num_frames):
        p = frame_idx * skip
        
        # Get ground truth and predicted stress directly from pickle
        gt_stress = single_trajectory['gt_stress'][p].cpu().numpy().flatten()
        pred_stress = single_trajectory['pred_stress'][p].cpu().numpy().flatten()
        
        # Compute stress error
        frame_stress_error = np.abs(pred_stress - gt_stress)
        all_stress_errors.extend(frame_stress_error)
    
    max_stress_error_all_frames = np.max(all_stress_errors) if all_stress_errors else 1e-6
    if max_stress_error_all_frames == 0:
        max_stress_error_all_frames = 1e-6

    # --- Pre-calculate bounds for stress values ---
    gt_y_displacement_all = abs((single_trajectory['gt_pos'] - single_trajectory['mesh_pos']).to('cpu'))
    displacement_min = gt_y_displacement_all.numpy().min(axis=(0, 1))
    displacement_max = gt_y_displacement_all.numpy().max(axis=(0, 1))
    
    gt_y_displacement_min = displacement_min[1]
    gt_y_displacement_max = displacement_max[1]

    for m in range(num_frames):
        p = m * skip
        
        # --- Data Prep ---
        # ✅ CORRECTED: Get actual stress values from pickle
        gt_stress_values = single_trajectory['gt_stress'][p].cpu().numpy().flatten()
        pred_stress_values = single_trajectory['pred_stress'][p].cpu().numpy().flatten()
        
        # For mesh coloring, use Y-displacement
        gt_y_displacement = abs((single_trajectory['gt_pos'][p] - single_trajectory['mesh_pos'][p])[:, 1].to('cpu'))
        pred_y_displacement = abs((single_trajectory['pred_pos'][p].to('cpu') - single_trajectory['mesh_pos'][p].to('cpu'))[:, 1].to('cpu'))

        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask = (node_type != 1) 
        mask_2 = (node_type == 1)

        # For visualization, use Y-displacement on meshes
        gt_stress_viz = gt_y_displacement.flatten()
        pred_stress_viz = pred_y_displacement.flatten()
        
        # ✅ CORRECTED: Use actual stress error (not displacement error)
        current_stress_error = gt_stress_values - pred_stress_values
        
        # Positions
        pred_xyz_pos = single_trajectory['pred_pos'][p].to('cpu')
        pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()

        gt_xyz_pos = single_trajectory['gt_pos'][p].to('cpu')
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()
        
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
            subplot_titles=('Ground Truth Stress', 'Predicted Stress', 'Stress Error')
        )

        # 1. Ground Truth Stress (Col 1)
        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x, y=gt_mask_y, z=gt_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=gt_stress_viz, showscale=False, coloraxis="coloraxis", opacity=1, flatshading=True,
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

        # 2. Predicted Stress (Col 2)
        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=pred_stress_viz, showscale=True, coloraxis="coloraxis", opacity=1, flatshading=True,
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

        # 3. Stress Error (Col 3) - FIXED LEGEND
        # Use pre-calculated maximum stress error across all frames for consistent legend
        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=np.abs(current_stress_error),  # ✅ Use actual stress error with absolute values
            showscale=True, 
            # Yellow (0 error) -> Purple (Max error across all frames)
            colorscale=[[0, 'yellow'], [1, 'purple']], 
            cmin=0,
            # FIXED: Uses maximum stress error across all frames for this shape
            cmax=max_stress_error_all_frames, 
            opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="ΔStress", x=1.05) 
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
                cmin=float(gt_y_displacement_min),
                cmax=float(gt_y_displacement_max),
                colorscale="jet",
                colorbar=dict(title="Stress", x=0.64)
            )
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_stress_{p}.png")
        frame_fig.write_image(file_save_path)
        print(f"frame ---{m}--- out of {num_frames} done", end="\r")
        
    print(f"All frames for rollout {i} done")
    return "frames saved"


def generate_gif(save_directory, group_name, fps=5, loop=0):
    """Generate GIF from PNG files in a directory."""
    directory = Path(save_directory)
    png_files = []

    for file in directory.rglob('predicted_step_[0-9]*.png'):
        # Skip stress files
        if 'stress' in file.name:
            continue
        try:
            n = int(file.stem.split('_')[-1])
            png_files.append((file, n))
        except ValueError:
            continue

    png_files.sort(key=lambda x: x[1])
    sorted_png_files = [file_path for file_path, _ in png_files]
    
    gif_outfile = os.path.join(save_directory, f'{group_name}_pos_fps{fps}.gif')
    
    if not sorted_png_files:
        print(f"No position PNGs found in {save_directory}")
        return

    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"Position GIF saved at {gif_outfile}")
    return 'animated gif saved'


def generate_gif_stress(save_directory, group_name, fps=5, loop=0):
    """Generate GIF for stress error animation."""
    directory = Path(save_directory)
    png_files = []

    for file in directory.rglob('predicted_step_stress*.png'):
        try:
            n = int(file.stem.split('_')[-1])
            png_files.append((file, n))
        except ValueError:
            continue

    png_files.sort(key=lambda x: x[1])
    sorted_png_files = [file_path for file_path, _ in png_files]
    
    gif_outfile = os.path.join(save_directory, f'{group_name}_stress_fps{fps}.gif')
    
    if not sorted_png_files:
        print(f"No stress PNGs found in {save_directory}")
        return

    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"Stress GIF saved at {gif_outfile}")
    return 'stress animated gif saved'


def animate_rollout(data_path, save_directory, epoch_number, key="stress"):
    """
    Animate both position and stress errors with reorganized folder structure.
    
    Structure:
    save_directory/
        epoch_{epoch_number}_pos/
            rollout_group_0/
                predicted_step_*.png
                rollout_group_0_pos_fps8.gif
            rollout_group_1/
                ...
        epoch_{epoch_number}_stress/
            rollout_group_0/
                predicted_step_stress_*.png
                rollout_group_0_stress_fps8.gif
            rollout_group_1/
                ...
    """
    with open(data_path, 'rb') as fp:
        rollout_data = pickle.load(fp)
    
    total_groups = len(rollout_data)
    print(f"Found {total_groups} groups in data.")

    # Create main directories for position and stress
    pos_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_pos")
    stress_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_stress")
    
    os.makedirs(pos_main_dir, exist_ok=True)
    os.makedirs(stress_main_dir, exist_ok=True)

    for i in range(total_groups):
        print(f"\n--- Processing Group {i}/{total_groups-1} ---")
        
        group_name = f"rollout_group_{i}"
        
        # Create separate directories for position and stress
        pos_group_dir = os.path.join(pos_main_dir, group_name)
        stress_group_dir = os.path.join(stress_main_dir, group_name)
        
        os.makedirs(pos_group_dir, exist_ok=True)
        os.makedirs(stress_group_dir, exist_ok=True)
        
        # Generate position error animation
        print(f"  Generating position error animation...")
        save_rollout_frames(rollout_data, pos_group_dir, i=i, key=key)
        generate_gif(pos_group_dir, group_name, fps=8)
        
        # Generate stress error animation
        print(f"  Generating stress error animation...")
        save_rollout_frames_stress(rollout_data, stress_group_dir, i=i)
        generate_gif_stress(stress_group_dir, group_name, fps=8)
        
    print(f"\n=== All animations complete ===")
    print(f"Position animations saved in: {pos_main_dir}")
    print(f"Stress animations saved in: {stress_main_dir}")
    
    return 


def main():
    data_path = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real/concatenated_rollout_all.pkl"
    save_directory = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real/animation"
    
    # Extract epoch number from the data path
    epoch_number = "Combined"  # You can also parse this from the filename
    
    os.makedirs(save_directory, exist_ok=True)
    animate_rollout(data_path, save_directory, epoch_number)

    return

if __name__ == '__main__':  
    main()