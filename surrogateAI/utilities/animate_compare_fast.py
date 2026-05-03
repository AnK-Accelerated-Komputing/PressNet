import plotly.graph_objects as go
import os
import torch
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import shutil
import concurrent.futures

def save_rollout_frames(rollout_data, save_directory, i=0, key="stress"):
    skip = 10
    
    # --- Handle Dimensions and Squeeze ---
    single_trajectory = rollout_data[i].copy() 
    
    if single_trajectory['gt_pos'].dim() == 4: # Assuming (1, Steps, Nodes, 3)
         num_steps = single_trajectory['gt_pos'].squeeze(0).shape[0]
    else:
         num_steps = single_trajectory['gt_pos'].shape[0]

    num_frames = num_steps // skip
    print(f"Total steps: {num_steps} | Frames to generate: {num_frames}")
    
    for keys in single_trajectory.keys():
        if single_trajectory[keys].dim() > 1 and single_trajectory[keys].shape[0] == 1:
            single_trajectory[keys] = single_trajectory[keys].squeeze(0)

    # --- Pre-calculate Bounds ---
    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range, y_range, z_range = bb_max - bb_min

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0].astype(int), faces[:, 1].astype(int), faces[:, 2].astype(int)

    gt_y_displacement_all = abs((single_trajectory['gt_pos'] - single_trajectory['mesh_pos']).to('cpu'))
    displacement_min = gt_y_displacement_all.numpy().min(axis=(0, 1))
    displacement_max = gt_y_displacement_all.numpy().max(axis=(0, 1))
    
    gt_y_displacement_min = displacement_min[1]
    gt_y_displacement_max = displacement_max[1]

    # OPTIMIZATION: Vectorized Max Error Calculation (No Loop)
    sampled_pred = single_trajectory['pred_pos'][0:num_frames*skip:skip].to('cpu')
    sampled_gt = single_trajectory['gt_pos'][0:num_frames*skip:skip].to('cpu')
    all_errors = torch.norm(sampled_pred - sampled_gt, dim=2).numpy()
    max_error_all_frames = np.max(all_errors) if all_errors.size > 0 else 1e-6
    if max_error_all_frames == 0:
        max_error_all_frames = 1e-6

    # OPTIMIZATION: Pre-calculate Edge Indices Once
    starts = np.concatenate([i_f, j_f, k_f])
    ends = np.concatenate([j_f, k_f, i_f])
    num_edges = len(starts)

    def build_edges(pos_array):
        arr = np.empty(num_edges * 3)
        arr[0::3] = pos_array[starts]
        arr[1::3] = pos_array[ends]
        arr[2::3] = np.nan # Plotly uses nan to break lines, faster than None
        return arr

    # Setup Thread Pool for parallel saving
    futures = []
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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

        pred_stress = pred_y_displacement.flatten().numpy()
        gt_stress = gt_y_displacement.flatten().numpy()

        pred_xyz_pos = single_trajectory['pred_pos'][p].to('cpu')
        pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()

        gt_xyz_pos = single_trajectory['gt_pos'][p].to('cpu')
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()

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

        # OPTIMIZATION: Instant Edge Generation
        pred_edges_x = build_edges(pred_x)
        pred_edges_y = build_edges(pred_y)
        pred_edges_z = build_edges(pred_z)
        
        gt_edges_x = build_edges(gt_x)
        gt_edges_y = build_edges(gt_y)
        gt_edges_z = build_edges(gt_z)

        # --- Plotting ---
        frame_fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Ground Truth', 'Prediction', 'Error (Positional)')
        )

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

        frame_fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode="markers", marker=dict(size=0.4, color='black'), showlegend=False), row=1, col=1)
        frame_fig.add_trace(go.Scatter3d(x=gt_edges_x, y=gt_edges_y, z=gt_edges_z, mode='lines', line=dict(color='black', width=0.4), showlegend=False), row=1, col=1)

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

        frame_fig.add_trace(go.Scatter3d(x=pred_x, y=pred_y, z=pred_z, mode="markers", marker=dict(size=0.2, color='black'), showlegend=False), row=1, col=2)
        frame_fig.add_trace(go.Scatter3d(x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.2), showlegend=False), row=1, col=2)

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=current_error, showscale=True, colorscale="jet", cmin=0, cmax=max_error_all_frames, opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Pos Error", x=1.05) 
        ), row=1, col=3)

        frame_fig.add_trace(go.Scatter3d(x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.1), showlegend=False), row=1, col=3)

        common_scene = dict(
            camera=dict(eye=dict(x=0.5, y=0.5, z=3.5), up=dict(x=0, y=1, z=0)),
            xaxis=dict(range=[bb_min[0], bb_max[0]]),
            yaxis=dict(range=[bb_min[1], bb_max[1]]),
            zaxis=dict(range=[bb_min[2], bb_max[2]]),
            aspectmode='manual',
            aspectratio=dict(x=x_range / x_range * 2, y=y_range / x_range * 2, z=z_range / x_range * 2)
        )

        frame_fig.update_layout(
            scene=common_scene, scene2=common_scene, scene3=common_scene,
            width=1800, height=800, legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=30, b=30),
            coloraxis=dict(cmin=float(gt_y_displacement_min), cmax=float(gt_y_displacement_max), colorscale="jet", colorbar=dict(title="y_disp", x=0.64))
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_{p}.png")
        # OPTIMIZATION: Submit to background thread
        futures.append(executor.submit(frame_fig.write_image, file_save_path))
        print(f"Dispatched position frame {m+1}/{num_frames} for rendering...", end="\r")

    print("\nWaiting for all position frames to save to disk...")
    concurrent.futures.wait(futures)
    executor.shutdown()
        
    print(f"All position frames for rollout {i} successfully saved.")
    return "frames saved"

def save_rollout_frames_stress(rollout_data, save_directory, i=0):
    skip = 10
    
    single_trajectory = rollout_data[i].copy() 
    
    if single_trajectory['gt_pos'].dim() == 4:
         num_steps = single_trajectory['gt_pos'].squeeze(0).shape[0]
    else:
         num_steps = single_trajectory['gt_pos'].shape[0]

    num_frames = num_steps // skip
    
    for keys in single_trajectory.keys():
        if single_trajectory[keys].dim() > 1 and single_trajectory[keys].shape[0] == 1:
            single_trajectory[keys] = single_trajectory[keys].squeeze(0)

    # --- Pre-calculate Bounds ---
    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range, y_range, z_range = bb_max - bb_min

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0].astype(int), faces[:, 1].astype(int), faces[:, 2].astype(int)

    # OPTIMIZATION: Vectorized Stress Error Calculation
    sampled_gt_stress = single_trajectory['gt_stress'][0:num_frames*skip:skip].cpu().numpy()
    sampled_pred_stress = single_trajectory['pred_stress'][0:num_frames*skip:skip].cpu().numpy()
    
    all_stress_errors = np.abs(sampled_pred_stress - sampled_gt_stress)
    max_stress_error_all_frames = np.max(all_stress_errors) if all_stress_errors.size > 0 else 1e-6
    if max_stress_error_all_frames == 0:
        max_stress_error_all_frames = 1e-6

    # FIX FOR PAPER: Force the colorbar minimum to 0 for physical realism
    stress_min = 0.0
    stress_max = max(np.max(sampled_gt_stress), np.max(sampled_pred_stress))

    # OPTIMIZATION: Pre-calculate Edge Indices Once
    starts = np.concatenate([i_f, j_f, k_f])
    ends = np.concatenate([j_f, k_f, i_f])
    num_edges = len(starts)

    def build_edges(pos_array):
        arr = np.empty(num_edges * 3)
        arr[0::3] = pos_array[starts]
        arr[1::3] = pos_array[ends]
        arr[2::3] = np.nan
        return arr

    futures = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4) 

    for m in range(num_frames):
        p = m * skip
        
        # --- Data Prep ---
        # RAW values kept intact for accurate scientific error reporting
        gt_stress_values = single_trajectory['gt_stress'][p].cpu().numpy().flatten()
        pred_stress_values = single_trajectory['pred_stress'][p].cpu().numpy().flatten()

        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask = (node_type != 1) 
        mask_2 = (node_type == 1)

        # PAPER BEST PRACTICE: Clip the arrays strictly for plotting the visual color
        gt_stress_viz = np.clip(gt_stress_values, a_min=0.0, a_max=None)
        pred_stress_viz = np.clip(pred_stress_values, a_min=0.0, a_max=None)
        
        # Calculate true error using the unclipped raw predictions to honestly reflect AI noise
        current_stress_error = np.abs(gt_stress_values - pred_stress_values)
        
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

        # OPTIMIZATION: Instant Edge Generation
        pred_edges_x = build_edges(pred_x)
        pred_edges_y = build_edges(pred_y)
        pred_edges_z = build_edges(pred_z)
        
        gt_edges_x = build_edges(gt_x)
        gt_edges_y = build_edges(gt_y)
        gt_edges_z = build_edges(gt_z)

        # --- Plotting ---
        frame_fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Ground Truth Stress', 'Predicted Stress', 'Stress Absolute Error')
        )

        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x, y=gt_mask_y, z=gt_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=gt_stress_viz, showscale=False, coloraxis="coloraxis", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0), lightposition=dict(x=0, y=0, z=0)
        ), row=1, col=1)
        
        frame_fig.add_trace(go.Mesh3d(
            x=gt_mask_x_2, y=gt_mask_y_2, z=gt_mask_z_2, i=i_f, j=j_f, k=k_f,
            showscale=False, color="#C4A484", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
        ), row=1, col=1)

        frame_fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode="markers", marker=dict(size=0.4, color='black'), showlegend=False), row=1, col=1)
        frame_fig.add_trace(go.Scatter3d(x=gt_edges_x, y=gt_edges_y, z=gt_edges_z, mode='lines', line=dict(color='black', width=0.4), showlegend=False), row=1, col=1)

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=pred_stress_viz, showscale=True, coloraxis="coloraxis", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0), lightposition=dict(x=0, y=0, z=0)
        ), row=1, col=2)

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x_2, y=pred_mask_y_2, z=pred_mask_z_2, i=i_f, j=j_f, k=k_f,
            showscale=True, color="#C4A484", opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
        ), row=1, col=2)

        frame_fig.add_trace(go.Scatter3d(x=pred_x, y=pred_y, z=pred_z, mode="markers", marker=dict(size=0.2, color='black'), showlegend=False), row=1, col=2)
        frame_fig.add_trace(go.Scatter3d(x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.2), showlegend=False), row=1, col=2)

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=current_stress_error, showscale=True, colorscale="jet", cmin=0, cmax=max_stress_error_all_frames, opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0), colorbar=dict(title="Error (Abs)", x=1.05) 
        ), row=1, col=3)

        frame_fig.add_trace(go.Scatter3d(x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.1), showlegend=False), row=1, col=3)

        common_scene = dict(
            camera=dict(eye=dict(x=0.5, y=0.5, z=3.5), up=dict(x=0, y=1, z=0)),
            xaxis=dict(range=[bb_min[0], bb_max[0]]),
            yaxis=dict(range=[bb_min[1], bb_max[1]]),
            zaxis=dict(range=[bb_min[2], bb_max[2]]),
            aspectmode='manual', aspectratio=dict(x=x_range / x_range * 2, y=y_range / x_range * 2, z=z_range / x_range * 2)
        )

        frame_fig.update_layout(
            scene=common_scene, scene2=common_scene, scene3=common_scene,
            width=1800, height=800, legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=30, b=30),
            coloraxis=dict(cmin=float(stress_min), cmax=float(stress_max), colorscale="jet", colorbar=dict(title="Stress", x=0.64))
        )

        file_save_path = os.path.join(save_directory, f"predicted_step_stress_{p}.png")
        futures.append(executor.submit(frame_fig.write_image, file_save_path))
        print(f"Dispatched stress frame {m+1}/{num_frames} for rendering...", end="\r")
        
    print("\nWaiting for all stress frames to save to disk...")
    concurrent.futures.wait(futures)
    executor.shutdown()
        
    print(f"All stress frames for rollout {i} successfully saved.")
    return "frames saved"

def generate_gif_to_directory(frames_directory, output_directory, group_name, fps=5, loop=0):
    directory = Path(frames_directory)
    png_files = []

    for file in directory.glob('predicted_step_[0-9]*.png'):
        if 'stress' in file.name:
            continue
        try:
            n = int(file.stem.split('_')[-1])
            png_files.append((file, n))
        except ValueError:
            continue

    png_files.sort(key=lambda x: x[1])
    sorted_png_files = [file_path for file_path, _ in png_files]
    
    gif_outfile = os.path.join(output_directory, f'{group_name}_pos_fps{fps}.gif')
    
    if not sorted_png_files:
        print(f"No position PNGs found in {frames_directory}")
        return

    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"  Position GIF saved: {group_name}_pos_fps{fps}.gif")
    return 'animated gif saved'

def generate_gif_stress_to_directory(frames_directory, output_directory, group_name, fps=5, loop=0):
    directory = Path(frames_directory)
    png_files = []

    for file in directory.glob('predicted_step_stress*.png'):
        try:
            n = int(file.stem.split('_')[-1])
            png_files.append((file, n))
        except ValueError:
            continue

    png_files.sort(key=lambda x: x[1])
    sorted_png_files = [file_path for file_path, _ in png_files]
    
    gif_outfile = os.path.join(output_directory, f'{group_name}_stress_fps{fps}.gif')
    
    if not sorted_png_files:
        print(f"No stress PNGs found in {frames_directory}")
        return

    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"  Stress GIF saved: {group_name}_stress_fps{fps}.gif")
    return 'stress animated gif saved'

def animate_rollout(data_path, save_directory, epoch_number, key="stress"):
    # FIX FOR CRASH: Load standard pickle file since it was saved with pickle.dump
    #print("Loading pickle file...")
    rollout_data = torch.load(data_path, map_location='cpu', weights_only=False)
    #with open(data_path, 'rb') as f:
        #rollout_data = pickle.load(f)
    
    total_groups = len(rollout_data)
    print(f"Found {total_groups} groups in data.")

    pos_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_pos")
    stress_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_stress")
    frames_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_frames")
    
    os.makedirs(pos_main_dir, exist_ok=True)
    os.makedirs(stress_main_dir, exist_ok=True)
    os.makedirs(frames_main_dir, exist_ok=True)

    for i in range(total_groups):
        print(f"\n{'='*60}")
        print(f"Processing Group {i}/{total_groups-1}")
        print(f"{'='*60}")
        
        group_name = f"rollout_group_{i}"
        
        group_frames_dir = os.path.join(frames_main_dir, f"group_{i}")
        os.makedirs(group_frames_dir, exist_ok=True)
        
        # === POSITION ERROR ANIMATION ===
        print(f"\n[POSITION] Generating frames...")
        save_rollout_frames(rollout_data, group_frames_dir, i=i, key=key)
        
        print(f"[POSITION] Creating GIF...")
        generate_gif_to_directory(group_frames_dir, pos_main_dir, group_name, fps=8)
        
        # === STRESS ERROR ANIMATION ===
        print(f"\n[STRESS] Generating frames...")
        save_rollout_frames_stress(rollout_data, group_frames_dir, i=i)
        
        print(f"[STRESS] Creating GIF...")
        generate_gif_stress_to_directory(group_frames_dir, stress_main_dir, group_name, fps=8)
        
        print(f"\n[FRAMES SAVED] Frames for group {i} kept in: {group_frames_dir}")
        
    print(f"\n{'='*60}")
    print(f"✓ All animations complete!")
    print(f"{'='*60}")
    print(f"Position GIFs: {pos_main_dir}")
    print(f"Stress GIFs:   {stress_main_dir}")
    print(f"Saved Frames:  {frames_main_dir}")
    print(f"{'='*60}\n")
    
    return

def main():
    data_path = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_Extrapolation/test/concatenated_rollout_all.pkl"
    save_directory = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/1_Inference_new/Dilated_Inference_Extrapolation/test/animation"
    epoch_number = "combined" 
    os.makedirs(save_directory, exist_ok=True)
    animate_rollout(data_path, save_directory, epoch_number)

    return

if __name__ == '__main__':  
    main()