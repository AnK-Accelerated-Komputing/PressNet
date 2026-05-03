import plotly.graph_objects as go
import os
import torch
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import shutil

def save_rollout_frames(rollout_data, save_directory, i=0, key="stress", skip=10):
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

    # Pre-calculate maximum error across all frames
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

        pred_xyz_pos = single_trajectory['pred_pos'][p].to('cpu')
        pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()

        gt_xyz_pos = single_trajectory['gt_pos'][p].to('cpu')
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()

        current_error = torch.norm(pred_xyz_pos - gt_xyz_pos, dim=1).numpy()
        
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

        frame_fig.add_trace(go.Mesh3d(
            x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f,
            intensity=current_error, 
            showscale=True, 
            colorscale="jet", 
            cmin=0,
            cmax=max_error_all_frames, 
            opacity=1, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Pos Error", x=1.05) 
        ), row=1, col=3)

        frame_fig.add_trace(go.Scatter3d(
            x=pred_edges_x, y=pred_edges_y, z=pred_edges_z, mode='lines', line=dict(color='black', width=0.1), showlegend=False
        ), row=1, col=3)

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
        
    print(f"\nAll frames for rollout {i} generated")
    return "frames saved"

def save_rollout_frames_stress(rollout_data, save_directory, i=0, skip=10):
    single_trajectory = rollout_data[i].copy() 
    
    if single_trajectory['gt_pos'].dim() == 4:
         num_steps = single_trajectory['gt_pos'].squeeze(0).shape[0]
    else:
         num_steps = single_trajectory['gt_pos'].shape[0]

    print("num_steps", num_steps)
    num_frames = 1 * num_steps // skip
    
    for keys in single_trajectory.keys():
        if single_trajectory[keys].dim() > 1 and single_trajectory[keys].shape[0] == 1:
            single_trajectory[keys] = single_trajectory[keys].squeeze(0)

    bb_min = single_trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
    bb_max = single_trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
    x_range, y_range, z_range = bb_max[0] - bb_min[0], bb_max[1] - bb_min[1], bb_max[2] - bb_min[2]

    faces = single_trajectory['faces'][0].to('cpu').numpy()
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]
    i_f, j_f, k_f = ([int(w) for w in i_f], [int(w) for w in j_f], [int(w) for w in k_f])

    all_stress_errors = []
    for frame_idx in range(num_frames):
        p = frame_idx * skip
        gt_stress = single_trajectory['gt_stress'][p].cpu().numpy().flatten()
        pred_stress = single_trajectory['pred_stress'][p].cpu().numpy().flatten()
        frame_stress_error = np.abs(pred_stress - gt_stress)
        all_stress_errors.extend(frame_stress_error)
    
    max_stress_error_all_frames = max(np.max(all_stress_errors), 1e-6)

    all_gt_stress, all_pred_stress = [], []
    for frame_idx in range(num_frames):
        p = frame_idx * skip
        all_gt_stress.extend(single_trajectory['gt_stress'][p].cpu().numpy().flatten())
        all_pred_stress.extend(single_trajectory['pred_stress'][p].cpu().numpy().flatten())
    
    stress_min = min(np.min(all_gt_stress), np.min(all_pred_stress))
    stress_max = max(np.max(all_gt_stress), np.max(all_pred_stress))

    for m in range(num_frames):
        p = m * skip
        gt_stress_values = single_trajectory['gt_stress'][p].cpu().numpy().flatten()
        pred_stress_values = single_trajectory['pred_stress'][p].cpu().numpy().flatten()
        
        gt_y_displacement = abs((single_trajectory['gt_pos'][p] - single_trajectory['mesh_pos'][p])[:, 1].to('cpu'))
        pred_y_displacement = abs((single_trajectory['pred_pos'][p].to('cpu') - single_trajectory['mesh_pos'][p].to('cpu'))[:, 1].to('cpu'))

        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask, mask_2 = (node_type != 1), (node_type == 1)
        gt_stress_viz, pred_stress_viz = gt_y_displacement.flatten(), pred_y_displacement.flatten()
        current_stress_error = gt_stress_values - pred_stress_values
        
        pred_xyz_pos, gt_xyz_pos = single_trajectory['pred_pos'][p].to('cpu'), single_trajectory['gt_pos'][p].to('cpu')
        pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()
        
        pred_mask_x, pred_mask_y, pred_mask_z = np.where(mask, pred_x, np.nan), np.where(mask, pred_y, np.nan), np.where(mask, pred_z, np.nan)
        pred_mask_x_2, pred_mask_y_2, pred_mask_z_2 = np.where(mask_2, pred_x, np.nan), np.where(mask_2, pred_y, np.nan), np.where(mask_2, pred_z, np.nan)
        gt_mask_x, gt_mask_y, gt_mask_z = np.where(mask, gt_x, np.nan), np.where(mask, gt_y, np.nan), np.where(mask, gt_z, np.nan)
        gt_mask_x_2, gt_mask_y_2, gt_mask_z_2 = np.where(mask_2, gt_x, np.nan), np.where(mask_2, gt_y, np.nan), np.where(mask_2, gt_z, np.nan)

        frame_fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}]*3], subplot_titles=('GT Stress', 'Pred Stress', 'Stress Error'))

        frame_fig.add_trace(go.Mesh3d(x=gt_mask_x, y=gt_mask_y, z=gt_mask_z, i=i_f, j=j_f, k=k_f, intensity=gt_stress_viz, coloraxis="coloraxis"), row=1, col=1)
        frame_fig.add_trace(go.Mesh3d(x=gt_mask_x_2, y=gt_mask_y_2, z=gt_mask_z_2, i=i_f, j=j_f, k=k_f, color="#C4A484"), row=1, col=1)
        frame_fig.add_trace(go.Mesh3d(x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f, intensity=pred_stress_viz, coloraxis="coloraxis"), row=1, col=2)
        frame_fig.add_trace(go.Mesh3d(x=pred_mask_x_2, y=pred_mask_y_2, z=pred_mask_z_2, i=i_f, j=j_f, k=k_f, color="#C4A484"), row=1, col=2)
        frame_fig.add_trace(go.Mesh3d(x=pred_mask_x, y=pred_mask_y, z=pred_mask_z, i=i_f, j=j_f, k=k_f, intensity=np.abs(current_stress_error), colorscale="jet", cmin=0, cmax=max_stress_error_all_frames, colorbar=dict(title="ΔStress", x=1.05)), row=1, col=3)

        common_scene = dict(camera=dict(eye=dict(x=0.5, y=0.5, z=3.5), up=dict(x=0, y=1, z=0)), xaxis=dict(range=[bb_min[0], bb_max[0]]), yaxis=dict(range=[bb_min[1], bb_max[1]]), zaxis=dict(range=[bb_min[2], bb_max[2]]), aspectmode='manual', aspectratio=dict(x=2, y=y_range/x_range*2, z=z_range/x_range*2))
        frame_fig.update_layout(scene=common_scene, scene2=common_scene, scene3=common_scene, width=1800, height=800, margin=dict(l=0, r=0, t=30, b=30), coloraxis=dict(cmin=float(stress_min), cmax=float(stress_max), colorscale="jet", colorbar=dict(title="Stress", x=0.64)))
        frame_fig.write_image(os.path.join(save_directory, f"predicted_step_stress_{p}.png"))
        print(f"frame ---{m}--- out of {num_frames} done", end="\r")
    return "frames saved"

def generate_gif_to_directory(frames_directory, output_directory, group_name, fps=5):
    directory = Path(frames_directory)
    png_files = sorted([f for f in directory.glob('predicted_step_[0-9]*.png') if 'stress' not in f.name], key=lambda x: int(x.stem.split('_')[-1]))
    if not png_files: return
    imgs = [Image.open(f) for f in png_files]
    imgs[0].save(fp=os.path.join(output_directory, f'{group_name}_pos_fps{fps}.gif'), format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=0)

def generate_gif_stress_to_directory(frames_directory, output_directory, group_name, fps=5):
    directory = Path(frames_directory)
    png_files = sorted([f for f in directory.glob('predicted_step_stress*.png')], key=lambda x: int(x.stem.split('_')[-1]))
    if not png_files: return
    imgs = [Image.open(f) for f in png_files]
    imgs[0].save(fp=os.path.join(output_directory, f'{group_name}_stress_fps{fps}.gif'), format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=0)

def animate_rollout(data_path, save_directory, epoch_number, skip=10):
    with open(data_path, 'rb') as fp: rollout_data = pickle.load(fp)
    pos_main_dir, stress_main_dir = os.path.join(save_directory, f"epoch_{epoch_number}_pos"), os.path.join(save_directory, f"epoch_{epoch_number}_stress")
    os.makedirs(pos_main_dir, exist_ok=True); os.makedirs(stress_main_dir, exist_ok=True)

    for i in range(len(rollout_data)):
        print(f"--- Epoch {epoch_number} | Animating Group {i} ---")
        temp_dir = os.path.join(save_directory, f"temp_group_{i}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            save_rollout_frames(rollout_data, temp_dir, i=i, skip=skip)
            generate_gif_to_directory(temp_dir, pos_main_dir, f"rollout_group_{i}", fps=8)
            save_rollout_frames_stress(rollout_data, temp_dir, i=i, skip=skip)
            generate_gif_stress_to_directory(temp_dir, stress_main_dir, f"rollout_group_{i}", fps=8)
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    return True

if __name__ == '__main__':
    data_path = r"/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/regDGCNN_seg/coarse_1500_train_val/DGCNN_C_ST1_K80(Retrain)/rollout/rollout_epoch_940.pkl"
    save_directory = r"/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/regDGCNN_seg/coarse_1500_train_val/DGCNN_C_ST1_K80(Retrain)/animation"
    epoch_number = "840" 
    os.makedirs(save_directory, exist_ok=True)
    animate_rollout(data_path, save_directory, epoch_number, skip=4)