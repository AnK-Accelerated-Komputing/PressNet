import os
import meshio
import torch
import numpy as np
import pickle


def save_time_series_vtu(gt_pos, pred_pos, connectivity, gt_deform, pred_deform, node_type, error, output_dir):
    """
    Save time series deformation data to VTU files and create PVD for ParaView.

    Vector fields:
        - gt_position (ground truth mesh positions)
        - pred_position (predicted mesh positions)
        - gt_deformation
        - pred_deformation
        - error
    """

    # Convert tensors to numpy
    gt_pos_np = gt_pos.detach().cpu().numpy() if torch.is_tensor(gt_pos) else np.array(gt_pos)
    pred_pos_np = pred_pos.detach().cpu().numpy() if torch.is_tensor(pred_pos) else np.array(pred_pos)
    connectivity_np = connectivity.detach().cpu().numpy() if torch.is_tensor(connectivity) else np.array(connectivity)
    gt_deform_np = gt_deform.detach().cpu().numpy() if torch.is_tensor(gt_deform) else np.array(gt_deform)
    pred_deform_np = pred_deform.detach().cpu().numpy() if torch.is_tensor(pred_deform) else np.array(pred_deform)
    error_np = error.detach().cpu().numpy() if torch.is_tensor(error) else np.array(error)
    node_type_np = node_type.detach().cpu().numpy() if torch.is_tensor(node_type) else np.array(node_type)


    connectivity_np = connectivity_np.astype("int32")

    # ---- Create static mask for nodes that are type 1 in any timestep ----
    # node_type_np shape: (T, N, 1)
    static_mask = (node_type_np == 1).any(axis=0, keepdims=True)  # shape (1, N, 1)
    # static_mask = ((node_type_np == 1) | (node_type_np == 3)).any(axis=0, keepdims=True)  # shape (1, N, 1)

    mask = np.repeat(static_mask, gt_deform_np.shape[0], axis=0)  # shape (T, N, 1)

    # -------- Apply mask to zero deformation and error ---------
    gt_deform_np = np.where(mask, 0.0, gt_deform_np)
    pred_deform_np = np.where(mask, 0.0, pred_deform_np)
    error_np = np.where(mask, 0.0, error_np)

    os.makedirs(output_dir, exist_ok=True)

    num_timesteps = gt_deform_np.shape[0]

    for t in range(num_timesteps):
        mesh = meshio.Mesh(
            points=gt_pos_np[t],
            cells=[("tetra", connectivity_np)],
            point_data={
                "gt_position": gt_pos_np[t],
                "pred_position": pred_pos_np[t],
                "gt_deformation": gt_deform_np[t],   # full vector
                "pred_deformation": pred_deform_np[t],
                "error": error_np[t]
            }
        )

        vtu_filename = f"pressnet_{t:04d}.vtu"
        vtu_path = os.path.join(output_dir, vtu_filename)
        mesh.write(vtu_path)

    # Write PVD file for time series
    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>'
    ]

    for t in range(num_timesteps):
        vtu_filename = f"pressnet_{t:04d}.vtu"
        pvd_lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{vtu_filename}"/>')
    pvd_lines.append('  </Collection>')
    pvd_lines.append('</VTKFile>')

    pvd_path = os.path.join(output_dir, "pressnet.pvd")
    with open(pvd_path, "w") as f:
        f.write("\n".join(pvd_lines))

    print(f"Saved {num_timesteps} VTU files and PVD file at {output_dir}")


def process_samples(data_path, base_output_dir):
    """
    Process rollout data and export VTU/PVD files for each sample.

    Parameters
    ----------
    data_path : str
        Path to the pickle file containing rollout data.
    base_output_dir : str
        Base directory to store output VTU/PVD files.
    """

    os.makedirs(base_output_dir, exist_ok=True)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for i, sample in enumerate(data):
        sample_dir = os.path.join(base_output_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)

        gt_pos = (sample['gt_pos'].cpu() if torch.is_tensor(sample['gt_pos']) else sample['gt_pos']).squeeze(0)
        pred_pos = (sample['pred_pos'].cpu() if torch.is_tensor(sample['pred_pos']) else sample['pred_pos']).squeeze(0)
        connectivity = (sample['cells'].cpu() if torch.is_tensor(sample['cells']) else sample['cells']).squeeze()[0]

        mesh_pos = (sample['mesh_pos'].cpu() if torch.is_tensor(sample['mesh_pos']) else sample['mesh_pos']).squeeze(0)
        node_type = (sample['node_type'].cpu() if torch.is_tensor(sample['node_type']) else sample['node_type']).squeeze(0)

        # Compute full vector deformations
        gt_deform = np.abs(mesh_pos - gt_pos)
        pred_deform = np.abs(mesh_pos - pred_pos)
        error = np.abs(pred_deform - gt_deform)

        save_time_series_vtu(gt_pos, pred_pos, connectivity, gt_deform, pred_deform, node_type, error, sample_dir)


def main():
    data_path = "/home/bipinshrestha228/PressNet/conference_paper/gcn_merged_400_abs/infer_rollout.pkl"
    base_output_dir = "/home/bipinshrestha228/PressNet/vtu_testing"
    process_samples(data_path, base_output_dir)

if __name__ == "__main__":
    main()
