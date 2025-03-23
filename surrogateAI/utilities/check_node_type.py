import plotly.graph_objects as go
import os
import torch

from plotly.subplots import make_subplots

from pathlib import Path
from PIL import Image
import numpy as np
import pickle

def save_rollout_frames(rollout_data,save_directory,i=0,key="stress"):
    skip = 4
    # print(rollout_data)
    num_steps = rollout_data[i]['gt_pos'].squeeze(0).shape[0]
    print("num_steps",num_steps)
    num_frames = 1 * num_steps // skip
    print("num_frames",num_frames)

    print(f"Number of frames: {num_frames}")
    
    single_trajectory = rollout_data[0]
    for keys in single_trajectory.keys():
        print(keys,single_trajectory[keys].shape)
        if single_trajectory[keys].shape[0] == 1:
            single_trajectory[keys] = single_trajectory[keys].squeeze(0)
        print("                                 ",keys,single_trajectory[keys].shape)
    bb_min = torch.squeeze(single_trajectory['gt_pos'], dim=0).cpu().numpy().min(axis=(0, 1))
    bb_max = torch.squeeze(single_trajectory['gt_pos'], dim=0).cpu().numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    faces = torch.squeeze(single_trajectory['faces'],dim=0)[0].to('cpu')
    faces = faces.numpy()
    i,j,k=faces[:, 0],faces[:, 1],faces[:, 2]
    i,j,k=([int(w)for w in i],
        [int(w) for w in j],
        [int(w) for w in k])

    m=0
    gt_y_displacement_all = abs((single_trajectory['node_type']).to('cpu'))
    # # print(gt_y_displacement_all.shape)

    displacement_min = torch.squeeze(gt_y_displacement_all, dim=0).cpu().numpy().min(axis=(0, 1))
    displacement_max =torch.squeeze(gt_y_displacement_all, dim=0).cpu().numpy().max(axis=(0, 1))


    # displacement_min = torch.squeeze(single_trajectory['stress'], dim=0).cpu().numpy().min(axis=(0, 1))
    # displacement_max =torch.squeeze(single_trajectory['stress'], dim=0).cpu().numpy().max(axis=(0, 1))
    print("minimum displacement in x,y,z dirn", displacement_min)
    print("maximum displacement in x,y,z dirn", displacement_max)
    # print(displacement_max, displacement_min)
    gt_y_displacement_min = displacement_min[0]
    gt_y_displacement_max = displacement_max[0]
    for m in range(1):
        gt_y_displacement = abs((single_trajectory['gt_pos'][m]-single_trajectory['mesh_pos'][m])[:,1].to('cpu'))
        # pred_y_displacement = abs((single_trajectory['pred_pos'][m].to('cpu')-single_trajectory['mesh_pos'][m].to('cpu'))[:,1].to('cpu'))

        node_type = single_trajectory['node_type'][m].to('cpu').flatten()
    
        # pred_stress = pred_y_displacement.flatten()

        gt_stress = gt_y_displacement.flatten()

        # pred_xyz_pos = torch.squeeze(single_trajectory['pred_pos'], dim=0)[m].to('cpu')
        # pred_x, pred_y, pred_z = pred_xyz_pos[:, 0].numpy(), pred_xyz_pos[:, 1].numpy(), pred_xyz_pos[:, 2].numpy()

        gt_xyz_pos = torch.squeeze(single_trajectory['gt_pos'], dim=0)[m].to('cpu')
        gt_x, gt_y, gt_z = gt_xyz_pos[:, 0].numpy(), gt_xyz_pos[:, 1].numpy(), gt_xyz_pos[:, 2].numpy()


        
        # z_displacement_min =  gt_z_displacement.min() #min(pred_z_displacement.min(),)
        # z_displacement_max =  gt_z_displacement.max() #max(pred_z_displacement.max(),)

        # edges_x, edges_y, edges_z = [], [], []
        # pred_edges_x, pred_edges_y, pred_edges_z = [], [], []
        gt_edges_x, gt_edges_y, gt_edges_z = [], [], []
        edges = [(i[m], j[m], k[m]) for m in range(len(i))]

        # for edge in edges:
        #     for start, end in [(edge[0], edge[1]), (edge[1], edge[2]), (edge[2], edge[0])]:
        #         pred_edges_x.extend([pred_x[start], pred_x[end], None])  # x-coordinates for each edge line segment
        #         pred_edges_y.extend([pred_y[start], pred_y[end], None])  # y-coordinates for each edge line segment
        #         pred_edges_z.extend([pred_z[start], pred_z[end], None])  # z-coordinates for each edge line segment

        for edge in edges:
            for start, end in [(edge[0], edge[1]), (edge[1], edge[2]), (edge[2], edge[0])]:
                gt_edges_x.extend([gt_x[start], gt_x[end], None])  # x-coordinates for each edge line segment
                gt_edges_y.extend([gt_y[start], gt_y[end], None])  # y-coordinates for each edge line segment
                gt_edges_z.extend([gt_z[start], gt_z[end], None])  # z-coordinates for each edge line segment

        # Create subplots with two figures
        frame_fig = make_subplots(rows=1, cols=1,
                                  specs=[[{'type': 'scene'}]],
                                  subplot_titles=('Ground Truth Positions', 'Predicted Positions'))

                
        # # Add predicted positions
        # frame_fig.add_trace(
        #     go.Mesh3d(
        #         x=pred_mask_x, y=pred_mask_y, z=pred_mask_z,
        #         i=i, j=j, k=k,
        #         intensity=pred_stress,
        #         showscale=True,
        #         coloraxis="coloraxis",
        #         # colorscale='viridis',
        #         opacity=1
        #     ), row=1, col=2
        # )

        # frame_fig.add_trace(
        #     go.Mesh3d(
        #         x=pred_mask_x_2, y=pred_mask_y_2, z=pred_mask_z_2,
        #         i=i, j=j, k=k,
        #         # intensity=pred_masked_intensity_2,
        #         showscale=True,
        #         color="white",
        #         # coloraxis="coloraxis",
        #         # colorscale='viridis',
        #         opacity=1
        #     ), row=1, col=2
        # )

        # frame_fig.add_trace(
        #     go.Scatter3d(
        #         x=pred_x, y=pred_y, z=pred_z,
        #         mode="markers",
        #         marker=dict(size=0.4, color='black'),
        #         name="Predicted Nodes"
        #     ), row=1, col=2
        # )

        # frame_fig.add_trace(
        #     go.Scatter3d(
        #         x=pred_edges_x, y=pred_edges_y, z=pred_edges_z,
        #         mode='lines',
        #         line=dict(color='black', width=0.8),
        #         name='Predicted Edges'
        #     ), row=1, col=2
        # )

        # Add ground truth positions
        frame_fig.add_trace(
            go.Mesh3d(
                x=gt_x, y=gt_y, z=gt_z,
                i=i, j=j, k=k,
                intensity=node_type,
                showscale=False,
                coloraxis="coloraxis",
                # colorscale='viridis',
                opacity=1
            ), row=1, col=1
        )

        
        frame_fig.add_trace(
            go.Scatter3d(
                x=gt_x, y=gt_y, z=gt_z,
                mode="markers",
                marker=dict(size=0.4, color='black'),
                name="Ground Truth Nodes"
            ), row=1, col=1
        )

        frame_fig.add_trace(
            go.Scatter3d(
                x=gt_edges_x, y=gt_edges_y, z=gt_edges_z,
                mode='lines',
                line=dict(color='black', width=0.8),
                name='Ground Truth Edges'
            ), row=1, col=1
        )

        # Update layout
        frame_fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=0.5,y=0.5,z=3.5),
                    up=dict(x=0, y=1, z=0)
                    ),  # Define y as the up axis
                xaxis=dict(range=[bb_min[0],bb_max[0]]),  # Set x-axis range
                yaxis=dict(range=[bb_min[1],bb_max[1]]),  # Set y-axis range
                zaxis=dict(range=[bb_min[2],bb_max[2]]),  # Set z-axis range
                aspectmode='manual',  # Use manual aspect ratio control
                aspectratio=dict(x=x_range/x_range*2, y=y_range/x_range*2, z=z_range/x_range*2),  # Set the aspect ratio as per actual data ranges
            ),
            # scene2=dict(  # Apply the same settings to the second subplot
            #     camera=dict(
            #         eye=dict(x=0.5, y=0.5, z=3.5),
            #         up=dict(x=0, y=1, z=0)
            #     ),
            #     xaxis=dict(range=[bb_min[0], bb_max[0]]),
            #     yaxis=dict(range=[bb_min[1], bb_max[1]]),
            #     zaxis=dict(range=[bb_min[2], bb_max[2]]),
            #     aspectmode='manual',
            #     aspectratio=dict(x=x_range / x_range * 2, y=y_range / x_range * 2, z=z_range / x_range * 2),
            # ),
            # Adjust layout for larger plots and smaller legend
            width=1400,  # Increase overall plot width
            height=800,  # Increase overall plot height
            legend=dict(
                font=dict(size=10),  # Smaller font for the legend
                itemsizing="constant",  # Ensure consistent sizing
                x=1.1,  # Position legend outside the plots
                y=0.9
            ),
            margin=dict(l=0, r=0, t=30, b=30),  # Reduce extra whitespace around plots
        
            coloraxis=dict(
                cmin=float(gt_y_displacement_min),  # Global min
                cmax=float(gt_y_displacement_max),  # Global max
                colorscale="jet",  # Or any preferred colorscale
                colorbar = dict(
                    title="y_disp"
                )
            ) # Ensure color range consistency
        )
        file_save_path =os.path.join(save_directory,f"predicted_step_{m}_node_type.png")
        frame_fig.write_image(file_save_path)
        print(f"frame ---{m}--- out of {num_frames} done",end="\r")
    print(f"frame{m} out of {num_frames} done")

    return "frames saved"
      
def animate_rollout(data_path, save_directory,i=0,key="stress"):
    with open(data_path, 'rb') as fp:
        rollout_data = pickle.load(fp)
    print(len(rollout_data))
    print(save_rollout_frames(rollout_data,save_directory,i=i,key=key))
    return 


def main():
    data_path =  '/home/user/PressNet/surrogateAI/training_output/regDGCNN_seg/Channel_rect_press_dataset/Tue-Mar-18-13-42-20-2025/rollout/rollout_epoch_350.pkl'
    save_directory = '/home/user/PressNet/surrogateAI/results/regDGCNN/Channel_rect_press_dataset/400_step'
    os.makedirs(save_directory,exist_ok=True)
    animate_rollout(data_path, save_directory)

    return


if __name__ == '__main__':  
    main()