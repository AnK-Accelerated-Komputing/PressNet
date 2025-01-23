import plotly.graph_objects as go
import os
import torch
# Get the parent directory and add it to the Python path
import sys
from pathlib import Path
# parent_dir = Path().resolve().parent
# sys.path.append(str(parent_dir))
sys.path.append('/home/user/AnK_MeshGraphNets')

from PIL import Image
import numpy as np
import pickle

from preprocessing.evalutation_trajectory import generate_evaluation_trajectory

def save_rollout_frames(single_trajectory,save_directory,key='world_pos'):
    skip = 5
    num_steps = single_trajectory[key].shape[0]
    num_frames = 1 * num_steps // skip

    print(f"Number of frames: {num_frames}")
    node_type = single_trajectory['node_type'][0].to('cpu').flatten().tolist()
    color_map = {
        0: 'blue',  # Type 0
        1: 'green', # Type 1
        3: 'red',   # Type 3
    }
    colors = [color_map[nt] for nt in node_type]


    bb_min = torch.squeeze(single_trajectory[key], dim=0).cpu().numpy().min(axis=(0, 1))
    bb_max = torch.squeeze(single_trajectory[key], dim=0).cpu().numpy().max(axis=(0, 1))
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]

    cells = single_trajectory['cells']
    faces_result = []
    # print(faces.shape)
    for faces_step in cells:
        later = torch.cat((faces_step[:, 2:4], torch.unsqueeze(faces_step[:, 0], 1)), -1)
        faces_step = torch.cat((faces_step[:, 0:3], later), 0)
        faces_result.append(faces_step)
        # print(faces_step.shape)
    faces_result = torch.stack(faces_result, 0)

    faces = torch.squeeze(faces_result,dim=0)[0].to('cpu')
    faces = faces.numpy()
    i,j,k=faces[:, 0],faces[:, 1],faces[:, 2]
    i,j,k=([int(w)for w in i],
        [int(w) for w in j],
        [int(w) for w in k])

    m=0
    displacement = abs(single_trajectory[key]-single_trajectory['mesh_pos']).to('cpu')
    flattened_displacement = displacement.reshape(-1, displacement.size(-1))
    print("shape",flattened_displacement.size())
    y_displacement =flattened_displacement[:,1]
    node_type = single_trajectory['node_type'].to('cpu').flatten()
    print(node_type.size())
    mask = (node_type == 0)
    masked_displacement = np.where(mask, y_displacement, 0)
    gt_y_displacement_min = masked_displacement.min()
    gt_y_displacement_max = masked_displacement.max()
    print('displacement(max,min):',gt_y_displacement_max,gt_y_displacement_min)
    stress = single_trajectory['stress'].to('cpu').flatten()
    max_stress = stress.max()
    min_stress = stress.min() 

    for m in range(num_frames):
        p = m*skip
        y_displacement = abs(single_trajectory[key][p]-single_trajectory['mesh_pos'][p])[:,1].to('cpu')
        node_type = single_trajectory['node_type'][p].to('cpu').flatten()
        mask = (node_type != 1) | (node_type != 3)
        mask_2 = (node_type == 1) | (node_type == 3)
        # masked_intensity = np.where(mask, y_displacement, np.nan)

        von_mises_stress = single_trajectory['stress'][p].to('cpu').flatten()        
       
        xyz_pos = torch.squeeze(single_trajectory[key],dim=0)[p].to('cpu')
        x, y, z = xyz_pos[:, 0], xyz_pos[:, 1], xyz_pos[:, 2]
        mask_x = np.where(mask, x, np.nan)
        mask_y = np.where(mask, y, np.nan)
        mask_z = np.where(mask, z, np.nan)

        mask_x_2 = np.where(mask_2, x, np.nan)
        mask_y_2 = np.where(mask_2, y, np.nan)
        mask_z_2 = np.where(mask_2, z, np.nan)

        edges_x, edges_y, edges_z = [], [], []
        edges = [(i[m], j[m], k[m]) for m in range(len(i))]

        for edge in edges:
            for start, end in [(edge[0], edge[1]), (edge[1], edge[2]), (edge[2], edge[0])]:
                edges_x.extend([x[start], x[end], None])  # x-coordinates for each edge line segment
                edges_y.extend([y[start], y[end], None])  # y-coordinates for each edge line segment
                edges_z.extend([z[start], z[end], None])  # z-coordinates for each edge line segment
                
        frame_fig = go.Figure(
            data=[go.Mesh3d(
                    x=mask_x, y=mask_y, z=mask_z,
                    i=i, j=j, k=k,
                    #intensity=masked_intensity,
                    intensity=von_mises_stress,
                    showscale=True,
                    coloraxis="coloraxis",
                    opacity=1
            ),
            go.Mesh3d(
                    x=mask_x_2, y=mask_y_2, z=mask_z_2,
                    i=i, j=j, k=k,
                    #intensity=masked_intensity,
                    showscale=False,
                    color="white",
                    opacity=1
            ),
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=0.5,
                            color='black',),
                name="Nodes"
            ),
            go.Scatter3d(
                x=edges_x,y=edges_y,z=edges_z,
                mode='lines',
                line=dict(color='black',width=1),
                name='Edges'
            )
            ],
            # name=str(i)
        )
       
        frame_fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=0.5,y=0.5,z=2),
                    up=dict(x=0, y=1, z=0)
                    ),  # Define y as the up axis
                xaxis=dict(range=[bb_min[0],bb_max[0]]),  # Set x-axis range
                yaxis=dict(range=[bb_min[1],bb_max[1]]),  # Set y-axis range
                zaxis=dict(range=[bb_min[2],bb_max[2]]),  # Set z-axis range
                aspectmode='manual',  # Use manual aspect ratio control
                aspectratio=dict(x=x_range/x_range*2, y=y_range/x_range*2, z=z_range/x_range*2),  # Set the aspect ratio as per actual data ranges
            ),
            width=1400,  # Increase overall plot width
            height=800,  # Increase overall plot height
            legend=dict(
                font=dict(size=10),  # Smaller font for the legend
                itemsizing="constant",  # Ensure consistent sizing
                x=1.1,  # Position legend outside the plots
                y=0.9
            ),
            coloraxis=dict(
                #cmin=float(gt_y_displacement_min),  # Global min
                #cmax=float(gt_y_displacement_max),
                cmin=float(min_stress),
                cmax=float(max_stress),    # Global max
                #colorscale="OrRd",  # Or any preferred colorscale
                colorscale="jet",
                colorbar = dict(
                    title="Von Mises stress"
                )
            )
                
        )
        file_save_path =os.path.join(save_directory,f"predicted_step_{p}.png")
        frame_fig.write_image(file_save_path)
        print(f"frame ---{m}--- out of {num_frames} done",end="\r")
        # if p == 0:
        #     frame_fig.write_html(os.path.join(save_directory,f"predicted_step_{p}.html"))
    print(f"frame{m} out of {num_frames} done")

    return "frames saved"

def generate_gif(save_directory,fps=5,loop=0):
    directory = Path(save_directory)
    # List to store the paths of the matching files
    png_files = []

    # Loop through the directory and find matching files
    for file in directory.rglob('predicted_step*.png'):
        # Extract the number 'n' from the filename
        try:
            n = int(file.stem.split('_')[-1])  # Extract the number n from the filename
            # Add the file path and its corresponding n value to the list
            png_files.append((file, n))
        except ValueError:
            continue  # Skip files where the number extraction fails

    # Sort the files based on the extracted number 'n'
    png_files.sort(key=lambda x: x[1])  # Sort by n value (second element in tuple)

    # Get the sorted file paths
    sorted_png_files = [file_path for file_path, _ in png_files]
    gif_outfile = os.path.join(save_directory, 'rollout.gif')
    # Generate the GIF
    imgs = [Image.open(file) for file in sorted_png_files]
    imgs[0].save(fp=gif_outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

    print(f"GIF saved at {gif_outfile}")

    return 'animated gif saved'
      
def animate_rollout(raw_data_directory, save_directory):
    device = "cpu"
    trajectory_data = generate_evaluation_trajectory(raw_data_directory,device)
    print(save_rollout_frames(trajectory_data,save_directory,key='target|world_pos')) #key: 'target|world_pos', "world_pos"
    print(generate_gif(save_directory))
    return 


def main():
    raw_data_directory = '/home/user/AnK_MeshGraphNets/raw_data/solid185/train/3mm_plate_data'
    save_directory = '/home/user/AnK_MeshGraphNets/test_results/GT/full_og_profile/3mm'
    animate_rollout(raw_data_directory, save_directory)

    return


if __name__ == '__main__':  
    main()