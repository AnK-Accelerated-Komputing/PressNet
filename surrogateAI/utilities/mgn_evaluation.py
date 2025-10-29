import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from PIL import Image
import json


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame

def load_pkl_file(file_path):
    """ Load the pkl file and return its contents. """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def obtain_step_loss(single_trajectory,loss='rmse'):
    step_loss = {
        'stress' : [],
        'max_stress' : [],
        'deform_y' : [],
        "max_deform_y" : []
    }
    gt = {
        'stress' : [],
        'max_stress' : [],
        'deform_y' : [],
        'max_deform_y' : []    
    }

    pred = {
        'stress' : [],
        'max_stress' : [],
        'deform_y' : [],
        'max_deform_y' : []    
    }
    for i in range(single_trajectory['node_type'].size(dim=0)):
        node_type = single_trajectory['node_type'][i].to('cpu').flatten()
        print(node_type.shape)
        mask = node_type ==0
        print(mask.shape)

        gt_stress = single_trajectory['gt_stress'][i].to('cpu').flatten()[mask]
        pred_stress = single_trajectory['pred_stress'][i].to('cpu').flatten()[mask]
        gt['stress'].append(torch.mean(gt_stress))
        gt['max_stress'].append(torch.max(gt_stress))
        pred['stress'].append(torch.mean(pred_stress))
        pred['max_stress'].append(torch.max(pred_stress))

        gt_pos = single_trajectory['gt_pos'][i].to('cpu')[mask]
        pred_pos = single_trajectory['pred_pos'][i].to('cpu')[mask]
        mesh_pos = single_trajectory['mesh_pos'][i].to('cpu')[mask]

        gt_y_deform = torch.abs((gt_pos-mesh_pos)[:,1].flatten())
        pred_y_deform = torch.abs((pred_pos-mesh_pos)[:,1].flatten())
        
        if i == 99 or i == 199:
            gt["deform_y"].append(gt["deform_y"][-1])
            gt["max_deform_y"].append(gt["max_deform_y"][-1])
        else:
            gt["deform_y"].append(torch.mean(gt_y_deform))
            gt["max_deform_y"].append(torch.max(gt_y_deform))
        
        pred["deform_y"].append(torch.mean(pred_y_deform))
        pred["max_deform_y"].append(torch.max(pred_y_deform))

        if loss == 'rmse':                
            # l2_stress = torch.norm(gt_stress-pred_stress,p=2)
            # loss_stress = torch.sqrt((torch.mean(gt_stress) - torch.mean(pred_stress)) ** 2)    #correction needed
            loss_stress = torch.sqrt(torch.mean((gt_stress - pred_stress) ** 2))
            loss_stress_max = torch.sqrt((torch.max(gt_stress) - torch.max(pred_stress))**2)
            
            loss_y_deform = torch.sqrt(torch.mean((gt_y_deform - pred_y_deform) ** 2))
            loss_y_deform_max = torch.sqrt((torch.max(gt_y_deform) - torch.max(pred_y_deform))**2)
        if i == 99 or i == 198:
            step_loss["deform_y"].append(step_loss["deform_y"][-1])
            step_loss["max_deform_y"].append(step_loss["max_deform_y"][-1])
        else:
            step_loss["deform_y"].append(loss_y_deform)
            step_loss["max_deform_y"].append(loss_y_deform_max)
        step_loss['stress'].append(loss_stress)
        step_loss['max_stress'].append(loss_stress_max)
        
    return step_loss, gt, pred

def plot_step_loss_gif(step_loss,gt,pred,output_dir,key='all'):
    steps = list(range(1, len(step_loss['stress']) + 1))
    if key == 'all':
        keys = ['stress', 'max_stress', 'deform_y', 'max_deform_y']
    else:
        keys = [key]

    for key in keys:
        y_max = max(max(step_loss[key]), max(gt[key]), max(pred[key]))
        for step in steps:
            # Plot the error over steps
            plt.figure(figsize=(8, 5))
            loss_key = [loss for loss in step_loss[key]][:step]
            gt_key = [loss for loss in gt[key]][:step]
            pred_key = [loss for loss in pred[key]][:step]
            plt.plot(steps[:step], gt_key,  linestyle='-', color='g',label=f'gt_{key} ')
            plt.plot(steps[:step], pred_key,  linestyle='-', color='b',label=f'pred_{key} ')
            plt.plot(steps[:step], loss_key,  linestyle='-', color='r',label=f'loss_{key} ')
            # print(stress_rmse)
            plt.title('Error over Steps')
            plt.xlabel('Step')
            plt.ylabel('Error')
            plt.grid(True)
            plt.legend()
            plt.xlim(0,steps[-1]+1)
            plt.ylim(0,y_max)
            filename = os.path.join(output_dir,key,"steps", f"frame_{step:03d}.png")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            if step == steps[-1]:
                filename = os.path.join(output_dir,key, f"frame_{step:03d}.png")
                plt.savefig(filename)

            plt.close()
        
        frames = []
        for step in steps:
            filename = os.path.join(output_dir,key,"steps", f"frame_{step:03d}.png")
            frames.append(Image.open(filename))

        # Save the GIF
        gif_dir = os.path.join(output_dir,key)
        gif_filename = os.path.join(gif_dir,f"{key}_error_over_steps.gif")
        fps=50
        frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)

        print(f"GIF saved as {gif_filename}")

    return

def compute_metrics(pred, gt):
    # error = torch.mean(pred) - torch.mean(gt)                   #correction needed       #change
    error = (pred-gt)
    l2 = torch.sqrt(torch.sum(error**2))
    rmse = torch.sqrt(torch.mean(error**2))                     #correction needed      #change

    max_rmse = torch.sqrt((torch.max(pred)-torch.max(gt))**2)
    return l2.item(), rmse.item(),max_rmse.item()


def obtain_domain_wise_loss_stress(single_trajectory):
    
    mesh_pos = single_trajectory['mesh_pos'].to('cpu')
    pred_stress = single_trajectory['pred_stress'].to('cpu')
    gt_stress = single_trajectory['gt_stress'].to('cpu')
    node_type= single_trajectory['node_type'].to('cpu')

    x_ranges = []
    x_max = mesh_pos[:,:,0].max()
    x_min = mesh_pos[:,:,0].min()
    # print("max",x_max,"min",x_min)
    discrete_size = 20
    x=x_min
    while x < x_max:
        # print(x, end=' ')
        if x+discrete_size < x_max:
            x_ranges.append((x+0,x+discrete_size))
        else:
            x_ranges.append((x+0,x_max))
        x += discrete_size
    # print()
    # print('x_ranges',x_ranges)
    results = {
        'range' : [],
        'l2' : [],
        'rmse' : [],
        'max_rmse': [],
        'gt': [],
        'max_gt': [],
        'pred': [],
        'max_pred': []
    }
    for x_min, x_max in x_ranges:
        # Create a mask for nodes in the x range and node_type = 0
        x_coords = mesh_pos[:, :, 0]  # Extract x-coordinates
        node_type_squeezed = node_type.squeeze(-1)
        mask = (x_coords <= x_max) & (x_coords > x_min) & (node_type_squeezed == 0)

        # Apply the mask
        filtered_pred_stress = pred_stress[mask]
        filtered_gt_stress = gt_stress[mask]

        # Skip if no nodes match
        if filtered_pred_stress.numel() == 0:
            # print(0)
            continue

        # Compute metrics
        l2, rmse,max_rmse = compute_metrics(filtered_pred_stress, filtered_gt_stress)

        results['range'].append((x_min, x_max))
        results['l2'].append(l2)
        results['rmse'].append(rmse)
        results['max_rmse'].append(max_rmse)
        results['gt'].append(torch.mean(filtered_gt_stress).item())
        results['max_gt'].append(torch.max(filtered_gt_stress).item())
        results['pred'].append(torch.mean(filtered_pred_stress).item())
        results['max_pred'].append(torch.max(filtered_pred_stress).item())
    return x_ranges,results

def obtain_domain_wise_loss_deform(single_trajectory):
    
    mesh_pos = single_trajectory['mesh_pos'].to('cpu')
    pred_stress = single_trajectory['pred_pos'].to('cpu')
    gt_stress = single_trajectory['gt_pos'].to('cpu')
    node_type= single_trajectory['node_type'].to('cpu')

    x_ranges = []
    x_max = mesh_pos[:,:,0].max()
    x_min = mesh_pos[:,:,0].min()
    # print("max",x_max,"min",x_min)
    discrete_size = 20
    x=x_min
    while x < x_max:
        # print(x, end=' ')
        if x+discrete_size < x_max:
            x_ranges.append((x+0,x+discrete_size))
        else:
            x_ranges.append((x+0,x_max))
        x += discrete_size
    # print()
    # print('x_ranges',x_ranges)
    results = {
        'range' : [],
        'l2' : [],
        'rmse' : [],
        'max_rmse': [],
        'gt': [],
        'max_gt': [],
        'pred': [],
        'max_pred': []
    }
    for x_min, x_max in x_ranges:
        # Create a mask for nodes in the x range and node_type = 0
        x_coords = mesh_pos[:, :, 0]  # Extract x-coordinates
        node_type_squeezed = node_type.squeeze(-1)
        mask = (x_coords <= x_max) & (x_coords > x_min) & (node_type_squeezed == 0)

        # Apply the mask
        filtered_pred_stress = pred_stress[mask]
        filtered_gt_stress = gt_stress[mask]

        # Skip if no nodes match
        if filtered_pred_stress.numel() == 0:
            # print(0)
            continue

        # Compute metrics
        l2, rmse,max_rmse = compute_metrics(filtered_pred_stress, filtered_gt_stress)

        results['range'].append((x_min, x_max))
        results['l2'].append(l2)
        results['rmse'].append(rmse)
        results['max_rmse'].append(max_rmse)
        results['gt'].append(torch.mean(filtered_gt_stress).item())
        results['max_gt'].append(torch.max(filtered_gt_stress).item())
        results['pred'].append(torch.mean(filtered_pred_stress).item())
        results['max_pred'].append(torch.max(filtered_pred_stress).item())
    return x_ranges,results

def plot_domain_wise_loss(x_ranges,results, output_dir,key='all'):
    # print(len(results['range']))
    x_range_labels = [(results['range'][p][0]+results['range'][p][1])/2 for p in range(len(results['range']))]
    if key == 'all':
        keys = ['stress','max_stress']
    else:
        keys = [key]
    
    for key in keys:
        if key == 'stress':
            errors = results['rmse']
            stress_gt = results['gt']
            stress_pred = results['pred']
        if key == 'max_stress':
            errors = results['max_rmse']
            stress_gt = results['max_gt']
            stress_pred = results['max_pred']

        colors = plt.cm.coolwarm([norm for norm in errors])  # Reverse colormap (blue low, red high)

        # Plot the histogram
        plt.figure(figsize=(10,6))
        bars2 = plt.bar(x_range_labels, stress_gt,width=20, color=colors, edgecolor='black',align='center')
        bar_centers2 = [bar.get_x() + bar.get_width() / 2 for bar in bars2]
        bar_heights2 = [bar.get_height() for bar in bars2]
        bars3 = plt.bar(x_range_labels, stress_pred,width=20, color=colors, edgecolor='black',align='center')
        bar_centers3 = [bar.get_x() + bar.get_width() / 2 for bar in bars3]
        bar_heights3 = [bar.get_height() for bar in bars3]
        plt.close()

        max_error = max(errors)
        min_error = min(errors)
        normalized_errors = [(error - min_error) / (max_error - min_error) for error in errors]
        colors = plt.cm.coolwarm([norm for norm in normalized_errors])  # Reverse colormap (blue low, red high)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.plot(bar_centers2, bar_heights2, color='g', marker='o', linewidth=2, label=f' GT {key}')
        plt.plot(bar_centers3, bar_heights3, color='b', marker='o', linewidth=2, label=f' Pred {key}')
        bars = plt.bar(x_range_labels, errors,width=20, color=colors, edgecolor='black',align='center')

        bar_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        bar_heights = [bar.get_height() for bar in bars]

        # Plot a curve connecting the highest points of the bars
        plt.plot(bar_centers, bar_heights, color='r', marker='o', linewidth=2, label='Error Trend')

        for i, (bar, norm_error) in enumerate(zip(bars, normalized_errors)):
            # Interpolating the color for the rectangle using the same normalized error
            color = plt.cm.coolwarm(norm_error)  # Get color from colormap based on error value
            
            # Rectangle's bottom-left corner at the bar's x position, y=-10, width=bar width, height=10
            rect = patches.Rectangle(
                (bar.get_x(), -20),   # Rectangle position
                bar.get_width(),      # Width of the rectangle (same as bar)
                10,                   # Height of the rectangle
                linewidth=0,
                edgecolor='black',
                facecolor=color,      # Apply the color based on the continuous error value
                alpha=0.5
            )
            plt.gca().add_patch(rect)

        plt.xlabel('X-dimension ranges', fontsize=14)
        plt.ylabel('L2 Error', fontsize=14)
        plt.title('Stress Error Across X-dimension Ranges', fontsize=16)
        plt.xticks(
            x_range_labels, 
            [f"[{x_max}, {x_min}]" for x_min, x_max in results['range']], 
            fontsize=7, 
            rotation=0
        )
        plt.ylim(-20, max(bar_heights2) + 50)
        plt.xlim(-300, 5) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min_error, vmax=max_error))
        sm.set_array([])  # Set array to an empty array to avoid "no mappable" error

        # Add the colorbar with the correct axis
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('Error Magnitude', fontsize=12)
        plt.tight_layout()
        plt.legend()
        filename= os.path.join(output_dir,key,'domain_wise_stress_error.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"plot saved at {filename}")
        plt.close()

    return


def evaluate_rollout(rollout_pth, base_output_dir,plot=True):
    print(f"Rollout Evaluation for {rollout_pth}")
    rollout = load_pkl_file(rollout_pth)
    for i in range(len(rollout)):
        start_time = time.time()
        single_trajectory = rollout[i]
        # single_trajectory = squeeze_data_frame(single_trajectory)
        print(single_trajectory.keys())
        print(single_trajectory["cells"].shape)
        print(single_trajectory["faces"].shape)
        print(single_trajectory["mesh_pos"].shape)
        print(single_trajectory["gt_pos"].shape)
        print(single_trajectory["pred_pos"].shape)
        print(single_trajectory["cur_positions"].shape)
        print(single_trajectory["cur_velocities"].shape)
        print(single_trajectory["gt_stress"].shape)
        print(single_trajectory["pred_stress"].shape)
        print(single_trajectory["node_type"].shape)
        output_dir = os.path.join(base_output_dir,str(i))
        os.makedirs(output_dir, exist_ok=True)
        print(f"working on the trajectory {i}")
        # break
        print("calculating step loss")
        step_loss,gt,pred = obtain_step_loss(single_trajectory)
        step_stress_loss_avg = torch.mean(torch.tensor(step_loss['stress']))
        step_stress_loss_max = torch.max(torch.tensor(step_loss['max_stress']))
        step_y_loss_avg = torch.mean(torch.tensor(step_loss['deform_y']))
        step_y_loss_max = torch.max(torch.tensor(step_loss['max_deform_y']))
        print("step loss:","stress_avg",step_stress_loss_avg,"stress_max",step_stress_loss_max,"y_avg",step_y_loss_avg,"y_max",step_y_loss_max)
        if plot:
            plot_step_loss_gif(step_loss,gt,pred,output_dir)
        print("step loss calculated")
        print("calculating domain loss")
        x_ranges, domain_loss = obtain_domain_wise_loss_stress(single_trajectory)
        x_ranges_deform, domain_loss_y_deform = obtain_domain_wise_loss_deform(single_trajectory)     #changes
        # print(domain_loss['rmse'])
        # print(domain_loss['max_rmse'])
        domain_stress_loss_avg = torch.mean(torch.tensor(domain_loss['rmse']))
        domain_stress_loss_max = torch.max(torch.tensor(domain_loss['max_rmse']))

        domain_y_deform_loss_avg = torch.mean(torch.tensor(domain_loss_y_deform['rmse']))           #change
        domain_y_deform_loss_max = torch.max(torch.tensor(domain_loss_y_deform['max_rmse']))
 
        print("Domain loss:","stress_avg",domain_stress_loss_avg,"stress_max",domain_stress_loss_max)
        if plot:
            plot_domain_wise_loss(x_ranges,domain_loss,output_dir)
        print("domain loss calculated")
        losses ={
            'step_stress_loss_avg':step_stress_loss_avg.item(),
            'step_stress_loss_max':step_stress_loss_max.item(),
            'step_y_deform_loss_avg':step_y_loss_avg.item(),
            'step_y_deform_loss_max':step_y_loss_max.item(),
            'domain_stress_loss_avg':domain_stress_loss_avg.item(),
            'domain_stress_loss_max':domain_stress_loss_max.item(),
            'domain_y_deform_loss_avg':domain_y_deform_loss_avg.item(),                                 #changes
            'domain_y_deform_loss_max':domain_y_deform_loss_max.item(),
            }
        json_file_path = os.path.join(os.path.dirname(base_output_dir),'rollout_info.json')
        try:
            print(json_file_path)
            with open(json_file_path,"r") as file:
                info = json.load(file)
        except:
            info = {
            'index': [],
            'file_name': [],
            'inference_time': [],
            'losses': []  
            }
        if 'losses' not in info or not isinstance(info['losses'], list):
            info['losses'] = []
        info['losses'].append(losses)

        with open(json_file_path,"w") as file:
            json.dump(info, file, indent=4)
        print("Appended new loss to json file")
        end_time = time.time()
        interval_sec = end_time - start_time
        print(f"Time taken for trajectory {i}: {interval_sec//60} minutes {interval_sec%60:.2f} seconds")

    return

def main():
    base_dir = "/home/ujwal/Documents/PressNet_Inference/PressNet/datasets/data/dis_and_stress"
    rollout_pth = os.path.join(base_dir,'rollout_epoch_860.pkl')
    output_dir = os.path.join(base_dir,'rollout_evaluation')
    evaluate_rollout(rollout_pth, output_dir,plot=False)

if __name__ == "__main__":
    main()