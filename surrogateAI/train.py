import os
from pathlib import Path
import pickle
import time
import datetime

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# from model import DGCNN, MagNet
from utilities import press_eval, common
from utilities.dataset import TrajectoryDataset

# from surrogateAI.models import press_model
from models import press_model


device = torch.device('cuda')


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame

def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def loss_fn(inputs, network_output, model):
    """L2 loss on position."""
    world_pos = inputs['curr_pos'].to(device)
    target_world_pos = inputs['next_pos'].to(device)
    
    cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - cur_position

    node_type = inputs['node_type'].to(device)

    world_pos_normalizer = model.get_output_normalizer()
    target_normalized = world_pos_normalizer(target_velocity)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    pos_prediction = network_output[:,:3]

    error = torch.sum((target_normalized - pos_prediction) ** 2, dim=1)
    # loss = torch.mean(error)
    loss = torch.mean(error[loss_mask])
    return  loss

def prepare_files_and_directories(output_dir,model_num,train_data_path):
    '''
        The following code is about creating all the necessary files and directories for the run
    '''
    train_data = train_data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(output_dir,str(model_num),train_data)
    run_create_time = time.time()
    run_create_datetime = datetime.datetime.fromtimestamp(run_create_time).strftime('%c')
    run_create_datetime_datetime_dash = run_create_datetime.replace(" ", "-").replace(":", "-")
    run_dir = os.path.join(output_dir, run_create_datetime_datetime_dash)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # make all the necessary directories
    checkpoint_dir = os.path.join(run_dir, 'checkpoint')
    log_dir = os.path.join(run_dir, 'log')
    rollout_dir = os.path.join(run_dir, 'rollout')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, log_dir, rollout_dir

def squeeze_data(data):
    transformed_data = {key: value.squeeze(0) for key, value in data.items()}
    return transformed_data


def main():
    device = torch.device('cuda')

    start_epoch = 0
    start_time = time.time()
    end_epoch = 1
    print(f"starting training from epoch {start_epoch} to {end_epoch}")
    train_data_path = "data/merged_400_trimmed_press_dataset.h5"
    output_dir = "training_output"
    train_dataset = TrajectoryDataset(train_data_path, split='train', stage=1)
    val_dataset = TrajectoryDataset(train_data_path, split='val', stage=1)
    # print(len(train_dataset),len(train_dataset)*3/399)
    # print(train_dataset[0])
    # print(len(val_dataset),len(val_dataset)*3/399)
    # print(val_dataset[0])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)



    ####_____________NEED TO SELECT AMONG DIFFERENT MODELS IN FUTURE_____________##########
    # model_num = 1
    # '''
    # if model 0: MGN
    # if model 1: GCN
    # '''
    # if model_num == 0:
    #     params = dict(field='world_pos', size=3, model=press_model, evaluator=press_eval)
    #     model = press_model.Model(params)
    # elif model_num == 1:
    #     model = press_model_GCN.GCN(nfeat=9,nhid=64,output=3,dropout=0.2,edge_dim=4)

    params = dict(field='world_pos', size=3, model=press_model, evaluator=press_eval)
    core_model = 'regDGCNN_seg'
    model = press_model.Model(params,core_model_name=core_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
    checkpoint_dir, log_dir, rollout_dir = prepare_files_and_directories(output_dir,core_model,train_data_path)
    
    epoch_training_losses = []
    step_training_losses = []
    epoch_run_times = []

    for epoch in range(start_epoch, end_epoch):
        print(f"running epoch {epoch+1}")
        epoch_start_time = time.time()

        epoch_training_loss = 0.0

        print(" training")
        for data in train_dataloader:
            frame = squeeze_data_frame(data)
            output = model(frame,is_training=True)
            loss = loss_fn(frame, output, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_training_losses.append(loss.detach().cpu())

            epoch_training_loss += loss.detach().cpu()

        epoch_training_losses.append(epoch_training_loss)
        print(f"epoch {epoch+1} training loss: {epoch_training_loss}, time taken: {time.time() - epoch_start_time}")


        loss_record = {}
        loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
        loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
        loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
        loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
        loss_record['train_epoch_losses'] = epoch_training_losses
        loss_record['all_step_train_losses'] = step_training_losses
        # save train loss
        temp_train_loss_pkl_file = os.path.join(log_dir, 'temp_train_loss.pkl')
        Path(temp_train_loss_pkl_file).touch()
        pickle_save(temp_train_loss_pkl_file, loss_record)
        if epoch%250 == 0:
            pickle_save(temp_train_loss_pkl_file.replace(".pkl",f'_{epoch}.pkl'), loss_record)



        model.save_model(os.path.join(checkpoint_dir,"epoch_model_checkpoint"))
        torch.save(optimizer.state_dict(),os.path.join(checkpoint_dir,"epoch_optimizer_checkpoint" + ".pth"))
        torch.save(scheduler.state_dict(),os.path.join(checkpoint_dir,"epoch_scheduler_checkpoint" + ".pth"))
        torch.save({'epoch': epoch}, os.path.join(checkpoint_dir, "epoch_checkpoint.pth"))
        
        if epoch == 13:
            scheduler.step()


        if epoch%50 == 0 or epoch == 0 or epoch == end_epoch-1:
            trajectories = []

            mse_losses = []
            l1_losses = []
            save_file = "rollout_epoch_" + str(epoch) + ".pkl"

            mse_loss_fn = torch.nn.MSELoss()
            l1_loss_fn = torch.nn.L1Loss()
            print(" evaluation")
            for data in val_loader:
                ##
                # print(data)
                data=squeeze_data(data)
                # print(len(data))
                # print(data['next_pos'].shape)
                # print(data['mesh_pos'].shape)
                # print(data['node_type'].shape)
                _, prediction_trajectory = press_eval.evaluate(model, data)
                mse_loss = mse_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                l1_loss = l1_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
                trajectories.append(prediction_trajectory)

            pickle_save(os.path.join(rollout_dir, save_file), trajectories)
            loss_record = {}
            loss_record['eval_total_mse_loss'] = torch.sum(torch.stack(mse_losses)).item()
            loss_record['eval_total_l1_loss'] = torch.sum(torch.stack(l1_losses)).item()
            loss_record['eval_mean_mse_loss'] = torch.mean(torch.stack(mse_losses)).item()
            loss_record['eval_max_mse_loss'] = torch.max(torch.stack(mse_losses)).item()
            loss_record['eval_min_mse_loss'] = torch.min(torch.stack(mse_losses)).item()
            loss_record['eval_mean_l1_loss'] = torch.mean(torch.stack(l1_losses)).item()
            loss_record['eval_max_l1_loss'] = torch.max(torch.stack(l1_losses)).item()
            loss_record['eval_min_l1_loss'] = torch.min(torch.stack(l1_losses)).item()
            loss_record['eval_mse_losses'] = mse_losses
            loss_record['eval_l1_losses'] = l1_losses
            pickle_save(os.path.join(log_dir, f'eval_loss_epoch_{epoch}.pkl'), loss_record)

        epoch_run_times.append(time.time() - epoch_start_time)

    pickle_save(os.path.join(log_dir, 'epoch_run_times.pkl'), epoch_run_times)
    model.save_model(os.path.join(checkpoint_dir, "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler_checkpoint.pth"))
    

    
    return
    
if __name__ == "__main__":
    main()