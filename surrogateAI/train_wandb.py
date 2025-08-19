import os
from pathlib import Path
import pickle
import time
import datetime
import gc

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb

from utilities import press_eval, common
from utilities.dataset import TrajectoryDataset
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
    loss = torch.mean(error[loss_mask])
    return loss

def prepare_files_and_directories(output_dir, model_name, train_data_path):
    '''
        Create necessary files and directories for the run
    '''
    train_data = train_data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(output_dir, model_name, train_data)
    run_create_time = time.time()
    run_create_datetime = datetime.datetime.fromtimestamp(run_create_time).strftime('%c')
    run_create_datetime_datetime_dash = run_create_datetime.replace(" ", "-").replace(":", "-")
    run_dir = os.path.join(output_dir, run_create_datetime_datetime_dash)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

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

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """Load model, optimizer, scheduler, and epoch from checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_checkpoint.pth")
    model_path = os.path.join(checkpoint_dir, "epoch_model_checkpoint_learned_model.pth")
    optimizer_path = os.path.join(checkpoint_dir, "epoch_optimizer_checkpoint.pth")
    scheduler_path = os.path.join(checkpoint_dir, "epoch_scheduler_checkpoint.pth")
    loss_record_path = os.path.join(checkpoint_dir, "../log/temp_train_loss.pkl")

    start_epoch = 0
    epoch_training_losses = []
    step_training_losses = []

    if os.path.exists(checkpoint_path) and os.path.exists(model_path):
        try:
            epoch_model_checkpoint_path = os.path.join(checkpoint_dir, "epoch_model_checkpoint")
            model.load_model(epoch_model_checkpoint_path)
            print(f"Loaded model checkpoint")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                print(f"Loaded optimizer checkpoint from {optimizer_path}")

            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                print(f"Loaded scheduler checkpoint from {scheduler_path}")

            if os.path.exists(loss_record_path):
                loss_record = pickle_load(loss_record_path)
                epoch_training_losses = loss_record.get('train_epoch_losses', [])
                step_training_losses = loss_record.get('all_step_train_losses', [])
                print(f"Loaded loss records from {loss_record_path}")

        except Exception as e:
            print(f"Error loading checkpoints: {e}. Starting from scratch.")
            start_epoch = 0
            epoch_training_losses = []
            step_training_losses = []
    else:
        print("No checkpoints found. Starting from scratch.")

    return start_epoch, epoch_training_losses, step_training_losses

def main():
    device = torch.device('cuda')

    # Initialize Wandb
    wandb.init(project="Early Stopping Checking", config={
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 1,
        "model": "MeshGraphNet",
        "dataset": "quarter s dataset: 400 step coarse",
        # "message_passing_step": "3",
        # "dropout": "0.4",
        "scheduler": "CosineAnnealingWarmRestarts",  # Updated config
        "T_0": 50,  # Initial cycle length
        "T_mult": 1,  # Cycle length multiplier
        "eta_min": 1e-6  # Minimum learning rate
    })

    start_epoch = 0
    start_time = time.time()
    end_epoch = 1000
    print(f"starting training from epoch {start_epoch} to {end_epoch}")
    train_data_path = "/home/ujwal/NEWPRESSNET/PressNet/Local/data/input/quarter_s_press_dataset.h5"
    output_dir = "/home/ujwal/NEWPRESSNET/PressNet/Local/data/output_model_eval"
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
    core_model = 'encode_process_decode'
    model = press_model.Model(params,core_model_name=core_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=50,  # Cycle length in epochs
    #     eta_min=1e-6  # Minimum learning rate
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Initial cycle length in epochs
        T_mult=1,  # Multiplier for cycle length after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.001,
    #     total_steps=(end_epoch) * len(train_dataloader),
    #     pct_start=0.3,
    #     anneal_strategy='cos',
    #     cycle_momentum=True,
    #     base_momentum=0.85,
    #     max_momentum=0.95,
    #     div_factor=25.0,
    #     final_div_factor=10000.0
    # )
    

    resume_from_existing = False

    if resume_from_existing:
        checkpoint_dir = '/home/ujwal/NEWPRESSNET/PressNet/Local/data/output/encode_process_decode/Channel_U_press_dataset/Fri-Jul-25-11-07-32-2025/checkpoint'
        log_dir = '/home/ujwal/NEWPRESSNET/PressNet/Local/data/output/encode_process_decode/Channel_U_press_dataset/Fri-Jul-25-11-07-32-2025/log'
        rollout_dir = '/home/ujwal/NEWPRESSNET/PressNet/Local/data/output/encode_process_decode/Channel_U_press_dataset/Fri-Jul-25-11-07-32-2025/rollout'
    else:   
        checkpoint_dir, log_dir, rollout_dir = prepare_files_and_directories(output_dir, core_model, train_data_path)
    
    start_epoch, epoch_training_losses, step_training_losses = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)

    
    # --- New: Early Stopping and Best Checkpoint Variables ---
    best_val_loss = float('inf')  # Track best validation loss
    patience = 100  # Number of epochs to wait for improvement
    patience_counter = 0  # Counter for epochs without improvement
    delta = 0.001  # Minimum improvement required to reset patience
    early_stop = False  # Flag to stop training
    
 
    epoch_run_times = []

    for epoch in range(start_epoch, end_epoch):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print(f"Running epoch {epoch+1}")
        epoch_start_time = time.time()

        epoch_training_loss = 0.0
        num_step = 0

        print("Training")
        model.train()
        for data in train_dataloader:
            frame = squeeze_data_frame(data)
            output = model(frame, is_training=True)
            loss = loss_fn(frame, output, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_training_losses.append(loss.detach().cpu())
            epoch_training_loss += loss.detach().cpu()
            num_step += 1
            wandb.log({"step_train_loss": loss.item()})

        mean_epoch_training_loss = epoch_training_loss / num_step
        epoch_training_losses.append(epoch_training_loss)
        print(f"Epoch {epoch+1} training loss: {epoch_training_loss}, time taken: {time.time() - epoch_start_time}")
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | "
              f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": epoch_training_loss.item(),
            "average_epoch_train_loss": mean_epoch_training_loss.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        })

        scheduler.step()

        loss_record = {}
        loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
        loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
        loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
        loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
        loss_record['train_epoch_losses'] = epoch_training_losses
        loss_record['all_step_train_losses'] = step_training_losses

        if epoch % 50 == 0:
            temp_train_loss_pkl_file = os.path.join(log_dir, 'temp_train_loss.pkl')
            Path(temp_train_loss_pkl_file).touch()
            pickle_save(temp_train_loss_pkl_file, loss_record)

        if epoch % 250 == 0:
            pickle_save(temp_train_loss_pkl_file.replace(".pkl", f'_{epoch}.pkl'), loss_record)

        if epoch % 50 == 0:
            model.save_model(os.path.join(checkpoint_dir, "epoch_model_checkpoint"))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "epoch_optimizer_checkpoint.pth"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "epoch_scheduler_checkpoint.pth"))
            torch.save({'epoch': epoch}, os.path.join(checkpoint_dir, "epoch_checkpoint.pth"))
        
        trajectories = []
        mse_losses = []
        l1_losses = []
        masked_losses = []
        save_file = f"rollout_epoch_{epoch}.pkl"

        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        print("Evaluation")
        model.eval()
        with torch.no_grad():
            num_data = 0
            masked_losses_sum = 0.0
            for data in val_loader:
                data = squeeze_data(data)
                _, prediction_trajectory = press_eval.evaluate(model, data)
                mse_loss = mse_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                l1_loss = l1_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
                trajectories.append(prediction_trajectory)

                pred = prediction_trajectory['pred_pos']
                target = torch.squeeze(data['next_pos'].to(device), dim=0)
                squared_error = torch.sum((target - pred) ** 2, dim=1)
                node_type = data['node_type'].to(device)
                loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
                loss_mask = loss_mask.squeeze(1)
                masked_loss = torch.mean(squared_error[loss_mask])
                masked_losses_sum += masked_loss.item()
                num_data += len(data["next_pos"])
                
        mean_masked_losses = masked_losses_sum / num_data
        if epoch % 50 == 0:
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

        if epoch % 50 == 0:
            pickle_save(os.path.join(log_dir, f'eval_loss_epoch_{epoch}.pkl'), loss_record)

        wandb.log({
            "epoch": epoch + 1,
            "eval_mean_mse_loss": loss_record['eval_mean_mse_loss'],
            "eval_mean_l1_loss": loss_record['eval_mean_l1_loss'],
            "mean_masked_loss": mean_masked_losses
        })

        current_val_loss = loss_record['eval_mean_l1_loss']
        if current_val_loss < best_val_loss - delta:
            print(f"New best validation loss: {current_val_loss:.6f} (previous: {best_val_loss:.6f})")
            best_val_loss = current_val_loss
            patience_counter = 0
            model.save_model(os.path.join(checkpoint_dir, "best_model_checkpoint"))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "best_optimizer_checkpoint.pth"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "best_scheduler_checkpoint.pth"))
            torch.save({'epoch': epoch}, os.path.join(checkpoint_dir, "best_epoch_checkpoint.pth"))
            print(f"Saved best checkpoint at epoch {epoch+1}")
            pickle_save(os.path.join(rollout_dir, save_file), trajectories)
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping: No improvement in validation loss for {patience} epochs")
            early_stop = True

        epoch_run_times.append(time.time() - epoch_start_time)
        pickle_save(os.path.join(log_dir, 'epoch_run_times_upto_epoch.pkl'), epoch_run_times)

    pickle_save(os.path.join(log_dir, 'epoch_run_times.pkl'), epoch_run_times)
    model.save_model(os.path.join(checkpoint_dir, "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler_checkpoint.pth"))
    
    wandb.finish()

if __name__ == "__main__":
    main()