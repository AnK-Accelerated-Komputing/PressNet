import os
import time
import torch
import datetime
import pickle
from pathlib import Path
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, get_rank
from torch.amp import GradScaler, autocast

from utilities import press_eval, common
from utilities.dataset import TrajectoryDataset
from models import press_model


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
    

def ddp_setup(rank, world_size):
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def prepare_dirs(output_dir, model_num, train_data_path):
    train_data = train_data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(output_dir, str(model_num), train_data)
    run_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir = os.path.join(run_dir, 'checkpoint')
    log_dir = os.path.join(run_dir, 'log')
    rollout_dir = os.path.join(run_dir, 'rollout')
    for d in [checkpoint_dir, log_dir, rollout_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, log_dir, rollout_dir


def loss_fn(inputs, outputs, model, device):
    world_pos = inputs['curr_pos'].to(device)
    target_pos = inputs['next_pos'].to(device)
    target_velocity = target_pos - world_pos

    node_type = inputs['node_type'].to(device)
    norm = model.get_output_normalizer()
    target_norm = norm(target_velocity)
    pred = outputs[:, :3]

    loss_mask = (node_type[:, 0] == common.NodeType.NORMAL.value)
    loss = ((target_norm - pred) ** 2).sum(dim=1)
    return loss[loss_mask].mean()


def train(rank, world_size):
    start_epoch = 0
    start_time = time.time()
    end_epoch = 500
    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Dataset
    data_path = "/home/ubuntu/ujwal/PressNetTest/PressNet/surrogateAI/data/conical_press_dataset.h5"
    train_dataset = TrajectoryDataset(data_path, split='train', stage=1)
    val_dataset = TrajectoryDataset(data_path, split='val', stage=1)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    params = dict(field='world_pos', size=3, model=press_model, evaluator=press_eval)
    core_model='regDGCNN_seg'
    model = press_model.Model(params, core_model_name=core_model).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 + 1e-6)

    checkpoint_dir, log_dir, rollout_dir = prepare_dirs("training_output", "regDGCNN_seg", data_path)
    
    # Mixed precision with GradScaler
    scaler = GradScaler()
    
    epoch_training_losses = []
    step_training_losses = []
    epoch_run_times = []

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        epoch_start_time = start_time
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            frame = squeeze_data_frame(batch)
            batch = {k: v.squeeze(0) for k, v in batch.items()}
            # output = model(batch, is_training=True)
            # zero gradients
            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                outputs = model(batch, is_training=True)
                loss = loss_fn(batch, outputs, model.module, device)
            
            step_training_losses.append(loss.detach().cpu())
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.detach().cpu()
        epoch_training_losses.append(total_loss)
        
        scheduler.step()
        
        # Logging and saving at only rank 0
        if get_rank() == 0:
            train_epoch_time_taken = time.time() - epoch_start_time
            print(f"Epoch {epoch} | Evaluation Loss: {total_loss:.4f} | Training Time: {train_epoch_time_taken:.2f}s")

            
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
            
            model.module.save_model(os.path.join(checkpoint_dir,"epoch_model_checkpoint"))
            torch.save(optimizer.state_dict(),os.path.join(checkpoint_dir,"epoch_optimizer_checkpoint" + ".pth"))
            torch.save(scheduler.state_dict(),os.path.join(checkpoint_dir,"epoch_scheduler_checkpoint" + ".pth"))
            torch.save({'epoch': epoch}, os.path.join(checkpoint_dir, "epoch_checkpoint.pth"))
            
            trajectories = []
            mse_losses = []
            l1_losses = []
            save_file = "rollout_epoch_" + str(epoch) + ".pkl"
            
            mse_loss_fn = torch.nn.MSELoss()
            l1_loss_fn = torch.nn.L1Loss()
            
            for batch in val_loader:
                
                # frame = squeeze_data_frame(batch)
                batch = {k: v.squeeze(0) for k, v in batch.items()}
                
                _, prediction_trajectory = press_eval.evaluate(model, batch)
                mse_loss = mse_loss_fn(torch.squeeze(batch['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                l1_loss = l1_loss_fn(torch.squeeze(batch['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
                trajectories.append(prediction_trajectory)
                
                validation_epoch_time_taken = time.time() - epoch_start_time - train_epoch_time_taken
                print(f"Epoch {epoch} | Validation MSE Loss: {mse_loss:.4f} | Validation Time: {validation_epoch_time_taken:.2f}s")

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
    # Save model on rank 0
    if get_rank() == 0:
        model.module.save_model(os.path.join(checkpoint_dir, "final_model"))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pth"))

    # Validation
    
    # Finally destroy the process group
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
