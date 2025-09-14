import os
from pathlib import Path
import pickle
import time
import datetime
import gc
import argparse
import json
import shutil

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
    target_stress = inputs['next_stress'].to(device)   ## need to check here
    
    cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - cur_position

    node_type = inputs['node_type'].to(device)

    world_pos_normalizer, stress_normalizer = model.get_output_normalizer()
    
    target_normalized = world_pos_normalizer(target_velocity)
    target_stress_normalized = stress_normalizer(target_stress)

    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    pos_prediction = network_output[:,:3]
    stress_prediction = network_output[:,3:4]

    error_pos = torch.sum((target_normalized - pos_prediction) ** 2, dim=1)
    error_stress = torch.sum((target_stress_normalized - stress_prediction) ** 2, dim=1)
    loss = torch.mean(error_pos[loss_mask])+torch.mean(error_stress)
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
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters from a config file or command line")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON config file (default: config.json)")
    
    # Parse known args to get config file path
    args, _ = parser.parse_known_args()

    # Load config file
    config = {}
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default argument values.")
    except json.JSONDecodeError:
        print(f"Error parsing {args.config}. Using default argument values.")

    # Define arguments with config file values as defaults (if available)
    parser.add_argument("--train_data_path", type=str,
                        default=config.get("train_data_path", "/home/ujwal/NEWPRESSNET/PressNet/Local/data/input/quarter_s_press_dataset.h5"),
                        help="Path to the training dataset HDF5 file")
    parser.add_argument("--output_dir", type=str,
                        default=config.get("output_dir", "/home/ujwal/NEWPRESSNET/PressNet/Local/data/output"),
                        help="Directory to store output files (checkpoints, logs, rollouts)")
    parser.add_argument("--model_name", type=str,
                        default=config.get("model_name", "gcn"), choices=["gcn", "encode_process_deocde","regDGCNN_seg","transolver"],
                        help="Model type to use:")
    parser.add_argument("--epochs", type=int,
                        default=config.get("epochs", 1000),
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float,
                        default=config.get("learning_rate", 0.0001),
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int,
                        default=config.get("batch_size", 1),
                        help="Batch size for training and validation")
    parser.add_argument("--scheduler_type", type=str,
                        default=config.get("scheduler_type", "CosineAnnealingWarmRestarts"),
                        choices=["CosineAnnealingWarmRestarts", "CosineAnnealingLR"],
                        help="Type of learning rate scheduler")
    parser.add_argument("--T_0", type=int,
                        default=config.get("T_0", 50),
                        help="Initial cycle length for CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--T_mult", type=int,
                        default=config.get("T_mult", 1),
                        help="Cycle length multiplier for CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--eta_min", type=float,
                        default=config.get("eta_min", 0.000001),
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--patience", type=int,
                        default=config.get("patience", 100),
                        help="Patience for early stopping")
    parser.add_argument("--resume", action="store_true",
                        default=config.get("resume", False),
                        help="Resume training from existing checkpoint")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=config.get("checkpoint_dir", None),
                        help="Directory containing checkpoints to resume from (if --resume is set)")
    parser.add_argument("--wandb_project", type=str,
                        default=config.get("wandb_project", "Making Metric Same"),
                        help="Wandb project name for logging")
    parser.add_argument("--shuffle", action="store_true",
                        default=config.get("shuffle", True),
                        help="Shuffle the dataset during training")
    parser.add_argument("--stage", type=int,
                        default=config.get("stage", 1),
                        help="Stage value for TrajectoryDataset (default: 1)")
    parser.add_argument("--delta", type=float,
                        default=config.get("delta", 0.001),
                        help="Minimum improvement in validation loss for early stopping (default: 0.001)")

    args = parser.parse_args()

    # Validate resume and checkpoint_dir
    if args.resume and not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required when --resume is set")

    # Initialize Wandb
    wandb.init(project=args.wandb_project, config={
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": args.model_name,
        "dataset": args.train_data_path.split("/")[-1].split(".")[0],
        "scheduler": args.scheduler_type,
        "T_0": args.T_0,
        "T_mult": args.T_mult,
        "eta_min": args.eta_min,
        "Shuffle": args.shuffle,
        "stage": args.stage,
        "delta": args.delta
    })

    start_epoch = 0
    start_time = time.time()
    end_epoch = args.epochs
    print(f"Starting training from epoch {start_epoch} to {end_epoch}")
    train_dataset = TrajectoryDataset(args.train_data_path, split='train', stage=args.stage)
    val_dataset = TrajectoryDataset(args.train_data_path, split='val', stage=args.stage)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    params = dict(field='world_pos', size=4, model=press_model, k_neighbor = 10, dilated_k_sample = 20, evaluator=press_eval)
    model = press_model.Model(params, core_model_name=args.model_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min
        )
    elif args.scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_0,
            eta_min=args.eta_min
        )

    if args.resume:
        checkpoint_dir = args.checkpoint_dir
        log_dir = os.path.join(checkpoint_dir, "../log")
        rollout_dir = os.path.join(checkpoint_dir, "../rollout")
    else:
        checkpoint_dir, log_dir, rollout_dir = prepare_files_and_directories(args.output_dir, args.model_name, args.train_data_path)
    
    start_epoch, epoch_training_losses, step_training_losses = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)

        # Copy config.json to log_dir
    config_copy_path = os.path.join(log_dir, "config.json")
    try:
        shutil.copy2(args.config, config_copy_path)
        print(f"Copied config file from {args.config} to {config_copy_path}")
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found. Skipping copy.")
    except Exception as e:
        print(f"Error copying config file to {config_copy_path}: {e}")

    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    delta = args.delta
    early_stop = False
    
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
        mse_losses_stress = []
        l1_losses_stress = []
        masked_losses = []
        save_file = f"rollout_epoch_{epoch}.pkl"

        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        print("Evaluation")
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = 0.0
            num_step = 0
            for data in val_loader:
                data = squeeze_data_frame(data)
                result_list = []
                
                for i in range(data['cells'].shape[0]):
                    # Extract the i-th slice for each key and squeeze the first dimension
                    result_list.append({
                        'cells': data['cells'][i].squeeze(0),       # Shape [953, 4]
                        'mesh_pos': data['mesh_pos'][i].squeeze(0), # Shape [446, 3]
                        'node_type': data['node_type'][i].squeeze(0), # Shape [446, 1]
                        'curr_pos': data['curr_pos'][i].squeeze(0), # Shape [446, 3]
                        'next_pos': data['next_pos'][i].squeeze(0),  # Shape [446, 3]
                        'next_stress': data['next_stress'][i].squeeze(0) # Shape [446, 1]
                    })
            
                
                for step_input in result_list:
                    frame = squeeze_data_frame(step_input)
                    output = model(frame, is_training=True)
                    loss = loss_fn(frame, output, model)
                    epoch_validation_loss += loss.detach().cpu()
                    num_step += 1

            mean_epoch_validation_loss = epoch_validation_loss / num_step    
 
        with torch.no_grad():
            num_data = 0
            masked_losses_sum = 0.0
            for data in val_loader:
                data = squeeze_data(data)
                _, prediction_trajectory = press_eval.evaluate(model, data)
                ## right now stress is not considered here , can add code later
                mse_loss_pos = mse_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                l1_loss_pos = l1_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                mse_loss_stress = mse_loss_fn(torch.squeeze(data['next_stress'].to(device), dim=0), prediction_trajectory['pred_stress'])
                l1_loss_stress = l1_loss_fn(torch.squeeze(data['next_stress'].to(device), dim=0), prediction_trajectory['pred_stress'])
                
                mse_losses.append(mse_loss_pos.cpu())
                l1_losses.append(l1_loss_pos.cpu())
                mse_losses_stress.append(mse_loss_stress.cpu())
                l1_losses_stress.append(l1_loss_stress.cpu())

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
        loss_record['eval_mse_losses_pos'] = mse_losses
        loss_record['eval_l1_losses_pos'] = l1_losses
        loss_record['eval_total_mse_loss_stress'] = torch.sum(torch.stack(mse_losses_stress)).item()
        loss_record['eval_total_l1_loss_stress'] = torch.sum(torch.stack(l1_losses_stress)).item()
        loss_record['eval_mean_mse_loss_stress'] = torch.mean(torch.stack(mse_losses_stress)).item()
        loss_record['eval_max_mse_loss_stress'] = torch.max(torch.stack(mse_losses_stress)).item()
        loss_record['eval_min_mse_loss_stress'] = torch.min(torch.stack(mse_losses_stress)).item()
        loss_record['eval_mean_l1_loss_stress'] = torch.mean(torch.stack(l1_losses_stress)).item()
        loss_record['eval_max_l1_loss_stress'] = torch.max(torch.stack(l1_losses_stress)).item()
        loss_record['eval_min_l1_loss_stress'] = torch.min(torch.stack(l1_losses_stress)).item()
        loss_record['eval_mse_losses_stress'] = mse_losses_stress
        loss_record['eval_l1_losses_stress'] = l1_losses_stress

        if epoch % 50 == 0:
            pickle_save(os.path.join(log_dir, f'eval_loss_epoch_{epoch}.pkl'), loss_record)

        wandb.log({
            "epoch": epoch + 1,
            "eval_mean_mse_loss": loss_record['eval_mean_mse_loss'],
            "eval_mean_l1_loss": loss_record['eval_mean_l1_loss'],
            "mean_masked_loss": mean_masked_losses,
            "mean_epoch_validation_loss": mean_epoch_validation_loss
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