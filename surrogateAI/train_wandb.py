from logging import config
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    loss_stress = torch.mean(error_stress[loss_mask])
    loss_pos = torch.mean(error_pos[loss_mask])
    loss = loss_stress + loss_pos
    return loss, loss_stress, loss_pos

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
    loss_record = None

    if os.path.exists(checkpoint_path) and os.path.exists(model_path):
        try:
            epoch_model_checkpoint_path = os.path.join(checkpoint_dir, "epoch_model_checkpoint")
            model.load_model(epoch_model_checkpoint_path)
            print(f"Loaded model checkpoint")
            
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
                print(f"Loaded optimizer checkpoint from {optimizer_path}")

            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
                print(f"Loaded scheduler checkpoint from {scheduler_path}")

            if os.path.exists(loss_record_path):
                loss_record = pickle_load(loss_record_path)
                print(f"Loaded loss records from {loss_record_path}")

        except Exception as e:
            print(f"Error loading checkpoints: {e}. Starting from scratch.")
            start_epoch = 0
            loss_record = None
    else:
        print("No checkpoints found. Starting from scratch.")

    return start_epoch, loss_record

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
    parser.add_argument("--train_val_data_path", type=str,
                        default=config.get("train_val_data_path", "/home/sushil/PressNet/datasets/data/Fine/fine_1500_train_val.h5"),
                        help="Path to the training/validation dataset HDF5 file")
    parser.add_argument("--output_dir", type=str,
                        default=config.get("output_dir", "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn"),
                        help="Directory to store output files (checkpoints, logs, rollouts)")
    parser.add_argument("--model_name", type=str,
                        default=config.get("model_name", "dilated_dgcnn"), choices=["gcn", "encode_process_decode","regDGCNN_seg","transolver","dilated_dgcnn"],
                        help="Model type to use:")
    parser.add_argument("--epochs", type=int,
                        default=config.get("epochs", 1260),
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
                        default=config.get("T_0", 20),
                        help="Initial cycle length for CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--T_mult", type=int,
                        default=config.get("T_mult", 2),
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
                        default=config.get("wandb_project", "Server3_Dilated_C_ST1_K50_D70"),
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
    parser.add_argument("--k_neighbor", type=int,
                    default=config.get("k_neighbor", 50),
                    help="Number of nearest neighbors")
    parser.add_argument("--dilated_k_sample", type=int,
                    default=config.get("dilated_k_sample", 100),
                    help="Dilated convolution sampling parameter")
    parser.add_argument("--message_passing_steps", type=int,
                    default=config.get("message_passing_steps", 4),
                    help="Number of message passing steps in the model")
    parser.add_argument("--slice", type=int,
                    default=config.get("slice", 1),
                    help="Slice parameter for transolver model")
    parser.add_argument("--edge_dim", type=int,
                    default=config.get("edge_dim", 16),
                    help="Edge dimension for the GCN model")
    

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
        "dataset": args.train_val_data_path.split("/")[-1].split(".")[0],
        "scheduler": args.scheduler_type,
        "T_0": args.T_0,
        "T_mult": args.T_mult,
        "eta_min": args.eta_min,
        "Shuffle": args.shuffle,
        "stage": args.stage,
        "delta": args.delta,
        "k_neighbor": args.k_neighbor,
        "dilated_k_sample": args.dilated_k_sample,
        "message_passing_steps": args.message_passing_steps,
        "slice": args.slice,
        "edge_dim": args.edge_dim,
    })
    
    # Set x-axis to be Epochs
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    end_epoch = args.epochs
    
    train_dataset = TrajectoryDataset(args.train_val_data_path, split='train', stage=args.stage)
    val_dataset = TrajectoryDataset(args.train_val_data_path, split='val', stage=args.stage)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    params = dict(field='world_pos', size=4, model=press_model, k_neighbor = args.k_neighbor, dilated_k_sample = args.dilated_k_sample,
                    message_passing_steps = args.message_passing_steps, slice=args.slice, edge_dim=args.edge_dim, evaluator=press_eval)
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

    # --- Initialize Variables based on Resume State ---
    start_epoch = 0
    epoch_training_losses = []
    epoch_training_losses_stress = []
    epoch_training_losses_pos = []
    step_training_losses = []
    epoch_learning_rate = []
    epoch_run_times = []
    epoch_eval_losses = []

    if args.resume:
        checkpoint_dir = args.checkpoint_dir
        # Assuming log and rollout dirs follow the standard relative path structure
        log_dir = os.path.join(checkpoint_dir, "../log")
        rollout_dir = os.path.join(checkpoint_dir, "../rollout")
        
        # Load the checkpoint and the full history record
        start_epoch, loss_record = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)
        
        # Restore lists from loss_record if it exists
        if loss_record:
            epoch_training_losses = loss_record.get('train_epoch_losses', [])
            step_training_losses = loss_record.get('all_step_train_losses', [])
            epoch_learning_rate = loss_record.get('learning_rate', [])
            epoch_run_times = loss_record.get('epoch_run_times', [])
            epoch_training_losses_stress = loss_record.get('train_epoch_losses_stress', [])
            epoch_training_losses_pos = loss_record.get('train_epoch_losses_pos', [])
            epoch_eval_losses = loss_record.get('epoch_eval_losses', [])
            print(f"Restored history: {len(epoch_training_losses)} epochs found.")
    else:
        # Create new directories for a fresh run
        checkpoint_dir, log_dir, rollout_dir = prepare_files_and_directories(args.output_dir, args.model_name, args.train_val_data_path)
    
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
    
    # Eval metric lists
    mse_losses_pos = []
    mse_losses_stress = []
    l1_losses_pos = []
    l1_losses_stress = [] 
    
    print(f"Starting training from epoch {start_epoch} to {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        print(f"\n=================== Running epoch {epoch+1} ===================")
        epoch_start_time = time.time()

        epoch_training_loss = 0.0
        num_step = 0

        print("Training")
        model.train()
        epoch_training_loss_stress = 0.0
        epoch_training_loss_pos = 0.0
        
        for data_idx, data in enumerate(train_dataloader):
            print(f"\n--- Batch {data_idx+1} ---")
            frame = squeeze_data_frame(data)
            print(f"1. Data loaded - Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            
            output = model(frame, is_training=True)
            print(f"2. Forward pass done - Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            
            loss, loss_stress, loss_pos = loss_fn(frame, output, model)
            print(f"3. Loss calculated - Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            
            optimizer.zero_grad()
            loss.backward()
            print(f"4. Backward pass done - Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            
            optimizer.step()
            print(f"5. Optimizer step done - Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            
            # Detach to save memory and add to lists
            step_training_losses.append(loss.detach().cpu())
            epoch_training_loss += loss.detach().cpu()
            epoch_training_loss_stress += loss_stress.detach().cpu()
            epoch_training_loss_pos += loss_pos.detach().cpu()
            num_step += 1

        mean_epoch_training_loss = epoch_training_loss / num_step
        epoch_training_losses.append(epoch_training_loss)
        epoch_training_losses_stress.append(epoch_training_loss_stress)
        epoch_training_losses_pos.append(epoch_training_loss_pos)
        epoch_learning_rate.append(optimizer.param_groups[0]['lr'])
        epoch_run_time = time.time() - epoch_start_time
        epoch_run_times.append(epoch_run_time)

        print(f"Epoch {epoch+1} training loss: {epoch_training_loss}, time taken: {time.time() - epoch_start_time}")
        
        # Store training metrics in a dictionary
        log_dict = {
            "epoch": epoch + 1,
            "epoch_train_loss": epoch_training_loss.item(),
            "epoch_train_loss_stress": epoch_training_loss_stress.item(),
            "epoch_train_loss_pos": epoch_training_loss_pos.item(),
            "average_epoch_train_loss": mean_epoch_training_loss.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        }

        scheduler.step()

        loss_record = {
            'train_total_loss': torch.sum(torch.stack(epoch_training_losses)).item(),
            'train_mean_epoch_loss': torch.mean(torch.stack(epoch_training_losses)).item(),
            'train_max_epoch_loss': torch.max(torch.stack(epoch_training_losses)).item(),
            'train_min_epoch_loss': torch.min(torch.stack(epoch_training_losses)).item(),
            'train_epoch_losses': epoch_training_losses,
            'train_epoch_losses_stress': epoch_training_losses_stress,
            'train_epoch_losses_pos': epoch_training_losses_pos,
            'all_step_train_losses': step_training_losses,
            'learning_rate': epoch_learning_rate,
            'epoch_run_times': epoch_run_times,
            'epoch_eval_losses': epoch_eval_losses
        }

        if epoch % 50 == 0:
            temp_train_loss_pkl_file = os.path.join(log_dir, 'temp_train_loss.pkl')
            # Ensure file exists
            if not os.path.exists(temp_train_loss_pkl_file):
                Path(temp_train_loss_pkl_file).touch()
            pickle_save(temp_train_loss_pkl_file, loss_record)

        if epoch % 250 == 0:
            temp_train_loss_pkl_file = os.path.join(log_dir, 'temp_train_loss.pkl')
            pickle_save(temp_train_loss_pkl_file.replace(".pkl", f'_{epoch}.pkl'), loss_record)

        # Standard periodic save
        if epoch % 50 == 0:
            model.save_model(os.path.join(checkpoint_dir, "epoch_model_checkpoint"))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "epoch_optimizer_checkpoint.pth"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "epoch_scheduler_checkpoint.pth"))
            torch.save({'epoch': epoch}, os.path.join(checkpoint_dir, "epoch_checkpoint.pth"))
        
        # Evaluation condition: Every 20 epochs, plus first and last
        should_evaluate = (epoch + 1) % 20 == 0 or epoch == start_epoch or (epoch + 1) == end_epoch

        if should_evaluate:
            trajectories = []
            mse_losses = []
            l1_losses = []
            mse_losses_stress = []
            l1_losses_stress = []
            mse_losses_pos = []
            l1_losses_pos = []
            masked_losses = []
            save_file = f"rollout_epoch_{epoch + 1}.pkl"

            mse_loss_fn = torch.nn.MSELoss()
            l1_loss_fn = torch.nn.L1Loss()
            print(f"Evaluation at epoch {epoch+1}")
            model.eval()
            
            with torch.no_grad():
                num_data = 0
                masked_losses_sum = 0.0
                for data in val_loader:
                    data = squeeze_data(data)
                    _, prediction_trajectory = press_eval.evaluate(model, data)
                    
                    mse_loss_pos = mse_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                    l1_loss_pos = l1_loss_fn(torch.squeeze(data['next_pos'].to(device), dim=0), prediction_trajectory['pred_pos'])
                    mse_loss_stress = mse_loss_fn(torch.squeeze(data['next_stress'].to(device), dim=0), prediction_trajectory['pred_stress'])
                    l1_loss_stress = l1_loss_fn(torch.squeeze(data['next_stress'].to(device), dim=0), prediction_trajectory['pred_stress'])
                    mse_loss = mse_loss_pos + mse_loss_stress
                    l1_loss = l1_loss_pos + l1_loss_stress

                    mse_losses.append(mse_loss.cpu()) # Total
                    l1_losses.append(l1_loss.cpu()) 
                    mse_losses_pos.append(mse_loss_pos.cpu())
                    l1_losses_pos.append(l1_loss_pos.cpu())
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
            
            # Calculate current mean MSE and add to history
            current_eval_mean_mse = torch.mean(torch.stack(mse_losses)).item()
            current_eval_mean_mse_pos = torch.mean(torch.stack(mse_losses_pos)).item()
            current_eval_mean_mse_stress = torch.mean(torch.stack(mse_losses_stress)).item()
            epoch_eval_losses.append(current_eval_mean_mse)

            # Save checkpoint in a separate folder for this specific evaluation
            eval_checkpoint_dir = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
            Path(eval_checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            model.save_model(os.path.join(eval_checkpoint_dir, "epoch_model_checkpoint"))
            torch.save(optimizer.state_dict(), os.path.join(eval_checkpoint_dir, "epoch_optimizer_checkpoint.pth"))
            torch.save(scheduler.state_dict(), os.path.join(eval_checkpoint_dir, "epoch_scheduler_checkpoint.pth"))
            torch.save({'epoch': epoch}, os.path.join(eval_checkpoint_dir, "epoch_checkpoint.pth"))
            print(f"Saved specific checkpoint to {eval_checkpoint_dir}")
            
            # Save rollout for this eval
            pickle_save(os.path.join(rollout_dir, save_file), trajectories)

            # Update loss record with evaluation metrics
            eval_loss_record = {
                'eval_total_mse_loss': torch.sum(torch.stack(mse_losses)).item(),
                'eval_total_l1_loss': torch.sum(torch.stack(l1_losses)).item(),
                'eval_mean_mse_loss_pos': current_eval_mean_mse_pos,
                'eval_mean_mse_loss_stress': current_eval_mean_mse_stress,
                'eval_mean_mse_loss': torch.mean(torch.stack(mse_losses)).item(),
                'eval_max_mse_loss': torch.max(torch.stack(mse_losses)).item(),
                'eval_min_mse_loss': torch.min(torch.stack(mse_losses)).item(),
                'eval_mean_l1_loss': torch.mean(torch.stack(l1_losses)).item(),
                'eval_max_l1_loss': torch.max(torch.stack(l1_losses)).item(),
                'eval_min_l1_loss': torch.min(torch.stack(l1_losses)).item(),
                'eval_mse_losses': mse_losses,
                'eval_l1_losses': l1_losses,
                'epoch_eval_losses': epoch_eval_losses,
                'eval_mse_losses_stress': mse_losses_stress,
                'eval_l1_losses_stress': l1_losses_stress,      
                'eval_mse_losses_pos': mse_losses_pos,
                'eval_l1_losses_pos': l1_losses_pos,
            }

            # Save evaluation loss record
            pickle_save(os.path.join(log_dir, f'eval_loss_epoch_{epoch + 1}.pkl'), eval_loss_record)

            # Update the dictionary with validation metrics for WandB
            log_dict.update({
                "eval_mean_mse_loss": eval_loss_record['eval_mean_mse_loss'],
                "eval_mean_l1_loss": eval_loss_record['eval_mean_l1_loss'],
                "mean_masked_loss": mean_masked_losses,
                "epoch_eval_mse_loss": current_eval_mean_mse,
                "eval_mean_mse_loss_pos": eval_loss_record['eval_mean_mse_loss_pos'],
                "eval_mean_mse_loss_stress": eval_loss_record['eval_mean_mse_loss_stress'],
            })
            
            # --- Check for best model and save to dedicated 'best_checkpoint' folder ---
            current_val_loss = eval_loss_record['eval_mean_l1_loss']
            if current_val_loss < best_val_loss - delta:
                print(f"New best validation loss: {current_val_loss:.6f} (previous: {best_val_loss:.6f})")
                best_val_loss = current_val_loss
                patience_counter = 0
                
                # Create dedicated directory for the BEST checkpoint
                best_checkpoint_dir = os.path.join(checkpoint_dir, "best_checkpoint")
                Path(best_checkpoint_dir).mkdir(parents=True, exist_ok=True)
                
                # Save using fixed names, which ensures replacement/overwriting
                model.save_model(os.path.join(best_checkpoint_dir, "best_model_checkpoint"))
                torch.save(optimizer.state_dict(), os.path.join(best_checkpoint_dir, "best_optimizer_checkpoint.pth"))
                torch.save(scheduler.state_dict(), os.path.join(best_checkpoint_dir, "best_scheduler_checkpoint.pth"))
                torch.save({'epoch': epoch}, os.path.join(best_checkpoint_dir, "best_epoch_checkpoint.pth"))
                
                # Also save a small JSON file with the epoch number for reference
                with open(os.path.join(best_checkpoint_dir, "best_info.json"), "w") as f:
                    json.dump({"best_epoch": epoch + 1, "best_loss": current_val_loss}, f, indent=4)
                
                print(f"Saved BEST checkpoint to {best_checkpoint_dir}")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        # Log consolidated metrics
        wandb.log(log_dict, step=epoch + 1)
        
        pickle_save(os.path.join(log_dir, 'epoch_run_times_upto_epoch.pkl'), epoch_run_times)

    print("\nTraining completed! Saving final results...")
    # Save final comprehensive training loss record
    final_loss_record = {
        'train_total_loss': torch.sum(torch.stack(epoch_training_losses)).item() if epoch_training_losses else 0,
        'train_mean_epoch_loss': torch.mean(torch.stack(epoch_training_losses)).item() if epoch_training_losses else 0,
        'train_max_epoch_loss': torch.max(torch.stack(epoch_training_losses)).item() if epoch_training_losses else 0,
        'train_min_epoch_loss': torch.min(torch.stack(epoch_training_losses)).item() if epoch_training_losses else 0,
        'train_epoch_losses': epoch_training_losses,
        'all_step_train_losses': step_training_losses,
        'learning_rate': epoch_learning_rate,
        'epoch_run_times': epoch_run_times,
        'epoch_eval_losses': epoch_eval_losses
    }
    pickle_save(os.path.join(log_dir, 'epoch_run_times.pkl'), epoch_run_times)
    pickle_save(os.path.join(log_dir, 'final_train_loss.pkl'), final_loss_record)

    model.save_model(os.path.join(checkpoint_dir, "final_model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "final_optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "final_scheduler_checkpoint.pth"))

    wandb.finish()

if __name__ == "__main__":
    main()