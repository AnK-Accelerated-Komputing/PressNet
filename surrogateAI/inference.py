import os
import time
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import press_model
from utilities import press_eval
from utilities.dataset import TrajectoryDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters dictionary
PARAMETERS = {
    'press': dict(
        noise=0.003,
        gamma=1.0,
        field='world_pos',
        history=False,
        size=3,
        batch=1,
        model=press_model,
        evaluator=press_eval,
        loss_type='deform',
        stochastic_message_passing_used=False
    )
}

def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame

def save_rollout(rollout_path, trajectories, disp=""):
    """Save predicted trajectories to a pickle file."""
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Rollout {disp} saved to {rollout_path}")


def load_model(inference_config, params, checkpoint_dir):
    """Load the model from a checkpoint."""
    model = params['model'].Model(
        params,
        inference_config['core_model'],
        inference_config['message_passing_aggregator'],
        inference_config['message_passing_steps'],
        inference_config['attention']
    )
    model.load_model(os.path.join(checkpoint_dir, 'epoch_model_checkpoint'))
    model.to(device)
    model.eval()
    return model


def get_meta_data(data_path):
    """Load metadata JSON corresponding to the dataset."""
    base_folder = os.path.dirname(data_path)
    data_name = os.path.basename(data_path).split('.')[0].replace('_press_dataset','')
    meta_file = os.path.join(base_folder, f'{data_name}_meta.json')

    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    return meta_data


def inference(checkpoint_dir, val_dir, rollout_dir):
    """Run inference on dataset and save rollout."""
    os.makedirs(rollout_dir, exist_ok=True)

    # Inference configuration
    inference_config = {
        "model": "press",
        "core_model": "regDGCNN_seg",
        "message_passing_aggregator": "sum",
        "message_passing_steps": 15,
        "attention": False
    }

    # Load dataset and model
    val_dataset = TrajectoryDataset(val_dir, split='val', stage=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    params = PARAMETERS[inference_config['model']]
    model = load_model(inference_config, params, checkpoint_dir)

    meta_data = get_meta_data(val_dir)
    groups = sorted(list(meta_data.keys()))
    print('meta data is', meta_data)
    info = {
        'index': [],
        'file_name' : [],
        'inference_time' : [],
    }
    # Run inference
    trajectories = []
    for idx, data in enumerate(val_loader):
        info['index'].append(idx)
        group_name = groups[idx]
        info['file_name'].append(meta_data[group_name]['data type'])
        
        start_time = time.time()
        _, prediction_trajectory = press_eval.evaluate(model, data)
        prediction_trajectory = squeeze_data_frame(prediction_trajectory)
        print(prediction_trajectory['cells'].shape)
        trajectories.append(prediction_trajectory)
        end_time = time.time()

        info['inference_time'].append(end_time-start_time)

        print(f"Processed sample {idx} in {end_time - start_time:.2f}s")

    # Save predictions
    rollout_file = Path(rollout_dir) / "infer_rollout.pkl"
    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f,indent=4)

    save_rollout(str(rollout_file), trajectories)


def main():
    # Define directories
    checkpoint_dir = "/home/bipinshrestha228/PressNet/final_data_and_checkpoitns/ujwal_dai_checkpoints/checkpoint"
    val_dir = "/home/bipinshrestha228/PressNet/final_data_and_checkpoitns/final_correct_data/400_step_dataset_h5/Channel_U_press_dataset.h5"
    rollout_dir = "/home/bipinshrestha228/PressNet/conference_paper/testing"

    # Run inference
    inference(checkpoint_dir, val_dir, rollout_dir)


if __name__ == "__main__":
    main()
