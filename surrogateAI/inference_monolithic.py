"""Single-network (monolithic baseline) inference.

One network, trained on the full 0-15 s trajectory (``train.py --stage 4``), rolls
out the whole trajectory on its own. This is the baseline that the staged
PressNet++ pipeline (``inference.py``) is compared against.

Example:
    python inference_monolithic.py --checkpoint ../checkpoints/transolver_coarse/monolithic \\
        --core_model transolver --data ../datasets/data/coarse/coarse_1500_train_val.h5 \\
        --split val --out ../outputs/infer_monolithic_val
"""
import argparse
import json
import os
import pickle
import time

import torch
from torch.utils.data import DataLoader

from models import press_model
from utilities import press_eval
from utilities.dataset import TrajectoryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture hyperparameters of the checkpoints are stored inside the checkpoint
# (the whole module is pickled); these are only needed to build the wrapper.
PARAMETERS = dict(
    field='world_pos',
    size=4,
    model=press_model,
    evaluator=press_eval,
    k_neighbor=20,
    dilated_k_sample=200,
    message_passing_steps=20,
    slice=64,
    edge_dim=8,
)


def save_rollout(rollout_path, trajectories):
    with open(rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Rollout saved to {rollout_path}")


def load_model(core_model, checkpoint_dir):
    """Load `best_model_checkpoint` (preferred) or `epoch_model_checkpoint` from `checkpoint_dir`."""
    model = press_model.Model(PARAMETERS, core_model_name=core_model)

    candidates = [
        os.path.join(checkpoint_dir, "best_model_checkpoint"),
        os.path.join(checkpoint_dir, "epoch_model_checkpoint"),
        os.path.join(checkpoint_dir, "best_checkpoint", "best_model_checkpoint"),
    ]
    for prefix in candidates:
        if os.path.exists(prefix + "_learned_model.pth"):
            print(f"Loading weights from: {prefix}")
            model.load_model(prefix)
            return model.to(device).eval()
    raise FileNotFoundError(f"No best/epoch checkpoint found in {checkpoint_dir}")


def inference(checkpoint_dir, data_path, rollout_dir, core_model, split='val', teacher_forcing=False):
    os.makedirs(rollout_dir, exist_ok=True)

    # stage=4 -> the whole trajectory, not one motion regime
    dataset = TrajectoryDataset(data_path, split=split, stage=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_model(core_model, checkpoint_dir)

    info = {'index': [], 'inference_time': []}
    trajectories = []
    print(f"Starting inference on {len(loader)} trajectories...")

    with torch.no_grad():
        for idx, data in enumerate(loader):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _, prediction = press_eval.evaluate(model, data, teacher_forcing=teacher_forcing)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            trajectories.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                                 for k, v in prediction.items()})
            info['index'].append(idx)
            info['inference_time'].append(elapsed)
            print(f"Trajectory {idx}: {elapsed * 1000:.1f} ms")

    with open(os.path.join(rollout_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    save_rollout(os.path.join(rollout_dir, 'concatenated_rollout_all.pkl'), trajectories)


def main():
    parser = argparse.ArgumentParser(description='Monolithic (single-network) PressNet++ baseline inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Checkpoint directory of the monolithic model')
    parser.add_argument('--core_model', required=True,
                        choices=['gcn', 'encode_process_decode', 'regDGCNN_seg', 'regpointnet_seg',
                                 'transolver', 'dilated_dgcnn'])
    parser.add_argument('--data', required=True, help='HDF5 file (with *_meta.json next to it)')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--one_step', action='store_true', help='One-step protocol (ground truth fed every step)')
    args = parser.parse_args()

    inference(args.checkpoint, args.data, args.out, args.core_model, args.split, args.one_step)


if __name__ == '__main__':
    main()
