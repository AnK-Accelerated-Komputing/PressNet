# surrogateAI

Training, inference and evaluation code. Run everything from this folder (`cd surrogateAI`) so that `models` and `utilities` are importable.

## Data flow

```
HDF5 + meta.json ──► utilities/dataset.py ──► train.py ──► checkpoints (best validation L1)
                       (stage 1/2/3/4, skip-3)                    │
                                                                  ▼
                     inference.py  (press ► dwell ► release)  ──► rollout_stage_*.pkl, concatenated_rollout_all.pkl
                     inference_monolithic.py (one network)                         │
                                                                                   ▼
                                       utilities/evaluate_*.py, plot_*.py, animate_compare.py, tools/paraview
```

## Training: `train.py`

```bash
python train.py --config ../configs/train/<arch>.json --stage {1,2,3,4} [--train_val_data_path ...] [--output_dir ...]
```

| Flag | Meaning |
|---|---|
| `--config` | JSON with defaults; every key can be overridden on the command line |
| `--model_name` | `gcn`, `regDGCNN_seg`, `encode_process_decode`, `dilated_dgcnn`, `transolver` |
| `--stage` | 1 press, 2 dwell, 3 release, 4 whole trajectory (monolithic baseline) |
| `--k_neighbor`, `--dilated_k_sample`, `--message_passing_steps`, `--slice`, `--edge_dim` | architecture hyperparameters (only the ones used by the chosen model matter) |
| `--epochs`, `--learning_rate`, `--scheduler_type`, `--T_0`, `--T_mult`, `--eta_min`, `--patience`, `--delta` | optimisation (paper values are the defaults in `configs/train`) |
| `--resume --checkpoint_dir <run>/checkpoint` | continue a run (model, optimiser, scheduler, loss history) |
| `--no_wandb` | disable Weights & Biases (metrics are always written to `<run>/log`) |

Output: `<output_dir>/<model>/stage<k>/<dataset>/<timestamp>/{checkpoint,log,rollout}`. Validation every 20 epochs runs a full autoregressive rollout of the stage window on the validation trajectories; `checkpoint/best_checkpoint/` holds the snapshot with the lowest validation L1 (this is the one used for the paper's results), `checkpoint/epoch_<n>/` the periodic ones.

Hyperparameters used in the paper:

| Model | Setting |
|---|---|
| MeshGraphNet | 20 message-passing steps, world-edge radius 30 (fixed in `press_model.py`) |
| Reg-DGCNN | k = 80 |
| Dilated-DGCNN | k = 20, k_dilated = 200 |
| Transolver | 64 slices, 8 layers (fixed in `press_model.py`) |
| GCN | `edge_dim` 8 |

## Inference

`inference.py` (staged) and `inference_monolithic.py` (single network); options are listed by `--help`. Key flags: `--config` or `--stage{1,2,3}_ckpt` + `--core_model`, `--data`, `--split {val,test}`, `--out`, `--one_step`, `--no_continuous`, `--evaluate`, `--animation`.

- **Rolled-out** (default): each step consumes the model's own previous prediction; stage k+1 starts from the predicted end state of stage k.
- **One-step** (`--one_step`): the ground-truth state is fed at every step; only the single learned update is scored.
- Die/handle nodes are never predicted: their positions are overwritten from the prescribed tool motion at every step.

Rollout pickles hold, per trajectory: `pred_pos`, `gt_pos`, `pred_stress`, `gt_stress`, `mesh_pos`, `cells`, `node_type`, `cur_positions`, `cur_velocities`.

## Utilities (`python -m utilities.<name> --help`)

| Module | Purpose |
|---|---|
| `evaluate_metrics` | pooled nMAE / nRMSE (y-displacement, stress) -> `journal_metrics_summary.json`, step-wise curves |
| `evaluate_metrics_xyz` | same per displacement component (x, y, z) and stress |
| `evaluate_stepwise` | per-trajectory step-averaged nRMSE -> `stepwise_metrics.json` (input of `plot_boxplots`) |
| `plot_boxplots` | box plots of step-wise nRMSE across runs (ablation figures) |
| `plot_stepwise_curves` | error-vs-step curves from the CSVs of `evaluate_metrics` |
| `plot_validation_loss` | validation loss of a training run |
| `analyze_timing` | inference time per trajectory and per step from `rollout_info.json` |
| `animate_compare` | ground truth vs prediction GIF |
| `dataset`, `press_eval`, `common` | data loading, rollout loop, node types / mesh helpers |

Metric definition: nMAE and nRMSE in percent over the blank (`NORMAL`) nodes and all time steps of a trajectory, normalised by the per-trajectory peak-to-peak range of the ground truth (floor 1e-6); tables report the mean over trajectories.

## Notes

- GPU is used if available; CPU works but is slow for the graph models.
- `pointnet2_ops_lib/` (farthest-point sampling CUDA op, from [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)) is optional; `dilated_dgcnn.py` falls back to a pure-PyTorch implementation.
