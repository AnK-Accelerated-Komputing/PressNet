# Pretrained checkpoints

One checkpoint per (mesh, architecture, stage). A staged model is the set of its three stage checkpoints; the monolithic baselines were trained on the full trajectory.

## Download

| Mesh | Architecture | Press / Dwell / Release | Monolithic |
|---|---|---|---|
| coarse | GCN | *(coming soon)* | n/a |
| coarse | Reg-DGCNN | *(coming soon)* | *(coming soon)* |
| coarse | MeshGraphNet | *(coming soon)* | *(coming soon)* |
| coarse | Dilated-DGCNN | *(coming soon)* | *(coming soon)* |
| coarse | Transolver | *(coming soon)* | *(coming soon)* |
| fine | GCN, Reg-DGCNN, MeshGraphNet, Dilated-DGCNN, Transolver | *(coming soon)* | not trained |

All-in-one archive: *(coming soon)*

## Layout

Unpack so that the paths in `configs/inference/*.json` resolve:

```
checkpoints/
└── <mesh>/                       # coarse | fine
    └── <model>/                  # gcn | reg_dgcnn | meshgraphnet | dilated_dgcnn | transolver
        ├── stage1/               # press
        ├── stage2/               # dwell
        ├── stage3/               # release
        └── monolithic/           # single network (coarse only)
```

Each stage folder holds the best-validation snapshot written by `train.py`:

```
best_model_checkpoint_learned_model.pth
best_model_checkpoint_output_normalizer.pth
best_model_checkpoint_stress_output_normalizer.pth
best_model_checkpoint_node_normalizer.pth
best_model_checkpoint_node_dynamic_normalizer.pth
best_model_checkpoint_mesh_edge_normalizer.pth
best_model_checkpoint_world_edge_normalizer.pth
```

(`inference.py` falls back to `epoch_model_checkpoint_*` if no `best_*` files are present.)

## Notes

- The network and the running-statistics normalisers are saved with `torch.save` of the whole module, so loading needs the `models/` package of this repository importable (run from `surrogateAI/`) and `torch.load(..., weights_only=False)`, which the code already does. Only load checkpoints from sources you trust.
- A checkpoint must be used with the architecture it was trained with (`core_model` in the inference config).
- Cross-resolution transfer (paper, mesh analysis) just uses a coarse config with `--data` pointing at a fine file, and vice versa.
