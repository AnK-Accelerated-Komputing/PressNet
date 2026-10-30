# Reproducing the paper

All paths relative to the repository root; commands that start with `python` run in `surrogateAI/`. `<M>` is one of `gcn reg_dgcnn meshgraphnet dilated_dgcnn transolver`, `<mesh>` is `coarse` or `fine`.

Prerequisite: dataset ([datasets/README.md](../datasets/README.md)) and either the released checkpoints ([checkpoints/README.md](../checkpoints/README.md)) or your own training run.

| Paper item | What to run |
|---|---|
| **Train all models** (Appendix, Training procedures) | `bash scripts/train_all_stages.sh <M> <mesh>` |
| **In-distribution benchmark**, per-stage and overall nMAE / nRMSE | `bash scripts/evaluate_all_splits.sh <M> <mesh>`, read `outputs/eval/<mesh>/<M>/in_distribution/evaluation/journal_metrics_summary.json` |
| **Staged vs monolithic** | train the monolithic model with `STAGES="4" bash scripts/train_all_stages.sh <M> coarse`, roll it out with `inference_monolithic.py`, score both with `python -m utilities.evaluate_metrics --rollout ...` |
| **Hyperparameter ablations** | train with the swept flag, e.g. `python train.py --config ../configs/train/meshgraphnet.json --stage 1 --message_passing_steps 10`; then `python -m utilities.evaluate_stepwise --rollout ...` and `python -m utilities.plot_boxplots --metrics 10=... --metrics 20=...`. Swept ranges: MeshGraphNet steps {3,5,10,14,20,25}, Reg-DGCNN k {20,...,120}, Dilated-DGCNN k {20,50,80} and k_dilated {70,...,400}, Transolver slices {32,...,256} |
| **Extrapolation and unseen-shape splits** | same script, outputs in `.../extrapolation/` and `.../unseen/` |
| **One-step vs rolled-out** | add `--one_step` to `inference.py` (`bash scripts/evaluate_all_splits.sh <M> <mesh> --one_step`) |
| **Per-step error curves** (dynamic nRMSE figures) | `python -m utilities.plot_stepwise_curves --csv_dir <eval>/plot_data_csvs --out figures --transitions 500 1000` |
| **Directional components** (Appendix) | `python -m utilities.evaluate_metrics_xyz --rollout <...>/concatenated_rollout_all.pkl` |
| **Mesh resolution and cross-resolution transfer** | train on `fine`; for transfer, use a coarse inference config with a fine `--data` file (and vice versa) |
| **Timing / compute table** | `rollout_info.json` from `inference.py`, then `python -m utilities.analyze_timing -i <...>/rollout_info.json -w 3` (first 3 trajectories dropped as warm-up) |
| **Qualitative figures** | `python -m utilities.animate_compare ...` for GIFs; ParaView figures: `python tools/paraview/convert_pkls_to_vtu.py --model NAME=<pkl> --out vtu` then `pvbatch tools/paraview/paraview_render.py --vtu-root vtu --out-root figures` |
| **Transolver slice figures** | `python tools/slice_visualisation/slice_visualisation.py export --press-ckpt <dir> --dwell-ckpt <dir> --release-ckpt <dir> --rollout <pkl> --out slices`, then `select` / `render` ([details](../tools/slice_visualisation/README.md)) |
| **Thermal benchmark** | temperature-network code: to be added |

Notes on determinism: the train/validation split uses a fixed seed (42) over the group order of the HDF5 file; training itself is not seeded, so retrained checkpoints will differ slightly from the released ones. Use the released checkpoints to reproduce the reported numbers exactly.
