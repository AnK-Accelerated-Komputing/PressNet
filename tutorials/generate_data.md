# Generating your own press-forming data

Pipeline: **ANSYS transient simulation -> text exports -> HDF5 + meta JSON** (`datasets/data_gen/`).

## 1. Raw ANSYS export layout

One folder per simulation (one parametric variation), exported from ANSYS Mechanical:

```
<raw_dir>/
└── <name>_data/
    ├── <name>.dat                          # solver input deck: nodes, elements, and `deltim,<dt>`
    ├── Directional_Deformation_X/Directional Deformation X <k>.txt     k = 2 ... 1500
    ├── Directional_Deformation_Y/Directional Deformation Y <k>.txt
    ├── Directional_Deformation_Z/Directional Deformation Z <k>.txt
    └── Equivalent_Stress/Equivalent Stress <k>.txt
```

Folder names must contain numbers (they are sorted by them). The reader (`generic_logic_dual_eblock.py`) parses the `.dat` file for nodes, tetrahedral cells and the time step, and classifies each node as blank (`NORMAL`), die (`OBSTACLE`) or handle from the displacement at the first and at the 1000th step (see `assign_body_id_to_nodes_generic`).

## 2. Convert to HDF5

```bash
python datasets/data_gen/generic_dataset_generate.py --raw_dir <raw_dir> --out_h5 datasets/data/coarse/p_full.h5
```

writes `p_full.h5` and `p_full_meta.json` (see [datasets/README.md](../datasets/README.md) for the layout).

## 3. Merge shapes into one dataset

```bash
python datasets/data_gen/merge_hf.py --h5 a.h5 b.h5 c.h5 --json a_meta.json b_meta.json c_meta.json \
    --out_h5 datasets/data/coarse/coarse_1500_train_val.h5
```

Groups are renumbered `group_0 ...`; the original file of each group is recorded under `group_mapping` in the merged meta JSON. Split the merged file into train/validation, extrapolation and unseen-shape files as described in the dataset README.

## 4. Check

```bash
python tutorials/inspect_h5.py datasets/data/coarse/coarse_1500_train_val.h5 --plot
cd surrogateAI && python -m utilities.dataset --data ../datasets/data/coarse/coarse_1500_train_val.h5 --split train --stage 1
```
