#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_pkls_to_vtu.py
================================================================================
STEP 1 of 2  --  run this with your NORMAL python (the one that has torch +
meshio). It turns each model's rollout .pkl into a per-model folder of VTU
files plus a `series.pvd` time-series that ParaView can read.

STEP 2 is `paraview_render.py`, run with pvpython / pvbatch.

Why two steps?
    ParaView's bundled python cannot unpickle your torch tensors (no torch
    inside pvpython). So we unpickle + compute the fields here, write plain
    VTU/PVD, and let ParaView just render.

Pass the rollouts on the command line: --model NAME=path/to/rollout.pkl (repeatable).

--------------------------------------------------------------------------------
WHAT EACH VTU CONTAINS  (point data, per node, per timestep)
--------------------------------------------------------------------------------
Vectors (used by ParaView's WarpByVector to deform the mesh):
    gt_disp     = gt_pos   - mesh_pos        ground-truth displacement
    pred_disp   = pred_pos - mesh_pos        predicted displacement

Scalars (used by ParaView's ColorBy):
    disp_y_gt   = |gt_disp_y|     (set ABS_Y=False for signed)
    disp_y_pred = |pred_disp_y|
    disp_mag_gt    = ||gt_disp||
    disp_mag_pred  = ||pred_disp||
    pos_error   = ||pred_pos - gt_pos||      <-- the positional error you color
                                                  the error panel by

Stress scalars (only written if the pkl has gt_stress / pred_stress):
    gt_stress, pred_stress
    stress_error = |pred_stress - gt_stress|

Node identity scalars:
    node_type, is_die   -> let ParaView MOVE the die (real displacement is kept),
                           colour only the workpiece by the field, draw the die
                           as a solid colour, and drop the die on error panels.

Mesh points are written at mesh_pos (the undeformed reference); the GT/Pred
shapes are produced in ParaView by WarpByVector(gt_disp / pred_disp).
================================================================================
"""

import os
import pickle
import numpy as np

# meshio is only needed here (Step 1), not in ParaView.
import meshio

# torch is only needed to UNPICKLE the rollout tensors. If your pkls were saved
# as numpy already, the code still works (torch import becomes optional).
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ==============================================================================
# 1. USER SETTINGS  --  edit only this block
# ==============================================================================

# name -> path of that model's rollout pkl.  The name becomes the output folder
# name and the label you will see in ParaView, so keep it short & filesystem-safe.
MODEL_PKLS = {}   # filled from --model NAME=PKL

# Parent folder where per-model VTU folders are written. Point Step 2 (the
# ParaView script's VTU_ROOT) at this same folder.
OUTPUT_VTU_ROOT = "vtu_root"   # overridden by --out

# Which geometry/sample(s) inside each pkl to export. Each pkl holds several
# tested geometries; every geometry gets its OWN series.pvd folder so the
# ParaView script can give it its OWN error legend (fixed in time, per geometry).
#   an int        -> just that one geometry, written flat to <model>/series.pvd
#   a list/tuple  -> those geometries, each to <model>/sample_<i>/series.pvd
#   "all"         -> every geometry,      each to <model>/sample_<i>/series.pvd
SAMPLE_INDEX = "all"

# Field options
ABS_Y = True                 # color y-displacement by |y| (matches a 0..max bar).
OBSTACLE_NODE_TYPE = 1       # node_type value for the rigid tool/die nodes.
                             # Die displacement is KEPT (so the die moves); the
                             # 'is_die' tag lets ParaView separate die/workpiece.

# Time stamp written into the PVD. If you know seconds-per-step, set DT so
# ParaView shows real time; otherwise integer step index is used.
DT = None                    # e.g. 0.03  -> time = step * DT ; None -> step index


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def _np(x):
    """torch.Tensor / list / np -> float64 numpy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _squeeze_leading_one(a):
    """Drop a leading batch dim of size 1, e.g. (1, T, N, 3) -> (T, N, 3)."""
    a = np.asarray(a)
    while a.ndim > 1 and a.shape[0] == 1 and a.ndim > 3:
        a = a[0]
    if a.ndim >= 1 and a.shape[0] == 1 and a.ndim in (3, 4):
        a = a[0]
    return a


def _get(sample, *keys):
    """Return the first present key from a trajectory dict."""
    for k in keys:
        if k in sample:
            return sample[k]
    raise KeyError(f"none of {keys} found in sample keys: {list(sample.keys())}")


def extract_arrays(sample):
    """
    Pull the standard PressNet++ rollout fields out of one sample dict and
    return clean numpy arrays:
        mesh_pos (T,N,3), gt_pos (T,N,3), pred_pos (T,N,3),
        node_type (T,N), cells (Ncells,4), gt_stress (T,N)|None, pred_stress|None
    """
    gt_pos   = _squeeze_leading_one(_np(_get(sample, "gt_pos")))
    pred_pos = _squeeze_leading_one(_np(_get(sample, "pred_pos")))
    mesh_pos = _squeeze_leading_one(_np(_get(sample, "mesh_pos")))

    # node_type: (T,N,1) or (T,N) -> (T,N)
    node_type = _squeeze_leading_one(_np(_get(sample, "node_type")))
    if node_type.ndim == 3 and node_type.shape[-1] == 1:
        node_type = node_type[..., 0]

    # cells / connectivity. press_eval stores 'cells' as (1,T,Ncells,4)-ish.
    cells = _np(_get(sample, "cells"))
    cells = np.squeeze(cells)
    if cells.ndim == 3:          # (T, Ncells, 4) -> take t=0 (connectivity is constant)
        cells = cells[0]
    cells = cells.astype(np.int64)

    T = gt_pos.shape[0]
    # broadcast a single reference mesh across time if needed
    if mesh_pos.ndim == 2:
        mesh_pos = np.broadcast_to(mesh_pos, gt_pos.shape).copy()
    if node_type.ndim == 1:
        node_type = np.broadcast_to(node_type, gt_pos.shape[:2]).copy()

    # ---- optional per-node stress scalars (None if this pkl has no stress) ----
    def _opt_scalar(*keys):
        for k in keys:
            if k in sample:
                s = _squeeze_leading_one(_np(sample[k]))
                if s.ndim == 3 and s.shape[-1] == 1:   # (T,N,1) -> (T,N)
                    s = s[..., 0]
                if s.ndim == 1:                        # (N,) constant in time
                    s = np.broadcast_to(s, gt_pos.shape[:2]).copy()
                return s
        return None

    gt_stress   = _opt_scalar("gt_stress")
    pred_stress = _opt_scalar("pred_stress")

    return mesh_pos, gt_pos, pred_pos, node_type, cells, gt_stress, pred_stress


def build_point_data(mesh_pos, gt_pos, pred_pos, node_type,
                     gt_stress=None, pred_stress=None):
    """Compute displacement vectors + scalar fields for every timestep."""
    gt_disp   = gt_pos   - mesh_pos          # (T,N,3)
    pred_disp = pred_pos - mesh_pos
    pos_err   = np.linalg.norm(pred_pos - gt_pos, axis=-1)   # (T,N)

    disp_y_gt   = gt_disp[..., 1].copy()
    disp_y_pred = pred_disp[..., 1].copy()
    if ABS_Y:
        disp_y_gt   = np.abs(disp_y_gt)
        disp_y_pred = np.abs(disp_y_pred)

    disp_mag_gt   = np.linalg.norm(gt_disp,   axis=-1)
    disp_mag_pred = np.linalg.norm(pred_disp, axis=-1)

    fields = dict(
        gt_disp=gt_disp, pred_disp=pred_disp,
        disp_y_gt=disp_y_gt, disp_y_pred=disp_y_pred,
        disp_mag_gt=disp_mag_gt, disp_mag_pred=disp_mag_pred,
        pos_error=pos_err,
    )

    # ---- stress (only if present in the pkl) ----
    if gt_stress is not None and pred_stress is not None:
        gt_s   = gt_stress.astype(np.float64).copy()
        pred_s = pred_stress.astype(np.float64).copy()
        stress_err = np.abs(pred_s - gt_s)
        fields.update(gt_stress=gt_s, pred_stress=pred_s, stress_error=stress_err)

    # Node identity so ParaView can (a) MOVE the die using its real displacement
    # and (b) separate die vs workpiece: colour the workpiece by the field, draw
    # the die as a solid colour, and drop the die entirely on the error panels.
    # Displacement is kept REAL for every node (incl. the die) so it moves.
    fields["node_type"] = node_type.astype(np.float32)
    fields["is_die"]    = (node_type == OBSTACLE_NODE_TYPE).astype(np.float32)

    return fields


def write_series(out_dir, mesh_pos, cells, fields):
    """Write one VTU per timestep + a series.pvd referencing them."""
    os.makedirs(out_dir, exist_ok=True)
    T = mesh_pos.shape[0]
    pvd_entries = []

    for t in range(T):
        point_data = {
            "gt_disp":       fields["gt_disp"][t].astype(np.float32),
            "pred_disp":     fields["pred_disp"][t].astype(np.float32),
            "disp_y_gt":     fields["disp_y_gt"][t].astype(np.float32),
            "disp_y_pred":   fields["disp_y_pred"][t].astype(np.float32),
            "disp_mag_gt":   fields["disp_mag_gt"][t].astype(np.float32),
            "disp_mag_pred": fields["disp_mag_pred"][t].astype(np.float32),
            "pos_error":     fields["pos_error"][t].astype(np.float32),
            "node_type":     fields["node_type"][t].astype(np.float32),
            "is_die":        fields["is_die"][t].astype(np.float32),
        }
        # stress arrays only exist when the pkl had them
        for sk in ("gt_stress", "pred_stress", "stress_error"):
            if sk in fields:
                point_data[sk] = fields[sk][t].astype(np.float32)

        mesh = meshio.Mesh(
            points=mesh_pos[t].astype(np.float32),
            cells=[("tetra", cells)],
            point_data=point_data,
        )
        fname = f"pressnet_{t:04d}.vtu"
        mesh.write(os.path.join(out_dir, fname))
        time_val = (t * DT) if DT is not None else t
        pvd_entries.append(
            f'    <DataSet timestep="{time_val}" group="" part="0" file="{fname}"/>'
        )

    pvd = (
        ['<?xml version="1.0"?>',
         '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
         '  <Collection>']
        + pvd_entries
        + ['  </Collection>', '</VTKFile>']
    )
    with open(os.path.join(out_dir, "series.pvd"), "w") as f:
        f.write("\n".join(pvd))
    print(f"    wrote {T} VTU + series.pvd  ->  {out_dir}")


# ==============================================================================
# 3. MAIN
# ==============================================================================
def load_rollout(pkl_path):
    """
    Robustly load a rollout file.

    These rollouts hold torch tensors and are typically written with torch.save
    (or a pickler that uses storage "persistent ids"), which plain pickle.load
    cannot read -> "A load persistent id instruction was encountered ...".
    torch.load supplies the persistent_load hook, so we try it first and fall
    back to plain pickle only for genuinely-plain pickles.
    """
    if _HAS_TORCH:
        try:
            # weights_only=False: the file is a dict/list of tensors, not a
            # state_dict; needed because recent torch defaults to True.
            return torch.load(pkl_path, map_location="cpu", weights_only=False)
        except TypeError:
            # older torch has no weights_only kwarg
            return torch.load(pkl_path, map_location="cpu")
        except Exception as e:
            print(f"  torch.load failed ({type(e).__name__}: {e}); trying pickle.load ...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_one(model_name, pkl_path):
    print(f"\n{'='*70}\n  MODEL: {model_name}\n  pkl  : {pkl_path}\n{'='*70}")
    if not os.path.exists(pkl_path):
        print(f"  ERROR: pkl not found -- skipping.")
        return

    data = load_rollout(pkl_path)

    # data is a list of sample dicts; allow a bare dict too.
    if isinstance(data, dict):
        data = [data]

    if SAMPLE_INDEX == "all":
        indices = list(range(len(data)))
    elif isinstance(SAMPLE_INDEX, (list, tuple)):
        indices = [int(x) for x in SAMPLE_INDEX]
    else:
        indices = [int(SAMPLE_INDEX)]

    # one geometry -> flat <model>/ ; several -> <model>/sample_<i>/
    multi = len(indices) > 1

    for i in indices:
        if i >= len(data):
            print(f"  WARN: sample {i} out of range (have {len(data)}) -- skipping.")
            continue
        mesh_pos, gt_pos, pred_pos, node_type, cells, gt_stress, pred_stress = extract_arrays(data[i])
        fields = build_point_data(mesh_pos, gt_pos, pred_pos, node_type, gt_stress, pred_stress)

        if multi:
            out_dir = os.path.join(OUTPUT_VTU_ROOT, model_name, f"sample_{i}")
        else:
            out_dir = os.path.join(OUTPUT_VTU_ROOT, model_name)
        has_stress = "stress_error" in fields
        print(f"  sample {i}: T={mesh_pos.shape[0]}  N={mesh_pos.shape[1]}  "
              f"cells={cells.shape[0]}  pos_error_max={fields['pos_error'].max():.4g}  "
              f"stress={'yes' if has_stress else 'no'}"
              + (f"  stress_err_max={fields['stress_error'].max():.4g}" if has_stress else ""))
        write_series(out_dir, mesh_pos, cells, fields)


def main():
    import argparse
    global OUTPUT_VTU_ROOT
    parser = argparse.ArgumentParser(description="Convert rollout pickles to VTU/PVD time series for ParaView")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PKL",
                        help="Label and rollout pickle, e.g. Transolver=out/concatenated_rollout_all.pkl (repeatable)")
    parser.add_argument("--out", default=OUTPUT_VTU_ROOT, help="Folder that receives one sub-folder per model")
    args = parser.parse_args()

    OUTPUT_VTU_ROOT = args.out
    for item in args.model:
        name, _, path = item.partition("=")
        if not path:
            parser.error(f"--model expects NAME=PKL, got {item!r}")
        MODEL_PKLS[name] = path

    os.makedirs(OUTPUT_VTU_ROOT, exist_ok=True)
    for model_name, pkl_path in MODEL_PKLS.items():
        convert_one(model_name, pkl_path)
    print(f"\n{'=' * 70}\n  DONE. VTU root: {OUTPUT_VTU_ROOT}")
    print(f"  Render with: pvbatch paraview_render.py --vtu-root {OUTPUT_VTU_ROOT} --out-root <figures>\n{'=' * 70}")


if __name__ == "__main__":
    main()
