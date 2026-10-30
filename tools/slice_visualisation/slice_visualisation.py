#!/usr/bin/env python3
"""Transolver learned slice decomposition: export, head selection, and figures.

WHAT THIS DOES
==============
Transolver replaces dense attention over mesh nodes with attention over a fixed
number of learned groups called slices. For each node, every attention head
produces a soft membership over the G slices:

    w = softmax( W_s x / tau )                              # shape B H N G

with a learned projection W_s and a learned temperature tau. The memberships of
a node sum to one, so the assignment is a soft partition of the domain. Nothing
in the training loss refers to these groups.

This script reads those memberships straight out of a trained checkpoint and
draws them, so the learned decomposition can be inspected directly.

FOUR SUBCOMMANDS
================
    export     run a checkpoint on one rollout trajectory, capture the slice
               memberships of every block and every head, and save them as .npz.
               Needs the model and the checkpoints, so it runs where they live.
    select     print, for every block and head, how well its partition matches
               the physical state, and which head the stated criterion picks.
               Reads .npz only.
    render     draw all G slice memberships of one head as a grid, one PNG per
               timestep. Reads .npz only.
    compare    draw one slice index across several heads of the same block, on
               the same frame. Reads .npz only.

TWO METHOD POINTS THAT MATTER
=============================
1. Never average over heads. The projection and the temperature belong to a
   single head, so a block with H heads computes H independent decompositions of
   the same mesh. The head average is not a quantity the model computes, and
   averaging several sharp partitions produces something diffuse that resembles
   none of them. This script always exports and always draws a single head.

2. Sharpness alone is the wrong way to choose which head to show. A head can
   reach a very high mean leading membership by assigning almost the whole part
   to one slice, and a criterion that rewards many equally sized groups instead
   will favour a head that cuts the domain into positional bands. Neither is a
   decomposition by physical state. The criterion used here tests that property
   directly; see `select` below.

THE SELECTION CRITERION
=======================
Requirement, applied first and absolutely. The blank has two flat end regions,
one at each end, which stay essentially stationary through the stroke. They are
physically identical to one another and spatially far apart, so a partition based
on physical state must place both in the same slice, and one based on position
cannot. A head that separates them is rejected however sharp it is.

Score among the heads that pass, averaged over the frames of the stage:

    flat coherence  x  eta squared

    flat coherence  share of the stationary material, taken as vertical travel
                    below FLAT_MM, held by its single largest slice. 1.0 means
                    the whole undeformed blank lands in one slice.
    eta squared     fraction of the variance of vertical travel explained
                    between slices rather than within them. 1.0 means every
                    slice is internally uniform in physical state. Needs no
                    threshold, which is why it is reported next to the
                    threshold-dependent quantity above.

Frames in which the punch has not engaged contain no variance in travel to
explain and are excluded from scoring, which leaves them available afterwards as
a held-out check.

COLOUR
======
One panel per slice. The colour at a node is that node's membership in that
slice, on an absolute scale from 0 to 1 shared by every panel and every figure.
Nothing is rescaled per panel and no colour transform is applied, so brightness
is comparable everywhere. Since a node's memberships sum to one, a node bright in
one panel is dark in all the others, so a slice the model uses appears as a
bright patch and an unused slice is dark everywhere. Most panels being dark is
the expected outcome, not a plotting fault.

TWO COUNTS, BOTH REPORTED
=========================
The number of slices that win at least one node is larger than the number that
are visible, because a slice can win a node merely by holding the least small
value there. `select` prints both. The second is the one that describes what a
figure shows.

USAGE
=====
    # where the model and checkpoints live
    python slice_visualisation.py export  --press-ckpt <dir> --dwell-ckpt <dir> --release-ckpt <dir> \n                                          --rollout <concatenated_rollout_all.pkl> --out <npz_dir>
    python slice_visualisation.py export  --dry-run

    # anywhere, from the .npz files
    python slice_visualisation.py select  --src <npz_dir>
    python slice_visualisation.py render  --src <npz_dir> --out <png_dir>
    python slice_visualisation.py render  --src <npz_dir> --stages press --at 3.0
    python slice_visualisation.py render  --src <npz_dir> --block 7 --head 2
    python slice_visualisation.py compare --src <npz_dir> --slice 3 --heads 2 3 1 0

REQUIREMENTS
============
    export     python 3.10+, torch, numpy, and the training repository importable
               so that `models.transolver` can be found
    select     numpy
    render     numpy, pyvista, matplotlib, pillow
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# configuration. Adjust the paths for the machine this runs on.
# ---------------------------------------------------------------------------

# models/transolver.py lives in <repo>/surrogateAI; found automatically from this file's location
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'surrogateAI')
OUT_NPZ = 'learned_slices'
OUT_PNG = os.path.join('figures', 'slices')

OBSTACLE_NODE_TYPE = 1     # node_type value marking tool nodes
DT = 0.01                  # seconds per simulation step
STRIDE = 3                 # the rollout stores every STRIDE-th step
STAGE_STEPS = {'press': (1, 500), 'dwell': (501, 1000), 'release': (1001, 1500)}
TIMES = {
    'press':   [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8],
    'dwell':   [5.4, 6.0, 6.6, 7.2, 7.8, 8.4, 9.0, 9.9],
    'release': [10.2, 10.8, 11.4, 12.0, 12.6, 13.2, 13.8, 14.4],
}
STAGES = ['press', 'dwell', 'release']

FLAT_MM = 2.0        # vertical travel below which material counts as stationary
MIN_RANGE_MM = 1.0   # travel spread below which a frame has no physics to explain
VISIBLE = 0.5        # a panel is visible when its membership exceeds this

# rendering
FOCAL = (-300.0, -19.0, -63.0)
CAMPOS = (560.0, 700.0, 800.0)
PANEL = (1000, 470)
ZOOM = 2.0

VERSION = 'slice_visualisation 1.0'


def log(*a):
    print(*a, flush=True)


def banner(msg):
    log('')
    log('=' * 78)
    log(msg)
    log('=' * 78)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def frame_of(t):
    """Stored-frame index for a time in seconds.

    The rollout keeps every STRIDE-th simulation step, so 1500 steps become 500
    stored frames and frame i holds step STRIDE*(i+1). Passing a raw timestep as
    a frame index silently clamps the later stages to the last stored frame and
    writes identical panels, so `export` checks the result and aborts instead.
    """
    return max(int(round(round(t / DT) / STRIDE)) - 1, 0)


def stage_of(t):
    step = int(round(t / DT))
    for s, (lo, hi) in STAGE_STEPS.items():
        if lo <= step <= hi:
            return s
    return None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def force_cpu_unpickling():
    import torch
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location='cpu', weights_only=False)


def pick_snapshot(ckpt_dir):
    """Prefer a best-validation snapshot; fall back in a stated order.

    `ckpt_dir` may be the folder holding the *_learned_model.pth files, a run's
    `checkpoint/` folder (its `best_checkpoint/` is searched first), or a parent of
    `best_checkpoint/`."""
    for sub in ('', 'best_checkpoint', os.path.join('checkpoint', 'best_checkpoint'), 'checkpoint'):
        path, pref = _pick_in(os.path.join(ckpt_dir, sub))
        if path:
            return path, pref
    return None, None


def _pick_in(ckpt_dir):
    for pref in ('best_model_checkpoint', 'epoch_model_checkpoint',
                 'model_checkpoint', 'final_model_checkpoint'):
        p = os.path.join(ckpt_dir, pref + '_learned_model.pth')
        if os.path.isfile(p):
            return p, pref
    return None, None


def make_inputs(geo, device):
    """The model takes unbatched tensors and adds the batch axis itself."""
    import torch
    pos = np.asarray(geo['pos'], dtype=np.float32).reshape(-1, 3)
    nt = np.asarray(geo['node_type']).reshape(-1, 1)
    return {'curr_pos': torch.tensor(pos, dtype=torch.float32, device=device),
            'node_type': torch.tensor(nt, dtype=torch.float32, device=device)}


def load_geometry(pkl, traj_index, frame):
    """One frame of one rollout trajectory: predicted positions and topology."""
    import pickle
    import torch
    log('  reading %s' % pkl)
    t0 = time.time()
    with open(pkl, 'rb') as fh:
        roll = pickle.load(fh)
    log('  loaded in %.0fs, %d trajectories' % (time.time() - t0, len(roll)))
    if traj_index >= len(roll):
        raise SystemExit('trajectory %d not in file (%d present)'
                         % (traj_index, len(roll)))
    t = roll[traj_index]

    def get(key):
        v = t[key]
        if isinstance(v, torch.Tensor):
            if v.dim() >= 3 and v.size(0) == 1:
                v = v.squeeze(0)
            return v.detach().cpu().numpy()
        return np.asarray(v)

    pred = get('pred_pos')
    n = pred.shape[0]
    f = min(max(frame, 0), n - 1)
    out = {'n_steps': n, 'frame': f, 'pos0': pred[0], 'pos': pred[f],
           'cells': get('cells')[f].astype(int),
           'node_type': get('node_type')[f]}
    del roll
    return out


def capture(model, geo, device):
    """Slice memberships of every block and head, unaveraged.

    Returns (block_names, per_head_temperatures, W) with W shaped
    (n_blocks, n_heads, n_nodes, n_slices). The hook reproduces the model's own
    softmax over the slice projection, including the per-head temperature, whose
    parameter has shape (1, H, 1, 1).
    """
    import torch
    from models import transolver as T

    got, handles = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, T.Physics_Attention_Irregular_Mesh):
            def make(parent, nm):
                def hook(_m, _inp, out):
                    w = torch.softmax(out / parent.temperature, dim=-1)
                    got.append((nm,
                                parent.temperature.detach().float().cpu()
                                .flatten().numpy().tolist(),
                                w.detach().float().cpu()))
                return hook
            handles.append(
                mod.in_project_slice.register_forward_hook(make(mod, name)))
    if not handles:
        raise SystemExit('no Physics_Attention_Irregular_Mesh found in the model')
    try:
        with torch.no_grad():
            model(make_inputs(geo, device))
    finally:
        for hd in handles:
            hd.remove()
    if not got:
        raise SystemExit('the hooks produced no output')
    names = [g[0] for g in got]
    temps = [g[1] for g in got]
    W = np.stack([g[2][0].numpy() for g in got], 0)
    return names, temps, W


def cmd_export(args):
    plan = []
    for stage in STAGES:
        for t in TIMES[stage]:
            if stage_of(t) != stage:
                log('  t=%.2f s is outside %s, skipped' % (t, stage))
                continue
            plan.append((stage, t, frame_of(t)))

    banner('%s  export  %d frames' % (VERSION, len(plan)))
    log('  press ckpt   : %s' % args.press_ckpt)
    log('  dwell ckpt   : %s' % args.dwell_ckpt)
    log('  release ckpt : %s' % args.release_ckpt)
    log('  rollout      : %s' % args.rollout)
    log('  repository   : %s' % args.repo)
    log('  output       : %s' % args.out)
    log('')
    log('  %-9s %8s %8s %8s' % ('stage', 't (s)', 'step', 'frame'))
    for stage, t, fr in plan:
        log('  %-9s %8.2f %8d %8d' % (stage, t, round(t / DT), fr))
    if args.dry_run:
        log('\n  dry run, nothing written')
        return

    import torch
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    log('\n  device: %s' % device)
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.repo)
    if device == 'cpu':
        force_cpu_unpickling()

    ckpts = {'press': args.press_ckpt, 'dwell': args.dwell_ckpt, 'release': args.release_ckpt}
    geom = args.rollout
    written, summary = [], []

    for stage in STAGES:
        frames = [(t, fr) for s, t, fr in plan if s == stage]
        if not frames:
            continue
        banner('%s  %d frames' % (stage.upper(), len(frames)))
        ckdir = ckpts[stage]
        path, pref = pick_snapshot(ckdir)
        if path is None:
            log('  no snapshot in %s, stage skipped' % ckdir)
            continue
        log('  checkpoint %s' % ckdir)
        log('  snapshot   %s' % pref)
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()

        for t, fr in frames:
            geo = load_geometry(geom, args.traj, fr)
            if geo['frame'] != fr:
                raise SystemExit(
                    'frame %d (t=%.2f s) was clamped to %d: only %d stored '
                    'frames. Check STRIDE, currently %d. Refusing to write.'
                    % (fr, t, geo['frame'], geo['n_steps'], STRIDE))
            names, temps, W = capture(model, geo, device)

            rows = W.sum(-1)
            if not np.allclose(rows, 1.0, atol=2e-3):
                raise SystemExit(
                    'memberships do not sum to one (min %.4f, max %.4f), so the '
                    'hook is not reading the model softmax. Refusing to write.'
                    % (rows.min(), rows.max()))

            nt = np.asarray(geo['node_type']).ravel().astype(int)
            is_die = nt == OBSTACLE_NODE_TYPE
            pos = np.asarray(geo['pos'], dtype=np.float32)
            travel = np.abs(pos[:, 1]
                            - np.asarray(geo['pos0'], dtype=np.float32)[:, 1])

            meta = {'version': VERSION, 'stage': stage, 't': t, 'frame': fr,
                    'step': int(round(t / DT)), 'stride': STRIDE,
                    'n_stored_frames': int(geo['n_steps']), 'traj': args.traj,
                    'checkpoint': rel[stage], 'snapshot': pref,
                    'blocks': names, 'temperatures': temps,
                    'n_blocks': int(W.shape[0]), 'n_heads': int(W.shape[1]),
                    'n_nodes': int(W.shape[2]), 'n_slices': int(W.shape[3]),
                    'rowsum_min': float(rows.min()),
                    'rowsum_max': float(rows.max()),
                    'n_workpiece': int((~is_die).sum())}
            name = 'slices_%s_t%04d.npz' % (stage, int(round(t * 100)))
            np.savez_compressed(
                os.path.join(args.out, name),
                weights=W.astype(np.float16), points=pos,
                node_type=nt.astype(np.int16), travel=travel.astype(np.float32),
                cells=np.asarray(geo['cells'], dtype=np.int32),
                is_die=is_die, meta=json.dumps(meta))
            mb = os.path.getsize(os.path.join(args.out, name)) / 1e6
            log('    wrote %s  (%d blocks x %d heads x %d nodes x %d slices, '
                '%.1f MB)' % (name, W.shape[0], W.shape[1], W.shape[2],
                              W.shape[3], mb))
            written.append(name)
            summary.append({'stage': stage, 't': t, 'snapshot': pref,
                            'n_blocks': int(W.shape[0])})

    with open(os.path.join(args.out, 'index.json'), 'w') as fh:
        json.dump({'version': VERSION, 'files': written, 'summary': summary},
                  fh, indent=1)
    banner('done: %d files plus index.json in %s' % (len(written), args.out))
    log('  Every block and head is stored unaveraged, so no figure needs a')
    log('  second export. float16 is sufficient: the smallest meaningful value')
    log('  is the equal-membership level 1/n_slices.')


# ---------------------------------------------------------------------------
# reading and metrics
# ---------------------------------------------------------------------------

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {'W': z['weights'].astype(np.float32), 'points': z['points'],
            'cells': z['cells'], 'is_die': z['is_die'],
            'travel': z['travel'], 'meta': json.loads(str(z['meta']))}


def by_stage(src, stages):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, 'slices_*.npz'))):
        st = os.path.basename(f).split('_')[1]
        if st in stages:
            out.setdefault(st, []).append(f)
    return out


def metrics(a, travel, x):
    """How well one head's partition matches the physical state.

    a       (n_nodes, n_slices) memberships for one block and one head
    travel  vertical travel per node, the physical state
    x       position along the blank, used only for the flat-ends test

    Returns a dict with
      eta2      variance of travel explained between slices
      ends      do the two flat ends share their dominant slice
      coh       share of the stationary material in its largest slice
      lead      mean membership of a node's leading slice
      n_win     slices winning at least one node
      n_vis     slices whose membership exceeds VISIBLE somewhere
    """
    own = a.argmax(1)
    gm = travel.mean()
    ss = float(((travel - gm) ** 2).sum())
    btw = 0.0
    for s in np.unique(own):
        m = own == s
        btw += m.sum() * (travel[m].mean() - gm) ** 2
    eta = float(btw / ss) if ss > 0 else 0.0

    flat = travel < FLAT_MM

    def dominant(mask):
        if mask.sum() == 0:
            return -2
        ids, c = np.unique(own[mask], return_counts=True)
        return int(ids[c.argmax()])

    xm = np.median(x)
    ends = dominant(flat & (x < xm)) == dominant(flat & (x >= xm))
    if flat.sum():
        _, fc = np.unique(own[flat], return_counts=True)
        coh = float(fc.max() / flat.sum())
    else:
        coh = 0.0
    return {'eta2': eta, 'ends': bool(ends), 'coh': coh,
            'lead': float(a.max(1).mean()),
            'n_win': int(len(np.unique(own))),
            'n_vis': int(sum(1 for s in range(a.shape[1])
                             if a[:, s].max() > VISIBLE))}


def choose_head(frames, block=None, head=None, quiet=False):
    """Pick one block and head for a whole stage, printing every candidate."""
    acc = {}
    scored = 0
    for d in frames:
        wp = ~d['is_die']
        tr = d['travel'][wp]
        x = d['points'][wp][:, 0]
        if tr.max() - tr.min() < MIN_RANGE_MM:
            continue                     # punch not engaged, nothing to explain
        scored += 1
        W = d['W']
        for b in range(W.shape[0]):
            for h in range(W.shape[1]):
                acc.setdefault((b, h), []).append(metrics(W[b, h][wp], tr, x))
    if not acc:
        raise SystemExit('no frame in this stage has any deformation to explain')

    cands = []
    for (b, h), v in acc.items():
        cands.append({
            'b': b, 'h': h,
            'eta2': float(np.mean([q['eta2'] for q in v])),
            'ends': all(q['ends'] for q in v),
            'coh': float(np.mean([q['coh'] for q in v])),
            'lead': float(np.mean([q['lead'] for q in v])),
            'n_win': float(np.mean([q['n_win'] for q in v])),
            'n_vis': float(np.mean([q['n_vis'] for q in v]))})
    for c in cands:
        c['score'] = c['coh'] * c['eta2']
    passing = sorted([c for c in cands if c['ends']], key=lambda c: -c['score'])
    failing = sorted([c for c in cands if not c['ends']],
                     key=lambda c: -c['score'])

    if not quiet:
        log('    %d of %d block-and-head pairs keep both flat ends in one slice '
            'on all %d scored frames' % (len(passing), len(cands), scored))
        log('      %-6s %-5s %8s %11s %7s %8s %8s %8s'
            % ('block', 'head', 'eta^2', 'flat coher', 'score', 'lead',
               'winning', 'visible'))
        for c in passing[:8]:
            log('      %-6d %-5d %8.3f %10.0f%% %7.3f %8.3f %8.1f %8.1f'
                % (c['b'], c['h'], c['eta2'], 100 * c['coh'], c['score'],
                   c['lead'], c['n_win'], c['n_vis']))
        if failing:
            c = failing[0]
            log('      rejected, separates the flat ends: block %d head %d, '
                'eta^2 %.3f, flat coherence %.0f%%'
                % (c['b'], c['h'], c['eta2'], 100 * c['coh']))
    if not passing:
        raise SystemExit('every head partitions by position; nothing to show')

    if block is not None and head is not None:
        sel = [c for c in cands if c['b'] == block and c['h'] == head]
        if not sel:
            raise SystemExit('block %d head %d does not exist for this stage'
                             % (block, head))
        if not sel[0]['ends'] and not quiet:
            log('      NOTE: the requested head separates the flat ends, so it '
                'partitions by position rather than by physical state.')
        return sel[0], 'requested'
    return passing[0], 'chosen by criterion'


def cmd_select(args):
    groups = by_stage(args.src, args.stages)
    if not groups:
        raise SystemExit('no slices_*.npz in %s' % args.src)
    banner('%s  head selection' % VERSION)
    for stage in args.stages:
        fl = groups.get(stage, [])
        if not fl:
            log('\n  %s: nothing exported' % stage)
            continue
        frames = [load_npz(f) for f in fl]
        ref = frames[len(frames) // 2]
        log('\n  %s: %d frames, %d blocks, %d heads, %d slices, %d nodes'
            % (stage.upper(), len(fl), ref['W'].shape[0], ref['W'].shape[1],
               ref['W'].shape[3], int(ref['meta']['n_workpiece'])))
        c, why = choose_head(frames, args.block, args.head)
        log('    using block %d head %d (%s)' % (c['b'], c['h'], why))
        log('      stage means: eta^2 %.3f, flat coherence %.0f%%, leading '
            'membership %.3f, %.1f slices winning nodes, %.1f visible'
            % (c['eta2'], 100 * c['coh'], c['lead'], c['n_win'], c['n_vis']))
        log('    checkpoint %s [%s], learned temperature %.3f'
            % (ref['meta']['checkpoint'], ref['meta']['snapshot'],
               ref['meta']['temperatures'][c['b']][c['h']]))
        log('    per frame:')
        log('      %7s %8s %8s %11s %8s %8s %11s'
            % ('t (s)', 'lead', 'eta^2', 'flat coher', 'winning', 'visible',
               'max travel'))
        for d in frames:
            wp = ~d['is_die']
            tr = d['travel'][wp]
            x = d['points'][wp][:, 0]
            m = metrics(d['W'][c['b'], c['h']][wp], tr, x)
            eta = ('%8.3f' % m['eta2']
                   if tr.max() - tr.min() >= MIN_RANGE_MM else '   undef')
            log('      %7.2f %8.3f %s %10.0f%% %8d %8d %10.2f'
                % (d['meta']['t'], m['lead'], eta, 100 * m['coh'],
                   m['n_win'], m['n_vis'], tr.max()))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def mesh_of(d):
    """Grid of the whole mesh; the tool nodes are removed at render time."""
    import pyvista as pv
    cells = np.asarray(d['cells'], dtype=np.int64).reshape(-1, 4)
    conn = np.hstack([np.full((len(cells), 1), 4, dtype=np.int64),
                      cells]).ravel()
    ctypes = np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8)
    g = pv.UnstructuredGrid(conn, ctypes, np.asarray(d['points'], dtype=float))
    g.point_data['is_die'] = d['is_die'].astype(float)
    g.point_data['travel'] = d['travel']
    return g


def render_one(mesh, field, out_png):
    """One panel. Fixed camera and matte lighting, so panels are comparable."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=PANEL)
    p.set_background('white')
    try:
        p.enable_anti_aliasing('ssaa')
    except Exception:
        pass
    p.add_mesh(mesh, scalars=field, cmap='viridis', clim=(0.0, 1.0),
               smooth_shading=True, interpolate_before_map=True, n_colors=256,
               lighting=True, show_scalar_bar=False, specular=0.0,
               specular_power=1.0, ambient=0.32, diffuse=0.78)
    p.camera.focal_point = FOCAL
    p.camera.position = CAMPOS
    p.camera.up = (0.0, 1.0, 0.0)
    p.camera.zoom(ZOOM)
    p.screenshot(out_png)
    p.close()


def content_box(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'))
    nz = np.where((a < 246).any(axis=2))
    if not len(nz[0]):
        return None
    return int(nz[1].min()), int(nz[0].min()), int(nz[1].max()), int(nz[0].max())


def crop_all(paths, pad=4):
    """One shared crop box for every panel, so the part is the same size in all.

    Cropping each panel to its own bounding box and letting the layout stretch it
    would make the part appear a different size in every panel and destroy any
    comparison between them.
    """
    from PIL import Image
    boxes = [b for b in (content_box(p) for p in paths) if b]
    if not boxes:
        return
    x0 = max(min(b[0] for b in boxes) - pad, 0)
    y0 = max(min(b[1] for b in boxes) - pad, 0)
    x1 = max(b[2] for b in boxes) + pad
    y1 = max(b[3] for b in boxes) + pad
    for p in paths:
        im = Image.open(p)
        im.crop((x0, y0, min(x1, im.width), min(y1, im.height))).save(p)


def cmd_render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from PIL import Image
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

    groups = by_stage(args.src, args.stages)
    if not groups:
        raise SystemExit('no slices_*.npz in %s' % args.src)
    os.makedirs(args.tmp, exist_ok=True)
    banner('%s  render' % VERSION)
    log('  source  : %s' % args.src)
    log('  output  : %s' % args.out)
    log('  scratch : %s  (local, per-panel intermediates)' % args.tmp)

    for stage in args.stages:
        fl = groups.get(stage, [])
        if not fl:
            log('\n  %s: nothing exported' % stage)
            continue
        frames = [load_npz(f) for f in fl]
        ref = frames[len(frames) // 2]
        log('\n  %s: %d frames, %d blocks, %d heads'
            % (stage.upper(), len(fl), ref['W'].shape[0], ref['W'].shape[1]))
        c, why = choose_head(frames, args.block, args.head)
        b, h = c['b'], c['h']
        log('    using block %d head %d (%s)' % (b, h, why))
        log('    checkpoint %s [%s], learned temperature %.3f'
            % (ref['meta']['checkpoint'], ref['meta']['snapshot'],
               ref['meta']['temperatures'][b][h]))

        outdir = os.path.join(args.out, stage)
        os.makedirs(outdir, exist_ok=True)
        for d in frames:
            t = d['meta']['t']
            if args.at is not None and abs(t - args.at) > 1e-6:
                continue
            g = mesh_of(d)
            A = d['W'][b, h]
            n_slices = A.shape[1]
            for s in range(n_slices):
                g.point_data['s%d' % s] = A[:, s]
            work = g.threshold(0.5, scalars='is_die', invert=True)

            paths = []
            for s in range(n_slices):
                p = os.path.join(args.tmp, '%s_%04d_%02d.png'
                                 % (stage, int(round(t * 100)), s))
                render_one(work, 's%d' % s, p)
                paths.append(p)
            crop_all(paths)

            ncol = args.cols
            nrow = int(np.ceil(len(paths) / ncol))
            iw, ih = Image.open(paths[0]).size
            pw, lab = 1.05, (0.0 if args.no_labels else 0.17)
            fig, axes = plt.subplots(
                nrow, ncol,
                figsize=(pw * ncol, (pw * ih / iw + lab) * nrow + 0.05),
                squeeze=False)
            for k in range(nrow * ncol):
                ax = axes[k // ncol][k % ncol]
                ax.set_axis_off()
                if k < len(paths):
                    ax.imshow(imread(paths[k]))
                    if not args.no_labels:
                        ax.set_title('Slice %d' % k, fontsize=6.4, pad=1.6)
            fig.subplots_adjust(left=0.004, right=0.996,
                                top=0.985 if args.no_labels else 0.962,
                                bottom=0.004, wspace=0.0,
                                hspace=0.0 if args.no_labels else 0.24)
            name = '%s_t%04d_b%dh%d.png' % (stage, int(round(t * 100)), b, h)
            fig.savefig(os.path.join(outdir, name), dpi=args.dpi,
                        facecolor='white', bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
            log('    wrote %s/%s' % (stage, name))

    log('\ndone. Colour is the softmax membership on an absolute 0 to 1 scale, '
        'shared by every panel.')


def cmd_compare(args):
    """One slice index across several heads of one block, on one frame.

    The geometry, the node set and the slice index are identical in every panel,
    so any difference between them is the learned assignment and nothing else.
    This distinguishes a learned membership field from a function of the geometry.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from PIL import Image

    f = os.path.join(args.src, 'slices_%s_t%04d.npz'
                     % (args.stage, int(round(args.at * 100))))
    if not os.path.exists(f):
        raise SystemExit('no export at %s' % f)
    d = load_npz(f)
    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    banner('%s  compare slice %d across heads' % (VERSION, args.sl))

    g = mesh_of(d)
    wp = ~d['is_die']
    tr = d['travel'][wp]
    flat, bend = tr < FLAT_MM, tr > 10.0

    paths, labels = [], []
    for h in args.heads:
        a = d['W'][args.block, h][:, args.sl]
        g.point_data['f'] = a
        work = g.threshold(0.5, scalars='is_die', invert=True)
        p = os.path.join(args.tmp, 'cmp_h%d.png' % h)
        render_one(work, 'f', p)
        paths.append(p)
        aw = a[wp]
        labels.append('head %d\nstationary %.3f   moving %.3f'
                      % (h, aw[flat].mean(), aw[bend].mean()))
        log('  head %d: mean on stationary material %.4f, on moving %.4f, '
            'peak %.4f' % (h, aw[flat].mean(), aw[bend].mean(), aw.max()))
    crop_all(paths)

    iw, ih = Image.open(paths[0]).size
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6 * ih / iw + 0.75),
                             squeeze=False)
    for k in range(n):
        ax = axes[0][k]
        ax.set_axis_off()
        ax.imshow(imread(paths[k]))
        ax.set_title(labels[k], fontsize=8, pad=4)
    fig.suptitle('Slice %d of block %d, %s, t = %.1f s: identical geometry and '
                 'slice index, different head'
                 % (args.sl, args.block, args.stage, args.at), fontsize=9)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.01,
                        wspace=0.02)
    out = os.path.join(args.out, '%s_t%04d_b%d_slice%d_across_heads.png'
                       % (args.stage, int(round(args.at * 100)), args.block,
                          args.sl))
    fig.savefig(out, dpi=args.dpi, facecolor='white', bbox_inches='tight',
                pad_inches=0.02)
    plt.close(fig)
    log('\nwrote %s' % out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    e = sub.add_parser('export', help='capture slice memberships from a checkpoint')
    e.add_argument('--press-ckpt', required=True, help='checkpoint folder of the press (stage 1) Transolver')
    e.add_argument('--dwell-ckpt', required=True, help='checkpoint folder of the dwell (stage 2) Transolver')
    e.add_argument('--release-ckpt', required=True, help='checkpoint folder of the release (stage 3) Transolver')
    e.add_argument('--rollout', required=True,
                   help='concatenated_rollout_all.pkl from inference.py (geometry for the figures)')
    e.add_argument('--repo', default=REPO, help='folder holding models/transolver.py (default: ../../surrogateAI)')
    e.add_argument('--out', default=OUT_NPZ)
    e.add_argument('--traj', type=int, default=0)
    e.add_argument('--cpu', action='store_true')
    e.add_argument('--dry-run', action='store_true')
    e.set_defaults(func=cmd_export)

    s = sub.add_parser('select', help='score every block and head, pick one')
    s.add_argument('--src', default=OUT_NPZ)
    s.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    s.add_argument('--block', type=int, default=None)
    s.add_argument('--head', type=int, default=None)
    s.set_defaults(func=cmd_select)

    r = sub.add_parser('render', help='draw all slices of one head as a grid')
    r.add_argument('--src', default=OUT_NPZ)
    r.add_argument('--out', default=OUT_PNG)
    r.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    r.add_argument('--at', type=float, default=None,
                   help='render only this time, in seconds')
    r.add_argument('--block', type=int, default=None)
    r.add_argument('--head', type=int, default=None)
    r.add_argument('--cols', type=int, default=8)
    r.add_argument('--no-labels', action='store_true')
    r.add_argument('--dpi', type=int, default=350)
    r.add_argument('--tmp', default=os.path.join(
        os.environ.get('TEMP', '/tmp'), 'slice_render_tmp'))
    r.set_defaults(func=cmd_render)

    c = sub.add_parser('compare', help='one slice index across several heads')
    c.add_argument('--src', default=OUT_NPZ)
    c.add_argument('--out', default=os.path.join(OUT_PNG, 'checks'))
    c.add_argument('--stage', default='press', choices=STAGES)
    c.add_argument('--at', type=float, default=3.0)
    c.add_argument('--block', type=int, default=7)
    c.add_argument('--slice', dest='sl', type=int, default=3)
    c.add_argument('--heads', type=int, nargs='+', default=[2, 3, 1, 0])
    c.add_argument('--dpi', type=int, default=300)
    c.add_argument('--tmp', default=os.path.join(
        os.environ.get('TEMP', '/tmp'), 'slice_cmp_tmp'))
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
