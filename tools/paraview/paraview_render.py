#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paraview_render.py
================================================================================
STEP 2 of 2  --  run with ParaView's python:

    pvpython paraview_render.py
    xvfb-run -a pvbatch paraview_render.py     # headless GLX build
    pvbatch  paraview_render.py                # headless OSMESA build

Parallel (N processes, each rendering a subset of units -- see
run_render_parallel.sh, which does exactly this):
    xvfb-run -a pvbatch paraview_render.py --scan-only
    xvfb-run -a pvbatch paraview_render.py --shard 0 --num-shards 4 &
    xvfb-run -a pvbatch paraview_render.py --shard 1 --num-shards 4 &
    ...

Reads the per-model / per-geometry VTU folders from `convert_pkls_to_vtu.py`
and renders, for each geometry of each model, GT / Prediction / Error panels
across time, then stitches them side-by-side (with legends beside, not
overlaid) into an mp4 + gif.

Design points:
  * 3D oblique view (CAMERA_AZIMUTH / CAMERA_ELEVATION), not a flat 2D look.
  * The die MOVES: it is warped by its real displacement and drawn as a solid
    colour (DIE_COLOR); the workpiece is coloured by the field.
  * Error panels show ONLY the workpiece (die thresholded out).
  * Legends are written as their OWN image files and placed BESIDE the panels
    (in-panel colour bars are turned off, so nothing overlays the mesh).
  * Only every SKIP-th timestep is rendered (default 10), not all 500.
  * Colour scales: GT & Pred share ONE range per geometry (time-invariant);
    the error legend has its OWN range per geometry (time-invariant).

Requires the VTU point arrays written by the converter, including 'is_die'.
================================================================================
"""

import os
import glob
import json
import shutil

from paraview.simple import *
from vtkmodules.vtkIOImage import vtkPNGReader, vtkPNGWriter
try:
    from vtk import vtkImageAppend, vtkImageExtractComponents
except ImportError:
    from vtkmodules.vtkImagingCore import vtkImageAppend, vtkImageExtractComponents

# matplotlib is OPTIONAL: the default legend backend draws with ParaView/VTK
# itself (vtkScalarBarActor, off-screen). matplotlib is only a fallback.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import Normalize as _Normalize, LogNorm as _LogNorm
    from matplotlib.cm import ScalarMappable as _ScalarMappable
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

LoadPalette('WhiteBackground')


# ==============================================================================
# 1. USER SETTINGS
# ==============================================================================
VTU_ROOT    = "vtu_root"      # overridden by --vtu-root
OUTPUT_ROOT = "figures"       # overridden by --out-root

# Optional model-name filter ([] = every model found). Geometries are all rendered.
MODELS = []

# Render every SKIP-th timestep (10 -> 50 frames out of 500).
SKIP = 10

# Share the colour scale across models for the SAME geometry: sample_0 uses one
# legend for GCN/Transolver/..., sample_1 uses its own, etc. (value range = union
# over all models incl. GT; error range = max over all models). A pre-pass scans
# every model first. Set False to scale each model/geometry independently.
SHARE_LEGEND_ACROSS_MODELS = True

# Manual legend-range overrides (applied last, after shared/per-unit ranges).
# Keyed by (sample_key, task_label); values: optional 'value'=(vmin, vmax)
# and/or 'error'=(emin, emax). Useful when ONE diverging model inflates the
# shared error range (max over all models) so much that a good model's error
# panel looks uniformly blue. Example:
#   RANGE_OVERRIDES = {("sample_9", "Displacement_Y"): dict(error=(0.0, 5.0))}
RANGE_OVERRIDES = {}

# True: reuse panel PNGs that already exist on disk (skips rendering, redoes
# legends + stitching + mp4/gif only). Handy after a stitch/legend-only fix.
SKIP_EXISTING_PANELS = False

# ---- parallel execution ------------------------------------------------------
# Units (model/geometry pairs) are independent, so N pvbatch processes can each
# render a subset ("shard"). Shared legends stay consistent because the range
# pre-pass is computed ONCE and cached to a JSON that every shard reads:
#     xvfb-run -a pvbatch render.py --scan-only                # 1) write cache
#     xvfb-run -a pvbatch render.py --shard 0 --num-shards 4 & # 2) N of these
#     ...                                                      #    in parallel
# (or just use run_render_parallel.sh). The cache is auto-recomputed when any
# series.pvd is newer than it (i.e. after re-running the converter).
USE_RANGES_CACHE = True
RANGES_CACHE = os.path.join(OUTPUT_ROOT, "ranges_cache.json")

# ---- comparison task(s) ------------------------------------------------------
#   gt/pred/error : (color_scalar, warp_vector)
# Error is drawn on the GT-warped workpiece by default ('gt_disp').
COMPARISON_TASKS = [
    dict(label="Displacement_Y",
         gt=("disp_y_gt", "gt_disp"), pred=("disp_y_pred", "pred_disp"),
         error=("pos_error", "pred_disp"),          # error on the PREDICTED shape
         value_title="y_disp", error_title="Pos Error", err_log=False),

    dict(label="Stress",
         gt=("gt_stress", "gt_disp"), pred=("pred_stress", "pred_disp"),
         error=("stress_error", "pred_disp"),       # error on the PREDICTED shape
         value_title="stress", error_title="Stress Error", err_log=False),
]

# ---- appearance --------------------------------------------------------------
VALUE_CMAP = "Jet"
ERROR_CMAP = "Jet"

# Legend rendering backend.
#   "paraview": drawn by ParaView itself -- a vtkScalarBarActor rendered in a
#       private OFF-SCREEN window (never a shared view, so it cannot capture
#       the wrong content). Colours are sampled from the SAME LUT the panels
#       use, and every call builds its own transfer function, so the value and
#       error legends can never share state. Output is RGB like the panels.
#   "mpl": matplotlib colourbar.
# Whichever is chosen, the other is the automatic fallback; a scalar-bar-view
# screenshot is the last resort.
LEGEND_BACKEND = "paraview"

# Die (rigid tool) appearance. Kept for context (now on the error panels too) but
# SEMI-TRANSPARENT so it never hides the workpiece. It is coloured the jet-MINIMUM
# blue, i.e. the "0" colour on the legend, to signal the die is rigid / does not
# deform. Set SHOW_DIE=False to drop it entirely.
SHOW_DIE    = True
DIE_COLOR   = [0.0, 0.0, 0.5]   # jet-min blue = "0 on the deformation scale"
DIE_OPACITY = 0.40              # semi-transparent so the workpiece shows through

# Draw element edges ("mesh grains") on the surfaces, like the reference image.
SHOW_MESH_EDGES = True
EDGE_COLOR      = [0.25, 0.25, 0.25]

# 3D view: looks straight down +Z (workpiece cross-section facing you, +Y up,
# like animate_rollout) then tilts by these small angles. Kept to a SLIGHT tilt.
# Raise for a more isometric look; set to 0/0 for perfectly front-on.
CAMERA_AZIMUTH   = 10.0
CAMERA_ELEVATION = 6.0
MANUAL_CAMERA = False
CAMERA = dict(
    position       = [1670.825, -906.853, 1470.499],
    focal_point    = [3.393, -55.007, -41.802],
    view_up        = [-0.302, 0.647, 0.698],
    parallel_scale = 351.635,
)

# ---- output / quality --------------------------------------------------------
KEEP_FRAMES        = True
MAKE_MP4           = True
MAKE_GIF           = True
STITCH_WITH_LEGEND = True     # place value/error legends beside the panels
USE_FXAA           = True     # anti-aliasing -> smoother edges
PANEL_WIDTH   = 1500          # higher resolution -> crisper render
PANEL_HEIGHT  = 1050
LEGEND_WIDTH  = 320
FRAME_RATE    = 12


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def discover_units():
    """[(label, pvd_path)] for every series.pvd under VTU_ROOT (any depth)."""
    units = []
    for pvd in sorted(glob.glob(os.path.join(VTU_ROOT, "**", "series.pvd"),
                                recursive=True)):
        label = os.path.relpath(os.path.dirname(pvd), VTU_ROOT)
        if MODELS and label.split(os.sep)[0] not in MODELS:
            continue
        units.append((label, pvd))
    return units


def sample_key(label):
    """Geometry identity, shared across models. 'GCN/sample_0' -> 'sample_0';
    a flat 'GCN' (single geometry per model) -> 'single'."""
    parts = label.split(os.sep)
    return os.sep.join(parts[1:]) if len(parts) > 1 else "single"


def precompute_sample_ranges(units):
    """
    Pre-pass: for each geometry (sample_key), union the value range and take the
    max error across ALL models, so every model of that geometry renders on one
    shared legend. Returns {(sample_key, task_label): [vmin, vmax, emax]}.
    """
    ranges = {}
    for label, pvd in units:
        reg = 'scan_' + label.replace(os.sep, "_").replace("-", "_")
        reader = PVDReader(registrationName=reg, FileName=pvd)
        ts = reader.TimestepValues
        if ts is None:
            ts = [0]
        step_times = [ts[t] for t in range(0, len(ts), SKIP)]
        reader.UpdatePipeline(ts[0])
        pdi = reader.GetDataInformation().GetPointDataInformation()
        avail = {pdi.GetArrayInformation(i).GetName() for i in range(pdi.GetNumberOfArrays())}
        skey = sample_key(label)
        for task in COMPARISON_TASKS:
            gt_c, pred_c, err_c = task["gt"], task["pred"], task["error"]
            needed = {gt_c[0], pred_c[0], err_c[0], gt_c[1], pred_c[1], err_c[1], 'is_die'}
            if needed - avail:
                continue
            gmin, gmax = workpiece_range(reader, gt_c[0],   gt_c[1],   step_times)
            pmin, pmax = workpiece_range(reader, pred_c[0], pred_c[1], step_times)
            _, emax    = workpiece_range(reader, err_c[0],  err_c[1],  step_times)
            vmin, vmax = min(gmin, pmin), max(gmax, pmax)
            # per-model line: shows WHICH model drives the shared error max
            print(f"   scan {label:28s} {task['label']:16s} "
                  f"value[{vmin:.4g}, {vmax:.4g}]  err_max {emax:.4g}")
            key = (skey, task["label"])
            if key not in ranges:
                ranges[key] = [vmin, vmax, emax]
            else:
                ranges[key][0] = min(ranges[key][0], vmin)
                ranges[key][1] = max(ranges[key][1], vmax)
                ranges[key][2] = max(ranges[key][2], emax)
        Delete(reader); del reader
    print("Shared per-sample ranges:")
    for (skey, tlabel), (vmn, vmx, emx) in sorted(ranges.items()):
        print(f"   {skey:16s} {tlabel:16s} value[{vmn:.4g}, {vmx:.4g}] error[0, {emx:.4g}]")
    return ranges


def load_ranges_cache(units):
    """Return cached shared ranges, or None if absent/stale/disabled."""
    if not (USE_RANGES_CACHE and os.path.exists(RANGES_CACHE)):
        return None
    try:
        newest = max(os.path.getmtime(p) for _, p in units) if units else 0.0
        if os.path.getmtime(RANGES_CACHE) < newest:
            print("Ranges cache is older than the VTU data -> recomputing.")
            return None
        with open(RANGES_CACHE) as f:
            raw = json.load(f)
        ranges = {tuple(k.split("||", 1)): list(v) for k, v in raw.items()}
        print(f"Loaded shared ranges from cache ({len(ranges)} entries): {RANGES_CACHE}")
        return ranges
    except Exception as e:
        print(f"NOTE: could not read ranges cache ({e}); recomputing.")
        return None


def save_ranges_cache(ranges):
    if not USE_RANGES_CACHE:
        return
    try:
        os.makedirs(os.path.dirname(RANGES_CACHE), exist_ok=True)
        tmp = f"{RANGES_CACHE}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({f"{k[0]}||{k[1]}": list(v) for k, v in ranges.items()},
                      f, indent=1)
        os.replace(tmp, RANGES_CACHE)   # atomic -> safe even if shards race
        print(f"Wrote ranges cache -> {RANGES_CACHE}")
    except Exception as e:
        print(f"NOTE: could not write ranges cache ({e}).")


def set_threshold(th, lo, hi):
    """Set threshold range across ParaView API versions (5.9 vs 5.10+)."""
    try:
        th.LowerThreshold = lo
        th.UpperThreshold = hi
    except Exception:
        pass
    try:
        th.ThresholdRange = [lo, hi]
    except Exception:
        pass


def workpiece(warp):
    """Threshold a warped source down to workpiece cells (is_die == 0)."""
    th = Threshold(Input=warp)
    th.Scalars = ['POINTS', 'is_die']
    set_threshold(th, -0.5, 0.5)
    return th


def die_part(warp):
    """Threshold a warped source down to die cells (is_die == 1)."""
    th = Threshold(Input=warp)
    th.Scalars = ['POINTS', 'is_die']
    set_threshold(th, 0.5, 1.5)
    return th


def get_global_range(src, color_var, step_times):
    """Min/max of color_var over the chosen timesteps -> time-invariant range."""
    lo_all, hi_all = float('inf'), float('-inf')
    for t in step_times:
        src.UpdatePipeline(t)
        arr = src.GetDataInformation().GetPointDataInformation().GetArrayInformation(color_var)
        if arr:
            nc = arr.GetNumberOfComponents()
            lo, hi = arr.GetComponentRange(0 if nc == 1 else -1)
            lo_all = min(lo_all, lo)
            hi_all = max(hi_all, hi)
    if lo_all == float('inf'):
        return 0.0, 1.0
    if lo_all == hi_all:
        hi_all = lo_all + 1.0
    return lo_all, hi_all


def workpiece_range(reader, color_var, warp_var, step_times):
    warp = WarpByVector(Input=reader); warp.Vectors = ['POINTS', warp_var]
    work = workpiece(warp)
    lo, hi = get_global_range(work, color_var, step_times)
    Delete(work); Delete(warp)
    return lo, hi


def config_lut(color_var, cmap, cmin, cmax, log_scale):
    lut = GetColorTransferFunction(color_var)
    pwf = GetOpacityTransferFunction(color_var)
    try:
        lut.ApplyPreset(cmap, True)
    except Exception:
        pass
    if log_scale:
        lo = max(cmin, 1e-10); hi = max(cmax, lo * 10)
        lut.UseLogScale = 1
        lut.RescaleTransferFunction(lo, hi); pwf.RescaleTransferFunction(lo, hi)
    else:
        lut.UseLogScale = 0
        lut.RescaleTransferFunction(cmin, cmax); pwf.RescaleTransferFunction(cmin, cmax)
    lut.AutomaticRescaleRangeMode = 'Never'
    return lut


def set_camera(view, reader):
    tmp = Show(reader, view); view.Update()
    SetActiveView(view)
    if MANUAL_CAMERA:
        view.CameraPosition      = CAMERA["position"]
        view.CameraFocalPoint    = CAMERA["focal_point"]
        view.CameraViewUp        = CAMERA["view_up"]
        view.CameraParallelScale = CAMERA["parallel_scale"]
    else:
        # Deterministic front view: look along +Z with +Y up (animate_rollout
        # style), then tilt slightly for a 3D feel. The second ResetCamera
        # refits the zoom while KEEPING this new orientation.
        ResetCamera(view)
        cam = GetActiveCamera()
        fp = cam.GetFocalPoint()
        dist = cam.GetDistance()
        cam.SetFocalPoint(fp[0], fp[1], fp[2])
        cam.SetPosition(fp[0], fp[1], fp[2] + dist)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.Azimuth(CAMERA_AZIMUTH)
        cam.Elevation(CAMERA_ELEVATION)
        ResetCamera(view)
        Render(view)              # renderer recomputes the clipping range here
    Hide(reader, view); Delete(tmp)


def _mpl_cmap_name(cmap):
    name = {"jet": "jet", "inferno": "inferno", "viridis": "viridis",
            "inferno (matplotlib)": "inferno", "viridis (matplotlib)": "viridis",
            "plasma (matplotlib)": "plasma", "magma (matplotlib)": "magma",
            "cool to warm": "coolwarm", "coolwarm": "coolwarm",
            "black-body radiation": "afmhot"}.get(cmap.lower(), cmap.lower())
    try:
        _plt.get_cmap(name)
        return name
    except Exception:
        return "jet"


def _vtk_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale=False):
    """
    Legend rendered by ParaView ITSELF (vtkScalarBarActor) -- no matplotlib.

    Why this is robust where the old scalar-bar-view screenshot was not:
      * the colours are sampled from the very LUT the panels use (config_lut),
        then copied into a FRESH vtkColorTransferFunction -- each call renders
        its own numbers, so value/error legends can never share state;
      * it draws into a private OFF-SCREEN vtkRenderWindow, not a view in the
        shared layout, so it cannot capture the wrong view's content;
      * output buffer is RGB, exactly like the panel screenshots (no
        vtkImageAppend component mismatch);
      * a blank/uniform output raises, which triggers the next backend.
    """
    import vtk as _vtk   # bundled with ParaView

    plut = config_lut(color_var, cmap, cmin, cmax, log_scale)   # same LUT as panels
    pts = [float(v) for v in plut.RGBPoints]                     # [x,r,g,b, ...]

    ctf = _vtk.vtkColorTransferFunction()
    for i in range(0, len(pts) - 3, 4):
        ctf.AddRGBPoint(pts[i], pts[i + 1], pts[i + 2], pts[i + 3])
    if log_scale and cmax > 0:
        try:
            ctf.SetScaleToLog10()
        except Exception:
            pass

    bar = _vtk.vtkScalarBarActor()
    bar.SetLookupTable(ctf)
    bar.SetTitle(title)
    bar.SetOrientationToVertical()
    bar.SetNumberOfLabels(6)
    bar.SetLabelFormat('%-#.4g')
    try:
        bar.UnconstrainedFontSizeOn()
        bar.GetTitleTextProperty().SetFontSize(26)
        bar.GetLabelTextProperty().SetFontSize(20)
    except Exception:
        pass
    for tp in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        tp.SetColor(0.0, 0.0, 0.0)
        tp.ShadowOff()
        tp.ItalicOff()
    bar.GetTitleTextProperty().BoldOn()
    bar.SetPosition(0.08, 0.04)     # normalized viewport (lower-left corner)
    bar.SetPosition2(0.62, 0.90)    # width / height fractions
    try:
        bar.SetBarRatio(0.45)
    except Exception:
        pass

    ren = _vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AddActor2D(bar)
    rw = _vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)     # xvfb/osmesa safe; never opens a window
    rw.SetSize(LEGEND_WIDTH, PANEL_HEIGHT)
    rw.AddRenderer(ren)
    rw.Render()

    w2i = _vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    try:
        w2i.SetInputBufferTypeToRGB()   # 3 components, like SaveScreenshot
    except Exception:
        pass
    w2i.ReadFrontBufferOff()
    w2i.Update()
    wr = vtkPNGWriter()
    wr.SetFileName(out_png)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()
    rw.Finalize()

    # sanity: a blank (all-one-colour) legend means the off-screen render
    # failed silently -> raise so save_legend() falls back to the next backend
    if not os.path.exists(out_png) or os.path.getsize(out_png) == 0:
        raise RuntimeError("off-screen legend wrote no image")
    chk = vtkPNGReader(); chk.SetFileName(out_png); chk.Update()
    rng = chk.GetOutput().GetScalarRange()
    if rng[0] == rng[1]:
        raise RuntimeError("off-screen legend rendered a uniform image")
    return out_png


def _mpl_legend(title, cmin, cmax, cmap, out_png, log_scale=False):
    """Matplotlib colourbar (RGBA; the stitcher normalises to RGB)."""
    dpi = 100
    fig = _plt.figure(figsize=(LEGEND_WIDTH / dpi, PANEL_HEIGHT / dpi), dpi=dpi)
    cax = fig.add_axes([0.30, 0.08, 0.16, 0.84])   # [left, bottom, w, h]
    if log_scale and cmax > 0:
        lo = max(cmin, 1e-10)
        norm = _LogNorm(vmin=lo, vmax=max(cmax, lo * 10))
    else:
        norm = _Normalize(vmin=cmin, vmax=cmax)
    cb = fig.colorbar(_ScalarMappable(norm=norm, cmap=_mpl_cmap_name(cmap)), cax=cax)
    cb.set_label(title, fontsize=13)
    cb.ax.tick_params(labelsize=11)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    _plt.close(fig)
    return out_png


def _pvview_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale=False):
    """Last resort: screenshot of a scalar-bar-only RenderView."""
    lut = config_lut(color_var, cmap, cmin, cmax, log_scale)
    lv = CreateView('RenderView')
    lv.ViewSize = [LEGEND_WIDTH, PANEL_HEIGHT]
    lv.Background = [1.0, 1.0, 1.0]
    lv.OrientationAxesVisibility = 0
    bar = GetScalarBar(lut, lv)
    bar.Visibility      = 1
    bar.Title           = title
    bar.ComponentTitle  = ''
    bar.WindowLocation  = 'Any Location'
    bar.Position        = [0.28, 0.1]
    bar.ScalarBarLength = 0.8
    for attr, val in (("ScalarBarThickness", 18),
                      ("TitleColor", [0, 0, 0]), ("LabelColor", [0, 0, 0])):
        try:
            setattr(bar, attr, val)
        except Exception:
            pass
    SetActiveView(lv)               # make sure the screenshot targets THIS view
    Render(lv)
    SaveScreenshot(out_png, lv, ImageResolution=[LEGEND_WIDTH, PANEL_HEIGHT])
    Delete(lv); del lv
    return out_png


def save_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale=False):
    """
    Write a standalone colour-bar PNG (LEGEND_WIDTH x PANEL_HEIGHT) straight
    from the numeric range (cmin, cmax) + title, so the value legend and the
    error legend ALWAYS reflect their own distinct ranges. Backend order is
    set by LEGEND_BACKEND; failures fall through to the next backend.
    """
    order = (["mpl", "paraview", "pvview"] if str(LEGEND_BACKEND).lower() == "mpl"
             else ["paraview", "mpl", "pvview"])
    last_err = None
    for how in order:
        try:
            if how == "paraview":
                return _vtk_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale)
            if how == "mpl" and _HAS_MPL:
                return _mpl_legend(title, cmin, cmax, cmap, out_png, log_scale)
            if how == "pvview":
                return _pvview_legend(color_var, title, cmin, cmax, cmap, out_png, log_scale)
        except Exception as e:
            last_err = e
            print(f"      NOTE: legend backend '{how}' failed ({e}); trying next.")
    raise RuntimeError(f"all legend backends failed: {last_err}")


def render_panel(reader, view, color_var, warp_var, cmin, cmax, cmap,
                 steps, timesteps, out_dir, include_die, log_scale=False):
    """Save one PNG per selected timestep: workpiece coloured (+ moving die)."""
    os.makedirs(out_dir, exist_ok=True)
    warp = WarpByVector(Input=reader); warp.Vectors = ['POINTS', warp_var]

    work = workpiece(warp)
    dwork = Show(work, view)
    dwork.Representation = 'Surface With Edges' if SHOW_MESH_EDGES else 'Surface'
    if SHOW_MESH_EDGES:
        try:
            dwork.EdgeColor = EDGE_COLOR
        except Exception:
            pass
    Hide(reader, view)
    ColorBy(dwork, ('POINTS', color_var))
    config_lut(color_var, cmap, cmin, cmax, log_scale)
    dwork.SetScalarBarVisibility(view, False)     # no overlaid bar

    die = ddie = None
    if include_die and SHOW_DIE:
        die = die_part(warp)
        ddie = Show(die, view)
        ddie.Representation = 'Surface With Edges' if SHOW_MESH_EDGES else 'Surface'
        # Solid colour. Avoid ColorBy(rep, None): some builds choke on the
        # 'NONE' association string. Setting an empty ColorArrayName turns off
        # scalar colouring so DiffuseColor/AmbientColor are used instead.
        for cname in (['', ''], ['POINTS', ''], [None, '']):
            try:
                ddie.ColorArrayName = cname
                break
            except Exception:
                continue
        try:
            ddie.DiffuseColor = DIE_COLOR
            ddie.AmbientColor = DIE_COLOR
            ddie.Opacity      = DIE_OPACITY   # semi-transparent -> see workpiece
            if SHOW_MESH_EDGES:
                ddie.EdgeColor = EDGE_COLOR
        except Exception:
            pass

    view.Update()
    for k, t in enumerate(steps):
        fn = os.path.join(out_dir, f"frame.{k:04d}.png")
        if SKIP_EXISTING_PANELS and os.path.exists(fn):
            continue
        view.ViewTime = timesteps[t]
        Render(view)
        SaveScreenshot(fn, view, ImageResolution=[PANEL_WIDTH, PANEL_HEIGHT])

    Hide(work, view)
    if ddie is not None:
        Hide(die, view)
    Delete(dwork)
    if ddie is not None:
        Delete(ddie)
    Delete(work)
    if die is not None:
        Delete(die)
    Delete(warp)


def _rgb_port(reader, keep):
    """Output port guaranteed to carry 3-component (RGB) image data.

    ParaView screenshots are RGB but matplotlib PNGs are RGBA. vtkImageAppend
    refuses to mix component counts ('Components of the inputs do not match')
    and silently drops the mismatched input, which blanked the legend columns.
    Normalise every input to RGB before appending.
    """
    scalars = reader.GetOutput().GetPointData().GetScalars()
    nc = scalars.GetNumberOfComponents() if scalars is not None else 0
    if nc == 3:
        return reader.GetOutputPort()
    ex = vtkImageExtractComponents()
    ex.SetInputConnection(reader.GetOutputPort())
    if nc >= 3:
        ex.SetComponents(0, 1, 2)      # RGBA -> RGB (drop alpha)
    else:
        ex.SetComponents(0, 0, 0)      # grayscale -> RGB
    ex.Update()
    keep.append(ex)                    # keep filter alive until append is done
    return ex.GetOutputPort()


def stitch(sources, combined_dir, n):
    """sources: list of ('dir', path) per-frame or ('img', path) static."""
    os.makedirs(combined_dir, exist_ok=True)
    warned = False
    for k in range(n):
        readers = []
        for kind, p in sources:
            fn = os.path.join(p, f"frame.{k:04d}.png") if kind == 'dir' else p
            if not os.path.exists(fn):
                if not warned:
                    print(f"      WARNING: missing image {fn} -- skipped")
                    warned = True
                continue
            r = vtkPNGReader(); r.SetFileName(fn); r.Update()
            readers.append(r)
        if not readers:
            continue
        keep, h0 = [], None
        app = vtkImageAppend(); app.SetAppendAxis(0)
        for r in readers:
            ext = r.GetOutput().GetExtent()
            h = ext[3] - ext[2] + 1
            if h0 is None:
                h0 = h
            elif h != h0 and not warned:
                print(f"      WARNING: image heights differ ({h} vs {h0}); "
                      f"stitch may clip -- check LEGEND/PANEL sizes")
                warned = True
            app.AddInputConnection(_rgb_port(r, keep))
        app.Update()
        w = vtkPNGWriter()
        w.SetFileName(os.path.join(combined_dir, f"frame.{k:04d}.png"))
        w.SetInputConnection(app.GetOutputPort()); w.Write()
    print(f"      stitched {n} combined frames -> {combined_dir}")


def encode_mp4(png_dir, out_mp4):
    import subprocess
    cmd = ['ffmpeg', '-y', '-framerate', str(FRAME_RATE),
           '-i', os.path.join(png_dir, 'frame.%04d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(FRAME_RATE), out_mp4]
    try:
        if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
            return True
    except FileNotFoundError:
        pass
    print(f"      NOTE: ffmpeg not found; PNGs are in {png_dir}")
    return False


def encode_gif(png_dir, out_gif):
    import subprocess
    flt = (f'fps={FRAME_RATE},scale=-1:-1:flags=lanczos,'
           f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse')
    cmd = ['ffmpeg', '-y', '-framerate', str(FRAME_RATE),
           '-i', os.path.join(png_dir, 'frame.%04d.png'),
           '-filter_complex', flt, out_gif]
    try:
        if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


# ==============================================================================
# 3. PER-UNIT (model/geometry) PROCESSING
# ==============================================================================
def process_unit(label, pvd, view, sample_ranges=None):
    sample_ranges = sample_ranges or {}
    skey = sample_key(label)
    out_dir = os.path.join(OUTPUT_ROOT, label)
    print(f"\n{'#'*70}\n  UNIT: {label}\n  in : {pvd}\n  out: {out_dir}\n{'#'*70}")
    if not os.path.exists(pvd):
        print("  ERROR: series.pvd not found -- skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)

    reg = label.replace(os.sep, "_").replace("-", "_")
    reader = PVDReader(registrationName=f'pvd_{reg}', FileName=pvd)
    timesteps = reader.TimestepValues
    if timesteps is None:
        timesteps = [0]
    T = len(timesteps)
    steps = list(range(0, T, SKIP))
    step_times = [timesteps[t] for t in steps]
    print(f"  {T} timesteps -> rendering {len(steps)} (skip={SKIP})")

    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
    GetLayout().SetSize(PANEL_WIDTH, PANEL_HEIGHT)
    set_camera(view, reader)

    # array availability check (peek at first timestep)
    reader.UpdatePipeline(timesteps[0])
    pdi = reader.GetDataInformation().GetPointDataInformation()
    available = {pdi.GetArrayInformation(i).GetName() for i in range(pdi.GetNumberOfArrays())}
    print(f"  arrays: {sorted(available)}")

    for task in COMPARISON_TASKS:
        label_t = task["label"]
        gt_c, pred_c, err_c = task["gt"], task["pred"], task["error"]
        err_log = task.get("err_log", False)
        value_title = task.get("value_title", gt_c[0])
        error_title = task.get("error_title", err_c[0])
        print(f"\n  {'='*56}\n  {label_t}\n  {'='*56}")

        needed = {gt_c[0], pred_c[0], err_c[0], gt_c[1], pred_c[1], err_c[1], 'is_die'}
        missing = needed - available
        if missing:
            print(f"  SKIP '{label_t}' -- missing arrays: {sorted(missing)}")
            continue

        # ranges on the WORKPIECE only. Prefer the shared per-sample range (same
        # legend across all models of this geometry); otherwise compute per-unit.
        key = (skey, label_t)
        if SHARE_LEGEND_ACROSS_MODELS and key in sample_ranges:
            value_min, value_max, emax = sample_ranges[key]
            src = f"shared per-sample ({skey})"
        else:
            gmin, gmax = workpiece_range(reader, gt_c[0],   gt_c[1],   step_times)
            pmin, pmax = workpiece_range(reader, pred_c[0], pred_c[1], step_times)
            _, emax    = workpiece_range(reader, err_c[0],  err_c[1],  step_times)
            value_min, value_max = min(gmin, pmin), max(gmax, pmax)
            src = "per-unit"
        error_min, error_max = 0.0, emax
        ov = RANGE_OVERRIDES.get(key, {})
        if "value" in ov:
            value_min, value_max = ov["value"]; src += " +value-override"
        if "error" in ov:
            error_min, error_max = ov["error"]; src += " +error-override"
        print(f"  value range (GT&Pred): [{value_min:.4g}, {value_max:.4g}]  ({src})")
        print(f"  error range: [{error_min:.4g}, {error_max:.4g}]  ({src})")

        base     = os.path.join(out_dir, "frames", label_t)
        gt_dir   = os.path.join(base, "gt")
        pred_dir = os.path.join(base, "pred")
        err_dir  = os.path.join(base, "error")
        comb_dir = os.path.join(base, "combined")

        # legends as their own files (beside, not overlaid)
        val_leg = os.path.join(out_dir, f"legend_{label_t}_value.png")
        err_leg = os.path.join(out_dir, f"legend_{label_t}_error.png")
        save_legend(pred_c[0], value_title, value_min, value_max, VALUE_CMAP, val_leg)
        save_legend(err_c[0],  error_title, error_min, error_max, ERROR_CMAP, err_leg, err_log)
        SetActiveView(view)

        print("  rendering GT ...")
        render_panel(reader, view, gt_c[0], gt_c[1], value_min, value_max, VALUE_CMAP,
                     steps, timesteps, gt_dir, include_die=True)
        print("  rendering Pred ...")
        render_panel(reader, view, pred_c[0], pred_c[1], value_min, value_max, VALUE_CMAP,
                     steps, timesteps, pred_dir, include_die=True)
        print("  rendering Error ...")
        render_panel(reader, view, err_c[0], err_c[1], error_min, error_max, ERROR_CMAP,
                     steps, timesteps, err_dir, include_die=True, log_scale=err_log)

        if STITCH_WITH_LEGEND:
            sources = [('dir', gt_dir), ('dir', pred_dir), ('img', val_leg),
                       ('dir', err_dir), ('img', err_leg)]
        else:
            sources = [('dir', gt_dir), ('dir', pred_dir), ('dir', err_dir)]
        stitch(sources, comb_dir, len(steps))

        if MAKE_MP4 and encode_mp4(comb_dir, os.path.join(out_dir, f"{label_t}.mp4")):
            print(f"  mp4 -> {label_t}.mp4")
        if MAKE_GIF and encode_gif(comb_dir, os.path.join(out_dir, f"{label_t}.gif")):
            print(f"  gif -> {label_t}.gif")

        if not KEEP_FRAMES:
            for d in (gt_dir, pred_dir, err_dir, comb_dir):
                shutil.rmtree(d, ignore_errors=True)

    Delete(reader); del reader


# ==============================================================================
# 4. MAIN
# ==============================================================================
def parse_args():
    """--shard I --num-shards N --scan-only; unknown (pvbatch) args ignored."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--vtu-root", default=None, help="Folder written by convert_pkls_to_vtu.py")
    p.add_argument("--models", nargs="*", default=None, help="Only render these model labels")
    p.add_argument("--skip", type=int, default=None, help="Render every SKIP-th frame (default 10)")
    p.add_argument("--out-root", default=None, help="Folder that receives the rendered figures")
    args, _ = p.parse_known_args()
    return args


def main():
    global VTU_ROOT, OUTPUT_ROOT, MODELS, SKIP
    args = parse_args()
    if args.vtu_root:
        VTU_ROOT = args.vtu_root
    if args.out_root:
        OUTPUT_ROOT = args.out_root
    if args.models:
        MODELS = args.models
    if args.skip:
        SKIP = args.skip
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    paraview.simple._DisableFirstRenderCameraReset()

    view = GetActiveViewOrCreate('RenderView')
    view.UseColorPaletteForBackground = 0
    view.Background                = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility      = 0
    try:
        view.UseFXAA = 1 if USE_FXAA else 0
    except Exception:
        pass

    units = discover_units()   # FULL list -- sharding must not alter shared legends
    if not units:
        print(f"No series.pvd found under {VTU_ROOT}. Run convert_pkls_to_vtu.py first.")
        return
    print(f"Units found ({len(units)}):")
    for label, _ in units:
        print(f"   - {label}")

    sample_ranges = {}
    if SHARE_LEGEND_ACROSS_MODELS:
        sample_ranges = load_ranges_cache(units)
        if sample_ranges is None:
            sample_ranges = precompute_sample_ranges(units)
            save_ranges_cache(sample_ranges)
    if args.scan_only:
        print("--scan-only: shared ranges cached; exiting.")
        return

    if args.num_shards > 1:
        units = [u for i, u in enumerate(units) if i % args.num_shards == args.shard]
        print(f"Shard {args.shard + 1}/{args.num_shards}: {len(units)} unit(s): "
              + ", ".join(l for l, _ in units))

    for idx, (label, pvd) in enumerate(units, 1):
        print(f"\n{'*'*70}\n  {idx}/{len(units)}: {label}\n{'*'*70}")
        process_unit(label, pvd, view, sample_ranges)

    print(f"\n{'='*70}\n  ALL DONE -> {OUTPUT_ROOT}\n{'='*70}")


if __name__ == "__main__":
    main()