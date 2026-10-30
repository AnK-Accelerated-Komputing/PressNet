#!/usr/bin/env python3
"""
analyze_timing.py
Post-process the rollout_info.json produced by inference.py
into a timing-summary JSON (per-trajectory time and ms/step, per stage and total).

Reports per-trajectory time (mean / std / median / IQR) and ms/step, per stage
and for the full rollout, after discarding cold-start (warm-up) trajectories.

Usage
-----
    python -m utilities.analyze_timing --input /path/to/rollout_info.json
    python -m utilities.analyze_timing -i rollout_info.json -o timing_summary.json -w 3 --label "Transolver-Fine"
"""
import os
import json
import argparse
import numpy as np


def _summarize(times, steps=None):
    """Statistics for one series of per-trajectory times (seconds)."""
    t = np.asarray(times, dtype=float)
    out = {
        "n": int(t.size),
        "per_traj_mean_s": float(t.mean()),
        "per_traj_std_s": float(t.std(ddof=1)) if t.size > 1 else 0.0,
        "per_traj_median_s": float(np.median(t)),
        "per_traj_iqr_s": float(np.percentile(t, 75) - np.percentile(t, 25)),
        "per_traj_min_s": float(t.min()),
        "per_traj_max_s": float(t.max()),
    }
    if steps is not None:
        s = np.asarray(steps, dtype=float)
        out["n_steps_mean"] = float(s.mean())
        valid = s > 0
        if valid.any():
            ms = t[valid] / s[valid] * 1e3  # milliseconds per step
            out["ms_per_step_mean"] = float(ms.mean())
            out["ms_per_step_std"] = float(ms.std(ddof=1)) if ms.size > 1 else 0.0
            out["ms_per_step_median"] = float(np.median(ms))
        else:
            out["ms_per_step_mean"] = out["ms_per_step_std"] = out["ms_per_step_median"] = None
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Summarize rollout_info.json timing into a JSON.")
    ap.add_argument("--input", "-i", default="rollout_info.json",
                    help="Path to rollout_info.json (default: ./rollout_info.json)")
    ap.add_argument("--output", "-o", default=None,
                    help="Output JSON path (default: <input_dir>/timing_summary.json)")
    ap.add_argument("--warmup", "-w", type=int, default=3,
                    help="Initial trajectories to drop as cold-start (default: 3)")
    ap.add_argument("--label", default=None,
                    help="Optional model/mesh label stored in the summary, e.g. 'Transolver-Fine'")
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)

    n_total = len(d.get("inference_time", []))
    if n_total == 0:
        raise SystemExit(f"No 'inference_time' entries found in {args.input}")

    warm = args.warmup
    if warm >= n_total:
        print(f"[warn] warmup ({warm}) >= trajectories ({n_total}); using warmup=0.")
        warm = 0
    n_used = n_total - warm

    # Step counts are written by inference.py. Degrade gracefully if absent.
    has_steps = all(k in d for k in ("stage_1_steps", "stage_2_steps", "stage_3_steps"))
    if has_steps:
        s1, s2, s3 = d["stage_1_steps"][warm:], d["stage_2_steps"][warm:], d["stage_3_steps"][warm:]
        total_steps = [a + b + c for a, b, c in zip(s1, s2, s3)]
    else:
        print("[warn] stage_*_steps not found -> ms/step omitted. "
              "Re-run inference.py to enable per-step numbers.")
        s1 = s2 = s3 = total_steps = None

    summary = {
        "source_file": os.path.abspath(args.input),
        "label": args.label,
        "n_trajectories_total": n_total,
        "warmup_dropped": warm,
        "n_trajectories_used": n_used,
        "units": {"time": "seconds", "per_step": "milliseconds"},
        "total": _summarize(d["inference_time"][warm:], total_steps),
    }
    if "stage_1_time" in d:
        summary["stage_1_press"] = _summarize(d["stage_1_time"][warm:], s1)
    if "stage_2_time" in d:
        summary["stage_2_dwell"] = _summarize(d["stage_2_time"][warm:], s2)
    if "stage_3_time" in d:
        summary["stage_3_release"] = _summarize(d["stage_3_time"][warm:], s3)

    # Compact row (rounded).
    tot = summary["total"]
    row = {
        "per_trajectory_s": f"{tot['per_traj_mean_s']:.3f} +/- {tot['per_traj_std_s']:.3f}",
        "per_trajectory_median_s": f"{tot['per_traj_median_s']:.3f}",
    }
    if tot.get("ms_per_step_mean") is not None:
        row["ms_per_step"] = f"{tot['ms_per_step_mean']:.3f} +/- {tot['ms_per_step_std']:.3f}"
        row["ms_per_step_median"] = f"{tot['ms_per_step_median']:.3f}"
    summary["summary_row"] = row

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "timing_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Console echo
    print(f"\nSaved timing summary -> {out_path}")
    print(f"  trajectories: {n_used} used ({warm} warm-up dropped of {n_total})")
    print(f"  per-trajectory: {row['per_trajectory_s']} s  (median {row['per_trajectory_median_s']} s)")
    if "ms_per_step" in row:
        print(f"  per-step:       {row['ms_per_step']} ms  (median {row['ms_per_step_median']} ms)")


if __name__ == "__main__":
    main()