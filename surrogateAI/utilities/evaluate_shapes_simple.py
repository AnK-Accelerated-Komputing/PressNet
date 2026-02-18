#!/usr/bin/env python3
"""
SIMPLE SHAPE METRICS EVALUATOR
Evaluate error metrics for each shape from pickle files
Generate summary table and report

Usage:
    python evaluate_shapes_simple.py --pkl_file rollout_data.pkl
    python evaluate_shapes_simple.py --pkl_file rollout_data.pkl --output summary.csv
"""

import torch
import pickle
import json
import csv
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from evaluate_shape_metrics import (
    evaluate_single_shape,
    evaluate_multiple_shapes,
    filter_shapes_by_threshold
)


def load_pkl(pkl_file):
    """Load pickle file with trajectories"""
    print(f"Loading: {pkl_file}")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"File not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        trajectories = data
        shape_ids = [f"shape_{i:04d}" for i in range(len(data))]
    elif isinstance(data, dict):
        trajectories = list(data.values())
        shape_ids = list(data.keys())
    else:
        raise ValueError("Pickle must contain list or dict of trajectories")
    
    print(f"✓ Loaded {len(trajectories)} shapes\n")
    return trajectories, shape_ids


def evaluate_all(trajectories, shape_ids):
    """Evaluate all shapes"""
    print("Evaluating shapes...")
    metrics = evaluate_multiple_shapes(
        trajectories_list=trajectories,
        shape_ids=shape_ids,
        verbose=False
    )
    print(f"✓ Evaluated {len(metrics)} shapes\n")
    return metrics


def print_summary_table(metrics):
    """Print summary table to console"""
    print("="*95)
    print("SUMMARY TABLE")
    print("="*95)
    print(f"{'Shape ID':<20} {'Max Error':<12} {'Mean Error':<12} {'RMSE Pos':<12} "
          f"{'Max Deform':<12} {'RMSE Deform':<12} {'Nodes':<10} {'Steps':<10}")
    print("-"*95)
    
    for shape_id, m in sorted(metrics.items()):
        print(f"{shape_id:<20} {m.max_pos_error:<12.6f} {m.mean_pos_error:<12.6f} "
              f"{m.rmse_position:<12.6f} {m.max_deform_y:<12.6f} {m.rmse_deformation:<12.6f} "
              f"{m.num_nodes:<10} {m.num_timesteps:<10}")
    
    print("="*95 + "\n")


def export_csv(metrics, output_file):
    """Export to CSV"""
    print(f"Saving CSV: {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Shape_ID', 'Num_Nodes', 'Num_Timesteps',
            'Max_Error', 'Mean_Error', 'Median_Error', 'Std_Error', 'RMSE_Position',
            'Max_Deform_Y', 'Mean_Deform_Y', 'Median_Deform_Y', 'Std_Deform_Y', 'RMSE_Deform'
        ])
        
        # Data
        for shape_id, m in sorted(metrics.items()):
            writer.writerow([
                shape_id,
                m.num_nodes,
                m.num_timesteps,
                f"{m.max_pos_error:.6f}",
                f"{m.mean_pos_error:.6f}",
                f"{m.median_pos_error:.6f}",
                f"{m.std_pos_error:.6f}",
                f"{m.rmse_position:.6f}",
                f"{m.max_deform_y:.6f}",
                f"{m.mean_deform_y:.6f}",
                f"{m.median_deform_y:.6f}",
                f"{m.std_deform_y:.6f}",
                f"{m.rmse_deformation:.6f}"
            ])
    
    print(f"✓ Saved to {output_file}\n")


def export_json(metrics, output_file):
    """Export to JSON"""
    print(f"Saving JSON: {output_file}")
    
    report = {
        shape_id: m.to_dict()
        for shape_id, m in metrics.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Saved to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate shape metrics from pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_shapes_simple.py --pkl_file rollout.pkl
  python evaluate_shapes_simple.py --pkl_file data.pkl --csv output.csv
  python evaluate_shapes_simple.py --pkl_file data.pkl --json report.json
        """
    )
    
    parser.add_argument('--pkl_file', type=str, required=True,
                       help='Path to pickle file with trajectories')
    parser.add_argument('--csv', type=str, default='metrics_summary.csv',
                       help='Output CSV file (default: metrics_summary.csv)')
    parser.add_argument('--json', type=str, default=None,
                       help='Output JSON file (default: None)')
    
    args = parser.parse_args()
    
    print("\n" + "="*95)
    print("SHAPE METRICS EVALUATOR")
    print("="*95 + "\n")
    
    # Load
    trajectories, shape_ids = load_pkl(args.pkl_file)
    
    # Evaluate
    metrics = evaluate_all(trajectories, shape_ids)
    
    # Print summary
    print_summary_table(metrics)
    
    # Export CSV
    export_csv(metrics, args.csv)
    
    # Export JSON if requested
    if args.json:
        export_json(metrics, args.json)
    
    print("="*95)
    print("DONE!")
    print("="*95 + "\n")


if __name__ == "__main__":
    main()
