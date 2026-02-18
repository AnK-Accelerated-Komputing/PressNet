"""
Module for computing error metrics for each die shape before animation.
This computes per-shape statistics like max error, mean error, RMSE, etc.
"""

import torch
import numpy as np
import json
import pickle
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class ShapeMetrics:
    """Container for storing error metrics for a single shape"""
    shape_id: str
    num_nodes: int
    num_timesteps: int
    
    # Position errors (L2 norm)
    max_pos_error: float
    mean_pos_error: float
    median_pos_error: float
    std_pos_error: float
    
    # Displacement errors (Y-direction deformation)
    max_deform_y: float
    mean_deform_y: float
    median_deform_y: float
    std_deform_y: float
    
    # RMSE metrics
    rmse_position: float
    rmse_deformation: float
    
    # Stress metrics (if available)
    max_stress_error: Optional[float] = None
    mean_stress_error: Optional[float] = None
    rmse_stress: Optional[float] = None
    
    # Spatial domain metrics
    domain_wise_rmse: Optional[Dict] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def print_summary(self):
        """Print a nice summary of metrics"""
        print(f"\n{'='*60}")
        print(f"SHAPE: {self.shape_id}")
        print(f"{'='*60}")
        print(f"Nodes: {self.num_nodes} | Timesteps: {self.num_timesteps}")
        print(f"\n--- POSITION ERROR METRICS ---")
        print(f"  Max Error:      {self.max_pos_error:.6f}")
        print(f"  Mean Error:     {self.mean_pos_error:.6f}")
        print(f"  Median Error:   {self.median_pos_error:.6f}")
        print(f"  Std Dev:        {self.std_pos_error:.6f}")
        print(f"  RMSE:           {self.rmse_position:.6f}")
        
        print(f"\n--- DEFORMATION (Y) METRICS ---")
        print(f"  Max Deform:     {self.max_deform_y:.6f}")
        print(f"  Mean Deform:    {self.mean_deform_y:.6f}")
        print(f"  Median Deform:  {self.median_deform_y:.6f}")
        print(f"  Std Dev:        {self.std_deform_y:.6f}")
        print(f"  RMSE:           {self.rmse_deformation:.6f}")
        
        if self.max_stress_error is not None:
            print(f"\n--- STRESS ERROR METRICS ---")
            print(f"  Max Error:      {self.max_stress_error:.6f}")
            print(f"  Mean Error:     {self.mean_stress_error:.6f}")
            print(f"  RMSE:           {self.rmse_stress:.6f}")
        
        print(f"{'='*60}\n")


def compute_position_errors(pred_pos: torch.Tensor, gt_pos: torch.Tensor, 
                           mesh_pos: torch.Tensor, node_type: torch.Tensor) -> Dict:
    """
    Compute position-based error metrics.
    
    Args:
        pred_pos: Predicted positions (Steps, Nodes, 3)
        gt_pos: Ground truth positions (Steps, Nodes, 3)
        mesh_pos: Mesh/reference positions (Steps, Nodes, 3)
        node_type: Node type mask (Steps, Nodes) - 0 for material, 1 for die
    
    Returns:
        Dictionary containing various position error metrics
    """
    # Move to CPU for computation
    pred_pos = pred_pos.to('cpu')
    gt_pos = gt_pos.to('cpu')
    mesh_pos = mesh_pos.to('cpu')
    node_type = node_type.to('cpu').flatten()
    
    # Create mask for material nodes (not die)
    material_mask = (node_type == 0)
    
    # Compute L2 norm of position error at each node
    pos_error = torch.norm(pred_pos - gt_pos, dim=-1)  # (Steps, Nodes)
    
    # Filter by material nodes
    filtered_errors = pos_error[:, material_mask].flatten()
    
    metrics = {
        'max_pos_error': float(torch.max(filtered_errors).item()),
        'mean_pos_error': float(torch.mean(filtered_errors).item()),
        'median_pos_error': float(torch.median(filtered_errors).item()),
        'std_pos_error': float(torch.std(filtered_errors).item()),
        'all_errors': filtered_errors.numpy()  # For histograms/analysis
    }
    
    return metrics


def compute_deformation_errors(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                              mesh_pos: torch.Tensor, node_type: torch.Tensor) -> Dict:
    """
    Compute Y-direction deformation error metrics.
    
    Args:
        pred_pos: Predicted positions (Steps, Nodes, 3)
        gt_pos: Ground truth positions (Steps, Nodes, 3)
        mesh_pos: Mesh/reference positions (Steps, Nodes, 3)
        node_type: Node type mask (Steps, Nodes)
    
    Returns:
        Dictionary containing deformation error metrics
    """
    pred_pos = pred_pos.to('cpu')
    gt_pos = gt_pos.to('cpu')
    mesh_pos = mesh_pos.to('cpu')
    node_type = node_type.to('cpu').flatten()
    
    material_mask = (node_type == 0)
    
    # Compute Y-displacement (absolute deformation)
    gt_y_deform = torch.abs((gt_pos - mesh_pos)[:, :, 1])  # (Steps, Nodes)
    pred_y_deform = torch.abs((pred_pos - mesh_pos)[:, :, 1])  # (Steps, Nodes)
    
    # Compute error in deformation
    deform_error = torch.abs(pred_y_deform - gt_y_deform)
    
    # Filter by material nodes
    filtered_deform_errors = deform_error[:, material_mask].flatten()
    filtered_gt_deform = gt_y_deform[:, material_mask].flatten()
    filtered_pred_deform = pred_y_deform[:, material_mask].flatten()
    
    # RMSE of deformation
    rmse_deform = torch.sqrt(torch.mean((filtered_pred_deform - filtered_gt_deform) ** 2))
    
    metrics = {
        'max_deform_y': float(torch.max(filtered_deform_errors).item()),
        'mean_deform_y': float(torch.mean(filtered_deform_errors).item()),
        'median_deform_y': float(torch.median(filtered_deform_errors).item()),
        'std_deform_y': float(torch.std(filtered_deform_errors).item()),
        'rmse_deformation': float(rmse_deform.item()),
        'all_deform_errors': filtered_deform_errors.numpy()
    }
    
    return metrics


def compute_stress_errors(pred_stress: torch.Tensor, gt_stress: torch.Tensor,
                         node_type: torch.Tensor) -> Dict:
    """
    Compute stress-based error metrics.
    
    Args:
        pred_stress: Predicted stress values (Steps, Nodes)
        gt_stress: Ground truth stress values (Steps, Nodes)
        node_type: Node type mask (Steps, Nodes)
    
    Returns:
        Dictionary containing stress error metrics
    """
    pred_stress = pred_stress.to('cpu').flatten()
    gt_stress = gt_stress.to('cpu').flatten()
    node_type = node_type.to('cpu').flatten()
    
    material_mask = (node_type == 0)
    
    # Filter by material nodes
    filtered_pred = pred_stress[material_mask]
    filtered_gt = gt_stress[material_mask]
    
    # Compute errors
    stress_error = torch.abs(filtered_pred - filtered_gt)
    rmse_stress = torch.sqrt(torch.mean((filtered_pred - filtered_gt) ** 2))
    
    metrics = {
        'max_stress_error': float(torch.max(stress_error).item()),
        'mean_stress_error': float(torch.mean(stress_error).item()),
        'rmse_stress': float(rmse_stress.item()),
        'all_stress_errors': stress_error.numpy()
    }
    
    return metrics


def compute_rmse_position(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                         node_type: torch.Tensor) -> float:
    """
    Compute overall RMSE for position predictions.
    """
    pred_pos = pred_pos.to('cpu')
    gt_pos = gt_pos.to('cpu')
    node_type = node_type.to('cpu').flatten()
    
    material_mask = (node_type == 0)
    
    error = pred_pos[:, material_mask, :] - gt_pos[:, material_mask, :]
    rmse = torch.sqrt(torch.mean(error ** 2))
    
    return float(rmse.item())


def compute_domain_wise_metrics(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                               mesh_pos: torch.Tensor, node_type: torch.Tensor,
                               num_domains: int = 20) -> Dict:
    """
    Compute error metrics subdivided by spatial domain (X-axis regions).
    Useful for understanding where the model performs well/poorly.
    
    Args:
        pred_pos, gt_pos, mesh_pos, node_type: Trajectory data
        num_domains: Number of spatial regions to divide the mesh into
    
    Returns:
        Dictionary with per-domain metrics
    """
    pred_pos = pred_pos.to('cpu')
    gt_pos = gt_pos.to('cpu')
    mesh_pos = mesh_pos.to('cpu')
    node_type = node_type.to('cpu').flatten()
    
    material_mask = (node_type == 0)
    
    # Get X-axis bounds
    x_coords = mesh_pos[:, :, 0]
    x_min = x_coords.min().item()
    x_max = x_coords.max().item()
    
    domain_size = (x_max - x_min) / num_domains
    
    domain_metrics = {}
    
    for i in range(num_domains):
        x_start = x_min + i * domain_size
        x_end = x_start + domain_size
        
        # Create domain mask
        x_mask = (x_coords >= x_start) & (x_coords < x_end)
        domain_mask = x_mask & material_mask.unsqueeze(0)
        
        if domain_mask.sum() == 0:
            continue
        
        # Compute metrics for this domain
        domain_error = torch.norm(pred_pos[domain_mask] - gt_pos[domain_mask], dim=-1)
        domain_rmse = torch.sqrt(torch.mean((pred_pos[domain_mask] - gt_pos[domain_mask]) ** 2))
        
        domain_metrics[f"domain_{i:02d}_{x_start:.2f}_to_{x_end:.2f}"] = {
            'max_error': float(torch.max(domain_error).item()),
            'mean_error': float(torch.mean(domain_error).item()),
            'rmse': float(domain_rmse.item()),
            'num_nodes': int(domain_mask.sum().item())
        }
    
    return domain_metrics


def evaluate_single_shape(trajectory_data: Dict, shape_id: str,
                         include_stress: bool = False,
                         include_domain_wise: bool = True) -> ShapeMetrics:
    """
    Compute all error metrics for a single shape/trajectory.
    
    Args:
        trajectory_data: Dictionary containing:
            - 'pred_pos': Predicted positions (Steps, Nodes, 3)
            - 'gt_pos': Ground truth positions (Steps, Nodes, 3)
            - 'mesh_pos': Mesh positions (Steps, Nodes, 3)
            - 'node_type': Node types (Steps, Nodes)
            - 'pred_stress' (optional): Predicted stress
            - 'gt_stress' (optional): Ground truth stress
        shape_id: Identifier for this shape
        include_stress: Whether to compute stress metrics
        include_domain_wise: Whether to compute spatial domain metrics
    
    Returns:
        ShapeMetrics object containing all computed metrics
    """
    
    pred_pos = trajectory_data['pred_pos']
    gt_pos = trajectory_data['gt_pos']
    mesh_pos = trajectory_data['mesh_pos']
    node_type = trajectory_data['node_type']
    
    num_timesteps = pred_pos.shape[0]
    num_nodes = pred_pos.shape[1]
    
    # Compute position errors
    pos_metrics = compute_position_errors(pred_pos, gt_pos, mesh_pos, node_type)
    
    # Compute deformation errors
    deform_metrics = compute_deformation_errors(pred_pos, gt_pos, mesh_pos, node_type)
    
    # Compute overall RMSE for position
    rmse_pos = compute_rmse_position(pred_pos, gt_pos, node_type)
    
    # Optional: Stress metrics
    stress_metrics = None
    if include_stress and 'pred_stress' in trajectory_data and 'gt_stress' in trajectory_data:
        stress_metrics = compute_stress_errors(
            trajectory_data['pred_stress'],
            trajectory_data['gt_stress'],
            node_type
        )
    
    # Optional: Domain-wise metrics
    domain_metrics = None
    if include_domain_wise:
        domain_metrics = compute_domain_wise_metrics(pred_pos, gt_pos, mesh_pos, node_type)
    
    # Create ShapeMetrics object
    metrics = ShapeMetrics(
        shape_id=shape_id,
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
        max_pos_error=pos_metrics['max_pos_error'],
        mean_pos_error=pos_metrics['mean_pos_error'],
        median_pos_error=pos_metrics['median_pos_error'],
        std_pos_error=pos_metrics['std_pos_error'],
        max_deform_y=deform_metrics['max_deform_y'],
        mean_deform_y=deform_metrics['mean_deform_y'],
        median_deform_y=deform_metrics['median_deform_y'],
        std_deform_y=deform_metrics['std_deform_y'],
        rmse_position=rmse_pos,
        rmse_deformation=deform_metrics['rmse_deformation'],
        max_stress_error=stress_metrics['max_stress_error'] if stress_metrics else None,
        mean_stress_error=stress_metrics['mean_stress_error'] if stress_metrics else None,
        rmse_stress=stress_metrics['rmse_stress'] if stress_metrics else None,
        domain_wise_rmse=domain_metrics
    )
    
    return metrics


def evaluate_multiple_shapes(trajectories_list: List[Dict], shape_ids: List[str],
                            include_stress: bool = False,
                            verbose: bool = True) -> Dict[str, ShapeMetrics]:
    """
    Evaluate error metrics for multiple shapes.
    
    Args:
        trajectories_list: List of trajectory dictionaries
        shape_ids: List of shape identifiers corresponding to trajectories
        include_stress: Whether to compute stress metrics
        verbose: Whether to print summaries
    
    Returns:
        Dictionary mapping shape_id -> ShapeMetrics
    """
    
    all_metrics = {}
    
    for shape_id, trajectory in zip(shape_ids, trajectories_list):
        metrics = evaluate_single_shape(trajectory, shape_id, include_stress=include_stress)
        all_metrics[shape_id] = metrics
        
        if verbose:
            metrics.print_summary()
    
    return all_metrics


def create_metrics_comparison_report(metrics_dict: Dict[str, ShapeMetrics],
                                     output_file: str) -> None:
    """
    Create a summary report comparing metrics across all shapes.
    
    Args:
        metrics_dict: Dictionary of shape metrics
        output_file: Path to save JSON report
    """
    
    # Convert to serializable format
    report = {
        shape_id: metrics.to_dict()
        for shape_id, metrics in metrics_dict.items()
    }
    
    # Add aggregate statistics
    aggregate = {
        'total_shapes': len(metrics_dict),
        'mean_max_pos_error': np.mean([m.max_pos_error for m in metrics_dict.values()]),
        'mean_rmse_position': np.mean([m.rmse_position for m in metrics_dict.values()]),
        'mean_rmse_deformation': np.mean([m.rmse_deformation for m in metrics_dict.values()]),
    }
    
    report['aggregate_statistics'] = aggregate
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nMetrics report saved to: {output_file}")


def filter_shapes_by_threshold(metrics_dict: Dict[str, ShapeMetrics],
                              metric_name: str = 'max_pos_error',
                              threshold: float = None,
                              percentile: int = None) -> Dict[str, ShapeMetrics]:
    """
    Filter shapes based on error metrics to identify problematic cases.
    Useful for focusing animation on worst-performing shapes.
    
    Args:
        metrics_dict: Dictionary of shape metrics
        metric_name: Which metric to filter by ('max_pos_error', 'rmse_position', etc.)
        threshold: Absolute threshold (shapes > threshold are returned)
        percentile: Percentile threshold (shapes in top N% are returned)
    
    Returns:
        Filtered dictionary of problematic shapes
    """
    
    # Extract metric values
    metric_values = [getattr(m, metric_name) for m in metrics_dict.values()]
    
    if percentile is not None:
        threshold = np.percentile(metric_values, percentile)
    elif threshold is None:
        raise ValueError("Either threshold or percentile must be specified")
    
    filtered = {
        shape_id: metrics
        for shape_id, metrics in metrics_dict.items()
        if getattr(metrics, metric_name) > threshold
    }
    
    print(f"\nFiltered {len(filtered)} shapes with {metric_name} > {threshold:.6f}")
    
    return filtered


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: How to use this module
    """
    
    # Example 1: Evaluate a single shape
    print("EXAMPLE 1: Evaluate Single Shape")
    print("-" * 70)
    
    # Load your trajectory data (from pickle, model inference, etc.)
    # trajectory_data = {
    #     'pred_pos': pred_pos_tensor,  # (Steps, Nodes, 3)
    #     'gt_pos': gt_pos_tensor,      # (Steps, Nodes, 3)
    #     'mesh_pos': mesh_pos_tensor,  # (Steps, Nodes, 3)
    #     'node_type': node_type_tensor # (Steps, Nodes)
    # }
    
    # metrics = evaluate_single_shape(trajectory_data, shape_id="die_001")
    # metrics.print_summary()
    
    
    # Example 2: Evaluate multiple shapes and create comparison report
    print("\nEXAMPLE 2: Evaluate Multiple Shapes")
    print("-" * 70)
    
    # all_metrics = evaluate_multiple_shapes(
    #     trajectories_list=[traj1, traj2, traj3],
    #     shape_ids=["die_001", "die_002", "die_003"],
    #     include_stress=True,
    #     verbose=True
    # )
    
    # create_metrics_comparison_report(
    #     all_metrics,
    #     output_file="metrics_report.json"
    # )
    
    
    # Example 3: Find problematic shapes for detailed animation
    print("\nEXAMPLE 3: Find Problematic Shapes")
    print("-" * 70)
    
    # problematic_shapes = filter_shapes_by_threshold(
    #     all_metrics,
    #     metric_name='max_pos_error',
    #     percentile=90  # Top 10% worst performing
    # )
    
    # # Now animate only these problematic shapes for detailed inspection
    # for shape_id, metrics in problematic_shapes.items():
    #     print(f"Animating {shape_id}...")
    #     save_rollout_frames(trajectory_data, save_directory=f"output/{shape_id}")
    
    print("\n✓ Module ready to use. See examples above.")
