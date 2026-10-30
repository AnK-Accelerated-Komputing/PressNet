"""Re-draw the step-wise error curves from the CSVs written by ``evaluate_metrics.py``.

    python -m utilities.plot_stepwise_curves --csv_dir <eval_dir>/plot_data_csvs --out figures \
        --ylim nrmse_deform 0 0.30 --ylim nrmse_stress 0 0.365 --transitions 500 1000
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replot_from_csv(csv_dir, output_dir, custom_ylims=None, transition_steps=None):
    """
    Reads previously saved CSVs and generates plots with custom Y-axis limits 
    and specific vertical transition lines.
    """
    if custom_ylims is None:
        custom_ylims = {}
    
    if transition_steps is None:
        transition_steps = []

    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all CSV files in the directory
    for file in os.listdir(csv_dir):
        if not file.endswith("_plot_data.csv"):
            continue
            
        metric_name = file.replace("_plot_data.csv", "")
        csv_path = os.path.join(csv_dir, file)
        
        # Load the data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            continue
            
        real_time_steps = df['Timestep'].values
        mean_err = df['Mean'].values
        std_err = df['Std_Dev'].values
        min_err = df['Min'].values
        max_err = df['Max'].values

        plt.figure(figsize=(10, 6))

        # Re-apply original logic for colors and std dev bounds
        if 'r2' in metric_name:
            color_theme = 'green'
            lower_std = mean_err - std_err
            upper_std = np.minimum(mean_err + std_err, 1.0)
            y_label = 'Spatial R² Score'
            default_ylim = (max(0.0, np.min(min_err)*0.9), 1.05)
        else:
            color_theme = 'blue'
            lower_std = np.maximum(mean_err - std_err, 0.0)
            upper_std = mean_err + std_err
            y_label = 'Relative Error' if 'nrmse' in metric_name else 'Absolute Error'
            default_ylim = (0, np.max(max_err) * 1.1)

        # Plotting shapes and lines
        plt.fill_between(real_time_steps, min_err, max_err, color='gray', alpha=0.15, label='Absolute Min/Max')
        plt.fill_between(real_time_steps, lower_std, upper_std, color=color_theme, alpha=0.3, label='±1 Std Dev')
        plt.plot(real_time_steps, mean_err, color=color_theme, linewidth=2.5, label=f'Mean {metric_name.upper()}')

        # --- SET Y-AXIS LIMITS ---
        if metric_name in custom_ylims:
            plt.ylim(custom_ylims[metric_name])
        else:
            plt.ylim(default_ylim)

        # --- VERTICAL TRANSITION LINES ---
        for i, step in enumerate(transition_steps):
            # Only add the label to the legend once
            label = 'Stage Transition' if i == 0 else None
            plt.axvline(x=step, color='black', linestyle='--', alpha=0.8, label=label)

        # Formatting
        plt.title(f'Aggregated Dynamic Performance: {metric_name.replace("_", " ").title()}')
        plt.xlabel('Simulation Timestep')
        plt.ylabel(y_label)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='best')
        
        # Ensure the x-axis starts at 0 and ends at the max timestep
        plt.xlim(0, real_time_steps[-1])

        plot_path = os.path.join(output_dir, f"{metric_name}_time_series_replot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"Saved plot for {metric_name} -> {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot step-wise error curves from CSVs")
    parser.add_argument("--csv_dir", required=True, help="Folder with the plot_data CSVs")
    parser.add_argument("--out", required=True, help="Output folder for the PNGs")
    parser.add_argument("--ylim", nargs=3, action="append", metavar=("METRIC", "LO", "HI"), default=[],
                        help="Custom y-limits for one metric, e.g. --ylim nrmse_deform 0 0.30 (repeatable)")
    parser.add_argument("--transitions", nargs="*", type=int, default=[500, 1000],
                        help="Time steps of the press/dwell and dwell/release hand-offs")
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        raise SystemExit(f"Directory not found: {args.csv_dir}")
    limits = {m: (float(lo), float(hi)) for m, lo, hi in args.ylim} or None
    replot_from_csv(csv_dir=args.csv_dir, output_dir=args.out, custom_ylims=limits,
                    transition_steps=args.transitions)
    print("Done!")
