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
    # Define where your CSVs are.
    base_dir = "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Inference_Output_Retrain/val_real/journal_evaluation_results"
    
    csv_directory = os.path.join(base_dir, "plot_data_csvs")
    output_directory = os.path.join(base_dir, "custom_plots")
    
    # ---------------------------------------------------------
    # DEFINE YOUR CUSTOM Y-AXIS LIMITS HERE
    # ---------------------------------------------------------
    my_custom_limits = {
        'nrmse_deform': (0, 0.30),    
        'nrmse_stress': (0, 0.365),    
    }
    
    # ---------------------------------------------------------
    # DEFINE YOUR TRANSITION TIMESTEPS HERE
    # ---------------------------------------------------------
    my_transition_steps = [500, 1000]

    print(f"Reading CSVs from: {csv_directory}")
    if os.path.exists(csv_directory):
        replot_from_csv(
            csv_dir=csv_directory, 
            output_dir=output_directory, 
            custom_ylims=my_custom_limits,
            transition_steps=my_transition_steps
        )
        print("Done!")
    else:
        print("Directory not found. Please check the 'base_dir' path.")