import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  # Added for percentage formatting
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths to point to the new JSON files (e.g., rollout_info.json)
file_map = {
    """"12": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_M/transolver/medium_1500_train_val/Transolver_M_ST2_Sl64/rollout/Transolver_M_ST2_Sl64_rollout_epoch_80_3_metric.json",
    "1180": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_M/transolver/medium_1500_train_val/Transolver_M_ST2_Sl64/rollout/Transolver_M_ST2_Sl64_rollout_epoch_740_3_metric.json",
    #"Reg_DGCNN": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/Inference_Output_OnestepPrediction/test/Inference_Output_OnestepPrediction_concatenated_rollout_all_3_metric.json","""

    "1020_5080":"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_C_ST1_MP20/rollout/MGN_C_ST1_MP20_rollout_epoch_1020_3_metric.json",
    "220_5080":"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_C_ST1_MP20/rollout/MGN_C_ST1_Messagepassing20_rollout_epoch_220_3_metric.json",
    "760_Spark2":"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_C_ST1_MP20_spark2/rollout/MGN_C_ST1_MP20_spark2_rollout_epoch_760_3_metric.json",
    "20_Spark2":"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/mgn/MGN_C_ST1_MP20_spark2/rollout/MGN_C_ST1_MP20_spark2_rollout_epoch_20_3_metric.json",
    
}

SAVE_FILENAME = "Meshgraphnet_Try.png"

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def load_and_process_data(paths):
    processed_records = []
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found for {label} at {path}")
            continue
            
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing JSON in {path}")
                continue
            
            # The new evaluation script stores trajectory data in a "losses" list
            losses_list = data.get("losses", [])
            
            for idx, entry in enumerate(losses_list):
                # Extract the Step-wise average nRMSE (as decimals)
                stress_val = entry.get("step_stress_nrmse_avg")
                y_disp_val = entry.get("step_y_deform_nrmse_avg")
                
                # Convert to Percentage (* 100) and append
                if stress_val is not None:
                    processed_records.append({
                        "Param": label,
                        "Trajectory": idx,
                        "Qty": "Stress",
                        "Error_Val": stress_val * 100.0  # Converted to %
                    })
                
                if y_disp_val is not None:
                    processed_records.append({
                        "Param": label,
                        "Trajectory": idx,
                        "Qty": "Y-Disp",
                        "Error_Val": y_disp_val * 100.0  # Converted to %
                    })
                    
    return pd.DataFrame(processed_records)

# ==========================================
# 3. DATA PREPARATION & OUTLIER LOGIC
# ==========================================
df = load_and_process_data(file_map)

if df.empty:
    print("❌ Error: No data was parsed. Check your JSON keys and file paths.")
else:
    # Use Categorical to maintain the order from file_map keys
    df['Param'] = pd.Categorical(df['Param'], categories=list(file_map.keys()), ordered=True)
    df = df.sort_values("Param")

    # Define the Y-limit for the plot dynamically to cut off extreme outliers
    y_limit = df["Error_Val"].quantile(0.96) * 1.6

    # Calculate Outliers for the terminal report
    def calculate_outliers(group):
        q1 = group["Error_Val"].quantile(0.25)
        q3 = group["Error_Val"].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        
        total_outliers = group[group["Error_Val"] > upper_whisker].shape[0]
        hidden_outliers = group[group["Error_Val"] > y_limit].shape[0]
        
        return pd.Series({
            "Total_Outliers": total_outliers,
            "Hidden_by_Limit": hidden_outliers,
            "Max_Error (%)": round(group["Error_Val"].max(), 4),
            "Mean_Error (%)": round(group["Error_Val"].mean(), 4)
        })

    outlier_summary = df.groupby(['Param', 'Qty'], observed=True).apply(calculate_outliers).reset_index()

    # --- Print Terminal Report ---
    print("\n" + "="*85)
    print("STATISTICAL SUMMARY (Metric: Step-wise nRMSE [%])")
    print(f"Current Y-Axis Display Limit: {y_limit:.2f}%")
    print("="*85)
    print(outlier_summary.to_string(index=False))
    print("-" * 85)
    print(f"TOTAL HIDDEN DATA POINTS ACROSS ALL GROUPS: {int(outlier_summary['Hidden_by_Limit'].sum())}")
    print("="*85 + "\n")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")

    # CHANGED: Made the figure narrower and taller (Width: 5.5, Height: 6.5)
    fig, ax = plt.subplots(figsize=(5.5, 6.5))

    bp = sns.boxplot(
        data=df, 
        x="Param", 
        y="Error_Val", 
        hue="Qty", 
        palette="Set1",
        width=0.5, 
        linewidth=1.2,
        fliersize=2.5,   
        showmeans=True,
        meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 5}
    )

    # Apply the adaptive Y-axis limit
    plt.ylim(0, y_limit)

    # CHANGED: Force Y-axis to show percentages (e.g., "2%" instead of "2")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # --- Styling ---
    sns.despine() 
    
    # CHANGED: Corrected axis labels
    plt.xlabel("Meshgraphnet", fontweight='bold')
    plt.ylabel("Step-wise nRMSE", fontweight='bold')
    
    # Slight rotation to prevent label overlap
    plt.xticks(rotation=0, fontsize=10) 
    
    # Position legend outside to the right
    plt.legend(
        title=None, 
        frameon=True, 
        fontsize=10, 
        loc='upper left', 
        bbox_to_anchor=(1, 1) 
    )

    # Final Adjustment and Save
    plt.tight_layout()
    plt.savefig(SAVE_FILENAME, dpi=600, bbox_inches='tight')
    print(f"✅ Success: Plot saved as {SAVE_FILENAME}")
    plt.show()