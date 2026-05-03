import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
file_map = {
    "Transolver": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/transolver/E1_C/Transolver_Slice64/Combined_Inference_OneStepPrediction/test/rollout_evaluation_max_error/combined_detailed_summary.json",
    "Dilated": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/dilated/Combined_Inference_K20_D100_onestepprediction/test/rollout_evaluation_max_error/combined_detailed_summary.json",
    "Reg_DGCNN": "/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/reg/Coarse_ST1/Inference_Output_OnestepPrediction/test/rollout_evaluation_max_error/combined_detailed_summary.json",

    
}

METRIC_CHOICE = "percent_error" 
SAVE_FILENAME = "Model_Comparison_OneStepPrediction_max_error.png"

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def load_and_process_data(paths, metric_name):
    processed_records = []
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found for {label} at {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            for entry in data:
                if entry.get("metric_type") == "max_error":
                    qty = entry.get("quantity")
                    if metric_name == "relative_error_to_max_groundtruth":
                        suffix = "(s)" if "stress" in qty.lower() else "(y)"
                        val = entry.get(f"{metric_name}{suffix}")
                    else:
                        val = entry.get(metric_name)
                    
                    if val is not None:
                        processed_records.append({
                            "Param": label,
                            "Trajectory": entry.get("time_index") if entry.get("time_index") else entry.get("trajectory_index"),
                            "Qty": "Stress" if "stress" in qty.lower() else "Y-Disp",
                            "Error_Val": val
                        })
    return pd.DataFrame(processed_records)

# ==========================================
# 3. DATA PREPARATION & OUTLIER LOGIC
# ==========================================
df = load_and_process_data(file_map, METRIC_CHOICE)

if df.empty:
    print("❌ Error: No data was parsed. Check your JSON keys and file paths.")
else:
    # FIX: Use Categorical to maintain the order from file_map keys
    df['Param'] = pd.Categorical(df['Param'], categories=list(file_map.keys()), ordered=True)
    df = df.sort_values("Param")

    # Define the Y-limit for the plot
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
            "Max_Error": round(group["Error_Val"].max(), 4)
        })

    outlier_summary = df.groupby(['Param', 'Qty'], observed=True).apply(calculate_outliers).reset_index()

    # --- Print Terminal Report ---
    print("\n" + "="*85)
    print(f"STATISTICAL SUMMARY (Metric: {METRIC_CHOICE.upper()})")
    print(f"Current Y-Axis Display Limit: {y_limit:.2f}")
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

    # ADJUSTMENT: Increased width from 5.5 to 8.5
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

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

    # --- Styling ---
    sns.despine() 
    plt.ylabel(f"{METRIC_CHOICE.replace('_', ' ').title()} [%]", fontweight='bold')
    plt.xlabel("Models", fontweight='bold')
    
    # Slight rotation to prevent label overlap
    plt.xticks(rotation=15, fontsize=9) 
    
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
    print(f"✅ Success: Verified plot saved as {SAVE_FILENAME}")
    plt.show()