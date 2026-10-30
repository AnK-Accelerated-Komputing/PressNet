"""Box plots of step-wise nRMSE across trajectories (ablation / one-step figures of the paper).

Each ``--metrics LABEL=PATH`` is a ``stepwise_metrics.json`` written by
``utilities/evaluate_stepwise.py``. One box per label; stress and y-displacement
are drawn side by side unless ``--quantity`` selects one of them.

    python -m utilities.plot_boxplots \\
        --metrics MGN-10=runs/mgn10/stepwise/stepwise_metrics.json \\
        --metrics MGN-20=runs/mgn20/stepwise/stepwise_metrics.json \\
        --quantity both --out mgn_message_passing.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

QUANTITIES = {
    "stress": ("Stress", "step_stress_nrmse_avg"),
    "y": ("Y-Disp", "step_y_deform_nrmse_avg"),
}


def load_records(metric_files, quantities):
    records = []
    for label, path in metric_files.items():
        if not os.path.exists(path):
            print(f"Warning: file not found for {label}: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for idx, entry in enumerate(data.get("losses", [])):
            for q in quantities:
                name, key = QUANTITIES[q]
                value = entry.get(key)
                if value is not None:
                    records.append({"Param": label, "Trajectory": idx, "Qty": name,
                                    "Error_Val": value * 100.0})  # stored as a fraction -> percent
    return pd.DataFrame(records)


def outlier_report(df, y_limit):
    def per_group(group):
        q1, q3 = group["Error_Val"].quantile(0.25), group["Error_Val"].quantile(0.75)
        upper_whisker = q3 + 1.5 * (q3 - q1)
        return pd.Series({
            "Total_Outliers": int((group["Error_Val"] > upper_whisker).sum()),
            "Hidden_by_Limit": int((group["Error_Val"] > y_limit).sum()),
            "Max_Error (%)": round(group["Error_Val"].max(), 4),
            "Mean_Error (%)": round(group["Error_Val"].mean(), 4),
        })
    return df.groupby(["Param", "Qty"], observed=True).apply(per_group).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Box plots of step-wise nRMSE",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metrics", action="append", required=True, metavar="LABEL=PATH",
                        help="Label and stepwise_metrics.json (repeatable; order = order on the x-axis)")
    parser.add_argument("--quantity", choices=["both", "stress", "y"], default="both")
    parser.add_argument("--xlabel", default="Model")
    parser.add_argument("--out", default="boxplot.png")
    parser.add_argument("--no_show", action="store_true", help="Do not open a window")
    args = parser.parse_args()

    metric_files = {}
    for item in args.metrics:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"--metrics expects LABEL=PATH, got {item!r}")
        metric_files[label] = path

    quantities = ["stress", "y"] if args.quantity == "both" else [args.quantity]
    df = load_records(metric_files, quantities)
    if df.empty:
        raise SystemExit("No data parsed: check the paths and that the JSON files contain a 'losses' list.")

    df["Param"] = pd.Categorical(df["Param"], categories=list(metric_files.keys()), ordered=True)
    df = df.sort_values("Param")

    # Cut the axis above extreme outliers; the report below says how many points are hidden.
    y_limit = df["Error_Val"].quantile(0.96) * 1.6
    report = outlier_report(df, y_limit)
    print("\n" + "=" * 85)
    print("STATISTICAL SUMMARY (Metric: step-wise nRMSE [%])")
    print(f"Y-axis display limit: {y_limit:.2f}%")
    print("=" * 85)
    print(report.to_string(index=False))
    print(f"TOTAL HIDDEN DATA POINTS: {int(report['Hidden_by_Limit'].sum())}\n")

    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    sns.boxplot(
        data=df, x="Param", y="Error_Val",
        hue="Qty" if len(quantities) > 1 else None,
        palette="Set1", width=0.5, linewidth=1.2, fliersize=2.5, showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
    )
    plt.ylim(0, y_limit)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0 if len(quantities) > 1 else 2))
    sns.despine()
    plt.xlabel(args.xlabel, fontweight="bold")
    plt.ylabel("Step-wise nRMSE" if len(quantities) > 1 else f"Step-wise {QUANTITIES[quantities[0]][0]} nRMSE",
               fontweight="bold")
    if len(quantities) > 1:
        plt.legend(title=None, frameon=True, fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"Saved plot to {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
