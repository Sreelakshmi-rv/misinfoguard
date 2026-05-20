"""
generate_final_results.py — Publication-quality results chart and summary table.

Run this after evaluation to generate:
    models/final_results_chart.png  — bar chart with error bars + significance
    models/final_results_table.csv  — clean summary table for report

Usage:
    python src/generate_final_results.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from simulation   import evaluate_scenarios, SCENARIO_LABELS, SCENARIO_COLORS
from stable_baselines3 import DQN
from rl_env       import MisinfoEnv


# ─────────────────────────────────────────────
# FINAL NUMBERS — from 200-seed evaluation
# Update these if you rerun evaluate_scenarios
# ─────────────────────────────────────────────

FINAL_RESULTS = {
    "none"  : {"mean": 63.9, "std": 5.7,  "peak": 40.3},
    "random": {"mean": 51.6, "std": 7.8,  "peak": 29.9},
    "greedy": {"mean": 32.6, "std": 9.4,  "peak": 17.8},
    "rl"    : {"mean": 30.1, "std": 8.0,  "peak": 17.3},
}

SCENARIOS = ["none", "random", "greedy", "rl"]

# Statistical test result (RL vs Greedy, n=200)
P_VALUE   = 0.0035
COHENS_D  = 0.294
N_SEEDS   = 200


# ─────────────────────────────────────────────
# CHART 1 — BAR CHART WITH ERROR BARS
# ─────────────────────────────────────────────

def plot_final_bar_chart():
    labels  = [SCENARIO_LABELS[s] for s in SCENARIOS]
    means   = [FINAL_RESULTS[s]["mean"] for s in SCENARIOS]
    stds    = [FINAL_RESULTS[s]["std"]  for s in SCENARIOS]
    colors  = [SCENARIO_COLORS[s]       for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.bar(
        labels, means, yerr=stds,
        color=colors, alpha=0.88,
        edgecolor="white", linewidth=1.0,
        capsize=7,
        error_kw={"linewidth": 2, "color": "white", "alpha": 0.7},
        zorder=3
    )

    # Annotate bars with mean ± std
    for bar, mean, std, scenario in zip(bars, means, stds, SCENARIOS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 1.2,
            f"{mean:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="white"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 3.8,
            f"±{std:.1f}%",
            ha="center", va="bottom",
            fontsize=8, color="#aaaaaa"
        )

    # Significance bracket between greedy and rl
    greedy_idx = SCENARIOS.index("greedy")
    rl_idx     = SCENARIOS.index("rl")
    greedy_bar = bars[greedy_idx]
    rl_bar     = bars[rl_idx]

    bracket_y  = max(means) + max(stds) + 10
    x1 = greedy_bar.get_x() + greedy_bar.get_width() / 2
    x2 = rl_bar.get_x()     + rl_bar.get_width()     / 2

    ax.plot([x1, x1, x2, x2],
            [bracket_y - 1, bracket_y, bracket_y, bracket_y - 1],
            color="gold", linewidth=1.5)
    ax.text(
        (x1 + x2) / 2, bracket_y + 0.5,
        f"p={P_VALUE} **(d={COHENS_D})",
        ha="center", va="bottom",
        fontsize=9, color="gold", fontweight="bold"
    )

    # Grid and styling
    ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, bracket_y + 6)
    ax.set_ylabel("Total Network Reached (%)", fontsize=12,
                  color="white", labelpad=10)
    ax.set_title(
        f"Intervention Strategy Comparison — {N_SEEDS} Network Instances\n"
        f"500 nodes | Partial Observability | Message Passing | "
        f"Echo Chamber Edge Weights",
        fontsize=13, fontweight="bold", color="white", pad=15
    )

    ax.tick_params(colors="white", labelsize=10)
    ax.spines[:].set_visible(False)
    for label in ax.get_xticklabels():
        label.set_color("white")

    # Legend for error bars
    ax.text(
        0.98, 0.03,
        f"Error bars = ±1 std | n={N_SEEDS} seeds",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#888888"
    )

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/final_results_chart.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("Saved → models/final_results_chart.png")


# ─────────────────────────────────────────────
# CHART 2 — PEAK VS TOTAL REACHED SCATTER
# ─────────────────────────────────────────────

def plot_peak_vs_total():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for scenario in SCENARIOS:
        r     = FINAL_RESULTS[scenario]
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        ax.scatter(r["peak"], r["mean"],
                   color=color, s=180, zorder=5,
                   edgecolors="white", linewidth=1.2)
        ax.errorbar(r["peak"], r["mean"],
                    yerr=r["std"],
                    fmt="none", color=color,
                    capsize=5, linewidth=1.5, alpha=0.6)
        ax.annotate(
            label,
            xy=(r["peak"], r["mean"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="white"
        )

    ax.set_xlabel("Peak Infected (%)", fontsize=11, color="white")
    ax.set_ylabel("Total Network Reached (%)", fontsize=11, color="white")
    ax.set_title("Peak vs Total Spread — All Strategies",
                 fontsize=12, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
    ax.xaxis.grid(True, alpha=0.2, color="white", linestyle="--")

    plt.tight_layout()
    plt.savefig("models/peak_vs_total.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("Saved → models/peak_vs_total.png")


# ─────────────────────────────────────────────
# TABLE — CSV + CONSOLE
# ─────────────────────────────────────────────

def generate_summary_table():
    rows = []
    baseline_mean = FINAL_RESULTS["none"]["mean"]

    for scenario in SCENARIOS:
        r          = FINAL_RESULTS[scenario]
        reduction  = round(baseline_mean - r["mean"], 1)
        rel_red    = round(reduction / baseline_mean * 100, 1)
        sig        = ""
        if scenario == "rl":
            sig    = f"p={P_VALUE}, d={COHENS_D}"

        rows.append({
            "Strategy"              : SCENARIO_LABELS[scenario],
            "Total Reached (mean)"  : f"{r['mean']}%",
            "Total Reached (±std)"  : f"±{r['std']}%",
            "Peak Infected"         : f"{r['peak']}%",
            "Reduction vs Baseline" : f"-{reduction}%",
            "Relative Reduction"    : f"-{rel_red}%",
            "Significance"          : sig,
        })

    df = pd.DataFrame(rows)

    # Console print
    print("\n" + "=" * 85)
    print("  FINAL RESULTS SUMMARY TABLE")
    print(f"  n={N_SEEDS} seeds | 500 nodes | Partial Observability")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85)
    print(f"\n  RL vs Greedy: {FINAL_RESULTS['greedy']['mean']}% → "
          f"{FINAL_RESULTS['rl']['mean']}% "
          f"(gap={round(FINAL_RESULTS['greedy']['mean']-FINAL_RESULTS['rl']['mean'],1)}%, "
          f"p={P_VALUE}, Cohen's d={COHENS_D})")
    print(f"  NLP Accuracy: 74.4% ± 0.7% (5-fold CV, LIAR dataset)")
    print("=" * 85)

    # Save CSV
    os.makedirs("models", exist_ok=True)
    df.to_csv("models/final_results_table.csv", index=False)
    print("\nSaved → models/final_results_table.csv")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating final results charts and table...")
    print("Using 200-seed evaluation numbers.\n")

    plot_final_bar_chart()
    plot_peak_vs_total()
    generate_summary_table()

    print("\nAll outputs saved to models/:")
    print("  final_results_chart.png")
    print("  peak_vs_total.png")
    print("  final_results_table.csv")