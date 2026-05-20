"""
multi_network_eval.py — Cross-network evaluation for report.

Loads trained agents per network type and evaluates RL vs Greedy
across all 4 network types. Generates charts for the report.

Run AFTER train_agents.py has trained all 4 network type agents.

Usage:
    python src/multi_network_eval.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from graph_env  import NETWORK_CONFIGS, plot_all_degree_distributions
from rl_env     import MisinfoEnv
from simulation import evaluate_scenarios, SCENARIO_LABELS, SCENARIO_COLORS
from stable_baselines3 import DQN


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

def load_models(graph_type, models_dir="models"):
    env0  = MisinfoEnv(agent_id=0, graph_type=graph_type)
    env1  = MisinfoEnv(agent_id=1, graph_type=graph_type)
    path0 = os.path.join(models_dir, f"agent0_{graph_type}_dqn")
    path1 = os.path.join(models_dir, f"agent1_{graph_type}_dqn")

    if not os.path.exists(path0 + ".zip"):
        print(f"  Warning: {path0}.zip not found, using base BA agents")
        path0 = os.path.join(models_dir, "agent0_dqn")
        path1 = os.path.join(models_dir, "agent1_dqn")

    return [DQN.load(path0, env=env0), DQN.load(path1, env=env1)]


# ─────────────────────────────────────────────
# RUN EVALUATION ACROSS ALL NETWORKS
# ─────────────────────────────────────────────

def run_all_networks(n_seeds=100):
    all_summaries = {}
    scenarios     = ["none", "random", "greedy", "rl"]

    for graph_type, config in NETWORK_CONFIGS.items():
        print(f"\n{'='*55}")
        print(f"  {config['label']}")
        print(f"{'='*55}")

        models  = load_models(graph_type)
        summary = evaluate_scenarios(
            scenarios      = scenarios,
            models         = models,
            num_seeds_eval = n_seeds,
            graph_type     = graph_type,
            verbose        = True,
        )
        summary["_n_seeds"] = n_seeds
        all_summaries[graph_type] = summary

    return all_summaries


# ─────────────────────────────────────────────
# PLOT 1 — GROUPED BAR CHART
# ─────────────────────────────────────────────

def plot_grouped_bar(all_summaries):
    network_types = list(all_summaries.keys())
    n     = len(network_types)
    x     = np.arange(n)
    width = 0.35

    greedy_means = [all_summaries[g]["greedy"]["reached_mean"]
                    for g in network_types]
    greedy_stds  = [all_summaries[g]["greedy"]["reached_std"]
                    for g in network_types]
    rl_means     = [all_summaries[g]["rl"]["reached_mean"]
                    for g in network_types]
    rl_stds      = [all_summaries[g]["rl"]["reached_std"]
                    for g in network_types]
    labels       = [NETWORK_CONFIGS[g]["label"].replace("\n", " ")
                    for g in network_types]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.bar(x - width/2, greedy_means, width, yerr=greedy_stds,
           label="Greedy (Degree)", color="#3498db", alpha=0.88,
           edgecolor="white", linewidth=0.8, capsize=5,
           error_kw={"linewidth": 1.5, "color": "white"})

    ax.bar(x + width/2, rl_means, width, yerr=rl_stds,
           label="RL Agents (Ours)", color="#2ecc71", alpha=0.88,
           edgecolor="white", linewidth=0.8, capsize=5,
           error_kw={"linewidth": 1.5, "color": "white"})

    for i in range(n):
        gap   = greedy_means[i] - rl_means[i]
        y_pos = max(greedy_means[i] + greedy_stds[i],
                    rl_means[i]     + rl_stds[i]) + 2
        color = "#2ecc71" if gap > 0 else "#e74c3c"
        ax.text(x[i], y_pos, f"{gap:+.1f}%",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

    ax.set_ylabel("Total Network Reached (%)", fontsize=12, color="white")
    ax.set_title(
        "RL Agents vs Greedy — Across 4 Network Topologies\n"
        "Agents trained independently per network | "
        f"{all_summaries[network_types[0]]['_n_seeds']} seeds per network",
        fontsize=13, fontweight="bold", color="white", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="white", rotation=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(greedy_means) + max(greedy_stds) + 14)
    ax.legend(fontsize=10, facecolor="#1a1a2e",
              labelcolor="white", edgecolor="white")

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/multi_network_results.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("Saved → models/multi_network_results.png")


# ─────────────────────────────────────────────
# PLOT 2 — ALL SCENARIOS ACROSS NETWORKS
# ─────────────────────────────────────────────

def plot_all_scenarios(all_summaries):
    """
    4-panel chart: one panel per network type.
    Each panel shows all 4 scenario bars.
    """
    network_types = list(all_summaries.keys())
    scenarios     = ["none", "random", "greedy", "rl"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0e1117")
    axes = axes.flatten()

    for ax, graph_type in zip(axes, network_types):
        ax.set_facecolor("#0e1117")
        config  = NETWORK_CONFIGS[graph_type]
        summary = all_summaries[graph_type]

        means  = [summary[s]["reached_mean"] for s in scenarios]
        stds   = [summary[s]["reached_std"]  for s in scenarios]
        colors = [SCENARIO_COLORS[s]          for s in scenarios]
        labels = [SCENARIO_LABELS[s]          for s in scenarios]

        bars = ax.bar(labels, means, yerr=stds,
                      color=colors, alpha=0.88,
                      edgecolor="white", linewidth=0.8,
                      capsize=5,
                      error_kw={"linewidth": 1.5, "color": "white"})

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.8,
                    f"{mean:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="white")

        ax.set_title(config["label"], fontsize=11,
                     fontweight="bold", color="white")
        ax.set_ylabel("Total Reached (%)", fontsize=9, color="white")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
        ax.set_ylim(0, max(means) + max(stds) + 10)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.suptitle(
        "All Intervention Strategies Across 4 Network Types",
        fontsize=14, fontweight="bold", color="white"
    )
    plt.tight_layout()
    plt.savefig("models/all_scenarios_all_networks.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("Saved → models/all_scenarios_all_networks.png")


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary_table(all_summaries):
    print("\n" + "=" * 78)
    print("  MULTI-NETWORK EVALUATION — RL vs GREEDY")
    print("=" * 78)
    print(f"  {'Network':<30} {'Greedy':>15} {'RL':>15} "
          f"{'Gap':>8} {'p-value':>10}")
    print("  " + "-" * 75)

    rows = []
    for graph_type, summary in all_summaries.items():
        label    = NETWORK_CONFIGS[graph_type]["label"].replace("\n", " ")
        gm       = summary["greedy"]["reached_mean"]
        gs       = summary["greedy"]["reached_std"]
        rm       = summary["rl"]["reached_mean"]
        rs       = summary["rl"]["reached_std"]
        gap      = gm - rm
        n        = summary.get("_n_seeds", 100)

        # Welch's t-test using raw arrays if available
        if "raw_reached" in summary.get("greedy", {}):
            _, p = stats.ttest_ind(
                summary["greedy"]["raw_reached"],
                summary["rl"]["raw_reached"],
                equal_var=False
            )
        else:
            se = np.sqrt(gs**2/n + rs**2/n)
            t  = gap / se if se > 0 else 0
            p  = 2 * stats.t.sf(abs(t), n-1)

        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<30} {gm:>10.1f}±{gs:.1f}%  "
              f"{rm:>10.1f}±{rs:.1f}%  "
              f"{gap:>+7.1f}%  {p:>8.4f} {sig}")

        rows.append({
            "Network"     : label,
            "Greedy Mean" : f"{gm:.1f}%",
            "Greedy Std"  : f"±{gs:.1f}%",
            "RL Mean"     : f"{rm:.1f}%",
            "RL Std"      : f"±{rs:.1f}%",
            "Gap"         : f"{gap:+.1f}%",
            "p-value"     : round(p, 4),
            "Significant" : sig,
        })

    print("=" * 78)
    print("  ** p<0.01   * p<0.05")

    df = pd.DataFrame(rows)
    os.makedirs("models", exist_ok=True)
    df.to_csv("models/multi_network_table.csv", index=False)
    print("\nSaved → models/multi_network_table.csv")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Multi-network evaluation — 4 network types, 100 seeds each")
    print("Make sure train_agents.py has been run for all network types first.")
    print("=" * 55)

    all_summaries = run_all_networks(n_seeds=100)

    print_summary_table(all_summaries)
    plot_grouped_bar(all_summaries)
    plot_all_scenarios(all_summaries)

    print("\nGenerating degree distribution comparison...")
    plot_all_degree_distributions()

    print("\nAll done. Files saved to models/:")
    print("  multi_network_results.png")
    print("  all_scenarios_all_networks.png")
    print("  all_degree_distributions.png")
    print("  multi_network_table.csv")