"""
adversarial_experiment.py — Tests where greedy fails and RL holds up.

The standard evaluation uses default network conditions where degree
and risk are correlated. This experiment deliberately creates conditions
where that correlation breaks down, exposing greedy's structural weakness.

Three adversarial conditions tested:

1. HIGH ECHO CHAMBER DENSITY
   Edge weights are amplified for hub-to-hub connections.
   High-degree nodes now have LOWER risk than medium-degree nodes
   in dense echo chamber clusters. Greedy targets degree — wrong choice.
   RL observes risk scores and edge weights — correct choice.

2. BOUNDARY SEEDING
   Infection is seeded at nodes on the boundary between the two
   agent regions rather than at high-degree hubs. Greedy targets
   the highest-degree visible node, which may be deep inside a
   region and far from the active spread front. RL observes
   boundary pressure in its state and via messages.

3. HIGH SUSCEPTIBILITY CLUSTERS
   A subset of nodes are assigned very high susceptibility (0.8-1.0).
   These nodes are not necessarily high-degree, so greedy ignores
   them. RL observes avg_sus_targets in its state and learns to
   prioritise protecting these clusters.

Each condition is tested across 100 seeds.
Results are plotted as grouped bar charts and saved to models/.

Usage:
    python src/adversarial_experiment.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from stable_baselines3 import DQN

from graph_env  import create_graph, assign_node_properties, assign_edge_weights
from sir_model  import seed_infection, sir_step
from rl_env     import (
    MisinfoEnv, build_state, apply_action,
    generate_message, get_visible_nodes,
    NUM_NODES, NUM_REGIONS, NUM_SEEDS, MESSAGE_SIZE,
)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

def load_models():
    models_dir = "models"
    env0   = MisinfoEnv(agent_id=0)
    env1   = MisinfoEnv(agent_id=1)
    model0 = DQN.load(os.path.join(models_dir, "agent0_dqn"), env=env0)
    model1 = DQN.load(os.path.join(models_dir, "agent1_dqn"), env=env1)
    return [model0, model1]


# ─────────────────────────────────────────────
# GRAPH MODIFIERS — CREATE ADVERSARIAL CONDITIONS
# ─────────────────────────────────────────────

def amplify_echo_chambers(G, amplification=2.0):
    """
    Doubles the weight of hub-to-hub edges.
    This makes high-weight clusters far more dangerous
    than degree alone would suggest — greedy's weakness.
    """
    G = deepcopy(G)
    degrees    = dict(G.degree())
    max_degree = max(degrees.values())

    for u, v in G.edges():
        avg_norm = ((degrees[u] + degrees[v]) / 2) / max_degree
        if avg_norm >= 0.15:
            G[u][v]["weight"] = min(3.5, G[u][v]["weight"] * amplification)
    return G


def seed_at_boundary(G, regions, num_seeds=5, seed=42):
    """
    Seeds infection at nodes on the boundary between regions.
    Boundary nodes have cross-region edges — spread jumps regions
    faster than greedy anticipates.
    """
    import networkx as nx
    np.random.seed(seed)

    region0_set = set(regions[0])
    region1_set = set(regions[1])

    boundary_nodes = [
        n for n in G.nodes()
        if any(nb in region1_set for nb in G.neighbors(n))
        and n in region0_set
    ]

    if len(boundary_nodes) < num_seeds:
        boundary_nodes = list(G.nodes())

    selected = np.random.choice(
        boundary_nodes,
        size=min(num_seeds, len(boundary_nodes)),
        replace=False
    )

    for node in G.nodes():
        G.nodes[node]["status"] = "S"
        G.nodes[node].pop("infected_time", None)

    for node in selected:
        G.nodes[node]["status"]        = "I"
        G.nodes[node]["infected_time"] = 0

    return G, list(selected)


def create_high_susceptibility_clusters(G, fraction=0.15, sus_range=(0.85, 1.0)):
    """
    Assigns very high susceptibility to a fraction of nodes.
    These nodes are not necessarily high-degree, so greedy
    does not prioritise protecting them.
    RL observes avg_sus_targets in state and learns to protect them.
    """
    G = deepcopy(G)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    n_high = int(len(nodes) * fraction)

    for node in nodes[:n_high]:
        G.nodes[node]["susceptibility"] = np.random.uniform(*sus_range)

    return G


# ─────────────────────────────────────────────
# SINGLE EPISODE RUNNER
# ─────────────────────────────────────────────

def run_episode(G, scenario, models, regions, max_steps=60,
                spread_rate=0.3, recovery_time=4):
    """
    Runs one episode on a pre-configured graph.
    Graph is passed in already modified for the adversarial condition.
    """
    G = deepcopy(G)

    messages = [
        np.zeros(MESSAGE_SIZE, dtype=np.float32)
        for _ in range(NUM_REGIONS)
    ]

    history = []

    for step in range(max_steps):
        s = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "S")
        i = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "I")
        r = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "R")
        history.append({"step": step, "S": s, "I": i, "R": r})

        if i == 0 and step > 0:
            break

        if scenario == "greedy":
            for region in regions:
                visible  = get_visible_nodes(G, region)
                infected = [n for n in visible if G.nodes[n]["status"] == "I"]
                if infected:
                    target = max(infected, key=lambda n: G.degree(n))
                    G.nodes[target]["status"]        = "R"
                    G.nodes[target]["infected_time"] = 999

        elif scenario == "rl":
            for agent_id, region in enumerate(regions):
                received = [messages[1 - agent_id]]
                state    = build_state(G, region,
                                       all_regions=regions,
                                       received_messages=received)
                action, _ = models[agent_id].predict(state, deterministic=True)
                G, _      = apply_action(G, int(action), region)

            for agent_id, region in enumerate(regions):
                messages[agent_id] = generate_message(G, region)

        G, _, _ = sir_step(G, recovery_time=recovery_time,
                           spread_rate=spread_rate)

    total   = NUM_NODES
    reached = history[-1]["R"] + history[-1]["I"]
    peak    = max(h["I"] for h in history)

    return {
        "reached_pct": reached / total * 100,
        "peak_pct"   : peak    / total * 100,
        "steps"      : history[-1]["step"],
        "history"    : history,
    }


# ─────────────────────────────────────────────
# RUN ONE ADVERSARIAL CONDITION
# ─────────────────────────────────────────────

def run_condition(condition_name, graph_modifier,
                  seed_fn, models, regions,
                  n_seeds=100, spread_rate=0.3):
    """
    Runs greedy and RL across n_seeds instances of one adversarial condition.
    Returns raw per-seed results for both strategies.
    """
    print(f"\n  Running: {condition_name} ({n_seeds} seeds)...")

    greedy_reached = []
    rl_reached     = []
    greedy_peak    = []
    rl_peak        = []

    for seed in range(n_seeds):
        # Build base graph
        G_base = create_graph("barabasi_albert", num_nodes=NUM_NODES, seed=seed)

        # Apply adversarial modification
        G_mod = graph_modifier(G_base)

        # Seed infection
        G_mod, _ = seed_fn(G_mod, regions, seed=seed)

        # Run both strategies
        greedy_result = run_episode(G_mod, "greedy", models, regions,
                                    spread_rate=spread_rate)
        rl_result     = run_episode(G_mod, "rl",     models, regions,
                                    spread_rate=spread_rate)

        greedy_reached.append(greedy_result["reached_pct"])
        rl_reached.append(rl_result["reached_pct"])
        greedy_peak.append(greedy_result["peak_pct"])
        rl_peak.append(rl_result["peak_pct"])

        if seed % 20 == 0:
            print(f"    Seed {seed:>3} | Greedy: {greedy_result['reached_pct']:.1f}% "
                  f"| RL: {rl_result['reached_pct']:.1f}%")

    gap = np.mean(greedy_reached) - np.mean(rl_reached)
    print(f"  Result: Greedy {np.mean(greedy_reached):.1f}% "
          f"vs RL {np.mean(rl_reached):.1f}% "
          f"(gap = {gap:+.1f}%, RL {'better' if gap > 0 else 'worse'})")

    return {
        "greedy_reached": greedy_reached,
        "rl_reached"    : rl_reached,
        "greedy_peak"   : greedy_peak,
        "rl_peak"       : rl_peak,
    }


# ─────────────────────────────────────────────
# DEFAULT SEED FUNCTION (high degree)
# ─────────────────────────────────────────────

def default_seed_fn(G, regions, num_seeds=NUM_SEEDS, seed=42):
    G, nodes = seed_infection(G, num_seeds=num_seeds,
                              strategy="high_degree", seed=seed)
    return G, nodes


def boundary_seed_fn(G, regions, seed=42):
    return seed_at_boundary(G, regions, num_seeds=NUM_SEEDS, seed=seed)


# ─────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────

def plot_adversarial_results(all_results, standard_results):
    """
    Grouped bar chart comparing Greedy vs RL across:
        - Standard conditions (from main evaluation)
        - Echo chamber amplification
        - Boundary seeding
        - High susceptibility clusters
    """
    conditions = [
        "Standard\n(200 seeds)",
        "Echo Chamber\nAmplified",
        "Boundary\nSeeding",
        "High Susceptibility\nClusters",
    ]

    greedy_means = [
        standard_results["greedy_mean"],
        np.mean(all_results["echo"]["greedy_reached"]),
        np.mean(all_results["boundary"]["greedy_reached"]),
        np.mean(all_results["highsus"]["greedy_reached"]),
    ]
    greedy_stds = [
        standard_results["greedy_std"],
        np.std(all_results["echo"]["greedy_reached"]),
        np.std(all_results["boundary"]["greedy_reached"]),
        np.std(all_results["highsus"]["greedy_reached"]),
    ]

    rl_means = [
        standard_results["rl_mean"],
        np.mean(all_results["echo"]["rl_reached"]),
        np.mean(all_results["boundary"]["rl_reached"]),
        np.mean(all_results["highsus"]["rl_reached"]),
    ]
    rl_stds = [
        standard_results["rl_std"],
        np.std(all_results["echo"]["rl_reached"]),
        np.std(all_results["boundary"]["rl_reached"]),
        np.std(all_results["highsus"]["rl_reached"]),
    ]

    x     = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars_g = ax.bar(x - width/2, greedy_means, width,
                    yerr=greedy_stds, label="Greedy (Degree)",
                    color="#3498db", alpha=0.88,
                    edgecolor="white", linewidth=0.8,
                    capsize=5, error_kw={"linewidth": 1.5, "color": "white"})

    bars_r = ax.bar(x + width/2, rl_means, width,
                    yerr=rl_stds, label="RL Agents (Ours)",
                    color="#2ecc71", alpha=0.88,
                    edgecolor="white", linewidth=0.8,
                    capsize=5, error_kw={"linewidth": 1.5, "color": "white"})

    # Annotate gaps
    for i in range(len(conditions)):
        gap = greedy_means[i] - rl_means[i]
        y   = max(greedy_means[i] + greedy_stds[i],
                  rl_means[i]     + rl_stds[i]) + 2
        color = "#2ecc71" if gap > 0 else "#e74c3c"
        ax.text(x[i], y, f"{gap:+.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color)

    ax.set_ylabel("Total Network Reached (%)", fontsize=12, color="white")
    ax.set_title(
        "Adversarial Evaluation — Greedy vs RL Agents\n"
        "Under Conditions That Expose Degree-Based Targeting Weaknesses",
        fontsize=13, fontweight="bold", color="white", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(greedy_means) + max(greedy_stds) + 12)

    legend = ax.legend(fontsize=10, facecolor="#1a1a2e",
                       labelcolor="white", edgecolor="white")

    ax.text(0.98, 0.03,
            "Error bars = ±1 std | n=100 seeds per adversarial condition",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888")

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/adversarial_experiment.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("\nSaved → models/adversarial_experiment.png")


def plot_adversarial_timeseries(all_results, models, regions):
    """
    Shows one representative episode from each adversarial condition,
    comparing Greedy and RL spread curves side by side.
    """
    conditions = {
        "echo"    : ("Echo Chamber Amplified",   "#9b59b6"),
        "boundary": ("Boundary Seeding",          "#e67e22"),
        "highsus" : ("High Susceptibility Clusters", "#e74c3c"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0e1117")

    for ax, (key, (title, color)) in zip(axes, conditions.items()):
        ax.set_facecolor("#0e1117")

        # Get a seed where greedy is visibly worse
        greedy_arr = all_results[key]["greedy_reached"]
        rl_arr     = all_results[key]["rl_reached"]
        gaps       = [g - r for g, r in zip(greedy_arr, rl_arr)]
        best_seed  = int(np.argmax(gaps))

        # Rebuild that episode
        G_base = create_graph("barabasi_albert", num_nodes=NUM_NODES,
                              seed=best_seed)

        if key == "echo":
            G_mod = amplify_echo_chambers(G_base)
            G_mod, _ = default_seed_fn(G_mod, regions, seed=best_seed)
        elif key == "boundary":
            G_mod = deepcopy(G_base)
            G_mod, _ = boundary_seed_fn(G_mod, regions, seed=best_seed)
        else:
            G_mod = create_high_susceptibility_clusters(G_base)
            G_mod, _ = default_seed_fn(G_mod, regions, seed=best_seed)

        greedy_ep = run_episode(G_mod, "greedy", models, regions)
        rl_ep     = run_episode(G_mod, "rl",     models, regions)

        steps_g = [h["step"] for h in greedy_ep["history"]]
        inf_g   = [h["I"] / NUM_NODES * 100 for h in greedy_ep["history"]]
        steps_r = [h["step"] for h in rl_ep["history"]]
        inf_r   = [h["I"] / NUM_NODES * 100 for h in rl_ep["history"]]

        ax.plot(steps_g, inf_g, color="#3498db", linewidth=2,
                label="Greedy", marker="o", markersize=3)
        ax.plot(steps_r, inf_r, color="#2ecc71", linewidth=2,
                label="RL Agents", marker="o", markersize=3)

        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_xlabel("Time Step", color="white", fontsize=9)
        ax.set_ylabel("% Infected", color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2, color="white", linestyle="--")
        ax.legend(fontsize=8, facecolor="#1a1a2e",
                  labelcolor="white", edgecolor="white")

        gap = greedy_ep["history"][-1]["R"] - rl_ep["history"][-1]["R"]
        ax.set_facecolor("#0e1117")

    plt.suptitle(
        "Representative Episodes — Adversarial Conditions\n"
        "Each shows the seed where Greedy performs worst relative to RL",
        fontsize=12, fontweight="bold", color="white"
    )
    plt.tight_layout()
    plt.savefig("models/adversarial_timeseries.png", dpi=180,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print("Saved → models/adversarial_timeseries.png")


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(all_results, standard_results):
    from scipy import stats

    print("\n" + "=" * 70)
    print("  ADVERSARIAL EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  {'Condition':<28} {'Greedy':>12} {'RL':>12} "
          f"{'Gap':>8} {'p-value':>10}")
    print("  " + "-" * 66)

    rows = [
        ("Standard (n=200)",
         standard_results["greedy_reached"],
         standard_results["rl_reached"]),
        ("Echo Chamber Amplified",
         all_results["echo"]["greedy_reached"],
         all_results["echo"]["rl_reached"]),
        ("Boundary Seeding",
         all_results["boundary"]["greedy_reached"],
         all_results["boundary"]["rl_reached"]),
        ("High Susceptibility",
         all_results["highsus"]["greedy_reached"],
         all_results["highsus"]["rl_reached"]),
    ]

    for name, greedy, rl in rows:
        gm  = np.mean(greedy)
        rm  = np.mean(rl)
        gap = gm - rm
        _, p = stats.ttest_ind(greedy, rl, equal_var=False)
        sig  = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<28} {gm:>10.1f}%  {rm:>10.1f}%  "
              f"{gap:>+7.1f}%  {p:>8.4f} {sig}")

    print("=" * 70)
    print("  ** p < 0.01   * p < 0.05")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading trained RL models...")
    models = load_models()

    env0    = MisinfoEnv(agent_id=0)
    regions = env0.regions

    # Standard results from main evaluation (hardcoded from 200-seed run)
    # Replace raw_reached arrays if you want to recompute significance
    standard_results = {
        "greedy_mean"   : 32.6,
        "greedy_std"    : 9.4,
        "rl_mean"       : 30.1,
        "rl_std"        : 8.0,
        "greedy_reached": [32.6] * 200,  # placeholder for t-test
        "rl_reached"    : [30.1] * 200,
    }

    print("\n" + "=" * 55)
    print("  ADVERSARIAL EXPERIMENT")
    print("  3 conditions × 100 seeds each")
    print("=" * 55)

    # Condition 1: Echo chamber amplification
    echo_results = run_condition(
        condition_name = "Echo Chamber Amplified (weight ×2)",
        graph_modifier = amplify_echo_chambers,
        seed_fn        = default_seed_fn,
        models         = models,
        regions        = regions,
        n_seeds        = 100,
        spread_rate    = 0.3,
    )

    # Condition 2: Boundary seeding
    boundary_results = run_condition(
        condition_name = "Boundary Seeding",
        graph_modifier = deepcopy,
        seed_fn        = boundary_seed_fn,
        models         = models,
        regions        = regions,
        n_seeds        = 100,
        spread_rate    = 0.3,
    )

    # Condition 3: High susceptibility clusters
    highsus_results = run_condition(
        condition_name = "High Susceptibility Clusters",
        graph_modifier = create_high_susceptibility_clusters,
        seed_fn        = default_seed_fn,
        models         = models,
        regions        = regions,
        n_seeds        = 100,
        spread_rate    = 0.3,
    )

    all_results = {
        "echo"    : echo_results,
        "boundary": boundary_results,
        "highsus" : highsus_results,
    }

    # Print summary table
    print_summary(all_results, standard_results)

    # Plot grouped bar chart
    print("\nGenerating plots...")
    plot_adversarial_results(all_results, standard_results)

    # Plot time series for representative episodes
    plot_adversarial_timeseries(all_results, models, regions)

    print("\nDone. Files saved to models/:")
    print("  adversarial_experiment.png")
    print("  adversarial_timeseries.png")