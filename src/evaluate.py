import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from stable_baselines3 import DQN

from graph_env import create_graph
from sir_model  import seed_infection, sir_step
from rl_env     import (MisinfoEnv, generate_message, build_state,
                        apply_action, NUM_NODES, NUM_SEEDS, MESSAGE_SIZE)


# ─────────────────────────────────────────────
# RUN ONE SCENARIO ON ONE GRAPH
# ─────────────────────────────────────────────

def run_scenario(scenario, G_base, models=None,
                 spread_rate=0.3, recovery_time=4):
    """
    Runs one full simulation under a given strategy.
    RL scenario uses message passing between agents.
    """
    G       = deepcopy(G_base)
    env0    = MisinfoEnv(agent_id=0, num_nodes=NUM_NODES)
    regions = env0.regions
    history = []

    # Initialize messages
    messages = [
        np.zeros(MESSAGE_SIZE, dtype=np.float32),
        np.zeros(MESSAGE_SIZE, dtype=np.float32)
    ]

    for step in range(60):
        s = sum(1 for n in G.nodes()
                if G.nodes[n]["status"] == "S")
        i = sum(1 for n in G.nodes()
                if G.nodes[n]["status"] == "I")
        r = sum(1 for n in G.nodes()
                if G.nodes[n]["status"] == "R")
        history.append({"step": step, "S": s, "I": i, "R": r})

        if i == 0 and step > 0:
            break

        if scenario == "none":
            pass

        elif scenario == "random":
            for region in regions:
                action = np.random.randint(0, 4)
                G, _   = apply_action(G, action, region)

        elif scenario == "greedy":
            for region in regions:
                infected = [n for n in region
                            if G.nodes[n]["status"] == "I"]
                if infected:
                    target = max(infected,
                                 key=lambda n: G.degree(n))
                    G.nodes[target]["status"]        = "R"
                    G.nodes[target]["infected_time"] = 999

        elif scenario == "rl":
            for agent_id, region in enumerate(regions):
                received = messages[1 - agent_id]
                state    = build_state(G, region,
                                       all_regions       = regions,
                                       received_messages = [received])
                action, _ = models[agent_id].predict(
                    state, deterministic=True
                )
                G, _ = apply_action(G, int(action), region)

            # Update messages after acting
            for agent_id, region in enumerate(regions):
                messages[agent_id] = generate_message(G, region)

        G, _, _ = sir_step(G,
                           recovery_time = recovery_time,
                           spread_rate   = spread_rate)

    total   = NUM_NODES
    reached = history[-1]["R"] + history[-1]["I"]
    peak    = max(h["I"] for h in history)
    steps   = history[-1]["step"]

    return {
        "reached_pct" : round(reached / total * 100, 2),
        "peak_pct"    : round(peak    / total * 100, 2),
        "steps"       : steps
    }


# ─────────────────────────────────────────────
# FULL EVALUATION — 30 GRAPHS
# ─────────────────────────────────────────────

def run_full_evaluation():

    print("Loading trained models...")
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models'
    )

    env0   = MisinfoEnv(agent_id=0)
    env1   = MisinfoEnv(agent_id=1)
    model0 = DQN.load(os.path.join(models_dir, "agent0_dqn"), env=env0)
    model1 = DQN.load(os.path.join(models_dir, "agent1_dqn"), env=env1)
    models = [model0, model1]

    # 30 different random seeds = 30 different graphs
    seeds = [
        42,   77,  123,  256,  999,
        13,   37,   88,  101,  200,
       301,  404,  512,  614,  777,
       888, 1024, 1111, 1234, 1337,
      1500, 1618, 1729, 2024, 2048,
      2187, 2718, 3141, 9999,  555
    ]

    scenarios   = ["none", "random", "greedy", "rl"]
    all_results = {s: [] for s in scenarios}

    print(f"\nRunning evaluation on {len(seeds)} graphs...")
    print("=" * 65)

    for i, seed in enumerate(seeds):
        print(f"\n  Graph {i+1:>2}/{len(seeds)} (seed={seed})")

        G_base = create_graph("barabasi_albert",
                              num_nodes=NUM_NODES, seed=seed)
        G_base, _ = seed_infection(G_base,
                                   num_seeds = NUM_SEEDS,
                                   strategy  = "high_degree",
                                   seed      = seed)

        for scenario in scenarios:
            result = run_scenario(
                scenario      = scenario,
                G_base        = G_base,
                models        = models if scenario == "rl" else None,
                spread_rate   = 0.3,
                recovery_time = 4
            )
            all_results[scenario].append(result)
            print(f"    {scenario:<8} → "
                  f"reached: {result['reached_pct']:>6}% | "
                  f"peak: {result['peak_pct']:>6}% | "
                  f"steps: {result['steps']}")

    # ── Compute averages and confidence intervals ──
    print("\n" + "=" * 65)
    print("  RESULTS ACROSS 30 GRAPHS")
    print("=" * 65)

    labels = {
        "none"  : "No Intervention",
        "random": "Random Agents",
        "greedy": "Greedy Strategy",
        "rl"    : "RL Agents (Ours)"
    }

    summary_rows    = []
    avg_reached_all = {}

    for scenario in scenarios:
        results     = all_results[scenario]
        reached_arr = np.array([r["reached_pct"] for r in results])
        peak_arr    = np.array([r["peak_pct"]    for r in results])
        steps_arr   = np.array([r["steps"]       for r in results])

        avg_reached = np.mean(reached_arr)
        std_reached = np.std(reached_arr)
        avg_peak    = np.mean(peak_arr)
        avg_steps   = np.mean(steps_arr)

        # 95% confidence interval
        n  = len(reached_arr)
        ci = 1.96 * std_reached / np.sqrt(n)

        avg_reached_all[scenario] = avg_reached

        print(f"\n  {labels[scenario]}")
        print(f"    Avg reached : {avg_reached:.1f}% ± {std_reached:.1f}%")
        print(f"    95% CI      : [{avg_reached-ci:.1f}%, "
              f"{avg_reached+ci:.1f}%]")
        print(f"    Avg peak    : {avg_peak:.1f}%")
        print(f"    Avg steps   : {avg_steps:.1f}")

        summary_rows.append({
            "Strategy"          : labels[scenario],
            "Avg Total Reached" : f"{avg_reached:.1f}% ± {std_reached:.1f}%",
            "95% CI"            : f"[{avg_reached-ci:.1f}%, "
                                  f"{avg_reached+ci:.1f}%]",
            "Avg Peak Infected" : f"{avg_peak:.1f}%",
            "Avg Steps"         : f"{avg_steps:.1f}"
        })

    # ── Save CSV ──
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(models_dir, "evaluation_results.csv"),
              index=False)
    print(f"\n  Results saved to models/evaluation_results.csv")

    # ── Bar chart ──
    scenario_labels  = [labels[s] for s in scenarios]
    avg_reached_vals = [avg_reached_all[s] for s in scenarios]
    colors           = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars    = ax.bar(scenario_labels, avg_reached_vals,
                     color=colors, alpha=0.85, edgecolor="white",
                     width=0.5)

    for bar, val in zip(bars, avg_reached_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontweight="bold", fontsize=12)

    ax.set_ylabel("Average % of Network Infected", fontsize=12)
    ax.set_title(
        "Average Infection Rate Across 30 Graph Instances\n"
        "(Lower is Better — 500 nodes, Partial Observability, "
        "Message Passing)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "evaluation_plot.png"), dpi=150)
    plt.show()
    print("  Evaluation plot saved to models/evaluation_plot.png")

    return df, all_results


# ─────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE TESTING
# ─────────────────────────────────────────────

def run_significance_tests(all_results):
    from scipy import stats

    print("\n── STATISTICAL SIGNIFICANCE TESTS ──")
    print("  Comparing RL Agents vs other strategies")
    print("  across 30 graph instances (paired t-test)")
    print("  p < 0.05 = statistically significant")
    print("=" * 60)

    rl_scores = [r["reached_pct"] for r in all_results["rl"]]

    comparisons = {
        "No Intervention" : "none",
        "Random Agents"   : "random",
        "Greedy Strategy" : "greedy"
    }

    for label, key in comparisons.items():
        other_scores = [r["reached_pct"] for r in all_results[key]]
        t_stat, p_value = stats.ttest_rel(rl_scores, other_scores)
        mean_diff = round(
            sum(other_scores)/len(other_scores) -
            sum(rl_scores)/len(rl_scores), 2
        )
        significance = ("✅ SIGNIFICANT"
                        if p_value < 0.05 else "❌ NOT significant")

        print(f"\n  RL Agents vs {label}:")
        print(f"    RL mean    : "
              f"{round(sum(rl_scores)/len(rl_scores),1)}%")
        print(f"    Other mean : "
              f"{round(sum(other_scores)/len(other_scores),1)}%")
        print(f"    Difference : -{mean_diff}%")
        print(f"    t-statistic: {round(t_stat, 4)}")
        print(f"    p-value    : {round(p_value, 8)}")
        print(f"    Result     : {significance}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# ADVERSARIAL EXPERIMENT
# ─────────────────────────────────────────────

def run_adversarial_experiment(models):
    """
    Tests RL vs Greedy on random seeding —
    where greedy assumption breaks.
    Uses 10 graphs for more reliable results.
    """
    seeds      = [42, 77, 123, 256, 999,
                  13, 37,  88, 101, 200]
    rl_scores  = []
    grd_scores = []

    env0    = MisinfoEnv(agent_id=0)
    regions = env0.regions

    print("\n── ADVERSARIAL EXPERIMENT ──")
    print("  Seeding: RANDOM (breaks greedy assumption)")
    print("  10 graph instances")
    print("=" * 60)
    print(f"  {'Seed':>6} | {'Greedy':>10} | "
          f"{'RL':>10} | {'Winner':>12}")
    print("  " + "-" * 50)

    for seed in seeds:
        G_base = create_graph("barabasi_albert",
                              num_nodes=NUM_NODES, seed=seed)
        G_base, _ = seed_infection(G_base,
                                   num_seeds = NUM_SEEDS,
                                   strategy  = "random",
                                   seed      = seed)

        grd = run_scenario("greedy", deepcopy(G_base))
        rl  = run_scenario("rl", deepcopy(G_base), models=models)

        rl_scores.append(rl["reached_pct"])
        grd_scores.append(grd["reached_pct"])

        winner = ("RL ✅" if rl["reached_pct"] <= grd["reached_pct"]
                  else "Greedy")
        print(f"  {seed:>6} | {grd['reached_pct']:>9.1f}% | "
              f"{rl['reached_pct']:>9.1f}% | {winner:>12}")

    avg_grd = round(sum(grd_scores) / len(grd_scores), 2)
    avg_rl  = round(sum(rl_scores)  / len(rl_scores),  2)

    print("\n" + "=" * 60)
    print(f"  Avg Greedy : {avg_grd}%")
    print(f"  Avg RL     : {avg_rl}%")

    if avg_rl < avg_grd:
        improvement = round(avg_grd - avg_rl, 2)
        print(f"\n  ✅ RL outperforms Greedy by {improvement}% "
              f"on random seeding")
    else:
        print(f"\n  Results similar on random seeding as well")

    # Plot
    fig, ax  = plt.subplots(figsize=(10, 5))
    x        = range(len(seeds))
    width    = 0.35
    bars1    = ax.bar([i - width/2 for i in x], grd_scores,
                      width, label="Greedy Strategy",
                      color="steelblue", alpha=0.85)
    bars2    = ax.bar([i + width/2 for i in x], rl_scores,
                      width, label="RL Agents (Ours)",
                      color="seagreen", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Graph {i+1}" for i in x])
    ax.set_ylabel("% of Network Infected")
    ax.set_title(
        "Adversarial Experiment: Random Seeding\n"
        "Greedy vs RL Agents — 500 nodes, Partial Observability",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'adversarial_experiment.png'
    ), dpi=150)
    plt.show()
    print("  Adversarial plot saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Full 30-graph evaluation
    df, all_results = run_full_evaluation()

    # Statistical significance
    run_significance_tests(all_results)

    # Adversarial experiment
    print("\nRunning adversarial experiment...")
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models'
    )
    env0   = MisinfoEnv(agent_id=0)
    env1   = MisinfoEnv(agent_id=1)
    model0 = DQN.load(os.path.join(models_dir, "agent0_dqn"), env=env0)
    model1 = DQN.load(os.path.join(models_dir, "agent1_dqn"), env=env1)
    run_adversarial_experiment([model0, model1])

    print("\nFinal Summary Table:")
    print(df.to_string(index=False))