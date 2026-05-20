"""
simulation.py — Unified scenario runner for MisinfoGuard.

Single source of truth used by:
    train_agents.py, page2_comparison.py, multi_network_eval.py

Updated to accept graph_type parameter so the same runner
works across all 4 network types.
"""

import numpy as np
from copy import deepcopy

from graph_env import create_graph, NETWORK_CONFIGS
from sir_model  import seed_infection, sir_step
from rl_env     import (
    MisinfoEnv,
    build_state,
    apply_action,
    generate_message,
    get_visible_nodes,
    NUM_NODES,
    NUM_REGIONS,
    NUM_SEEDS,
    MESSAGE_SIZE,
)


# ─────────────────────────────────────────────
# SCENARIO LABELS AND COLORS
# ─────────────────────────────────────────────

SCENARIO_LABELS = {
    "none"  : "No Intervention",
    "random": "Random Agents",
    "greedy": "Greedy (Degree)",
    "rl"    : "RL Agents (Ours)",
}

SCENARIO_COLORS = {
    "none"  : "#e74c3c",
    "random": "#e67e22",
    "greedy": "#3498db",
    "rl"    : "#2ecc71",
}


# ─────────────────────────────────────────────
# SINGLE EPISODE RUNNER
# ─────────────────────────────────────────────

def run_episode(
    scenario,
    models         = None,
    num_nodes      = NUM_NODES,
    num_seeds      = NUM_SEEDS,
    seed_strategy  = "high_degree",
    spread_rate    = 0.3,
    recovery_time  = 4,
    max_steps      = 60,
    seed           = 42,
    graph_type     = "barabasi_albert",
):
    if scenario == "rl" and models is None:
        raise ValueError("models must be provided for rl scenario")

    G = create_graph(graph_type, num_nodes=num_nodes, seed=seed)
    G, _ = seed_infection(
        G, num_seeds=num_seeds, strategy=seed_strategy, seed=seed
    )

    env0    = MisinfoEnv(agent_id=0, num_nodes=num_nodes,
                         graph_type=graph_type)
    regions = env0.regions

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

        if scenario == "none":
            pass

        elif scenario == "random":
            for region in regions:
                action = np.random.randint(0, 4)
                G, _   = apply_action(G, action, region)

        elif scenario == "greedy":
            for region in regions:
                visible  = get_visible_nodes(G, region)
                infected = [n for n in visible
                            if G.nodes[n]["status"] == "I"]
                if infected:
                    target = max(infected, key=lambda n: G.degree(n))
                    G.nodes[target]["status"]        = "R"
                    G.nodes[target]["infected_time"] = 999

        elif scenario == "rl":
            for agent_id, region in enumerate(regions):
                received = [messages[1 - agent_id]]
                state    = build_state(
                    G, region,
                    all_regions       = regions,
                    received_messages = received,
                )
                action, _ = models[agent_id].predict(
                    state, deterministic=True
                )
                G, _ = apply_action(G, int(action), region)

            for agent_id, region in enumerate(regions):
                messages[agent_id] = generate_message(G, region)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        G, _, _ = sir_step(
            G, recovery_time=recovery_time, spread_rate=spread_rate
        )

    return history


# ─────────────────────────────────────────────
# MULTI-SEED EVALUATOR
# ─────────────────────────────────────────────

def evaluate_scenarios(
    scenarios,
    models         = None,
    num_seeds_eval = 30,
    num_nodes      = NUM_NODES,
    num_seeds      = NUM_SEEDS,
    seed_strategy  = "high_degree",
    spread_rate    = 0.3,
    recovery_time  = 4,
    graph_type     = "barabasi_albert",
    verbose        = True,
):
    results = {s: {"peak": [], "reached": [], "steps": []}
               for s in scenarios}

    for seed in range(num_seeds_eval):
        if verbose and seed % 5 == 0:
            print(f"  Seed {seed}/{num_seeds_eval}...")

        for scenario in scenarios:
            history = run_episode(
                scenario      = scenario,
                models        = models,
                num_nodes     = num_nodes,
                num_seeds     = num_seeds,
                seed_strategy = seed_strategy,
                spread_rate   = spread_rate,
                recovery_time = recovery_time,
                seed          = seed,
                graph_type    = graph_type,
            )

            total   = num_nodes
            peak_i  = max(h["I"] for h in history)
            final   = history[-1]
            reached = final["R"] + final["I"]
            steps   = final["step"]

            results[scenario]["peak"].append(peak_i / total * 100)
            results[scenario]["reached"].append(reached / total * 100)
            results[scenario]["steps"].append(steps)

    summary = {}
    for scenario in scenarios:
        peak_arr    = results[scenario]["peak"]
        reached_arr = results[scenario]["reached"]
        steps_arr   = results[scenario]["steps"]

        summary[scenario] = {
            "peak_mean"    : round(np.mean(peak_arr),    2),
            "peak_std"     : round(np.std(peak_arr),     2),
            "reached_mean" : round(np.mean(reached_arr), 2),
            "reached_std"  : round(np.std(reached_arr),  2),
            "steps_mean"   : round(np.mean(steps_arr),   1),
            "raw_peak"     : peak_arr,
            "raw_reached"  : reached_arr,
            "n_seeds"      : num_seeds_eval,
        }

    if verbose:
        _print_summary(summary, scenarios, graph_type)

    return summary


# ─────────────────────────────────────────────
# QUICK SINGLE-RUN COMPARISON
# ─────────────────────────────────────────────

def run_comparison(
    scenarios,
    models        = None,
    num_nodes     = NUM_NODES,
    num_seeds     = NUM_SEEDS,
    seed_strategy = "high_degree",
    spread_rate   = 0.3,
    recovery_time = 4,
    seed          = 42,
    graph_type    = "barabasi_albert",
):
    results = {}
    for scenario in scenarios:
        results[scenario] = run_episode(
            scenario      = scenario,
            models        = models,
            num_nodes     = num_nodes,
            num_seeds     = num_seeds,
            seed_strategy = seed_strategy,
            spread_rate   = spread_rate,
            recovery_time = recovery_time,
            seed          = seed,
            graph_type    = graph_type,
        )
    return results


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def _print_summary(summary, scenarios, graph_type="barabasi_albert"):
    label = NETWORK_CONFIGS.get(graph_type, {}).get("label", graph_type)
    print("\n" + "=" * 65)
    print(f"  RESULTS — {label}")
    print("=" * 65)
    print(f"  {'Scenario':<28} {'Peak Infected':>15} {'Total Reached':>15}")
    print("  " + "-" * 60)

    for s in scenarios:
        m         = summary[s]
        lbl       = SCENARIO_LABELS.get(s, s)
        peak_str  = f"{m['peak_mean']:.1f}% ± {m['peak_std']:.1f}%"
        reach_str = f"{m['reached_mean']:.1f}% ± {m['reached_std']:.1f}%"
        print(f"  {lbl:<28} {peak_str:>15} {reach_str:>15}")

    print("=" * 65)

    if "rl" in summary and "greedy" in summary:
        gap = summary["greedy"]["reached_mean"] - summary["rl"]["reached_mean"]
        print(f"\n  RL vs Greedy gap: {gap:+.1f}% "
              f"({'RL better' if gap > 0 else 'Greedy better'})")
    print()