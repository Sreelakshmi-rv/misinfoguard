"""
page2_comparison.py — Intervention Comparison dashboard page.

Updated to support all 4 network types with per-network trained agents.
"""

import streamlit as st
import sys
import os
import pandas as pd
import altair as alt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation import (
    run_comparison,
    evaluate_scenarios,
    SCENARIO_LABELS,
    SCENARIO_COLORS,
)
from graph_env  import NETWORK_CONFIGS
from rl_env     import MisinfoEnv, NUM_NODES
from stable_baselines3 import DQN


# ─────────────────────────────────────────────
# LOAD MODELS — CACHED PER NETWORK TYPE
# ─────────────────────────────────────────────

@st.cache_resource
def load_models(graph_type="barabasi_albert"):
    """
    Loads trained agent pair for the selected network type.
    Falls back to base BA agents if network-specific not found.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    env0 = MisinfoEnv(agent_id=0, num_nodes=500, graph_type=graph_type)
    env1 = MisinfoEnv(agent_id=1, num_nodes=500, graph_type=graph_type)

    path0 = os.path.join(models_dir, f"agent0_{graph_type}_dqn")
    path1 = os.path.join(models_dir, f"agent1_{graph_type}_dqn")

    # Fall back to base model
    if not os.path.exists(path0 + ".zip"):
        st.warning(f"Network-specific agents for {graph_type} not found. "
                   f"Using Barabási-Albert trained agents.")
        path0 = os.path.join(models_dir, "agent0_dqn")
        path1 = os.path.join(models_dir, "agent1_dqn")

    model0 = DQN.load(path0, env=env0)
    model1 = DQN.load(path1, env=env1)
    return model0, model1


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

def show():
    st.title("Intervention Comparison")
    st.markdown(
        "Compare four intervention strategies across different network types. "
        "Each network type has independently trained RL agents."
    )
    st.markdown("---")

    # ── Sidebar ──
    st.sidebar.markdown("### Network Type")

    graph_type = st.sidebar.selectbox(
        "Select Network",
        options = list(NETWORK_CONFIGS.keys()),
        format_func = lambda x: NETWORK_CONFIGS[x]["label"],
        help = "Each network type models a different social structure."
    )

    # Show network description
    st.sidebar.info(NETWORK_CONFIGS[graph_type]["description"])
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Simulation Settings")
    st.sidebar.info("Network size fixed at 500 nodes.")

    num_seeds     = st.sidebar.slider("Initial Infected Nodes", 1, 10, 5)
    seed_strategy = st.sidebar.selectbox("Seed Strategy",
                                         ["high_degree", "random"])
    spread_rate   = st.sidebar.slider("Spread Rate", 0.1, 0.9, 0.3, 0.1)
    recovery_time = st.sidebar.slider("Recovery Time (steps)", 2, 10, 4)
    random_seed   = st.sidebar.number_input("Random Seed", value=8, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Scenarios")
    run_none   = st.sidebar.checkbox("No Intervention",  value=True)
    run_random = st.sidebar.checkbox("Random Agents",    value=True)
    run_greedy = st.sidebar.checkbox("Greedy (Degree)",  value=True)
    run_rl     = st.sidebar.checkbox("RL Agents (Ours)", value=True)

    st.sidebar.markdown("---")
    multi_seed_n  = st.sidebar.slider("Multi-seed runs", 5, 100, 30)
    run_multiseed = st.sidebar.checkbox("Run multi-seed evaluation",
                                        value=False)

    # ── Network info banner ──
    config = NETWORK_CONFIGS[graph_type]
    st.info(f"**{config['label']}** — {config['description']}")

    # ── Run button ──
    if st.button("Run Comparison", use_container_width=True):

        with st.spinner(f"Loading RL agents for {config['label']}..."):
            model0, model1 = load_models(graph_type)
            models = [model0, model1]

        scenarios_to_run = []
        if run_none:   scenarios_to_run.append("none")
        if run_random: scenarios_to_run.append("random")
        if run_greedy: scenarios_to_run.append("greedy")
        if run_rl:     scenarios_to_run.append("rl")

        if not scenarios_to_run:
            st.warning("Select at least one scenario.")
            return

        progress = st.progress(0)

        with st.spinner("Running single-seed comparison..."):
            results = run_comparison(
                scenarios     = scenarios_to_run,
                models        = models,
                num_nodes     = NUM_NODES,
                num_seeds     = num_seeds,
                seed_strategy = seed_strategy,
                spread_rate   = spread_rate,
                recovery_time = recovery_time,
                seed          = int(random_seed),
                graph_type    = graph_type,
            )
        progress.progress(0.6)

        summary = None
        if run_multiseed:
            with st.spinner(f"Running {multi_seed_n}-seed evaluation..."):
                summary = evaluate_scenarios(
                    scenarios      = scenarios_to_run,
                    models         = models,
                    num_seeds_eval = multi_seed_n,
                    num_nodes      = NUM_NODES,
                    num_seeds      = num_seeds,
                    seed_strategy  = seed_strategy,
                    spread_rate    = spread_rate,
                    recovery_time  = recovery_time,
                    graph_type     = graph_type,
                    verbose        = False,
                )
        progress.progress(1.0)
        progress.empty()

        st.session_state["comparison_results"] = results
        st.session_state["comparison_summary"] = summary
        st.session_state["comparison_nodes"]   = NUM_NODES
        st.session_state["scenarios_run"]      = scenarios_to_run
        st.session_state["graph_type"]         = graph_type
        st.success("Done!")

    # ── Display results ──
    if "comparison_results" not in st.session_state:
        return

    results   = st.session_state["comparison_results"]
    summary   = st.session_state["comparison_summary"]
    num_nodes = st.session_state["comparison_nodes"]
    scenarios = st.session_state["scenarios_run"]
    gt        = st.session_state.get("graph_type", "barabasi_albert")

    st.markdown(f"### Infection Rate Over Time — "
                f"{NETWORK_CONFIGS[gt]['label']}")

    chart_data = []
    for scenario, history in results.items():
        for h in history:
            chart_data.append({
                "Step"      : h["step"],
                "Infected %": round(h["I"] / num_nodes * 100, 2),
                "Scenario"  : SCENARIO_LABELS.get(scenario, scenario),
            })

    df = pd.DataFrame(chart_data)
    color_scale = alt.Scale(
        domain = [SCENARIO_LABELS[s] for s in scenarios],
        range  = [SCENARIO_COLORS[s]  for s in scenarios],
    )

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x     = alt.X("Step:Q",        title="Time Step"),
        y     = alt.Y("Infected %:Q",  title="% of Network Infected"),
        color = alt.Color("Scenario:N", scale=color_scale),
    ).properties(
        height = 380,
        title  = f"Misinformation Spread — {NETWORK_CONFIGS[gt]['label']}"
    )
    st.altair_chart(chart, use_container_width=True)

    # ── Single-seed table ──
    st.markdown("### Single-Seed Summary")
    table_rows = []
    for scenario in scenarios:
        history = results[scenario]
        peak_i  = max(h["I"] for h in history)
        final   = history[-1]
        reached = final["R"] + final["I"]
        table_rows.append({
            "Strategy"         : SCENARIO_LABELS.get(scenario, scenario),
            "Peak Infected"    : f"{round(peak_i/num_nodes*100, 1)}%",
            "Total Reached"    : f"{reached} ({round(reached/num_nodes*100,1)}%)",
            "Steps to Contain" : final["step"],
        })

    st.dataframe(pd.DataFrame(table_rows),
                 use_container_width=True, hide_index=True)

    # ── Multi-seed table ──
    if summary is not None:
        st.markdown("---")
        st.markdown("### Multi-Seed Evaluation (Mean ± Std)")

        ms_rows = []
        for scenario in scenarios:
            m = summary[scenario]
            ms_rows.append({
                "Strategy"    : SCENARIO_LABELS.get(scenario, scenario),
                "Peak Infected": f"{m['peak_mean']:.1f}% ± {m['peak_std']:.1f}%",
                "Total Reached": f"{m['reached_mean']:.1f}% ± {m['reached_std']:.1f}%",
                "Avg Steps"   : m["steps_mean"],
            })

        st.dataframe(pd.DataFrame(ms_rows),
                     use_container_width=True, hide_index=True)

        if "rl" in summary and "greedy" in summary:
            gap = round(summary["greedy"]["reached_mean"]
                        - summary["rl"]["reached_mean"], 1)
            if gap > 0:
                st.success(
                    f"**RL Agents reduced total infection by {gap}% vs "
                    f"Greedy on {NETWORK_CONFIGS[gt]['label']} network.**"
                )
            else:
                st.warning(
                    f"**Greedy outperformed RL by {abs(gap)}% on this "
                    f"network type. Consider retraining agents.**"
                )

    # ── Key insight ──
    st.markdown("---")
    if "none" in results and "rl" in results:
        none_r   = results["none"][-1]["R"] + results["none"][-1]["I"]
        rl_r     = results["rl"][-1]["R"]   + results["rl"][-1]["I"]
        reduction = round((none_r - rl_r) / none_r * 100, 1)
        st.info(
            f"**RL Agents reduced total network infection by "
            f"{reduction}% compared to no intervention.**"
        )