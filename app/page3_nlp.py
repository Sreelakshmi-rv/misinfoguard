import streamlit as st
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_classifier import load_classifier, predict
from graph_env import create_graph
from sir_model import seed_infection, sir_step
from rl_env import (build_state, apply_action, MisinfoEnv,
                    generate_message, MESSAGE_SIZE)
from stable_baselines3 import DQN


# ─────────────────────────────────────────────
# LOAD CLASSIFIER
# ─────────────────────────────────────────────

@st.cache_resource
def load_nlp_model():
    # Returns embedder, xgb_model, party_encoder
    embedder, model, party_encoder = load_classifier()
    return embedder, model, party_encoder


@st.cache_resource
def load_rl_models():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    env0   = MisinfoEnv(agent_id=0, num_nodes=500)
    env1   = MisinfoEnv(agent_id=1, num_nodes=500)
    model0 = DQN.load(os.path.join(models_dir, "agent0_dqn"), env=env0)
    model1 = DQN.load(os.path.join(models_dir, "agent1_dqn"), env=env1)
    return model0, model1


# ─────────────────────────────────────────────
# SIMULATE SPREAD FOR NLP PAGE
# ─────────────────────────────────────────────

def simulate_spread(scenario, rl_models=None, seed=42,
                    spread_rate=0.4, num_seeds=5):
    from copy import deepcopy

    G_base = create_graph("barabasi_albert",
                          num_nodes=500, seed=seed)
    G_base, _ = seed_infection(G_base,
                               num_seeds = num_seeds,
                               strategy  = "high_degree",
                               seed      = seed)
    G = deepcopy(G_base)

    env0    = MisinfoEnv(agent_id=0, num_nodes=500)
    regions = env0.regions

    # Initialize messages
    messages = [
        np.zeros(MESSAGE_SIZE, dtype=np.float32),
        np.zeros(MESSAGE_SIZE, dtype=np.float32)
    ]

    history = []

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

        elif scenario == "rl":
            for agent_id, region in enumerate(regions):
                received = [messages[1 - agent_id]]
                state    = build_state(
                    G, region,
                    all_regions       = regions,
                    received_messages = received
                )
                action, _ = rl_models[agent_id].predict(
                    state, deterministic=True
                )
                G, _ = apply_action(G, int(action), region)

            for agent_id, region in enumerate(regions):
                messages[agent_id] = generate_message(G, region)

        G, _, _ = sir_step(G,
                           recovery_time = 4,
                           spread_rate   = spread_rate)

    return history


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

def show():
    st.title("NLP Misinformation Classifier")
    st.markdown("Paste any news claim or headline. The system classifies it and simulates how it would spread if it entered the network.")
    st.markdown("---")

    # ── Load models ──
    with st.spinner("Loading classifier..."):
        embedder, nlp_model, party_encoder = load_nlp_model()

    # ── Text input ──
    st.markdown("### Enter a News Claim")

    example_claims = [
        "Select an example or type your own...",
        "The government is putting chemicals in the water supply.",
        "Scientists confirm that climate change is accelerating.",
        "This politician voted against the healthcare bill 12 times.",
        "The unemployment rate has reached a record low this quarter.",
        "5G towers are causing health problems in nearby residents.",
    ]

    selected_example = st.selectbox("Quick examples:", example_claims)

    user_text = st.text_area(
        "Or type your own claim:",
        value = "" if selected_example == example_claims[0]
                else selected_example,
        height = 100,
        placeholder = "Type a news headline or claim here..."
    )

    if st.button("Analyze Claim", use_container_width=True):

        if not user_text.strip():
            st.warning("Please enter a claim to analyze.")
            return

        # Default seed count — overridden by threat assessment below
        num_seeds = 3

        # ── Run classification ──
        with st.spinner("Classifying..."):
            result = predict(user_text.strip(), embedder, nlp_model, party_encoder)

        st.markdown("---")
        st.markdown("### Classification Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            if result["label"] == "FAKE":
                st.error(f"## FAKE\nConfidence: {result['confidence']}%")
            else:
                st.success(f"## REAL\nConfidence: {result['confidence']}%")

        with col2:
            st.markdown("**Probability Breakdown**")
            st.markdown(f"- 🔴 Fake probability: **{result['fake_prob']}%**")
            st.markdown(f"- 🟢 Real probability: **{result['real_prob']}%**")

            # Visual probability bar
            fake_frac = result["fake_prob"] / 100
            real_frac = result["real_prob"] / 100
            st.progress(real_frac)
            st.caption(f"← Real ({result['real_prob']}%)  |  "
                       f"Fake ({result['fake_prob']}%) →")

        # ── Simulate spread if fake ──
        st.markdown("---")
        st.markdown("### Threat Assessment")

        fake_prob = result["fake_prob"]

        # Determine threat level based on fake probability
        # fake_prob drives BOTH seed count AND spread rate.
        # Higher fake probability = more seeds + faster spread.
        # This makes the NLP output genuinely affect simulation behavior.
        if fake_prob >= 70:
            threat_level = "🔴 CRITICAL"
            threat_color = "error"
            num_seeds    = 5
            spread_rate  = 0.55
            threat_desc  = ("High confidence this is misinformation. "
                           "Simulating aggressive spread (rate=0.55, seeds=5).")
        elif fake_prob >= 50:
            threat_level = "🟠 HIGH"
            threat_color = "warning"
            num_seeds    = 3
            spread_rate  = 0.40
            threat_desc  = ("Moderate-high probability of misinformation. "
                           "Simulating standard spread (rate=0.40, seeds=3).")
        elif fake_prob >= 30:
            threat_level = "🟡 MEDIUM"
            threat_color = "warning"
            num_seeds    = 2
            spread_rate  = 0.25
            threat_desc  = ("Uncertain classification. "
                           "Simulating limited spread (rate=0.25, seeds=2).")
        else:
            threat_level = "🟢 LOW"
            threat_color = "info"
            spread_rate  = 0.15
            num_seeds    = 1
            threat_desc  = ("Low probability of misinformation. "
                           "Simulating minimal spread scenario.")

        # Display threat assessment
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Threat Level",    threat_level)
        col_t2.metric("Fake Probability", f"{fake_prob}%")
        col_t3.metric("Simulated Seeds", num_seeds)
        col_t4.metric("Spread Rate",     spread_rate,
                      help="Higher fake probability → faster spread simulation")

        if threat_color == "error":
            st.error(f"⚠️ {threat_desc}")
        elif threat_color == "warning":
            st.warning(f"⚠️ {threat_desc}")
        else:
            st.info(f"ℹ️ {threat_desc}")

        st.markdown("### Network Spread Simulation")
        st.caption(
            f"Simulation uses **{num_seeds} seed node(s)** based on "
            f"fake probability score of {fake_prob}%. "
            f"Higher fake probability → more aggressive spread simulation."
        )

        with st.spinner("Running spread simulations..."):
            rl_model0, rl_model1 = load_rl_models()
            rl_models = [rl_model0, rl_model1]

            history_none = simulate_spread(
                "none",
                seed        = 42,
                num_seeds   = num_seeds,
                spread_rate = spread_rate
            )
            history_rl = simulate_spread(
                "rl",
                rl_models   = rl_models,
                seed        = 42,
                num_seeds   = num_seeds,
                spread_rate = spread_rate
            )

        # ── Plot comparison ──
        import pandas as pd
        import altair as alt

        chart_data = []
        total = 500

        for h in history_none:
            chart_data.append({
                "Step"      : h["step"],
                "Infected %": round(h["I"] / total * 100, 2),
                "Scenario"  : "No Intervention"
            })
        for h in history_rl:
            chart_data.append({
                "Step"      : h["step"],
                "Infected %": round(h["I"] / total * 100, 2),
                "Scenario"  : "RL Agents (Ours)"
            })

        df = pd.DataFrame(chart_data)

        color_scale = alt.Scale(
            domain = ["No Intervention", "RL Agents (Ours)"],
            range  = ["#e74c3c", "#2ecc71"]
        )

        chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
            x     = alt.X("Step:Q",        title="Time Step"),
            y     = alt.Y("Infected %:Q",  title="% of Network Infected"),
            color = alt.Color("Scenario:N", scale=color_scale)
        ).properties(
            height = 300,
            title  = "Spread With vs Without RL Intervention"
        )

        st.altair_chart(chart, use_container_width=True)

        # ── Final metrics ──
        col_a, col_b, col_c = st.columns(3)

        none_reached = history_none[-1]["R"] + history_none[-1]["I"]
        rl_reached   = history_rl[-1]["R"]   + history_rl[-1]["I"]
        reduction    = round((none_reached - rl_reached) /
                             none_reached * 100, 1)

        col_a.metric("Without Intervention",
                     f"{round(none_reached/total*100,1)}% reached")
        col_b.metric("With RL Agents",
                     f"{round(rl_reached/total*100,1)}% reached")
        col_c.metric("Spread Reduction",
                     f"{reduction}%",
                     delta=f"-{reduction}%")

        st.markdown("---")
        st.caption(
            "Note: The NLP classifier achieves 74.4% accuracy (5-fold CV) on the LIAR "
            "political fact-checking dataset. Performance is lower on scientific "
            "consensus claims, which are underrepresented in LIAR. "
            "Spread simulation uses a 500-node scale-free network."
        )