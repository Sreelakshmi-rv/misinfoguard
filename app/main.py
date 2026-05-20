import streamlit as st

# ── Page config — must be first Streamlit call ──
st.set_page_config(
    page_title = "MisinfoGuard",
    layout     = "wide"
)

# ── Sidebar navigation ──
st.sidebar.title("MisinfoGuard")
st.sidebar.markdown("**Active Control of Misinformation Propagation using Multi-Agent RL**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home",
     "Network Visualizer",
     "Intervention Comparison",
     "NLP Classifier"]
)


# ── Route to pages ──
if page == "Home":

    st.title("MisinfoGuard")
    st.subheader("Active Control of Misinformation Propagation in Social Networks using Multi-Agent Reinforcement Learning")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### Network Visualizer\nGenerate a synthetic social network and watch misinformation spread in real time through the SIR model.")

    with col2:
        st.success("### Intervention Comparison\nCompare four strategies: No control, Random, Greedy, and trained RL Agents side by side.")

    with col3:
        st.warning("### NLP Classifier\nPaste any news claim. The system classifies it as Fake or Real and simulates how it would spread.")

    st.markdown("---")
    st.markdown("### How it works")

    st.markdown("""
    1. **Social Network** — A 500-node scale-free graph with heterogeneous edge weights modeling echo chambers.
    2. **SIR Spread Model** — Misinformation spreads like an epidemic: Susceptible → Infected → Recovered. Edge weights amplify spread through echo chambers.
    3. **Multi-Agent RL** — Two DQN agents control Louvain community regions under partial observability. Agents communicate via learned 8-dimensional message vectors.
    4. **NLP Layer** — A TF-IDF + XGBoost classifier detects misinformation and sets simulation threat level based on fake probability score.
    """)

    st.markdown("---")
    st.markdown("### System Architecture")

    import os
    arch_path = os.path.join(os.path.dirname(__file__), "architecture.png")
    if os.path.exists(arch_path):
        st.image(arch_path, use_container_width=True)
    else:
        st.info("Architecture diagram not found.")

    st.markdown("---")
    st.markdown("### Key Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("No Intervention",  "63.9%", "baseline", delta_color="off")
    r2.metric("Random Agents",    "51.6%", "-12.3%")
    r3.metric("Greedy Strategy",  "32.6%", "-31.3%")
    r4.metric("RL Agents (Ours)", "30.1%", "-33.8%")

    st.caption(
        "Average % of network infected across 200 graph instances "
        "(500 nodes, partial observability, message passing). "
        "RL vs Greedy gap: 2.56% (p=0.0035, Cohen's d=0.29)."
    )

elif page == "Network Visualizer":
    from page1_network import show
    show()

elif page == "Intervention Comparison":
    from page2_comparison import show
    show()

elif page == "NLP Classifier":
    from page3_nlp import show
    show()