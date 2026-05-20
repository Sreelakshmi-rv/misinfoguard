import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_env import create_graph, print_graph_summary
from sir_model  import seed_infection, sir_step
from pyvis.network import Network
import networkx as nx


# ─────────────────────────────────────────────
# HELPER — BUILD PYVIS HTML WITH SIR COLORS
# ─────────────────────────────────────────────

def build_pyvis_html(G):
    """
    Renders the graph as an interactive pyvis HTML string.
    Node colors reflect SIR status:
        S = steelblue
        I = crimson
        R = seagreen
    Node size reflects influence.
    """
    status_colors = {
        "S": "#4a90d9",   # blue
        "I": "#e74c3c",   # red
        "R": "#2ecc71",   # green
    }

    net = Network(height="500px", width="100%",
                  bgcolor="#0e1117", font_color="white")

    for node in G.nodes():
        status = G.nodes[node]["status"]
        inf    = G.nodes[node]["influence"]
        sus    = G.nodes[node]["susceptibility"]
        color  = status_colors[status]
        size   = 6 + inf * 40

        net.add_node(
            node,
            label = str(node),
            color = color,
            size  = size,
            title = (f"Node {node}<br>"
                     f"Status: {status}<br>"
                     f"Influence: {inf:.3f}<br>"
                     f"Susceptibility: {sus:.3f}")
        )

    for edge in G.edges():
        net.add_edge(edge[0], edge[1], color="#333333")

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -40,
          "springLength": 80
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 100}
      }
    }
    """)

    # Return HTML string
    html = net.generate_html()
    return html


# ─────────────────────────────────────────────
# COUNT SIR STATES
# ─────────────────────────────────────────────

def count_states(G):
    s = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "S")
    i = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "I")
    r = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "R")
    return s, i, r


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

def show():
    st.title("Network Visualizer")
    st.markdown("Generate a synthetic social network and simulate misinformation spread using the SIR model.")
    st.markdown("---")

    # ── Sidebar controls ──
    st.sidebar.markdown("### Network Settings")

    graph_type = st.sidebar.selectbox(
        "Graph Type",
        ["barabasi_albert", "erdos_renyi"],
        help="Barabási-Albert is more realistic (scale-free). Erdős-Rényi is random."
    )

    num_nodes = st.sidebar.slider(
        "Number of Nodes (People)",
        min_value = 50,
        max_value = 500,
        value     = 200,
        step      = 50
    )

    num_seeds = st.sidebar.slider(
        "Initial Infected Nodes",
        min_value = 1,
        max_value = 10,
        value     = 3
    )

    seed_strategy = st.sidebar.selectbox(
        "Seed Strategy",
        ["high_degree", "random"],
        help="High-degree seeds the most connected nodes. Random picks randomly."
    )

    spread_rate = st.sidebar.slider(
        "Spread Rate",
        min_value = 0.1,
        max_value = 0.9,
        value     = 0.4,
        step      = 0.1
    )

    recovery_time = st.sidebar.slider(
        "Recovery Time (steps)",
        min_value = 2,
        max_value = 10,
        value     = 4
    )

    # ── Generate network ──
    if st.button("Generate Network", use_container_width=True):
        st.session_state["G_base"] = create_graph(
            graph_type = graph_type,
            num_nodes  = num_nodes,
            seed       = 42
        )
        st.session_state["G"] = None
        st.session_state["history"] = []
        st.session_state["spread_done"] = False
        st.session_state["graph_type"]    = graph_type
        st.session_state["num_nodes"]     = num_nodes
        st.session_state["num_seeds"]     = num_seeds
        st.session_state["seed_strategy"] = seed_strategy
        st.session_state["spread_rate"]   = spread_rate
        st.session_state["recovery_time"] = recovery_time

    # ── Show network if generated ──
    if "G_base" in st.session_state and st.session_state["G_base"] is not None:

        G_display = (st.session_state["G"]
                     if st.session_state.get("G") is not None
                     else st.session_state["G_base"])

        # Network stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes (People)",    G_display.number_of_nodes())
        col2.metric("Edges (Connections)", G_display.number_of_edges())
        degrees = [d for _, d in G_display.degree()]
        col3.metric("Max Connections", max(degrees))

        # SIR counters
        s, i, r = count_states(G_display)
        total    = G_display.number_of_nodes()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔵 Susceptible", f"{s} ({round(s/total*100,1)}%)")
        c2.metric("🔴 Infected",    f"{i} ({round(i/total*100,1)}%)")
        c3.metric("🟢 Recovered",   f"{r} ({round(r/total*100,1)}%)")

        # Graph visualization
        st.markdown("**Interactive Network** — hover over nodes for details")
        html = build_pyvis_html(G_display)
        components.html(html, height=520, scrolling=False)

        st.caption("🔵 Susceptible  |  🔴 Infected  |  🟢 Recovered")

        # ── Simulate spread ──
        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Simulate Spread (Step by Step)",
                         use_container_width=True):

                from copy import deepcopy

                # Start fresh from base graph
                G = deepcopy(st.session_state["G_base"])
                G, seed_nodes = seed_infection(
                    G,
                    num_seeds = st.session_state["num_seeds"],
                    strategy  = st.session_state["seed_strategy"],
                    seed      = 42
                )

                st.info(f"Infection seeded at nodes: {seed_nodes}")

                history = []
                placeholder_graph   = st.empty()
                placeholder_metrics = st.empty()

                for step in range(60):

                    # ── Run SIR step FIRST, then display ──
                    # This ensures red nodes are visible before recovery
                    G, new_infected, new_recovered = sir_step(
                        G,
                        recovery_time = st.session_state["recovery_time"],
                        spread_rate   = st.session_state["spread_rate"]
                    )

                    s, i, r = count_states(G)
                    history.append({"step": step,
                                    "S": s, "I": i, "R": r})

                    # Update graph display
                    with placeholder_graph:
                        html = build_pyvis_html(G)
                        components.html(html, height=520, scrolling=False)

                    # Update metrics
                    with placeholder_metrics:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Step", step + 1)
                        m2.metric("🔵 Susceptible",
                                  f"{s} ({round(s/total*100,1)}%)")
                        m3.metric("🔴 Infected",
                                  f"{i} ({round(i/total*100,1)}%)")
                        m4.metric("🟢 Recovered",
                                  f"{r} ({round(r/total*100,1)}%)")

                    # Slow down so red nodes are visible
                    # Recovery time controls animation speed too
                    delay = max(0.6, st.session_state["recovery_time"] * 0.2)
                    time.sleep(delay)

                    if i == 0:
                        st.success(f"Spread stopped at step {step + 1}. "
                                   f"Misinformation reached "
                                   f"{round((r)/total*100,1)}% of network.")
                        break

                st.session_state["G"]       = G
                st.session_state["history"] = history
                st.session_state["spread_done"] = True

        with col_btn2:
            if st.button("Reset Network", use_container_width=True):
                st.session_state["G"] = None
                st.session_state["history"] = []
                st.session_state["spread_done"] = False
                st.rerun()

        # ── SIR curve after spread ──
        if st.session_state.get("spread_done") and st.session_state["history"]:
            st.markdown("---")
            st.markdown("### SIR Curve")

            import pandas as pd
            import altair as alt

            history = st.session_state["history"]
            df = pd.DataFrame(history)
            df_melt = df.melt(id_vars="step",
                              value_vars=["S", "I", "R"],
                              var_name="Status",
                              value_name="Count")

            color_scale = alt.Scale(
                domain = ["S", "I", "R"],
                range  = ["#4a90d9", "#e74c3c", "#2ecc71"]
            )

            chart = alt.Chart(df_melt).mark_line(
                strokeWidth=2.5
            ).encode(
                x     = alt.X("step:Q", title="Time Step"),
                y     = alt.Y("Count:Q", title="Number of Nodes"),
                color = alt.Color("Status:N", scale=color_scale),
            ).properties(
                height = 300,
                title  = "SIR Spread Over Time"
            )

            st.altair_chart(chart, use_container_width=True)
            