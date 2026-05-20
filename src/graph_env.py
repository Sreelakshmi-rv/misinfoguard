import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import os

# ─────────────────────────────────────────────
# NETWORK TYPE REGISTRY
# ─────────────────────────────────────────────

NETWORK_CONFIGS = {
    "barabasi_albert": {
        "label"      : "Barabási-Albert (Scale-Free)",
        "short"      : "BA",
        "color"      : "#3498db",
        "description": "Scale-free network with power-law degree distribution. "
                       "Hub nodes dominate — models Twitter/Instagram.",
    },
    "watts_strogatz": {
        "label"      : "Watts-Strogatz (Small-World)",
        "short"      : "WS",
        "color"      : "#e67e22",
        "description": "High clustering with short path lengths. "
                       "Models tight community groups — WhatsApp, offline networks.",
    },
    "erdos_renyi": {
        "label"      : "Erdős-Rényi (Random)",
        "short"      : "ER",
        "color"      : "#9b59b6",
        "description": "Uniform random connections. Theoretical baseline — "
                       "no hub structure, no community clustering.",
    },
    "holme_kim": {
        "label"      : "Holme-Kim (Scale-Free + Clustering)",
        "short"      : "HK",
        "color"      : "#e74c3c",
        "description": "Scale-free with added triangle formation. "
                       "Closest to real social networks — Facebook-like.",
    },
}


# ─────────────────────────────────────────────
# GRAPH GENERATION
# ─────────────────────────────────────────────

def create_graph(graph_type="barabasi_albert", num_nodes=500, seed=42):
    """
    Creates a synthetic social network graph.

    Args:
        graph_type : one of "barabasi_albert", "watts_strogatz",
                     "erdos_renyi", "holme_kim"
        num_nodes  : number of nodes
        seed       : random seed

    Returns:
        G : NetworkX graph with node properties and edge weights
    """
    if graph_type == "barabasi_albert":
        G = nx.barabasi_albert_graph(n=num_nodes, m=2, seed=seed)

    elif graph_type == "watts_strogatz":
        # k=6: each node connected to 6 nearest neighbours
        # p=0.1: 10% rewiring probability — creates small-world shortcuts
        G = nx.watts_strogatz_graph(n=num_nodes, k=6, p=0.3, seed=seed)

    elif graph_type == "erdos_renyi":
        # p=0.008: sparse random graph, avg degree ~4 matching BA
        G = nx.erdos_renyi_graph(n=num_nodes, p=0.008, seed=seed)
        # Ensure connectivity
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(1, len(components)):
                u = list(components[0])[0]
                v = list(components[i])[0]
                G.add_edge(u, v)

    elif graph_type == "holme_kim":
        # m=2: edges per new node (same as BA)
        # p=0.5: probability of adding triangle after each edge
        G = nx.powerlaw_cluster_graph(n=num_nodes, m=2, p=0.5, seed=seed)

    else:
        raise ValueError(
            f"graph_type must be one of: {list(NETWORK_CONFIGS.keys())}"
        )

    G = assign_node_properties(G, seed=seed)
    G = assign_edge_weights(G, seed=seed)

    return G


# ─────────────────────────────────────────────
# NODE PROPERTY ASSIGNMENT
# ─────────────────────────────────────────────

def assign_node_properties(G, seed=42):
    np.random.seed(seed)
    degree_centrality = nx.degree_centrality(G)

    for node in G.nodes():
        G.nodes[node]["influence"]      = round(degree_centrality[node], 4)
        G.nodes[node]["susceptibility"] = round(np.random.uniform(0.1, 0.9), 4)
        G.nodes[node]["credibility"]    = round(np.random.uniform(0.1, 1.0), 4)
        G.nodes[node]["status"]         = "S"

    return G


# ─────────────────────────────────────────────
# EDGE WEIGHT ASSIGNMENT
# ─────────────────────────────────────────────

def assign_edge_weights(G, seed=42):
    """
    Assigns echo-chamber weights to edges.
    Hub-to-hub edges get high weights (strong echo chamber bond).
    Peripheral edges get low weights (weak bridging connections).
    Works for all network types — adapts to each graph's degree distribution.
    """
    np.random.seed(seed + 1000)

    degrees    = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for u, v in G.edges():
        if max_degree > 0:
            avg_norm = ((degrees[u] + degrees[v]) / 2) / max_degree
        else:
            avg_norm = 0.0

        if avg_norm >= 0.15:
            weight = np.random.uniform(1.5, 2.5)
        elif avg_norm >= 0.05:
            weight = np.random.uniform(0.8, 1.5)
        else:
            weight = np.random.uniform(0.3, 0.8)

        G[u][v]["weight"] = round(weight, 3)

    return G


# ─────────────────────────────────────────────
# GRAPH SUMMARY
# ─────────────────────────────────────────────

def print_graph_summary(G, graph_type="barabasi_albert"):
    degrees = [d for _, d in G.degree()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    config  = NETWORK_CONFIGS.get(graph_type, {})

    print("=" * 50)
    print(f"  {config.get('label', graph_type)}")
    print("=" * 50)
    print(f"  Nodes           : {G.number_of_nodes()}")
    print(f"  Edges           : {G.number_of_edges()}")
    print(f"  Avg degree      : {round(sum(degrees)/len(degrees), 2)}")
    print(f"  Max degree      : {max(degrees)}")
    print(f"  Min degree      : {min(degrees)}")
    print(f"  Avg edge weight : {round(sum(weights)/len(weights), 3)}")
    print(f"  High-w edges    : "
          f"{sum(1 for w in weights if w > 1.5)} "
          f"({round(sum(1 for w in weights if w > 1.5)/len(weights)*100, 1)}%)")
    print("=" * 50)


# ─────────────────────────────────────────────
# DEGREE DISTRIBUTION — POWER LAW VALIDATION
# ─────────────────────────────────────────────

def plot_degree_distribution(G, title="Degree Distribution"):
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    x = list(degree_counts.keys())
    y = list(degree_counts.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x, y, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Number of Nodes")
    ax1.set_title("Degree Distribution (Linear Scale)")
    ax1.grid(alpha=0.3)

    ax2.scatter(x, y, color="crimson", alpha=0.7, s=30)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Degree (log scale)")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title("Log-Log Scale\nStraight line = Power Law")
    ax2.grid(alpha=0.3, which="both")

    if len(x) > 1:
        log_x  = np.log(np.array(x, dtype=float))
        log_y  = np.log(np.array(y, dtype=float))
        coeffs = np.polyfit(log_x, log_y, 1)
        slope  = round(coeffs[0], 2)
        fit    = np.poly1d(coeffs)
        x_fit  = np.linspace(min(log_x), max(log_x), 100)
        ax2.plot(np.exp(x_fit), np.exp(fit(x_fit)),
                 color="gold", linewidth=2, linestyle="--",
                 label=f"Power law fit (slope={slope})")
        ax2.legend()

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("degree_distribution.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MULTI-NETWORK DEGREE DISTRIBUTION COMPARISON
# ─────────────────────────────────────────────

def plot_all_degree_distributions(num_nodes=500, seed=42):
    """
    Plots degree distributions for all 4 network types side by side.
    Used in the report methodology chapter to justify network choice.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (graph_type, config) in zip(axes, NETWORK_CONFIGS.items()):
        G       = create_graph(graph_type, num_nodes=num_nodes, seed=seed)
        degrees = [d for _, d in G.degree()]
        deg_counts = {}
        for d in degrees:
            deg_counts[d] = deg_counts.get(d, 0) + 1

        x = list(deg_counts.keys())
        y = list(deg_counts.values())

        ax.scatter(x, y, color=config["color"], alpha=0.8, s=40)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (log)", fontsize=9)
        ax.set_ylabel("Frequency (log)", fontsize=9)
        ax.set_title(
            f"{config['label']}\n"
            f"Nodes: {G.number_of_nodes()} | "
            f"Edges: {G.number_of_edges()} | "
            f"Max degree: {max(degrees)}",
            fontsize=10, fontweight="bold"
        )
        ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "Degree Distributions — All 4 Network Types (500 nodes, log-log scale)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/all_degree_distributions.png", dpi=150)
    plt.show()
    print("Saved → models/all_degree_distributions.png")


# ─────────────────────────────────────────────
# PYVIS VISUALIZATION
# ─────────────────────────────────────────────

def visualize_pyvis(G, filename="network.html"):
    net = Network(height="700px", width="100%",
                  bgcolor="#1a1a2e", font_color="white")

    for node in G.nodes():
        inf   = G.nodes[node]["influence"]
        sus   = G.nodes[node]["susceptibility"]
        red   = int(sus * 255)
        green = int((1 - sus) * 255)
        color = f"rgb({red},{green},80)"
        size  = 5 + inf * 40

        net.add_node(node, label=str(node), color=color, size=size,
                     title=f"Node {node}\n"
                           f"Influence: {inf:.3f}\n"
                           f"Susceptibility: {sus:.3f}")

    for edge in G.edges():
        w = G[edge[0]][edge[1]]["weight"]
        net.add_edge(edge[0], edge[1], color="#444444", width=w * 1.5)

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 100
        },
        "solver": "forceAtlas2Based"
      }
    }
    """)
    net.write_html(filename)
    print(f"Interactive graph saved as {filename}")


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for graph_type in NETWORK_CONFIGS:
        print(f"\nTesting {graph_type}...")
        G = create_graph(graph_type, num_nodes=500, seed=42)
        print_graph_summary(G, graph_type)

    print("\nGenerating degree distribution comparison...")
    plot_all_degree_distributions()