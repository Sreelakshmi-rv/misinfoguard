import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy

from graph_env import create_graph, print_graph_summary


# ─────────────────────────────────────────────
# SEED INITIAL INFECTION
# ─────────────────────────────────────────────

def seed_infection(G, num_seeds=3, strategy="high_degree", seed=42):
    """
    Sets the starting infected nodes — where misinformation enters the network.

    Args:
        G          : the graph
        num_seeds  : how many nodes start infected
        strategy   : "high_degree" (target influencers) or "random"
        seed       : for reproducibility

    Returns:
        G          : graph with seed nodes set to status "I"
        seed_nodes : list of nodes that were seeded
    """
    np.random.seed(seed)

    if strategy == "high_degree":
        # Sort nodes by number of connections, pick top ones
        sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        seed_nodes = [n for n, _ in sorted_nodes[:num_seeds]]

    elif strategy == "random":
        seed_nodes = list(np.random.choice(G.nodes(), num_seeds, replace=False))

    else:
        raise ValueError("strategy must be 'high_degree' or 'random'")

    for node in seed_nodes:
        G.nodes[node]["status"] = "I"
        G.nodes[node]["infected_time"] = 0  # track how long it's been infected

    return G, seed_nodes


# ─────────────────────────────────────────────
# ONE STEP OF SIR SPREAD
# ─────────────────────────────────────────────

def sir_step(G, recovery_time=4, spread_rate=0.4):
    """
    Runs one time step of the SIR model.

    Now uses edge weights in the spread formula:
    prob = spread_rate × edge_weight × receiver_susceptibility
           × credibility_modifier

    Higher edge weight = stronger connection = 
    misinformation spreads more easily along that edge.
    This models echo chambers — tightly bonded communities
    spread misinformation much faster internally.
    """
    new_infected  = []
    new_recovered = []
    nodes_to_infect  = []
    nodes_to_recover = []

    for node in G.nodes():
        status = G.nodes[node]["status"]

        if status == "I":
            G.nodes[node]["infected_time"] += 1

            # Stochastic recovery
            time_infected = G.nodes[node]["infected_time"]
            if time_infected >= recovery_time:
                recovery_prob = min(1.0, 0.4 + 0.2 * (
                    time_infected - recovery_time
                ))
                if np.random.random() < recovery_prob:
                    nodes_to_recover.append(node)
                    continue

            # Try to spread to each susceptible neighbor
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]["status"] == "S":

                    # Get edge weight between this node and neighbor
                    edge_weight = G[node][neighbor].get("weight", 1.0)

                    # Spread probability incorporates:
                    # - base spread rate
                    # - edge weight (connection strength / echo chamber)
                    # - receiver susceptibility
                    # - sender credibility modifier
                    sender_credibility      = G.nodes[node]["credibility"]
                    receiver_susceptibility = G.nodes[neighbor]["susceptibility"]

                    prob = (spread_rate
                            * edge_weight
                            * receiver_susceptibility
                            * (0.5 + 0.5 * sender_credibility))

                    # Clamp between 0.02 and 0.95
                    prob = max(0.02, min(0.95, prob))

                    if np.random.random() < prob:
                        nodes_to_infect.append(neighbor)

    # Apply changes
    for node in nodes_to_infect:
        if G.nodes[node]["status"] == "S":
            G.nodes[node]["status"]        = "I"
            G.nodes[node]["infected_time"] = 0
            new_infected.append(node)

    for node in nodes_to_recover:
        G.nodes[node]["status"] = "R"
        new_recovered.append(node)

    return G, new_infected, new_recovered

# ─────────────────────────────────────────────
# FULL SIMULATION RUN
# ─────────────────────────────────────────────

def run_simulation(G, num_steps=60, recovery_time=4,
                   spread_rate=0.3, num_seeds=5,
                   seed_strategy="high_degree", seed=42):
    """
    Runs the full SIR simulation from start to finish.

    Returns:
        history : list of dicts, one per time step
                  each dict has counts of S, I, R nodes
                  and which nodes changed state
    """

    # Seed the infection
    G, seed_nodes = seed_infection(G,
                                   num_seeds=num_seeds,
                                   strategy=seed_strategy,
                                   seed=seed)

    print(f"\nInfection seeded at nodes: {seed_nodes}")

    history = []

    for step in range(num_steps):

        # Count current states
        statuses = [G.nodes[n]["status"] for n in G.nodes()]
        s_count = statuses.count("S")
        i_count = statuses.count("I")
        r_count = statuses.count("R")

        # Record this step
        history.append({
            "step"     : step,
            "S"        : s_count,
            "I"        : i_count,
            "R"        : r_count,
            "total"    : G.number_of_nodes()
        })

        # Print progress every 5 steps
        if step % 5 == 0:
            pct = round((i_count / G.number_of_nodes()) * 100, 1)
            print(f"  Step {step:>3} | S: {s_count:>4} | "
                  f"I: {i_count:>4} | R: {r_count:>4} | "
                  f"Infected: {pct}%")

        # Run one step
        G, new_infected, new_recovered = sir_step(
            G,
            recovery_time=recovery_time,
            spread_rate=spread_rate
        )

        # Stop early if no infected nodes remain
        if i_count == 0 and step > 0:
            print(f"\n  Spread stopped at step {step} — no active spreaders.")
            break

    return history, G


# ─────────────────────────────────────────────
# PLOT SIR CURVE
# ─────────────────────────────────────────────

def plot_sir_curve(history, title="SIR Spread Over Time"):
    """
    Plots how S, I, R counts change over time.
    This is the classic epidemic curve.
    """
    steps = [h["step"]   for h in history]
    s     = [h["S"]      for h in history]
    i     = [h["I"]      for h in history]
    r     = [h["R"]      for h in history]
    total = history[0]["total"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ── Top plot: raw counts ──
    ax1.plot(steps, s, color="steelblue",  linewidth=2, label="Susceptible (S)")
    ax1.plot(steps, i, color="crimson",    linewidth=2, label="Infected (I)")
    ax1.plot(steps, r, color="seagreen",   linewidth=2, label="Recovered (R)")
    ax1.set_ylabel("Number of Nodes")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Bottom plot: percentage ──
    ax2.fill_between(steps, [x/total*100 for x in s],
                     color="steelblue", alpha=0.4, label="Susceptible %")
    ax2.fill_between(steps, [x/total*100 for x in i],
                     color="crimson",   alpha=0.4, label="Infected %")
    ax2.fill_between(steps, [x/total*100 for x in r],
                     color="seagreen",  alpha=0.4, label="Recovered %")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Percentage of Network")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sir_curve.png", dpi=150)
    plt.show()
    print("SIR curve saved as sir_curve.png")


# ─────────────────────────────────────────────
# PLOT NETWORK STATE AT A GIVEN STEP
# ─────────────────────────────────────────────

def plot_network_state(G, step_number, pos=None):
    """
    Draws the network with nodes colored by SIR status.
    Blue = S, Red = I, Green = R
    """
    color_map = {"S": "steelblue", "I": "crimson", "R": "seagreen"}
    colors    = [color_map[G.nodes[n]["status"]] for n in G.nodes()]
    sizes     = [G.nodes[n]["influence"] * 2000 + 50 for n in G.nodes()]

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors,
                           node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="gray")

    # Legend
    patches = [
        mpatches.Patch(color="steelblue", label="Susceptible (S)"),
        mpatches.Patch(color="crimson",   label="Infected (I)"),
        mpatches.Patch(color="seagreen",  label="Recovered (R)"),
    ]
    plt.legend(handles=patches, loc="upper right")
    plt.title(f"Network State at Step {step_number}", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"network_state_step{step_number}.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# SEEDING EXPERIMENT
# ─────────────────────────────────────────────

def run_seeding_experiment(num_nodes=200):
    """
    Compares 4 seeding scenarios:
    - 1 seed random vs high-degree
    - 5 seeds random vs high-degree

    Shows how starting point and number of seeds
    affects how far misinformation spreads.
    """

    scenarios = [
        {"num_seeds": 1, "strategy": "random",       "label": "1 seed — Random",      "color": "steelblue",  "ls": "--"},
        {"num_seeds": 1, "strategy": "high_degree",  "label": "1 seed — High-Degree",  "color": "crimson",    "ls": "--"},
        {"num_seeds": 5, "strategy": "random",       "label": "5 seeds — Random",      "color": "steelblue",  "ls": "-"},
        {"num_seeds": 5, "strategy": "high_degree",  "label": "5 seeds — High-Degree", "color": "crimson",    "ls": "-"},
    ]

    plt.figure(figsize=(12, 5))
    
    print("\n── SEEDING EXPERIMENT RESULTS ──")
    print(f"  {'Scenario':<30} {'Peak Infected':>15} {'Total Reached':>15}")
    print("  " + "-" * 62)

    for s in scenarios:
        # Fresh graph for each scenario
        G = create_graph(graph_type="barabasi_albert",
                         num_nodes=num_nodes, seed=42)

        history, _ = run_simulation(
            G,
            num_steps     = 40,
            recovery_time = 4,
            spread_rate   = 0.4,
            num_seeds     = s["num_seeds"],
            seed_strategy = s["strategy"],
            seed          = 42
        )

        infected_over_time = [h["I"] for h in history]
        total_reached      = history[-1]["I"] + history[-1]["R"]
        peak_infected      = max(infected_over_time)
        steps              = [h["step"] for h in history]

        print(f"  {s['label']:<30} {peak_infected:>15} "
              f"{total_reached:>15} ({round(total_reached/num_nodes*100,1)}%)")

        plt.plot(steps, infected_over_time,
                 label=s["label"],
                 color=s["color"],
                 linestyle=s["ls"],
                 linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Number of Infected Nodes")
    plt.title("Seeding Experiment: How Starting Conditions Affect Spread")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("seeding_experiment.png", dpi=150)
    plt.show()
    print("\n  Plot saved as seeding_experiment.png")

# ─────────────────────────────────────────────
# SEIR MODEL — COMPARISON EXPERIMENT
# ─────────────────────────────────────────────

def run_seir_comparison(G_input, num_steps=40, spread_rate=0.4,
                        recovery_time=4, exposure_time=2,
                        num_seeds=3, seed=42):
    """
    Runs SEIR model alongside SIR for comparison.

    SEIR adds an Exposed state:
        S → E → I → R

    E (Exposed): The node has received the misinformation
    but hasn't started sharing yet. There's a delay
    between seeing content and deciding to spread it.
    This is realistic — people don't immediately share
    everything they see.

    exposure_time: how many steps a node stays Exposed
    before becoming Infected (actively spreading)

    Returns history for both SIR and SEIR for comparison.
    """
    from copy import deepcopy

    # ── Run standard SIR ──
    G_sir = deepcopy(G_input)
    G_sir, _ = seed_infection(G_sir, num_seeds=num_seeds,
                              strategy="high_degree", seed=seed)

    sir_history = []
    for step in range(num_steps):
        statuses = [G_sir.nodes[n]["status"] for n in G_sir.nodes()]
        sir_history.append({
            "step": step,
            "S": statuses.count("S"),
            "I": statuses.count("I"),
            "R": statuses.count("R")
        })
        if statuses.count("I") == 0 and step > 0:
            break
        G_sir, _, _ = sir_step(G_sir,
                                recovery_time=recovery_time,
                                spread_rate=spread_rate)

    # ── Run SEIR ──
    G_seir = deepcopy(G_input)

    # Seed infection — start as E not I for SEIR
    sorted_nodes = sorted(G_seir.degree(),
                          key=lambda x: x[1], reverse=True)
    seed_nodes   = [n for n, _ in sorted_nodes[:num_seeds]]
    for node in seed_nodes:
        G_seir.nodes[node]["status"]        = "E"
        G_seir.nodes[node]["exposed_time"]  = 0
        G_seir.nodes[node]["infected_time"] = 0

    seir_history = []

    for step in range(num_steps):
        statuses = [G_seir.nodes[n]["status"] for n in G_seir.nodes()]
        seir_history.append({
            "step": step,
            "S": statuses.count("S"),
            "E": statuses.count("E"),
            "I": statuses.count("I"),
            "R": statuses.count("R")
        })

        if statuses.count("I") == 0 and statuses.count("E") == 0 and step > 0:
            break

        nodes_to_expose   = []
        nodes_to_infect   = []
        nodes_to_recover  = []

        for node in G_seir.nodes():
            status = G_seir.nodes[node]["status"]

            # E → I transition
            if status == "E":
                G_seir.nodes[node]["exposed_time"] += 1
                if G_seir.nodes[node]["exposed_time"] >= exposure_time:
                    nodes_to_infect.append(node)

            # I → spread and recover
            elif status == "I":
                G_seir.nodes[node]["infected_time"] += 1

                # Recovery check
                time_inf = G_seir.nodes[node]["infected_time"]
                if time_inf >= recovery_time:
                    recovery_prob = min(1.0, 0.4 + 0.2 * (
                        time_inf - recovery_time
                    ))
                    if np.random.random() < recovery_prob:
                        nodes_to_recover.append(node)
                        continue

                # Spread to S neighbors
                for neighbor in G_seir.neighbors(node):
                    if G_seir.nodes[neighbor]["status"] == "S":
                        sender_cred = G_seir.nodes[node]["credibility"]
                        recv_sus    = G_seir.nodes[neighbor]["susceptibility"]
                        prob = (spread_rate * recv_sus
                                * (0.5 + 0.5 * sender_cred))
                        prob = max(0.05, min(0.95, prob))
                        if np.random.random() < prob:
                            nodes_to_expose.append(neighbor)

        # Apply transitions
        for node in nodes_to_expose:
            if G_seir.nodes[node]["status"] == "S":
                G_seir.nodes[node]["status"]       = "E"
                G_seir.nodes[node]["exposed_time"] = 0
        for node in nodes_to_infect:
            G_seir.nodes[node]["status"]        = "I"
            G_seir.nodes[node]["infected_time"] = 0
        for node in nodes_to_recover:
            G_seir.nodes[node]["status"] = "R"

    return sir_history, seir_history


def plot_seir_comparison(sir_history, seir_history, total_nodes=200):
    """
    Plots SIR vs SEIR spread curves side by side.
    Shows the effect of adding the Exposed delay state.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── SIR plot ──
    steps_sir = [h["step"] for h in sir_history]
    ax1.plot(steps_sir, [h["S"]/total_nodes*100 for h in sir_history],
             color="steelblue", linewidth=2, label="Susceptible (S)")
    ax1.plot(steps_sir, [h["I"]/total_nodes*100 for h in sir_history],
             color="crimson",   linewidth=2, label="Infected (I)")
    ax1.plot(steps_sir, [h["R"]/total_nodes*100 for h in sir_history],
             color="seagreen",  linewidth=2, label="Recovered (R)")
    ax1.set_title("SIR Model\n(Immediate spread after exposure)",
                  fontweight="bold")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("% of Network")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)

    # Peak annotation
    peak_i   = max(h["I"] for h in sir_history)
    peak_pct = round(peak_i / total_nodes * 100, 1)
    ax1.annotate(f"Peak: {peak_pct}%",
                 xy=(steps_sir[
                     [h["I"] for h in sir_history].index(peak_i)
                 ], peak_pct),
                 xytext=(10, peak_pct + 5),
                 fontsize=9, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson"))

    # ── SEIR plot ──
    steps_seir = [h["step"] for h in seir_history]
    ax2.plot(steps_seir, [h["S"]/total_nodes*100 for h in seir_history],
             color="steelblue",  linewidth=2, label="Susceptible (S)")
    ax2.plot(steps_seir, [h["E"]/total_nodes*100 for h in seir_history],
             color="orange",     linewidth=2, label="Exposed (E)")
    ax2.plot(steps_seir, [h["I"]/total_nodes*100 for h in seir_history],
             color="crimson",    linewidth=2, label="Infected (I)")
    ax2.plot(steps_seir, [h["R"]/total_nodes*100 for h in seir_history],
             color="seagreen",   linewidth=2, label="Recovered (R)")
    ax2.set_title("SEIR Model\n(Delay between exposure and spreading)",
                  fontweight="bold")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("% of Network")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    # Peak annotation
    peak_i_seir   = max(h["I"] for h in seir_history)
    peak_pct_seir = round(peak_i_seir / total_nodes * 100, 1)
    ax2.annotate(f"Peak: {peak_pct_seir}%",
                 xy=(steps_seir[
                     [h["I"] for h in seir_history].index(peak_i_seir)
                 ], peak_pct_seir),
                 xytext=(10, peak_pct_seir + 5),
                 fontsize=9, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson"))

    plt.suptitle("SIR vs SEIR: Effect of Exposure Delay on Spread",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sir_vs_seir.png", dpi=150)
    plt.show()

    # Print comparison
    sir_reached  = sir_history[-1]["R"]  + sir_history[-1]["I"]
    seir_reached = seir_history[-1]["R"] + seir_history[-1]["I"]

    print("\n── SIR vs SEIR COMPARISON ──")
    print(f"  SIR  — Total reached: "
          f"{round(sir_reached/total_nodes*100,1)}% | "
          f"Peak: {peak_pct}%")
    print(f"  SEIR — Total reached: "
          f"{round(seir_reached/total_nodes*100,1)}% | "
          f"Peak: {peak_pct_seir}%")
    print(f"\n  The exposure delay in SEIR slows the peak "
          f"and shifts it later in time.")
    print("  Saved as sir_vs_seir.png")

# ─────────────────────────────────────────────
# RUN THIS FILE DIRECTLY TO TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Build graph
    print("Building graph...")
    G = create_graph(graph_type="barabasi_albert", num_nodes=500)
    print_graph_summary(G)

    # Fix layout so all plots use same node positions
    pos = nx.spring_layout(G, seed=42)

    # Show network before spread
    print("\nNetwork state BEFORE spread:")
    plot_network_state(G, step_number=0, pos=pos)

    # Run simulation
    print("\nRunning SIR simulation...")
    history, G_final = run_simulation(
        G,
        num_steps      = 60,
        recovery_time  = 4,
        spread_rate    = 0.3,
        num_seeds      = 5,
        seed_strategy  = "high_degree"
    )

    # Show network after spread
    print("\nNetwork state AFTER spread:")
    plot_network_state(G_final, step_number=len(history), pos=pos)

    # Plot SIR curve
    plot_sir_curve(history)

    # Print final summary
    final = history[-1]
    total = final["total"]
    print("\n── FINAL RESULTS ──")
    print(f"  Total nodes       : {total}")
    print(f"  Never infected (S): {final['S']} "
          f"({round(final['S']/total*100, 1)}%)")
    print(f"  Still infected (I): {final['I']} "
          f"({round(final['I']/total*100, 1)}%)")
    print(f"  Recovered (R)     : {final['R']} "
          f"({round(final['R']/total*100, 1)}%)")
    print(f"\n  >> Misinformation reached "
          f"{round((final['I']+final['R'])/total*100, 1)}% of the network")
    
    # Run seeding experiment
    print("\nRunning seeding experiment...")
    run_seeding_experiment(num_nodes=200)

    # SEIR comparison
    print("\nRunning SIR vs SEIR comparison...")
    G_fresh = create_graph("barabasi_albert", num_nodes=200, seed=42)
    sir_hist, seir_hist = run_seir_comparison(G_fresh)
    plot_seir_comparison(sir_hist, seir_hist)