import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_env import create_graph
from sir_model  import seed_infection, sir_step

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NUM_NODES         = 500
NUM_REGIONS       = 2
NUM_SEEDS         = 5
MESSAGE_SIZE      = 8
RECEIVED_MSG_SIZE = MESSAGE_SIZE
STATE_SIZE        = 20   # 12 local features + 8 message values


# ─────────────────────────────────────────────
# COMMUNITY DETECTION — LOUVAIN REGION SPLIT
# ─────────────────────────────────────────────

def get_regions(G, num_regions=2, seed=42):
    import community as community_louvain
    partition       = community_louvain.best_partition(G, random_state=seed)
    community_ids   = set(partition.values())
    raw_communities = {cid: [] for cid in community_ids}
    for node, cid in partition.items():
        raw_communities[cid].append(node)

    sorted_communities = sorted(
        raw_communities.values(), key=len, reverse=True
    )

    if len(sorted_communities) > num_regions:
        regions = sorted_communities[:num_regions]
        for extra in sorted_communities[num_regions:]:
            smallest_idx = min(range(num_regions),
                               key=lambda i: len(regions[i]))
            regions[smallest_idx].extend(extra)
    else:
        regions = sorted_communities

    print(f"\n  Louvain detected {len(sorted_communities)} communities")
    print(f"  Network split into {num_regions} regions:")
    for i, r in enumerate(regions):
        print(f"    Region {i}: {len(r)} nodes")

    return regions


# ─────────────────────────────────────────────
# PARTIAL OBSERVABILITY — VISIBLE NODES
# ─────────────────────────────────────────────

def get_visible_nodes(G, region_nodes):
    """
    Infection frontier: infected nodes + their in-region neighbors.
    """
    region_set = set(region_nodes)
    visible    = set()

    for node in region_nodes:
        if G.nodes[node]["status"] == "I":
            visible.add(node)
            for nb in G.neighbors(node):
                if nb in region_set:
                    visible.add(nb)

    if len(visible) == 0:
        top_nodes = sorted(region_nodes,
                           key=lambda n: G.degree(n),
                           reverse=True)[:10]
        visible.update(top_nodes)

    return visible


# ─────────────────────────────────────────────
# COMPACT STATE VECTOR — 12 + 8 = 20 FEATURES
# ─────────────────────────────────────────────

def build_state(G, region_nodes, all_regions=None,
                received_messages=None):
    """
    20-feature compact state. Replaces the old 2500-dim per-node vector.

    DQN with MLP cannot extract signal from 2500 mostly-zero inputs.
    20 meaningful aggregates encode the same situation more efficiently.

    Local features (12):
        0.  Region infection rate
        1.  Visible infection rate (frontier only)
        2.  Hub infection rate (top-20% degree nodes)
        3.  Boundary pressure (infected nodes with cross-region edges)
        4.  Susceptible rate remaining
        5.  High-susceptibility node rate (sus > 0.6)
        6.  Max risk score among visible infected (normalized)
        7.  Avg risk score among visible infected (normalized)
        8.  Max influence among visible infected
        9.  Avg edge weight of visible infected (echo chamber signal)
        10. Avg susceptibility of S-neighbors of infected
        11. Best counter-message target score (normalized)

    Message from other agent (8):
        12-19. The 8-value message vector from the other agent
    """
    region_set  = set(region_nodes)
    region_size = len(region_nodes)

    visible      = get_visible_nodes(G, region_nodes)
    infected     = [n for n in region_nodes if G.nodes[n]["status"] == "I"]
    susceptible  = [n for n in region_nodes if G.nodes[n]["status"] == "S"]
    vis_infected = [n for n in visible      if G.nodes[n]["status"] == "I"]

    # Feature 0: Region infection rate
    infection_rate = len(infected) / region_size

    # Feature 1: Visible infection rate
    vis_inf_rate = len(vis_infected) / max(1, len(visible))

    # Feature 2: Hub infection rate
    sorted_by_deg = sorted(region_nodes, key=lambda n: G.degree(n),
                           reverse=True)
    top_hubs      = sorted_by_deg[:max(1, region_size // 5)]
    hub_inf_rate  = (sum(1 for n in top_hubs
                         if G.nodes[n]["status"] == "I")
                     / len(top_hubs))

    # Feature 3: Boundary pressure
    boundary = (
        sum(1 for n in infected
            if any(nb not in region_set for nb in G.neighbors(n)))
        / max(1, len(infected))
    ) if infected else 0.0

    # Feature 4: Susceptible rate
    susceptible_rate = len(susceptible) / region_size

    # Feature 5: High-susceptibility node rate
    high_sus_rate = (
        sum(1 for n in susceptible if G.nodes[n]["susceptibility"] > 0.6)
        / max(1, region_size)
    )

    # Features 6-7: Risk scores
    def risk_score(node):
        return sum(
            G.nodes[nb]["susceptibility"]
            * G[node][nb].get("weight", 1.0)
            * G.nodes[node]["influence"]
            for nb in G.neighbors(node)
            if G.nodes[nb]["status"] == "S"
        )

    if vis_infected:
        risks      = [risk_score(n) for n in vis_infected]
        max_risk_n = min(1.0, max(risks) / 10.0)
        avg_risk_n = min(1.0, np.mean(risks) / 5.0)
    else:
        max_risk_n = avg_risk_n = 0.0

    # Feature 8: Max influence of visible infected
    max_influence = (max(G.nodes[n]["influence"] for n in vis_infected)
                     if vis_infected else 0.0)

    # Feature 9: Avg edge weight of visible infected
    if vis_infected:
        avg_ew = np.mean([
            np.mean([G[n][nb].get("weight", 1.0) for nb in G.neighbors(n)])
            for n in vis_infected
        ])
        avg_ew_n = min(1.0, avg_ew / 2.5)
    else:
        avg_ew_n = 0.0

    # Feature 10: Avg susceptibility of S-neighbors of infected
    sus_nb_vals = [
        G.nodes[nb]["susceptibility"]
        for n in vis_infected
        for nb in G.neighbors(n)
        if nb in region_set and G.nodes[nb]["status"] == "S"
    ]
    avg_sus_targets = np.mean(sus_nb_vals) if sus_nb_vals else 0.0

    # Feature 11: Best counter-message score
    def counter_score(node):
        return sum(
            G.nodes[nb]["susceptibility"]
            for nb in G.neighbors(node)
            if G.nodes[nb]["status"] == "S" and nb in region_set
        )
    best_counter = (max(counter_score(n) for n in vis_infected)
                    if vis_infected else 0.0)
    best_counter_n = min(1.0, best_counter / 20.0)

    local = np.array([
        infection_rate,   # 0
        vis_inf_rate,     # 1
        hub_inf_rate,     # 2
        boundary,         # 3
        susceptible_rate, # 4
        high_sus_rate,    # 5
        max_risk_n,       # 6
        avg_risk_n,       # 7
        max_influence,    # 8
        avg_ew_n,         # 9
        avg_sus_targets,  # 10
        best_counter_n,   # 11
    ], dtype=np.float32)

    # Message from other agent
    if received_messages is not None:
        msg = received_messages[0] if isinstance(received_messages, list) \
              else received_messages
    else:
        msg = np.zeros(MESSAGE_SIZE, dtype=np.float32)

    return np.concatenate([local, msg]).astype(np.float32)


# ─────────────────────────────────────────────
# MESSAGE GENERATION
# ─────────────────────────────────────────────

def generate_message(G, region_nodes):
    """8-value summary of this region for the other agent."""
    region_set  = set(region_nodes)
    region_size = len(region_nodes)
    infected    = [n for n in region_nodes if G.nodes[n]["status"] == "I"]
    susceptible = [n for n in region_nodes if G.nodes[n]["status"] == "S"]

    infection_rate = len(infected) / region_size

    sorted_by_deg = sorted(region_nodes, key=lambda n: G.degree(n),
                            reverse=True)
    top_hubs      = sorted_by_deg[:max(1, region_size // 5)]
    hub_infection = (sum(1 for n in top_hubs if G.nodes[n]["status"] == "I")
                     / len(top_hubs))

    boundary_rate = (
        sum(1 for n in infected
            if any(nb not in region_set for nb in G.neighbors(n)))
        / max(1, len(infected))
    ) if infected else 0.0

    if infected:
        avg_ew = np.mean([
            np.mean([G[n][nb].get("weight", 1.0) for nb in G.neighbors(n)])
            for n in infected
        ])
        avg_infected_weight = min(1.0, avg_ew / 2.5)
    else:
        avg_infected_weight = 0.0

    susceptible_rate       = len(susceptible) / region_size
    max_infected_influence = (max(G.nodes[n]["influence"] for n in infected)
                               if infected else 0.0)

    sus_nb = [
        G.nodes[nb]["susceptibility"]
        for n in infected
        for nb in G.neighbors(n)
        if nb in region_set and G.nodes[nb]["status"] == "S"
    ]
    avg_sus_of_targets = np.mean(sus_nb) if sus_nb else 0.0
    momentum           = min(1.0, infection_rate * 3.0)

    return np.array([
        infection_rate, hub_infection, boundary_rate,
        avg_infected_weight, susceptible_rate,
        max_infected_influence, avg_sus_of_targets, momentum,
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# APPLY ACTION — DIFFERENTIATED TARGETING
# ─────────────────────────────────────────────

def apply_action(G, action, region_nodes):
    """
    Each action targets a DIFFERENT node using DIFFERENT criteria.
    The agent must learn which action fits the current situation.

    Action 0: Do nothing
    Action 1: FLAG — target max-influence infected node (reduce influence 60%)
    Action 2: COUNTER-MESSAGE — target node with most susceptible neighbors
              (reduce their susceptibility 40%)
    Action 3: QUARANTINE — target highest risk-score infected node
              (risk = influence × edge_weight × neighbor susceptibility)

    Greedy only ever does action 3 on highest-DEGREE node.
    RL learns when action 1 or 2 is better — this is the edge.
    """
    if action == 0:
        return G, False

    visible      = get_visible_nodes(G, region_nodes)
    vis_infected = [n for n in visible if G.nodes[n]["status"] == "I"]

    if not vis_infected:
        return G, False

    if action == 1:
        target = max(vis_infected, key=lambda n: G.nodes[n]["influence"])
        G.nodes[target]["influence"] *= 0.4
        return G, True

    elif action == 2:
        region_set = set(region_nodes)
        def counter_score(node):
            return sum(
                G.nodes[nb]["susceptibility"]
                for nb in G.neighbors(node)
                if G.nodes[nb]["status"] == "S" and nb in region_set
            )
        target = max(vis_infected, key=counter_score)
        for nb in G.neighbors(target):
            if G.nodes[nb]["status"] == "S":
                G.nodes[nb]["susceptibility"] = max(
                    0.05, G.nodes[nb]["susceptibility"] * 0.6
                )
        return G, True

    elif action == 3:
        def risk_score(node):
            return sum(
                G.nodes[nb]["susceptibility"]
                * G[node][nb].get("weight", 1.0)
                * G.nodes[node]["influence"]
                for nb in G.neighbors(node)
                if G.nodes[nb]["status"] == "S"
            )
        target = max(vis_infected, key=risk_score)
        G.nodes[target]["status"]        = "R"
        G.nodes[target]["infected_time"] = 999
        return G, True

    return G, False


# ─────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class MisinfoEnv(gym.Env):
    """
    Multi-agent misinformation control environment.
    State: 20-feature compact vector (was 2500-dim).
    Actions: 4 types, each targeting a different node.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 graph_type    = "barabasi_albert",
                 num_nodes     = NUM_NODES,
                 num_regions   = NUM_REGIONS,
                 agent_id      = 0,
                 recovery_time = 4,
                 spread_rate   = 0.3,
                 max_steps     = 60):

        super().__init__()
        self.graph_type    = graph_type
        self.num_nodes     = num_nodes
        self.num_regions   = num_regions
        self.agent_id      = agent_id
        self.recovery_time = recovery_time
        self.spread_rate   = spread_rate
        self.max_steps     = max_steps

        base_G       = create_graph(graph_type=graph_type,
                                    num_nodes=num_nodes, seed=42)
        self.regions   = get_regions(base_G, num_regions=num_regions)
        self.my_region = self.regions[agent_id]

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.action_space      = spaces.Discrete(4)

        self.received_messages = [np.zeros(MESSAGE_SIZE, dtype=np.float32)]
        self.G                 = None
        self.step_num          = 0
        self.prev_i_count      = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed   = np.random.randint(0, 10000) if seed is None else seed
        self.G    = create_graph(self.graph_type, self.num_nodes, seed=ep_seed)
        self.G, _ = seed_infection(self.G, num_seeds=NUM_SEEDS,
                                   strategy="high_degree", seed=ep_seed)
        self.step_num     = 0
        self.prev_i_count = sum(1 for n in self.my_region
                                if self.G.nodes[n]["status"] == "I")
        self.received_messages = [np.zeros(MESSAGE_SIZE, dtype=np.float32)]

        state = build_state(self.G, self.my_region,
                            all_regions=self.regions,
                            received_messages=self.received_messages)
        return state, {}

    def step(self, action):
        self.G, action_taken = apply_action(self.G, action, self.my_region)

        self.G, new_infected, _ = sir_step(
            self.G, recovery_time=self.recovery_time,
            spread_rate=self.spread_rate
        )

        region_set          = set(self.my_region)
        region_size         = len(self.my_region)
        current_i           = sum(1 for n in self.my_region
                                  if self.G.nodes[n]["status"] == "I")
        region_new_infected = sum(1 for n in new_infected if n in region_set)
        hub_infections      = sum(
            1 for n in new_infected
            if n in region_set and self.G.nodes[n]["influence"] > 0.3
        ) if new_infected else 0

        # Reward
        reward          = 0.0
        delta           = self.prev_i_count - current_i
        reward         += delta * 3.0
        reward         -= region_new_infected * 2.0
        reward         -= hub_infections      * 1.5
        if action_taken:
            reward     -= 0.1
        infection_rate  = current_i / region_size
        if action == 0 and infection_rate > 0.15:
            reward     -= 1.5
        if current_i == 0 and self.prev_i_count > 0:
            reward     += 10.0

        self.step_num  += 1
        total_infected  = sum(1 for n in self.G.nodes()
                              if self.G.nodes[n]["status"] == "I")
        terminated      = (total_infected == 0)
        truncated       = (self.step_num >= self.max_steps)

        next_state = build_state(self.G, self.my_region,
                                 all_regions=self.regions,
                                 received_messages=self.received_messages)
        self.prev_i_count = current_i

        return next_state, reward, terminated, truncated, {}

    def render(self):
        i = sum(1 for n in self.G.nodes() if self.G.nodes[n]["status"] == "I")
        print(f"  Step {self.step_num:>3} | Infected: {i:>5}")


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing compact MisinfoEnv — 20-feature state...")
    env   = MisinfoEnv(agent_id=0)
    state, _ = env.reset(seed=42)
    print(f"State length : {len(state)}  (was ~2500)")
    print(f"State values : {state.round(3)}")

    total_reward, done, step = 0, False, 0
    while not done:
        env.received_messages = [generate_message(env.G, env.regions[1])]
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

    print(f"Episode: {step} steps | Reward: {total_reward:.2f}")
    print("Test PASSED.")