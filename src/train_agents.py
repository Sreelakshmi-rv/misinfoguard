"""
train_agents.py — Trains DQN agents for each network type independently.

Each network type gets its own pair of trained agents because:
- WS has no hubs — flagging is less useful, counter-messaging matters more
- ER has uniform degree — greedy and RL targeting differ substantially
- HK has clustering — echo chamber dynamics differ from pure BA

Model files saved as:
    models/agent0_{network_type}_dqn.zip
    models/agent1_{network_type}_dqn.zip
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from rl_env     import MisinfoEnv, NUM_NODES, NUM_REGIONS
from graph_env  import NETWORK_CONFIGS
from simulation import (
    run_comparison,
    evaluate_scenarios,
    SCENARIO_LABELS,
    SCENARIO_COLORS,
)


# ─────────────────────────────────────────────
# CALLBACK
# ─────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, agent_id, verbose=0):
        super().__init__(verbose)
        self.log_path  = log_path
        self.agent_id  = agent_id
        self.episode   = 0
        self.ep_reward = 0.0

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward"])

    def _on_step(self):
        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode   += 1
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode,
                                 round(self.ep_reward, 3)])
            if self.episode % 20 == 0:
                print(f"    Agent {self.agent_id} | "
                      f"Episode {self.episode:>4} | "
                      f"Reward: {self.ep_reward:.2f}")
            self.ep_reward = 0.0
        return True


# ─────────────────────────────────────────────
# TRAIN ONE AGENT
# ─────────────────────────────────────────────

def train_agent(agent_id, graph_type="barabasi_albert",
                total_timesteps=200_000):
    """
    Trains one DQN agent for a specific network type.
    Model saved as models/agent{id}_{graph_type}_dqn.zip
    """
    config = NETWORK_CONFIGS[graph_type]

    print(f"\n{'='*55}")
    print(f"  Training Agent {agent_id} | {config['label']}")
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"  Network   : {NUM_NODES} nodes, {NUM_REGIONS} regions")
    print(f"{'='*55}")

    env = MisinfoEnv(
        graph_type  = graph_type,
        num_nodes   = NUM_NODES,
        num_regions = NUM_REGIONS,
        agent_id    = agent_id,
    )

    os.makedirs("models", exist_ok=True)
    log_path = f"models/agent{agent_id}_{graph_type}_rewards.csv"
    callback = RewardLoggerCallback(log_path=log_path,
                                    agent_id=agent_id)

    model = DQN(
        policy                = "MlpPolicy",
        env                   = env,
        learning_rate         = 5e-4,
        batch_size            = 128,
        buffer_size           = 100_000,
        exploration_fraction  = 0.4,
        exploration_final_eps = 0.05,
        verbose               = 0,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = f"models/agent{agent_id}_{graph_type}_dqn"
    model.save(save_path)
    print(f"\n  Agent {agent_id} saved to {save_path}.zip")

    return model


# ─────────────────────────────────────────────
# LOAD MODELS FOR ONE NETWORK TYPE
# ─────────────────────────────────────────────

def load_models(graph_type="barabasi_albert"):
    """
    Loads trained agent pair for a given network type.
    Falls back to base agents if network-specific not found.
    """
    models_dir = "models"
    env0 = MisinfoEnv(agent_id=0, graph_type=graph_type)
    env1 = MisinfoEnv(agent_id=1, graph_type=graph_type)

    path0 = os.path.join(models_dir, f"agent0_{graph_type}_dqn")
    path1 = os.path.join(models_dir, f"agent1_{graph_type}_dqn")

    if not os.path.exists(path0 + ".zip"):
        print(f"  Warning: {path0}.zip not found, using base BA agents")
        path0 = os.path.join(models_dir, "agent0_dqn")
        path1 = os.path.join(models_dir, "agent1_dqn")

    model0 = DQN.load(path0, env=env0)
    model1 = DQN.load(path1, env=env1)
    return [model0, model1]


# ─────────────────────────────────────────────
# TRAIN ALL NETWORKS
# ─────────────────────────────────────────────

def train_all_networks(network_types=None, total_timesteps=200_000):
    """
    Trains agent pairs for each network type sequentially.
    Returns dict mapping graph_type to [model0, model1].
    """
    if network_types is None:
        network_types = list(NETWORK_CONFIGS.keys())

    all_models = {}

    for graph_type in network_types:
        print(f"\n{'#'*55}")
        print(f"  NETWORK: {NETWORK_CONFIGS[graph_type]['label']}")
        print(f"{'#'*55}")

        model0 = train_agent(0, graph_type, total_timesteps)
        model1 = train_agent(1, graph_type, total_timesteps)
        all_models[graph_type] = [model0, model1]

    return all_models


# ─────────────────────────────────────────────
# PLOT LEARNING CURVES FOR ONE NETWORK TYPE
# ─────────────────────────────────────────────

def plot_learning_curves(graph_type="barabasi_albert"):
    colors = ["steelblue", "crimson"]
    label  = NETWORK_CONFIGS[graph_type]["label"]

    plt.figure(figsize=(13, 5))

    for agent_id in range(NUM_REGIONS):
        log_path = f"models/agent{agent_id}_{graph_type}_rewards.csv"
        if not os.path.exists(log_path):
            log_path = f"models/agent{agent_id}_rewards.csv"
        if not os.path.exists(log_path):
            print(f"  Warning: {log_path} not found, skipping")
            continue

        episodes, rewards = [], []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["total_reward"]))

        smoothed = []
        window   = 30
        for i in range(len(rewards)):
            start = max(0, i - window)
            smoothed.append(np.mean(rewards[start:i+1]))

        plt.plot(episodes, rewards,
                 alpha=0.15, color=colors[agent_id])
        plt.plot(episodes, smoothed, linewidth=2,
                 label=f"Agent {agent_id} (smoothed, window=30)",
                 color=colors[agent_id])

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(
        f"Agent Learning Curves — {label}\n"
        "Partial Observability + Message Passing + Echo Chamber Edge Weights"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = f"models/learning_curves_{graph_type}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# PLOT SINGLE-SEED COMPARISON
# ─────────────────────────────────────────────

def plot_comparison(results, graph_type="barabasi_albert"):
    label  = NETWORK_CONFIGS[graph_type]["label"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    print(f"\n── SINGLE-SEED COMPARISON — {label} ──")
    print(f"  {'Scenario':<28} {'Peak':>12} {'Total Reached':>15}")
    print("  " + "-" * 58)

    for scenario, history in results.items():
        total   = history[0]["S"] + history[0]["I"] + history[0]["R"]
        steps   = [h["step"] for h in history]
        inf_pct = [h["I"] / total * 100 for h in history]
        reached = max(h["I"] + h["R"] for h in history)
        peak    = max(inf_pct)
        lbl     = SCENARIO_LABELS.get(scenario, scenario)
        color   = SCENARIO_COLORS.get(scenario, "gray")

        print(f"  {lbl:<28} {peak:>11.1f}% "
              f"{reached:>9} ({round(reached/total*100, 1)}%)")

        ax1.plot(steps, inf_pct, label=lbl, color=color, linewidth=2)

        if scenario in ["greedy", "rl"]:
            ax2.plot(steps, inf_pct, label=lbl, color=color,
                     linewidth=2.5, marker="o", markersize=4)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("% of Network Infected")
    ax1.set_title(f"All Scenarios — {label}\n"
                  "Greedy under same partial observability as RL")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("% of Network Infected")
    ax2.set_title("Greedy vs RL — Zoomed")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 20)

    plt.suptitle(
        f"Intervention Comparison — {NUM_NODES} Nodes, "
        f"{NUM_REGIONS} Agents, {label}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    save_path = f"models/comparison_plot_{graph_type}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nComparison plot saved to {save_path}")


# ─────────────────────────────────────────────
# PLOT MULTI-SEED BAR CHART
# ─────────────────────────────────────────────

def plot_multiseed_results(summary, scenarios, graph_type="barabasi_albert"):
    label  = NETWORK_CONFIGS[graph_type]["label"]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    means  = [summary[s]["reached_mean"]  for s in scenarios]
    stds   = [summary[s]["reached_std"]   for s in scenarios]
    colors = [SCENARIO_COLORS.get(s, "gray") for s in scenarios]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, means, yerr=stds,
                  color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.2,
                  capsize=6, error_kw={"linewidth": 2})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.5,
            f"{mean:.1f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Total Network Reached (%)", fontsize=12)
    ax.set_title(
        f"Multi-Seed Evaluation — {label}\n"
        f"{NUM_NODES} nodes | Greedy under partial observability",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, max(means) + max(stds) + 8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    save_path = f"models/multiseed_results_{graph_type}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Multi-seed chart saved to {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    ALL_SCENARIOS  = ["none", "random", "greedy", "rl"]
    NETWORK_TYPES  = list(NETWORK_CONFIGS.keys())

    # ── Step 1: Train all networks ──
    print("\nStep 1: Training agents for all 4 network types...")
    all_models = train_all_networks(
        network_types   = NETWORK_TYPES,
        total_timesteps = 200_000,
    )

    # ── Step 2: Evaluate and plot per network ──
    for graph_type in NETWORK_TYPES:
        label  = NETWORK_CONFIGS[graph_type]["label"]
        models = all_models[graph_type]

        print(f"\nStep 2: Learning curves — {label}...")
        plot_learning_curves(graph_type)

        print(f"\nStep 3: Single-seed comparison — {label}...")
        single_results = run_comparison(
            scenarios  = ALL_SCENARIOS,
            models     = models,
            seed       = 8,
            graph_type = graph_type,
        )
        plot_comparison(single_results, graph_type)

        print(f"\nStep 4: Multi-seed evaluation — {label}...")
        summary = evaluate_scenarios(
            scenarios      = ALL_SCENARIOS,
            models         = models,
            num_seeds_eval = 100,
            graph_type     = graph_type,
            verbose        = True,
        )
        plot_multiseed_results(summary, ALL_SCENARIOS, graph_type)

    print(f"\nAll done. Files in models/:")
    for g in NETWORK_TYPES:
        short = NETWORK_CONFIGS[g]["short"]
        print(f"  agent0_{g}_dqn.zip")
        print(f"  agent1_{g}_dqn.zip")
        print(f"  learning_curves_{g}.png")
        print(f"  comparison_plot_{g}.png")
        print(f"  multiseed_results_{g}.png")