"""
train_non_ba_networks.py — Trains agents for ER, HK, WS only.

BA agents are loaded from existing models (agent0_dqn.zip renamed).
Each non-BA network uses network-specific timesteps and spread rates
based on observed learning curve behaviour.

Changes from standard training:
    ER  — 400k steps, spread_rate=0.25 (high baseline infection, lower rate helps)
    HK  — 300k steps, spread_rate=0.30 (same as BA, just needs more time)
    WS  — 500k steps, spread_rate=0.40 (uniform degree needs stronger signal)
          Also uses rewiring p=0.3 in graph_env.py for more degree heterogeneity

Run this AFTER renaming BA models:
    copy models\\agent0_dqn.zip models\\agent0_barabasi_albert_dqn.zip
    copy models\\agent1_dqn.zip models\\agent1_barabasi_albert_dqn.zip

Usage:
    python src/train_non_ba_networks.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from rl_env    import MisinfoEnv, NUM_NODES, NUM_REGIONS
from graph_env import NETWORK_CONFIGS


# ─────────────────────────────────────────────
# NETWORK-SPECIFIC TRAINING CONFIG
# ─────────────────────────────────────────────

TRAINING_CONFIGS = {
    "erdos_renyi": {
        "total_timesteps": 400_000,
        "spread_rate"    : 0.25,
        "reason"         : "High baseline infection — lower spread rate "
                           "creates more controllable episodes",
    },
    "holme_kim": {
        "total_timesteps": 300_000,
        "spread_rate"    : 0.30,
        "reason"         : "Similar to BA — just needs more training time",
    },
    "watts_strogatz": {
        "total_timesteps": 500_000,
        "spread_rate"    : 0.40,
        "reason"         : "Uniform degree kills risk signal — higher spread "
                           "rate creates severe enough infections for "
                           "clear reward differentiation",
    },
}


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
# TRAIN ONE AGENT WITH CUSTOM CONFIG
# ─────────────────────────────────────────────

def train_agent(agent_id, graph_type, config):
    total_timesteps = config["total_timesteps"]
    spread_rate     = config["spread_rate"]
    label           = NETWORK_CONFIGS[graph_type]["label"]

    print(f"\n{'='*60}")
    print(f"  Training Agent {agent_id} | {label}")
    print(f"  Timesteps   : {total_timesteps:,}")
    print(f"  Spread rate : {spread_rate}")
    print(f"  Reason      : {config['reason']}")
    print(f"{'='*60}")

    env = MisinfoEnv(
        graph_type    = graph_type,
        num_nodes     = NUM_NODES,
        num_regions   = NUM_REGIONS,
        agent_id      = agent_id,
        spread_rate   = spread_rate,
    )

    os.makedirs("models", exist_ok=True)
    log_path = f"models/agent{agent_id}_{graph_type}_rewards.csv"
    callback = RewardLoggerCallback(log_path=log_path, agent_id=agent_id)

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
    print(f"\n  Saved to {save_path}.zip")

    return model


# ─────────────────────────────────────────────
# PLOT LEARNING CURVE
# ─────────────────────────────────────────────

def plot_learning_curve(graph_type):
    colors = ["steelblue", "crimson"]
    label  = NETWORK_CONFIGS[graph_type]["label"]
    plt.figure(figsize=(13, 5))

    for agent_id in range(NUM_REGIONS):
        log_path = f"models/agent{agent_id}_{graph_type}_rewards.csv"
        if not os.path.exists(log_path):
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

        plt.plot(episodes, rewards, alpha=0.15, color=colors[agent_id])
        plt.plot(episodes, smoothed, linewidth=2,
                 label=f"Agent {agent_id} (smoothed)",
                 color=colors[agent_id])

    config = TRAINING_CONFIGS[graph_type]
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(
        f"Learning Curves — {label}\n"
        f"Timesteps: {config['total_timesteps']:,} | "
        f"Spread rate: {config['spread_rate']}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = f"models/learning_curves_{graph_type}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# VERIFY BA MODELS EXIST
# ─────────────────────────────────────────────

def check_ba_models():
    path0 = "models/agent0_barabasi_albert_dqn.zip"
    path1 = "models/agent1_barabasi_albert_dqn.zip"

    if not os.path.exists(path0) or not os.path.exists(path1):
        print("\n" + "=" * 60)
        print("  WARNING: BA models not found.")
        print("  Run these commands first:")
        print("    copy models\\agent0_dqn.zip "
              "models\\agent0_barabasi_albert_dqn.zip")
        print("    copy models\\agent1_dqn.zip "
              "models\\agent1_barabasi_albert_dqn.zip")
        print("=" * 60)
        return False

    print("  BA models found. Using existing trained agents.")
    return True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  TARGETED TRAINING — ER, HK, WS")
    print("  BA uses existing trained models")
    print("=" * 60)

    # Verify BA models are renamed and ready
    check_ba_models()

    # Train each non-BA network with its specific config
    for graph_type, config in TRAINING_CONFIGS.items():
        label = NETWORK_CONFIGS[graph_type]["label"]
        print(f"\n{'#'*60}")
        print(f"  STARTING: {label}")
        print(f"{'#'*60}")

        model0 = train_agent(0, graph_type, config)
        model1 = train_agent(1, graph_type, config)

        print(f"\n  Plotting learning curves for {label}...")
        plot_learning_curve(graph_type)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("  Files saved in models/:")
    for graph_type in TRAINING_CONFIGS:
        print(f"    agent0_{graph_type}_dqn.zip")
        print(f"    agent1_{graph_type}_dqn.zip")
        print(f"    learning_curves_{graph_type}.png")
    print("\n  Next step: run multi_network_eval.py")
    print("=" * 60).