"""
action_audit.py — Check what actions trained agents actually pick.

If agents are mostly picking action 0 (do nothing), they've
learned passivity as a local optimum. This tells us the reward
signal isn't guiding them toward useful behavior.

Run this before any more retraining.
"""

import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_env       import MisinfoEnv, build_state, generate_message, MESSAGE_SIZE, NUM_NODES
from graph_env    import create_graph
from sir_model    import seed_infection, sir_step
from stable_baselines3 import DQN


def audit_actions(models, num_episodes=20):
    """
    Runs episodes and records which actions each agent picks.
    Prints action frequency distribution.
    """
    action_labels = {
        0: "Do Nothing",
        1: "Flag (reduce influence)",
        2: "Counter-message (protect neighbors)",
        3: "Quarantine (remove node)",
    }

    action_counts = [
        {0: 0, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 0},
    ]
    total_steps = [0, 0]

    env0 = MisinfoEnv(agent_id=0)
    env1 = MisinfoEnv(agent_id=1)
    regions = env0.regions

    for ep in range(num_episodes):
        G = create_graph("barabasi_albert", num_nodes=NUM_NODES, seed=ep)
        G, _ = seed_infection(G, num_seeds=5, strategy="high_degree", seed=ep)

        messages = [
            np.zeros(MESSAGE_SIZE, dtype=np.float32),
            np.zeros(MESSAGE_SIZE, dtype=np.float32),
        ]

        for step in range(60):
            i_count = sum(1 for n in G.nodes() if G.nodes[n]["status"] == "I")
            if i_count == 0:
                break

            for agent_id, region in enumerate(regions):
                received = [messages[1 - agent_id]]
                state    = build_state(G, region,
                                       all_regions=regions,
                                       received_messages=received)
                action, _ = models[agent_id].predict(state, deterministic=True)
                action_counts[agent_id][int(action)] += 1
                total_steps[agent_id] += 1

            for agent_id, region in enumerate(regions):
                messages[agent_id] = generate_message(G, region)

            from sir_model import sir_step
            G, _, _ = sir_step(G, recovery_time=4, spread_rate=0.3)

    print("\n" + "=" * 55)
    print("  ACTION FREQUENCY AUDIT")
    print("=" * 55)

    for agent_id in range(2):
        total = total_steps[agent_id]
        print(f"\n  Agent {agent_id} ({total} total decisions):")
        for action, label in action_labels.items():
            count = action_counts[agent_id][action]
            pct   = count / total * 100 if total > 0 else 0
            bar   = "█" * int(pct / 2)
            print(f"    Action {action} ({label:<30}): "
                  f"{pct:5.1f}%  {bar}")

    print("\n" + "=" * 55)

    # Verdict
    for agent_id in range(2):
        total    = total_steps[agent_id]
        do_nothing_pct = action_counts[agent_id][0] / total * 100
        if do_nothing_pct > 60:
            print(f"\n  ⚠️  Agent {agent_id} is doing nothing {do_nothing_pct:.0f}%"
                  f" of the time — learned passivity.")
        elif action_counts[agent_id][3] / total > 0.7:
            print(f"\n  ⚠️  Agent {agent_id} is always quarantining"
                  f" ({action_counts[agent_id][3]/total*100:.0f}%) —"
                  f" same as greedy, no learned policy.")
        else:
            print(f"\n  ✅ Agent {agent_id} is using a mix of actions.")

    return action_counts


if __name__ == "__main__":
    models_dir = "models"
    env0   = MisinfoEnv(agent_id=0)
    env1   = MisinfoEnv(agent_id=1)
    model0 = DQN.load(os.path.join(models_dir, "agent0_dqn"), env=env0)
    model1 = DQN.load(os.path.join(models_dir, "agent1_dqn"), env=env1)

    audit_actions([model0, model1], num_episodes=20)