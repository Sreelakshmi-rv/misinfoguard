import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation import evaluate_scenarios
from rl_env import MisinfoEnv
from stable_baselines3 import DQN
from scipy import stats
import numpy as np

for graph_type in ["erdos_renyi", "watts_strogatz"]:
    env0   = MisinfoEnv(agent_id=0, graph_type=graph_type)
    env1   = MisinfoEnv(agent_id=1, graph_type=graph_type)
    model0 = DQN.load(f"models/agent0_{graph_type}_dqn", env=env0)
    model1 = DQN.load(f"models/agent1_{graph_type}_dqn", env=env1)

    summary = evaluate_scenarios(
        scenarios      = ["greedy", "rl"],
        models         = [model0, model1],
        num_seeds_eval = 200,
        graph_type     = graph_type,
        verbose        = False,
    )

    greedy = summary["greedy"]["raw_reached"]
    rl     = summary["rl"]["raw_reached"]
    t, p   = stats.ttest_ind(greedy, rl, equal_var=False)
    d      = (np.mean(greedy) - np.mean(rl)) / np.sqrt(
             (np.std(greedy)**2 + np.std(rl)**2) / 2)

    print(f"\n{graph_type}")
    print(f"  Gap     : {np.mean(greedy) - np.mean(rl):.2f}%")
    print(f"  p-value : {p:.4f}")
    print(f"  Cohen d : {d:.4f}")