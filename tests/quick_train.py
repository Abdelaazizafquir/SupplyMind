"""Quick training validation: 30 episodes to verify learning happens."""
import sys
sys.path.insert(0, "C:/Users/yoga/Desktop/pfe")

import numpy as np
import time
import torch
from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.agents.mappo import MAPPOAgent
from src.agents.graph import SupplyChainGraph
from src.utils.data_processing import generate_synthetic_demand

NUM_EP = 30
NUM_PERIODS = 100

env = MultiEchelonInventoryEnv({
    "num_retailers": 3, "num_distributors": 2, "num_suppliers": 1,
    "num_periods": NUM_PERIODS
})
demand = generate_synthetic_demand(NUM_PERIODS, 3)
env.set_demand_data(demand)

graph = SupplyChainGraph.from_env_config(3, 2, 1)
adj = graph.get_adjacency_matrix().numpy()
static = graph.node_features.numpy()
obs, _ = env.reset(seed=42)
obs_dim = obs["agent_0"].shape[0]

agent = MAPPOAgent(
    6, obs_dim, 1,
    {"use_gnn": True, "gnn_hidden_dim": 64, "num_epochs": 5,
     "num_minibatches": 2, "hidden_dim": 256},
    adj, static
)

print(f"Training {NUM_EP} episodes x {NUM_PERIODS} steps...")
rewards_log = []
t0 = time.time()

for ep in range(NUM_EP):
    env.set_demand_data(demand.copy())
    obs, _ = env.reset(seed=42 + ep)
    ep_rew = 0

    while True:
        actions, lps, vals = agent.get_actions(obs, env.agent_ids)
        next_obs, rew, term, trunc, info = env.step(actions)
        done = trunc.get("__all__", False)
        dones = {a: done for a in env.agent_ids}
        agent.buffer.add(obs, actions, rew, dones, lps, vals)
        ep_rew += sum(rew.values())
        obs = next_obs

        if len(agent.buffer) >= 64:
            if not done:
                with torch.no_grad():
                    _, _, bv = agent.get_actions(next_obs, env.agent_ids)
                    bootstrap = float(np.mean([bv[a] for a in env.agent_ids]))
            else:
                bootstrap = 0.0
            agent.update(env.agent_ids, last_value=bootstrap)

        if done:
            break

    if len(agent.buffer) > 0:
        agent.update(env.agent_ids, last_value=0.0)

    rewards_log.append(ep_rew)
    if (ep + 1) % 5 == 0:
        avg5 = np.mean(rewards_log[-5:])
        print(f"  Ep {ep+1:3d}/{NUM_EP} | Avg5 Reward: {avg5:>10.2f} | Time: {time.time()-t0:.0f}s")

first5 = np.mean(rewards_log[:5])
last5 = np.mean(rewards_log[-5:])
print()
print(f"First 5 avg: {first5:.2f}")
print(f"Last  5 avg: {last5:.2f}")
pct = ((last5 - first5) / abs(first5)) * 100 if first5 != 0 else 0
print(f"Improvement:  {pct:.1f}%")
if last5 > first5:
    print("LEARNING IS HAPPENING - reward is improving")
else:
    print("WARNING: No improvement detected")
