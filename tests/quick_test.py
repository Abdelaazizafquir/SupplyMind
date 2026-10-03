"""Quick integration test for all fixes."""
import sys
sys.path.insert(0, "C:/Users/yoga/Desktop/pfe")

import numpy as np
import torch
from src.agents.graph import SupplyChainGraph
from src.agents.gnn_comm import GNNCommunication
from src.agents.mappo import MAPPOAgent
from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.utils.data_processing import generate_synthetic_demand

print("=" * 50)
print("  QUICK INTEGRATION TEST")
print("=" * 50)

# 1. Test env with role indicators
env = MultiEchelonInventoryEnv({
    "num_retailers": 3, "num_distributors": 2,
    "num_suppliers": 1, "num_periods": 50
})
demand = generate_synthetic_demand(50, 3)
env.set_demand_data(demand)
obs, _ = env.reset(seed=42)

print(f"\n1. Env obs shape: {obs['agent_0'].shape}")
print(f"   Role (agent_0/retailer):  {obs['agent_0'][6:9]}")
print(f"   Role (agent_3/distrib):   {obs['agent_3'][6:9]}")
print(f"   Role (agent_5/supplier):  {obs['agent_5'][6:9]}")

assert obs["agent_0"][6] == 1.0 and obs["agent_0"][7] == 0.0, "Role bug: retailer"
assert obs["agent_3"][7] == 1.0 and obs["agent_3"][6] == 0.0, "Role bug: distributor"
assert obs["agent_5"][8] == 1.0 and obs["agent_5"][6] == 0.0, "Role bug: supplier"
print("   OK: Role indicators correct")

# 2. Test graph
graph = SupplyChainGraph.from_env_config(3, 2, 1)
adj = graph.get_adjacency_matrix().numpy()
static = graph.node_features.numpy()
print(f"\n2. Graph: {graph.num_nodes} nodes, adj={adj.shape}, static={static.shape}")
print("   OK: Graph built")

# 3. Test MAPPO agent with GNN
obs_dim = obs["agent_0"].shape[0]
agent = MAPPOAgent(
    6, obs_dim, 1,
    {"use_gnn": True, "gnn_hidden_dim": 64, "num_epochs": 2, "num_minibatches": 2},
    adj, static
)
actions, lps, vals = agent.get_actions(obs, env.agent_ids)
print(f"\n3. MAPPO actions OK, V(s0)={vals['agent_0']:.4f}")

# 4. Run 10 steps and collect rollout
for step in range(10):
    next_obs, rew, term, trunc, info = env.step(actions)
    done = trunc.get("__all__", False)
    dones = {aid: done for aid in env.agent_ids}
    agent.buffer.add(obs, actions, rew, dones, lps, vals)
    obs = next_obs
    actions, lps, vals = agent.get_actions(obs, env.agent_ids)

print(f"   Buffer size: {len(agent.buffer)}")

# 5. Test bootstrap value update (THE FIX)
bootstrap_val = float(np.mean([vals[aid] for aid in env.agent_ids]))
print(f"\n4. Bootstrap value: {bootstrap_val:.4f}")
stats = agent.update(env.agent_ids, last_value=bootstrap_val)
print(f"   Update stats: policy_loss={stats.get('policy_loss', 0):.4f}, "
      f"value_loss={stats.get('value_loss', 0):.4f}")
print("   OK: Update with bootstrap value works")

# 6. Test episode termination
env2 = MultiEchelonInventoryEnv({"num_retailers": 3, "num_distributors": 2,
                                  "num_suppliers": 1, "num_periods": 5})
env2.set_demand_data(generate_synthetic_demand(5, 3))
obs, _ = env2.reset(seed=42)
for s in range(10):
    act = {aid: np.array([0.5]) for aid in env2.agent_ids}
    obs, rew, term, trunc, info = env2.step(act)
    if trunc.get("__all__", False):
        print(f"\n5. Episode terminated at step {s+1} (expected: 5)")
        assert s + 1 == 5, f"Termination bug: ended at {s+1}"
        print("   OK: Episode termination correct")
        break

print(f"\n{'=' * 50}")
print("  ALL TESTS PASSED!")
print(f"{'=' * 50}")
