"""
Cost Audit Script - Breaks down MAPPO vs (s,S) costs granularly
"""
import sys, numpy as np
sys.path.insert(0, '.')

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.baselines.classical_policies import SsPoliciy

env_config = {
    'num_retailers': 3, 'num_distributors': 2, 'num_suppliers': 1,
    'num_periods': 100, 'holding_cost': 1.0, 'ordering_cost': 50.0,
    'shortage_cost': 10.0, 'max_order_quantity': 500
}

# ── RUN S,S ──
env = MultiEchelonInventoryEnv(env_config)
policy = SsPoliciy(env, s=30, S=100)
obs, _ = env.reset(seed=42)
base = env.unwrapped if hasattr(env, 'unwrapped') else env

ss_hold, ss_order, ss_short = 0.0, 0.0, 0.0
for t in range(100):
    actions = policy.get_action(obs)
    obs, rewards, terminated, truncated, infos = env.step(actions)
    for aid in base.agent_ids:
        if aid in infos:
            ss_hold += infos[aid].get('holding_cost', 0)
            ss_order += infos[aid].get('ordering_cost', 0)
            ss_short += infos[aid].get('shortage_cost', 0)
    if terminated.get('__all__', False):
        break

print("=======  (s,S) Cost Breakdown ========")
print(f"  Holding Cost:  ${ss_hold:,.2f}")
print(f"  Ordering Cost: ${ss_order:,.2f}")
print(f"  Shortage Cost: ${ss_short:,.2f}")
print(f"  TOTAL:         ${ss_hold+ss_order+ss_short:,.2f}")

# ── RUN MAPPO ──
import torch, warnings
from src.agents.mappo import MAPPOAgent
from src.agents.graph import SupplyChainGraph
from pathlib import Path

project_root = Path('.').resolve()
model_paths = list((project_root / "experiments" / "mappo_training").glob("mappo_gnn_*.pt"))
best_model = sorted(model_paths)[-1]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    chkpt = torch.load(best_model, map_location="cpu", weights_only=False)

config = chkpt.get("config", {})
graph = SupplyChainGraph.from_env_config(3, 2, 1, 3)
agent = MAPPOAgent(
    num_agents=6, obs_dim=config.get("obs_dim", 12), action_dim=1, config=config,
    adjacency=graph.get_adjacency_matrix().numpy(),
    static_node_features=graph.node_features.numpy()
)
gnn_state = {k.replace("layers.", "gnn_layers.").replace("norms.", "layer_norms."): v
             for k, v in chkpt["gnn"].items()}
agent.gnn.load_state_dict(gnn_state, strict=False)
agent.actor.load_state_dict(chkpt["actor"], strict=False)
agent.gnn.eval(); agent.actor.eval()

env2 = MultiEchelonInventoryEnv(env_config)
obs, _ = env2.reset(seed=42)
base2 = env2.unwrapped if hasattr(env2, 'unwrapped') else env2

mr_hold, mr_order, mr_short = 0.0, 0.0, 0.0
for t in range(100):
    agent_actions = {}
    with torch.no_grad():
        for i, aid in enumerate(base2.agent_ids):
            o = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0)
            r = 0 if i < base2.num_retailers else (1 if i < base2.num_retailers + base2.num_distributors else 2)
            r_t = torch.zeros((1, 3)); r_t[0, r] = 1.0
            if agent.gnn:
                all_obs = torch.tensor(np.array([obs[a] for a in base2.agent_ids]), dtype=torch.float32)
                gnn_in = torch.cat([all_obs, agent.static_features.cpu()], dim=-1)
                gnn_emb = agent.gnn(gnn_in, agent.adjacency.cpu())
                aug_o = torch.cat([o, gnn_emb[i:i+1]], dim=-1)
                action, _ = agent.actor.get_action(aug_o, r_t, deterministic=True)
            else:
                action, _ = agent.actor.get_action(o, r_t, deterministic=True)
            agent_actions[aid] = action.detach().cpu().numpy()[0]
    
    obs, rewards, terminated, truncated, infos = env2.step(agent_actions)
    for aid in base2.agent_ids:
        if aid in infos:
            mr_hold += infos[aid].get('holding_cost', 0)
            mr_order += infos[aid].get('ordering_cost', 0)
            mr_short += infos[aid].get('shortage_cost', 0)
    if terminated.get('__all__', False): break

print()
print("=======  MAPPO+GNN Cost Breakdown ======")
print(f"  Holding Cost:  ${mr_hold:,.2f}")
print(f"  Ordering Cost: ${mr_order:,.2f}")
print(f"  Shortage Cost: ${mr_short:,.2f}")
print(f"  TOTAL:         ${mr_hold+mr_order+mr_short:,.2f}")

print()
print("=======  IMPROVEMENT ========")
ss_total = ss_hold + ss_order + ss_short
mr_total = mr_hold + mr_order + mr_short
print(f"  Cost Reduction: {(1 - mr_total/ss_total)*100:.1f}%")
print(f"  Savings: ${ss_total - mr_total:,.2f}")
