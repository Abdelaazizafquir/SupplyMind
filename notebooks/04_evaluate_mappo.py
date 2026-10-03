"""
Evaluate MAPPO against Baselines
================================
Runs the MAPPO agent using the best Kaggle checkpoint through 
both Normal and Disrupted environments, then natively appends 
its score to the baseline PNG charts.
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.agents.mappo import MAPPOAgent
from src.agents.graph import SupplyChainGraph
class MAPPOObsWrapper:
    def __init__(self, env):
        self.env = env
        self.demand_history = []
        self.last_demand = None
        self.nr = 3
        self.nd = 2
        self.ns = 1
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.demand_history = []
        self.last_demand = np.zeros(self.nr)
        return self._wrap_obs(), info
        
    def step(self, action):
        obs, rewards, term, trunc, info = self.env.step(action)
        
        # Extract demand from info
        d = np.zeros(self.nr)
        for i in range(self.nr):
            if f'agent_{i}' in info and 'demand' in info[f'agent_{i}']:
                d[i] = info[f'agent_{i}']['demand']
        self.last_demand = d
        self.demand_history.append(float(np.mean(d)))
        
        return self._wrap_obs(), rewards, term, trunc, info
        
    def _wrap_obs(self):
        obs = {}
        td = float(np.sum(self.last_demand)) if self.last_demand is not None else 0.
        avg_d = np.mean(self.demand_history[-5:]) if self.demand_history else 0.
        trend = 0.
        if len(self.demand_history) >= 10:
            r = np.mean(self.demand_history[-5:])
            o = np.mean(self.demand_history[-10:-5])
            trend = (r-o)/(o+1e-8)
            
        base_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        for i, aid in enumerate(base_env.agent_ids):
            node = base_env.nodes[i]
            pipe = sum(q for _,q in node.pipeline)
            ld = self.last_demand[i]/100. if i < self.nr else td/(self.nr*100.)
            cap = 1000 if i < self.nr else (2500 if i < self.nr+self.nd else 5000)
            ip = (node.inventory + pipe)/cap
            
            obs[aid] = np.array([
                node.inventory/1000, 
                pipe/1000, 
                ld, 
                ip, 
                avg_d/100.,
                base_env.current_step/base_env.num_periods, 
                1 if i < self.nr else 0,
                1 if self.nr <= i < self.nr+self.nd else 0, 
                1 if i >= self.nr+self.nd else 0,
                trend, 
                td/(self.nr*100.), 
                float(len([p for p in node.pipeline if p[1]>0]))/5.
            ], dtype=np.float32)
            
        return obs


def plot_results(results_dict, title, filename):
    plt.figure(figsize=(10, 6))
    policies = list(results_dict.keys())
    costs = []
    for v in results_dict.values():
        if isinstance(v, dict) and "metrics" in v:
            costs.append(float(v["metrics"]["total_cost"]["mean"]))
        elif isinstance(v, dict) and "mean" in v:
            costs.append(float(v["mean"]))
        else:
            costs.append(float(v))
    
    sns.barplot(x=policies, y=costs, palette="viridis")
    plt.title(title)
    plt.ylabel("Total Cost ($)")
    
    formatter = plt.FuncFormatter(lambda x, pos: f'${x*1e-6:.1f}M')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    for i, cost in enumerate(costs):
        plt.text(i, cost, f'${cost*1e-6:.2f}M', ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_mappo(env, agent):
    obs, _ = env.reset(seed=42)
    if hasattr(env, 'env'):
        base_env = env.env
        if hasattr(base_env, 'unwrapped'):
            base_env = base_env.unwrapped
    else:
        base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    done = False
    
    total_cost = 0.0
    while not done:
        # Formulate observation inputs
        agent_actions = {}
        for i, aid in enumerate(base_env.agent_ids):
            o = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0).to(agent.device)
            # Find the role (3 role types)
            r = 0 if i < base_env.num_retailers else (1 if i < base_env.num_retailers + base_env.num_distributors else 2)
            r_tensor = torch.zeros((1, 3)).to(agent.device)
            r_tensor[0, r] = 1.0
            
            with torch.no_grad():
                # Get GNN Context
                if agent.gnn is not None:
                    # Get augmented dimension block correctly ordered
                    all_obs = torch.tensor(np.array([obs[a] for a in base_env.agent_ids]), dtype=torch.float32).to(agent.device)
                    # GNN expects concat [obs, static_feats]
                    gnn_in = torch.cat([all_obs, agent.static_features], dim=-1)
                    gnn_emb = agent.gnn(gnn_in, agent.adjacency)
                    
                    # Augment observeration
                    aug_o = torch.cat([o, gnn_emb[i:i+1]], dim=-1)
                    action, _ = agent.actor.get_action(aug_o, r_tensor, deterministic=True)
                else:
                    action, _ = agent.actor.get_action(o, r_tensor, deterministic=True)
            
            agent_actions[aid] = action.cpu().numpy()[0]

        obs, rewards, terminated, truncated, infos = env.step(agent_actions)
        
        # Un-scale the reward (env divides by 1000 and adds 6% cooperative penalty)
        total_cost += sum(-r for r in rewards.values()) * 1000.0 / 1.06
        done = terminated["__all__"] or truncated["__all__"]
        
    return total_cost

if __name__ == "__main__":
    mappo_dir = project_root / "models" / "agents"
    baseline_dir = project_root / "experiments" / "baseline_results"
    
    model_paths = list(mappo_dir.glob("mappo_gnn_final.pt"))
    if not model_paths:
        print("Error: No MAPPO checkpoint found in models/agents/")
        sys.exit(1)
        
    best_model = sorted(model_paths)[-1]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chkpt = torch.load(best_model, map_location="cpu", weights_only=False)
    
    config = chkpt.get("config", {"hidden_dim": 512, "gnn_hidden_dim": 64, "obs_dim": 12})
    obs_dim = config.get("obs_dim", 12)
    
    # ── Graph Setup ──
    graph = SupplyChainGraph.from_env_config(
        num_retailers=3,
        num_distributors=2,
        num_suppliers=1,
        lead_time_mean=3, # Must match env
    )
    adjacency = graph.get_adjacency_matrix().numpy()
    static_features = graph.node_features.numpy()
    
    # Initialize exactly how 03_train_mappo.py did
    agent = MAPPOAgent(
        num_agents=6,
        obs_dim=obs_dim,
        action_dim=1,
        config=config,
        adjacency=adjacency,
        static_node_features=static_features
    )
    # Remap Kaggle checkpoint keys to Local Architectures
    gnn_state = chkpt["gnn"]
    fixed_gnn_state = {}
    for k, v in gnn_state.items():
        k = k.replace("layers.", "gnn_layers.").replace("norms.", "layer_norms.")
        fixed_gnn_state[k] = v
        
    agent.gnn.load_state_dict(fixed_gnn_state, strict=False)
    agent.actor.load_state_dict(chkpt["actor"], strict=False)
    agent.gnn.eval()
    agent.actor.eval()
    
    env_config = {
        "num_retailers": 3,
        "num_distributors": 2,
        "num_suppliers": 1,
        "num_periods": 100,
        "holding_cost": 1.0,
        "ordering_cost": 50.0,
        "shortage_cost": 10.0,
        "max_order_quantity": 500,
        "seed": 42
    }
    
    # Eval Normal
    env_norm = MAPPOObsWrapper(MultiEchelonInventoryEnv(env_config))
    cost_norm = evaluate_mappo(env_norm, agent)
    
    from src.environment.disruption_wrapper import DisruptionWrapper
    
    # Eval Disrupted
    env_disr_base = MultiEchelonInventoryEnv(env_config)
    disruption_config = {
        "demand_spike_prob": 0.03,
        "demand_spike_factor": 3.0,
        "lead_time_delay_prob": 0.04,
        "lead_time_delay_factor": 3.0,
        "node_failure_prob": 0.02,
        "node_failure_duration": 10,
        "cascade_enabled": True,
    }
    env_disr = MAPPOObsWrapper(DisruptionWrapper(env_disr_base, disruption_config))
    cost_disr = evaluate_mappo(env_disr, agent)
    
    print(f"MAPPO Normal Cost: ${float(cost_norm):,.2f}")
    print(f"MAPPO Disrupted Cost: ${float(cost_disr):,.2f}")
    
    # Merge and replot Normal
    with open(baseline_dir / "results_normal.json", "r") as f:
        res_n = json.load(f)
    res_n["MAPPO+GNN (Fine-Tuned)"] = {"mean": float(cost_norm), "std": 0.0}
    with open(baseline_dir / "results_normal.json", "w") as f:
        json.dump(res_n, f, indent=4)
    plot_results(res_n, "Policy Comparison: Normal Operations", baseline_dir / "baseline_comparison_normal.png")
    
    # Merge and replot Disrupted
    with open(baseline_dir / "results_disrupted.json", "r") as f:
        res_d = json.load(f)
    res_d["MAPPO+GNN (Fine-Tuned)"] = {"mean": float(cost_disr), "std": 0.0}
    with open(baseline_dir / "results_disrupted.json", "w") as f:
        json.dump(res_d, f, indent=4)
    plot_results(res_d, "Policy Comparison: Active Disruption (Factory Fire)", baseline_dir / "baseline_comparison_disrupted.png")
    
    print("Successfully appended MAPPO results to baseline plots!")
