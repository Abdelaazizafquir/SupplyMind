"""
Policy Explainability Engine
============================
Runs exactly 1 episode of (s, S) and MAPPO+GNN identically 
on a Disrupted env, tracking daily inventory and orders.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.environment.disruption_wrapper import DisruptionWrapper
from src.agents.mappo import MAPPOAgent
from src.agents.graph import SupplyChainGraph
from src.baselines.classical_policies import SsPolicy

def extract_episode_trace(env, agent=None, is_mappo=False, steps=100):
    obs, _ = env.reset(seed=42)
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    # We will trace "Distributor 0" (Node 3) which is hit by factory fire
    trace_inventory = []
    trace_orders = []
    
    for t in range(steps):
        # Record pre-action inventory
        target_node = base_env.nodes[3]
        trace_inventory.append(float(target_node.inventory))
        
        agent_actions = {}
        target_order = 0.0
        
        if is_mappo:
            for i, aid in enumerate(base_env.agent_ids):
                o = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0).to(agent.device)
                r = 0 if i < base_env.num_retailers else (1 if i < base_env.num_retailers + base_env.num_distributors else 2)
                r_tensor = torch.zeros((1, 3)).to(agent.device)
                r_tensor[0, r] = 1.0
                
                with torch.no_grad():
                    if agent.gnn is not None:
                        all_obs = torch.tensor(np.array([obs[a] for a in base_env.agent_ids]), dtype=torch.float32).to(agent.device)
                        gnn_in = torch.cat([all_obs, agent.static_features], dim=-1)
                        gnn_emb = agent.gnn(gnn_in, agent.adjacency)
                        aug_o = torch.cat([o, gnn_emb[i:i+1]], dim=-1)
                        action, _ = agent.actor.get_action(aug_o, r_tensor, deterministic=True)
                    else:
                        action, _ = agent.actor.get_action(o, r_tensor, deterministic=True)
                
                agent_actions[aid] = action.cpu().numpy()[0]
                
                # Reverse Action Squaring manually to calculate the true order qty placed
                if i == 3:
                    target_ratio = agent_actions[aid]
                    true_qty = float((target_ratio ** 2) * target_node.capacity)
                    target_order = true_qty
        else:
            # Baseline policy
            actions = agent.get_action(obs)
            agent_actions = actions
            target_ratio = actions[base_env.agent_ids[3]][0]
            true_qty = float((target_ratio ** 2) * target_node.capacity)
            target_order = true_qty
            
        trace_orders.append(target_order)
        obs, rewards, terminated, truncated, infos = env.step(agent_actions)

    return trace_inventory, trace_orders

if __name__ == "__main__":
    mappo_dir = project_root / "experiments" / "mappo_training"
    baseline_dir = project_root / "experiments" / "baseline_results"
    
    # ── Env Setup ──
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
    
    disruption_config = {
        "demand_spike_prob": 0.03,
        "demand_spike_factor": 3.0,
        "lead_time_delay_prob": 0.04,
        "lead_time_delay_factor": 3.0,
        "node_failure_prob": 0.02,
        "node_failure_duration": 10,
        "cascade_enabled": True,
    }
    
    # ── Mappo Agent Initialization ──
    model_paths = list(mappo_dir.glob("mappo_gnn_*.pt"))
    best_model = sorted(model_paths)[-1]
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chkpt = torch.load(best_model, map_location="cpu", weights_only=False)
        
    config = chkpt.get("config", {})
    obs_dim = config.get("obs_dim", 12)
    
    graph = SupplyChainGraph.from_env_config(3, 2, 1, 3)
    agent_mappo = MAPPOAgent(
        num_agents=6, obs_dim=obs_dim, action_dim=1,
        config=config, adjacency=graph.get_adjacency_matrix().numpy(),
        static_node_features=graph.node_features.numpy()
    )
    
    gnn_state = chkpt["gnn"]
    fixed_gnn_state = {}
    for k, v in gnn_state.items():
        k = k.replace("layers.", "gnn_layers.").replace("norms.", "layer_norms.")
        fixed_gnn_state[k] = v
        
    agent_mappo.gnn.load_state_dict(fixed_gnn_state, strict=False)
    agent_mappo.actor.load_state_dict(chkpt["actor"], strict=False)
    agent_mappo.gnn.eval()
    agent_mappo.actor.eval()
    
    # ── Baseline Agent ──
    agent_ss = SsPoliciy(MultiEchelonInventoryEnv(env_config), s=30, S=100) # Aggressive (s,S)
    
    # ── Extraction ──
    print("Tracing (s,S) trajectory...")
    env1 = DisruptionWrapper(MultiEchelonInventoryEnv(env_config), disruption_config)
    inv_ss, ord_ss = extract_episode_trace(env1, agent_ss, is_mappo=False)
    
    print("Tracing MAPPO trajectory...")
    env2 = DisruptionWrapper(MultiEchelonInventoryEnv(env_config), disruption_config)
    inv_mr, ord_mr = extract_episode_trace(env2, agent_mappo, is_mappo=True)
    
    # ── Save ──
    out = {
        "ss_inventory": inv_ss,
        "ss_orders": ord_ss,
        "mappo_inventory": inv_mr,
        "mappo_orders": ord_mr,
    }
    
    with open(baseline_dir / "explainability_traces.json", "w") as f:
        json.dump(out, f)
        
    print(f"Explanation Trace extraction successful! Saved to {baseline_dir / 'explainability_traces.json'}")
