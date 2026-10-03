"""
Full Comparison: MAPPO+GNN vs Classical Baselines
===================================================
Runs EVERYTHING on the SAME environment with normalized demand.
Produces the final comparison table and plots for the PFE report.

Usage:
    python notebooks/05_full_comparison.py
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Add project root
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.environment.disruption_wrapper import DisruptionWrapper
from src.utils.metrics import SupplyChainMetrics, EpisodeMetrics

# ═══════════════════════════════════════════════════════════
# INLINE KAGGLE MODELS (to avoid local/Kaggle architecture mismatch)
# These match EXACTLY what was trained on Kaggle
# ═══════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x, adj):
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj_hat.sum(dim=1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt.unsqueeze(1) * adj_hat * deg_inv_sqrt.unsqueeze(0)
        if x.dim() == 3:
            return torch.einsum('ij,bjk->bik', norm, self.linear(x))
        return torch.mm(norm, self.linear(x))


class GNNCommunication(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(GCNLayer(in_d, out_d))
            self.norms.append(nn.LayerNorm(out_d))
    
    def forward(self, x, adj):
        h = F.relu(self.input_proj(x))
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_prev = h
            h = norm(layer(h, adj))
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if h.shape == h_prev.shape:
                    h = h + h_prev
        return h


NUM_ROLE_TYPES = 3

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=1, hidden_dim=256, num_roles=NUM_ROLE_TYPES):
        super().__init__()
        total_in = obs_dim + num_roles
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs, role):
        x = torch.cat([obs, role], dim=-1)
        features = self.net(x)
        mean = torch.sigmoid(self.mean_head(features))
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std
    
    def get_action(self, obs, role, deterministic=False):
        mean, std = self.forward(obs, role)
        if deterministic:
            return mean, torch.zeros(1)
        dist = Normal(mean, std)
        action = torch.clamp(dist.sample(), 0.0, 1.0)
        return action, dist.log_prob(action).sum(-1)


class CriticNetwork(nn.Module):
    def __init__(self, per_agent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(per_agent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, agent_states):
        if agent_states.dim() == 2:
            agent_states = agent_states.unsqueeze(0)
        encoded = self.encoder(agent_states)
        pooled = encoded.mean(dim=1)
        return self.value_head(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════
# KAGGLE-MATCHING ENVIRONMENT (obs layout must match training)
# ═══════════════════════════════════════════════════════════

class KaggleMatchingEnv:
    """
    Wraps MultiEchelonInventoryEnv to produce observations matching 
    the Kaggle training env's obs layout (12-dim with demand signals).
    Also uses linear action scaling (no squaring) matching Kaggle env.
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.num_agents = base_env.num_agents
        self.agent_ids = base_env.agent_ids
        self.nr = base_env.num_retailers
        self.nd = base_env.num_distributors
        self.ns = base_env.num_suppliers
        self.last_demand = np.zeros(self.nr)
        self.demand_history = []
    
    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        self.last_demand = np.zeros(self.nr)
        self.demand_history = []
        return self._transform_obs(obs), info
    
    def _transform_obs(self, obs):
        """Transform local env obs to Kaggle's 12-dim format."""
        total_retail_demand = float(np.sum(self.last_demand))
        avg_demand = np.mean(self.demand_history[-5:]) if len(self.demand_history) > 0 else 0.0
        if len(self.demand_history) >= 10:
            recent = np.mean(self.demand_history[-5:])
            older = np.mean(self.demand_history[-10:-5])
            trend = (recent - older) / (older + 1e-8)
        else:
            trend = 0.0
        
        transformed = {}
        for i, aid in enumerate(self.agent_ids):
            raw = obs[aid]
            node = self.base_env.nodes[i]
            pipe_total = sum(q for _, q in node.pipeline)
            
            if i < self.nr:
                local_demand = self.last_demand[i] / 100.0
            else:
                local_demand = total_retail_demand / (self.nr * 100.0)
            
            capacity = 1000 if i < self.nr else (2500 if i < self.nr + self.nd else 5000)
            inv_position = (node.inventory + pipe_total) / capacity
            
            transformed[aid] = np.array([
                node.inventory / 1000,
                pipe_total / 1000,
                local_demand,
                inv_position,
                avg_demand / 100.0,
                self.base_env.current_step / self.base_env.num_periods,
                1 if i < self.nr else 0,
                1 if self.nr <= i < self.nr + self.nd else 0,
                1 if i >= self.nr + self.nd else 0,
                trend,
                total_retail_demand / (self.nr * 100.0),
                float(len([p for p in node.pipeline if p[1] > 0])) / 5.0,
            ], dtype=np.float32)
        return transformed
    
    def step(self, actions):
        """Step with LINEAR action scaling (matching Kaggle env)."""
        # Convert actions: Kaggle uses linear target_inv = a * capacity
        # But local env uses SQUARED: target_inv = (a²) * capacity
        # So we need to sqrt the action to compensate
        converted_actions = {}
        for i, aid in enumerate(self.agent_ids):
            a = float(actions[aid][0]) if hasattr(actions[aid], '__len__') else float(actions[aid])
            # Local env does (a²)*capacity, Kaggle does a*capacity
            # To get same target: sqrt(a) into local env → (sqrt(a))² * cap = a * cap
            converted_actions[aid] = np.array([np.sqrt(max(0, a))], dtype=np.float32)
        
        obs, rewards, term, trunc, infos = self.base_env.step(converted_actions)
        
        # Track demand for next obs
        demand = self.base_env._get_demand(self.base_env.current_step - 1)
        self.last_demand = demand.copy() if isinstance(demand, np.ndarray) else np.array(demand)
        self.demand_history.append(float(np.mean(self.last_demand)))
        
        return self._transform_obs(obs), rewards, term, trunc, infos


# ═══════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════

NUM_RETAILERS = 3
NUM_DISTRIBUTORS = 2
NUM_SUPPLIERS = 1
NUM_AGENTS = 6
OBS_DIM = 12
GNN_OUTPUT_DIM = 64

def build_graph():
    n = NUM_AGENTS
    adj = np.zeros((n, n), dtype=np.float32)
    for r in range(NUM_RETAILERS):
        d = NUM_RETAILERS + (r % NUM_DISTRIBUTORS)
        adj[r, d] = adj[d, r] = 1.0
    for d_idx in range(NUM_DISTRIBUTORS):
        s = NUM_RETAILERS + NUM_DISTRIBUTORS + (d_idx % NUM_SUPPLIERS)
        adj[NUM_RETAILERS + d_idx, s] = adj[s, NUM_RETAILERS + d_idx] = 1.0
    
    static = []
    for i in range(n):
        if i < NUM_RETAILERS:
            static.append([1, 0, 0, 0.0])
        elif i < NUM_RETAILERS + NUM_DISTRIBUTORS:
            static.append([0, 1, 0, 0.5])
        else:
            static.append([0, 0, 1, 1.0])
    return adj, np.array(static, dtype=np.float32)


def get_roles(device):
    roles = []
    for _ in range(NUM_RETAILERS): roles.append([1, 0, 0])
    for _ in range(NUM_DISTRIBUTORS): roles.append([0, 1, 0])
    for _ in range(NUM_SUPPLIERS): roles.append([0, 0, 1])
    return torch.tensor(roles, dtype=torch.float32).to(device)


def evaluate_mappo(env_wrapper, gnn, actor, roles_t, static_t, adj_t, device, 
                   num_episodes=10, seeds=None):
    """Evaluate MAPPO agent on the wrapped env."""
    gnn.eval()
    actor.eval()
    
    all_metrics = []
    
    for ep in range(num_episodes):
        seed = seeds[ep] if seeds else 42 + ep
        obs, _ = env_wrapper.reset(seed=seed)
        episode = EpisodeMetrics()
        done = False
        
        while not done:
            obs_list = [torch.tensor(obs[f"agent_{i}"], dtype=torch.float32) for i in range(NUM_AGENTS)]
            obs_t = torch.stack(obs_list).to(device)
            gnn_in = torch.cat([obs_t, static_t], dim=-1)
            
            with torch.no_grad():
                gnn_out = gnn(gnn_in, adj_t)
                aug = torch.cat([obs_t, gnn_out], dim=-1)
                
                actions_dict = {}
                for i in range(NUM_AGENTS):
                    action, _ = actor.get_action(aug[i], roles_t[i], deterministic=True)
                    actions_dict[f"agent_{i}"] = action.cpu().numpy()
            
            obs, rewards, term, trunc, infos = env_wrapper.step(actions_dict)
            done = trunc.get("__all__", False) or term.get("__all__", False)
            
            episode.total_periods += 1
            base = env_wrapper.base_env
            for agent_id in base.agent_ids:
                ai = infos.get(agent_id, {})
                if "holding_cost" in ai:
                    episode.total_holding_cost += ai["holding_cost"]
                if "shortage" in ai:
                    episode.total_shortage_cost += ai["shortage"] * base.shortage_cost
                    if ai["shortage"] > 0:
                        episode.stockout_periods += 1
                if "demand" in ai:
                    episode.total_demand += ai["demand"]
                if "fulfilled" in ai:
                    episode.total_fulfilled += ai["fulfilled"]
                if "order_placed" in ai:
                    episode.orders_placed.append(ai["order_placed"])
                    if ai["order_placed"] > 0:
                        episode.total_ordering_cost += base.ordering_cost
            
            episode.inventory_history.append(
                [node.inventory for node in base.nodes]
            )
            if base.episode_data["demand_history"]:
                last_d = base.episode_data["demand_history"][-1]
                if isinstance(last_d, list):
                    episode.demand_history.extend(last_d)
        
        episode.total_cost = (episode.total_holding_cost + 
                              episode.total_ordering_cost + 
                              episode.total_shortage_cost)
        metrics = SupplyChainMetrics.compute_all(episode)
        all_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    
    return {"policy": "MAPPO+GNN", "num_episodes": num_episodes, "metrics": aggregated}


def evaluate_baseline(env, policy_fn, name, num_episodes=10, seeds=None):
    """Evaluate a baseline policy."""
    all_metrics = []
    base = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    for ep in range(num_episodes):
        seed = seeds[ep] if seeds else 42 + ep
        obs, info = env.reset(seed=seed)
        episode = EpisodeMetrics()
        done = False
        
        while not done:
            actions = policy_fn(obs, base)
            obs, rewards, term, trunc, info = env.step(actions)
            done = trunc.get("__all__", False) or term.get("__all__", False)
            
            episode.total_periods += 1
            for agent_id in base.agent_ids:
                ai = info.get(agent_id, {})
                if "holding_cost" in ai:
                    episode.total_holding_cost += ai["holding_cost"]
                if "shortage" in ai:
                    episode.total_shortage_cost += ai["shortage"] * base.shortage_cost
                    if ai["shortage"] > 0:
                        episode.stockout_periods += 1
                if "demand" in ai:
                    episode.total_demand += ai["demand"]
                if "fulfilled" in ai:
                    episode.total_fulfilled += ai["fulfilled"]
                if "order_placed" in ai:
                    episode.orders_placed.append(ai["order_placed"])
                    if ai["order_placed"] > 0:
                        episode.total_ordering_cost += base.ordering_cost
            
            episode.inventory_history.append(
                [node.inventory for node in base.nodes]
            )
            if base.episode_data["demand_history"]:
                last_d = base.episode_data["demand_history"][-1]
                if isinstance(last_d, list):
                    episode.demand_history.extend(last_d)
        
        episode.total_cost = (episode.total_holding_cost + 
                              episode.total_ordering_cost + 
                              episode.total_shortage_cost)
        metrics = SupplyChainMetrics.compute_all(episode)
        all_metrics.append(metrics)
    
    aggregated = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    
    return {"policy": name, "num_episodes": num_episodes, "metrics": aggregated}


# ═══════════════════════════════════════════════════════════
# BASELINE POLICIES (using env's native action space with squaring)
# ═══════════════════════════════════════════════════════════

def ss_policy(s, S):
    """(s,S) policy: order up to S when inventory < s."""
    def policy_fn(obs, env):
        actions = {}
        for i, aid in enumerate(env.agent_ids):
            node = env.nodes[i]
            if node.inventory < s:
                order_qty = max(0, S - node.inventory)
                action_val = np.sqrt(min(order_qty / node.capacity, 1.0))
                actions[aid] = np.array([action_val], dtype=np.float32)
            else:
                actions[aid] = np.array([0.0], dtype=np.float32)
        return actions
    return policy_fn


def basestock_policy(target):
    """Base-stock: order up to target each period."""
    def policy_fn(obs, env):
        actions = {}
        for i, aid in enumerate(env.agent_ids):
            node = env.nodes[i]
            order_qty = max(0, target - node.inventory)
            action_val = np.sqrt(min(order_qty / node.capacity, 1.0))
            actions[aid] = np.array([action_val], dtype=np.float32)
        return actions
    return policy_fn


def random_policy(obs, env):
    return {aid: np.array([np.random.random()], dtype=np.float32) for aid in env.agent_ids}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    results_dir = project_root / "experiments" / "final_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  FULL COMPARISON: MAPPO+GNN vs BASELINES")
    print("=" * 60)
    
    # -- Generate demand (same as Kaggle, normalized to ~25/retailer) --
    TARGET_MEAN = 25.0
    NUM_PERIODS = 100
    np.random.seed(42)
    t = np.arange(NUM_PERIODS)
    demand = np.zeros((NUM_PERIODS, NUM_RETAILERS))
    for r in range(NUM_RETAILERS):
        base = TARGET_MEAN * (1 + 0.2 * np.random.random())
        demand[:, r] = np.maximum(0, base + 3*np.sin(2*np.pi*t/7) + 
                                   5*np.sin(2*np.pi*t/365) + np.random.normal(0, 5, NUM_PERIODS))
    
    print(f"\n  Demand: mean={demand.mean():.1f}/retailer/day, shape={demand.shape}")
    
    # -- Common env config --
    env_config = {
        "num_retailers": NUM_RETAILERS,
        "num_distributors": NUM_DISTRIBUTORS,
        "num_suppliers": NUM_SUPPLIERS,
        "num_periods": NUM_PERIODS,
        "holding_cost": 1.0,
        "ordering_cost": 50.0,
        "shortage_cost": 500.0,
        "initial_inventory": 100,
        "max_order_quantity": 500,
    }
    
    NUM_EVAL = 10
    seeds = list(range(42, 42 + NUM_EVAL))
    
    # -- Load MAPPO model --
    model_path = project_root / "models" / "agents" / "mappo_gnn_final.pt"
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        return
    
    device = torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chkpt = torch.load(model_path, map_location=device, weights_only=False)
    
    config = chkpt.get("config", {})
    HIDDEN_DIM = config.get("hidden_dim", 512)
    
    adj, static_feat = build_graph()
    adj_t = torch.tensor(adj).to(device)
    static_t = torch.tensor(static_feat).to(device)
    roles_t = get_roles(device)
    
    gnn_input_dim = OBS_DIM + static_feat.shape[1]
    actor_obs_dim = OBS_DIM + GNN_OUTPUT_DIM
    
    gnn = GNNCommunication(gnn_input_dim, 64, GNN_OUTPUT_DIM, 2).to(device)
    actor = ActorNetwork(actor_obs_dim, 1, HIDDEN_DIM).to(device)
    
    gnn.load_state_dict(chkpt["gnn"])
    actor.load_state_dict(chkpt["actor"])
    print(f"  Loaded MAPPO model from {model_path.name}")
    print(f"  Config: hidden_dim={HIDDEN_DIM}, obs_dim={OBS_DIM}")
    
    # ═══════════ EVALUATE ALL ═══════════
    all_results = {}
    
    # --- MAPPO+GNN ---
    print(f"\n{'-'*50}")
    print(f"  Evaluating: MAPPO+GNN (trained 1500 episodes)")
    print(f"{'-'*50}")
    
    env_mappo = MultiEchelonInventoryEnv(env_config)
    env_mappo.set_demand_data(demand.copy())
    wrapper = KaggleMatchingEnv(env_mappo)
    
    mappo_result = evaluate_mappo(wrapper, gnn, actor, roles_t, static_t, adj_t, device,
                                   num_episodes=NUM_EVAL, seeds=seeds)
    all_results["MAPPO+GNN"] = mappo_result
    m = mappo_result["metrics"]
    print(f"  Total Cost:  ${m['total_cost']['mean']:>12,.0f} ± ${m['total_cost']['std']:>8,.0f}")
    print(f"  Fill Rate:    {m['fill_rate']['mean']:>10.2%} ± {m['fill_rate']['std']:>7.2%}")
    print(f"  Stockout:     {m['stockout_rate']['mean']:>10.2%}")
    print(f"  Shortage $:   ${m['shortage_cost']['mean']:>10,.0f}")
    
    # --- Baselines ---
    baselines = [
        ("(s,S) Tuned", ss_policy(100, 300)),
        ("(s,S) Conservative", ss_policy(130, 350)),
        ("(s,S) Aggressive", ss_policy(80, 250)),
        ("Base-Stock 200", basestock_policy(200)),
        ("Base-Stock 300", basestock_policy(300)),
        ("Base-Stock 400", basestock_policy(400)),
        ("Random", random_policy),
    ]
    
    for name, policy_fn in baselines:
        print(f"\n{'-'*50}")
        print(f"  Evaluating: {name}")
        print(f"{'-'*50}")
        
        env_bl = MultiEchelonInventoryEnv(env_config)
        env_bl.set_demand_data(demand.copy())
        
        result = evaluate_baseline(env_bl, policy_fn, name, num_episodes=NUM_EVAL, seeds=seeds)
        all_results[name] = result
        m = result["metrics"]
        print(f"  Total Cost:  ${m['total_cost']['mean']:>12,.0f} ± ${m['total_cost']['std']:>8,.0f}")
        print(f"  Fill Rate:    {m['fill_rate']['mean']:>10.2%} ± {m['fill_rate']['std']:>7.2%}")
        print(f"  Stockout:     {m['stockout_rate']['mean']:>10.2%}")
        print(f"  Shortage $:   ${m['shortage_cost']['mean']:>10,.0f}")
    
    # ═══════════ SAVE RESULTS ═══════════
    with open(results_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # ═══════════ PLOTS ═══════════
    policies = list(all_results.keys())
    
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'ytick.labelsize': 16})
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    colors = ['#E53935' if p == 'MAPPO+GNN' else '#42A5F5' for p in policies]
    
    # 1. Total Cost
    ax = axes[0, 0]
    costs = [all_results[p]["metrics"]["total_cost"]["mean"] for p in policies]
    cost_std = [all_results[p]["metrics"]["total_cost"]["std"] for p in policies]
    bars = ax.barh(policies, costs, xerr=cost_std, color=colors, edgecolor="white")
    ax.set_title("Total Cost (Lower = Better)", fontweight="bold", fontsize=18)
    ax.set_xlabel("Total Cost ($)")
    for i, (c, p) in enumerate(zip(costs, policies)):
        ax.text(c + max(costs)*0.01, i, f'${c:,.0f}', va='center', fontsize=18)
    
    # 2. Fill Rate
    ax = axes[0, 1]
    fills = [all_results[p]["metrics"]["fill_rate"]["mean"] for p in policies]
    fill_std = [all_results[p]["metrics"]["fill_rate"]["std"] for p in policies]
    ax.barh(policies, fills, xerr=fill_std, color=colors, edgecolor="white")
    ax.set_title("Fill Rate (Higher = Better)", fontweight="bold", fontsize=18)
    ax.set_xlabel("Fill Rate")
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, label='95% target')
    ax.legend(fontsize=14)
    for i, f in enumerate(fills):
        ax.text(f + 0.01, i, f'{f:.1%}', va='center', fontsize=18)
    
    # 3. Stockout Rate
    ax = axes[1, 0]
    stockouts = [all_results[p]["metrics"]["stockout_rate"]["mean"] for p in policies]
    ax.barh(policies, stockouts, color=colors, edgecolor="white")
    ax.set_title("Stockout Rate (Lower = Better)", fontweight="bold", fontsize=18)
    ax.set_xlabel("Stockout Rate")
    for i, s in enumerate(stockouts):
        ax.text(s + 0.01, i, f'{s:.2%}', va='center', fontsize=18)
    
    # 4. Shortage Cost
    ax = axes[1, 1]
    shortage = [all_results[p]["metrics"]["shortage_cost"]["mean"] for p in policies]
    shortage_std = [all_results[p]["metrics"]["shortage_cost"]["std"] for p in policies]
    ax.barh(policies, shortage, xerr=shortage_std, color=colors, edgecolor="white")
    ax.set_title("Shortage Cost (Lower = Better)", fontweight="bold", fontsize=18)
    ax.set_xlabel("Shortage Cost ($)")
    for i, s in enumerate(shortage):
        ax.text(s + max(shortage)*0.01, i, f'${s:,.0f}', va='center', fontsize=18)
    
    plt.suptitle("MAPPO+GNN vs Classical Baselines — Supply Chain Optimization",
                 fontsize=24, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "full_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ═══════════ SUMMARY TABLE ═══════════
    print(f"\n\n{'='*90}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Policy':<22} {'Cost ($)':>14} {'Fill Rate':>12} {'Stockout':>12} {'Shortage $':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*12} {'-'*12} {'-'*14}")
    
    for p in policies:
        m = all_results[p]["metrics"]
        marker = " *" if p == "MAPPO+GNN" else ""
        print(f"  {p:<22} ${m['total_cost']['mean']:>12,.0f} "
              f"{m['fill_rate']['mean']:>11.1%} "
              f"{m['stockout_rate']['mean']:>11.2%} "
              f"${m['shortage_cost']['mean']:>12,.0f}{marker}")
    
    print(f"\n  Results saved to: {results_dir}")
    print(f"  Plot saved to: {results_dir / 'full_comparison.png'}")


if __name__ == "__main__":
    main()
