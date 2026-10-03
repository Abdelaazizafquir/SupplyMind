"""
Disruption Resilience: MAPPO+GNN vs Classical Baselines
=========================================================
Tests how each policy handles 3 disruption scenarios:
  1. Demand Spike  — 3x demand at a retailer for 10 days
  2. Supplier Failure — supplier goes offline for 15 days
  3. Combined Crisis — demand spike + lead-time delays simultaneously

This is the key test for the PFE: static baselines break under disruptions
while MAPPO (trained to generalize) should adapt.

Usage:
    python notebooks/06_disruption_comparison.py
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

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.environment.disruption_wrapper import DisruptionWrapper
from src.utils.metrics import SupplyChainMetrics, EpisodeMetrics


# ═══════════════════════════════════════════════════════════
# INLINE KAGGLE MODELS (same as 05_full_comparison.py)
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
        for i, (layer, norm_layer) in enumerate(zip(self.layers, self.norms)):
            h_prev = h
            h = norm_layer(layer(h, adj))
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


# ═══════════════════════════════════════════════════════════
# KAGGLE-MATCHING ENV WRAPPER
# ═══════════════════════════════════════════════════════════

class KaggleMatchingEnv:
    """Wraps env to produce Kaggle-compatible 12-dim obs + linear action scaling."""
    def __init__(self, base_env):
        self.base_env = base_env
        # Access attributes through unwrapped for gym.Wrapper chains
        core = base_env.unwrapped if hasattr(base_env, 'unwrapped') else base_env
        self.num_agents = core.num_agents
        self.agent_ids = core.agent_ids
        self.nr = core.num_retailers
        self.nd = core.num_distributors
        self.ns = core.num_suppliers
        self.last_demand = np.zeros(self.nr)
        self.demand_history = []

    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        self.last_demand = np.zeros(self.nr)
        self.demand_history = []
        return self._transform_obs(obs), info

    def _transform_obs(self, obs):
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
            node = self.base_env.unwrapped.nodes[i] if hasattr(self.base_env, 'unwrapped') else self.base_env.nodes[i]
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
                self.base_env.unwrapped.current_step / self.base_env.unwrapped.num_periods if hasattr(self.base_env, 'unwrapped') else self.base_env.current_step / self.base_env.num_periods,
                1 if i < self.nr else 0,
                1 if self.nr <= i < self.nr + self.nd else 0,
                1 if i >= self.nr + self.nd else 0,
                trend,
                total_retail_demand / (self.nr * 100.0),
                float(len([p for p in node.pipeline if p[1] > 0])) / 5.0,
            ], dtype=np.float32)
        return transformed

    def step(self, actions):
        converted = {}
        for aid in self.agent_ids:
            a = float(actions[aid][0]) if hasattr(actions[aid], '__len__') else float(actions[aid])
            converted[aid] = np.array([np.sqrt(max(0, a))], dtype=np.float32)

        obs, rewards, term, trunc, infos = self.base_env.step(converted)

        base = self.base_env.unwrapped if hasattr(self.base_env, 'unwrapped') else self.base_env
        demand = base._get_demand(base.current_step - 1)
        self.last_demand = demand.copy() if isinstance(demand, np.ndarray) else np.array(demand)
        self.demand_history.append(float(np.mean(self.last_demand)))

        return self._transform_obs(obs), rewards, term, trunc, infos


# ═══════════════════════════════════════════════════════════
# GRAPH / ROLES / CONSTANTS
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


# ═══════════════════════════════════════════════════════════
# DISRUPTION SCENARIOS
# ═══════════════════════════════════════════════════════════

SCENARIOS = {
    "Normal (Baseline)": {
        "disruption_config": {
            "demand_spike_prob": 0.0,
            "lead_time_delay_prob": 0.0,
            "node_failure_prob": 0.0,
            "scheduled_disruptions": [],
        },
        "description": "No disruptions - control scenario",
    },
    "Demand Spike": {
        "disruption_config": {
            "demand_spike_prob": 0.0,
            "lead_time_delay_prob": 0.0,
            "node_failure_prob": 0.0,
            "scheduled_disruptions": [
                {"start_step": 30, "type": 1, "duration": 10,
                 "severity": 3.0, "affected_nodes": [0, 1]},
            ],
        },
        "description": "3x demand at retailers 0,1 from day 30-40",
    },
    "Supplier Failure": {
        "disruption_config": {
            "demand_spike_prob": 0.0,
            "lead_time_delay_prob": 0.0,
            "node_failure_prob": 0.0,
            "cascade_enabled": True,
            "scheduled_disruptions": [
                {"start_step": 25, "type": 3, "duration": 15,
                 "severity": 1.0, "affected_nodes": [5]},
            ],
        },
        "description": "Supplier offline from day 25-40 with cascade",
    },
    "Combined Crisis": {
        "disruption_config": {
            "demand_spike_prob": 0.0,
            "lead_time_delay_prob": 0.0,
            "node_failure_prob": 0.0,
            "cascade_enabled": True,
            "scheduled_disruptions": [
                {"start_step": 20, "type": 1, "duration": 12,
                 "severity": 2.5, "affected_nodes": [0, 1, 2]},
                {"start_step": 25, "type": 2, "duration": 15,
                 "severity": 2.0, "affected_nodes": [0, 1, 2, 3, 4, 5]},
                {"start_step": 35, "type": 3, "duration": 10,
                 "severity": 1.0, "affected_nodes": [3]},
            ],
        },
        "description": "Demand spike + lead-time delays + distributor failure",
    },
}


# ═══════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════

def run_episode(env, get_action_fn, seed=42):
    """Run one episode and return metrics + per-step data."""
    obs, info = env.reset(seed=seed)
    episode = EpisodeMetrics()
    done = False
    step_costs = []
    step_fill = []

    base = env.base_env.unwrapped if hasattr(env, 'base_env') else (env.unwrapped if hasattr(env, 'unwrapped') else env)

    while not done:
        actions = get_action_fn(obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        done = trunc.get("__all__", False) or term.get("__all__", False)

        step_cost = 0.0
        step_demand = 0.0
        step_fulfilled = 0.0

        episode.total_periods += 1
        for agent_id in base.agent_ids:
            ai = infos.get(agent_id, {})
            if "holding_cost" in ai:
                episode.total_holding_cost += ai["holding_cost"]
                step_cost += ai["holding_cost"]
            if "shortage" in ai:
                shortage_cost = ai["shortage"] * base.shortage_cost
                episode.total_shortage_cost += shortage_cost
                step_cost += shortage_cost
                if ai["shortage"] > 0:
                    episode.stockout_periods += 1
            if "demand" in ai:
                episode.total_demand += ai["demand"]
                step_demand += ai["demand"]
            if "fulfilled" in ai:
                episode.total_fulfilled += ai["fulfilled"]
                step_fulfilled += ai["fulfilled"]
            if "order_placed" in ai:
                episode.orders_placed.append(ai["order_placed"])
                if ai["order_placed"] > 0:
                    episode.total_ordering_cost += base.ordering_cost
                    step_cost += base.ordering_cost

        episode.inventory_history.append([node.inventory for node in base.nodes])
        if base.episode_data["demand_history"]:
            last_d = base.episode_data["demand_history"][-1]
            if isinstance(last_d, list):
                episode.demand_history.extend(last_d)

        step_costs.append(step_cost)
        step_fill.append(step_fulfilled / max(step_demand, 1e-8) if step_demand > 0 else 1.0)

    episode.total_cost = (episode.total_holding_cost +
                          episode.total_ordering_cost +
                          episode.total_shortage_cost)
    metrics = SupplyChainMetrics.compute_all(episode)
    return metrics, step_costs, step_fill


def evaluate_policy(policy_name, get_action_fn, env_factory, num_episodes=10, seeds=None):
    """Evaluate a policy across episodes, collect aggregate + per-step data."""
    all_metrics = []
    all_step_costs = []
    all_step_fill = []

    for ep in range(num_episodes):
        seed = seeds[ep] if seeds else 42 + ep
        env = env_factory()
        metrics, step_costs, step_fill = run_episode(env, get_action_fn, seed=seed)
        all_metrics.append(metrics)
        all_step_costs.append(step_costs)
        all_step_fill.append(step_fill)

    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    # Average per-step curves
    min_len = min(len(sc) for sc in all_step_costs)
    avg_step_costs = np.mean([sc[:min_len] for sc in all_step_costs], axis=0)
    avg_step_fill = np.mean([sf[:min_len] for sf in all_step_fill], axis=0)

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "metrics": aggregated,
        "step_costs": avg_step_costs.tolist(),
        "step_fill": avg_step_fill.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# BASELINE POLICIES
# ═══════════════════════════════════════════════════════════

def make_ss_action_fn(s, S, env_ref):
    """(s,S) policy returning action dict."""
    def action_fn(obs):
        base = env_ref[0].base_env.unwrapped if hasattr(env_ref[0], 'base_env') else (env_ref[0].unwrapped if hasattr(env_ref[0], 'unwrapped') else env_ref[0])
        actions = {}
        for i, aid in enumerate(base.agent_ids):
            node = base.nodes[i]
            if node.inventory < s:
                order_qty = max(0, S - node.inventory)
                action_val = np.sqrt(min(order_qty / node.capacity, 1.0))
                actions[aid] = np.array([action_val], dtype=np.float32)
            else:
                actions[aid] = np.array([0.0], dtype=np.float32)
        return actions
    return action_fn


def make_basestock_action_fn(target, env_ref):
    """Base-stock policy returning action dict."""
    def action_fn(obs):
        base = env_ref[0].base_env.unwrapped if hasattr(env_ref[0], 'base_env') else (env_ref[0].unwrapped if hasattr(env_ref[0], 'unwrapped') else env_ref[0])
        actions = {}
        for i, aid in enumerate(base.agent_ids):
            node = base.nodes[i]
            order_qty = max(0, target - node.inventory)
            action_val = np.sqrt(min(order_qty / node.capacity, 1.0))
            actions[aid] = np.array([action_val], dtype=np.float32)
        return actions
    return action_fn


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    results_dir = project_root / "experiments" / "disruption_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  DISRUPTION RESILIENCE: MAPPO+GNN vs BASELINES")
    print("=" * 65)

    # ── Demand ──
    TARGET_MEAN = 25.0
    NUM_PERIODS = 100
    np.random.seed(42)
    t = np.arange(NUM_PERIODS)
    demand = np.zeros((NUM_PERIODS, NUM_RETAILERS))
    for r in range(NUM_RETAILERS):
        base_d = TARGET_MEAN * (1 + 0.2 * np.random.random())
        demand[:, r] = np.maximum(0, base_d + 3*np.sin(2*np.pi*t/7) +
                                   5*np.sin(2*np.pi*t/365) + np.random.normal(0, 5, NUM_PERIODS))
    print(f"\n  Demand: mean={demand.mean():.1f}/retailer/day, shape={demand.shape}")

    # ── Common config ──
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

    # ── Load MAPPO model ──
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
    gnn.eval()
    actor.eval()
    print(f"  Loaded MAPPO model: hidden_dim={HIDDEN_DIM}")

    # ── MAPPO action function ──
    def mappo_action_fn(obs):
        obs_list = [torch.tensor(obs[f"agent_{i}"], dtype=torch.float32) for i in range(NUM_AGENTS)]
        obs_t = torch.stack(obs_list).to(device)
        gnn_in = torch.cat([obs_t, static_t], dim=-1)
        with torch.no_grad():
            gnn_out = gnn(gnn_in, adj_t)
            aug = torch.cat([obs_t, gnn_out], dim=-1)
            actions = {}
            for i in range(NUM_AGENTS):
                action, _ = actor.get_action(aug[i], roles_t[i], deterministic=True)
                actions[f"agent_{i}"] = action.cpu().numpy()
        return actions

    # ── Define policies ──
    # Best classical baselines from 05_full_comparison: (s,S) Conservative s=130,S=350 and Base-Stock 400
    POLICIES = ["MAPPO+GNN", "(s,S) Conservative", "Base-Stock 400"]

    # ═══════════ RUN ALL SCENARIOS ═══════════
    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        print(f"\n{'='*65}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"  {scenario_cfg['description']}")
        print(f"{'='*65}")

        scenario_results = {}

        for policy_name in POLICIES:
            print(f"\n  >> {policy_name}...", end=" ", flush=True)

            # Create a mutable ref so the action fn can access current env
            env_ref = [None]

            def make_env(pname=policy_name):
                base = MultiEchelonInventoryEnv(env_config)
                base.set_demand_data(demand.copy())
                wrapped = DisruptionWrapper(base, scenario_cfg["disruption_config"])
                if pname == "MAPPO+GNN":
                    kaggle_env = KaggleMatchingEnv(wrapped)
                    env_ref[0] = kaggle_env
                    return kaggle_env
                else:
                    env_ref[0] = wrapped
                    return wrapped

            if policy_name == "MAPPO+GNN":
                action_fn = mappo_action_fn
            elif policy_name == "(s,S) Conservative":
                action_fn = make_ss_action_fn(130, 350, env_ref)
            elif policy_name == "Base-Stock 400":
                action_fn = make_basestock_action_fn(400, env_ref)

            result = evaluate_policy(
                policy_name, action_fn,
                env_factory=make_env,
                num_episodes=NUM_EVAL, seeds=seeds,
            )
            scenario_results[policy_name] = result
            m = result["metrics"]
            print(f"Cost=${m['total_cost']['mean']:>10,.0f}  Fill={m['fill_rate']['mean']:.1%}  "
                  f"Stockout={m['stockout_rate']['mean']:.2%}")

        all_results[scenario_name] = scenario_results

    # ═══════════ SAVE RESULTS ═══════════
    save_data = {}
    for scenario_name, scenario_results in all_results.items():
        save_data[scenario_name] = {}
        for pname, result in scenario_results.items():
            save_data[scenario_name][pname] = {
                "policy": result["policy"],
                "metrics": result["metrics"],
            }
    with open(results_dir / "disruption_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # ═══════════ SUMMARY TABLE ═══════════
    print(f"\n\n{'='*90}")
    print(f"  DISRUPTION RESILIENCE SUMMARY")
    print(f"{'='*90}")

    header = f"  {'Scenario':<22} {'Policy':<22} {'Cost ($)':>12} {'Fill Rate':>10} {'Stockout':>10}"
    print(header)
    print(f"  {'='*22} {'='*22} {'='*12} {'='*10} {'='*10}")

    for scenario_name in SCENARIOS:
        sr = all_results[scenario_name]
        for i, pname in enumerate(POLICIES):
            m = sr[pname]["metrics"]
            s_label = scenario_name if i == 0 else ""
            print(f"  {s_label:<22} {pname:<22} ${m['total_cost']['mean']:>10,.0f} "
                  f"{m['fill_rate']['mean']:>9.1%} {m['stockout_rate']['mean']:>9.2%}")
        print(f"  {'-'*22} {'-'*22} {'-'*12} {'-'*10} {'-'*10}")

    # ═══════════ DEGRADATION TABLE ═══════════
    print(f"\n\n{'='*80}")
    print(f"  RESILIENCE SCORE (% degradation from Normal baseline)")
    print(f"{'='*80}")
    print(f"  {'Policy':<22} {'Demand Spike':>15} {'Supplier Fail':>15} {'Combined':>15}")
    print(f"  {'='*22} {'='*15} {'='*15} {'='*15}")

    for pname in POLICIES:
        normal_fill = all_results["Normal (Baseline)"][pname]["metrics"]["fill_rate"]["mean"]
        vals = []
        for scenario_name in ["Demand Spike", "Supplier Failure", "Combined Crisis"]:
            disrupted_fill = all_results[scenario_name][pname]["metrics"]["fill_rate"]["mean"]
            degradation = (normal_fill - disrupted_fill) / max(normal_fill, 1e-8) * 100
            vals.append(degradation)
        print(f"  {pname:<22} {vals[0]:>13.1f}% {vals[1]:>13.1f}% {vals[2]:>13.1f}%")

    # ═══════════ PLOTS ═══════════
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'ytick.labelsize': 16})
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    scenario_list = list(SCENARIOS.keys())
    policy_colors = {"MAPPO+GNN": "#E53935", "(s,S) Conservative": "#42A5F5", "Base-Stock 400": "#66BB6A"}

    # Panel 1: Fill Rate across scenarios (grouped bar)
    ax = axes[0, 0]
    x = np.arange(len(scenario_list))
    width = 0.25
    for j, pname in enumerate(POLICIES):
        fills = [all_results[s][pname]["metrics"]["fill_rate"]["mean"] for s in scenario_list]
        stds = [all_results[s][pname]["metrics"]["fill_rate"]["std"] for s in scenario_list]
        bars = ax.bar(x + j*width, fills, width, label=pname, color=policy_colors[pname],
                      yerr=stds, capsize=3, edgecolor="white")
        for k, v in enumerate(fills):
            ax.text(x[k] + j*width, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=18)
    ax.set_ylabel("Fill Rate")
    ax.set_title("Fill Rate by Scenario", fontweight="bold", fontsize=18)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(" (Baseline)", "") for s in scenario_list], fontsize=16)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.4, label='95% target')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=16)

    # Panel 2: Total Cost across scenarios
    ax = axes[0, 1]
    for j, pname in enumerate(POLICIES):
        costs = [all_results[s][pname]["metrics"]["total_cost"]["mean"] for s in scenario_list]
        stds = [all_results[s][pname]["metrics"]["total_cost"]["std"] for s in scenario_list]
        ax.bar(x + j*width, costs, width, label=pname, color=policy_colors[pname],
               yerr=stds, capsize=3, edgecolor="white")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Total Cost by Scenario", fontweight="bold", fontsize=18)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(" (Baseline)", "") for s in scenario_list], fontsize=16)
    ax.legend(fontsize=16)

    # Panel 3: Fill Rate over time for Combined Crisis
    ax = axes[1, 0]
    combined = all_results["Combined Crisis"]
    for pname in POLICIES:
        step_fill = combined[pname]["step_fill"]
        # Smooth with window of 3
        fill_smooth = np.convolve(step_fill, np.ones(3)/3, mode='valid')
        ax.plot(fill_smooth, label=pname, color=policy_colors[pname], linewidth=2)
    # Mark disruption events
    ax.axvspan(20, 32, alpha=0.1, color='orange', label='Demand spike')
    ax.axvspan(25, 40, alpha=0.1, color='red', label='Lead-time delay')
    ax.axvspan(35, 45, alpha=0.1, color='purple', label='Distributor failure')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.4)
    ax.set_xlabel("Day")
    ax.set_ylabel("Fill Rate")
    ax.set_title("Fill Rate Over Time - Combined Crisis", fontweight="bold", fontsize=18)
    ax.legend(fontsize=16, loc='lower left')
    ax.set_ylim(0, 1.1)

    # Panel 4: Degradation comparison (bar chart)
    ax = axes[1, 1]
    disruption_names = ["Demand\nSpike", "Supplier\nFailure", "Combined\nCrisis"]
    x2 = np.arange(3)
    for j, pname in enumerate(POLICIES):
        normal_fill = all_results["Normal (Baseline)"][pname]["metrics"]["fill_rate"]["mean"]
        degs = []
        for sn in ["Demand Spike", "Supplier Failure", "Combined Crisis"]:
            d_fill = all_results[sn][pname]["metrics"]["fill_rate"]["mean"]
            degs.append((normal_fill - d_fill) / max(normal_fill, 1e-8) * 100)
        ax.bar(x2 + j*width, degs, width, label=pname, color=policy_colors[pname], edgecolor="white")
        for k, v in enumerate(degs):
            ax.text(x2[k] + j*width, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=18)
    ax.set_ylabel("Fill Rate Degradation (%)")
    ax.set_title("Resilience: Fill Rate Degradation from Normal", fontweight="bold", fontsize=18)
    ax.set_xticks(x2 + width)
    ax.set_xticklabels(disruption_names, fontsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim(bottom=0)

    plt.suptitle("Disruption Resilience Analysis: MAPPO+GNN vs Classical Baselines",
                 fontsize=24, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "disruption_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Results saved to: {results_dir}")
    print(f"  Plot saved to: {results_dir / 'disruption_comparison.png'}")


if __name__ == "__main__":
    main()
