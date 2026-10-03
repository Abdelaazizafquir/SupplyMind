"""
Baseline Evaluation Script
============================
Runs all classical baselines on the environment and saves results.
This runs locally (no GPU needed).

Usage:
    python notebooks/02_run_baselines.py
"""

import sys
import os
import json
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
from src.environment.disruption_wrapper import DisruptionWrapper
from src.baselines.classical_policies import (
    SsPoliciy, BaseStockPolicy, RandomPolicy, run_baseline_comparison
)
from src.utils.data_processing import generate_synthetic_demand
from src.utils.metrics import SupplyChainMetrics


def plot_baseline_comparison(results: dict, save_path: Path):
    """Create comparison plots for baselines."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    policies = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))
    
    # 1. Total Cost
    costs = [results[p]["metrics"]["total_cost"]["mean"] for p in policies]
    cost_stds = [results[p]["metrics"]["total_cost"]["std"] for p in policies]
    axes[0, 0].barh(policies, costs, xerr=cost_stds, color=colors, edgecolor="white")
    axes[0, 0].set_title("Total Cost (Lower is Better)", fontweight="bold")
    axes[0, 0].set_xlabel("Total Cost ($)")
    
    # 2. Fill Rate
    fills = [results[p]["metrics"]["fill_rate"]["mean"] for p in policies]
    fill_stds = [results[p]["metrics"]["fill_rate"]["std"] for p in policies]
    axes[0, 1].barh(policies, fills, xerr=fill_stds, color=colors, edgecolor="white")
    axes[0, 1].set_title("Fill Rate (Higher is Better)", fontweight="bold")
    axes[0, 1].set_xlabel("Fill Rate")
    axes[0, 1].set_xlim(0, 1.1)
    
    # 3. Stockout Rate
    stockouts = [results[p]["metrics"]["stockout_rate"]["mean"] for p in policies]
    so_stds = [results[p]["metrics"]["stockout_rate"]["std"] for p in policies]
    axes[1, 0].barh(policies, stockouts, xerr=so_stds, color=colors, edgecolor="white")
    axes[1, 0].set_title("Stockout Rate (Lower is Better)", fontweight="bold")
    axes[1, 0].set_xlabel("Stockout Rate")
    
    # 4. Bullwhip Ratio
    bullwhip = [min(results[p]["metrics"]["bullwhip_ratio"]["mean"], 10) for p in policies]
    bw_stds = [min(results[p]["metrics"]["bullwhip_ratio"]["std"], 5) for p in policies]
    axes[1, 1].barh(policies, bullwhip, xerr=bw_stds, color=colors, edgecolor="white")
    axes[1, 1].set_title("Bullwhip Ratio (Closer to 1 is Better)", fontweight="bold")
    axes[1, 1].set_xlabel("Bullwhip Ratio")
    axes[1, 1].axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Ideal")
    axes[1, 1].legend()
    
    plt.suptitle("Baseline Policy Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  [SUCCESS] Comparison plot saved to: {save_path}")


def main():
    results_dir = project_root / "experiments" / "baseline_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  BASELINE EVALUATION")
    print("=" * 60)
    
    # Try to load processed demand, fall back to synthetic
    processed_dir = project_root / "data" / "processed"
    demand_file = processed_dir / "demand_matrix.npy"
    
    if demand_file.exists():
        demand = np.load(demand_file)
        print(f"  Using processed demand data: {demand.shape}")
    else:
        print("  Using synthetic demand data")
        demand = generate_synthetic_demand(num_periods=365, num_retailers=3)
    
    num_retailers = demand.shape[1]
    
    # Create environment
    env_config = {
        "num_retailers": num_retailers,
        "num_distributors": 2,
        "num_suppliers": 1,
        "num_periods": min(365, len(demand)),
        "lead_time_mean": 3,
        "lead_time_std": 1,
        "holding_cost": 1.0,
        "ordering_cost": 50.0,
        "shortage_cost": 10.0,
        "initial_inventory": 100,
        "max_order_quantity": 500,
    }
    
    # === Test 1: Normal conditions ===
    print("\n" + "-" * 40)
    print("  TEST 1: NORMAL CONDITIONS")
    print("-" * 40)
    
    env = MultiEchelonInventoryEnv(env_config)
    env.set_demand_data(demand.copy())
    
    results_normal = run_baseline_comparison(env, num_episodes=5, seed=42)
    
    plot_baseline_comparison(results_normal, results_dir / "baseline_comparison_normal.png")
    
    # === Test 2: With disruptions ===
    print("\n" + "-" * 40)
    print("  TEST 2: WITH DISRUPTIONS")
    print("-" * 40)
    
    env = MultiEchelonInventoryEnv(env_config)
    env.set_demand_data(demand.copy())
    
    disruption_config = {
        "demand_spike_prob": 0.03,
        "demand_spike_factor": 3.0,
        "lead_time_delay_prob": 0.04,
        "lead_time_delay_factor": 3.0,
        "node_failure_prob": 0.02,
        "node_failure_duration": 10,
        "cascade_enabled": True,
    }
    
    env_disrupted = DisruptionWrapper(env, disruption_config)
    results_disrupted = run_baseline_comparison(env_disrupted, num_episodes=5, seed=42)
    
    plot_baseline_comparison(results_disrupted, results_dir / "baseline_comparison_disrupted.png")
    
    # Save numerical results
    def serialize_results(res):
        """Convert numpy types for JSON serialization."""
        serialized = {}
        for policy, data in res.items():
            serialized[policy] = {
                "num_episodes": data["num_episodes"],
                "metrics": {
                    k: {mk: float(mv) for mk, mv in v.items()}
                    for k, v in data["metrics"].items()
                }
            }
        return serialized
    
    with open(results_dir / "results_normal.json", "w") as f:
        json.dump(serialize_results(results_normal), f, indent=2)
    
    with open(results_dir / "results_disrupted.json", "w") as f:
        json.dump(serialize_results(results_disrupted), f, indent=2)
    
    print("\n" + "=" * 60)
    print("  [SUCCESS] BASELINE EVALUATION COMPLETE!")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
