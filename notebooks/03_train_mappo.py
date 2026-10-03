"""
MAPPO+GNN Training Loop
=========================
Full training script that can run locally (CPU) or on Kaggle (GPU).

Usage:
    python notebooks/03_train_mappo.py --config configs/default_config.yaml
    
For Kaggle: upload as notebook with GPU accelerator.
"""

import sys
import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.environment.disruption_wrapper import DisruptionWrapper
from src.agents.mappo import MAPPOAgent
from src.agents.graph import SupplyChainGraph
from src.utils.data_processing import generate_synthetic_demand
from src.utils.metrics import SupplyChainMetrics, EpisodeMetrics
from src.utils.config import Config


def train_mappo(
    config: Config = None,
    total_episodes: int = 1000,
    rollout_length: int = 128,
    eval_interval: int = 25,
    save_interval: int = 100,
    output_dir: str = None,
):
    """
    Main MAPPO training loop.
    
    Args:
        config: Configuration object
        total_episodes: Total training episodes
        rollout_length: Steps per rollout before update
        eval_interval: Episodes between evaluations
        save_interval: Episodes between model saves
        output_dir: Directory to save results
    """
    cfg = config or Config()
    out_dir = Path(output_dir) if output_dir else project_root / "experiments" / "mappo_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Seeds
    np.random.seed(cfg.seed)
    
    print("=" * 60)
    print("  MAPPO+GNN TRAINING")
    print("=" * 60)
    print(f"  Device: {cfg.get_device()}")
    print(f"  Episodes: {total_episodes}")
    print(f"  GNN: {'Enabled' if cfg.mappo.use_gnn else 'Disabled'}")
    print()
    
    # ── Environment Setup ──
    env_config = {
        "num_retailers": cfg.environment.num_retailers,
        "num_distributors": cfg.environment.num_distributors,
        "num_suppliers": cfg.environment.num_suppliers,
        "num_periods": cfg.environment.num_periods,
        "lead_time_mean": cfg.environment.lead_time_mean,
        "lead_time_std": cfg.environment.lead_time_std,
        "holding_cost": cfg.environment.holding_cost,
        "ordering_cost": cfg.environment.ordering_cost,
        "shortage_cost": cfg.environment.shortage_cost,
        "initial_inventory": cfg.environment.initial_inventory,
        "max_order_quantity": cfg.environment.max_order_quantity,
    }
    
    env = MultiEchelonInventoryEnv(env_config)
    
    # Load demand data
    processed_dir = project_root / "data" / "processed"
    demand_file = processed_dir / "demand_matrix.npy"
    
    if demand_file.exists():
        demand_data = np.load(demand_file)
        print(f"  Loaded processed demand: {demand_data.shape}")
    else:
        demand_data = generate_synthetic_demand(
            num_periods=cfg.environment.num_periods,
            num_retailers=cfg.environment.num_retailers,
        )
        print(f"  Using synthetic demand: {demand_data.shape}")
    
    env.set_demand_data(demand_data.copy())
    
    # Add disruption wrapper for some episodes
    disruption_config = {
        "demand_spike_prob": cfg.disruption.demand_spike_prob,
        "demand_spike_factor": cfg.disruption.demand_spike_factor,
        "lead_time_delay_prob": cfg.disruption.lead_time_delay_prob,
        "lead_time_delay_factor": cfg.disruption.lead_time_delay_factor,
        "node_failure_prob": cfg.disruption.node_failure_prob,
        "node_failure_duration": cfg.disruption.node_failure_duration,
        "cascade_enabled": cfg.disruption.cascade_enabled,
    }
    
    env_disrupted = DisruptionWrapper(
        MultiEchelonInventoryEnv(env_config), disruption_config
    )
    env_disrupted.env.set_demand_data(demand_data.copy())
    
    # ── Graph Setup ──
    graph = SupplyChainGraph.from_env_config(
        num_retailers=cfg.environment.num_retailers,
        num_distributors=cfg.environment.num_distributors,
        num_suppliers=cfg.environment.num_suppliers,
        lead_time_mean=cfg.environment.lead_time_mean,
    )
    
    adjacency = graph.get_adjacency_matrix().numpy()
    static_features = graph.node_features.numpy()
    
    # ── Agent Setup ──
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs[env.agent_ids[0]].shape[0]
    
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=obs_dim,
        action_dim=1,
        config={
            "hidden_dim": cfg.mappo.hidden_dim,
            "learning_rate": cfg.mappo.learning_rate,
            "gamma": cfg.mappo.gamma,
            "gae_lambda": cfg.mappo.gae_lambda,
            "clip_epsilon": cfg.mappo.clip_epsilon,
            "entropy_coef": cfg.mappo.entropy_coef,
            "value_loss_coef": cfg.mappo.value_loss_coef,
            "max_grad_norm": cfg.mappo.max_grad_norm,
            "num_epochs": cfg.mappo.num_epochs,
            "num_minibatches": cfg.mappo.num_minibatches,
            "use_gnn": cfg.mappo.use_gnn,
            "gnn_type": cfg.mappo.gnn_type,
            "gnn_hidden_dim": cfg.mappo.gnn_hidden_dim,
            "gnn_num_layers": cfg.mappo.gnn_num_layers,
        },
        adjacency=adjacency,
        static_node_features=static_features,
    )
    
    print(f"  Agents: {env.num_agents}")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Actor params: {sum(sum(p.numel() for p in a.parameters()) for a in agent.actors):,}")
    print(f"  Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")
    if agent.gnn:
        print(f"  GNN params: {sum(p.numel() for p in agent.gnn.parameters()):,}")
    
    # ── Learning Rate Scheduler ──
    import torch.optim as optim_module
    actor_scheduler = optim_module.lr_scheduler.CosineAnnealingLR(
        agent.actor_optimizer, T_max=total_episodes, eta_min=1e-5
    )
    critic_scheduler = optim_module.lr_scheduler.CosineAnnealingLR(
        agent.critic_optimizer, T_max=total_episodes, eta_min=1e-5
    )
    
    # ── Curriculum Learning ──
    # Phase 1: Normal only (learn basics)
    # Phase 2: 15% disrupted (gentle introduction)
    # Phase 3: 30% disrupted (moderate challenge)
    # Phase 4: 45% disrupted (full resilience training)
    def get_disruption_prob(ep):
        """Curriculum: gradually increase disruption frequency."""
        progress = ep / total_episodes
        if progress < 0.3:    return 0.0   # Phase 1: pure normal
        elif progress < 0.5:  return 0.15  # Phase 2: gentle
        elif progress < 0.75: return 0.30  # Phase 3: moderate
        else:                 return 0.45  # Phase 4: hard
    
    # ── Training Loop ──
    episode_rewards = []
    episode_costs = []
    eval_results = []
    training_losses = []
    best_eval_cost = float('inf')
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        # Curriculum-based disruption selection
        disruption_prob = get_disruption_prob(episode)
        use_disrupted = disruption_prob > 0 and np.random.random() < disruption_prob
        current_env = env_disrupted if use_disrupted else env
        
        # Always reset with fresh demand data
        if use_disrupted:
            current_env.env.set_demand_data(demand_data.copy())
        else:
            current_env.set_demand_data(demand_data.copy())
        
        obs, info = current_env.reset(seed=cfg.seed + episode)
        
        episode_reward = 0
        episode_cost = 0
        steps = 0
        
        while True:
            # Get actions
            actions, log_probs, values = agent.get_actions(
                obs, env.agent_ids, deterministic=False
            )
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = current_env.step(actions)
            
            done = truncated.get("__all__", False) or terminated.get("__all__", False)
            dones = {aid: done for aid in env.agent_ids}
            
            # Store in buffer
            agent.buffer.add(obs, actions, rewards, dones, log_probs, values)
            
            episode_reward += sum(rewards.values())
            episode_cost += sum(
                info[aid].get("holding_cost", 0) for aid in env.agent_ids
            )
            
            obs = next_obs
            steps += 1
            
            # Update after rollout_length steps
            if len(agent.buffer) >= rollout_length:
                # CRITICAL: compute bootstrap value for mid-episode rollouts.
                # Without this, GAE assumes future value=0 at the rollout boundary,
                # which makes the agent extremely short-sighted (can't plan beyond 128 steps).
                if not done:
                    with torch.no_grad():
                        _, _, bootstrap_vals = agent.get_actions(
                            next_obs, current_env.agent_ids, deterministic=False
                        )
                        bootstrap_value = float(np.mean(
                            [bootstrap_vals[aid] for aid in current_env.agent_ids]
                        ))
                else:
                    bootstrap_value = 0.0
                stats = agent.update(env.agent_ids, last_value=bootstrap_value)
                training_losses.append(stats)
            
            if done:
                break
        
        # Final update if buffer not empty (episode ended naturally → last_value=0)
        if len(agent.buffer) > 0:
            stats = agent.update(env.agent_ids, last_value=0.0)
            training_losses.append(stats)
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        
        # Step LR schedulers
        actor_scheduler.step()
        critic_scheduler.step()
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            elapsed = time.time() - start_time
            phase = f"P1-NORMAL" if disruption_prob == 0 else f"P{int(disruption_prob*100)}%-DISRUPT"
            
            print(
                f"  Ep {episode+1:>4d}/{total_episodes} | "
                f"Reward: {avg_reward:>10.2f} | "
                f"Cost: {avg_cost:>10.1f} | "
                f"{phase:>14s} | "
                f"LR: {actor_scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.0f}s"
            )
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            try:
                eval_metrics = evaluate_agent(agent, env, env.agent_ids, demand_data, num_episodes=3)
                eval_metrics['episode'] = episode + 1
                eval_results.append(eval_metrics)
                
                # Track best model
                if eval_metrics.get('total_cost', float('inf')) < best_eval_cost:
                    best_eval_cost = eval_metrics['total_cost']
                    agent.save(str(out_dir / "mappo_best.pt"))
                
                print(f"  -- EVAL -- Cost: ${eval_metrics['total_cost']:.0f} | "
                      f"Fill: {eval_metrics['fill_rate']:.2%} | "
                      f"Bullwhip: {eval_metrics.get('bullwhip_ratio', 0):.4f}"
                      f"{' *** BEST' if eval_metrics['total_cost'] == best_eval_cost else ''}")
            except Exception as e:
                print(f"  -- EVAL FAILED -- {e}")
                eval_results.append({'episode': episode + 1, 'error': str(e)})
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(str(out_dir / f"mappo_ep{episode+1}.pt"))
    
    # ── Save Final Results ──
    agent.save(str(out_dir / "mappo_final.pt"))
    
    # Save training curves
    plot_training_curves(episode_rewards, episode_costs, eval_results, out_dir)
    
    # Save metrics
    results = {
        "total_episodes": total_episodes,
        "final_avg_reward": float(np.mean(episode_rewards[-50:])),
        "final_avg_cost": float(np.mean(episode_costs[-50:])),
        "eval_results": eval_results,
        "config": {
            "use_gnn": cfg.mappo.use_gnn,
            "hidden_dim": cfg.mappo.hidden_dim,
            "lr": cfg.mappo.learning_rate,
        },
        "training_time_seconds": time.time() - start_time,
    }
    
    with open(out_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  Time: {time.time() - start_time:.0f}s")
    print(f"  Final avg reward: {results['final_avg_reward']:.1f}")
    print(f"  Results saved to: {out_dir}")
    print(f"{'='*60}")
    
    return results


def evaluate_agent(
    agent: MAPPOAgent,
    env: MultiEchelonInventoryEnv,
    agent_ids: list,
    demand_data: np.ndarray,
    num_episodes: int = 5,
) -> Dict:
    """Evaluate agent performance."""
    all_metrics = []
    
    for ep in range(num_episodes):
        env.set_demand_data(demand_data.copy())
        obs, _ = env.reset(seed=1000 + ep)
        
        episode = EpisodeMetrics()
        done = False
        
        while not done:
            actions, _, _ = agent.get_actions(obs, agent_ids, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = truncated.get("__all__", False)
            episode.total_periods += 1
            
            for aid in agent_ids:
                ai = info.get(aid, {})
                episode.total_holding_cost += ai.get("holding_cost", 0)
                if "shortage" in ai:
                    episode.total_shortage_cost += ai["shortage"] * env.shortage_cost
                    if ai["shortage"] > 0:
                        episode.stockout_periods += 1
                if "demand" in ai:
                    episode.total_demand += ai["demand"]
                if "fulfilled" in ai:
                    episode.total_fulfilled += ai["fulfilled"]
                if "order_placed" in ai:
                    episode.orders_placed.append(ai["order_placed"])
                    if ai["order_placed"] > 0:
                        episode.total_ordering_cost += env.ordering_cost
            
            if env.episode_data["demand_history"]:
                last = env.episode_data["demand_history"][-1]
                if isinstance(last, list):
                    episode.demand_history.extend(last)
            
            episode.inventory_history.append([n.inventory for n in env.nodes])
        
        metrics = SupplyChainMetrics.compute_all(episode)
        all_metrics.append(metrics)
    
    # Average
    avg = {}
    for key in all_metrics[0]:
        avg[key] = float(np.mean([m[key] for m in all_metrics]))
    
    return avg


def plot_training_curves(rewards, costs, eval_results, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, color="#2196F3")
    if len(rewards) > 10:
        window = min(50, len(rewards) // 3)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), smoothed, color="#F44336", linewidth=2)
    axes[0, 0].set_title("Episode Rewards", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    
    # Costs
    axes[0, 1].plot(costs, alpha=0.3, color="#FF9800")
    if len(costs) > 10:
        window = min(50, len(costs) // 3)
        smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(costs)), smoothed, color="#4CAF50", linewidth=2)
    axes[0, 1].set_title("Episode Costs", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total Cost")
    
    # Eval metrics
    if eval_results:
        eval_costs = [e.get("total_cost", 0) for e in eval_results]
        eval_fills = [e.get("fill_rate", 0) for e in eval_results]
        
        axes[1, 0].plot(eval_costs, "o-", color="#9C27B0", linewidth=2)
        axes[1, 0].set_title("Eval: Total Cost", fontweight="bold")
        axes[1, 0].set_xlabel("Evaluation #")
        axes[1, 0].set_ylabel("Cost")
        
        axes[1, 1].plot(eval_fills, "o-", color="#E91E63", linewidth=2)
        axes[1, 1].set_title("Eval: Fill Rate", fontweight="bold")
        axes[1, 1].set_xlabel("Evaluation #")
        axes[1, 1].set_ylabel("Fill Rate")
        axes[1, 1].set_ylim(0, 1.05)
    
    plt.suptitle("MAPPO+GNN Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to: {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MAPPO+GNN")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--no-gnn", action="store_true", help="Disable GNN")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    if args.no_gnn:
        config.mappo.use_gnn = False
    
    train_mappo(
        config=config,
        total_episodes=args.episodes,
        output_dir=args.output,
    )
