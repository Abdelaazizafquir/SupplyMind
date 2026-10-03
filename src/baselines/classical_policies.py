"""
Classical Baseline Policies
============================
Implements standard inventory management policies for benchmarking:
- (s, S) policy: Reorder up to S when inventory falls below s
- Base-stock policy: Order up to a fixed target level each period
"""

import numpy as np
from typing import Dict, List, Tuple
from ..utils.metrics import SupplyChainMetrics, EpisodeMetrics


class BasePolicy:
    """Abstract base class for inventory policies."""
    
    def __init__(self, env, name: str = "base"):
        self.env = env
        self.base_env = getattr(self.env, "unwrapped", self.env)
        self.name = name
    
    def get_action(self, obs: Dict, info: Dict = None) -> Dict[str, np.ndarray]:
        """Compute action for all agents."""
        raise NotImplementedError
    
    def evaluate(self, num_episodes: int = 10, seed: int = 42) -> Dict:
        """
        Run the policy for multiple episodes and return aggregated metrics.
        """
        all_metrics = []
        
        for ep in range(num_episodes):
            episode_metrics = self._run_episode(seed=seed + ep)
            metrics = SupplyChainMetrics.compute_all(episode_metrics)
            all_metrics.append(metrics)
        
        # Aggregate across episodes
        aggregated = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        return {
            "policy": self.name,
            "num_episodes": num_episodes,
            "metrics": aggregated,
        }
    
    def _run_episode(self, seed: int = 42) -> EpisodeMetrics:
        """Run a single episode and collect metrics."""
        obs, info = self.env.reset(seed=seed)
        episode = EpisodeMetrics()
        done = False
        
        while not done:
            actions = self.get_action(obs, info)
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            done = truncated.get("__all__", False) or terminated.get("__all__", False)
            episode.total_periods += 1
            
            # Collect metrics from info
            for agent_id in self.base_env.agent_ids:
                agent_info = info.get(agent_id, {})
                
                if "holding_cost" in agent_info:
                    episode.total_holding_cost += agent_info["holding_cost"]
                if "shortage" in agent_info:
                    episode.total_shortage_cost += agent_info["shortage"] * self.base_env.shortage_cost
                    if agent_info["shortage"] > 0:
                        episode.stockout_periods += 1
                if "demand" in agent_info:
                    episode.total_demand += agent_info["demand"]
                if "fulfilled" in agent_info:
                    episode.total_fulfilled += agent_info["fulfilled"]
                if "order_placed" in agent_info:
                    episode.orders_placed.append(agent_info["order_placed"])
            
            # Demand history
            if hasattr(self.base_env, 'episode_data'):
                if self.base_env.episode_data["demand_history"]:
                    last_demand = self.base_env.episode_data["demand_history"][-1]
                    if isinstance(last_demand, list):
                        episode.demand_history.extend(last_demand)
            
            # Inventory history
            episode.inventory_history.append(
                [node.inventory for node in self.base_env.nodes]
            )
            
            # Ordering cost
            for agent_id in self.base_env.agent_ids:
                agent_info = info.get(agent_id, {})
                if agent_info.get("order_placed", 0) > 0:
                    episode.total_ordering_cost += self.base_env.ordering_cost
        
        episode.total_cost = (
            episode.total_holding_cost + 
            episode.total_ordering_cost + 
            episode.total_shortage_cost
        )
        
        return episode


class SsPolicy(BasePolicy):
    """
    (s, S) Policy: When inventory falls below reorder point s,
    order up to target level S.
    
    Parameters:
    - s: Reorder point (trigger level)
    - S: Order-up-to level (target)
    """
    
    def __init__(self, env, s: float = 30, S: float = 100, name: str = "(s,S) Policy"):
        super().__init__(env, name)
        self.s = s
        self.S = S
    
    def get_action(self, obs: Dict, info: Dict = None) -> Dict[str, np.ndarray]:
        actions = {}
        
        for i, agent_id in enumerate(self.base_env.agent_ids):
            node = self.base_env.nodes[i]
            
            if node.inventory < self.s:
                # Action space represents squared target inventory ratio
                target_ratio = np.sqrt(min(self.S / node.capacity, 1.0))
                actions[agent_id] = np.array([target_ratio], dtype=np.float32)
            else:
                # Target inventory of 0 guarantees no order is placed
                actions[agent_id] = np.array([0.0], dtype=np.float32)
        
        return actions


class BaseStockPolicy(BasePolicy):
    """
    Base-Stock Policy: Order the difference between target stock level
    and current inventory position (inventory + pipeline) every period.
    """
    
    def __init__(self, env, target_stock: float = 80, name: str = "Base-Stock Policy"):
        super().__init__(env, name)
        self.target_stock = target_stock
    
    def get_action(self, obs: Dict, info: Dict = None) -> Dict[str, np.ndarray]:
        actions = {}
        
        for i, agent_id in enumerate(self.base_env.agent_ids):
            node = self.base_env.nodes[i]
            
            # Action space represents squared target inventory ratio
            target_ratio = np.sqrt(min(self.target_stock / node.capacity, 1.0))
            actions[agent_id] = np.array([target_ratio], dtype=np.float32)
        
        return actions


class RandomPolicy(BasePolicy):
    """Random policy for sanity checking."""
    
    def __init__(self, env, name: str = "Random Policy"):
        super().__init__(env, name)
    
    def get_action(self, obs: Dict, info: Dict = None) -> Dict[str, np.ndarray]:
        return {
            agent_id: np.array([np.random.random()], dtype=np.float32)
            for agent_id in self.base_env.agent_ids
        }


def run_baseline_comparison(env, num_episodes: int = 10, seed: int = 42) -> Dict:
    """
    Run all baseline policies and return comparative results.
    """
    policies = [
        SsPolicy(env, s=30, S=100),
        SsPolicy(env, s=20, S=80, name="(s,S) Conservative"),
        SsPolicy(env, s=40, S=120, name="(s,S) Aggressive"),
        BaseStockPolicy(env, target_stock=80),
        BaseStockPolicy(env, target_stock=60, name="Base-Stock Conservative"),
        BaseStockPolicy(env, target_stock=100, name="Base-Stock Aggressive"),
        RandomPolicy(env),
    ]
    
    results = {}
    for policy in policies:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {policy.name}")
        print(f"{'='*50}")
        
        result = policy.evaluate(num_episodes=num_episodes, seed=seed)
        results[policy.name] = result
        
        # Print summary
        m = result["metrics"]
        print(f"  Total Cost:    ${m['total_cost']['mean']:>10,.2f} ± ${m['total_cost']['std']:>8,.2f}")
        print(f"  Fill Rate:      {m['fill_rate']['mean']:>9.2%} ± {m['fill_rate']['std']:>7.2%}")
        print(f"  Stockout Rate:  {m['stockout_rate']['mean']:>9.2%} ± {m['stockout_rate']['std']:>7.2%}")
        print(f"  Bullwhip:       {m['bullwhip_ratio']['mean']:>9.4f} ± {m['bullwhip_ratio']['std']:>7.4f}")
    
    return results
