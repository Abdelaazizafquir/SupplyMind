"""
Supply Chain Metrics Module
============================
All evaluation metrics for multi-echelon inventory optimization.
Used across baselines, MAPPO, and full system experiments.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Container for metrics collected during one simulation episode."""
    total_holding_cost: float = 0.0
    total_ordering_cost: float = 0.0
    total_shortage_cost: float = 0.0
    total_cost: float = 0.0
    stockout_periods: int = 0
    total_periods: int = 0
    total_demand: float = 0.0
    total_fulfilled: float = 0.0
    orders_placed: List[float] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)
    inventory_history: List[List[float]] = field(default_factory=list)


class SupplyChainMetrics:
    """
    Computes all supply chain performance metrics.
    
    Metrics computed:
    - Total cost (holding + ordering + shortage)
    - Stockout rate
    - Fill rate (service level)
    - Inventory turnover
    - Bullwhip ratio
    - Recovery time (for disruption scenarios)
    - Cost-of-disruption
    """
    
    @staticmethod
    def total_cost(holding_cost: float, ordering_cost: float, shortage_cost: float) -> float:
        """Total supply chain cost = holding + ordering + shortage."""
        return holding_cost + ordering_cost + shortage_cost
    
    @staticmethod
    def stockout_rate(stockout_periods: int, total_periods: int) -> float:
        """Fraction of periods where demand could not be fully met."""
        if total_periods == 0:
            return 0.0
        return stockout_periods / total_periods
    
    @staticmethod
    def fill_rate(total_fulfilled: float, total_demand: float) -> float:
        """Fraction of total demand that was satisfied from stock."""
        if total_demand == 0:
            return 1.0
        return min(total_fulfilled / total_demand, 1.0)
    
    @staticmethod
    def inventory_turnover(total_demand: float, avg_inventory: float) -> float:
        """How many times inventory is sold and replaced over a period."""
        if avg_inventory == 0:
            return float('inf')
        return total_demand / avg_inventory
    
    @staticmethod
    def bullwhip_ratio(orders: np.ndarray, demand: np.ndarray) -> float:
        """
        Bullwhip effect = Var(orders)/Mean(orders) / (Var(demand)/Mean(demand))
        
        A ratio > 1 indicates order amplification (bullwhip effect).
        A ratio ≈ 1 means orders track demand well.
        A ratio < 1 means the system dampens demand variability.
        """
        orders = np.asarray(orders, dtype=float)
        demand = np.asarray(demand, dtype=float)
        
        mean_demand = np.mean(demand)
        mean_orders = np.mean(orders)
        
        if mean_demand == 0 or mean_orders == 0:
            return float('inf')
        
        cv_demand = np.var(demand) / mean_demand
        cv_orders = np.var(orders) / mean_orders
        
        if cv_demand == 0:
            return float('inf')
        
        return cv_orders / cv_demand
    
    @staticmethod
    def recovery_time(
        inventory_levels: np.ndarray, 
        disruption_start: int, 
        normal_level: float,
        threshold: float = 0.9
    ) -> int:
        """
        Number of periods after disruption start until inventory 
        recovers to threshold * normal_level.
        
        Args:
            inventory_levels: Array of inventory levels over time
            disruption_start: Period index when disruption began
            normal_level: Pre-disruption average inventory level
            threshold: Recovery is when inventory >= threshold * normal_level
            
        Returns:
            Number of periods to recover, or -1 if never recovered
        """
        target = threshold * normal_level
        post_disruption = inventory_levels[disruption_start:]
        
        for i, level in enumerate(post_disruption):
            if level >= target:
                return i
        
        return -1  # Never recovered within the episode
    
    @staticmethod
    def cost_of_disruption(
        cost_with_disruption: float, 
        cost_without_disruption: float
    ) -> float:
        """Extra cost incurred due to the disruption."""
        return cost_with_disruption - cost_without_disruption
    
    @staticmethod
    def adaptive_capacity(
        recovery_time: int,
        cost_of_disruption: float,
        disruption_severity: float
    ) -> float:
        """
        Combined resilience metric.
        Higher is better (faster recovery, lower extra cost per unit of disruption).
        
        adaptive_capacity = disruption_severity / (recovery_time * cost_of_disruption)
        """
        if recovery_time <= 0 or cost_of_disruption <= 0:
            return 0.0
        return disruption_severity / (recovery_time * cost_of_disruption)
    
    @classmethod
    def compute_all(cls, episode: EpisodeMetrics) -> Dict[str, float]:
        """Compute all metrics from an episode's collected data."""
        orders = np.array(episode.orders_placed) if episode.orders_placed else np.array([0])
        demand = np.array(episode.demand_history) if episode.demand_history else np.array([0])
        
        # Average inventory across all echelons and periods
        if episode.inventory_history:
            inv_array = np.array(episode.inventory_history)
            avg_inventory = np.mean(inv_array)
        else:
            avg_inventory = 0.0
        
        return {
            "total_cost": cls.total_cost(
                episode.total_holding_cost,
                episode.total_ordering_cost,
                episode.total_shortage_cost
            ),
            "holding_cost": episode.total_holding_cost,
            "ordering_cost": episode.total_ordering_cost,
            "shortage_cost": episode.total_shortage_cost,
            "stockout_rate": cls.stockout_rate(
                episode.stockout_periods, 
                episode.total_periods
            ),
            "fill_rate": cls.fill_rate(
                episode.total_fulfilled, 
                episode.total_demand
            ),
            "inventory_turnover": cls.inventory_turnover(
                episode.total_demand, 
                avg_inventory
            ),
            "bullwhip_ratio": cls.bullwhip_ratio(orders, demand),
            "avg_inventory": avg_inventory,
            "total_periods": episode.total_periods,
        }
    
    @staticmethod
    def format_results(metrics: Dict[str, float]) -> str:
        """Pretty-print metrics for logging."""
        lines = [
            "=" * 50,
            "  SUPPLY CHAIN PERFORMANCE METRICS",
            "=" * 50,
            f"  Total Cost:          ${metrics['total_cost']:>12,.2f}",
            f"    ├─ Holding:        ${metrics['holding_cost']:>12,.2f}",
            f"    ├─ Ordering:       ${metrics['ordering_cost']:>12,.2f}",
            f"    └─ Shortage:       ${metrics['shortage_cost']:>12,.2f}",
            f"  Stockout Rate:        {metrics['stockout_rate']:>11.2%}",
            f"  Fill Rate:            {metrics['fill_rate']:>11.2%}",
            f"  Inventory Turnover:   {metrics['inventory_turnover']:>11.2f}",
            f"  Bullwhip Ratio:       {metrics['bullwhip_ratio']:>11.4f}",
            f"  Avg Inventory:        {metrics['avg_inventory']:>11.2f}",
            f"  Total Periods:        {metrics['total_periods']:>11d}",
            "=" * 50,
        ]
        return "\n".join(lines)
