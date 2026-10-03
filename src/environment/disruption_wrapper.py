"""
Disruption Wrapper
==================
Gymnasium wrapper that injects realistic disruptions into the supply chain.

Disruption Types:
- Type 1: Demand spikes (sudden surge in customer demand)
- Type 2: Lead-time delays (upstream delays)
- Type 3: Node failures (supplier/distributor goes offline)
- Type 4: Cascading disruptions (chain reaction from Type 3)
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass


class DisruptionType(Enum):
    NONE = 0
    DEMAND_SPIKE = 1
    LEAD_TIME_DELAY = 2
    NODE_FAILURE = 3
    CASCADE = 4


@dataclass
class ActiveDisruption:
    """Tracks an active disruption event."""
    disruption_type: DisruptionType
    start_step: int
    duration: int
    severity: float
    affected_nodes: List[int]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_active(self) -> bool:
        return True  # Checked externally
    
    def remaining(self, current_step: int) -> int:
        return max(0, (self.start_step + self.duration) - current_step)


class DisruptionWrapper(gym.Wrapper):
    """
    Wraps the MultiEchelonInventoryEnv to inject stochastic disruptions.
    
    Usage:
        env = MultiEchelonInventoryEnv(config)
        env = DisruptionWrapper(env, disruption_config)
    """
    
    def __init__(self, env: gym.Env, config: dict = None):
        super().__init__(env)
        
        cfg = config or {}
        
        # Disruption probabilities (per step)
        self.demand_spike_prob = cfg.get("demand_spike_prob", 0.02)
        self.demand_spike_factor = cfg.get("demand_spike_factor", 3.0)
        
        self.lead_time_delay_prob = cfg.get("lead_time_delay_prob", 0.03)
        self.lead_time_delay_factor = cfg.get("lead_time_delay_factor", 3.0)
        
        self.node_failure_prob = cfg.get("node_failure_prob", 0.01)
        self.node_failure_duration = cfg.get("node_failure_duration", 10)
        
        self.cascade_enabled = cfg.get("cascade_enabled", True)
        
        # Track active disruptions
        self.active_disruptions: List[ActiveDisruption] = []
        self.disruption_history: List[ActiveDisruption] = []
        
        # Disruption schedule (for reproducible experiments)
        self.scheduled_disruptions: List[Dict] = cfg.get("scheduled_disruptions", [])
        
        # Track failed nodes
        self.failed_nodes: set = set()
        
        # Non-destructive demand overrides (FIX: don't mutate demand_data in-place)
        self._demand_overrides: Dict[int, Dict[int, float]] = {}  # step -> {node_id -> multiplier}
        
        # Original lead time values
        self._original_lead_time_mean = env.lead_time_mean
        self._original_lead_time_std = env.lead_time_std
    
    def reset(self, **kwargs):
        """Reset environment and clear all disruptions."""
        self.active_disruptions = []
        self.disruption_history = []
        self.failed_nodes = set()
        self._demand_overrides = {}
        self.env.lead_time_mean = self._original_lead_time_mean
        self.env.lead_time_std = self._original_lead_time_std
        
        return self.env.reset(**kwargs)
    
    def step(self, actions: Dict[str, np.ndarray]):
        """
        Step with disruption injection.
        
        1. Check for new disruptions (scheduled or stochastic)
        2. Apply active disruption effects
        3. Apply demand overrides NON-DESTRUCTIVELY
        4. Execute base environment step
        5. Clean up expired disruptions
        """
        current_step = self.env.current_step
        
        # 1. Check scheduled disruptions
        self._check_scheduled_disruptions(current_step)
        
        # 2. Check stochastic disruptions
        self._check_stochastic_disruptions(current_step)
        
        # 3. Apply disruption effects (actions + demand overrides)
        modified_actions = self._apply_disruption_effects(actions, current_step)
        
        # 4. Apply demand overrides NON-DESTRUCTIVELY before stepping
        #    Save original demand, apply spike, step, then restore
        original_demand_row = None
        if current_step in self._demand_overrides and self.env.demand_data is not None:
            step_idx = min(current_step, len(self.env.demand_data) - 1)
            original_demand_row = self.env.demand_data[step_idx].copy()
            for node_id, multiplier in self._demand_overrides[current_step].items():
                self.env.demand_data[step_idx][node_id] *= multiplier
        
        # 5. Execute step
        obs, rewards, terminated, truncated, infos = self.env.step(modified_actions)
        
        # 6. Restore original demand data (FIX: never permanently mutate)
        if original_demand_row is not None:
            step_idx = min(current_step, len(self.env.demand_data) - 1)
            self.env.demand_data[step_idx] = original_demand_row
        
        # 7. Add disruption info to observations
        for agent_id in self.env.agent_ids:
            infos[agent_id]["disruptions_active"] = len(self.active_disruptions)
            infos[agent_id]["disruption_types"] = [
                d.disruption_type.value for d in self.active_disruptions
            ]
            infos[agent_id]["failed_nodes"] = list(self.failed_nodes)
        
        # 8. Clean up expired disruptions
        self._cleanup_disruptions(current_step)
        
        return obs, rewards, terminated, truncated, infos
    
    def _check_scheduled_disruptions(self, current_step: int):
        """Activate any scheduled disruptions for this step."""
        for sched in self.scheduled_disruptions:
            if sched.get("start_step") == current_step:
                d_type = DisruptionType(sched["type"])
                disruption = ActiveDisruption(
                    disruption_type=d_type,
                    start_step=current_step,
                    duration=sched.get("duration", 10),
                    severity=sched.get("severity", 1.0),
                    affected_nodes=sched.get("affected_nodes", []),
                    metadata=sched.get("metadata", {}),
                )
                self.active_disruptions.append(disruption)
                self.disruption_history.append(disruption)
    
    def _check_stochastic_disruptions(self, current_step: int):
        """Randomly trigger disruptions based on configured probabilities."""
        rng = np.random.default_rng()
        
        # Type 1: Demand spike
        if rng.random() < self.demand_spike_prob:
            affected = rng.choice(
                [n.node_id for n in self.env.nodes if n.node_type == "retailer"],
                size=1
            ).tolist()
            
            disruption = ActiveDisruption(
                disruption_type=DisruptionType.DEMAND_SPIKE,
                start_step=current_step,
                duration=rng.integers(3, 15),
                severity=self.demand_spike_factor,
                affected_nodes=affected,
            )
            self.active_disruptions.append(disruption)
            self.disruption_history.append(disruption)
        
        # Type 2: Lead-time delay
        if rng.random() < self.lead_time_delay_prob:
            disruption = ActiveDisruption(
                disruption_type=DisruptionType.LEAD_TIME_DELAY,
                start_step=current_step,
                duration=rng.integers(5, 20),
                severity=self.lead_time_delay_factor,
                affected_nodes=list(range(self.env.num_agents)),
            )
            self.active_disruptions.append(disruption)
            self.disruption_history.append(disruption)
        
        # Type 3: Node failure
        if rng.random() < self.node_failure_prob:
            # Choose a non-retailer node to fail
            candidates = [n.node_id for n in self.env.nodes 
                         if n.node_type in ("distributor", "supplier")
                         and n.node_id not in self.failed_nodes]
            
            if candidates:
                failed_id = rng.choice(candidates)
                disruption = ActiveDisruption(
                    disruption_type=DisruptionType.NODE_FAILURE,
                    start_step=current_step,
                    duration=self.node_failure_duration,
                    severity=1.0,
                    affected_nodes=[failed_id],
                )
                self.active_disruptions.append(disruption)
                self.disruption_history.append(disruption)
                self.failed_nodes.add(failed_id)
                
                # Type 4: Cascade (if enabled)
                if self.cascade_enabled:
                    self._trigger_cascade(failed_id, current_step)
    
    def _trigger_cascade(self, failed_node_id: int, current_step: int):
        """Trigger cascading effects when a node fails."""
        rng = np.random.default_rng()
        failed_node = self.env.nodes[failed_node_id]
        
        # Find downstream nodes that depend on the failed node
        downstream = [
            j for j in range(self.env.num_agents)
            if self.env.adjacency[failed_node_id, j] > 0
            and self.env.nodes[j].echelon < failed_node.echelon
        ]
        
        if downstream:
            # Demand spike at downstream nodes (panic ordering)
            cascade = ActiveDisruption(
                disruption_type=DisruptionType.CASCADE,
                start_step=current_step,
                duration=self.node_failure_duration + 5,
                severity=1.5,
                affected_nodes=downstream,
                metadata={"triggered_by": failed_node_id},
            )
            self.active_disruptions.append(cascade)
            self.disruption_history.append(cascade)
    
    def _apply_disruption_effects(
        self, 
        actions: Dict[str, np.ndarray], 
        current_step: int
    ) -> Dict[str, np.ndarray]:
        """Apply active disruption effects to the environment and actions."""
        modified_actions = dict(actions)
        
        for disruption in self.active_disruptions:
            if disruption.remaining(current_step) <= 0:
                continue
            
            if disruption.disruption_type == DisruptionType.DEMAND_SPIKE:
                # Register demand override (NON-DESTRUCTIVE — don't mutate demand_data)
                if self.env.demand_data is not None:
                    if current_step not in self._demand_overrides:
                        self._demand_overrides[current_step] = {}
                    for node_id in disruption.affected_nodes:
                        if node_id < self.env.num_retailers:
                            # Stack multipliers if multiple spikes hit same node
                            prev = self._demand_overrides[current_step].get(node_id, 1.0)
                            self._demand_overrides[current_step][node_id] = prev * disruption.severity
            
            elif disruption.disruption_type == DisruptionType.LEAD_TIME_DELAY:
                # Increase lead times
                self.env.lead_time_mean = int(
                    self._original_lead_time_mean * disruption.severity
                )
                self.env.lead_time_std = int(
                    self._original_lead_time_std * disruption.severity
                )
            
            elif disruption.disruption_type == DisruptionType.NODE_FAILURE:
                # Zero out actions for failed nodes
                for node_id in disruption.affected_nodes:
                    agent_id = self.env.agent_ids[node_id]
                    modified_actions[agent_id] = np.array([0.0])
                    # Also clear their inventory (frozen)
                    self.env.nodes[node_id].inventory = max(
                        0, self.env.nodes[node_id].inventory * 0.95
                    )
            
            elif disruption.disruption_type == DisruptionType.CASCADE:
                # Amplify orders at affected nodes (panic buying)
                for node_id in disruption.affected_nodes:
                    agent_id = self.env.agent_ids[node_id]
                    if agent_id in modified_actions:
                        current = modified_actions[agent_id]
                        modified_actions[agent_id] = np.clip(
                            current * disruption.severity, 0.0, 1.0
                        )
        
        return modified_actions
    
    def _cleanup_disruptions(self, current_step: int):
        """Remove expired disruptions."""
        still_active = []
        for d in self.active_disruptions:
            if d.remaining(current_step) > 0:
                still_active.append(d)
            else:
                # Restore state for ended disruptions
                if d.disruption_type == DisruptionType.LEAD_TIME_DELAY:
                    self.env.lead_time_mean = self._original_lead_time_mean
                    self.env.lead_time_std = self._original_lead_time_std
                elif d.disruption_type == DisruptionType.NODE_FAILURE:
                    for node_id in d.affected_nodes:
                        self.failed_nodes.discard(node_id)
        
        self.active_disruptions = still_active
    
    def get_disruption_summary(self) -> Dict[str, Any]:
        """Get a summary of all disruptions that occurred during the episode."""
        return {
            "total_disruptions": len(self.disruption_history),
            "by_type": {
                dtype.name: sum(1 for d in self.disruption_history 
                              if d.disruption_type == dtype)
                for dtype in DisruptionType if dtype != DisruptionType.NONE
            },
            "total_disrupted_periods": sum(d.duration for d in self.disruption_history),
            "active_now": len(self.active_disruptions),
        }
