"""
Multi-Echelon Supply Chain Environment
=======================================
Custom Gymnasium environment for multi-agent inventory management.
Supports configurable number of echelons and agents.
Integrates with disruption wrapper.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a node (agent) in the supply chain network."""
    node_id: int
    node_type: str  # "retailer", "distributor", "supplier"
    echelon: int    # 0 = retailer (closest to customer), higher = upstream
    inventory: float = 0.0
    capacity: float = 1000.0
    pipeline: List[Tuple[int, float]] = None  # (arrival_time, quantity)
    
    def __post_init__(self):
        if self.pipeline is None:
            self.pipeline = []


class MultiEchelonInventoryEnv(gym.Env):
    """
    Multi-echelon inventory management environment for MARL.
    
    Architecture:
        Suppliers → Distributors → Retailers → Customers
        
    Each agent controls one node and decides order quantities.
    External demand arrives at retailers from real data (M5) or synthetic.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        # Default config
        cfg = config or {}
        self.num_retailers = cfg.get("num_retailers", 3)
        self.num_distributors = cfg.get("num_distributors", 2)
        self.num_suppliers = cfg.get("num_suppliers", 1)
        self.num_periods = cfg.get("num_periods", 365)
        self.lead_time_mean = cfg.get("lead_time_mean", 3)
        self.lead_time_std = cfg.get("lead_time_std", 1)
        self.holding_cost = cfg.get("holding_cost", 1.0)
        self.ordering_cost = cfg.get("ordering_cost", 50.0)
        self.shortage_cost = cfg.get("shortage_cost", 500.0)
        self.initial_inventory = cfg.get("initial_inventory", 100)
        self.max_order = cfg.get("max_order_quantity", 500)
        
        # External demand data (set via set_demand_data())
        self.demand_data = None
        self.demand_forecast = None  # From PatchTST
        
        # Total number of agents
        self.num_agents = self.num_retailers + self.num_distributors + self.num_suppliers
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        
        # Build network topology
        self.nodes = self._build_network()
        self.adjacency = self._build_adjacency()
        
        # Observation: [inventory, pipeline, demand, downstream, upstream, time,
        #               is_retailer, is_distributor, is_supplier, forecast_1..H]
        self.forecast_horizon = cfg.get("forecast_horizon", 7)
        obs_dim = 6 + 3 + self.forecast_horizon  # base(6) + role(3) + forecast
        
        # Per-agent spaces
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for agent_id in self.agent_ids
        })
        
        self.action_space = spaces.Dict({
            agent_id: spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )  # Normalized [0,1] → scaled to [0, max_order]
            for agent_id in self.agent_ids
        })
        
        # State tracking
        self.current_step = 0
        self.episode_data = {
            "inventory_history": [],
            "order_history": [],
            "demand_history": [],
            "cost_history": [],
        }
    
    def _build_network(self) -> List[Node]:
        """Build the supply chain network topology."""
        nodes = []
        node_id = 0
        
        # Retailers (echelon 0)
        for i in range(self.num_retailers):
            nodes.append(Node(
                node_id=node_id, node_type="retailer", echelon=0,
                inventory=self.initial_inventory, capacity=self.max_order * 2
            ))
            node_id += 1
        
        # Distributors (echelon 1)
        for i in range(self.num_distributors):
            nodes.append(Node(
                node_id=node_id, node_type="distributor", echelon=1,
                inventory=self.initial_inventory * 2, capacity=self.max_order * 5
            ))
            node_id += 1
        
        # Suppliers (echelon 2)
        for i in range(self.num_suppliers):
            nodes.append(Node(
                node_id=node_id, node_type="supplier", echelon=2,
                inventory=self.initial_inventory * 5, capacity=self.max_order * 10
            ))
            node_id += 1
        
        return nodes
    
    def _build_adjacency(self) -> np.ndarray:
        """
        Build adjacency matrix for the supply chain graph.
        Edges go from upstream to downstream (supplier → distributor → retailer).
        """
        adj = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        
        retailers = [n for n in self.nodes if n.node_type == "retailer"]
        distributors = [n for n in self.nodes if n.node_type == "distributor"]
        suppliers = [n for n in self.nodes if n.node_type == "supplier"]
        
        # Connect retailers to distributors (round-robin assignment)
        for i, r in enumerate(retailers):
            d = distributors[i % len(distributors)]
            adj[r.node_id, d.node_id] = 1.0
            adj[d.node_id, r.node_id] = 1.0  # Bidirectional for GNN
        
        # Connect distributors to suppliers
        for i, d in enumerate(distributors):
            s = suppliers[i % len(suppliers)]
            adj[d.node_id, s.node_id] = 1.0
            adj[s.node_id, d.node_id] = 1.0
        
        return adj
    
    def get_edge_index(self) -> np.ndarray:
        """Get edge index in COO format for PyTorch Geometric."""
        edges = np.where(self.adjacency > 0)
        return np.array([edges[0], edges[1]], dtype=np.int64)
    
    def set_demand_data(self, demand_series: np.ndarray):
        """
        Set external demand data (e.g., from M5).
        Shape: (num_periods, num_retailers)
        """
        self.demand_data = demand_series
    
    def set_demand_forecast(self, forecasts: np.ndarray):
        """
        Set demand forecasts from PatchTST.
        Shape: (num_periods, num_retailers, horizon)
        """
        self.demand_forecast = forecasts
    
    def _get_demand(self, step: int) -> np.ndarray:
        """Get demand for all retailers at the current step."""
        if self.demand_data is not None and step < len(self.demand_data):
            return self.demand_data[step]
        else:
            # Synthetic fallback: Poisson demand
            return np.random.poisson(lam=20, size=self.num_retailers).astype(float)
    
    def _get_forecast(self, step: int, retailer_idx: int) -> np.ndarray:
        """Get demand forecast for a retailer."""
        if self.demand_forecast is not None and step < len(self.demand_forecast):
            return self.demand_forecast[step, retailer_idx]
        else:
            # No forecast available — return zeros
            return np.zeros(self.forecast_horizon, dtype=np.float32)
    
    def _get_lead_time(self) -> int:
        """Sample a stochastic lead time."""
        lt = max(1, int(np.random.normal(self.lead_time_mean, self.lead_time_std)))
        return lt
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observations for all agents."""
        obs = {}
        
        for i, node in enumerate(self.nodes):
            agent_id = self.agent_ids[i]
            
            # Pipeline total (goods in transit)
            pipeline_total = sum(qty for _, qty in node.pipeline)
            
            # Incoming demand (only retailers see external demand)
            if node.node_type == "retailer":
                incoming_demand = self._get_demand(self.current_step)[
                    node.node_id  # Retailer index
                ] if self.current_step < self.num_periods else 0.0
                forecast = self._get_forecast(self.current_step, node.node_id)
            else:
                # Upstream nodes see aggregate orders from downstream
                incoming_demand = 0.0
                forecast = np.zeros(self.forecast_horizon, dtype=np.float32)
            
            # Downstream / upstream average inventory
            downstream_ids = [j for j in range(self.num_agents) 
                            if self.adjacency[i, j] > 0 and self.nodes[j].echelon < node.echelon]
            upstream_ids = [j for j in range(self.num_agents) 
                          if self.adjacency[i, j] > 0 and self.nodes[j].echelon > node.echelon]
            
            downstream_avg = np.mean([self.nodes[j].inventory for j in downstream_ids]) if downstream_ids else 0.0
            upstream_avg = np.mean([self.nodes[j].inventory for j in upstream_ids]) if upstream_ids else 0.0
            
            # Time feature (normalized)
            time_feature = self.current_step / self.num_periods
            
            # Role indicators (one-hot)
            is_retailer = 1.0 if node.node_type == "retailer" else 0.0
            is_distributor = 1.0 if node.node_type == "distributor" else 0.0
            is_supplier = 1.0 if node.node_type == "supplier" else 0.0
            
            # Concatenate observation
            base_obs = np.array([
                node.inventory / node.capacity,  # idx 0: Normalized inventory
                pipeline_total / node.capacity,  # idx 1: Normalized pipeline
                incoming_demand / 100.0,          # idx 2: Normalized demand
                downstream_avg / 500.0,           # idx 3: Normalized downstream
                upstream_avg / 500.0,             # idx 4: Normalized upstream
                time_feature,                     # idx 5: Time progress
                is_retailer,                      # idx 6: Role indicator
                is_distributor,                   # idx 7: Role indicator
                is_supplier,                      # idx 8: Role indicator
            ], dtype=np.float32)
            
            obs[agent_id] = np.concatenate([base_obs, forecast / 100.0])
        
        return obs
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.nodes = self._build_network()
        
        self.episode_data = {
            "inventory_history": [],
            "order_history": [],
            "demand_history": [],
            "cost_history": [],
        }
        
        obs = self._get_obs()
        info = {agent_id: {} for agent_id in self.agent_ids}
        
        return obs, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dict mapping agent_id → order quantity (normalized [0,1])
            
        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        total_costs = {agent_id: 0.0 for agent_id in self.agent_ids}
        infos = {agent_id: {} for agent_id in self.agent_ids}
        
        # 1. Process arriving shipments (pipeline)
        for node in self.nodes:
            arrived = [(t, q) for t, q in node.pipeline if t <= self.current_step]
            for _, qty in arrived:
                node.inventory += qty
                node.inventory = min(node.inventory, node.capacity)
            node.pipeline = [(t, q) for t, q in node.pipeline if t > self.current_step]
        
        # 2. Get external demand at retailers
        demand = self._get_demand(self.current_step)
        
        # 3. Fulfill demand at retailers
        for i, node in enumerate(self.nodes):
            if node.node_type == "retailer":
                d = demand[i]
                fulfilled = min(node.inventory, d)
                node.inventory -= fulfilled
                shortage = d - fulfilled
                
                # Costs
                total_costs[self.agent_ids[i]] += shortage * self.shortage_cost
                infos[self.agent_ids[i]]["demand"] = d
                infos[self.agent_ids[i]]["fulfilled"] = fulfilled
                infos[self.agent_ids[i]]["shortage"] = shortage
        
        # 4. Process orders (actions)
        period_orders = []
        for i, (agent_id, node) in enumerate(zip(self.agent_ids, self.nodes)):
            action = actions.get(agent_id, np.array([0.0]))
            
            # Action space transformation: Action is a squared base-stock target ratio [0, 1] mapped to capacity
            target_inventory = (float(action[0]) ** 2) * node.capacity
            pipeline_total = sum(q for _, q in node.pipeline)
            inventory_position = node.inventory + pipeline_total
            
            # Order to reach target, bounded by max_order for physical transport limits
            desired_order = max(0.0, target_inventory - inventory_position)
            order_qty = min(desired_order, float(self.max_order))
            
            if order_qty > 0:
                # Ordering cost
                total_costs[agent_id] += self.ordering_cost
                
                # Find upstream supplier
                upstream_ids = [j for j in range(self.num_agents)
                              if self.adjacency[i, j] > 0 and self.nodes[j].echelon > node.echelon]
                
                if upstream_ids:
                    supplier = self.nodes[upstream_ids[0]]
                    # Supplier fulfills from their inventory
                    actual_shipped = min(supplier.inventory, order_qty)
                    supplier.inventory -= actual_shipped
                    
                    # Add to pipeline with lead time
                    lt = self._get_lead_time()
                    node.pipeline.append((self.current_step + lt, actual_shipped))
                elif node.node_type == "supplier":
                    # Suppliers have unlimited raw material (external source)
                    lt = self._get_lead_time()
                    node.pipeline.append((self.current_step + lt, order_qty))
                
                period_orders.append(order_qty)
            else:
                period_orders.append(0.0)
            
            infos[agent_id]["order_placed"] = order_qty
        
        # 5. Holding costs
        for i, (agent_id, node) in enumerate(zip(self.agent_ids, self.nodes)):
            holding = max(0, node.inventory) * self.holding_cost
            total_costs[agent_id] += holding
            infos[agent_id]["holding_cost"] = holding
        
        # 6. Compute rewards (negative costs — we want to minimize cost)
        rewards = {
            agent_id: -total_costs[agent_id] 
            for agent_id in self.agent_ids
        }
        
        # Add shared component to reward (cooperative)
        # NOTE: 0.01 instead of 0.1 — with 6 agents, 0.1 meant 60% total penalty
        # which drowned individual learning signals
        total_system_cost = sum(total_costs.values())
        for agent_id in self.agent_ids:
            rewards[agent_id] -= 0.01 * total_system_cost  # Cooperative penalty
        
        # Scale rewards to keep critic MSE loss stable
        # Without this, raw costs (~50-500) cause value loss to explode
        for agent_id in self.agent_ids:
            rewards[agent_id] /= 1000.0
        
        # 7. Track episode data
        self.episode_data["inventory_history"].append(
            [node.inventory for node in self.nodes]
        )
        self.episode_data["order_history"].append(period_orders)
        self.episode_data["demand_history"].append(demand.tolist() if isinstance(demand, np.ndarray) else demand)
        self.episode_data["cost_history"].append(total_system_cost)
        
        # 8. Step forward
        self.current_step += 1
        
        # Check termination
        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: self.current_step >= self.num_periods for agent_id in self.agent_ids}
        terminated["__all__"] = False
        truncated["__all__"] = self.current_step >= self.num_periods
        
        obs = self._get_obs()
        
        return obs, rewards, terminated, truncated, infos
    
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get current graph data for GNN processing.
        Returns node features and edge indices.
        """
        node_features = []
        for node in self.nodes:
            pipeline_total = sum(qty for _, qty in node.pipeline)
            node_features.append([
                node.inventory / node.capacity,
                pipeline_total / node.capacity,
                node.echelon / 2.0,  # Normalized echelon
                1.0 if node.node_type == "retailer" else 0.0,
                1.0 if node.node_type == "distributor" else 0.0,
                1.0 if node.node_type == "supplier" else 0.0,
            ])
        
        return {
            "node_features": np.array(node_features, dtype=np.float32),
            "edge_index": self.get_edge_index(),
            "adjacency": self.adjacency.copy(),
        }
