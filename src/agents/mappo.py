"""
MAPPO Agent Module — Parameter Sharing Architecture
=====================================================
Multi-Agent Proximal Policy Optimization with:
- Centralized Training, Decentralized Execution (CTDE)
- SHARED actor with role-type conditioning (topology-agnostic)
- Mean-pooled critic (works with any number of agents)
- GNN communication layer integration
- Support for continuous action spaces

The shared actor receives role-type encoding (retailer/distributor/supplier)
so a single model generalizes across different supply chain topologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .gnn_comm import GNNCommunication


# Number of distinct agent roles in a supply chain
NUM_ROLE_TYPES = 3  # retailer=0, distributor=1, supplier=2


@dataclass
class RolloutBuffer:
    """Stores rollout data for PPO training."""
    observations: List[Dict[str, np.ndarray]]
    actions: List[Dict[str, np.ndarray]]
    rewards: List[Dict[str, float]]
    dones: List[Dict[str, bool]]
    log_probs: List[Dict[str, float]]
    values: List[Dict[str, float]]
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.__init__()
    
    def __len__(self):
        return len(self.observations)


class ActorNetwork(nn.Module):
    """
    Shared Actor (Policy) network with role-type conditioning.
    
    Input: obs + GNN_embedding + role_type_onehot
    Output: mean and std for continuous action distribution.
    
    The role_type_onehot (3-dim) tells the network whether this agent is
    a retailer, distributor, or supplier — enabling a single network
    to handle all agent types.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_role_types: int = NUM_ROLE_TYPES,
    ):
        super().__init__()
        
        # Total input = obs features + role encoding
        total_input_dim = obs_dim + num_role_types
        
        layers = []
        in_dim = total_input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.num_role_types = num_role_types
    
    def forward(self, obs: torch.Tensor, role_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Agent observation [obs_dim] or [batch, obs_dim]
            role_type: One-hot role vector [num_role_types] or [batch, num_role_types]
        Returns:
            action mean and std
        """
        x = torch.cat([obs, role_type], dim=-1)
        features = self.backbone(x)
        mean = torch.sigmoid(self.mean_head(features))  # Actions in [0, 1]
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std
    
    def get_action(self, obs: torch.Tensor, role_type: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy."""
        mean, std = self.forward(obs, role_type)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, role_type: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probability and entropy of given actions."""
        mean, std = self.forward(obs, role_type)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized Critic with Mean-Pooling (topology-agnostic).
    
    Instead of concatenating all agents' states (which fixes the agent count),
    this critic processes each agent individually and then mean-pools across agents.
    This makes it work with ANY number of agents.
    
    Architecture:
        1. Per-agent encoder: maps each agent's augmented obs to a hidden representation
        2. Mean-pool across all agents → single global representation
        3. Value head: maps global representation to scalar value
    """
    
    def __init__(
        self,
        per_agent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Per-agent encoder
        encoder_layers = []
        in_dim = per_agent_dim
        for _ in range(num_layers):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.agent_encoder = nn.Sequential(*encoder_layers)
        
        # Value head (after mean-pooling)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_states: [num_agents, per_agent_dim] or [batch, num_agents, per_agent_dim]
        Returns:
            value estimate: scalar or [batch]
        """
        if agent_states.dim() == 2:
            # [num_agents, dim] -> [1, num_agents, dim]
            agent_states = agent_states.unsqueeze(0)
        
        # Encode each agent: [batch, num_agents, hidden_dim]
        encoded = self.agent_encoder(agent_states)
        
        # Mean-pool across agents: [batch, hidden_dim]
        pooled = encoded.mean(dim=1)
        
        # Value: [batch]
        value = self.value_head(pooled).squeeze(-1)
        return value


class MAPPOAgent:
    """
    Multi-Agent PPO with CTDE, parameter sharing, and optional GNN.
    
    Architecture:
        - SHARED Actor network with role-type conditioning (decentralized execution)
        - Mean-pooled Critic network (centralized training, topology-agnostic)
        - Optional GNN layer for structured communication
    
    This architecture works with ANY number of agents — no retraining needed
    when the supply chain topology changes.
    
    Args:
        num_agents: Number of agents (used for graph setup, not model size)
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        config: MAPPO configuration dict
        adjacency: Supply chain adjacency matrix [num_agents, num_agents]
        static_node_features: Static features for GNN [num_agents, feat_dim]
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int = 1,
        config: dict = None,
        adjacency: np.ndarray = None,
        static_node_features: np.ndarray = None,
    ):
        cfg = config or {}
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.lr = cfg.get("learning_rate", 3e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.num_epochs = cfg.get("num_epochs", 10)
        self.num_minibatches = cfg.get("num_minibatches", 4)
        
        # GNN config
        self.use_gnn = cfg.get("use_gnn", True)
        gnn_output_dim = cfg.get("gnn_hidden_dim", 64)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GNN Communication Layer
        if self.use_gnn and adjacency is not None:
            static_dim = static_node_features.shape[1] if static_node_features is not None else 0
            gnn_input_dim = obs_dim + static_dim
            
            self.gnn = GNNCommunication(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_output_dim,
                output_dim=gnn_output_dim,
                num_layers=cfg.get("gnn_num_layers", 2),
                gnn_type=cfg.get("gnn_type", "gcn"),
            ).to(self.device)
            
            actor_obs_dim = obs_dim + gnn_output_dim  # obs + GNN embedding
            
            self.adjacency = torch.tensor(adjacency, dtype=torch.float32).to(self.device)
            self.static_features = (
                torch.tensor(static_node_features, dtype=torch.float32).to(self.device)
                if static_node_features is not None else None
            )
        else:
            self.gnn = None
            actor_obs_dim = obs_dim
            self.adjacency = None
            self.static_features = None
        
        # SHARED Actor (one network for all agents, with role conditioning)
        self.actor = ActorNetwork(
            obs_dim=actor_obs_dim,  # GNN augmented obs dim (role is added inside)
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_role_types=NUM_ROLE_TYPES,
        ).to(self.device)
        
        # Keep self.actors as a ModuleList with single actor for backward compat
        self.actors = nn.ModuleList([self.actor])
        
        # Mean-pooled Critic (topology-agnostic)
        # Input per agent = augmented obs + role encoding
        critic_per_agent_dim = actor_obs_dim + NUM_ROLE_TYPES
        self.critic = CriticNetwork(
            per_agent_dim=critic_per_agent_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        # Optimizers
        actor_params = list(self.actor.parameters())
        if self.gnn is not None:
            actor_params.extend(self.gnn.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training stats
        self.training_step = 0
        self.training_stats = []
    
    def _get_role_type(self, agent_id: str, agent_ids: List[str]) -> torch.Tensor:
        """
        Get one-hot role type for an agent based on its position in the topology.
        
        Convention (from supply_chain_env.py):
            - First N agents are retailers (role=0)
            - Next M agents are distributors (role=1)
            - Last K agents are suppliers (role=2)
        
        The role is encoded in the observation vector itself (dims 6,7,8 are
        retailer/distributor/supplier indicators), but we also pass it explicitly
        to the actor for clean conditioning.
        """
        idx = agent_ids.index(agent_id)
        total = len(agent_ids)
        
        # Try to infer from environment node types
        # The observation already contains role indicators at positions 6,7,8
        # But we need a clean one-hot for the actor input
        role = torch.zeros(NUM_ROLE_TYPES, device=self.device)
        
        # Heuristic: use the observation's built-in role indicators
        # This is set during _get_obs in supply_chain_env.py
        # For now, we use position-based inference
        # (The env sets obs[6]=retailer, obs[7]=distributor, obs[8]=supplier)
        if idx < total:  # Always true, but defensive
            role_idx = 0  # Default: retailer
            # We'll override this in _augment_observations using the actual obs
        role[role_idx] = 1.0
        return role
    
    def _extract_role_from_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract role one-hot from the observation vector.
        Obs indices 6, 7, 8 correspond to is_retailer, is_distributor, is_supplier.
        
        Falls back to static_features if available and obs doesn't have 
        valid role indicators (sum != 1).
        """
        if obs_tensor.dim() == 1:
            role = obs_tensor[6:9].clone()
            # Validate: role should be a proper one-hot (sums to 1)
            if role.sum().item() < 0.5:
                # Obs doesn't contain role — fall back to zero (will be overridden)
                role = torch.zeros(NUM_ROLE_TYPES, device=obs_tensor.device)
        else:
            role = obs_tensor[:, 6:9].clone()
        return role
    
    def _get_gnn_embeddings(self, obs_dict: Dict[str, np.ndarray], agent_ids: List[str]) -> torch.Tensor:
        """Compute GNN communication embeddings."""
        obs_list = [torch.tensor(obs_dict[aid], dtype=torch.float32) for aid in agent_ids]
        obs_tensor = torch.stack(obs_list).to(self.device)
        
        if self.static_features is not None:
            gnn_input = torch.cat([obs_tensor, self.static_features], dim=-1)
        else:
            gnn_input = obs_tensor
        
        embeddings = self.gnn(gnn_input, self.adjacency)
        return embeddings
    
    def _augment_observations(
        self, obs_dict: Dict[str, np.ndarray], agent_ids: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Augment observations with GNN embeddings and extract role types.
        
        Role priority:
          1. Extract from obs indices 6-8 (if env embeds them)
          2. Fall back to static_features[:, 0:3] (from SupplyChainGraph)
        
        Returns:
            augmented_obs: dict of augmented observation tensors
            role_types: dict of role one-hot tensors
        """
        obs_list = [torch.tensor(obs_dict[aid], dtype=torch.float32) for aid in agent_ids]
        obs_tensor = torch.stack(obs_list).to(self.device)
        
        # Extract roles from observation
        roles = self._extract_role_from_obs(obs_tensor)  # [num_agents, 3]
        
        # Validate roles — if obs doesn't contain valid role indicators, 
        # use static features from the graph (indices 0-2 are role one-hot)
        role_sums = roles.sum(dim=-1)  # each agent should sum to 1.0
        if (role_sums < 0.5).any() and self.static_features is not None:
            roles = self.static_features[:len(agent_ids), :NUM_ROLE_TYPES].clone()
        
        if self.gnn is not None:
            gnn_embeddings = self._get_gnn_embeddings(obs_dict, agent_ids)
            augmented = torch.cat([obs_tensor, gnn_embeddings], dim=-1)
        else:
            augmented = obs_tensor
        
        aug_dict = {aid: augmented[i] for i, aid in enumerate(agent_ids)}
        role_dict = {aid: roles[i] for i, aid in enumerate(agent_ids)}
        
        return aug_dict, role_dict
    
    def get_actions(
        self, 
        obs_dict: Dict[str, np.ndarray], 
        agent_ids: List[str],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """
        Get actions for all agents using the SHARED actor.
        
        Returns:
            actions, log_probs, values (all as dicts)
        """
        with torch.no_grad():
            augmented, roles = self._augment_observations(obs_dict, agent_ids)
            
            actions = {}
            log_probs = {}
            
            for aid in agent_ids:
                action, log_prob = self.actor.get_action(
                    augmented[aid], roles[aid], deterministic
                )
                actions[aid] = action.cpu().numpy()
                log_probs[aid] = log_prob.item()
            
            # Centralized critic: stack all agent states with roles
            agent_states = torch.stack([
                torch.cat([augmented[aid], roles[aid]])
                for aid in agent_ids
            ])  # [num_agents, per_agent_dim]
            
            value = self.critic(agent_states).item()
            values = {aid: value for aid in agent_ids}
        
        return actions, log_probs, values
    
    def compute_gae(
        self,
        rewards: List[Dict[str, float]],
        values: List[Dict[str, float]],
        dones: List[Dict[str, bool]],
        agent_ids: List[str],
        last_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        Uses shared (cooperative) returns.
        """
        T = len(rewards)
        
        # Average rewards across agents (cooperative)
        avg_rewards = np.array([
            np.mean([rewards[t][aid] for aid in agent_ids])
            for t in range(T)
        ])
        
        avg_values = np.array([
            np.mean([values[t][aid] for aid in agent_ids])
            for t in range(T)
        ])
        
        any_done = np.array([
            any(dones[t].get(aid, False) for aid in agent_ids)
            for t in range(T)
        ], dtype=float)
        
        # GAE computation
        advantages = np.zeros(T)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - any_done[t]
            else:
                next_value = avg_values[t + 1]
                next_non_terminal = 1.0 - any_done[t]
            
            delta = avg_rewards[t] + self.gamma * next_value * next_non_terminal - avg_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + avg_values
        
        return advantages, returns
    
    def update(self, agent_ids: List[str], last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        Uses the SHARED actor for all agents.
        
        Args:
            agent_ids: List of agent IDs
            last_value: Bootstrap value V(s_T) for non-terminal rollout ends.
                        MUST be set to critic's value estimate when the rollout
                        is truncated mid-episode (not at a natural done state).
        
        Returns:
            Training statistics dict
        """
        if len(self.buffer) == 0:
            return {}
        
        # Compute GAE
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            agent_ids,
            last_value=last_value,
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        T = len(self.buffer)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        
        for epoch in range(self.num_epochs):
            # Mini-batch indices
            indices = np.random.permutation(T)
            batch_size = max(1, T // self.num_minibatches)
            
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = indices[start:end]
                
                total_policy_loss = 0.0
                total_entropy = 0.0
                
                for t_idx in batch_idx:
                    obs = self.buffer.observations[t_idx]
                    augmented, roles = self._augment_observations(obs, agent_ids)
                    
                    for aid in agent_ids:
                        old_log_prob = self.buffer.log_probs[t_idx][aid]
                        action = torch.tensor(
                            self.buffer.actions[t_idx][aid], 
                            dtype=torch.float32
                        ).to(self.device)
                        
                        # Shared actor with role conditioning
                        new_log_prob, entropy = self.actor.evaluate_actions(
                            augmented[aid], roles[aid], action
                        )
                        
                        # PPO clipped objective
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        adv = torch.tensor(advantages[t_idx], dtype=torch.float32).to(self.device)
                        
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                        
                        policy_loss = -torch.min(surr1, surr2)
                        total_policy_loss += policy_loss
                        total_entropy += entropy
                
                # Value loss (mean-pooled critic)
                value_targets = torch.tensor(
                    returns[batch_idx], dtype=torch.float32
                ).to(self.device)
                
                value_preds = []
                for t_idx in batch_idx:
                    obs = self.buffer.observations[t_idx]
                    augmented, roles = self._augment_observations(obs, agent_ids)
                    
                    # Stack agent states with roles for critic
                    agent_states = torch.stack([
                        torch.cat([augmented[aid], roles[aid]])
                        for aid in agent_ids
                    ])  # [num_agents, per_agent_dim]
                    
                    value_preds.append(self.critic(agent_states))
                
                value_preds = torch.cat(value_preds)
                value_loss = F.mse_loss(value_preds, value_targets)
                
                # Total loss
                n_samples = len(batch_idx) * len(agent_ids)
                avg_policy_loss = total_policy_loss / n_samples
                avg_entropy = total_entropy / n_samples
                
                loss = (
                    avg_policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * avg_entropy
                )
                
                # Optimize actor + GNN + critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                all_params = list(self.critic.parameters()) + list(self.actor.parameters())
                if self.gnn is not None:
                    all_params.extend(self.gnn.parameters())
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                stats["policy_loss"] += avg_policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += avg_entropy.item()
        
        # Average stats
        num_updates = self.num_epochs * max(1, T // max(1, T // self.num_minibatches))
        for key in stats:
            stats[key] /= max(1, num_updates)
        
        self.training_step += 1
        self.training_stats.append(stats)
        
        # Clear buffer
        self.buffer.clear()
        
        return stats
    
    def save(self, path: str):
        """Save model weights."""
        checkpoint = {
            "actor": self.actor.state_dict(),  # Single shared actor
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
            "training_stats": self.training_stats,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "use_gnn": self.use_gnn,
                "num_role_types": NUM_ROLE_TYPES,
            },
        }
        if self.gnn is not None:
            checkpoint["gnn"] = self.gnn.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  Model saved to: {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle both old (multi-actor) and new (shared actor) checkpoints
        if "actor" in checkpoint:
            # New format: single shared actor
            self.actor.load_state_dict(checkpoint["actor"])
        elif "actors" in checkpoint:
            # Legacy format: load first actor as shared actor
            print("  ⚠️  Loading legacy multi-actor checkpoint into shared actor")
            self.actor.load_state_dict(checkpoint["actors"][0])
        
        self.critic.load_state_dict(checkpoint["critic"])
        
        if self.gnn is not None and "gnn" in checkpoint:
            self.gnn.load_state_dict(checkpoint["gnn"])
        
        if "actor_optimizer" in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            except (ValueError, KeyError):
                print("  ⚠️  Optimizer state mismatch — reinitializing optimizers")
        
        self.training_step = checkpoint.get("training_step", 0)
        self.training_stats = checkpoint.get("training_stats", [])
        
        print(f"  Model loaded from: {path} (step {self.training_step})")

    def update_topology(self, adjacency: np.ndarray, static_node_features: np.ndarray = None):
        """
        Update the GNN topology for a new supply chain configuration.
        Call this when the user changes the number of agents.
        
        The actor and critic do NOT need to change — they are topology-agnostic.
        Only the GNN adjacency and static features need updating.
        """
        self.num_agents = adjacency.shape[0]
        self.adjacency = torch.tensor(adjacency, dtype=torch.float32).to(self.device)
        
        if static_node_features is not None:
            self.static_features = torch.tensor(
                static_node_features, dtype=torch.float32
            ).to(self.device)
        else:
            self.static_features = None
        
        print(f"  Topology updated: {self.num_agents} agents")
