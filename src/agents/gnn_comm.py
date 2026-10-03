"""
GNN Communication Layer for MAPPO
====================================
Graph Neural Network that enables structured communication
between agents in the multi-echelon supply chain.

Supports:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE

The GNN output augments each agent's observation before
being fed to the MAPPO policy/value networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer (Kipf & Welling, 2017)."""
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
        """
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Degree normalization
        deg = adj_hat.sum(dim=1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt.unsqueeze(1) * adj_hat * deg_inv_sqrt.unsqueeze(0)
        
        # Message passing
        support = self.linear(x)
        output = torch.mm(norm, support)
        
        return output


class GATLayer(nn.Module):
    """Graph Attention Layer (Veličković et al., 2018)."""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
        """
        N = x.size(0)
        h = self.W(x)  # [N, out_dim]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, heads, head_dim]
        
        # Compute attention scores
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [N, heads, N, head_dim]
        h_j = h.unsqueeze(0).expand(N, -1, -1, -1)  # [N, heads, N, head_dim]
        
        concat = torch.cat([h_i, h_j], dim=-1)  # [N, heads, N, 2*head_dim]
        e = (concat * self.a.unsqueeze(0).unsqueeze(2)).sum(dim=-1)  # [N, heads, N]
        e = self.leaky_relu(e)
        
        # Mask non-neighbors
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1)
        adj_with_self = adj_expanded + torch.eye(N, device=adj.device).unsqueeze(1)
        e = e.masked_fill(adj_with_self == 0, float('-inf'))
        
        # Softmax
        alpha = F.softmax(e, dim=-1)  # [N, heads, N]
        alpha = self.dropout(alpha)
        
        # Weighted aggregation
        out = torch.bmm(
            alpha.view(N * self.num_heads, 1, N).squeeze(1).unsqueeze(1),
            h.view(N, self.num_heads, self.head_dim).permute(1, 0, 2).reshape(self.num_heads * N, 1, self.head_dim).squeeze(1).unsqueeze(1)
        )
        
        # Simplified: use matrix multiply per head
        out_list = []
        for head in range(self.num_heads):
            attn = alpha[:, head, :]  # [N, N]
            head_feat = h[:, head, :]  # [N, head_dim]
            out_list.append(torch.mm(attn, head_feat))
        
        out = torch.cat(out_list, dim=-1)  # [N, out_dim]
        
        return out


class GNNCommunication(nn.Module):
    """
    Multi-layer GNN for agent communication in supply chain.
    
    Takes agent observations and supply chain graph structure,
    produces communication embeddings that augment each agent's
    observation space.
    
    Args:
        input_dim: Dimension of node features (obs_dim + static_features)
        hidden_dim: Hidden dimension of GNN layers
        output_dim: Output dimension (communication embedding size)
        num_layers: Number of GNN layers
        gnn_type: Type of GNN layer ("gcn", "gat")
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_d = hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNLayer(in_d, out_d))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATLayer(in_d, out_d, num_heads=4))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.layer_norms.append(nn.LayerNorm(out_d))
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through GNN communication layer.
        
        Args:
            node_features: [num_nodes, input_dim] or [batch, num_nodes, input_dim]
            adjacency: [num_nodes, num_nodes]
            
        Returns:
            communication_embeddings: [num_nodes, output_dim]
        """
        # Handle batched input
        if node_features.dim() == 3:
            batch_size = node_features.size(0)
            outputs = []
            for b in range(batch_size):
                out = self._forward_single(node_features[b], adjacency)
                outputs.append(out)
            return torch.stack(outputs)
        else:
            return self._forward_single(node_features, adjacency)
    
    def _forward_single(
        self, 
        node_features: torch.Tensor, 
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for a single graph."""
        h = self.input_proj(node_features)
        h = F.relu(h)
        
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_prev = h
            h = gnn(h, adjacency)
            h = norm(h)
            
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
                # Residual connection (if dimensions match)
                if h.shape == h_prev.shape:
                    h = h + h_prev
        
        return h
    
    def get_augmented_observations(
        self,
        observations: dict,
        agent_ids: list,
        static_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict:
        """
        Augment agent observations with GNN communication embeddings.
        
        Args:
            observations: Dict of agent observations {agent_id: obs_array}
            agent_ids: List of agent IDs
            static_features: Static node features [num_nodes, static_dim]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            augmented_observations: Dict with GNN embeddings concatenated
        """
        # Stack observations
        obs_list = [torch.tensor(observations[aid], dtype=torch.float32) for aid in agent_ids]
        obs_tensor = torch.stack(obs_list)  # [num_agents, obs_dim]
        
        # Concatenate with static features
        if static_features is not None:
            node_input = torch.cat([obs_tensor, static_features], dim=-1)
        else:
            node_input = obs_tensor
        
        # GNN forward
        with torch.no_grad():
            comm_embeddings = self.forward(node_input, adjacency)
        
        # Augment observations
        augmented = {}
        for i, aid in enumerate(agent_ids):
            aug_obs = torch.cat([obs_tensor[i], comm_embeddings[i]])
            augmented[aid] = aug_obs.numpy()
        
        return augmented
