"""
Supply Chain Graph Module
==========================
Defines the supply chain network as a graph for GNN-based communication.
Uses PyTorch Geometric data structures.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SupplyChainGraph:
    """
    Graph representation of the supply chain network.
    
    Attributes:
        num_nodes: Total number of nodes (agents)
        node_types: Type of each node ("retailer", "distributor", "supplier")
        edge_index: COO-format edge indices [2, num_edges]
        edge_attr: Edge features (lead time, capacity, cost)
        node_features: Node feature matrix [num_nodes, feature_dim]
    """
    num_nodes: int
    node_types: List[str]
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    node_features: Optional[torch.Tensor] = None
    
    @classmethod
    def from_env_config(
        cls,
        num_retailers: int = 3,
        num_distributors: int = 2,
        num_suppliers: int = 1,
        lead_time_mean: float = 3.0,
    ) -> "SupplyChainGraph":
        """
        Build graph from environment configuration.
        
        Network topology:
            Suppliers → Distributors → Retailers
        """
        num_nodes = num_retailers + num_distributors + num_suppliers
        
        # Node types
        node_types = (
            ["retailer"] * num_retailers +
            ["distributor"] * num_distributors +
            ["supplier"] * num_suppliers
        )
        
        # Build edges (bidirectional)
        src, dst = [], []
        edge_features = []
        
        # Retailers → Distributors
        for r in range(num_retailers):
            d_idx = num_retailers + (r % num_distributors)
            # Forward (order flow: retailer → distributor)
            src.append(r)
            dst.append(d_idx)
            edge_features.append([lead_time_mean, 1.0, 0])  # [lead_time, capacity_norm, edge_type]
            
            # Backward (material flow: distributor → retailer)
            src.append(d_idx)
            dst.append(r)
            edge_features.append([lead_time_mean, 1.0, 1])
        
        # Distributors → Suppliers
        for d in range(num_distributors):
            d_idx = num_retailers + d
            s_idx = num_retailers + num_distributors + (d % num_suppliers)
            
            # Forward
            src.append(d_idx)
            dst.append(s_idx)
            edge_features.append([lead_time_mean * 1.5, 1.0, 0])
            
            # Backward
            src.append(s_idx)
            dst.append(d_idx)
            edge_features.append([lead_time_mean * 1.5, 1.0, 1])
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Initial node features (type encoding + echelon)
        node_feat_list = []
        for nt in node_types:
            if nt == "retailer":
                node_feat_list.append([1, 0, 0, 0.0])  # one-hot + echelon
            elif nt == "distributor":
                node_feat_list.append([0, 1, 0, 0.5])
            else:
                node_feat_list.append([0, 0, 1, 1.0])
        
        node_features = torch.tensor(node_feat_list, dtype=torch.float32)
        
        return cls(
            num_nodes=num_nodes,
            node_types=node_types,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_features=node_features,
        )
    
    def update_node_features(self, observations: Dict[str, np.ndarray], agent_ids: List[str]):
        """
        Update node features with current observations from environment.
        Called at each step before GNN forward pass.
        
        Args:
            observations: Dict of agent observations from environment
            agent_ids: List of agent IDs in order
        """
        features = []
        for agent_id in agent_ids:
            obs = observations[agent_id]
            features.append(obs)
        
        # Concatenate static node features with dynamic observations
        dynamic = torch.tensor(np.array(features), dtype=torch.float32)
        
        if self.node_features is not None:
            self.node_features = torch.cat([
                self.node_features[:, :4],  # Keep static features
                dynamic
            ], dim=1)
        else:
            self.node_features = dynamic
    
    def to_pyg_data(self):
        """
        Convert to PyTorch Geometric Data object.
        Requires torch_geometric to be installed.
        """
        try:
            from torch_geometric.data import Data
            
            return Data(
                x=self.node_features,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                num_nodes=self.num_nodes,
            )
        except ImportError:
            raise ImportError(
                "torch_geometric is required for GNN features. "
                "Install with: pip install torch-geometric"
            )
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get dense adjacency matrix."""
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            adj[src, dst] = 1.0
        return adj
    
    def get_neighbors(self, node_idx: int) -> List[int]:
        """Get neighbor indices for a node."""
        mask = self.edge_index[0] == node_idx
        return self.edge_index[1, mask].tolist()
    
    def visualize(self, save_path: str = None):
        """
        Visualize the supply chain graph.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Position nodes by echelon
        positions = {}
        echelon_counts = {"retailer": 0, "distributor": 0, "supplier": 0}
        echelon_y = {"retailer": 0, "distributor": 1, "supplier": 2}
        
        num_per_type = {
            "retailer": sum(1 for nt in self.node_types if nt == "retailer"),
            "distributor": sum(1 for nt in self.node_types if nt == "distributor"),
            "supplier": sum(1 for nt in self.node_types if nt == "supplier"),
        }
        
        for i, nt in enumerate(self.node_types):
            count = echelon_counts[nt]
            total = num_per_type[nt]
            x = (count + 0.5) / total  # Centered x position
            y = echelon_y[nt]
            positions[i] = (x, y)
            echelon_counts[nt] += 1
        
        # Draw edges
        for e in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, e].item(), self.edge_index[1, e].item()
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.5))
        
        # Draw nodes
        colors = {"retailer": "#4CAF50", "distributor": "#2196F3", "supplier": "#FF9800"}
        for i, nt in enumerate(self.node_types):
            x, y = positions[i]
            ax.scatter(x, y, s=800, c=colors[nt], zorder=5, edgecolors="white", linewidths=2)
            ax.text(x, y, f"{nt[0].upper()}{i}", ha="center", va="center",
                   fontsize=10, fontweight="bold", color="white", zorder=6)
        
        # Legend
        patches = [mpatches.Patch(color=c, label=l.title()) for l, c in colors.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=11)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Retailers\n(Echelon 0)", "Distributors\n(Echelon 1)", "Suppliers\n(Echelon 2)"])
        ax.set_title("Supply Chain Network Topology", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Graph saved to: {save_path}")
        plt.show()
