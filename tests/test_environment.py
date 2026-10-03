"""
Environment Sanity Test
========================
Quick test to verify the environment and disruption wrapper work correctly.

Usage:
    python tests/test_environment.py
"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.environment.supply_chain_env import MultiEchelonInventoryEnv
from src.environment.disruption_wrapper import DisruptionWrapper, DisruptionType
from src.utils.data_processing import generate_synthetic_demand
from src.agents.graph import SupplyChainGraph


def test_basic_environment():
    """Test that the environment runs without errors."""
    print("\n  Test 1: Basic Environment")
    
    config = {
        "num_retailers": 3,
        "num_distributors": 2,
        "num_suppliers": 1,
        "num_periods": 50,
    }
    
    env = MultiEchelonInventoryEnv(config)
    
    # Generate synthetic demand
    demand = generate_synthetic_demand(num_periods=50, num_retailers=3)
    env.set_demand_data(demand)
    
    obs, info = env.reset(seed=42)
    
    assert len(obs) == env.num_agents, f"Expected {env.num_agents} obs, got {len(obs)}"
    print(f"    ✅ Reset OK — {env.num_agents} agents, obs shapes: {[obs[k].shape for k in obs]}")
    
    # Run 50 steps with random actions
    total_reward = 0
    for step in range(50):
        actions = {
            agent_id: np.array([np.random.random()], dtype=np.float32)
            for agent_id in env.agent_ids
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += sum(rewards.values())
    
    print(f"    ✅ Ran 50 steps — Total reward: {total_reward:.2f}")
    print(f"    ✅ Episode data: {len(env.episode_data['cost_history'])} entries")
    

def test_disruption_wrapper():
    """Test disruption wrapper."""
    print("\n  Test 2: Disruption Wrapper")
    
    config = {
        "num_retailers": 3,
        "num_distributors": 2,
        "num_suppliers": 1,
        "num_periods": 100,
    }
    
    env = MultiEchelonInventoryEnv(config)
    demand = generate_synthetic_demand(num_periods=100, num_retailers=3)
    env.set_demand_data(demand)
    
    # Scheduled disruption for testing
    disruption_config = {
        "demand_spike_prob": 0.0,  # Disable stochastic
        "lead_time_delay_prob": 0.0,
        "node_failure_prob": 0.0,
        "scheduled_disruptions": [
            {
                "type": DisruptionType.DEMAND_SPIKE.value,
                "start_step": 20,
                "duration": 5,
                "severity": 3.0,
                "affected_nodes": [0],
            },
            {
                "type": DisruptionType.NODE_FAILURE.value,
                "start_step": 50,
                "duration": 10,
                "severity": 1.0,
                "affected_nodes": [4],  # Distributor 2
            },
        ],
    }
    
    env_d = DisruptionWrapper(env, disruption_config)
    obs, info = env_d.reset(seed=42)
    
    disruptions_seen = set()
    for step in range(100):
        actions = {
            agent_id: np.array([0.3], dtype=np.float32)
            for agent_id in env.agent_ids
        }
        obs, rewards, terminated, truncated, info = env_d.step(actions)
        
        for agent_id in env.agent_ids:
            for dt in info[agent_id].get("disruption_types", []):
                disruptions_seen.add(DisruptionType(dt).name)
    
    summary = env_d.get_disruption_summary()
    print(f"    ✅ Disruptions triggered: {disruptions_seen}")
    print(f"    ✅ Total disruptions: {summary['total_disruptions']}")
    print(f"    ✅ By type: {summary['by_type']}")


def test_graph():
    """Test supply chain graph construction."""
    print("\n  Test 3: Supply Chain Graph")
    
    graph = SupplyChainGraph.from_env_config(
        num_retailers=3,
        num_distributors=2,
        num_suppliers=1,
    )
    
    print(f"    ✅ Nodes: {graph.num_nodes}")
    print(f"    ✅ Edges: {graph.edge_index.shape[1]}")
    print(f"    ✅ Node types: {graph.node_types}")
    print(f"    ✅ Edge index shape: {graph.edge_index.shape}")
    
    # Test adjacency
    adj = graph.get_adjacency_matrix()
    print(f"    ✅ Adjacency matrix shape: {adj.shape}")
    
    # Test neighbors
    for i in range(graph.num_nodes):
        neighbors = graph.get_neighbors(i)
        print(f"      Node {i} ({graph.node_types[i]}): neighbors = {neighbors}")
    
    # Test PyG conversion
    try:
        pyg_data = graph.to_pyg_data()
        print(f"    ✅ PyG Data: {pyg_data}")
    except ImportError:
        print(f"    ⚠️  torch_geometric not installed, skipping PyG test")


def test_gnn_communication():
    """Test GNN communication layer."""
    print("\n  Test 4: GNN Communication")
    
    try:
        from src.agents.gnn_comm import GNNCommunication
        import torch
        
        num_nodes = 6
        input_dim = 12  # obs_dim
        
        gnn = GNNCommunication(
            input_dim=input_dim,
            hidden_dim=32,
            output_dim=16,
            num_layers=2,
            gnn_type="gcn",
        )
        
        # Fake data
        node_features = torch.randn(num_nodes, input_dim)
        adj = torch.zeros(num_nodes, num_nodes)
        adj[0, 3] = adj[3, 0] = 1  # R0 ↔ D0
        adj[1, 3] = adj[3, 1] = 1  # R1 ↔ D0
        adj[2, 4] = adj[4, 2] = 1  # R2 ↔ D1
        adj[3, 5] = adj[5, 3] = 1  # D0 ↔ S0
        adj[4, 5] = adj[5, 4] = 1  # D1 ↔ S0
        
        output = gnn(node_features, adj)
        print(f"    ✅ GNN output shape: {output.shape}")
        assert output.shape == (num_nodes, 16), f"Expected ({num_nodes}, 16), got {output.shape}"
        
        # Test batched
        batch_features = torch.randn(4, num_nodes, input_dim)
        batch_output = gnn(batch_features, adj)
        print(f"    ✅ Batched GNN output shape: {batch_output.shape}")
        
        print(f"    ✅ GNN parameters: {sum(p.numel() for p in gnn.parameters()):,}")
        
    except ImportError as e:
        print(f"    ⚠️  Skipping GNN test: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  ENVIRONMENT & COMPONENT TESTS")
    print("=" * 60)
    
    test_basic_environment()
    test_disruption_wrapper()
    test_graph()
    test_gnn_communication()
    
    print("\n" + "=" * 60)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 60)
