"""
Configuration Management
========================
YAML-based configuration for all experiments.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@dataclass
class EnvironmentConfig:
    """Configuration for the supply chain environment."""
    num_retailers: int = 3
    num_distributors: int = 2
    num_suppliers: int = 1
    num_periods: int = 365
    lead_time_mean: int = 3
    lead_time_std: int = 1
    holding_cost: float = 1.0
    ordering_cost: float = 50.0
    shortage_cost: float = 500.0
    initial_inventory: int = 100
    max_order_quantity: int = 500
    demand_source: str = "m5"  # "m5" or "synthetic"


@dataclass
class DisruptionConfig:
    """Configuration for disruption scenarios."""
    enabled: bool = True
    demand_spike_prob: float = 0.02
    demand_spike_factor: float = 3.0
    lead_time_delay_prob: float = 0.03
    lead_time_delay_factor: float = 3.0
    node_failure_prob: float = 0.01
    node_failure_duration: int = 10
    cascade_enabled: bool = True


@dataclass
class ForecastingConfig:
    """Configuration for demand forecasting."""
    model_type: str = "patchtst"  # "patchtst", "prophet", "sarimax"
    horizon: int = 7
    lookback: int = 30
    patch_length: int = 16
    num_heads: int = 4
    d_model: int = 128
    num_layers: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5


@dataclass
class MAPPOConfig:
    """Configuration for the MAPPO algorithm."""
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"
    
    # GNN communication
    use_gnn: bool = True
    gnn_type: str = "gcn"  # "gcn", "gat", "sage"
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    
    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    num_minibatches: int = 4
    total_timesteps: int = 1_000_000
    
    # Execution
    num_workers: int = 4
    rollout_length: int = 128


@dataclass
class CoordinatorConfig:
    """Configuration for the LLM Coordinator."""
    llm_provider: str = "ollama"  # "ollama", "openai"
    llm_model: str = "llama3:8b"
    temperature: float = 0.3
    advisory_enabled: bool = True
    override_enabled: bool = True
    override_threshold: float = 0.8  # Disruption severity to trigger override
    explanation_enabled: bool = True
    max_tokens: int = 512


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    experiment_name: str = "default"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    disruption: DisruptionConfig = field(default_factory=DisruptionConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]
        if "seed" in data:
            config.seed = data["seed"]
        if "device" in data:
            config.device = data["device"]
        
        # Load sub-configs
        if "environment" in data:
            config.environment = EnvironmentConfig(**data["environment"])
        if "disruption" in data:
            config.disruption = DisruptionConfig(**data["disruption"])
        if "forecasting" in data:
            config.forecasting = ForecastingConfig(**data["forecasting"])
        if "mappo" in data:
            config.mappo = MAPPOConfig(**data["mappo"])
        if "coordinator" in data:
            config.coordinator = CoordinatorConfig(**data["coordinator"])
        
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict
        data = asdict(self)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
