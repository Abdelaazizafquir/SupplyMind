"""Pydantic schemas — request/response validation"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    id: str
    email: str
    name: Optional[str]
    avatar_url: Optional[str]

    model_config = {"from_attributes": True}


# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    mode: str = "simulation"  # "simulation" or "live"


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    topology: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    mode: str = "simulation"
    topology: Optional[Dict[str, Any]]
    dataset_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Dataset / Upload ──────────────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    date: Optional[str] = None
    sku_id: Optional[str] = None
    quantity: Optional[str] = None
    store_id: Optional[str] = None
    price: Optional[str] = None


class DatasetOut(BaseModel):
    id: str
    filename: str
    original_name: Optional[str]
    row_count: Optional[int]
    columns: Optional[Dict[str, Any]]
    eda_summary: Optional[Dict[str, Any]]
    uploaded_at: datetime

    model_config = {"from_attributes": True}


# ── Topology (Agent Canvas) ───────────────────────────────────────────────────

class AgentNode(BaseModel):
    id: str
    type: str                         # "supplier"|"distributor"|"manufacturer"|"retailer"
    name: str
    capacity: int = 500
    lead_time_min: int = 1
    lead_time_max: int = 3
    holding_cost: float = 1.0
    stockout_penalty: float = 500.0
    policy: str = "mappo_gnn"         # "mappo_gnn"|"ss_policy"|"base_stock"
    # visual position
    position_x: float = 0
    position_y: float = 0


class TopologyEdge(BaseModel):
    id: str
    source: str
    target: str
    transport_cost: float = 0.1
    capacity: Optional[int] = None


class Topology(BaseModel):
    agents: List[AgentNode]
    edges: List[TopologyEdge]


# ── Simulation ────────────────────────────────────────────────────────────────

class CustomDisruption(BaseModel):
    node_id: str          # e.g. "agent_0" or "supplier_0"
    start_day: int
    end_day: int
    disruption_type: int  # 1=demand spike, 2=lead-time delay, 3=node failure
    magnitude: float = 2.0


class SimulationCreate(BaseModel):
    project_id: str
    algorithm: str = "mappo_gnn"
    scenario: str = "normal"
    episodes: int = 300
    horizon: int = 365
    holding_cost_mult: float = 1.0
    stockout_penalty_mult: float = 1.0
    custom_disruptions: List[CustomDisruption] = []

    @field_validator("algorithm")
    @classmethod
    def valid_algorithm(cls, v: str) -> str:
        allowed = {"mappo_gnn", "mappo", "ss_policy", "base_stock", "all"}
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}")
        return v

    @field_validator("scenario")
    @classmethod
    def valid_scenario(cls, v: str) -> str:
        allowed = {"normal", "demand_spike", "lead_time_delay", "node_failure", "combined"}
        if v not in allowed:
            raise ValueError(f"scenario must be one of {allowed}")
        return v

    @field_validator("episodes")
    @classmethod
    def clamp_episodes(cls, v: int) -> int:
        return max(10, min(v, 2000))

    @field_validator("horizon")
    @classmethod
    def clamp_horizon(cls, v: int) -> int:
        return max(30, min(v, 365))

    @field_validator("holding_cost_mult", "stockout_penalty_mult")
    @classmethod
    def clamp_cost_mult(cls, v: float) -> float:
        return max(0.1, min(v, 10.0))


class SimulationOut(BaseModel):
    id: str
    project_id: str
    algorithm: str
    scenario: str
    episodes: int
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    kpis: Optional[Dict[str, Any]]

    model_config = {"from_attributes": True}


class SimulationDetail(SimulationOut):
    timeline: Optional[List[Dict[str, Any]]]
    agent_logs: Optional[List[Dict[str, Any]]]
    llm_summary: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]]


# ── LLM / Recommendations ─────────────────────────────────────────────────────

class RecommendationCard(BaseModel):
    agent_id: str
    agent_name: str
    action: str          # "Order X units from ..."
    reason: str          # one-sentence explanation
    confidence: str      # "High" | "Medium" | "Low"
    urgency: str         # "Urgent" | "Normal" | "Monitor"
    quantity: int
    cost_estimate: float
    lead_time_days: int = 1


class WhatIfRequest(BaseModel):
    simulation_id: str
    scenario_type: str   # "demand_spike"|"supplier_failure"|"lead_time_delay"|"custom"
    parameters: Dict[str, Any] = {}


class WhatIfResult(BaseModel):
    scenario_description: str
    fill_rate_delta: float
    cost_delta: float
    recovery_days: int
    recommendations: List[RecommendationCard]
    llm_narrative: str


# ── DB Connector ──────────────────────────────────────────────────────────────

class DBConnectionConfig(BaseModel):
    db_type: str            # "postgresql" | "mysql" | "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str           # DB name or file path for SQLite
    username: str = ""
    password: str = ""
    table_name: str = ""
    inventory_col: str = "inventory"
    demand_col: str = "demand"
    node_col: str = "node_id"

    @field_validator("db_type")
    @classmethod
    def valid_db_type(cls, v: str) -> str:
        allowed = {"postgresql", "mysql", "sqlite"}
        if v not in allowed:
            raise ValueError(f"db_type must be one of {allowed}")
        return v


class DBPreviewResponse(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    row_count: int


class TableConfig(BaseModel):
    """Per-table import configuration (column mapping for one table)."""
    table_name: str
    inventory_col: str = "inventory"
    demand_col: str = "demand"
    node_col: str = "node_id"


class MultiTableImportRequest(BaseModel):
    """Import from multiple tables in one request."""
    db_type: str
    host: str = "localhost"
    port: int = 5432
    database: str
    username: str = ""
    password: str = ""
    tables: List[TableConfig]

    @field_validator("db_type")
    @classmethod
    def valid_db_type(cls, v: str) -> str:
        if v not in {"postgresql", "mysql", "sqlite"}:
            raise ValueError("db_type must be postgresql, mysql, or sqlite")
        return v


# ── Live Control ──────────────────────────────────────────────────────────────

class NodeInventoryInput(BaseModel):
    node_id: str                    # user-defined, e.g. "store_paris"
    node_name: str = ""             # display name (optional)
    node_type: str = "retailer"     # "retailer" | "distributor" | "supplier"
    inventory_units: float
    capacity_units: float = 1000.0
    demand_units: float = 20.0
    unit_cost: Optional[float] = None
    lead_time_days: Optional[int] = None

    @field_validator("node_type")
    @classmethod
    def valid_node_type(cls, v: str) -> str:
        if v not in {"retailer", "distributor", "supplier"}:
            raise ValueError("node_type must be retailer, distributor, or supplier")
        return v


class LiveControlRequest(BaseModel):
    project_id: str
    nodes: List[NodeInventoryInput]     # 1–20 nodes, any type mix

    @field_validator("nodes")
    @classmethod
    def valid_nodes_list(cls, v: list) -> list:
        if not v:
            raise ValueError("At least 1 node is required")
        if len(v) > 10000:
            raise ValueError("Maximum 10000 nodes supported per request for Live Control")
        return v


class NodeRecommendation(BaseModel):
    node_id: str
    node_name: str
    node_type: str
    inventory_units: float
    inventory_pct: float
    capacity_units: float
    order_units: float
    urgency: str            # "critical" | "high" | "medium" | "low"
    estimated_cost: float
    estimated_lead_time_days: int


class LiveControlResponse(BaseModel):
    recommendations: List[NodeRecommendation]
    total_order_units: float
    total_estimated_cost: float
    analysis: str           # Groq LLM explanation
    topology_note: str = "" # e.g. "8 nodes mapped to 6-agent model"


class FeatureExplanation(BaseModel):
    feature_name: str
    shap_value: float

class ExplainRequest(BaseModel):
    project_id: str
    nodes: List[NodeInventoryInput]
    target_node_id: str

class ExplainResponse(BaseModel):
    node_id: str
    base_value: float
    positive_features: List[FeatureExplanation]
    negative_features: List[FeatureExplanation]
    summary: str
