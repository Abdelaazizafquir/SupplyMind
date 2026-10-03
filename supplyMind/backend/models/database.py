"""
Database models — SQLAlchemy ORM + async init
Uses SQLite for local dev, PostgreSQL in production (set DATABASE_URL env var)
"""
import os
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from datetime import datetime, timezone

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./supplymind.db"
)

# For sync operations (Celery tasks)
SYNC_DATABASE_URL = DATABASE_URL.replace("+aiosqlite", "").replace("postgresql+asyncpg", "postgresql")

engine = create_async_engine(DATABASE_URL, echo=False)
sync_engine = create_engine(SYNC_DATABASE_URL.replace("postgresql+asyncpg", "postgresql"), echo=False)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
SyncSessionLocal = sessionmaker(sync_engine)

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(String, primary_key=True)   # UUID
    email      = Column(String, unique=True, nullable=False, index=True)
    name       = Column(String)
    avatar_url = Column(String)
    provider   = Column(String, default="email")    # "google" | "github" | "email"
    hashed_pw  = Column(String)                     # null for OAuth users
    created_at = Column(DateTime, default=utcnow)

    projects   = relationship("Project", back_populates="owner", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id          = Column(String, primary_key=True)  # UUID
    owner_id    = Column(String, ForeignKey("users.id"), nullable=False)
    name        = Column(String, nullable=False)
    description = Column(Text)
    mode        = Column(String, default="simulation")  # "simulation" | "live"
    created_at  = Column(DateTime, default=utcnow)
    updated_at  = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Supply chain topology (serialized from React Flow canvas)
    topology    = Column(JSON)                      # {"agents": [...], "edges": [...]}
    # Raw demand data reference
    dataset_id  = Column(String, ForeignKey("datasets.id"))

    owner       = relationship("User", back_populates="projects")
    dataset     = relationship("Dataset", back_populates="projects")
    simulations = relationship("Simulation", back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id           = Column(String, primary_key=True)
    owner_id     = Column(String, ForeignKey("users.id"), nullable=False)
    filename     = Column(String, nullable=False)
    original_name= Column(String)
    file_path    = Column(String)                  # local path or S3 key
    file_size    = Column(Integer)                 # bytes
    row_count    = Column(Integer)
    columns      = Column(JSON)                    # detected column mapping
    eda_summary  = Column(JSON)                    # stats: mean, std, date_range, top_skus
    uploaded_at  = Column(DateTime, default=utcnow)

    projects     = relationship("Project", back_populates="dataset")


class Simulation(Base):
    __tablename__ = "simulations"

    id           = Column(String, primary_key=True)
    project_id   = Column(String, ForeignKey("projects.id"), nullable=False)
    algorithm    = Column(String, default="mappo_gnn")  # "mappo_gnn"|"mappo"|"ss_policy"|"base_stock"
    scenario     = Column(String, default="normal")     # "normal"|"demand_spike"|"lead_time_delay"|"node_failure"|"combined"
    episodes     = Column(Integer, default=300)
    status       = Column(String, default="pending")    # "pending"|"running"|"completed"|"failed"
    progress     = Column(Float, default=0.0)           # 0–100
    started_at   = Column(DateTime)
    completed_at = Column(DateTime)
    created_at   = Column(DateTime, default=utcnow)
    error_msg    = Column(Text)

    # Results stored as JSON blobs
    kpis         = Column(JSON)    # fill_rate, total_cost, bullwhip_ratio, disruptions_handled
    timeline     = Column(JSON)    # per-step inventory/orders/demand time series
    agent_logs   = Column(JSON)    # per-agent decisions
    llm_summary  = Column(JSON)    # {"global": "...", "agents": {"retailer_0": "..."}}
    comparison   = Column(JSON)    # side-by-side vs baselines

    project      = relationship("Project", back_populates="simulations")


# ── Session helpers ───────────────────────────────────────────────────────────

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
