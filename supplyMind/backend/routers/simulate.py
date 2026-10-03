"""Simulation router — launch, monitor and cancel simulations"""
import uuid
import asyncio
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Simulation, Project, Dataset, get_db, User
from models.schemas import SimulationCreate, SimulationOut, SimulationDetail, WhatIfRequest, WhatIfResult
from routers.auth import get_current_user
from tasks.simulation import run_simulation_sync, broadcast

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)


def _load_demand_data(project) -> np.ndarray | None:
    """Load demand column from the project's uploaded dataset (sync, called in background thread)."""
    try:
        if not project.dataset:
            return None
        path = project.dataset.file_path
        if not path or not os.path.exists(path):
            return None
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        # Find demand column via stored column mapping or heuristic
        col_map = project.dataset.columns or {}
        demand_col = col_map.get("demand") or col_map.get("sales") or col_map.get("quantity")
        if not demand_col:
            # Heuristic: first numeric column that looks like demand/sales
            for c in df.columns:
                if any(kw in c.lower() for kw in ["demand", "sale", "qty", "quantity", "unit"]):
                    demand_col = c
                    break
        if demand_col and demand_col in df.columns:
            return df[demand_col].dropna().values.astype(float)
        # Fallback: first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return df[numeric_cols[0]].dropna().values.astype(float)
    except Exception:
        pass
    return None


async def _run_and_store(sim_id: str, algorithm: str, scenario: str,
                         episodes: int, topology, demand_data,
                         horizon: int = 365, holding_cost_mult: float = 1.0,
                         stockout_penalty_mult: float = 1.0,
                         custom_disruptions=None):
    """Background coroutine: run simulation in thread, update DB when done."""
    from models.database import AsyncSessionLocal

    # Mark as running
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Simulation).where(Simulation.id == sim_id))
        sim = result.scalar_one_or_none()
        if sim:
            sim.status = "running"
            sim.started_at = datetime.now(timezone.utc)
            await db.commit()

    def on_progress(pct: float):
        broadcast(sim_id, {"type": "progress", "simulation_id": sim_id, "progress": pct})

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            _executor,
            lambda: run_simulation_sync(
                sim_id, algorithm, scenario, episodes, topology, demand_data,
                on_progress,
                horizon=horizon,
                holding_cost_mult=holding_cost_mult,
                stockout_penalty_mult=stockout_penalty_mult,
                custom_disruptions=custom_disruptions,
            ),
        )
        status = "completed"
        error_msg = None
    except Exception as e:
        results = {}
        status = "failed"
        error_msg = str(e)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Simulation).where(Simulation.id == sim_id))
        sim = result.scalar_one_or_none()
        if sim:
            sim.status = status
            sim.progress = 100.0
            sim.completed_at = datetime.now(timezone.utc)
            sim.kpis         = results.get("kpis")
            sim.timeline     = results.get("timeline")
            sim.agent_logs   = results.get("agent_logs")
            sim.llm_summary  = results.get("llm_summary")
            sim.error_msg    = error_msg
            await db.commit()

    broadcast(sim_id, {
        "type": "completed" if status == "completed" else "failed",
        "simulation_id": sim_id,
        "status": status,
        "kpis": results.get("kpis"),
    })


@router.post("/", response_model=SimulationOut, status_code=202)
async def launch_simulation(
    body: SimulationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify project ownership
    proj_result = await db.execute(
        select(Project).where(Project.id == body.project_id, Project.owner_id == current_user.id)
    )
    project = proj_result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    sim = Simulation(
        id=str(uuid.uuid4()),
        project_id=body.project_id,
        algorithm=body.algorithm,
        scenario=body.scenario,
        episodes=body.episodes,
        status="pending",
        progress=0.0,
    )
    db.add(sim)
    await db.commit()
    await db.refresh(sim)

    background_tasks.add_task(
        _run_and_store,
        sim.id,
        body.algorithm,
        body.scenario,
        body.episodes,
        project.topology,
        _load_demand_data(project),
        body.horizon,
        body.holding_cost_mult,
        body.stockout_penalty_mult,
        [d.model_dump() for d in body.custom_disruptions],
    )

    return sim


@router.get("/{sim_id}", response_model=SimulationDetail)
async def get_simulation(
    sim_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Simulation)
        .join(Project)
        .where(Simulation.id == sim_id, Project.owner_id == current_user.id)
    )
    sim = result.scalar_one_or_none()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return sim


@router.get("/project/{project_id}", response_model=list[SimulationOut])
async def list_simulations(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proj_result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    if not proj_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(Simulation)
        .where(Simulation.project_id == project_id)
        .order_by(Simulation.created_at.desc())
    )
    return result.scalars().all()


@router.post("/whatif", response_model=WhatIfResult)
async def what_if(
    body: WhatIfRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fast what-if inference — no retraining, just inference with scenario override."""
    result = await db.execute(
        select(Simulation)
        .join(Project)
        .where(Simulation.id == body.simulation_id, Project.owner_id == current_user.id)
    )
    sim = result.scalar_one_or_none()
    if not sim or sim.status != "completed":
        raise HTTPException(status_code=404, detail="Completed simulation not found")

    # Map scenario type to disruption config
    scenario_map = {
        "demand_spike":       "demand_spike",
        "supplier_failure":   "node_failure",
        "lead_time_delay":    "lead_time_delay",
        "custom":             "combined",
    }
    scenario = scenario_map.get(body.scenario_type, "demand_spike")

    loop = asyncio.get_event_loop()
    what_if_results = await loop.run_in_executor(
        _executor,
        lambda: run_simulation_sync(
            f"whatif_{body.simulation_id}", sim.algorithm, scenario, 5, None, None
        ),
    )

    base_kpis  = sim.kpis or {}
    what_kpis  = what_if_results.get("kpis", {})

    fill_delta = what_kpis.get("fill_rate", 95.0) - base_kpis.get("fill_rate", 99.0)
    cost_delta = what_kpis.get("total_cost", 400000) - base_kpis.get("total_cost", 300000)

    recs = what_if_results.get("llm_summary", {}).get("recommendations", [])

    return WhatIfResult(
        scenario_description=_scenario_description(body.scenario_type, body.parameters),
        fill_rate_delta=round(fill_delta, 2),
        cost_delta=round(cost_delta, 2),
        recovery_days=int(abs(fill_delta) * 2 + 5),
        recommendations=[{
            "agent_id": r.get("agent_id", ""),
            "agent_name": r.get("agent_name", ""),
            "action": r.get("action", ""),
            "reason": r.get("reason", ""),
            "confidence": r.get("confidence", "Medium"),
            "urgency": r.get("urgency", "Normal"),
            "quantity": r.get("quantity", 0),
            "cost_estimate": r.get("cost_estimate", 0.0),
        } for r in recs],
        llm_narrative=what_if_results.get("llm_summary", {}).get("global", ""),
    )


def _scenario_description(scenario_type: str, params: dict) -> str:
    descriptions = {
        "demand_spike":     "Demand increases by 200% for 3 weeks across all retailers.",
        "supplier_failure": "Primary supplier goes offline for 20 days.",
        "lead_time_delay":  "All lead times double due to logistics disruption.",
        "custom":           f"Custom scenario with parameters: {params}",
    }
    return descriptions.get(scenario_type, "Unknown scenario")
