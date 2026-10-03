"""Projects router — CRUD for supply chain projects"""
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Project, get_db, User
from models.schemas import ProjectCreate, ProjectOut, ProjectUpdate, Topology
from routers.auth import get_current_user

router = APIRouter()


@router.get("/templates", response_model=List[dict])
async def get_templates(_: User = Depends(get_current_user)):
    """Return built-in supply chain topology templates."""
    return [
        {
            "id": "simple",
            "name": "Simple (3 nodes)",
            "description": "Supplier → Distributor → Retailer",
            "agents": [
                {"id": "supplier_0", "type": "supplier", "name": "Supplier A",
                 "capacity": 500, "lead_time_min": 2, "lead_time_max": 4,
                 "holding_cost": 0.5, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 100, "position_y": 200},
                {"id": "distributor_0", "type": "distributor", "name": "Distributor A",
                 "capacity": 300, "lead_time_min": 1, "lead_time_max": 2,
                 "holding_cost": 1.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 400, "position_y": 200},
                {"id": "retailer_0", "type": "retailer", "name": "Retailer A",
                 "capacity": 200, "lead_time_min": 1, "lead_time_max": 1,
                 "holding_cost": 2.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 700, "position_y": 200},
            ],
            "edges": [
                {"id": "e1", "source": "supplier_0", "target": "distributor_0", "transport_cost": 0.1},
                {"id": "e2", "source": "distributor_0", "target": "retailer_0", "transport_cost": 0.2},
            ],
        },
        {
            "id": "medium",
            "name": "Medium (6 nodes)",
            "description": "1 Supplier → 2 Distributors → 3 Retailers (PFE default)",
            "agents": [
                {"id": "supplier_0", "type": "supplier", "name": "Supplier A",
                 "capacity": 1000, "lead_time_min": 2, "lead_time_max": 5,
                 "holding_cost": 0.5, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 350, "position_y": 50},
                {"id": "distributor_0", "type": "distributor", "name": "Distributor A",
                 "capacity": 500, "lead_time_min": 1, "lead_time_max": 3,
                 "holding_cost": 1.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 150, "position_y": 250},
                {"id": "distributor_1", "type": "distributor", "name": "Distributor B",
                 "capacity": 500, "lead_time_min": 1, "lead_time_max": 3,
                 "holding_cost": 1.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 550, "position_y": 250},
                {"id": "retailer_0", "type": "retailer", "name": "Retailer A",
                 "capacity": 200, "lead_time_min": 1, "lead_time_max": 1,
                 "holding_cost": 2.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 50, "position_y": 450},
                {"id": "retailer_1", "type": "retailer", "name": "Retailer B",
                 "capacity": 200, "lead_time_min": 1, "lead_time_max": 1,
                 "holding_cost": 2.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 350, "position_y": 450},
                {"id": "retailer_2", "type": "retailer", "name": "Retailer C",
                 "capacity": 200, "lead_time_min": 1, "lead_time_max": 1,
                 "holding_cost": 2.0, "stockout_penalty": 500, "policy": "mappo_gnn",
                 "position_x": 650, "position_y": 450},
            ],
            "edges": [
                {"id": "e1", "source": "supplier_0", "target": "distributor_0", "transport_cost": 0.1},
                {"id": "e2", "source": "supplier_0", "target": "distributor_1", "transport_cost": 0.1},
                {"id": "e3", "source": "distributor_0", "target": "retailer_0", "transport_cost": 0.2},
                {"id": "e4", "source": "distributor_0", "target": "retailer_1", "transport_cost": 0.2},
                {"id": "e5", "source": "distributor_1", "target": "retailer_1", "transport_cost": 0.2},
                {"id": "e6", "source": "distributor_1", "target": "retailer_2", "transport_cost": 0.2},
            ],
        },
    ]


@router.get("/", response_model=List[ProjectOut])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.owner_id == current_user.id).order_by(Project.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = Project(
        id=str(uuid.uuid4()),
        owner_id=current_user.id,
        name=body.name,
        description=body.description,
        mode=body.mode,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(project, field, value)
    await db.commit()
    await db.refresh(project)
    return project


@router.put("/{project_id}/topology", response_model=ProjectOut)
async def save_topology(
    project_id: str,
    topology: Topology,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save the supply chain topology from the React Flow canvas."""
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.topology = topology.model_dump()
    await db.commit()
    await db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.delete(project)
    await db.commit()

