# SupplyMind

AI-powered supply chain optimisation SaaS built on top of the PFE MAPPO+GNN research engine.

## Tech stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 14 · TypeScript · Tailwind CSS |
| Backend | FastAPI · SQLAlchemy 2 async · SQLite (dev) / PostgreSQL (prod) |
| AI engine | MAPPO + GNN (PyTorch Geometric) — trained in PFE research |
| Auth | JWT (python-jose) |
| Realtime | WebSocket (simulation progress) |

## Quick start (development)

### 1. Activate the existing PFE venv and install backend deps

```powershell
# From the root pfe/ folder
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" "sqlalchemy[asyncio]" aiosqlite `
    "python-jose[cryptography]" "passlib[bcrypt]" python-multipart `
    aiofiles reportlab pydantic-settings pandas openpyxl
```

### 2. Install frontend deps

```powershell
cd supplyMind\frontend
npm install
```

### 3. Start both servers at once

```powershell
cd supplyMind
.\start.ps1
```

Or start individually:

```powershell
# Terminal 1 — backend
cd supplyMind\backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd supplyMind\frontend
npm run dev
```

### 4. Open the app

| URL | Purpose |
|-----|---------|
| http://localhost:3000 | Frontend |
| http://localhost:8000/docs | FastAPI Swagger UI |

## Docker (one-command start)

```powershell
cd supplyMind
docker compose up --build
```

> Requires Docker Desktop. The AI engine source and trained model are mounted from the parent `pfe/` folder.

## Project structure

```
supplyMind/
├── backend/
│   ├── main.py               # FastAPI entry point
│   ├── models/
│   │   ├── database.py       # SQLAlchemy ORM models
│   │   └── schemas.py        # Pydantic request / response schemas
│   ├── routers/
│   │   ├── auth.py           # JWT auth routes
│   │   ├── projects.py       # Project CRUD + topology
│   │   ├── upload.py         # CSV / Excel upload + EDA
│   │   ├── simulate.py       # Simulation launch + what-if
│   │   ├── results.py        # Results + CSV/PDF export
│   │   └── ws.py             # WebSocket live progress
│   └── tasks/
│       └── simulation.py     # Core AI runner (MAPPO+GNN)
├── frontend/
│   ├── app/
│   │   ├── page.tsx                          # Landing page
│   │   ├── auth/login/page.tsx               # Login
│   │   ├── auth/register/page.tsx            # Register
│   │   ├── dashboard/page.tsx                # Project list
│   │   └── projects/[id]/
│   │       ├── upload/page.tsx               # Data upload + EDA
│   │       ├── builder/page.tsx              # Topology builder
│   │       ├── simulate/page.tsx             # Simulation runner
│   │       └── results/page.tsx              # Results dashboard
│   └── lib/
│       ├── api.ts                            # Typed API client
│       └── auth-context.tsx                  # React auth context
├── docker-compose.yml
├── start.ps1                                 # Dev launcher
└── README.md
```

## Environment variables

### backend/.env
```
DATABASE_URL=sqlite+aiosqlite:///./supplymind.db
JWT_SECRET=change-this-in-production-supplymind-secret
FRONTEND_URL=http://localhost:3000
```

### frontend/.env.local
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Simulation scenarios

| Scenario | Description |
|----------|-------------|
| `normal` | Baseline — steady demand |
| `demand_spike` | +200 % demand spike for 3 weeks |
| `supplier_failure` | Primary supplier offline 20 days |
| `lead_time_delay` | All lead times doubled |
| `seasonal` | Sinusoidal seasonal demand pattern |

## AI algorithms

| Algo | Notes |
|------|-------|
| `mappo` | Multi-Agent PPO + GNN (default, best performance) |
| `dqn` | Deep Q-Network baseline |
| `greedy` | Deterministic reorder-point baseline |
