# SupplyMind — Intelligent Multi-Agent Supply Chain Optimization

![SupplyMind Dashboard Preview](supplyMind/frontend/public/icon.png)

SupplyMind is a state-of-the-art SaaS platform designed to optimize multi-echelon supply chain inventory management under deep uncertainty. Built as a Master's Thesis Project, it replaces brittle heuristic policies (like Base-Stock or `(s,S)`) with an advanced **Multi-Agent Reinforcement Learning (MARL)** architecture, integrated with **Graph Neural Networks (GNN)**, **Transformers (PatchTST)**, and **Large Language Models (LLMs)**.

## 🚀 Key Features

*   **Multi-Agent Reinforcement Learning (MAPPO)**: Autonomous agents (Retailers, Distributors, Suppliers) learn cooperative ordering policies to minimize total chain cost while avoiding stockouts.
*   **Graph Neural Network (GNN) Communication Layer**: Agents share hidden state representations through the supply chain topology, naturally preventing the **Bullwhip Effect**.
*   **Proactive Transformer Forecasting**: Integrated **PatchTST** models analyze historical data to inject 7-day rolling forecasts directly into the agents' observation space.
*   **Live Control Engine**: Connect your live PostgreSQL/MySQL database to the platform to receive real-time, explainable AI ordering recommendations.
*   **What-If Scenarios**: Simulate extreme disruptions (e.g., massive demand spikes, sudden lead time delays, node failures) and observe how the MARL policy outperforms classical heuristics by 300%+.
*   **LLM Coordinator**: Integrated `Llama 3.3` (via Groq) acts as a human-readable supervisor, explaining agent behavior, summarizing vulnerabilities, and outputting narrative analysis.
*   **Explainable AI (SHAP)**: Gradient attribution algorithms explain *why* the neural network generated a specific order quantity (e.g., "Current Inventory pushed the order UP, while Neighbor Signals pushed it DOWN").

---

## 🏗️ Architecture

The system utilizes a **CTDE (Centralized Training, Decentralized Execution)** paradigm.

1.  **Frontend**: Next.js 14, React Flow (Topology Builder), Tailwind CSS.
2.  **Backend API**: FastAPI, SQLAlchemy, PostgreSQL/SQLite.
3.  **RL Environment**: Custom Python Gymnasium environment (`MultiEchelonInventoryEnv`).
4.  **Deep Learning Stack**: PyTorch, Torch Geometric (GNNs), Ray RLlib (MAPPO), NeuralForecast (PatchTST).
5.  **LLM Integration**: LangChain & LangGraph.

---

## 💻 Local Setup & Installation

### 1. Backend Setup

Ensure you have Python 3.12+ installed.

```bash
cd supplyMind/backend
python -m venv .venv
# Activate virtual environment
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

**Environment Variables**
Create a `.env` file in the `backend` directory:
```env
JWT_SECRET=your_super_secret_key_change_me
GROQ_API_KEY=your_groq_api_key
```

**Start the Backend Server**
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

Ensure you have Node.js 18+ installed.

```bash
cd supplyMind/frontend
npm install
```

**Start the Frontend Server**
```bash
npm run dev
```

The platform will now be accessible at `http://localhost:3000`.

---

## 🧠 Using the Platform

### Creating a Project
When creating a project, you can choose between two modes:
1.  **Simulation Mode**: Upload historical CSV/Excel data, build your supply chain graph, and simulate years of data in seconds. Test the AI against disruptions and compare it to classical policies.
2.  **Live Control Mode**: Skip the simulation and connect directly to your live database. The system will ingest your current inventory levels and output real-time ordering recommendations.

### Dataset Overrides
If your dataset lacks explicit cost columns or continuous dates, the platform allows you to specify global **Data Overrides** (e.g., Default Unit Cost, Time Horizon) during the upload phase to ensure the AI has the necessary financial context.

---

## 📊 Academic Performance

Under normal conditions, the MAPPO+GNN architecture performs comparably to optimized `(s,S)` policies. 
However, under **severe supply chain disruptions**, classical heuristics collapse (resulting in catastrophic holding or shortage costs). The MAPPO+GNN agent degrades gracefully, often reducing the financial impact of the disruption by over **85%** compared to baselines.
