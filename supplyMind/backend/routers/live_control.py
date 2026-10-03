"""
live_control.py — real-time MAPPO+GNN inference with fully flexible topology.

The GNN uses parameter-sharing message passing (KaggleGCNLayer uses the same
weights for every node regardless of graph size), so the model naturally supports
any number of agents N. We pass all user nodes directly — no slot aggregation.

Architecture:
  KaggleGNN  : forward(node_features: (1, N, 16), adj: (1, N, N)) → (1, N, 64)
  KaggleActor: forward(x: (N, 79)) → (N, 1)   — 79 = obs(12) + gnn(64) + role(3)
Both networks are fully agnostic to N.
"""

import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from models.schemas import (
    LiveControlRequest, LiveControlResponse,
    NodeInventoryInput, NodeRecommendation,
)
from routers.auth import get_current_user

router = APIRouter()

_ROLE_VEC = {
    "retailer":    [1, 0, 0],
    "distributor": [0, 1, 0],
    "supplier":    [0, 0, 1],
}


def _urgency(inv_pct: float) -> str:
    if inv_pct < 0.10: return "critical"
    if inv_pct < 0.25: return "high"
    if inv_pct < 0.50: return "medium"
    return "low"


# -- Forecasting (PatchTST) ---------------------------------------------------

FORECAST_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "forecasting", "patchtst_model"
)

_patchtst_cache = None

def load_patchtst():
    global _patchtst_cache
    if _patchtst_cache is not None:
        return _patchtst_cache
    if os.path.exists(FORECAST_MODEL_PATH):
        try:
            from neuralforecast import NeuralForecast
            _patchtst_cache = NeuralForecast.load(path=FORECAST_MODEL_PATH)
        except Exception:
            _patchtst_cache = "FAILED"
    else:
        _patchtst_cache = "FAILED"
    return _patchtst_cache


# -- Build 16-dim observation per node ----------------------------------------

def _node_obs(node: NodeInventoryInput, forecast_array: np.ndarray = None) -> np.ndarray:
    """Build the same 16-dim observation the GNN/actor expects."""
    obs = np.zeros(16, dtype=np.float32)
    inv_pct     = node.inventory_units / max(node.capacity_units, 1.0)
    demand_norm = node.demand_units    / max(node.capacity_units, 1.0)
    obs[0] = max(0.0, min(1.0, inv_pct))
    obs[2] = demand_norm
    obs[5] = 0.5   # mid-episode time feature
    role   = _ROLE_VEC.get(node.node_type, [0, 0, 0])
    obs[6], obs[7], obs[8] = role
    
    # Inject 7-day forecast if available and node is a retailer
    if forecast_array is not None and node.node_type == "retailer":
        # Ensure array is length 7 and normalize it by 100.0 as done in the env
        obs[9:16] = (forecast_array[:7] / 100.0).astype(np.float32)
        
    return obs


# -- Build supply-chain adjacency for N nodes ---------------------------------

def _supply_chain_adj(nodes: list) -> np.ndarray:
    """
    Adjacency matrix reflecting the supply chain hierarchy:
      - same-tier nodes are fully connected (shared market info)
      - retailers connect to all distributors (ordering relationship)
      - distributors connect to all suppliers
    Row-normalised so GCN aggregation is a weighted mean.
    """
    n = len(nodes)
    tier_map = {"retailer": 0, "distributor": 1, "supplier": 2}
    tiers = np.array([tier_map.get(node.node_type, -1) for node in nodes], dtype=np.float32)

    # Vectorized adjacency construction
    adj = np.eye(n, dtype=np.float32)
    # Broadcast subtraction to find tier differences
    diffs = np.abs(tiers[:, None] - tiers[None, :])
    # Connect same tier or adjacent tier (diff == 0 or diff == 1) => diff <= 1
    # Avoid connecting unknown tiers (-1) unless handling defaults. 
    # Both tiers must be valid (>= 0)
    valid = (tiers[:, None] >= 0) & (tiers[None, :] >= 0)
    adj[valid & (diffs <= 1)] = 1.0

    row_sums = adj.sum(axis=1, keepdims=True).clip(min=1e-8)
    return adj / row_sums


# -- Groq analysis ------------------------------------------------------------

def _groq_analysis(recommendations, topology_note: str) -> str:
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return "AI analysis unavailable (GROQ_API_KEY not set)."
        
        # Sort by urgency for the summary
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_recs = sorted(recommendations, key=lambda r: urgency_order.get(r.urgency, 99))
        
        if len(sorted_recs) <= 20:
            lines = [
                f"  - {r.node_name} ({r.node_type}): "
                f"inv {r.inventory_units:.0f}/{r.capacity_units:.0f} "
                f"({r.inventory_pct*100:.1f}%), ORDER {r.order_units:.0f} -- {r.urgency}"
                for r in sorted_recs
            ]
        else:
            top_critical = sorted_recs[:15]
            lines = [
                f"  - {r.node_name} ({r.node_type}): "
                f"inv {r.inventory_units:.0f}/{r.capacity_units:.0f} "
                f"({r.inventory_pct*100:.1f}%), ORDER {r.order_units:.0f} -- {r.urgency}"
                for r in top_critical
            ]
            lines.append(f"  ... and {len(sorted_recs)-15} other nodes (summarized).")

        prompt = (
            f"You are a supply chain AI advisor. {topology_note}\n\n"
            "Live inventory snapshot with MAPPO+GNN recommendations (showing top critical nodes if dataset is large):\n"
            + "\n".join(lines)
            + "\n\nProvide a concise 4-6 sentence briefing: which network areas are most at risk, "
            "whether the aggregate ordering plan prevents stockouts, and one specific network-wide action."
        )
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI analysis unavailable: {exc}"


def _get_defaults(node_type: str):
    if node_type == "supplier": return 5.0, 5
    if node_type == "distributor": return 12.0, 3
    return 15.0, 1  # retailer


# -- Heuristic fallback -------------------------------------------------------

def _heuristic_response(req: LiveControlRequest) -> LiveControlResponse:
    recs = []
    total = 0.0
    total_cost = 0.0
    for n in req.nodes:
        inv_pct = n.inventory_units / max(n.capacity_units, 1.0)
        order   = round(max(0.0, 1.0 - inv_pct) * 0.5 * n.capacity_units, 1)
        total  += order
        
        default_cost, default_lt = _get_defaults(n.node_type)
        cost = n.unit_cost if n.unit_cost is not None else default_cost
        lt = n.lead_time_days if n.lead_time_days is not None else default_lt
        est_cost = order * cost
        total_cost += est_cost
        
        recs.append(NodeRecommendation(
            node_id=n.node_id, node_name=n.node_name or n.node_id, node_type=n.node_type,
            inventory_units=n.inventory_units, inventory_pct=round(inv_pct, 4),
            capacity_units=n.capacity_units, order_units=order, urgency=_urgency(inv_pct),
            estimated_cost=est_cost, estimated_lead_time_days=lt
        ))
    return LiveControlResponse(
        recommendations=recs, total_order_units=round(total, 1), total_estimated_cost=round(total_cost, 2),
        analysis="MAPPO+GNN model not loaded -- heuristic deficit-based recommendations shown.",
        topology_note="",
    )


# -- Endpoint -----------------------------------------------------------------

@router.post("/infer", response_model=LiveControlResponse)
async def live_infer(
    req: LiveControlRequest,
    current_user=Depends(get_current_user),
) -> LiveControlResponse:
    """
    Run MAPPO+GNN on a live inventory snapshot with any number of user nodes.

    The GNN uses shared weights (KaggleGCNLayer) applied to a (1, N, 16) tensor,
    and the actor is applied row-wise to (N, 79). Both are size-agnostic — we pass
    all user nodes directly as N individual agents with an N×N supply-chain adjacency.
    """
    # 1. Fetch ML Models
    try:
        from tasks.simulation import load_model, mappo_actions
        gnn, actor = load_model()
    except Exception as exc:
        return _heuristic_response(req)
        
    patchtst_model = load_patchtst()

    # 2. Extract state & adjacency
    nodes = req.nodes
    n = len(nodes)
    adj = _supply_chain_adj(nodes)

    # 3. Generate proactive demand forecasts for retailers
    # Since Live Control only receives current demand, we mock a 7-day forecast 
    # based on current demand + noise. If real history was provided, we'd use patchtst_model.predict()
    forecasts = {}
    for node in nodes:
        if node.node_type == "retailer":
            base_demand = float(node.demand_units)
            # Add ±10% noise to create a realistic-looking forecast curve
            noise = np.random.uniform(0.9, 1.1, size=7)
            forecasts[node.node_id] = base_demand * noise

    # 4. Build observations
    agent_ids = [f"node_{i}" for i in range(n)]
    obs_dict = {}
    for i, node in enumerate(nodes):
        node_forecast = forecasts.get(node.node_id, None)
        obs_dict[agent_ids[i]] = _node_obs(node, forecast_array=node_forecast)

    try:
        raw_actions = mappo_actions(obs_dict, agent_ids, gnn, actor, adj)
    except Exception as exc:
        raise HTTPException(500, f"Model inference failed: {exc}") from exc

    recommendations = []
    total_order     = 0.0
    total_cost      = 0.0

    for i, node in enumerate(nodes):
        action  = max(0.0, min(1.0, float(raw_actions[agent_ids[i]])))
        order   = round(action ** 2 * node.capacity_units, 1)
        inv_pct = node.inventory_units / max(node.capacity_units, 1.0)
        total_order += order
        
        default_cost, default_lt = _get_defaults(node.node_type)
        cost = node.unit_cost if node.unit_cost is not None else default_cost
        lt = node.lead_time_days if node.lead_time_days is not None else default_lt
        est_cost = order * cost
        total_cost += est_cost
        
        recommendations.append(NodeRecommendation(
            node_id        = node.node_id,
            node_name      = node.node_name or node.node_id,
            node_type      = node.node_type,
            inventory_units= node.inventory_units,
            inventory_pct  = round(max(0.0, min(1.0, inv_pct)), 4),
            capacity_units = node.capacity_units,
            order_units    = order,
            urgency        = _urgency(inv_pct),
            estimated_cost = est_cost,
            estimated_lead_time_days = lt,
        ))

    order_key = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    type_key  = {"retailer": 0, "distributor": 1, "supplier": 2}
    recommendations.sort(key=lambda r: (order_key[r.urgency], type_key.get(r.node_type, 9)))

    n_ret  = sum(1 for nd in nodes if nd.node_type == "retailer")
    n_dist = sum(1 for nd in nodes if nd.node_type == "distributor")
    n_sup  = sum(1 for nd in nodes if nd.node_type == "supplier")
    topology_note = (
        f"{n} node{'s' if n != 1 else ''} "
        f"({n_ret} retailer, {n_dist} distributor, {n_sup} supplier) — "
        "each agent runs through the GNN independently (parameter-shared weights)."
    )

    return LiveControlResponse(
        recommendations  = recommendations,
        total_order_units= round(total_order, 1),
        total_estimated_cost = round(total_cost, 2),
        analysis         = _groq_analysis(recommendations, topology_note),
        topology_note    = topology_note,
    )


from models.schemas import ExplainRequest, ExplainResponse, FeatureExplanation

@router.post("/explain", response_model=ExplainResponse)
async def live_infer_explain(
    req: ExplainRequest,
    current_user=Depends(get_current_user),
) -> ExplainResponse:
    """
    Generate gradient-based explanations for a specific node's recommendation.
    Uses Integrated Gradients-style attribution (gradient × (input - baseline))
    which works reliably with any PyTorch model including those with LayerNorm.
    """
    try:
        from tasks.simulation import load_model
        import torch
    except Exception as exc:
        raise HTTPException(500, f"Model unavailable: {exc}")

    nodes = req.nodes
    n = len(nodes)
    if n == 0:
        raise HTTPException(400, "No nodes provided")

    target_node = next((nd for nd in nodes if nd.node_id == req.target_node_id), None)
    if not target_node:
        raise HTTPException(404, "Target node not found")

    try:
        target_i = nodes.index(target_node)
    except ValueError:
        raise HTTPException(404, "Target node not in list")

    obs_dict = {f"node_{i}": _node_obs(nodes[i]) for i in range(n)}
    adj = _supply_chain_adj(nodes)

    try:
        gnn, actor = load_model()
    except Exception as exc:
        raise HTTPException(500, f"Model load failed: {exc}")

    try:
        import torch
        obs_list = [obs_dict[f"node_{i}"] for i in range(n)]
        obs_arr  = np.array(obs_list, dtype=np.float32)

        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0)  # (1, N, 16)
        adj_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)      # (1, N, N)

        with torch.no_grad():
            gnn_out = gnn(obs_tensor, adj_tensor)  # (1, N, 64)

        obs_12  = obs_tensor[0, :, :12]   # (N, 12)
        gnn_64  = gnn_out[0]              # (N, 64)
        role_3  = obs_tensor[0, :, 6:9]  # (N, 3)
        actor_in_all = torch.cat([obs_12, gnn_64, role_3], dim=-1)  # (N, 79)

        target_vec = actor_in_all[target_i:target_i + 1].detach().clone()  # (1, 79)
        target_vec.requires_grad_(True)

        output = actor(target_vec)
        output.sum().backward()

        grads = target_vec.grad[0].detach().numpy()  # (79,)

        # Baseline = mean of all other nodes (or zeros if only 1 node)
        if n > 1:
            others = actor_in_all[np.arange(n) != target_i].detach().numpy()
            baseline = others.mean(axis=0)
        else:
            baseline = np.zeros(actor_in_all.shape[-1], dtype=np.float32)

        input_vals = target_vec.detach().numpy()[0]
        sv = grads * (input_vals - baseline)  # gradient × (input - baseline)

        base_value = float(output.item()) - float(sv.sum())

    except Exception as exc:
        raise HTTPException(500, f"Attribution failed: {exc}")

    sv_flat = sv.flatten()

    # Aggregate into 3 human-readable features
    # Index 0:   inventory %
    # Index 2:   demand %
    # Index 12–75: GNN context embedding (supply chain neighbours)
    inv_shap = float(sv_flat[0])
    dem_shap = float(sv_flat[2])
    gnn_shap = float(sv_flat[12:76].sum())

    features = [
        FeatureExplanation(feature_name="Current Stock on Hand",          shap_value=inv_shap),
        FeatureExplanation(feature_name="Recent Customer Orders",         shap_value=dem_shap),
        FeatureExplanation(feature_name="Activity at Other Locations",    shap_value=gnn_shap),
    ]

    positive = sorted([f for f in features if f.shap_value > 0],  key=lambda x: x.shap_value, reverse=True)
    negative = sorted([f for f in features if f.shap_value <= 0], key=lambda x: x.shap_value)

    # Plain-language summary
    dominant = max(features, key=lambda x: abs(x.shap_value))
    if dominant.feature_name == "Current Stock on Hand":
        if inv_shap > 0:
            summary = "The AI recommends ordering MORE because your current stock levels are running low."
        else:
            summary = "The AI recommends ordering LESS — you already have sufficient stock on hand."
    elif dominant.feature_name == "Recent Customer Orders":
        if dem_shap > 0:
            summary = "The AI recommends ordering MORE to meet rising customer demand."
        else:
            summary = "The AI recommends ordering LESS because current demand is low."
    else:
        if gnn_shap > 0:
            summary = "The AI recommends ordering MORE to protect against shortages elsewhere in the supply chain."
        else:
            summary = "The AI is reducing your order to help balance excess inventory across the network."

    return ExplainResponse(
        node_id=req.target_node_id,
        base_value=base_value,
        positive_features=positive,
        negative_features=negative,
        summary=summary,
    )
