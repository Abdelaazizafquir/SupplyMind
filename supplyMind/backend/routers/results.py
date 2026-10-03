"""Results router — export PDF, CSV and comparison data"""
import io
import csv
import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Simulation, Project, get_db, User
from models.schemas import SimulationDetail
from routers.auth import get_current_user

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

router = APIRouter()


@router.get("/{sim_id}", response_model=SimulationDetail)
async def get_results(
    sim_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Simulation)
        .join(Project)
        .where(Simulation.id == sim_id, Project.owner_id == current_user.id, Simulation.status == "completed")
    )
    sim = result.scalar_one_or_none()
    if not sim:
        raise HTTPException(status_code=404, detail="Completed simulation not found")
    return sim


@router.get("/{sim_id}/export/csv")
async def export_csv(
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

    output = io.StringIO()
    writer = csv.writer(output)

    kpis = sim.kpis or {}
    llm  = sim.llm_summary or {}
    comparison = llm.get("comparison")  # present when algorithm="all"

    if comparison:
        # ── Comparison export: one section per algorithm ──────────────────────
        algos = ["mappo_gnn", "base_stock", "ss_policy"]
        kpi_keys = ["fill_rate", "total_cost", "avg_inventory", "avg_lead_time", "disruptions_handled", "episodes_run"]

        # Header row: metric | MAPPO+GNN | Base-Stock | (s,S) Policy
        labels = [comparison.get(a, {}).get("label", a) for a in algos]
        writer.writerow(["metric"] + labels)

        for key in kpi_keys:
            row = [key.replace("_", " ").title()]
            for a in algos:
                row.append(comparison.get(a, {}).get("kpis", {}).get(key, ""))
            writer.writerow(row)

        # Per-algorithm timeline sections
        for a in algos:
            tl = comparison.get(a, {}).get("timeline", [])
            if tl:
                label = comparison.get(a, {}).get("label", a)
                writer.writerow([])
                writer.writerow([f"Timeline — {label}"])
                writer.writerow(["step", "cost"])
                for row in tl:
                    writer.writerow([row.get("step", ""), row.get("cost", "")])

        # Per-algorithm agent logs
        for a in algos:
            logs = comparison.get(a, {}).get("agent_logs", [])
            if logs:
                label = comparison.get(a, {}).get("label", a)
                writer.writerow([])
                writer.writerow([f"Agent Logs — {label}"])
                writer.writerow(["agent_id", "agent_name", "role", "avg_inventory", "orders_placed", "fill_rate", "avg_cost_per_step"])
                for log in logs:
                    writer.writerow([
                        log.get("agent_id", ""), log.get("agent_name", ""), log.get("role", ""),
                        log.get("avg_inventory", ""), log.get("orders_placed", ""),
                        log.get("fill_rate", ""), log.get("avg_cost_per_step", ""),
                    ])
    else:
        # ── Single-algorithm export (original behaviour) ───────────────────────
        writer.writerow(["metric", "value"])
        for k, v in kpis.items():
            writer.writerow([k, v])

        if sim.timeline:
            writer.writerow([])
            writer.writerow(["step", "cost"])
            for row in sim.timeline:
                writer.writerow([row.get("step", ""), row.get("cost", "")])

        if sim.agent_logs:
            writer.writerow([])
            writer.writerow(["agent_id", "agent_name", "role", "avg_inventory", "orders_placed", "fill_rate", "avg_cost_per_step"])
            for log in sim.agent_logs:
                writer.writerow([
                    log.get("agent_id", ""), log.get("agent_name", ""), log.get("role", ""),
                    log.get("avg_inventory", ""), log.get("orders_placed", ""),
                    log.get("fill_rate", ""), log.get("avg_cost_per_step", ""),
                ])

    output.seek(0)
    return StreamingResponse(
        iter([output.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=simulation_{sim_id[:8]}.csv"},
    )


@router.get("/{sim_id}/export/pdf")
async def export_pdf(
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

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("SupplyMind — Simulation Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Simulation ID: {sim_id}", styles["Normal"]))
        story.append(Paragraph(f"Algorithm: {sim.algorithm}  |  Scenario: {sim.scenario}", styles["Normal"]))
        story.append(Spacer(1, 20))

        kpis = sim.kpis or {}
        llm  = sim.llm_summary or {}
        comparison = llm.get("comparison")

        if comparison:
            # ── Algorithm Comparison section ─────────────────────────────────
            story.append(Paragraph("Algorithm Comparison", styles["Heading2"]))
            if llm.get("comparison_summary"):
                import re
                clean_summary = re.sub(r"\*\*(.*?)\*\*", r"\1", llm["comparison_summary"])
                clean_summary = re.sub(r"_(.*?)_", r"\1", clean_summary)
                story.append(Paragraph(clean_summary, styles["Normal"]))
                story.append(Spacer(1, 10))

            algos = ["mappo_gnn", "base_stock", "ss_policy"]
            kpi_keys = ["fill_rate", "total_cost", "avg_inventory", "avg_lead_time", "disruptions_handled", "episodes_run"]
            header = ["Metric"] + [comparison.get(a, {}).get("label", a) for a in algos]
            comp_data = [header]
            for key in kpi_keys:
                row = [key.replace("_", " ").title()]
                for a in algos:
                    val = comparison.get(a, {}).get("kpis", {}).get(key, "")
                    if isinstance(val, float):
                        val = f"{val:.2f}"
                    row.append(str(val))
                comp_data.append(row)

            ct = Table(comp_data, colWidths=[130, 120, 110, 100])
            ct.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ]))
            story.append(ct)
            story.append(Spacer(1, 20))

            # Per-algorithm agent logs
            for a in algos:
                logs = comparison.get(a, {}).get("agent_logs", [])
                if logs:
                    label = comparison.get(a, {}).get("label", a)
                    story.append(Paragraph(f"Agent Logs — {label}", styles["Heading3"]))
                    log_data = [["Agent", "Role", "Avg Inventory", "Orders", "Fill Rate", "Cost/Step"]]
                    for log in logs:
                        log_data.append([
                            log.get("agent_name", ""), log.get("role", ""),
                            f"{log.get('avg_inventory', 0):.0f}", str(log.get("orders_placed", "")),
                            f"{log.get('fill_rate', 0):.1f}%", f"${log.get('avg_cost_per_step', 0):.2f}",
                        ])
                    lt = Table(log_data, colWidths=[90, 70, 80, 60, 70, 80])
                    lt.setStyle(TableStyle([
                        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
                        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
                        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE",      (0, 0), (-1, -1), 8),
                    ]))
                    story.append(lt)
                    story.append(Spacer(1, 12))
        else:
            # ── Single-algorithm KPI section (original behaviour) ─────────────
            story.append(Paragraph("Key Performance Indicators", styles["Heading2"]))
            kpi_data = [["Metric", "Value"]]
            for k, v in kpis.items():
                kpi_data.append([str(k).replace("_", " ").title(), str(v)])
            t = Table(kpi_data, colWidths=[200, 200])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

            if sim.agent_logs:
                story.append(Paragraph("Agent Logs", styles["Heading2"]))
                log_data = [["Agent", "Role", "Avg Inventory", "Orders", "Fill Rate", "Cost/Step"]]
                for log in sim.agent_logs:
                    log_data.append([
                        log.get("agent_name", ""), log.get("role", ""),
                        f"{log.get('avg_inventory', 0):.0f}", str(log.get("orders_placed", "")),
                        f"{log.get('fill_rate', 0):.1f}%", f"${log.get('avg_cost_per_step', 0):.2f}",
                    ])
                lt = Table(log_data, colWidths=[90, 70, 80, 60, 70, 80])
                lt.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                    ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
                    ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ]))
                story.append(lt)
                story.append(Spacer(1, 20))

        if llm.get("global"):
            story.append(Paragraph("AI Analysis", styles["Heading2"]))
            story.append(Paragraph(llm["global"], styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.read()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{sim_id[:8]}.pdf"},
        )

    except ImportError:
        raise HTTPException(status_code=501, detail="PDF export requires reportlab: pip install reportlab")


# ── Groq-powered AI analysis & chat ──────────────────────────────────────────

SCENARIO_DESCRIPTIONS = {
    "normal":           "Normal operations — no disruptions.",
    "demand_spike":     "Demand spike — retailer demand surged 200% for ~20 days (days 20-40).",
    "lead_time_delay":  "Lead-time delay — supplier lead times tripled for 30 days (days 15-45).",
    "node_failure":     "Node failure — primary supplier completely offline for 20 days (days 25-45).",
    "combined":         "Combined crisis — simultaneous demand spike (+250%) on two retailers AND supplier failure (days 20-50).",
}

ALGO_DESCRIPTIONS = {
    "mappo_gnn": "MAPPO+GNN (Multi-Agent Proximal Policy Optimization with Graph Neural Network) — deep RL trained end-to-end on the full supply chain graph.",
    "base_stock": "Base-Stock policy — classical continuous order-up-to baseline (20% capacity target).",
    "ss_policy": "(s,S) Policy — classical reorder-point/order-to-level baseline (10%/40% capacity thresholds).",
    "all": "All-algorithm comparison run.",
    "mappo": "MAPPO — multi-agent RL without GNN message-passing.",
}


def _build_sim_context(sim: Simulation) -> str:
    """Build a rich text summary of simulation results for the LLM."""
    kpis       = sim.kpis or {}
    llm        = sim.llm_summary or {}
    comparison = llm.get("comparison")
    timeline   = sim.timeline or []
    logs       = sim.agent_logs or []

    algo    = sim.algorithm
    scenario = sim.scenario

    lines = [
        "=== SUPPLYMIND SIMULATION RESULTS ===",
        f"Algorithm  : {ALGO_DESCRIPTIONS.get(algo, algo)}",
        f"Scenario   : {SCENARIO_DESCRIPTIONS.get(scenario, scenario)}",
        f"Episodes   : {kpis.get('episodes_run', sim.episodes)} evaluation runs",
    ]

    # ── Timeline stats ───────────────────────────────────────────────────────
    if timeline:
        costs = [t.get("cost", 0) for t in timeline]
        avg_step = sum(costs) / len(costs) if costs else 0
        peak_day = timeline[costs.index(max(costs))].get("step", "?") if costs else "?"
        best_day = timeline[costs.index(min(costs))].get("step", "?") if costs else "?"
        lines += [
            f"\nTimeline ({len(timeline)} recorded steps):",
            f"  Avg step cost  : ${avg_step:.1f}",
            f"  Peak cost day  : day {peak_day}  (${max(costs):.1f})",
            f"  Lowest cost day: day {best_day}  (${min(costs):.1f})",
        ]
        # Inventory trend if available
        if timeline[0].get("inventories"):
            inv_totals = [sum(t.get("inventories", {}).values()) for t in timeline]
            lines += [
                f"  Avg total inventory: {sum(inv_totals)/len(inv_totals):.0f} units",
                f"  Min total inventory: {min(inv_totals):.0f} units (potential stockout risk)",
                f"  Max total inventory: {max(inv_totals):.0f} units",
            ]

    if comparison:
        # ── Multi-algorithm comparison ────────────────────────────────────────
        lines.append("\n=== ALGORITHM COMPARISON ===")
        best_fill_algo  = max(comparison, key=lambda a: comparison[a].get("kpis", {}).get("fill_rate", 0))
        best_cost_algo  = min(comparison, key=lambda a: comparison[a].get("kpis", {}).get("total_cost", float("inf")))
        best_lt_algo    = min(comparison, key=lambda a: comparison[a].get("kpis", {}).get("avg_lead_time", float("inf")))

        for a in ["mappo_gnn", "base_stock", "ss_policy"]:
            entry = comparison.get(a, {})
            k     = entry.get("kpis", {})
            label = entry.get("label", a)
            badges = []
            if a == best_fill_algo: badges.append("BEST FILL RATE")
            if a == best_cost_algo: badges.append("LOWEST COST")
            if a == best_lt_algo: badges.append("FASTEST TRANSPORT")
            badge_str = f"  [{', '.join(badges)}]" if badges else ""
            lines.append(
                f"\n  {label}{badge_str}:"
                f"\n    Fill rate     : {k.get('fill_rate','?')}%"
                f"\n    Total cost    : ${k.get('total_cost','?'):,}"
                f"\n    Avg Lead Time : {k.get('avg_lead_time','?')} days"
                f"\n    Avg inventory : {k.get('avg_inventory','?')} units"
            )
            # Per-algo agent logs
            algo_logs = entry.get("agent_logs", [])
            if algo_logs:
                lines.append(f"    Agent breakdown:")
                for ag in algo_logs[:6]:
                    lines.append(
                        f"      {ag.get('agent_name','?')} ({ag.get('role','?')}): "
                        f"inv={ag.get('avg_inventory',0):.0f}u  "
                        f"orders={ag.get('orders_placed','?')}  "
                        f"cost/step=${ag.get('avg_cost_per_step',0):.2f}"
                    )
    else:
        # ── Single-algorithm KPIs ─────────────────────────────────────────────
        lines += [
            f"\n=== KEY PERFORMANCE INDICATORS ===",
            f"  Fill rate        : {kpis.get('fill_rate', '?')}%  {'✓ Excellent' if float(kpis.get('fill_rate',0)) >= 95 else '⚠ Below target (95%)'}",
            f"  Total chain cost : ${kpis.get('total_cost', '?'):,}",
            f"  Avg inventory    : {kpis.get('avg_inventory', '?')} units",
            f"  Avg Lead Time    : {kpis.get('avg_lead_time', '?')} days  {'✓ Fast' if float(kpis.get('avg_lead_time', 99)) < 4.0 else '⚠ Delayed'}",
            f"  Disruptions handled: {kpis.get('disruptions_handled', 0)}",
        ]

        if logs:
            lines.append(f"\n=== AGENT PERFORMANCE ({len(logs)} agents) ===")
            roles = {}
            for ag in logs:
                r = ag.get("role", "unknown")
                roles.setdefault(r, []).append(ag)
            for role, agents in roles.items():
                lines.append(f"\n  {role.upper()} nodes ({len(agents)}):")
                for ag in agents:
                    lines.append(
                        f"    {ag.get('agent_name','?')}: "
                        f"avg_inv={ag.get('avg_inventory',0):.0f}u  "
                        f"orders={ag.get('orders_placed','?')}  "
                        f"fill={ag.get('fill_rate','?')}%  "
                        f"cost/step=${ag.get('avg_cost_per_step',0):.2f}"
                    )

    # ── Step explanations from LLM coordinator (if any) ──────────────────────
    step_expls = llm.get("step_explanations", [])
    if step_expls:
        lines.append(f"\n=== AI COORDINATOR OBSERVATIONS (during simulation) ===")
        for e in step_expls[:3]:
            lines.append(f"  Day {e.get('step','?')}: {e.get('explanation','')[:200]}")

    return "\n".join(lines)


def _call_groq(system_prompt: str, user_message: str) -> str:
    """Call Groq API. Auto-enables when GROQ_API_KEY is present."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "⚠️ AI Analysis unavailable. Set GROQ_API_KEY in .env and restart the server."
    try:
        from groq import Groq
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"


class AnalyzeResponse(BaseModel):
    analysis: str


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str


@router.post("/{sim_id}/analyze", response_model=AnalyzeResponse)
async def analyze_simulation(
    sim_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a fresh Groq/Gemma AI analysis of the simulation results."""
    result = await db.execute(
        select(Simulation)
        .join(Project)
        .where(Simulation.id == sim_id, Project.owner_id == current_user.id)
    )
    sim = result.scalar_one_or_none()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    context = _build_sim_context(sim)
    system = (
        "You are SupplyMind, an expert supply chain AI analyst specializing in multi-echelon inventory systems. "
        "The simulation was run by a MAPPO+GNN (Multi-Agent Proximal Policy Optimization with Graph Neural Network) AI model "
        "on a 6-node supply chain: 1 supplier → 2 distributors → 3 retailers. "
        "Given the full simulation results below, write a concise but insightful analysis (4-6 sentences). "
        "Always comment on: (1) customer service level (fill rate vs 95% target), "
        "(2) cost efficiency and where costs concentrate, "
        "(3) average lead times and whether pipeline delays impacted customer fill rate, "
        "(4) how well the chain handled the disruption scenario, "
        "(5) which agents performed best/worst and why. "
        "Use **bold** for key numbers. Be specific — reference actual values from the data. "
        "Do NOT use markdown headers or bullet points — write flowing prose."
    )
    user_msg = f"Analyze these supply chain simulation results:\n\n{context}"
    analysis = _call_groq(system, user_msg)
    return AnalyzeResponse(analysis=analysis)


@router.post("/{sim_id}/chat", response_model=ChatResponse)
async def chat_about_simulation(
    sim_id: str,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Answer a user question about the simulation using Groq/Gemma."""
    result = await db.execute(
        select(Simulation)
        .join(Project)
        .where(Simulation.id == sim_id, Project.owner_id == current_user.id)
    )
    sim = result.scalar_one_or_none()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    context = _build_sim_context(sim)
    history_text = ""
    for msg in (body.history or [])[-8:]:   # last 4 turns
        prefix = "User" if msg.role == "user" else "Assistant"
        history_text += f"{prefix}: {msg.content}\n"

    system = (
        "You are SupplyMind, an expert supply chain AI assistant specializing in multi-echelon inventory systems. "
        "The simulation covers a 6-node supply chain: 1 supplier → 2 distributors → 3 retailers, "
        "optimized by a MAPPO+GNN deep reinforcement learning model. "
        "Answer questions about the simulation results concisely and precisely. "
        "Reference specific numbers from the data when relevant. "
        "Use **bold** for key numbers. Keep answers under 180 words. "
        "If asked to compare algorithms, use the comparison data provided. "
        "If asked about a specific agent/node, use the agent performance data."
    )
    user_msg = (
        f"Simulation data:\n{context}\n\n"
        + (f"Conversation so far:\n{history_text}\n" if history_text else "")
        + f"User question: {body.message}"
    )
    reply = _call_groq(system, user_msg)
    return ChatResponse(reply=reply)

