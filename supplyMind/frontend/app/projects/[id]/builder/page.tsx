"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, AgentNode, TopologyEdge, Topology, TopologyTemplate } from "@/lib/api";
import ProjectNav from "@/components/ProjectNav";
import DashboardLayout from "@/components/DashboardLayout";
import { Layout, Settings, Trash2, Plus, ArrowRight, Save, Play } from "lucide-react";

const NODE_COLORS: Record<string, string> = {
  supplier:     "bg-warning/10 border-warning/30 text-warning",
  distributor:  "bg-primary/10 border-primary/30 text-primary",
  manufacturer: "bg-success/10 border-success/30 text-success",
  retailer:     "bg-danger/10 border-danger/30 text-danger",
};

const NODE_ICONS: Record<string, string> = {
  supplier: "🏭", distributor: "🚛", manufacturer: "⚙️", retailer: "🏪",
};

export default function BuilderPage() {
  const params = useParams();
  const projectId = params.id as string;
  const router = useRouter();

  const [agents, setAgents]       = useState<AgentNode[]>([]);
  const [edges, setEdges]         = useState<TopologyEdge[]>([]);
  const [templates, setTemplates] = useState<TopologyTemplate[]>([]);
  const [selected, setSelected]   = useState<string | null>(null);
  const [saving, setSaving]       = useState(false);
  const [saved, setSaved]         = useState(false);
  const [showCustom, setShowCustom] = useState(false);
  const [customCounts, setCustomCounts] = useState({ suppliers: 1, distributors: 2, retailers: 3 });
  const [projectMode, setProjectMode] = useState<string>("simulation");

  useEffect(() => {
    api.projects.templates().then(setTemplates).catch(console.error);
    api.projects.get(projectId).then((p) => {
      if (p.topology) {
        if (p.topology.agents) setAgents(p.topology.agents);
        if (p.topology.edges) setEdges(p.topology.edges);
        if (p.topology.mode) setProjectMode(p.topology.mode);
      }
    }).catch(console.error);
  }, [projectId]);

  const loadTemplate = (tpl: TopologyTemplate) => {
    setAgents(tpl.agents);
    setEdges(tpl.edges);
    setSelected(null);
  };

  const updateAgent = (id: string, updates: Partial<AgentNode>) => {
    setAgents((prev) => prev.map((a) => a.id === id ? { ...a, ...updates } : a));
  };

  const removeAgent = (id: string) => {
    setAgents((prev) => prev.filter((a) => a.id !== id));
    setEdges((prev) => prev.filter((e) => e.source !== id && e.target !== id));
    if (selected === id) setSelected(null);
  };

  const addEdge = (source: string, target: string) => {
    const id = `e_${source}_${target}`;
    if (edges.find((e) => e.id === id)) return;
    setEdges((prev) => [...prev, { id, source, target, transport_cost: 0.1 }]);
  };

  const generateCustomTopology = () => {
    const { suppliers, distributors, retailers } = customCounts;
    const newAgents: AgentNode[] = [];
    const newEdges: TopologyEdge[] = [];

    const xSpacing = (n: number, idx: number, total: number) =>
      total === 1 ? 350 : 100 + idx * Math.floor(600 / (total - 1));

    for (let i = 0; i < suppliers; i++) {
      newAgents.push({ id: `supplier_${i}`, type: "supplier", name: `Supplier ${String.fromCharCode(65 + i)}`,
        capacity: 1000, lead_time_min: 2, lead_time_max: 5,
        holding_cost: 0.5, stockout_penalty: 500, policy: "mappo_gnn",
        position_x: xSpacing(suppliers, i, suppliers), position_y: 50 });
    }
    for (let i = 0; i < distributors; i++) {
      newAgents.push({ id: `distributor_${i}`, type: "distributor", name: `Distributor ${String.fromCharCode(65 + i)}`,
        capacity: 500, lead_time_min: 1, lead_time_max: 3,
        holding_cost: 1.0, stockout_penalty: 500, policy: "mappo_gnn",
        position_x: xSpacing(distributors, i, distributors), position_y: 250 });
    }
    for (let i = 0; i < retailers; i++) {
      newAgents.push({ id: `retailer_${i}`, type: "retailer", name: `Retailer ${String.fromCharCode(65 + i)}`,
        capacity: 200, lead_time_min: 1, lead_time_max: 1,
        holding_cost: 2.0, stockout_penalty: 500, policy: "mappo_gnn",
        position_x: xSpacing(retailers, i, retailers), position_y: 450 });
    }

    // Connect each supplier to each distributor
    for (let s = 0; s < suppliers; s++) {
      for (let d = 0; d < distributors; d++) {
        newEdges.push({ id: `e_s${s}_d${d}`, source: `supplier_${s}`, target: `distributor_${d}`, transport_cost: 0.1 });
      }
    }
    // Distribute retailers evenly across distributors
    for (let r = 0; r < retailers; r++) {
      const d = r % (distributors || 1);
      newEdges.push({ id: `e_d${d}_r${r}`, source: `distributor_${d}`, target: `retailer_${r}`, transport_cost: 0.2 });
    }

    setAgents(newAgents);
    setEdges(newEdges);
    setShowCustom(false);
    setSelected(null);
  };

  const saveTopology = async () => {
    setSaving(true);
    try {
      const topology: Topology = { agents, edges, mode: projectMode } as any;
      await api.projects.saveTopology(projectId, topology);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const selectedAgent = agents.find((a) => a.id === selected);

  return (
    <DashboardLayout>
      <div className="h-full flex flex-col overflow-hidden">
        <ProjectNav projectId={projectId} step="builder" />

        <div className="flex-1 flex overflow-hidden select-none">
          {/* Left: templates + canvas */}
          <div className="flex-1 flex flex-col p-6 overflow-auto space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h1 className="text-base font-bold text-text-primary">Supply Chain Topology</h1>
                <p className="text-[11px] text-text-secondary">Visually connect nodes representing echelons.</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={saveTopology}
                  disabled={saving || agents.length === 0}
                  className="bg-surface-card border border-border-custom hover:border-text-secondary/30 disabled:opacity-40 text-text-primary text-xs font-semibold px-4 py-2 rounded-lg transition flex items-center gap-1.5 shadow-premium"
                >
                  <Save size={14} /> {saved ? "✓ Saved" : saving ? "Saving…" : "Save topology"}
                </button>
                <button
                  onClick={() => { saveTopology(); router.push(projectMode === "live" ? `/projects/${projectId}/live` : `/projects/${projectId}/simulate`); }}
                  disabled={agents.length === 0}
                  className="bg-primary hover:bg-primary/95 disabled:opacity-40 text-white text-xs font-semibold px-4 py-2 rounded-lg transition flex items-center gap-1.5 shadow-premium"
                >
                  {projectMode === "live" ? "Continue to Live Control" : "Continue"} <ArrowRight size={14} />
                </button>
              </div>
            </div>

            {/* Templates */}
            <div className="flex flex-wrap gap-2.5 items-center">
              <span className="text-text-secondary text-xs">Templates:</span>
              {templates.map((t) => (
                <button
                  key={t.id}
                  onClick={() => loadTemplate(t)}
                  className="text-xs bg-surface-card hover:bg-primary/5 border border-border-custom hover:border-primary/30 text-text-secondary hover:text-text-primary px-3 py-1.5 rounded-lg transition"
                >
                  {t.name}
                </button>
              ))}
              <button
                onClick={() => setShowCustom((v) => !v)}
                className={`text-xs border px-3 py-1.5 rounded-lg transition font-semibold ${
                  showCustom 
                    ? "bg-primary border-primary text-white" 
                    : "bg-surface-card border-border-custom hover:border-primary/30 text-text-secondary hover:text-text-primary"
                }`}
              >
                ✏️ Custom…
              </button>
            </div>

            {/* Custom topology builder */}
            {showCustom && (
              <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4 shadow-premium">
                <h3 className="font-semibold text-xs text-text-primary">Configure custom network counts</h3>
                <div className="grid grid-cols-3 gap-4">
                  {(["suppliers", "distributors", "retailers"] as const).map((role) => (
                    <div key={role} className="space-y-1.5">
                      <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider capitalize">{role}</label>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setCustomCounts((c) => ({ ...c, [role]: Math.max(1, c[role] - 1) }))}
                          className="w-8 h-8 rounded-lg border border-border-custom bg-surface hover:bg-primary/5 text-text-primary font-bold flex items-center justify-center transition"
                        >−</button>
                        <span className="text-sm font-bold w-6 text-center text-text-primary">{customCounts[role]}</span>
                        <button
                          type="button"
                          onClick={() => setCustomCounts((c) => ({ ...c, [role]: Math.min(10, c[role] + 1) }))}
                          className="w-8 h-8 rounded-lg border border-border-custom bg-surface hover:bg-primary/5 text-text-primary font-bold flex items-center justify-center transition"
                        >+</button>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-between pt-2">
                  <button
                    onClick={generateCustomTopology}
                    className="bg-primary hover:bg-primary/95 text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow-premium"
                  >
                    Generate {customCounts.suppliers + customCounts.distributors + customCounts.retailers}-node topology
                  </button>
                  <span className="text-text-tertiary text-xs">
                    {customCounts.suppliers} supplier{customCounts.suppliers !== 1 ? "s" : ""} →{" "}
                    {customCounts.distributors} distributor{customCounts.distributors !== 1 ? "s" : ""} →{" "}
                    {customCounts.retailers} retailer{customCounts.retailers !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>
            )}

            {/* Canvas — SVG visual topology */}
            {agents.length === 0 ? (
              <div className="flex-1 border border-dashed border-border-custom bg-surface-card rounded-card flex items-center justify-center text-text-tertiary" style={{ minHeight: 450 }}>
                <div className="text-center space-y-2">
                  <Layout size={32} className="mx-auto text-text-tertiary opacity-60" />
                  <p className="text-sm font-semibold text-text-primary">Canvas is empty</p>
                  <p className="text-xs">Select a baseline template or generate a custom model to begin.</p>
                </div>
              </div>
            ) : (
              <TopologyCanvas
                agents={agents}
                edges={edges}
                selected={selected}
                onSelect={setSelected}
              />
            )}
          </div>

          {/* Right: config panel */}
          {selectedAgent && (
            <aside className="w-72 border-l border-border-custom p-6 overflow-y-auto bg-surface-card flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-2">
                    <Settings size={16} className="text-primary" />
                    <h2 className="font-semibold text-sm text-text-primary">Configure Node</h2>
                  </div>
                  <button onClick={() => setSelected(null)} className="text-text-tertiary hover:text-text-primary text-sm font-semibold">×</button>
                </div>

                <NodeConfigForm
                  agent={selectedAgent}
                  onChange={(updates) => updateAgent(selectedAgent.id, updates)}
                  onDelete={() => removeAgent(selectedAgent.id)}
                  allAgents={agents}
                  edges={edges}
                  onAddEdge={addEdge}
                />
              </div>
            </aside>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}

function TopologyCanvas({
  agents, edges, selected, onSelect,
}: {
  agents: AgentNode[];
  edges: TopologyEdge[];
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const W = 700, H = 480;
  const PAD = 60;
  const NODER = 26; // node radius

  const xs = agents.map((a) => a.position_x);
  const ys = agents.map((a) => a.position_y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const cx = (x: number) => PAD + ((x - minX) / rangeX) * (W - 2 * PAD);
  const cy = (y: number) => PAD + ((y - minY) / rangeY) * (H - 2 * PAD);

  // Colors mapping to B2B specifications
  const NODE_FILL: Record<string, string> = {
    supplier: "#EF9F27",
    distributor: "#534AB7",
    manufacturer: "#1D9E75",
    retailer: "#D85A30",
  };
  const NODE_STROKE: Record<string, string> = {
    supplier: "#F5B041",
    distributor: "#6A60D0",
    manufacturer: "#27AE60",
    retailer: "#E57373",
  };
  const NODE_EMOJI: Record<string, string> = {
    supplier: "🏭", distributor: "🚛", manufacturer: "⚙️", retailer: "🏪",
  };

  return (
    <div className="flex-1 bg-surface-card border border-border-custom rounded-card overflow-hidden relative" style={{ minHeight: 450 }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {/* Background grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
          </pattern>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="6" refY="2" orient="auto">
            <path d="M0,0 L0,4 L6,2 z" fill="rgba(83, 74, 183, 0.4)" />
          </marker>
          <marker id="arrow-sel" markerWidth="6" markerHeight="6" refX="6" refY="2" orient="auto">
            <path d="M0,0 L0,4 L6,2 z" fill="#534AB7" />
          </marker>
        </defs>
        <rect width={W} height={H} fill="url(#grid)" />

        {/* Edges */}
        {edges.map((e) => {
          const src = agents.find((a) => a.id === e.source);
          const tgt = agents.find((a) => a.id === e.target);
          if (!src || !tgt) return null;
          const x1 = cx(src.position_x), y1 = cy(src.position_y);
          const x2 = cx(tgt.position_x), y2 = cy(tgt.position_y);
          const dx = x2 - x1, dy = y2 - y1;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const ex1 = x1 + (dx / len) * (NODER + 2);
          const ey1 = y1 + (dy / len) * (NODER + 2);
          const ex2 = x2 - (dx / len) * (NODER + 8);
          const ey2 = y2 - (dy / len) * (NODER + 8);
          const isSrcSel = selected === e.source || selected === e.target;
          return (
            <g key={e.id}>
              <line
                x1={ex1} y1={ey1} x2={ex2} y2={ey2}
                stroke={isSrcSel ? "#534AB7" : "rgba(255,255,255,0.12)"}
                strokeWidth={isSrcSel ? 2 : 1.2}
                strokeDasharray={isSrcSel ? undefined : "4 3"}
                markerEnd={isSrcSel ? "url(#arrow-sel)" : "url(#arrow)"}
              />
              {e.transport_cost > 0 && (
                <text
                  x={(x1 + x2) / 2}
                  y={(y1 + y2) / 2 - 6}
                  fill="rgba(255,255,255,0.3)"
                  fontSize={8}
                  textAnchor="middle"
                >
                  ${e.transport_cost}/u
                </text>
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {agents.map((agent) => {
          const x = cx(agent.position_x);
          const y = cy(agent.position_y);
          const fill = NODE_FILL[agent.type] ?? "#534AB7";
          const stroke = NODE_STROKE[agent.type] ?? "#6A60D0";
          const isSel = selected === agent.id;
          return (
            <g
              key={agent.id}
              style={{ cursor: "pointer" }}
              onClick={() => onSelect(agent.id === selected ? null : agent.id)}
            >
              {/* Selection ring */}
              {isSel && (
                <circle cx={x} cy={y} r={NODER + 6} fill="none" stroke="#534AB7" strokeWidth={1.5} strokeDasharray="3 2" />
              )}
              {/* Node circle */}
              <circle
                cx={x} cy={y} r={NODER}
                fill={`${fill}20`}
                stroke={isSel ? "#534AB7" : stroke}
                strokeWidth={isSel ? 2.5 : 1.5}
              />
              {/* Emoji icon */}
              <text x={x} y={y - 3} textAnchor="middle" fontSize={14} dominantBaseline="middle">
                {NODE_EMOJI[agent.type] ?? "📦"}
              </text>
              {/* Agent name */}
              <text x={x} y={y + NODER + 12} textAnchor="middle" fontSize={9} fill={isSel ? "#534AB7" : "var(--foreground)"} fontWeight={isSel ? "600" : "normal"}>
                {agent.name.length > 14 ? agent.name.slice(0, 13) + "…" : agent.name}
              </text>
              {/* Capacity badge */}
              <text x={x} y={y + NODER + 22} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.25)">
                cap {agent.capacity}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-3 left-4 flex gap-3">
        {(["supplier", "distributor", "manufacturer", "retailer"] as const)
          .filter((role) => agents.some((a) => a.type === role))
          .map((role) => (
            <div key={role} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: NODE_FILL[role] }} />
              <span className="text-[10px] text-text-secondary capitalize">{role}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

function NodeConfigForm({ agent, onChange, onDelete, allAgents, edges, onAddEdge }: {
  agent: AgentNode;
  onChange: (u: Partial<AgentNode>) => void;
  onDelete: () => void;
  allAgents: AgentNode[];
  edges: TopologyEdge[];
  onAddEdge: (src: string, tgt: string) => void;
}) {
  const Field = ({ label, field, type = "number" }: { label: string; field: keyof AgentNode; type?: string }) => (
    <div className="mb-4">
      <label className="block text-[11px] text-text-secondary mb-1 font-medium">{label}</label>
      <input
        type={type}
        value={String(agent[field])}
        onChange={(e) => onChange({ [field]: type === "number" ? Number(e.target.value) : e.target.value } as Partial<AgentNode>)}
        className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-text-primary text-xs focus:outline-none focus:border-primary transition"
      />
    </div>
  );

  const connectableAgents = allAgents.filter((a) => {
    if (a.id === agent.id) return false;
    return !edges.find((e) => (e.source === agent.id && e.target === a.id) || (e.source === a.id && e.target === agent.id));
  });

  return (
    <div className="space-y-4">
      <div className="bg-surface border border-border-custom rounded-lg p-3 flex items-center gap-2">
        <span className="text-base">{NODE_ICONS[agent.type]}</span>
        <div>
          <p className="text-xs font-semibold text-text-primary uppercase tracking-wider">{agent.type}</p>
          <p className="text-[10px] text-text-tertiary">ID: {agent.id}</p>
        </div>
      </div>

      <div className="space-y-1">
        <Field label="Name" field="name" type="text" />
        <Field label="Capacity (units)" field="capacity" />
        <Field label="Lead time min (days)" field="lead_time_min" />
        <Field label="Lead time max (days)" field="lead_time_max" />
        <Field label="Holding cost ($/unit/day)" field="holding_cost" />
        <Field label="Stockout penalty ($/unit)" field="stockout_penalty" />

        <div className="mb-4">
          <label className="block text-[11px] text-text-secondary mb-1 font-medium">Policy</label>
          <select
            value={agent.policy}
            onChange={(e) => onChange({ policy: e.target.value })}
            className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-text-primary text-xs focus:outline-none focus:border-primary transition"
          >
            <option value="mappo_gnn">MAPPO + GNN (recommended)</option>
            <option value="base_stock">Base-stock policy</option>
            <option value="ss_policy">(s,S) policy</option>
          </select>
        </div>

        {connectableAgents.length > 0 && (
          <div className="mb-4">
            <label className="block text-[11px] text-text-secondary mb-2 font-medium">Connect to</label>
            <div className="space-y-1">
              {connectableAgents.map((a) => (
                <button
                  key={a.id}
                  onClick={() => onAddEdge(agent.id, a.id)}
                  className="w-full text-left text-xs bg-surface hover:bg-primary/5 border border-border-custom px-3 py-1.5 rounded-lg transition text-text-secondary hover:text-text-primary"
                >
                  + {a.name}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <button
        onClick={onDelete}
        className="w-full text-xs text-danger hover:text-danger/90 border border-danger/20 hover:border-danger/30 py-2 rounded-lg transition flex items-center justify-center gap-1 bg-danger/5"
      >
        <Trash2 size={12} /> Remove node
      </button>
    </div>
  );
}
