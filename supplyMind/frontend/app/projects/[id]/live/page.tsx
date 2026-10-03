"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import {
  api,
  DBConnectionConfig, DBTestResult, DBPreviewResponse,
  LiveControlResponse, TableConfig, MultiTableImportResult,
  ExplainResponse, ExplainRequest, NodeRecommendation
} from "@/lib/api";
import ProjectNav from "@/components/ProjectNav";
import DashboardLayout from "@/components/DashboardLayout";
import { Database, Plus, ShieldCheck, Download, Server, Maximize2, X } from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, Legend
} from "recharts";

type NodeType = "retailer" | "distributor" | "supplier";

interface ExtTableCfg extends Required<TableConfig> {
  node_name_col: string;
  node_type_col: string;
  capacity_col: string;
}

interface UserNode {
  uid: string;
  node_id: string;
  name: string;
  node_type: NodeType;
  capacity: number;
  inventory: number;
  demand: number;
  unit_cost?: number;
  lead_time_days?: number;
}

const DEFAULT_DEMAND: Record<NodeType, number> = { retailer: 20, distributor: 60, supplier: 180 };
const DEFAULT_CAP:    Record<NodeType, number> = { retailer: 1000, distributor: 2500, supplier: 5000 };

const TYPE_COLOR: Record<NodeType, string> = {
  retailer:    "bg-danger/5 border-danger/20 text-danger",
  distributor: "bg-primary/5 border-primary/20 text-primary",
  supplier:    "bg-warning/5 border-warning/20 text-warning",
};

const URGENCY_STYLE: Record<string, string> = {
  critical: "bg-danger/10 text-danger border border-danger/20",
  high:     "bg-warning/10 text-warning border border-warning/20",
  medium:   "bg-primary/10 text-primary border border-primary/20",
  low:      "bg-success/10 text-success border border-success/20",
};

function invBarColor(pct: number) {
  if (pct < 0.1)  return "bg-danger";
  if (pct < 0.25) return "bg-warning";
  if (pct < 0.5)  return "bg-primary";
  return "bg-success";
}

function mkUid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function makeDefaultNodes(): UserNode[] {
  return [
    { uid: mkUid(), node_id: "retailer_0", name: "Retailer A",    node_type: "retailer",    capacity: 1000, inventory: 300,  demand: 20  },
    { uid: mkUid(), node_id: "retailer_1", name: "Retailer B",    node_type: "retailer",    capacity: 1000, inventory: 300,  demand: 20  },
    { uid: mkUid(), node_id: "retailer_2", name: "Retailer C",    node_type: "retailer",    capacity: 1000, inventory: 300,  demand: 20  },
    { uid: mkUid(), node_id: "dist_0",     name: "Distributor 1", node_type: "distributor", capacity: 2500, inventory: 750,  demand: 60  },
    { uid: mkUid(), node_id: "dist_1",     name: "Distributor 2", node_type: "distributor", capacity: 2500, inventory: 750,  demand: 60  },
    { uid: mkUid(), node_id: "supplier_0", name: "Supplier",      node_type: "supplier",    capacity: 5000, inventory: 1500, demand: 180 },
  ];
}

export default function LiveControlPage() {
  const { id: projectId } = useParams<{ id: string }>();

  const [nodes, setNodes]             = useState<UserNode[]>(makeDefaultNodes);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName]         = useState("");
  const [newType, setNewType]         = useState<NodeType>("retailer");
  const [newCap,  setNewCap]          = useState(1000);

  const [inferLoading, setInferLoading] = useState(false);
  const [result, setResult]             = useState<LiveControlResponse | null>(null);
  const [inferError, setInferError]     = useState<string | null>(null);

  const [zoomedChart, setZoomedChart]   = useState<{ title: string; render: () => React.ReactNode } | null>(null);

  const [explainLoading, setExplainLoading] = useState<string | null>(null);
  const [explainData, setExplainData]       = useState<ExplainResponse | null>(null);

  const [showDb, setShowDb]             = useState(false);
  const [dbType, setDbType]             = useState<"postgresql" | "mysql" | "sqlite">("postgresql");
  const [host, setHost]                 = useState("localhost");
  const [port, setPort]                 = useState(5432);
  const [database, setDatabase]         = useState("");
  const [username, setUsername]         = useState("");
  const [password, setPassword]         = useState("");
  const [testResult, setTestResult]     = useState<DBTestResult | null>(null);
  
  const [selectedTables, setSelectedTables] = useState<ExtTableCfg[]>([]);
  const [preview, setPreview]               = useState<DBPreviewResponse | null>(null);
  const [previewFor, setPreviewFor]         = useState<string>("");
  const [importResults, setImportResults]   = useState<MultiTableImportResult["results"]>([]);
  const [importErrors, setImportErrors]     = useState<MultiTableImportResult["errors"]>([]);
  const [dbLoading, setDbLoading]           = useState(false);
  const [dbError, setDbError]               = useState<string | null>(null);

  function addNode() {
    if (!newName.trim()) return;
    const uid = mkUid();
    setNodes(prev => [...prev, {
      uid, node_id: uid, name: newName.trim(), node_type: newType,
      capacity: newCap, inventory: Math.round(newCap * 0.3),
      demand: DEFAULT_DEMAND[newType],
    }]);
    setNewName(""); setNewType("retailer"); setNewCap(1000); setShowAddForm(false);
  }

  function removeNode(uid: string) {
    setNodes(prev => prev.filter(n => n.uid !== uid));
    setResult(null);
  }

  function updateNode(uid: string, field: keyof UserNode, val: any) {
    setNodes(prev => prev.map(n => n.uid === uid ? { ...n, [field]: val } : n));
  }

  function resetTopology() { setNodes(makeDefaultNodes()); setResult(null); }

  function buildConnCfg(): DBConnectionConfig {
    return { db_type: dbType, host, port, database, username, password };
  }

  function toggleTable(name: string) {
    setSelectedTables(prev =>
      prev.some(t => t.table_name === name)
        ? prev.filter(t => t.table_name !== name)
        : [...prev, { table_name: name, inventory_col: "inventory", demand_col: "demand", node_col: "node_id", node_name_col: "node_name", node_type_col: "node_type", capacity_col: "capacity" }]
    );
  }

  function updateTableCfg(name: string, field: string, val: string) {
    setSelectedTables(prev => prev.map(t => t.table_name === name ? { ...t, [field]: val } : t));
  }

  async function handleTest() {
    setDbLoading(true); setDbError(null); setTestResult(null); setSelectedTables([]);
    try {
      const r = await api.dbConnector.test(buildConnCfg());
      setTestResult(r);
      if (!r.success) setDbError(r.error ?? "Connection failed");
    } catch (e) { setDbError(e instanceof Error ? e.message : String(e)); }
    finally { setDbLoading(false); }
  }

  async function handlePreviewTable(tc: ExtTableCfg) {
    setDbLoading(true); setDbError(null); setPreviewFor(tc.table_name);
    try {
      const cfg = { ...buildConnCfg(), table_name: tc.table_name,
                    inventory_col: tc.inventory_col, demand_col: tc.demand_col };
      setPreview(await api.dbConnector.preview(cfg));
    } catch (e) { setDbError(e instanceof Error ? e.message : String(e)); }
    finally { setDbLoading(false); }
  }

  async function handleImportAll() {
    if (!selectedTables.length) { setDbError("Select at least one table first"); return; }
    setDbLoading(true); setDbError(null); setImportResults([]); setImportErrors([]);
    try {
      const req: MultiTableImportRequest = {
        db_type: dbType, host, port, database, username, password,
        tables: selectedTables,
      };
      const r = await api.dbConnector.importMulti(req, projectId);
      setImportResults(r.results);
      setImportErrors(r.errors);

      if (r.results.length > 0) {
        const newNodes: UserNode[] = [];
        for (const res of r.results) {
          const tc = selectedTables.find(t => t.table_name === res.table_name);
          if (!tc || !res.data) continue;
          const nodeIdx  = res.columns.indexOf(tc.node_col);
          const invIdx   = res.columns.indexOf(tc.inventory_col);
          const demIdx   = res.columns.indexOf(tc.demand_col);
          const nameIdx  = res.columns.indexOf(tc.node_name_col);
          const typeIdx  = res.columns.indexOf(tc.node_type_col);
          const flexCapIdx = res.columns.indexOf(tc.capacity_col);
          if (nodeIdx < 0) continue;

          for (const row of res.data) {
            const nId  = String(row[nodeIdx]);
            const inv  = invIdx >= 0 ? Number(row[invIdx]) : 0;
            const dem  = demIdx >= 0 ? Number(row[demIdx]) : 10;
            const name = nameIdx >= 0 ? String(row[nameIdx]) : nId;
            const rawType = typeIdx >= 0 ? String(row[typeIdx]).toLowerCase() : "retailer";
            const nodeType: NodeType = rawType === "distributor" ? "distributor" : rawType === "supplier" ? "supplier" : "retailer";
            const cap  = flexCapIdx >= 0 ? Number(row[flexCapIdx]) || 1000 : 1000;
            if (isNaN(inv) && isNaN(dem)) continue;

            newNodes.push({
              uid: mkUid(),
              node_id: nId,
              name,
              node_type: nodeType,
              capacity: cap,
              inventory: isNaN(inv) ? 0 : inv,
              demand: isNaN(dem) ? 10 : dem,
            });
          }
        }
        if (newNodes.length > 0) {
          setNodes(newNodes);
          setResult(null);
        }
      }
    } catch (e) { setDbError(e instanceof Error ? e.message : String(e)); }
    finally { setDbLoading(false); }
  }

  async function handleInfer() {
    if (!nodes.length) { setInferError("Add at least one node first"); return; }
    setInferLoading(true); setInferError(null); setResult(null); setExplainData(null);
    try {
      const payload = nodes.map(n => ({
        node_id: n.node_id, node_name: n.name, node_type: n.node_type,
        inventory_units: n.inventory, capacity_units: n.capacity, demand_units: n.demand,
        unit_cost: n.unit_cost, lead_time_days: n.lead_time_days
      }));
      setResult(await api.liveControl.infer({ project_id: projectId, nodes: payload }));
    } catch (e) { setInferError(e instanceof Error ? e.message : String(e)); }
    finally { setInferLoading(false); }
  }

  async function handleExplain(nodeId: string) {
    setExplainLoading(nodeId); setExplainData(null);
    try {
      const payload = nodes.map(n => ({
        node_id: n.node_id, node_name: n.name, node_type: n.node_type,
        inventory_units: n.inventory, capacity_units: n.capacity, demand_units: n.demand,
        unit_cost: n.unit_cost, lead_time_days: n.lead_time_days
      }));
      const req: ExplainRequest = { project_id: projectId, nodes: payload, target_node_id: nodeId };
      const res = await api.liveControl.explain(req);
      setExplainData(res);
    } catch (e) { 
      alert("Explanation failed: " + (e instanceof Error ? e.message : String(e)));
    } finally { setExplainLoading(null); }
  }

  function exportCsv() {
    if (!result) return;
    const header = ["node_id","node_name","node_type","inventory","capacity","inv_pct","order_units","urgency"];
    const rows = result.recommendations.map(r => [
      r.node_id, r.node_name, r.node_type, r.inventory_units, r.capacity_units,
      (r.inventory_pct * 100).toFixed(1) + "%", r.order_units, r.urgency,
    ]);
    const csv = [header, ...rows].map(r => r.join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "live_recommendations_" + Date.now() + ".csv";
    a.click();
  }

  const retailers    = nodes.filter(n => n.node_type === "retailer");
  const distributors = nodes.filter(n => n.node_type === "distributor");
  const suppliers    = nodes.filter(n => n.node_type === "supplier");
  const recMap = result
    ? Object.fromEntries(result.recommendations.map(r => [r.node_id, r]))
    : {} as Record<string, NodeRecommendation>;

  // Pre-compute live KPIs and chart data (only populated when result exists)
  const recs           = result?.recommendations ?? [];
  const criticalNodes  = recs.filter(r => r.urgency === "critical").length;
  const healthyNodes   = recs.filter(r => r.inventory_pct >= 0.5).length;
  const shortageRisk   = recs.filter(r => r.inventory_pct < 0.1).length;
  const avgStockPct    = recs.length > 0
    ? Math.round(recs.reduce((s, r) => s + r.inventory_pct * 100, 0) / recs.length)
    : 0;
  const networkFillRate = recs.length > 0
    ? Math.round((healthyNodes / recs.length) * 100)
    : 0;
  const invChartData = recs.map(r => ({
    name:  r.node_name.length > 12 ? r.node_name.slice(0, 12) + "…" : r.node_name,
    stock: Math.round(r.inventory_pct * 100),
    order: Math.round(r.order_units),
    type:  r.node_type,
  }));
  const nodeColors: Record<string, string> = {
    retailer: "#D85A30", distributor: "#534AB7", supplier: "#EF9F27"
  };


  return (
    <DashboardLayout>
      <div className="h-full flex flex-col overflow-hidden">
        <ProjectNav projectId={projectId} step="live" />

        <div className="flex-1 overflow-auto p-8 select-none space-y-6 max-w-7xl mx-auto w-full">
          <div>
            <h1 className="text-xl font-bold text-text-primary tracking-tight">Live Database Control</h1>
            <p className="text-xs text-text-secondary mt-1">
              Verify live inventory levels, trigger instant MAPPO+GNN evaluations, and get optimal GNN action orders.
            </p>
          </div>

          {/* Topology summary bar */}
          <div className="bg-surface-card border border-border-custom rounded-card p-4 flex items-center gap-4 flex-wrap text-xs font-semibold">
            <span className="text-text-primary">{nodes.length} Active Nodes</span>
            <span className="h-3.5 w-px bg-border-custom" />
            <span className="text-danger">{retailers.length} Retailers</span>
            <span className="text-primary">{distributors.length} Distributors</span>
            <span className="text-warning">{suppliers.length} Suppliers</span>
            <span className="h-3.5 w-px bg-border-custom" />
            <span className="text-[10px] text-text-tertiary font-mono italic">Independent GNN evaluation thread</span>
            
            <div className="ml-auto flex gap-2">
              <button onClick={resetTopology} className="px-3 py-1.5 border border-border-custom hover:border-text-secondary/25 bg-surface text-text-secondary hover:text-text-primary rounded-lg transition">
                Reset Baseline
              </button>
              <button onClick={() => setShowAddForm(s => !s)} className="px-3 py-1.5 bg-primary hover:bg-primary/95 text-white rounded-lg transition shadow-premium flex items-center gap-1">
                <Plus size={14} /> Add node
              </button>
            </div>
          </div>

          {/* Add node form */}
          {showAddForm && (
            <div className="bg-surface-card border border-primary/20 rounded-card p-5 space-y-4 shadow-premium">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Add Custom Echelon Node</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 items-end">
                <div>
                  <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider block mb-1">Node Name</label>
                  <input
                    value={newName}
                    onChange={e => setNewName(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") addNode(); }}
                    placeholder="e.g. Paris Node"
                    className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider block mb-1">Echelon Type</label>
                  <select
                    value={newType}
                    onChange={e => { const t = e.target.value as NodeType; setNewType(t); setNewCap(DEFAULT_CAP[t]); }}
                    className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition"
                  >
                    <option value="retailer">Retailer</option>
                    <option value="distributor">Distributor</option>
                    <option value="supplier">Supplier</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider block mb-1">Capacity (units)</label>
                  <input
                    type="number" value={newCap} min={1}
                    onChange={e => setNewCap(Number(e.target.value))}
                    className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition"
                  />
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={addNode} disabled={!newName.trim()}
                    className="flex-1 py-2 bg-primary hover:bg-primary/95 disabled:opacity-50 text-white rounded-lg text-xs font-semibold transition shadow-premium"
                  >Add</button>
                  <button
                    onClick={() => setShowAddForm(false)}
                    className="py-2 px-3 border border-border-custom hover:bg-primary/5 rounded-lg text-xs text-text-secondary transition"
                  >x</button>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
            
            {/* Left panel: inventory snapshot */}
            <section className="space-y-4">
              <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Inventory Snapshot</h2>

              {nodes.length === 0 && (
                <div className="bg-surface-card border border-dashed border-border-custom rounded-card p-10 text-center text-text-tertiary text-xs">
                  No echelons defined. Click &quot;Add node&quot; to begin.
                </div>
              )}

              {(["retailer", "distributor", "supplier"] as NodeType[]).map(type => {
                const group = nodes.filter(n => n.node_type === type);
                if (!group.length) return null;
                const label = type === "retailer" ? "Retailers" : type === "distributor" ? "Distributors" : "Suppliers";
                return (
                  <div key={type} className="space-y-2">
                    <p className="text-[10px] font-bold text-text-tertiary uppercase tracking-wider">{label}</p>
                    <div className="space-y-2">
                      {group.map(node => {
                        const pct = node.inventory / node.capacity;
                        const rec = recMap[node.node_id];
                        return (
                          <div key={node.uid} className={`border border-border-custom bg-surface-card rounded-card p-4 hover:border-primary/10 transition`}>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="font-semibold text-xs text-text-primary truncate">{node.name}</span>
                                {rec && (
                                  <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full shrink-0 ${URGENCY_STYLE[rec.urgency]}`}>
                                    {rec.urgency}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                <span className="text-[10px] text-text-secondary font-mono">{node.inventory.toFixed(0)} / {node.capacity} u</span>
                                <button
                                  onClick={() => removeNode(node.uid)}
                                  className="text-text-tertiary hover:text-danger transition text-xs font-bold"
                                  title="Remove node"
                                >x</button>
                              </div>
                            </div>
                            <div className="h-1.5 bg-surface border border-border-custom rounded-full mb-3 overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${invBarColor(pct)}`}
                                style={{ width: `${Math.min(pct * 100, 100)}%` }}
                              />
                            </div>
                            <input
                              type="range" min={0} max={node.capacity} step={node.capacity / 100}
                              value={node.inventory}
                              onChange={e => updateNode(node.uid, "inventory", Number(e.target.value))}
                              className="w-full accent-primary mb-3"
                            />
                            
                            <div className="flex flex-wrap items-center gap-3 pt-2.5 border-t border-border-custom">
                              <div className="flex items-center gap-1.5">
                                <label className="text-[10px] text-text-secondary">Daily demand:</label>
                                <input
                                  type="number" min={0} value={node.demand}
                                  onChange={e => updateNode(node.uid, "demand", Number(e.target.value))}
                                  className="w-14 bg-surface border border-border-custom rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-primary transition"
                                />
                              </div>
                              <div className="flex items-center gap-1.5">
                                <label className="text-[10px] text-text-secondary" title="Unit Cost">Cost ($):</label>
                                <input
                                  type="number" min={0} value={node.unit_cost ?? ""} placeholder="auto"
                                  onChange={e => updateNode(node.uid, "unit_cost", e.target.value ? Number(e.target.value) : undefined)}
                                  className="w-12 bg-surface border border-border-custom rounded px-1.5 py-1 text-[11px] text-text-primary focus:outline-none focus:border-primary transition"
                                />
                              </div>
                              <div className="flex items-center gap-1.5">
                                <label className="text-[10px] text-text-secondary" title="Lead Time (Days)">LT (d):</label>
                                <input
                                  type="number" min={0} value={node.lead_time_days ?? ""} placeholder="auto"
                                  onChange={e => updateNode(node.uid, "lead_time_days", e.target.value ? Number(e.target.value) : undefined)}
                                  className="w-12 bg-surface border border-border-custom rounded px-1.5 py-1 text-[11px] text-text-primary focus:outline-none focus:border-primary transition"
                                />
                              </div>
                              {rec && (
                                <div className="ml-auto text-[11px] text-text-secondary">
                                  Order Action: <span className="text-primary font-bold">{rec.order_units.toFixed(0)} units</span>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}

              <button
                onClick={handleInfer} disabled={inferLoading || nodes.length === 0}
                className="w-full py-3 bg-primary hover:bg-primary/95 disabled:opacity-50 rounded-lg font-semibold text-xs transition flex items-center justify-center gap-1.5 shadow-premium"
              >
                {inferLoading ? "Running GNN Engine..." : "Get AI recommendations"}
              </button>

              {inferError && (
                <div className="bg-danger/10 border border-danger/20 text-danger rounded-lg p-3 text-xs">{inferError}</div>
              )}
            </section>

            {/* Right panel: results + DB connector */}
            <section className="space-y-6">
              {!result && (
                <div className="bg-surface-card border border-dashed border-border-custom rounded-card p-10 text-center text-text-tertiary text-xs">
                  Snapshot configuration ready. Click &quot;Get AI recommendations&quot;.
                </div>
              )}

              {result && (
                <div className="space-y-5">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Inference Recommendations</h2>
                    <button onClick={exportCsv} className="border border-border-custom hover:border-text-secondary/25 bg-surface-card rounded px-2.5 py-1 text-[10px] font-semibold text-text-primary flex items-center gap-1 shadow-premium">
                      <Download size={10} /> CSV
                    </button>
                  </div>

                  {/* KPI Strip */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: "Network Fill Rate",  value: `${networkFillRate}%`,                                   good: networkFillRate >= 80 },
                      { label: "Total Order Units",   value: result.total_order_units.toLocaleString(),               good: null },
                      { label: "Estimated Cost",      value: `$${result.total_estimated_cost.toLocaleString()}`,      good: null },
                      { label: "Critical Nodes",      value: String(criticalNodes),                                   good: criticalNodes === 0 },
                      { label: "Avg Stock Level",     value: `${avgStockPct}%`,                                       good: avgStockPct >= 50 },
                      { label: "Healthy Nodes",       value: `${healthyNodes} / ${recs.length}`,                     good: healthyNodes === recs.length },
                      { label: "Shortage Risk Nodes", value: String(shortageRisk),                                    good: shortageRisk === 0 },
                      { label: "Nodes Analysed",      value: String(recs.length),                                     good: null },
                    ].map(k => (
                      <div key={k.label} className="bg-surface-card border border-border-custom rounded-card p-4 flex flex-col justify-between h-[95px] hover:border-primary/20 transition">
                        <p className="text-[9px] font-semibold text-text-tertiary uppercase tracking-wider">{k.label}</p>
                        <h3 className="text-lg font-bold text-text-primary tracking-tight">{k.value}</h3>
                        {k.good !== null && (
                          <span className={`text-[9px] font-bold ${k.good ? "text-success" : "text-warning"}`}>
                            {k.good ? "Optimal" : "Monitor"}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Charts */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Stock Level chart */}
                    <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-3">
                      <div className="flex justify-between items-start">
                        <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Stock Level by Node (%)</p>
                        <button
                          onClick={() => setZoomedChart({ title: "Stock Level by Node (%)", render: () => (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={invChartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <YAxis domain={[0, 100]} tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" tickFormatter={v => `${v}%`} />
                                <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v: any) => [`${v}%`, "Stock"]} />
                                <Bar dataKey="stock" radius={[8, 8, 0, 0]}>
                                  {invChartData.map((d, i) => <Cell key={i} fill={nodeColors[d.type] ?? "#534AB7"} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )})}
                          className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={invChartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" tickFormatter={v => `${v}%`} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v: any) => [`${v}%`, "Stock"]} />
                          <Bar dataKey="stock" radius={[4, 4, 0, 0]}>
                            {invChartData.map((d, i) => <Cell key={i} fill={nodeColors[d.type] ?? "#534AB7"} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Order Units chart */}
                    <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-3">
                      <div className="flex justify-between items-start">
                        <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Recommended Order Units by Node</p>
                        <button
                          onClick={() => setZoomedChart({ title: "Recommended Order Units by Node", render: () => (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={invChartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v: any) => [`${v} units`, "Order"]} />
                                <Bar dataKey="order" radius={[8, 8, 0, 0]}>
                                  {invChartData.map((d, i) => <Cell key={i} fill={nodeColors[d.type] ?? "#534AB7"} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )})}
                          className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={invChartData} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v: any) => [`${v} units`, "Order"]} />
                          <Bar dataKey="order" radius={[4, 4, 0, 0]}>
                            {invChartData.map((d, i) => <Cell key={i} fill={nodeColors[d.type] ?? "#534AB7"} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Node cards */}
                  {result.topology_note && (
                    <p className="text-[10px] text-text-tertiary italic">{result.topology_note}</p>
                  )}
                  <div className="grid grid-cols-2 gap-3">
                    {result.recommendations.map(r => (
                      <div key={r.node_id} className="bg-surface-card border border-border-custom rounded-lg p-4 space-y-2 hover:border-primary/10 transition">
                        <div className="flex items-start justify-between gap-1.5">
                          <span className="text-xs font-semibold text-text-primary truncate">{r.node_name}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full shrink-0 ${URGENCY_STYLE[r.urgency]}`}>{r.urgency}</span>
                        </div>
                        <div className="h-1 bg-surface border border-border-custom rounded-full overflow-hidden">
                          <div className={`h-full ${invBarColor(r.inventory_pct)}`} style={{ width: `${Math.min(r.inventory_pct * 100, 100)}%` }} />
                        </div>
                        <p className="text-lg font-bold text-text-primary">{r.order_units.toFixed(0)} <span className="text-[10px] text-text-secondary font-normal">units</span></p>
                        <div className="flex flex-wrap gap-1.5 pt-1.5 border-t border-border-custom text-[9px] font-medium">
                          <span className="bg-success/10 border border-success/20 text-success px-1.5 py-0.5 rounded">
                            Cost: ${r.estimated_cost.toLocaleString(undefined, {maximumFractionDigits: 0})}
                          </span>
                          <span className="bg-primary/10 border border-primary/20 text-primary px-1.5 py-0.5 rounded">
                            ETA: {r.estimated_lead_time_days}d
                          </span>
                        </div>
                        <div className="flex items-center justify-between pt-1 text-[10px]">
                          <span className="text-text-tertiary">Stock: {(r.inventory_pct * 100).toFixed(0)}%</span>
                          <button
                            onClick={() => handleExplain(r.node_id)}
                            disabled={explainLoading === r.node_id}
                            className="font-bold text-primary hover:underline uppercase text-[9px]"
                          >
                            {explainLoading === r.node_id ? "Explaining..." : "Explain"}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* AI Narrative */}
                  <div className="bg-primary/5 border border-primary/25 rounded-card p-4 space-y-1">
                    <p className="text-[10px] font-semibold text-primary uppercase tracking-wider">AI Narrative Analysis</p>
                    <p className="text-xs text-text-secondary leading-relaxed whitespace-pre-wrap">{result.analysis}</p>
                  </div>
                </div>
              )}

              {/* DB Connector */}
              <div className="bg-surface-card border border-border-custom rounded-card overflow-hidden">
                <button
                  onClick={() => setShowDb(s => !s)}
                  className="w-full px-5 py-4 flex items-center justify-between hover:bg-primary/5 transition font-semibold text-xs text-text-primary uppercase tracking-wider border-b border-border-custom"
                >
                  <span className="flex items-center gap-1.5"><Server size={14} className="text-primary" /> Connect Production Database</span>
                  <span className="text-text-tertiary">{showDb ? "Hide" : "Expand"}</span>
                </button>

                {showDb && (
                  <div className="p-5 space-y-4">
                    {/* DB Type select */}
                    <div className="grid grid-cols-3 gap-2">
                      {(["postgresql", "mysql", "sqlite"] as const).map(t => (
                        <button
                          key={t}
                          onClick={() => { setDbType(t); setPort(t === "sqlite" ? 0 : t === "mysql" ? 3306 : 5432); }}
                          className={`py-2 px-3 rounded-lg text-xs border font-semibold transition ${
                            dbType === t
                              ? "bg-primary/10 border-primary text-primary"
                              : "border-border-custom text-text-secondary hover:border-text-secondary/20"
                          }`}
                        >
                          {t === "postgresql" ? "PostgreSQL" : t === "mysql" ? "MySQL" : "SQLite"}
                        </button>
                      ))}
                    </div>

                    {dbType !== "sqlite" ? (
                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Host</label>
                          <input value={host} onChange={e => setHost(e.target.value)} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" placeholder="localhost" />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Port</label>
                          <input type="number" value={port} onChange={e => setPort(Number(e.target.value))} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Database</label>
                          <input value={database} onChange={e => setDatabase(e.target.value)} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" placeholder="my_db" />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Username</label>
                          <input value={username} onChange={e => setUsername(e.target.value)} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" />
                        </div>
                        <div className="col-span-2 space-y-1">
                          <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Password</label>
                          <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" />
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <label className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">File Path</label>
                        <input value={database} onChange={e => setDatabase(e.target.value)} className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition" placeholder="C:/path/to/db.sqlite" />
                      </div>
                    )}

                    <button onClick={handleTest} disabled={dbLoading} className="px-4 py-2 bg-primary hover:bg-primary/95 disabled:opacity-50 rounded-lg text-xs font-semibold transition w-full shadow-premium">
                      {dbLoading && !testResult ? "Connecting…" : "Test DB Connection"}
                    </button>

                    {dbError && <div className="bg-danger/10 border border-danger/25 text-danger rounded-lg p-3 text-xs">{dbError}</div>}
                    {testResult?.success && (
                      <div className="bg-success/10 border border-success/25 text-success rounded-lg p-3 text-xs flex items-center gap-2">
                        <ShieldCheck size={16} /> Connected — {testResult.tables.length} table{testResult.tables.length !== 1 ? "s" : ""} found.
                      </div>
                    )}

                    {testResult?.success && testResult.tables.length > 0 && (
                      <div className="space-y-2 pt-2 border-t border-border-custom">
                        <p className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Select tables to map</p>
                        <div className="flex flex-wrap gap-1.5">
                          {testResult.tables.map(t => {
                            const isSelected = selectedTables.some(s => s.table_name === t);
                            return (
                              <button
                                key={t}
                                onClick={() => toggleTable(t)}
                                className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition ${
                                  isSelected
                                    ? "bg-primary/10 border-primary text-primary"
                                    : "border-border-custom text-text-secondary hover:text-text-primary bg-surface"
                                }`}
                              >
                                {isSelected ? "✓ " : ""}{t}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {selectedTables.length > 0 && (
                      <div className="space-y-3 pt-3 border-t border-border-custom">
                        <p className="text-[10px] text-text-secondary font-semibold uppercase tracking-wider">Column mapping configurations</p>
                        {selectedTables.map(st => (
                          <div key={st.table_name} className="bg-surface border border-border-custom rounded-lg p-3 space-y-3">
                            <div className="flex items-center justify-between border-b border-border-custom pb-2">
                              <span className="text-xs font-semibold text-text-primary">{st.table_name}</span>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => handlePreviewTable(st)}
                                  disabled={dbLoading}
                                  className="px-2 py-0.5 border border-border-custom hover:border-text-secondary/25 text-[10px] rounded transition bg-surface-card"
                                >
                                  Preview
                                </button>
                                <button
                                  onClick={() => toggleTable(st.table_name)}
                                  className="text-xs text-danger font-bold"
                                >
                                  ×
                                </button>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              {/* Inputs for mapping */}
                              {["node_col", "node_name_col", "node_type_col", "capacity_col", "inventory_col", "demand_col"].map((field) => (
                                <div key={field} className="space-y-0.5">
                                  <label className="text-[8px] text-text-tertiary uppercase font-bold tracking-wider capitalize">{field.replace("_", " ")}</label>
                                  <input
                                    value={(st as any)[field]}
                                    onChange={e => updateTableCfg(st.table_name, field, e.target.value)}
                                    className="w-full bg-surface-card border border-border-custom rounded px-2 py-1 text-[10px] text-text-primary focus:outline-none focus:border-primary transition"
                                  />
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {selectedTables.length > 0 && (
                      <button
                        onClick={handleImportAll}
                        disabled={dbLoading}
                        className="w-full px-4 py-2.5 bg-success hover:bg-success/95 disabled:opacity-50 rounded-lg text-xs font-semibold text-white transition shadow-premium"
                      >
                        {dbLoading ? "Importing…" : `Import ${selectedTables.length} mapped table${selectedTables.length !== 1 ? "s" : ""}`}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>

      {/* Chart Zoom Modal */}
      {zoomedChart && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-6"
          onClick={() => setZoomedChart(null)}
        >
          <div
            className="bg-surface-card border border-border-custom rounded-card w-full max-w-4xl overflow-hidden shadow-premium flex flex-col"
            style={{ height: "75vh" }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-border-custom bg-surface shrink-0">
              <h3 className="font-semibold text-xs text-text-primary uppercase tracking-widest">{zoomedChart.title}</h3>
              <button
                onClick={() => setZoomedChart(null)}
                className="text-text-tertiary hover:text-text-primary transition p-1 rounded hover:bg-primary/10"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 p-6">
              {zoomedChart.render()}
            </div>
          </div>
        </div>
      )}

      {/* SHAP Explanation Modal */}
      {explainData && (() => {
        const maxShap = Math.max(
          0.001,
          ...explainData.positive_features.map(f => f.shap_value),
          ...explainData.negative_features.map(f => Math.abs(f.shap_value))
        );
        
        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <div className="bg-surface-card border border-border-custom rounded-card w-full max-w-md overflow-hidden shadow-premium">
              <div className="flex items-center justify-between p-4 border-b border-border-custom bg-surface">
                <h3 className="font-semibold text-xs text-text-primary uppercase tracking-widest">AI Explanation (SHAP Value)</h3>
                <button onClick={() => setExplainData(null)} className="text-text-tertiary hover:text-text-primary font-bold">×</button>
              </div>
              
              <div className="p-5 space-y-5">
                <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
                  <p className="text-[10px] text-primary font-semibold mb-1">Impact Summary</p>
                  <p className="text-xs text-text-secondary leading-relaxed">{explainData.summary}</p>
                </div>

                <div>
                  <p className="text-[10px] text-text-tertiary mb-3 uppercase tracking-wider font-bold">Features pushing order UP</p>
                  <div className="space-y-3">
                    {explainData.positive_features.length > 0 ? explainData.positive_features.map((f, i) => (
                      <div key={i} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-success font-medium">{f.feature_name}</span>
                        </div>
                        <div className="h-1 bg-surface rounded-full overflow-hidden">
                          <div className="h-full bg-success rounded-full" style={{ width: `${Math.max((f.shap_value / maxShap) * 100, 2)}%` }} />
                        </div>
                      </div>
                    )) : <p className="text-xs text-text-tertiary italic">No positive drivers.</p>}
                  </div>
                </div>

                <div>
                  <p className="text-[10px] text-text-tertiary mb-3 uppercase tracking-wider font-bold">Features pushing order DOWN</p>
                  <div className="space-y-3">
                    {explainData.negative_features.length > 0 ? explainData.negative_features.map((f, i) => (
                      <div key={i} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-danger font-medium">{f.feature_name}</span>
                        </div>
                        <div className="h-1 bg-surface rounded-full overflow-hidden">
                          <div className="h-full bg-danger rounded-full" style={{ width: `${Math.max((Math.abs(f.shap_value) / maxShap) * 100, 2)}%` }} />
                        </div>
                      </div>
                    )) : <p className="text-xs text-text-tertiary italic">No negative drivers.</p>}
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </DashboardLayout>
  );
}
