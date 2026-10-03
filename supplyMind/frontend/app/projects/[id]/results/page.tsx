"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, SimulationDetail, RecommendationCard, WhatIfResult, ExplainResponse, NodeInventoryInput } from "@/lib/api";
import ProjectNav from "@/components/ProjectNav";
import DashboardLayout from "@/components/DashboardLayout";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from "recharts";
import { FileText, Share2, RefreshCw, Send, HelpCircle, Check, AlertTriangle, ArrowRight, Maximize2, X } from "lucide-react";

const CONFIDENCE_COLORS = {
  High:   "bg-success/10 text-success border-success/25",
  Medium: "bg-warning/10 text-warning border-warning/25",
  Low:    "bg-danger/10 text-danger border-danger/25",
};

const URGENCY_COLORS = {
  Urgent:  "bg-danger/15 text-danger",
  Normal:  "bg-primary/15 text-primary",
  Monitor: "bg-surface border border-border-custom text-text-secondary",
};

interface ChatMsg { role: "user" | "assistant"; content: string; }

export default function ResultsPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const projectId = params.id as string;
  const simId = searchParams.get("sim");
  const router = useRouter();

  const [sim, setSim]           = useState<SimulationDetail | null>(null);
  const [loading, setLoading]   = useState(true);
  const [whatIf, setWhatIf]     = useState<WhatIfResult | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "agents" | "timeline" | "whatif" | "chat">("overview");

  // Explain
  const [explainLoading, setExplainLoading] = useState<string | null>(null);
  const [explainData, setExplainData]       = useState<ExplainResponse | null>(null);

  // Live AI analysis
  const [aiAnalysis, setAiAnalysis]     = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  // Chat
  const [chatHistory, setChatHistory]   = useState<ChatMsg[]>([]);
  const [chatInput, setChatInput]       = useState("");
  const [chatLoading, setChatLoading]   = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [zoomedContent, setZoomedContent] = useState<{title: string, type: "kpi" | "chart", render: () => React.ReactNode} | null>(null);

  useEffect(() => {
    if (!simId) { setLoading(false); return; }
    api.results.get(simId)
      .then((data) => {
        setSim(data);
        // Auto-fetch live Groq analysis after data loads
        setAnalysisLoading(true);
        api.results.analyze(simId)
          .then((r) => setAiAnalysis(r.analysis))
          .catch(() => setAiAnalysis(null))
          .finally(() => setAnalysisLoading(false));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [simId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleExplain = async (agentId: string) => {
    if (!projectId || !sim?.llm_summary?.worst_day_state) return;
    setExplainLoading(agentId);
    try {
      const res = await api.liveControl.explain({
        project_id: projectId,
        nodes: sim.llm_summary.worst_day_state as NodeInventoryInput[],
        target_node_id: agentId,
      });
      setExplainData(res);
    } catch (err) {
      console.error(err);
      alert("Failed to explain recommendation.");
    } finally {
      setExplainLoading(null);
    }
  };

  const sendChat = async () => {
    if (!chatInput.trim() || !simId || chatLoading) return;
    const userMsg: ChatMsg = { role: "user", content: chatInput.trim() };
    const newHistory = [...chatHistory, userMsg];
    setChatHistory(newHistory);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await api.results.chat(simId, userMsg.content, chatHistory);
      setChatHistory([...newHistory, { role: "assistant", content: res.reply }]);
    } catch {
      setChatHistory([...newHistory, { role: "assistant", content: "⚠️ Could not reach AI. Check your GROQ_API_KEY in .env." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const runWhatIf = async (scenarioType: string) => {
    if (!simId) return;
    setWhatIfLoading(true);
    setWhatIf(null);
    try {
      const result = await api.simulations.whatIf(simId, scenarioType);
      setWhatIf(result);
      setActiveTab("whatif");
    } catch (err) {
      console.error(err);
    } finally {
      setWhatIfLoading(false);
    }
  };

  const downloadCSV = async () => {
    if (!simId) return;
    const res = await api.results.exportCsv(simId);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `results_${simId.slice(0,8)}.csv`; a.click();
  };

  const downloadPDF = async () => {
    if (!simId) return;
    const res = await api.results.exportPdf(simId);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `report_${simId.slice(0,8)}.pdf`; a.click();
  };

  if (loading) return (
    <div className="min-h-screen bg-surface flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-xs text-text-secondary font-medium">Assembling simulation metrics…</p>
      </div>
    </div>
  );

  if (!sim && !simId) return (
    <DashboardLayout>
      <ProjectNav projectId={projectId} step="results" />
      <div className="max-w-md mx-auto px-8 py-20 text-center space-y-4 select-none">
        <div className="text-4xl">📊</div>
        <h2 className="text-sm font-bold text-text-primary uppercase tracking-widest">No results yet</h2>
        <p className="text-xs text-text-secondary">Run a simulation first to generate reports.</p>
        <Link
          href={`/projects/${projectId}/simulate`}
          className="bg-primary hover:bg-primary/95 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition inline-block shadow-premium"
        >
          Go to Simulation
        </Link>
      </div>
    </DashboardLayout>
  );

  const kpis = sim?.kpis;
  const llm  = sim?.llm_summary;
  const recs: RecommendationCard[] = llm?.recommendations ?? [];

  return (
    <DashboardLayout>
      <div className="h-full flex flex-col overflow-hidden">
        <ProjectNav projectId={projectId} step="results" />

        <div className="flex-1 overflow-auto p-8 select-none space-y-6">
          
          {/* Header Area */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h1 className="text-xl font-bold text-text-primary tracking-tight">Simulation Results</h1>
              {kpis && (
                <div className="flex gap-2 items-center mt-1">
                  <span className="text-[10px] bg-primary/10 text-primary font-bold px-2 py-0.5 rounded-full uppercase">
                    {kpis.algorithm?.toUpperCase()}
                  </span>
                  <span className="text-[10px] bg-surface-card border border-border-custom text-text-secondary font-medium px-2 py-0.5 rounded-full capitalize">
                    {kpis.scenario?.replace("_", " ")} scenario
                  </span>
                </div>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={downloadCSV}
                className="bg-surface-card border border-border-custom hover:border-text-secondary/30 text-text-secondary hover:text-text-primary text-xs font-semibold px-3 py-2 rounded-lg transition shadow-premium"
              >
                Export CSV
              </button>
              <button
                onClick={downloadPDF}
                className="bg-surface-card border border-border-custom hover:border-text-secondary/30 text-text-secondary hover:text-text-primary text-xs font-semibold px-3 py-2 rounded-lg transition shadow-premium flex items-center gap-1"
              >
                <FileText size={14} /> Export PDF
              </button>
              <Link
                href={`/projects/${projectId}/simulate`}
                className="bg-primary hover:bg-primary/95 text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow-premium flex items-center gap-1"
              >
                <RefreshCw size={12} /> Run again
              </Link>
            </div>
          </div>

          {/* KPI Strip */}
          {kpis && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: "Customer Fill Rate", value: `${kpis.fill_rate.toFixed(1)}%`, good: kpis.fill_rate >= 95 },
                { label: "Total Chain Cost", value: `$${kpis.total_cost.toLocaleString(undefined, {maximumFractionDigits: 0})}`, good: null },
                { label: "Average Lead Time", value: `${kpis.avg_lead_time?.toFixed(1)} days`, good: (kpis.avg_lead_time ?? 0) < 4.0 },
                { label: "Disruptions", value: String(kpis.disruptions_handled), good: null },
                ...(kpis.inventory_turnover !== undefined ? [
                  { label: "Inventory Turnover", value: `${kpis.inventory_turnover.toFixed(1)}×`, good: kpis.inventory_turnover >= 4.0 },
                  { label: "Shortage Days", value: String(kpis.shortage_days), good: (kpis.shortage_days ?? 0) === 0 },
                  { label: "Holding / Transport", value: `$${(kpis.holding_cost ?? 0).toLocaleString()} / $${(kpis.transport_cost ?? 0).toLocaleString()}`, good: null },
                  { label: "AI Savings", value: `$${kpis.ai_savings?.toLocaleString()}`, good: (kpis.ai_savings ?? 0) > 0 }
                ] : [])
              ].map((k) => (
                <div key={k.label} className="group relative bg-surface-card border border-border-custom rounded-card p-5 flex flex-col justify-between h-[110px] hover:border-primary/20 transition-all duration-200">
                  <div className="flex justify-between items-start">
                    <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">{k.label}</p>
                    <button
                      onClick={() => setZoomedContent({ title: k.label, type: "kpi", render: () => (
                        <div className="flex flex-col items-center justify-center space-y-6">
                          <h3 className="text-[8rem] font-bold text-text-primary tracking-tight leading-none">{k.value}</h3>
                          {k.good !== null && (
                            <span className={`text-4xl font-bold ${k.good ? "text-success" : "text-warning"}`}>
                              {k.good ? "Optimal" : "Monitor"}
                            </span>
                          )}
                        </div>
                      )})}
                      className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5"
                    >
                      <Maximize2 size={12} />
                    </button>
                  </div>
                  <h3 className="text-xl font-bold text-text-primary mt-1 tracking-tight">{k.value}</h3>
                  {k.good !== null && (
                    <span className={`text-[9px] font-bold ${k.good ? "text-success" : "text-warning"}`}>
                      {k.good ? "Optimal" : "Monitor"}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Tabs Nav */}
          <div className="flex gap-1 bg-surface-card border border-border-custom rounded-lg p-1 w-fit">
            {(["overview", "agents", "timeline", "whatif", "chat"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition capitalize ${
                  activeTab === tab ? "bg-primary text-white" : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {tab === "whatif" ? "What-If Sandbox" : tab === "chat" ? "🤖 Ask AI" : tab}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === "overview" && (
            <div className="space-y-6">
              
              {/* Comparison table */}
              {llm?.comparison && (() => {
                const algos = ["mappo_gnn", "base_stock", "ss_policy"] as const;
                const labels: Record<string, string> = {
                  mappo_gnn: "MAPPO + GNN",
                  base_stock: "Base-Stock",
                  ss_policy: "(s,S) Policy",
                };
                const chartData = algos.map((a) => ({
                  name: labels[a],
                  fill_rate: parseFloat(((llm.comparison as any)[a]?.kpis?.fill_rate ?? 0).toFixed(1)),
                  total_cost: parseFloat(((llm.comparison as any)[a]?.kpis?.total_cost ?? 0).toFixed(2)),
                  avg_lead_time: parseFloat(((llm.comparison as any)[a]?.kpis?.avg_lead_time ?? 0).toFixed(1)),
                }));

                const ALGO_COLORS = ["#534AB7", "#1D9E75", "#EF9F27"];

                return (
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {/* Fill rate chart */}
                    <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-4 flex flex-col">
                      <div className="flex justify-between items-start">
                        <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Fill Rate (%)</p>
                        <button
                          onClick={() => setZoomedContent({ title: "Fill Rate (%)", type: "chart", render: () => (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <YAxis domain={[0, 100]} tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" tickFormatter={(v) => `${v}%`} />
                                <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`${v}%`, "Fill Rate"]} />
                                <Bar dataKey="fill_rate" radius={[8, 8, 0, 0]}>
                                  {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )})}
                          className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5 z-10"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={100}>
                        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" tickFormatter={(v) => `${v}%`} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`${v}%`, "Fill Rate"]} />
                          <Bar dataKey="fill_rate" radius={[4, 4, 0, 0]}>
                            {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Cost chart */}
                    <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-4 flex flex-col">
                      <div className="flex justify-between items-start">
                        <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Total Cost ($)</p>
                        <button
                          onClick={() => setZoomedContent({ title: "Total Cost ($)", type: "chart", render: () => (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                                <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`$${v.toLocaleString()}`, "Total Cost"]} />
                                <Bar dataKey="total_cost" radius={[8, 8, 0, 0]}>
                                  {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )})}
                          className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5 z-10"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={100}>
                        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`$${v.toLocaleString()}`, "Total Cost"]} />
                          <Bar dataKey="total_cost" radius={[4, 4, 0, 0]}>
                            {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Average Lead Time */}
                    <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-4 flex flex-col">
                      <div className="flex justify-between items-start">
                        <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Avg Lead Time (Days)</p>
                        <button
                          onClick={() => setZoomedContent({ title: "Avg Lead Time (Days)", type: "chart", render: () => (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 14 }} stroke="var(--border-custom)" />
                                <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`${v} days`, "Avg Lead Time"]} />
                                <Bar dataKey="avg_lead_time" radius={[8, 8, 0, 0]}>
                                  {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          )})}
                          className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5 z-10"
                        >
                          <Maximize2 size={12} />
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={100}>
                        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                          <XAxis dataKey="name" tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <YAxis tick={{ fill: "var(--text-secondary)", fontSize: 9 }} stroke="var(--border-custom)" />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} formatter={(v) => [`${v} days`, "Avg Lead Time"]} />
                          <Bar dataKey="avg_lead_time" radius={[4, 4, 0, 0]}>
                            {chartData.map((_, idx) => <Cell key={idx} fill={ALGO_COLORS[idx]} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })()}

              {/* Insights Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
                {/* Left: LLM analysis */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="bg-surface-card border border-border-custom rounded-card p-6 space-y-4">
                    <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">AI Performance Narrative</h3>
                    {analysisLoading ? (
                      <div className="flex items-center gap-2 text-xs text-text-secondary">
                        <RefreshCw size={14} className="animate-spin text-primary" />
                        <span>Synthesizing multi-echelon behaviors…</span>
                      </div>
                    ) : aiAnalysis ? (
                      <p className="text-xs text-text-secondary leading-relaxed whitespace-pre-wrap">{aiAnalysis}</p>
                    ) : (
                      <p className="text-xs text-text-tertiary italic">Executive summary unavailable.</p>
                    )}
                  </div>
                </div>

                {/* Right: Recommendations */}
                <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
                  <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Recommended Actions</h3>
                  {recs.length === 0 ? (
                    <p className="text-xs text-text-tertiary italic">No recommendations produced.</p>
                  ) : (
                    <div className="space-y-3">
                      {recs.map((r, idx) => (
                        <RecCard
                          key={idx}
                          rec={r}
                          onExplain={handleExplain}
                          explainLoading={explainLoading}
                          canExplain={!!sim?.llm_summary?.worst_day_state}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === "timeline" && sim?.timeline && (() => {
            const timelineData = sim.timeline.map((t) => {
              const totalInventory = Object.values(t.inventories || {}).reduce((a, b) => a + Number(b), 0);
              const stockouts = Object.values(t.inventories || {}).filter((v) => Number(v) <= 0).length;
              return {
                day: t.step,
                inventory: totalInventory,
                stockout: stockouts,
                cost: t.cost,
              };
            });

            return (
              <div className="group relative bg-surface-card border border-border-custom rounded-card p-5 space-y-4 flex flex-col">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-sm font-semibold text-text-primary">Operational Timeline Chart</h3>
                    <p className="text-xs text-text-secondary mt-0.5">Inventory levels and stockouts over the full horizon</p>
                  </div>
                  <button
                    onClick={() => setZoomedContent({ title: "Operational Timeline Chart", type: "chart", render: () => (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={timelineData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                          <XAxis dataKey="day" stroke="#5F5E5A" fontSize={14} tickLine={false} />
                          <YAxis stroke="#5F5E5A" fontSize={14} tickLine={false} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} />
                          <Legend wrapperStyle={{ fontSize: 14, paddingTop: 20 }} />
                          <Line type="monotone" dataKey="inventory" stroke="#534AB7" strokeWidth={3} dot={false} name="Total Inventory" />
                          <Line type="monotone" dataKey="stockout" stroke="#D85A30" strokeWidth={3} dot={false} name="Stockout Nodes" />
                        </LineChart>
                      </ResponsiveContainer>
                    )})}
                    className="opacity-0 group-hover:opacity-100 transition-opacity bg-surface border border-border-custom hover:border-primary text-text-secondary hover:text-primary rounded-md p-1.5 z-10"
                  >
                    <Maximize2 size={12} />
                  </button>
                </div>
                <div className="flex-1 min-h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timelineData}>
                      <XAxis dataKey="day" stroke="#5F5E5A" fontSize={9} tickLine={false} />
                      <YAxis stroke="#5F5E5A" fontSize={9} tickLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: "var(--surface-card)", border: "1px solid var(--border-custom)", borderRadius: "8px", color: "var(--text-primary)" }} itemStyle={{ color: "var(--text-secondary)" }} labelStyle={{ color: "var(--text-primary)", fontWeight: "bold", marginBottom: "4px" }} />
                      <Line type="monotone" dataKey="inventory" stroke="#534AB7" strokeWidth={1.5} dot={false} name="Total Inventory" />
                      <Line type="monotone" dataKey="stockout" stroke="#D85A30" strokeWidth={1.5} dot={false} name="Stockout Nodes" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })()}

          {activeTab === "agents" && sim?.agent_logs && (
            <div className="bg-surface-card border border-border-custom rounded-card overflow-hidden">
              <div className="p-5 border-b border-border-custom">
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Per-Agent Metrics</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-border-custom bg-surface text-text-secondary font-semibold">
                      <th className="p-4">Agent ID</th>
                      <th className="p-4">Name</th>
                      <th className="p-4">Mean Inventory</th>
                      <th className="p-4">Fill Rate</th>
                      <th className="p-4">Avg Cost/Step</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sim.agent_logs.map((log) => (
                      <tr key={log.agent_id} className="border-b border-border-custom hover:bg-primary/5 transition">
                        <td className="p-4 font-mono font-bold text-text-primary">{log.agent_id}</td>
                        <td className="p-4 text-text-primary">{log.agent_name}</td>
                        <td className="p-4 text-text-secondary">{log.avg_inventory?.toFixed(1)} u</td>
                        <td className="p-4 text-text-primary font-semibold">{log.fill_rate?.toFixed(1)}%</td>
                        <td className="p-4 text-danger font-medium">${log.avg_cost_per_step?.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          )}

          {activeTab === "whatif" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
              {/* Controls */}
              <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Disruption Scenarios</h3>
                <p className="text-xs text-text-secondary">Simulate alternative stress tests against the trained policy.</p>
                <div className="flex flex-col gap-2">
                  {[
                    { label: "Demand Spike (+150%)", val: "demand_spike" },
                    { label: "Supplier Failure (15d)", val: "node_failure" },
                    { label: "Combined Stress Run", val: "combined" }
                  ].map((sc) => (
                    <button
                      key={sc.val}
                      onClick={() => runWhatIf(sc.val)}
                      disabled={whatIfLoading}
                      className="w-full text-left px-4 py-3 bg-surface border border-border-custom rounded-lg text-xs font-semibold text-text-primary hover:border-primary/30 transition flex items-center justify-between"
                    >
                      <span>{sc.label}</span>
                      <ArrowRight size={12} className="text-text-tertiary" />
                    </button>
                  ))}
                </div>
              </div>

              {/* Sandbox results */}
              <div className="md:col-span-2 space-y-4">
                {whatIfLoading && (
                  <div className="bg-surface-card border border-border-custom rounded-card py-20 text-center text-xs text-text-secondary flex items-center justify-center gap-2">
                    <RefreshCw size={14} className="animate-spin text-primary" />
                    Running Monte Carlo iterations…
                  </div>
                )}

                {whatIf && !whatIfLoading && (
                  <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-6">
                    <div>
                      <h3 className="text-sm font-semibold text-text-primary">Stress Test Sandbox Results</h3>
                      <p className="text-xs text-text-secondary mt-0.5">Policy resilience metrics compared to baseline</p>
                    </div>

                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-surface border border-border-custom rounded-lg p-3">
                        <p className="text-[9px] text-text-tertiary uppercase font-semibold">New Fill Rate</p>
                        <p className={`text-base font-bold mt-1 ${whatIf.new_fill_rate >= 90 ? "text-success" : "text-warning"}`}>
                          {whatIf.new_fill_rate.toFixed(1)}%
                        </p>
                      </div>
                      <div className="bg-surface border border-border-custom rounded-lg p-3">
                        <p className="text-[9px] text-text-tertiary uppercase font-semibold">New Cost</p>
                        <p className="text-base font-bold text-text-primary mt-1">
                          ${whatIf.new_cost.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </p>
                      </div>
                      <div className="bg-surface border border-border-custom rounded-lg p-3">
                        <p className="text-[9px] text-text-tertiary uppercase font-semibold">Cost Increase</p>
                        <p className="text-base font-bold text-danger mt-1">
                          +{((whatIf.new_cost - (kpis?.total_cost ?? whatIf.new_cost)) / (kpis?.total_cost ?? 1) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 space-y-1.5">
                      <p className="text-xs font-semibold text-primary uppercase tracking-wider">AI Impact Forecast</p>
                      <p className="text-xs text-text-secondary leading-relaxed">{whatIf.impact_summary}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "chat" && (
            <div className="max-w-xl mx-auto bg-surface-card border border-border-custom rounded-card overflow-hidden h-[420px] flex flex-col justify-between shadow-premium">
              <div className="p-4 border-b border-border-custom bg-surface flex items-center justify-between">
                <span className="text-xs font-bold text-text-primary uppercase tracking-widest">Ask SupplyMind Coordinator</span>
                <span className="text-[10px] text-text-tertiary">GPT-4o Engine</span>
              </div>

              {/* Chat feed */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 font-sans text-xs">
                {chatHistory.length === 0 && (
                  <div className="text-center text-text-tertiary py-12 space-y-2">
                    <HelpCircle size={24} className="mx-auto opacity-50" />
                    <p className="font-semibold text-text-primary">Interactive sandbox chat</p>
                    <p className="max-w-xs mx-auto text-[11px]">Ask questions like: "What triggered the stockout on day 40?" or "How can we reduce Supplier A delay impacts?"</p>
                  </div>
                )}
                {chatHistory.map((m, idx) => (
                  <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[80%] rounded-lg px-4 py-2.5 leading-relaxed ${
                      m.role === "user" 
                        ? "bg-primary text-white" 
                        : "bg-surface border border-border-custom text-text-primary"
                    }`}>
                      {m.content}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-surface border border-border-custom rounded-lg px-4 py-2.5 text-text-tertiary flex items-center gap-1.5">
                      <RefreshCw size={12} className="animate-spin text-primary" /> Thinking…
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input */}
              <div className="p-4 border-t border-border-custom bg-surface flex gap-2">
                <input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") sendChat(); }}
                  placeholder="Ask a question about the GNN optimization policies..."
                  className="flex-1 bg-surface-card border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-primary transition"
                />
                <button
                  onClick={sendChat}
                  disabled={chatLoading || !chatInput.trim()}
                  className="bg-primary hover:bg-primary/95 text-white font-semibold p-2 rounded-lg transition disabled:opacity-50"
                >
                  <Send size={14} />
                </button>
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
                    <h3 className="font-semibold text-xs text-text-primary uppercase tracking-widest">AI Reasoning (SHAP Analysis)</h3>
                    <button onClick={() => setExplainData(null)} className="text-text-tertiary hover:text-text-primary font-bold">×</button>
                  </div>
                  
                  <div className="p-5 space-y-5">
                    <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
                      <p className="text-[10px] text-primary font-semibold mb-1">Impact Summary</p>
                      <p className="text-xs text-text-secondary leading-relaxed">{explainData.summary}</p>
                    </div>

                    <div>
                      <p className="text-[10px] text-text-tertiary mb-3 uppercase tracking-wider font-bold">Drivers for larger replenishment</p>
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
                      <p className="text-[10px] text-text-tertiary mb-3 uppercase tracking-wider font-bold">Drivers for smaller replenishment</p>
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

        </div>
      </div>
      {/* Zoom Modal */}
      {zoomedContent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8" onClick={(e) => { e.stopPropagation(); setZoomedContent(null); }}>
          <div className="bg-surface-card border border-border-custom rounded-card w-full max-w-5xl h-[80vh] flex flex-col overflow-hidden shadow-2xl relative" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-6 border-b border-border-custom bg-surface/50">
              <h3 className="font-bold text-lg text-text-primary uppercase tracking-widest">{zoomedContent.title}</h3>
              <button onClick={() => setZoomedContent(null)} className="text-text-tertiary hover:text-text-primary font-bold p-2 bg-surface border border-border-custom rounded-lg transition-colors">
                <X size={20} />
              </button>
            </div>
            <div className="flex-1 p-8 min-h-0 bg-surface-card/50 w-full h-full">
              {zoomedContent.type === "kpi" ? (
                <div className="w-full h-full flex items-center justify-center">
                  {zoomedContent.render()}
                </div>
              ) : (
                zoomedContent.render()
              )}
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}

function RecCard({ rec, onExplain, explainLoading, canExplain }: { rec: RecommendationCard; onExplain: (id: string) => void; explainLoading: string | null; canExplain: boolean }) {
  const conf = CONFIDENCE_COLORS[rec.confidence] ?? CONFIDENCE_COLORS.Medium;
  const urg  = URGENCY_COLORS[rec.urgency] ?? URGENCY_COLORS.Normal;

  return (
    <div className="bg-surface border border-border-custom rounded-lg p-4 space-y-2 hover:border-primary/10 transition duration-150">
      <div className="flex items-center justify-between">
        <p className="font-semibold text-xs text-text-primary">{rec.agent_name}</p>
        <div className="flex gap-1.5">
          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${conf}`}>{rec.confidence}</span>
          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full uppercase ${urg}`}>{rec.urgency}</span>
        </div>
      </div>
      <p className="text-xs font-semibold text-primary">{rec.action}</p>
      <p className="text-[11px] text-text-secondary leading-relaxed">{rec.reason}</p>
      <div className="flex justify-between items-center mt-3 pt-3 border-t border-border-custom">
        <div className="flex gap-1.5">
          <span className="text-[9px] bg-success/10 border border-success/20 text-success px-1.5 py-0.5 rounded">
            Cost: ${rec.cost_estimate.toLocaleString(undefined, {minimumFractionDigits: 0})}
          </span>
          <span className="text-[9px] bg-primary/10 border border-primary/20 text-primary px-1.5 py-0.5 rounded">
            ETA: {rec.lead_time_days ?? 1}d
          </span>
        </div>
        <div>
          {canExplain && (
            <button
              onClick={() => onExplain(rec.agent_id)}
              disabled={explainLoading === rec.agent_id}
              className="text-[9px] uppercase font-bold text-primary hover:underline disabled:opacity-50 transition"
            >
              {explainLoading === rec.agent_id ? "Explaining..." : "Explain"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
