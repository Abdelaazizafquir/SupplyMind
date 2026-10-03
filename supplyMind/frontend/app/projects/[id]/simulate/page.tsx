"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, Simulation, SimulationCreate, KPIs, CustomDisruption } from "@/lib/api";
import { createSimulationSocket } from "@/lib/api";
import ProjectNav from "@/components/ProjectNav";
import DashboardLayout from "@/components/DashboardLayout";
import { Play, Activity, Settings, RefreshCw, BarChart2, ShieldAlert, Cpu } from "lucide-react";

const ALGORITHMS = [
  { value: "all",        label: "All Algorithms", desc: "Compare all policies simultaneously", badge: "📊 Compare" },
  { value: "mappo_gnn",  label: "MAPPO + GNN",   desc: "Multi-agent RL over graph topology", badge: "⭐ Recommended" },
  { value: "mappo",      label: "MAPPO",          desc: "Independent multi-agent RL",     badge: "" },
  { value: "base_stock", label: "Base-Stock",     desc: "Classical inventory policy",     badge: "" },
  { value: "ss_policy",  label: "(s,S) Policy",   desc: "Min/Max classical policy",       badge: "" },
];

const SCENARIOS = [
  { value: "normal",          label: "Normal operations", desc: "No disruptions", icon: "🟢" },
  { value: "demand_spike",    label: "Demand spike",       desc: "+200% demand for 3 weeks", icon: "📈" },
  { value: "lead_time_delay", label: "Lead-time delay",    desc: "Supplier delays 3×",        icon: "⏱️" },
  { value: "node_failure",    label: "Supplier failure",   desc: "Primary supplier offline",  icon: "🔴" },
  { value: "combined",        label: "Combined crisis",    desc: "Spike + failure simultaneously", icon: "⚠️" },
];

export default function SimulatePage() {
  const params = useParams();
  const projectId = params.id as string;
  const router = useRouter();

  const [algorithm, setAlgorithm] = useState<string>("mappo_gnn");
  const [scenario, setScenario]   = useState<string>("normal");
  const [episodes, setEpisodes]   = useState<number>(50);
  const [horizon, setHorizon]     = useState<number>(365);
  const [holdingMult, setHoldingMult]   = useState<number>(1.0);
  const [stockoutMult, setStockoutMult] = useState<number>(1.0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showArchitect, setShowArchitect] = useState(false);
  const [customDisruptions, setCustomDisruptions] = useState<CustomDisruption[]>([]);
  const [magnitude, setMagnitude] = useState<number>(2.0);
  const [targetNode, setTargetNode] = useState<string>("retailer_0");

  useEffect(() => {
    if (scenario === "demand_spike") { setMagnitude(2.0); setTargetNode("retailer_0"); }
    else if (scenario === "lead_time_delay") { setMagnitude(3.0); setTargetNode("supplier_0"); }
    else if (scenario === "node_failure") { setMagnitude(0.0); setTargetNode("supplier_0"); }
  }, [scenario]);
  const [running, setRunning]     = useState(false);
  const [progress, setProgress]   = useState(0);
  const [simId, setSimId]         = useState<string | null>(null);
  const [status, setStatus]       = useState<string | null>(null);
  const [kpis, setKpis]           = useState<KPIs | null>(null);
  const [logs, setLogs]           = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Simulation history
  const [history, setHistory] = useState<Simulation[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  // Load simulation history
  useEffect(() => {
    api.simulations.listByProject(projectId)
      .then(setHistory)
      .catch(console.error)
      .finally(() => setHistoryLoading(false));
  }, [projectId]);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev.slice(-30), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const launchSimulation = async () => {
    setRunning(true);
    setProgress(0);
    setKpis(null);
    setStatus("pending");
    setLogs([]);
    addLog("Initializing agent training session...");

    const appliedDisruptions: CustomDisruption[] = [...customDisruptions];
    if (scenario !== "normal" && scenario !== "combined") {
      appliedDisruptions.push({
        node_id: targetNode,
        start_day: 20,
        end_day: 45,
        disruption_type: scenario === "demand_spike" ? 1 : scenario === "lead_time_delay" ? 2 : 3,
        magnitude: magnitude
      });
    }

    const body: SimulationCreate = {
      project_id: projectId,
      algorithm: algorithm as SimulationCreate["algorithm"],
      scenario: scenario as SimulationCreate["scenario"],
      episodes,
      horizon,
      holding_cost_mult: holdingMult,
      stockout_penalty_mult: stockoutMult,
      custom_disruptions: appliedDisruptions,
    };

    try {
      addLog(`Launching training run: ${algorithm} on scenario: ${scenario}`);
      const sim = await api.simulations.launch(body);
      setSimId(sim.id);
      addLog(`Created simulation session: ${sim.id}`);

      // Open WebSocket for live progress
      const ws = createSimulationSocket(sim.id, (msg) => {
        if (msg.type === "progress" && msg.progress !== undefined) {
          setProgress(msg.progress);
          addLog(`Episode ${Math.round(msg.progress * (episodes / 100))} / ${episodes} processed.`);
        }
        if (msg.type === "completed") {
          setProgress(100);
          setStatus("completed");
          setKpis(msg.kpis ?? null);
          setRunning(false);
          addLog("Convergence achieved. Storing neural model weights.");
          ws.close();
          // Auto-redirect to results
          setTimeout(() => {
            router.push(`/projects/${projectId}/results?sim=${sim.id}`);
          }, 1500);
        }
        if (msg.type === "failed") {
          setStatus("failed");
          setRunning(false);
          addLog("Simulation thread crashed due to local environment limits.");
          ws.close();
        }
      });
      wsRef.current = ws;

      // Poll as fallback (WebSocket may not connect in all environments)
      const poll = setInterval(async () => {
        try {
          const updated = await api.simulations.get(sim.id);
          if (updated.progress) {
            setProgress(updated.progress);
            addLog(`GNN epoch converged. Progress: ${Math.round(updated.progress)}%`);
          }
          if (updated.status === "completed") {
            setProgress(100);
            setStatus("completed");
            setKpis(updated.kpis ?? null);
            setRunning(false);
            clearInterval(poll);
          }
          if (updated.status === "failed") {
            setStatus("failed");
            setRunning(false);
            clearInterval(poll);
          }
        } catch {
          // ignore poll errors
        }
      }, 2500);
      setTimeout(() => clearInterval(poll), 300_000); // max 5 min
    } catch (err) {
      console.error(err);
      setRunning(false);
      setStatus("failed");
      addLog("Launch thread rejected connection.");
    }
  };

  const radius = 40;
  const strokeDash = 2 * Math.PI * radius;

  return (
    <DashboardLayout>
      <div className="h-full flex flex-col overflow-hidden">
        <ProjectNav projectId={projectId} step="simulate" />

        <div className="flex-1 overflow-auto p-8 select-none max-w-4xl mx-auto w-full">
          {running || status === "completed" || status === "failed" ? (
            /* Running/Completed Progress View */
            <div className="flex flex-col items-center justify-center py-10 space-y-6">
              <div className="w-full max-w-md bg-surface-card border border-border-custom rounded-card p-6 space-y-6 shadow-premium text-center">
                <h2 className="text-sm font-bold text-text-primary uppercase tracking-widest">
                  {running ? "AI Training Progress" : status === "completed" ? "Simulation Completed" : "Simulation Failed"}
                </h2>

                {/* Circular Progress SVG */}
                <div className="flex items-center justify-center relative">
                  <svg className="w-28 h-28 transform -rotate-90">
                    <circle
                      cx="56" cy="56" r={radius}
                      stroke="rgba(255,255,255,0.06)" strokeWidth="6"
                      fill="transparent"
                    />
                    <circle
                      cx="56" cy="56" r={radius}
                      stroke="#534AB7" strokeWidth="6"
                      fill="transparent"
                      strokeDasharray={strokeDash}
                      strokeDashoffset={strokeDash * (1 - progress / 100)}
                      className="transition-all duration-300"
                    />
                  </svg>
                  <span className="absolute text-sm font-bold font-mono text-text-primary">
                    {Math.round(progress)}%
                  </span>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-text-secondary">
                    {running ? `Running neural simulation on GPU...` : "Converged to optimal policy."}
                  </p>
                  <p className="text-[10px] text-text-tertiary">
                    {episodes} episodes · Horizon: {horizon} days
                  </p>
                </div>

                {kpis && (
                  <div className="grid grid-cols-2 gap-3 pt-2 text-left">
                    {[
                      { label: "Fill Rate", value: `${kpis.fill_rate.toFixed(1)}%` },
                      { label: "Total Cost", value: `$${kpis.total_cost.toLocaleString(undefined, {maximumFractionDigits: 0})}` },
                      { label: "Avg Lead Time", value: `${kpis.avg_lead_time?.toFixed(1)} days` },
                      { label: "Disruptions", value: String(kpis.disruptions_handled) },
                    ].map((k) => (
                      <div key={k.label} className="bg-surface border border-border-custom rounded-lg p-3">
                        <p className="text-[9px] text-text-tertiary uppercase font-semibold">{k.label}</p>
                        <p className="text-sm font-bold text-text-primary mt-0.5">{k.value}</p>
                      </div>
                    ))}
                  </div>
                )}

                {status === "completed" && simId && (
                  <button
                    onClick={() => router.push(`/projects/${projectId}/results?sim=${simId}`)}
                    className="w-full bg-primary hover:bg-primary/95 text-white font-semibold py-2.5 rounded-lg text-xs transition shadow-premium"
                  >
                    View report cards →
                  </button>
                )}

                {!running && (
                  <button
                    onClick={() => { setStatus(null); setProgress(0); setKpis(null); }}
                    className="w-full text-xs text-text-secondary hover:text-text-primary py-2 transition"
                  >
                    Run another model
                  </button>
                )}
              </div>

              {/* Scrolling monospace log console */}
              <div className="w-full max-w-md bg-surface-card border border-border-custom rounded-card p-4 space-y-2 font-mono text-[10px] text-text-secondary h-44 overflow-y-auto">
                <p className="text-text-tertiary border-b border-border-custom pb-1 select-none">SYSTEM CONSOLE LOGS</p>
                {logs.length === 0 && <p className="text-text-tertiary italic">Waiting for compiler logs...</p>}
                {logs.map((log, idx) => (
                  <p key={idx} className="whitespace-pre-wrap leading-relaxed">{log}</p>
                ))}
              </div>
            </div>
          ) : (
            /* Configure Simulation Page */
            <div className="space-y-8">
              <div>
                <h1 className="text-xl font-bold text-text-primary tracking-tight">Run GNN Optimization</h1>
                <p className="text-xs text-text-secondary mt-1">
                  Configure simulation episodes, algorithm echelons, and disruption scenarios.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                
                {/* Left side: Algorithms & Training Budget */}
                <div className="space-y-6">
                  <div>
                    <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest mb-3">Model Core</h2>
                    <div className="grid grid-cols-1 gap-2.5">
                      {ALGORITHMS.map((a) => (
                        <button
                          key={a.value}
                          onClick={() => setAlgorithm(a.value)}
                          className={`flex items-start justify-between px-4 py-3 rounded-lg border transition text-left ${
                            algorithm === a.value
                              ? "border-primary bg-primary/5 text-text-primary shadow-sm"
                              : "border-border-custom bg-surface-card text-text-secondary hover:border-text-secondary/20"
                          }`}
                        >
                          <div>
                            <p className="text-xs font-semibold">{a.label}</p>
                            <p className="text-[10px] text-text-secondary mt-0.5">{a.desc}</p>
                          </div>
                          {a.badge && (
                            <span className="text-[9px] bg-primary-soft text-primary font-bold px-2 py-0.5 rounded-full uppercase">
                              {a.badge}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Episodes Training Budget */}
                  <div>
                    <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest mb-3">
                      Training Budget: <span className="text-primary">{episodes} episodes</span>
                    </h2>
                    <div className="flex gap-2">
                      {[10, 25, 50, 100].map((n) => (
                        <button
                          key={n}
                          onClick={() => setEpisodes(n)}
                          className={`flex-1 py-2 rounded-lg border text-xs transition font-semibold ${
                            episodes === n
                              ? "border-primary bg-primary/5 text-primary"
                              : "border-border-custom bg-surface-card text-text-secondary hover:border-text-secondary/20"
                          }`}
                        >
                          {n === 10 ? "Fast (10)" : n === 100 ? "Thorough (100)" : String(n)}
                        </button>
                      ))}
                    </div>
                    <span className="text-[10px] text-text-tertiary mt-2 inline-block">Estimated runtime: ~{Math.round(episodes * 0.4)} seconds on GPU</span>
                  </div>
                </div>

                {/* Right side: Scenario Cards & Advanced Settings */}
                <div className="space-y-6">
                  <div>
                    <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest mb-3">Disruption Scenario</h2>
                    <div className="grid grid-cols-1 gap-2.5">
                      {SCENARIOS.map((s) => (
                        <button
                          key={s.value}
                          onClick={() => setScenario(s.value)}
                          className={`flex items-start gap-3 px-4 py-3 rounded-lg border transition text-left ${
                            scenario === s.value
                              ? "border-primary bg-primary/5 text-text-primary shadow-sm"
                              : "border-border-custom bg-surface-card text-text-secondary hover:border-text-secondary/20"
                          }`}
                        >
                          <span className="text-base">{s.icon}</span>
                          <div>
                            <p className="text-xs font-semibold">{s.label}</p>
                            <p className="text-[10px] text-text-secondary mt-0.5">{s.desc}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {scenario !== "normal" && scenario !== "combined" && (
                    <div className="border border-border-custom bg-surface-card rounded-card p-4 space-y-4 animate-in fade-in slide-in-from-top-2">
                      <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Customize Disruption</h3>
                      
                      {scenario !== "node_failure" && (
                        <div>
                          <div className="flex justify-between text-xs mb-1.5">
                            <span className="text-text-secondary">Severity / Magnitude</span>
                            <span className="text-text-primary font-bold">{magnitude.toFixed(1)}×</span>
                          </div>
                          <input
                            type="range" min={1.0} max={5.0} step={0.1}
                            value={magnitude}
                            onChange={(e) => setMagnitude(Number(e.target.value))}
                            className="w-full accent-primary"
                          />
                        </div>
                      )}

                      <div>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-text-secondary">Target Node</span>
                        </div>
                        <select
                          value={targetNode}
                          onChange={(e) => setTargetNode(e.target.value)}
                          className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-xs text-text-primary outline-none focus:border-primary transition"
                        >
                          <optgroup label="Retailers (Downstream)">
                            <option value="retailer_0">Retailer 0</option>
                            <option value="retailer_1">Retailer 1</option>
                            <option value="retailer_2">Retailer 2</option>
                          </optgroup>
                          <optgroup label="Distributors (Midstream)">
                            <option value="distributor_0">Distributor 0</option>
                            <option value="distributor_1">Distributor 1</option>
                          </optgroup>
                          <optgroup label="Suppliers (Upstream)">
                            <option value="supplier_0">Supplier 0</option>
                          </optgroup>
                        </select>
                      </div>
                    </div>
                  )}

                  {/* Collapsible Advanced parameters */}
                  <div className="border border-border-custom bg-surface-card rounded-card overflow-hidden">
                    <button
                      className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-text-primary hover:bg-primary/5 transition"
                      onClick={() => setShowAdvanced((v) => !v)}
                    >
                      <span className="flex items-center gap-1.5"><Settings size={14} className="text-primary" /> Advanced cost model overrides</span>
                      <span className="text-[10px] text-text-tertiary">{showAdvanced ? "Hide" : "Expand"}</span>
                    </button>
                    {showAdvanced && (
                      <div className="px-4 pb-4 pt-1 space-y-4 border-t border-border-custom">
                        <div>
                          <div className="flex justify-between text-xs mb-1.5">
                            <span className="text-text-secondary">Simulation Horizon</span>
                            <span className="text-text-primary font-bold">{horizon} days</span>
                          </div>
                          <input
                            type="range" min={30} max={365} step={5}
                            value={horizon}
                            onChange={(e) => setHorizon(Number(e.target.value))}
                            className="w-full accent-primary"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-xs mb-1.5">
                            <span className="text-text-secondary">Holding Cost Multiplier</span>
                            <span className="text-text-primary font-bold">{holdingMult.toFixed(1)}×</span>
                          </div>
                          <input
                            type="range" min={0.5} max={3} step={0.1}
                            value={holdingMult}
                            onChange={(e) => setHoldingMult(Number(e.target.value))}
                            className="w-full accent-primary"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-xs mb-1.5">
                            <span className="text-text-secondary">Shortage Penalty Multiplier</span>
                            <span className="text-text-primary font-bold">{stockoutMult.toFixed(1)}×</span>
                          </div>
                          <input
                            type="range" min={0.5} max={5} step={0.1}
                            value={stockoutMult}
                            onChange={(e) => setStockoutMult(Number(e.target.value))}
                            className="w-full accent-primary"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

              </div>

              <button
                onClick={launchSimulation}
                className="w-full bg-primary hover:bg-primary/95 text-white font-semibold py-3.5 rounded-lg text-sm transition shadow-premium flex items-center justify-center gap-1.5"
              >
                <Cpu size={16} /> Launch GNN Simulation
              </button>
            </div>
          )}

          {/* Simulation History */}
          {!historyLoading && history.length > 0 && !running && !status && (
            <div className="mt-12 space-y-4">
              <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest">
                Simulation History
              </h2>
              <div className="space-y-2">
                {history.slice(0, 8).map((sim) => (
                  <SimHistoryRow key={sim.id} sim={sim} projectId={projectId} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}

function SimHistoryRow({ sim, projectId }: { sim: Simulation; projectId: string }) {
  const STATUS_COLORS = {
    completed: "bg-success/10 text-success border border-success/20",
    running:   "bg-primary/10 text-primary border border-primary/20 animate-pulse",
    pending:   "bg-warning/10 text-warning border border-warning/20",
    failed:    "bg-danger/10 text-danger border border-danger/20",
  };
  const color = STATUS_COLORS[sim.status as keyof typeof STATUS_COLORS] ?? "bg-surface-card border border-border-custom text-text-tertiary";

  const ALGO_LABELS: Record<string, string> = {
    mappo_gnn:  "MAPPO+GNN",
    mappo:      "MAPPO",
    base_stock: "Base-Stock",
    ss_policy:  "(s,S)",
  };

  const dateStr = sim.completed_at
    ? new Date(sim.completed_at).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : new Date(sim.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" });

  return (
    <div className="flex items-center gap-4 bg-surface-card border border-border-custom hover:border-primary/20 rounded-lg px-5 py-3 transition duration-150">
      {/* Status badge */}
      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase shrink-0 ${color}`}>
        {sim.status}
      </span>

      {/* Algo + Scenario */}
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <span className="text-xs font-bold text-text-primary shrink-0">{ALGO_LABELS[sim.algorithm] ?? sim.algorithm}</span>
        <span className="text-text-tertiary text-xs select-none">·</span>
        <span className="text-xs text-text-secondary truncate capitalize">{sim.scenario.replace("_", " ")}</span>
        <span className="text-text-tertiary text-xs select-none">·</span>
        <span className="text-[10px] text-text-tertiary font-mono">{sim.episodes} episodes</span>
      </div>

      {/* KPIs preview */}
      {sim.kpis && sim.status === "completed" && (
        <div className="hidden sm:flex items-center gap-4 text-[10px] text-text-secondary shrink-0 font-medium">
          <span>Fill Rate: <span className={sim.kpis.fill_rate >= 95 ? "text-success" : "text-warning"}>{sim.kpis.fill_rate.toFixed(1)}%</span></span>
          <span>Cost: <span className="text-text-primary">${sim.kpis.total_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span></span>
        </div>
      )}

      {/* Date */}
      <span className="text-[10px] text-text-tertiary shrink-0 hidden md:block">{dateStr}</span>

      {/* View button */}
      {sim.status === "completed" && (
        <Link
          href={`/projects/${projectId}/results?sim=${sim.id}`}
          className="text-xs text-primary hover:underline font-semibold shrink-0"
        >
          View report →
        </Link>
      )}
    </div>
  );
}
