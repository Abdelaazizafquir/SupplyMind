"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, Project, Simulation } from "@/lib/api";
import DashboardLayout from "@/components/DashboardLayout";
import {
  TrendingDown,
  Activity,
  Plus,
  ArrowUpRight,
  Layers,
  Zap,
  Database,
  Clock,
  Trash2
} from "lucide-react";

export default function DashboardPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [projects, setProjects]         = useState<Project[]>([]);
  const [fetching, setFetching]         = useState(true);
  const [creating, setCreating]         = useState(false);
  const [newName, setNewName]           = useState("");
  const [newMode, setNewMode]           = useState("simulation");
  const [showForm, setShowForm]         = useState(false);
  const [latestSims, setLatestSims]     = useState<Simulation[]>([]);
  const [simsLoading, setSimsLoading]   = useState(true);

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    if (user) {
      api.projects.list()
        .then(async (ps) => {
          setProjects(ps);
          // Load simulations for all projects to compute real KPIs
          const allSims: Simulation[] = [];
          await Promise.all(
            ps.map((p) =>
              api.simulations.listByProject(p.id)
                .then((sims) => allSims.push(...sims))
                .catch(() => {})
            )
          );
          // Sort by most recent
          allSims.sort(
            (a, b) =>
              new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          );
          setLatestSims(allSims);
        })
        .catch(console.error)
        .finally(() => { setFetching(false); setSimsLoading(false); });
    }
  }, [user]);

  const createProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const p = await api.projects.create(newName.trim(), newMode);
      setProjects([p, ...projects]);
      setNewName("");
      setNewMode("simulation");
      setShowForm(false);
      router.push(newMode === "live" ? `/projects/${p.id}/live` : `/projects/${p.id}/upload`);
    } catch (err) {
      console.error(err);
      alert("Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async (id: string) => {
    if (!confirm("Are you sure you want to delete this project? This action cannot be undone.")) return;
    try {
      await api.projects.delete(id);
      setProjects((prev) => prev.filter((p) => p.id !== id));
      setLatestSims((prev) => prev.filter((s) => s.project_id !== id));
    } catch (err) {
      console.error(err);
      alert("Failed to delete project");
    }
  };

  // Compute real KPIs from completed simulations
  const completedSims = latestSims.filter((s) => s.status === "completed");
  const runningSims   = latestSims.filter((s) => s.status === "running");
  const totalProjects = projects.length;
  const liveProjects  = projects.filter((p) => p.mode === "live").length;
  const simProjects   = projects.filter((p) => p.mode !== "live").length;

  if (loading || !user) return null;

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-8 py-8 space-y-8 select-none">

        {/* Hero Section */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-text-primary">
              Good morning, {user.name || user.email.split("@")[0]}.
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
              </span>
              <p className="text-xs text-text-secondary">
                {runningSims.length > 0
                  ? `${runningSims.length} simulation${runningSims.length > 1 ? "s" : ""} running · ${totalProjects} project${totalProjects !== 1 ? "s" : ""} active`
                  : totalProjects > 0
                  ? `${totalProjects} project${totalProjects !== 1 ? "s" : ""} · ${completedSims.length} simulation${completedSims.length !== 1 ? "s" : ""} completed`
                  : "No projects yet — create your first study below."}
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowForm(true)}
            className="bg-primary hover:bg-primary/95 text-white font-semibold px-4 py-2 rounded-lg transition text-xs shadow-premium flex items-center gap-1.5 self-start sm:self-auto"
          >
            <Plus size={14} /> New project
          </button>
        </div>

        {/* New Project Form */}
        {showForm && (
          <form
            onSubmit={createProject}
            className="bg-surface-card border border-primary/20 rounded-card p-5 space-y-4 shadow-premium"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-text-primary uppercase tracking-wider">Create New Study</span>
              <button type="button" onClick={() => setShowForm(false)} className="text-text-tertiary hover:text-text-primary text-xs">×</button>
            </div>

            <input
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Project name (e.g. Tomato Supply Chain Q2)"
              className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-primary transition"
            />

            <div className="flex items-center gap-6">
              <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-text-secondary">
                <input
                  type="radio"
                  name="mode"
                  value="simulation"
                  checked={newMode === "simulation"}
                  onChange={(e) => setNewMode(e.target.value)}
                  className="accent-primary"
                />
                Simulation Project
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-text-secondary">
                <input
                  type="radio"
                  name="mode"
                  value="live"
                  checked={newMode === "live"}
                  onChange={(e) => setNewMode(e.target.value)}
                  className="accent-primary"
                />
                Live Database Control Project
              </label>
            </div>

            <div className="flex gap-2 justify-end">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-3 py-1.5 border border-border-custom rounded-lg text-xs text-text-secondary hover:text-text-primary transition"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={creating || !newName.trim()}
                className="bg-primary hover:bg-primary/95 disabled:opacity-50 text-white font-medium px-3 py-1.5 rounded-lg transition text-xs shadow-premium"
              >
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </form>
        )}

        {/* KPI Strip — real data from projects & simulations */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-surface-card border border-border-custom rounded-card p-5 flex flex-col justify-between h-[110px] hover:border-primary/20 transition-all duration-200">
            <div>
              <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Total Projects</p>
              <h3 className="text-2xl font-bold text-text-primary tracking-tight mt-1">
                {fetching ? "—" : totalProjects}
              </h3>
            </div>
            <div className="flex items-center gap-1 text-[10px] font-medium text-text-secondary">
              <Layers size={11} />
              <span>{simProjects} simulation · {liveProjects} live</span>
            </div>
          </div>

          <div className="bg-surface-card border border-border-custom rounded-card p-5 flex flex-col justify-between h-[110px] hover:border-primary/20 transition-all duration-200">
            <div>
              <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Simulations Run</p>
              <h3 className="text-2xl font-bold text-text-primary tracking-tight mt-1">
                {simsLoading ? "—" : latestSims.length}
              </h3>
            </div>
            <div className="flex items-center gap-1 text-[10px] font-medium text-success">
              <Activity size={11} />
              <span>{completedSims.length} completed · {runningSims.length} running</span>
            </div>
          </div>

          <div className="bg-surface-card border border-border-custom rounded-card p-5 flex flex-col justify-between h-[110px] hover:border-primary/20 transition-all duration-200">
            <div>
              <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Live DB Projects</p>
              <h3 className="text-2xl font-bold text-text-primary tracking-tight mt-1">
                {fetching ? "—" : liveProjects}
              </h3>
            </div>
            <div className="flex items-center gap-1 text-[10px] font-medium text-text-secondary">
              <Database size={11} />
              <span>{liveProjects > 0 ? "Real-time control active" : "No live projects yet"}</span>
            </div>
          </div>

          <div className="bg-surface-card border border-border-custom rounded-card p-5 flex flex-col justify-between h-[110px] hover:border-primary/20 transition-all duration-200">
            <div>
              <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Last Activity</p>
              <h3 className="text-sm font-bold text-text-primary tracking-tight mt-2 leading-tight">
                {simsLoading
                  ? "—"
                  : latestSims.length > 0
                  ? new Date(latestSims[0].created_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" })
                  : projects.length > 0
                  ? new Date(projects[0].updated_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" })
                  : "—"}
              </h3>
            </div>
            <div className="flex items-center gap-1 text-[10px] font-medium text-text-secondary">
              <Clock size={11} />
              <span>{latestSims.length > 0 ? `Latest: ${latestSims[0].algorithm ?? "—"}` : "No simulations yet"}</span>
            </div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">

          {/* Left: Projects List */}
          <div className="lg:col-span-2 space-y-6">
            <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Active Studies</h2>

            {fetching ? (
              <div className="bg-surface-card border border-border-custom rounded-card py-20 text-center text-xs text-text-secondary">
                Loading projects…
              </div>
            ) : projects.length === 0 ? (
              <div className="bg-surface-card border border-border-custom rounded-card py-20 text-center space-y-4">
                <div className="text-3xl">📦</div>
                <p className="text-sm text-text-primary font-medium">No projects yet</p>
                <p className="text-xs text-text-secondary">Create your first study to begin GNN modeling.</p>
                <button
                  onClick={() => setShowForm(true)}
                  className="bg-primary hover:bg-primary/95 text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow-premium"
                >
                  Create project
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {projects.map((p) => (
                  <ProjectCard key={p.id} project={p} onDelete={handleDeleteProject} />
                ))}
              </div>
            )}

            {/* Recent Simulations Table */}
            {!simsLoading && latestSims.length > 0 && (
              <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
                <h3 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Recent Simulations</h3>
                <div className="space-y-1">
                  {latestSims.slice(0, 5).map((sim) => {
                    const proj = projects.find((p) => p.id === sim.project_id);
                    const statusColor =
                      sim.status === "completed" ? "text-success" :
                      sim.status === "running"   ? "text-[#EF9F27]" :
                      sim.status === "failed"    ? "text-danger" :
                      "text-text-tertiary";
                    return (
                      <div key={sim.id} className="flex items-center justify-between py-2.5 border-b border-border-custom last:border-0">
                        <div>
                          <p className="text-xs font-semibold text-text-primary">{proj?.name ?? "Unknown project"}</p>
                          <p className="text-[10px] text-text-tertiary mt-0.5">{sim.algorithm} · {sim.scenario}</p>
                        </div>
                        <div className="text-right">
                          <span className={`text-[10px] font-semibold ${statusColor}`}>{sim.status}</span>
                          {sim.status === "running" && (
                            <div className="w-16 h-1 bg-border-custom rounded-full mt-1 overflow-hidden">
                              <div className="h-full bg-[#EF9F27] rounded-full transition-all" style={{ width: `${sim.progress}%` }} />
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {latestSims.length > 5 && (
                  <p className="text-[10px] text-text-tertiary text-center">
                    +{latestSims.length - 5} more simulations across all projects
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Right: Quick Actions & System Status */}
          <div className="space-y-6">

            {/* System Status */}
            <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">System Status</h3>
              </div>
              <div className="space-y-3">
                {[
                  { label: "API Server", ok: true },
                  { label: "AI Engine (GNN)", ok: completedSims.length > 0 },
                  { label: "Database", ok: true },
                  { label: "Live Control", ok: liveProjects > 0 },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <span className="text-xs text-text-secondary">{item.label}</span>
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${item.ok ? "bg-success" : "bg-border-custom"}`} />
                      <span className={`text-[10px] font-medium ${item.ok ? "text-success" : "text-text-tertiary"}`}>
                        {item.ok ? "Operational" : "Idle"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Workspace Actions */}
            <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Workspace Actions</h3>
              <div className="flex flex-col gap-2">
                <button
                  onClick={() => setShowForm(true)}
                  className="w-full text-left px-4 py-3 bg-surface border border-border-custom rounded-lg text-xs font-semibold text-text-primary hover:border-primary/30 transition flex items-center justify-between group"
                >
                  <span>New simulation project</span>
                  <ArrowUpRight size={14} className="text-text-tertiary group-hover:text-primary transition" />
                </button>
                <button
                  onClick={() => { setNewMode("live"); setShowForm(true); }}
                  className="w-full text-left px-4 py-3 bg-surface border border-border-custom rounded-lg text-xs font-semibold text-text-primary hover:border-primary/30 transition flex items-center justify-between group"
                >
                  <span>New live control project</span>
                  <Zap size={14} className="text-text-tertiary group-hover:text-primary transition" />
                </button>
                {completedSims.length > 0 && (
                  <Link
                    href={`/projects/${completedSims[0].project_id}/results`}
                    className="w-full text-left px-4 py-3 bg-surface border border-border-custom rounded-lg text-xs font-semibold text-text-primary hover:border-primary/30 transition flex items-center justify-between group"
                  >
                    <span>View latest results</span>
                    <ArrowUpRight size={14} className="text-text-tertiary group-hover:text-primary transition" />
                  </Link>
                )}
              </div>
            </div>

            {/* Project breakdown */}
            {projects.length > 0 && (
              <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-3">
                <h3 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Project Breakdown</h3>
                {projects.slice(0, 4).map((p) => (
                  <div key={p.id} className="flex items-center justify-between">
                    <span className="text-xs text-text-primary truncate max-w-[130px]">{p.name}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                      p.mode === "live"
                        ? "bg-[#1D9E75]/10 text-[#1D9E75]"
                        : "bg-primary/10 text-primary"
                    }`}>
                      {p.mode === "live" ? "Live" : "Sim"}
                    </span>
                  </div>
                ))}
                {projects.length > 4 && (
                  <p className="text-[10px] text-text-tertiary">+{projects.length - 4} more projects</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

function ProjectCard({ project, onDelete }: { project: Project; onDelete: (id: string) => void }) {
  const hasTopology = !!project.topology?.agents?.length || !!project.topology?.edges?.length;
  const hasDataset  = !!project.dataset_id;
  const isLiveMode  = project.mode === "live";

  let steps: { label: string; done: boolean; href: string }[] = [];
  if (isLiveMode) {
    steps = [
      { label: "Live Control", done: false, href: `/projects/${project.id}/live` },
    ];
  } else {
    steps = [
      { label: "Data",     done: hasDataset,  href: `/projects/${project.id}/upload`   },
      { label: "Topology", done: hasTopology, href: `/projects/${project.id}/builder`  },
      { label: "Simulate", done: false,       href: `/projects/${project.id}/simulate` },
      { label: "Results",  done: false,       href: `/projects/${project.id}/results`  },
    ];
  }

  const nextStep = steps.find((s) => !s.done) ?? steps[steps.length - 1];
  const createdAt = new Date(project.created_at).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

  return (
    <div className="bg-surface-card border border-border-custom rounded-card p-5 hover:border-primary/40 transition-all duration-200 group flex flex-col justify-between h-[185px]">
      <div>
        <div className="flex items-center justify-between gap-2">
          <h3 className="font-semibold text-sm text-text-primary truncate">{project.name}</h3>
          <div className="flex items-center gap-2 shrink-0">
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
              isLiveMode
                ? "bg-[#1D9E75]/10 text-[#1D9E75] border border-[#1D9E75]/20"
                : "bg-primary/10 text-primary border border-primary/20"
            }`}>
              {isLiveMode ? "Live DB" : "Simulation"}
            </span>
            <button 
              onClick={(e) => { e.preventDefault(); onDelete(project.id); }}
              className="text-text-tertiary hover:text-danger transition p-1 rounded-md hover:bg-white/5"
              title="Delete Project"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>
        <p className="text-text-tertiary text-[10px] mt-0.5">Created {createdAt}</p>

        {/* Progress steps */}
        <div className="flex gap-2.5 mt-4 flex-wrap">
          {steps.map((s) => (
            <div key={s.label} className="flex items-center gap-1">
              <span className={`w-1.5 h-1.5 rounded-full ${s.done ? "bg-success" : "bg-border-custom"}`} />
              <span className={`text-[10px] ${s.done ? "text-success font-medium" : "text-text-tertiary"}`}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-2 mt-4">
        <Link
          href={nextStep.href}
          className="flex-1 text-center bg-primary hover:bg-primary/95 text-white text-xs font-semibold py-2 rounded-lg transition duration-150 shadow-premium"
        >
          {isLiveMode ? "Open Live Control →" : "Continue →"}
        </Link>
        {!isLiveMode && (
          <Link
            href={`/projects/${project.id}/results`}
            className="text-xs text-text-secondary hover:text-text-primary py-2 px-3 rounded-lg border border-border-custom hover:border-text-secondary/30 transition duration-150"
          >
            Results
          </Link>
        )}
      </div>
    </div>
  );
}
