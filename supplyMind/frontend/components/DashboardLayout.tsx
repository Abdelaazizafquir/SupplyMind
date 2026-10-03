"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, Project, Simulation } from "@/lib/api";
import {
  LayoutDashboard,
  Play,
  Network,
  Lightbulb,
  Plug,
  Users,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Bell,
  Search,
  Zap,
  Sun,
  Moon,
  X,
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2,
  FolderOpen,
  ArrowRight,
} from "lucide-react";
import { useTheme } from "@/lib/theme-context";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const { user, logout } = useAuth();
  const params = useParams();
  const pathname = usePathname();
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);
  const [projectMode, setProjectMode] = useState<string | null>(null);
  const { theme, toggle: toggleTheme } = useTheme();

  // Search
  const [showSearch, setShowSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchProjects, setSearchProjects] = useState<Project[]>([]);
  const [allProjects, setAllProjects] = useState<Project[]>([]);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Notifications
  const [showNotifications, setShowNotifications] = useState(false);
  const [notifications, setNotifications] = useState<Simulation[]>([]);
  const [notifsRead, setNotifsRead] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);

  const projectId = params?.id as string;

  // Fetch project mode
  useEffect(() => {
    if (projectId) {
      api.projects.get(projectId)
        .then((p) => setProjectMode(p.mode ?? "simulation"))
        .catch(() => setProjectMode("simulation"));
    } else {
      setProjectMode(null);
    }
  }, [projectId]);

  // Pre-load projects for search
  useEffect(() => {
    api.projects.list().then(setAllProjects).catch(() => {});
  }, []);

  // Load recent simulations as notifications
  useEffect(() => {
    // Fetch sims across all projects for notification feed
    const fetchSims = async () => {
      try {
        const projects = await api.projects.list();
        const allSims: Simulation[] = [];
        await Promise.allSettled(
          projects.map(async (p) => {
            try {
              const sims = await api.simulations.list(p.id);
              allSims.push(...sims);
            } catch {}
          })
        );
        allSims.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        setNotifications(allSims.slice(0, 8));
      } catch {}
    };
    fetchSims();
  }, []);

  // Search filtering
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchProjects(allProjects);
    } else {
      const q = searchQuery.toLowerCase();
      setSearchProjects(allProjects.filter(p => p.name.toLowerCase().includes(q)));
    }
  }, [searchQuery, allProjects]);

  // Open search with Cmd/Ctrl+K
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setShowSearch(true);
      }
      if (e.key === "Escape") {
        setShowSearch(false);
        setShowNotifications(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Focus search input when modal opens
  useEffect(() => {
    if (showSearch) setTimeout(() => searchInputRef.current?.focus(), 50);
  }, [showSearch]);

  // Close notifications on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifications(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const unreadCount = notifsRead ? 0 : notifications.filter(n => n.status === "completed").length;

  const isLiveProject = projectMode === "live";

  const workspaceLinks = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard, active: pathname === "/dashboard" },
  ];

  const intelligenceLinks = projectId
    ? isLiveProject
      ? [{ name: "Live Control", href: `/projects/${projectId}/live`, icon: Zap, active: pathname?.includes("/live") }]
      : [
          { name: "Agents",          href: `/projects/${projectId}/builder`,  icon: Network,   active: pathname?.includes("/builder")  },
          { name: "Simulations",     href: `/projects/${projectId}/simulate`, icon: Play,      active: pathname?.includes("/simulate") },
          { name: "Recommendations", href: `/projects/${projectId}/results`,  icon: Lightbulb, active: pathname?.includes("/results")  },
        ]
    : [];

  const settingsLinks = [
    { name: "Integrations", href: "/settings/integrations", icon: Plug,  disabled: false },
  ];

  function renderNavItem(link: { name: string; href: string; icon: React.ElementType; active?: boolean; disabled?: boolean }) {
    const Icon = link.icon;
    const cls = `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-150 ${
      link.active
        ? "bg-primary/10 border-l-2 border-primary text-primary font-medium"
        : "text-text-secondary hover:bg-black/5 dark:hover:bg-white/5 hover:text-text-primary"
    }`;
    if (link.disabled) {
      return (
        <div className={cls + " opacity-40 cursor-not-allowed"} title={collapsed ? link.name : undefined}>
          <Icon size={18} className="text-text-secondary" />
          {!collapsed && <span>{link.name}</span>}
        </div>
      );
    }
    return (
      <Link href={link.href} className={cls} title={collapsed ? link.name : undefined}>
        <Icon size={18} className={link.active ? "text-primary" : "text-text-secondary"} />
        {!collapsed && <span>{link.name}</span>}
      </Link>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-surface text-text-primary">
      {/* Sidebar */}
      <aside
        className={`border-r border-border-custom bg-surface-card flex flex-col justify-between transition-all duration-300 ${
          collapsed ? "w-16" : "w-[260px]"
        }`}
      >
        <div>
          {/* Logo */}
          <div className="h-16 flex items-center justify-between px-4 border-b border-border-custom">
            {!collapsed && (
              <Link href="/dashboard" className="flex items-center gap-1.5 font-semibold text-lg tracking-tight select-none">
                <span className="text-text-primary font-medium">Supply</span>
                <span className="text-primary font-bold">Mind</span>
              </Link>
            )}
            {collapsed && (
              <Link href="/dashboard" className="mx-auto text-primary font-black text-xl select-none">S</Link>
            )}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="p-1 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg text-text-tertiary hover:text-text-primary transition"
            >
              {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
          </div>

          {/* Navigation */}
          <nav className="p-3 space-y-6 overflow-y-auto">
            {/* WORKSPACE */}
            <div>
              {!collapsed && <p className="px-3 text-[10px] font-semibold tracking-wider text-text-tertiary uppercase mb-2">Workspace</p>}
              <ul className="space-y-1">
                {workspaceLinks.map(link => <li key={link.name}>{renderNavItem(link)}</li>)}
              </ul>
            </div>

            {/* INTELLIGENCE */}
            {intelligenceLinks.length > 0 && (
              <div>
                {!collapsed && <p className="px-3 text-[10px] font-semibold tracking-wider text-text-tertiary uppercase mb-2">Intelligence</p>}
                <ul className="space-y-1">
                  {intelligenceLinks.map(link => <li key={link.name}>{renderNavItem(link)}</li>)}
                </ul>
              </div>
            )}

            {/* SETTINGS */}
            <div>
              {!collapsed && <p className="px-3 text-[10px] font-semibold tracking-wider text-text-tertiary uppercase mb-2">Settings</p>}
              <ul className="space-y-1">
                {settingsLinks.map(link => (
                  <li key={link.name}>{renderNavItem({ ...link, active: pathname?.startsWith(link.href) })}</li>
                ))}
              </ul>
            </div>
          </nav>
        </div>

        {/* User footer */}
        <div className="p-3 border-t border-border-custom bg-surface-card">
          <div className="flex items-center justify-between gap-2">
            {!collapsed && (
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs uppercase shrink-0">
                  {user?.name?.slice(0, 2) || user?.email?.slice(0, 2) || "U"}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-text-primary truncate">{user?.name || user?.email}</p>
                  <p className="text-[10px] text-primary bg-primary-soft font-medium inline-block px-1.5 py-0.5 rounded mt-0.5">Enterprise</p>
                </div>
              </div>
            )}
            {collapsed && (
              <div className="w-8 h-8 mx-auto rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs uppercase shrink-0">
                {user?.name?.slice(0, 2) || user?.email?.slice(0, 2) || "U"}
              </div>
            )}
            {!collapsed && (
              <button onClick={logout} className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg text-text-tertiary hover:text-danger transition shrink-0" title="Sign out">
                <LogOut size={16} />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Layout */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Topbar */}
        <header className="h-16 border-b border-border-custom bg-surface-card flex items-center justify-between px-8 z-10 shrink-0">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <Link href="/dashboard" className="hover:text-text-primary transition">SupplyMind</Link>
            {projectId && (
              <>
                <span className="text-text-tertiary">/</span>
                <span className="font-mono text-text-tertiary">{projectId.slice(0, 8)}</span>
              </>
            )}
            {pathname && (
              <>
                <span className="text-text-tertiary">/</span>
                <span className="capitalize text-text-primary font-medium">{pathname.split("/").pop()}</span>
              </>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            {/* Search */}
            <button
              onClick={() => setShowSearch(true)}
              className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-surface border border-border-custom rounded-lg text-xs text-text-tertiary hover:border-primary/30 hover:text-text-primary transition-all duration-200 w-44"
            >
              <Search size={13} />
              <span>Search…</span>
              <span className="ml-auto text-[10px] bg-surface-card border border-border-custom rounded px-1.5 py-0.5 font-mono">⌘K</span>
            </button>

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              className="p-2 rounded-lg border border-border-custom bg-surface hover:border-primary/30 text-text-secondary hover:text-text-primary transition-all duration-200"
            >
              {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            </button>

            {/* Notifications */}
            <div className="relative" ref={notifRef}>
              <button
                onClick={() => { setShowNotifications(s => !s); setNotifsRead(true); }}
                className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg text-text-secondary hover:text-text-primary transition relative"
              >
                <Bell size={16} />
                {unreadCount > 0 && (
                  <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
                )}
              </button>

              {/* Notifications dropdown */}
              {showNotifications && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-surface-card border border-border-custom rounded-card shadow-2xl z-50 overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-border-custom">
                    <p className="text-xs font-bold text-text-primary uppercase tracking-wider">Notifications</p>
                    <button onClick={() => setShowNotifications(false)} className="text-text-tertiary hover:text-text-primary transition">
                      <X size={14} />
                    </button>
                  </div>
                  <div className="max-h-80 overflow-y-auto">
                    {notifications.length === 0 ? (
                      <div className="py-10 text-center text-xs text-text-tertiary">No recent activity</div>
                    ) : (
                      notifications.map(sim => {
                        const Icon = sim.status === "completed" ? CheckCircle2
                          : sim.status === "running" ? Loader2
                          : sim.status === "failed" ? AlertCircle
                          : Clock;
                        const color = sim.status === "completed" ? "text-success"
                          : sim.status === "running" ? "text-warning"
                          : sim.status === "failed" ? "text-danger"
                          : "text-text-tertiary";
                        return (
                          <Link
                            key={sim.id}
                            href={`/projects/${sim.project_id}/results`}
                            onClick={() => setShowNotifications(false)}
                            className="flex items-start gap-3 px-4 py-3 hover:bg-black/5 dark:hover:bg-white/5 transition border-b border-border-custom last:border-0"
                          >
                            <Icon size={15} className={`${color} shrink-0 mt-0.5 ${sim.status === "running" ? "animate-spin" : ""}`} />
                            <div className="min-w-0 flex-1">
                              <p className="text-xs font-semibold text-text-primary truncate">
                                Simulation — {sim.algorithm ?? "Unknown"}
                              </p>
                              <p className="text-[10px] text-text-tertiary mt-0.5">
                                {sim.scenario} · {new Date(sim.created_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}
                              </p>
                            </div>
                            <span className={`text-[9px] font-bold uppercase shrink-0 mt-1 ${color}`}>{sim.status}</span>
                          </Link>
                        );
                      })
                    )}
                  </div>
                  <div className="px-4 py-2 border-t border-border-custom">
                    <p className="text-[10px] text-text-tertiary text-center">{notifications.length} recent events</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto bg-surface">
          {children}
        </main>
      </div>

      {/* Search Modal */}
      {showSearch && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 px-4" onClick={() => setShowSearch(false)}>
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
          <div
            className="relative bg-surface-card border border-border-custom rounded-card w-full max-w-lg shadow-2xl overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            {/* Search Input */}
            <div className="flex items-center gap-3 px-4 py-3.5 border-b border-border-custom">
              <Search size={16} className="text-text-tertiary shrink-0" />
              <input
                ref={searchInputRef}
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search projects…"
                className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-tertiary outline-none"
              />
              <button onClick={() => setShowSearch(false)} className="text-text-tertiary hover:text-text-primary transition">
                <X size={15} />
              </button>
            </div>

            {/* Results */}
            <div className="max-h-72 overflow-y-auto">
              {searchProjects.length === 0 ? (
                <div className="py-10 text-center text-xs text-text-tertiary">No projects found</div>
              ) : (
                <div className="py-2">
                  <p className="px-4 pb-1.5 text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">
                    Projects ({searchProjects.length})
                  </p>
                  {searchProjects.map(p => (
                    <Link
                      key={p.id}
                      href={p.mode === "live" ? `/projects/${p.id}/live` : `/projects/${p.id}/simulate`}
                      onClick={() => { setShowSearch(false); setSearchQuery(""); }}
                      className="flex items-center gap-3 px-4 py-2.5 hover:bg-black/5 dark:hover:bg-white/5 transition group"
                    >
                      <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <FolderOpen size={13} className="text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-text-primary truncate">{p.name}</p>
                        <p className="text-[10px] text-text-tertiary">{p.mode === "live" ? "Live DB" : "Simulation"} · {new Date(p.created_at).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}</p>
                      </div>
                      <ArrowRight size={13} className="text-text-tertiary group-hover:text-primary transition shrink-0" />
                    </Link>
                  ))}
                </div>
              )}
            </div>

            <div className="px-4 py-2 border-t border-border-custom flex items-center justify-between">
              <p className="text-[10px] text-text-tertiary">Press <kbd className="bg-surface border border-border-custom rounded px-1.5 py-0.5 font-mono text-[9px]">Esc</kbd> to close</p>
              <p className="text-[10px] text-text-tertiary"><kbd className="bg-surface border border-border-custom rounded px-1.5 py-0.5 font-mono text-[9px]">⌘K</kbd> to open anytime</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
