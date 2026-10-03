"use client";

import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { ArrowRight, Brain, Cpu, BarChart3, Database, ShieldAlert, FileText, CheckCircle2 } from "lucide-react";

export default function LandingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) router.push("/dashboard");
  }, [user, loading, router]);

  return (
    <main className="min-h-screen bg-surface text-text-primary flex flex-col justify-between selection:bg-primary/30">
      {/* Navigation */}
      <nav className="border-b border-border-custom bg-surface-card sticky top-0 z-50">
        <div className="flex items-center justify-between px-8 py-4 max-w-7xl mx-auto w-full">
          <div className="flex items-center gap-2 select-none">
            <span className="text-xl font-bold tracking-tight text-text-primary">
              Supply<span className="text-primary">Mind</span>
            </span>
            <span className="text-[10px] bg-primary/10 border border-primary/20 text-primary font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider">
              AI Engine
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/auth/login"
              className="text-xs text-text-secondary hover:text-text-primary font-medium transition py-2 px-3 rounded-lg hover:bg-white/5"
            >
              Sign in
            </Link>
            <Link
              href="/auth/register"
              className="text-xs bg-primary hover:bg-primary/95 text-white font-semibold px-4 py-2 rounded-lg transition-all duration-150 shadow-premium"
            >
              Get started free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-5xl mx-auto px-8 pt-24 pb-20 text-left w-full">
        <div className="inline-flex items-center gap-2 bg-primary-soft border border-primary/20 text-primary text-xs font-semibold px-4 py-1.5 rounded-full mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          MAPPO + Graph Neural Networks Core
        </div>
        
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-text-primary leading-[1.1] mb-6 max-w-4xl">
          Optimize your supply chain <br />
          <span className="text-primary font-extrabold">with multi-agent reinforcement learning.</span>
        </h1>
        
        <p className="text-base sm:text-lg text-text-secondary max-w-3xl mb-10 leading-relaxed">
          Upload demand data, structure your topology visually, and let our MAPPO+GNN engine learn the optimal inventory control and procurement policies in real-time.
        </p>

        <div className="flex items-center gap-4 flex-wrap">
          <Link
            href="/auth/register"
            className="bg-primary hover:bg-primary/95 text-white font-semibold px-6 py-3 rounded-lg transition-all duration-150 shadow-premium text-sm flex items-center gap-2"
          >
            Start optimizing <ArrowRight size={16} />
          </Link>
          <Link
            href="/auth/login"
            className="border border-border-custom hover:border-text-secondary/30 text-text-secondary hover:text-text-primary font-medium px-6 py-3 rounded-lg transition text-sm"
          >
            Sign in
          </Link>
        </div>
      </section>

      {/* Grid Features */}
      <section className="max-w-7xl mx-auto px-8 pb-24 w-full">
        <p className="text-[10px] font-bold text-text-tertiary uppercase tracking-widest mb-6">Platform Capabilities</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            {
              icon: Brain,
              title: "MAPPO + GNN Engine",
              desc: "Train autonomous agents representing manufacturers, distributors, and retailers to negotiate supply chain nodes."
            },
            {
              icon: Cpu,
              title: "Live Training Simulation",
              desc: "Watch state values, order recommendations, and rewards converge step-by-step on an active dashboard."
            },
            {
              icon: BarChart3,
              title: "Disruption Scenarios",
              desc: "Simulate supplier failures, demand spikes, and logistical lead-time delays to stress-test policies."
            },
            {
              icon: Database,
              title: "Enterprise Connectors",
              desc: "Import inventory and demand tables directly from your PostgreSQL, MySQL, or SQLite database."
            },
            {
              icon: ShieldAlert,
              title: "What-If Sandbox",
              desc: "Predict operational impact for custom disruption periods and analyze alternative multi-echelon policies."
            },
            {
              icon: FileText,
              title: "AI Narrative Reports",
              desc: "Plain-English executive summaries and action plans written by LLMs trained on simulation outputs."
            }
          ].map((f) => (
            <div
              key={f.title}
              className="bg-surface-card border border-border-custom rounded-lg p-6 hover:border-primary/30 transition duration-200"
            >
              <div className="w-10 h-10 rounded-lg bg-primary-soft text-primary flex items-center justify-center mb-4">
                <f.icon size={20} />
              </div>
              <h3 className="font-semibold text-text-primary text-sm mb-2">{f.title}</h3>
              <p className="text-text-secondary text-xs leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Core Performance Stats */}
      <section className="border-t border-border-custom py-16 bg-surface-card w-full">
        <div className="max-w-7xl mx-auto px-8 grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            { value: "97.3%", label: "Average Fill Rate achieved under MAPPO" },
            { value: "−22%", label: "Total Cost reduction compared to classical (s,Q) policies" },
            { value: "Real-time", label: "GNN recommendation generation under live DB control" },
          ].map((s) => (
            <div key={s.label} className="flex flex-col border-l-2 border-primary/20 pl-6">
              <span className="text-2xl sm:text-3xl font-bold text-primary tracking-tight">{s.value}</span>
              <span className="text-xs text-text-secondary mt-1">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border-custom py-8 text-center text-text-tertiary text-xs bg-surface-card">
        <div className="max-w-7xl mx-auto px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <span>© 2026 SupplyMind. Built on PFE research.</span>
          <div className="flex gap-4">
            <span className="hover:text-text-secondary cursor-not-allowed">Terms</span>
            <span className="hover:text-text-secondary cursor-not-allowed">Privacy</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
