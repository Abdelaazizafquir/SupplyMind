"use client";

import DashboardLayout from "@/components/DashboardLayout";
import { Plug, Check, ExternalLink } from "lucide-react";

const INTEGRATIONS = [
  {
    name: "PostgreSQL",
    description: "Connect to your production PostgreSQL database for live inventory control.",
    status: "available",
    color: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    icon: "🐘",
    href: null,
  },
  {
    name: "MySQL",
    description: "Import supply chain data directly from a MySQL database.",
    status: "available",
    color: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    icon: "🐬",
    href: null,
  },
  {
    name: "SQLite",
    description: "Connect a local SQLite file for offline or development workflows.",
    status: "available",
    color: "bg-success/10 text-success border-success/20",
    icon: "📁",
    href: null,
  },
  {
    name: "CSV / Excel Upload",
    description: "Upload flat-file datasets to seed inventory and demand data.",
    status: "available",
    color: "bg-primary/10 text-primary border-primary/20",
    icon: "📊",
    href: null,
  },


];

export default function IntegrationsPage() {
  const available   = INTEGRATIONS.filter(i => i.status === "available");

  return (
    <DashboardLayout>
      <div className="p-8 max-w-5xl mx-auto space-y-10">
        {/* Header */}
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
              <Plug size={18} className="text-primary" />
            </div>
            <h1 className="text-xl font-bold text-text-primary tracking-tight">Integrations</h1>
          </div>
          <p className="text-sm text-text-secondary max-w-xl">
            Connect SupplyMind to your existing data infrastructure. Use live database integrations for real-time control or upload files to run simulations.
          </p>
        </div>

        {/* Active integrations */}
        <div className="space-y-4">
          <h2 className="text-xs font-bold text-text-secondary uppercase tracking-widest">Available Now</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {available.map(intg => (
              <div
                key={intg.name}
                className="bg-surface-card border border-border-custom rounded-card p-5 flex items-start gap-4 hover:border-primary/20 transition group"
              >
                <div className={`w-10 h-10 rounded-xl border flex items-center justify-center text-lg shrink-0 ${intg.color}`}>
                  {intg.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-sm font-semibold text-text-primary">{intg.name}</p>
                    <span className="flex items-center gap-1 text-[10px] font-bold text-success bg-success/10 border border-success/20 px-2 py-0.5 rounded-full">
                      <Check size={10} /> Active
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary leading-relaxed">{intg.description}</p>
                </div>
                {intg.href && (
                  <a href={intg.href} target="_blank" className="text-text-tertiary hover:text-primary transition shrink-0 mt-1">
                    <ExternalLink size={14} />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>




      </div>
    </DashboardLayout>
  );
}
