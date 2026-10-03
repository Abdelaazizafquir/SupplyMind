"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Check } from "lucide-react";

export default function ProjectNav({ projectId, step }: { projectId: string; step: string }) {
  const [mode, setMode] = useState<string>("simulation");

  useEffect(() => {
    api.projects.get(projectId)
      .then(p => {
        // Use dedicated mode field; fall back to topology.mode for old data
        setMode(p.mode || (p.topology as any)?.mode || "simulation");
      })
      .catch(console.error);
  }, [projectId]);

  const simSteps = [
    { id: "upload", label: "Data", href: `/projects/${projectId}/upload` },
    { id: "builder", label: "Agents", href: `/projects/${projectId}/builder` },
    { id: "simulate", label: "Config", href: `/projects/${projectId}/simulate` },
    { id: "results", label: "Review", href: `/projects/${projectId}/results` },
  ];

  if (mode === "live") {
    return (
      <div className="px-8 py-5 border-b border-border-custom bg-surface-card flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-text-primary">Live Database Control</h1>
          <p className="text-xs text-text-secondary mt-0.5">Real-time GNN actions & recommendations</p>
        </div>
      </div>
    );
  }

  const activeIdx = simSteps.findIndex((s) => s.id === step);

  return (
    <div className="border-b border-border-custom bg-surface-card px-8 py-6 flex flex-col items-center justify-center">
      {/* Stepper Wizard Container */}
      <div className="w-full max-w-xl flex items-center justify-between relative">
        {/* Connecting Lines */}
        <div className="absolute left-0 right-0 top-5 h-[2px] bg-border-custom -z-0" />
        <div
          className="absolute left-0 top-5 h-[2px] bg-primary transition-all duration-300 -z-0"
          style={{ width: `${(Math.max(0, activeIdx) / (simSteps.length - 1)) * 100}%` }}
        />

        {simSteps.map((s, idx) => {
          const isActive = s.id === step;
          const isDone = idx < activeIdx;
          const isUpcoming = idx > activeIdx;

          return (
            <div key={s.id} className="flex flex-col items-center z-10 relative">
              <Link href={s.href} className="group flex flex-col items-center">
                {/* Circle */}
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-mono text-sm font-medium transition-all duration-300 border-2 ${
                    isDone
                      ? "bg-success border-success text-white"
                      : isActive
                      ? "bg-primary border-primary text-white shadow-lg shadow-primary/20"
                      : "bg-surface-card border-border-custom text-text-tertiary group-hover:border-text-secondary group-hover:text-text-secondary"
                  }`}
                >
                  {isDone ? <Check size={16} strokeWidth={3} /> : idx + 1}
                </div>

                {/* Label */}
                <span
                  className={`text-xs mt-2 transition-all duration-300 ${
                    isActive
                      ? "text-primary font-semibold"
                      : isDone
                      ? "text-success font-medium"
                      : "text-text-tertiary group-hover:text-text-secondary"
                  }`}
                >
                  {s.label}
                </span>
              </Link>
            </div>
          );
        })}
      </div>
    </div>
  );
}
