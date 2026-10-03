"use client";

import { useState, useCallback } from "react";
import { useRouter, useParams } from "next/navigation";
import { api, Dataset, EDASummary } from "@/lib/api";
import ProjectNav from "@/components/ProjectNav";
import DashboardLayout from "@/components/DashboardLayout";
import { UploadCloud, CheckCircle2, AlertTriangle, ArrowRight, Database } from "lucide-react";

export default function UploadPage() {
  const params = useParams();
  const projectId = params.id as string;
  const router = useRouter();

  const [dragging, setDragging]   = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dataset, setDataset]     = useState<Dataset | null>(null);
  const [error, setError]         = useState("");

  const handleFile = useCallback(async (file: File) => {
    setError("");
    setUploading(true);
    try {
      const ds = await api.upload.uploadFile(file);
      setDataset(ds);
      // Link dataset to project
      await api.projects.update(projectId, { dataset_id: ds.id });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [projectId]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <DashboardLayout>
      <ProjectNav projectId={projectId} step="upload" />

      <div className="max-w-3xl mx-auto px-8 py-10 space-y-6 select-none">
        <div>
          <h1 className="text-xl font-bold text-text-primary tracking-tight">Upload Your Data</h1>
          <p className="text-xs text-text-secondary mt-1">
            Provide historical sales, demand logs, or inventory files. Supported formats: CSV, Excel.
          </p>
        </div>

        {!dataset ? (
          <>
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`border border-dashed rounded-card p-16 text-center transition duration-150 cursor-pointer ${
                dragging
                  ? "border-primary bg-primary/10"
                  : "border-border-custom bg-surface-card hover:border-primary/50"
              }`}
            >
              {uploading ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  <p className="text-xs text-text-secondary">Analyzing demand schemas…</p>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <UploadCloud size={36} className="text-primary mb-4" />
                  <p className="text-sm font-semibold text-text-primary mb-1">Drop your CSV or Excel file here</p>
                  <p className="text-[11px] text-text-secondary mb-5">sales_history.csv, inventory.xlsx — columns auto-detected</p>
                  <label className="cursor-pointer bg-primary hover:bg-primary/95 text-white font-semibold px-4 py-2 rounded-lg transition text-xs shadow-premium">
                    Browse Files
                    <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileInput} className="hidden" />
                  </label>
                </div>
              )}
            </div>

            {error && (
              <div className="bg-danger/10 border border-danger/20 text-danger rounded-lg px-4 py-2.5 text-xs">
                {error}
              </div>
            )}

            {/* Demo data option */}
            <div className="bg-surface-card border border-border-custom rounded-card p-5 flex items-center justify-between">
              <div>
                <p className="font-semibold text-xs text-text-primary">Use SupplyChain M5 Demo Dataset</p>
                <p className="text-[10px] text-text-tertiary mt-0.5">Walmart model · 1 store · 50 SKUs · 1913 days</p>
              </div>
              <button
                onClick={() => router.push(`/projects/${projectId}/builder`)}
                className="text-xs text-primary hover:underline font-semibold flex items-center gap-1"
              >
                Use demo <ArrowRight size={12} />
              </button>
            </div>
          </>
        ) : (
          <EDACard dataset={dataset} projectId={projectId} onContinue={() => router.push(`/projects/${projectId}/builder`)} />
        )}
      </div>
    </DashboardLayout>
  );
}

function EDACard({ dataset, onContinue, projectId }: { dataset: Dataset; onContinue: () => void; projectId: string }) {
  const eda: EDASummary = dataset.eda_summary || {};
  
  const [defaultCost, setDefaultCost] = useState("50");
  const [horizon, setHorizon] = useState("365");
  const [saving, setSaving] = useState(false);

  const handleContinue = async () => {
    setSaving(true);
    try {
      const project = await api.projects.get(projectId);
      let updatedTopology: any = { ...project.topology, default_cost: Number(defaultCost), horizon_days: Number(horizon) };

      if (eda.multi_echelon) {
        const agents: any[] = [];
        const edges: any[] = [];
        
        const suppliers = eda.supplier_names || [];
        const distributors = eda.distributor_names || [];
        const retailers = eda.retailer_names || [];
        
        suppliers.forEach((name, i) => {
          const id = `supplier_${name}`;
          agents.push({ id, type: "supplier", name: name.toUpperCase(), capacity: 1000, lead_time_min: 1, lead_time_max: 3, unit_cost: Number(defaultCost), holding_cost: 1.0, stockout_penalty: 500, policy: "mappo_gnn", position_x: 100, position_y: 100 + i * 150 });
        });
        
        distributors.forEach((name, i) => {
          const id = `distributor_${name}`;
          agents.push({ id, type: "distributor", name: name.toUpperCase(), capacity: 1000, lead_time_min: 1, lead_time_max: 3, unit_cost: Number(defaultCost), holding_cost: 1.5, stockout_penalty: 500, policy: "mappo_gnn", position_x: 400, position_y: 100 + i * 150 });
          suppliers.forEach(sup => {
            edges.push({ id: `e_${sup}_${name}`, source: `supplier_${sup}`, target: id, transport_cost: 0.1 });
          });
        });
        
        retailers.forEach((name, i) => {
          const id = `retailer_${name}`;
          agents.push({ id, type: "retailer", name: name.toUpperCase(), capacity: 500, lead_time_min: 1, lead_time_max: 3, unit_cost: Number(defaultCost), holding_cost: 2.0, stockout_penalty: 500, policy: "mappo_gnn", position_x: 700, position_y: 100 + i * 150 });
          distributors.forEach(dist => {
            edges.push({ id: `e_${dist}_${name}`, source: `distributor_${dist}`, target: id, transport_cost: 0.2 });
          });
        });
        
        updatedTopology = { ...updatedTopology, agents, edges };
      }

      await api.projects.update(projectId, { topology: updatedTopology });
      onContinue();
    } catch (e) {
      console.error(e);
      onContinue();
    }
  };

  // If there's an anomaly (e.g. no rows, zero stores)
  const hasAnomaly = !eda.row_count || eda.row_count === 0;

  return (
    <div className="space-y-6">
      <div className="bg-success/10 border border-success/20 text-success rounded-card px-4 py-3 text-xs flex items-center gap-2">
        <CheckCircle2 size={16} />
        <span>File uploaded successfully: <strong>{dataset.original_name}</strong></span>
      </div>

      {hasAnomaly && (
        <div className="bg-warning/10 border border-warning/20 text-warning rounded-card px-4 py-3 text-xs flex items-center gap-2">
          <AlertTriangle size={16} />
          <span>Potential data anomalies detected: empty rows or incomplete periods found.</span>
        </div>
      )}

      {/* Stats Chips */}
      <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
        <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Dataset Details</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {(eda.multi_echelon ? [
            { label: "Rows (Days)", value: eda.row_count?.toLocaleString() ?? "—" },
            { label: "Retailers", value: eda.num_retailers ?? "0" },
            { label: "Distributors", value: eda.num_distributors ?? "0" },
            { label: "Suppliers", value: eda.num_suppliers ?? "0" },
          ] : [
            { label: "Rows Detected", value: eda.row_count?.toLocaleString() ?? "—" },
            { label: "Columns", value: eda.column_count ?? "—" },
            { label: "Total SKUs", value: eda.num_skus ?? "—" },
            { label: "Unique Stores", value: eda.num_stores ?? "—" },
          ]).map((s) => (
            <div key={s.label} className="bg-surface border border-border-custom rounded-lg p-3">
              <p className="text-text-tertiary text-[10px] uppercase font-semibold">{s.label}</p>
              <p className="text-sm font-bold text-text-primary mt-1">{String(s.value)}</p>
            </div>
          ))}
        </div>

        {eda.demand_mean !== undefined && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-border-custom">
            {[
              { label: "Mean Demand", value: eda.demand_mean?.toFixed(1) ?? "—" },
              { label: "Demand Std Dev", value: eda.demand_std?.toFixed(1) ?? "—" },
              { label: "Min demand", value: eda.demand_min ?? "—" },
              { label: "Max demand", value: eda.demand_max ?? "—" },
            ].map((s) => (
              <div key={s.label} className="bg-surface border border-border-custom rounded-lg p-3">
                <p className="text-text-tertiary text-[10px] uppercase font-semibold">{s.label}</p>
                <p className="text-sm font-bold text-text-primary mt-1">{String(s.value)}</p>
              </div>
            ))}
          </div>
        )}

        {eda.date_min && (
          <div className="text-[10px] text-text-secondary flex items-center gap-1.5 pt-2">
            <span className="w-1 h-1 rounded-full bg-primary" />
            <span>Timeline: {eda.date_min} to {eda.date_max} ({eda.num_periods} periods mapped)</span>
          </div>
        )}

        {eda.top_skus && eda.top_skus.length > 0 && (
          <div className="pt-2 space-y-1.5">
            <p className="text-[10px] font-semibold text-text-tertiary uppercase tracking-wider">Top SKUs Catalog</p>
            <div className="flex flex-wrap gap-1.5">
              {eda.top_skus.slice(0, 8).map((sku) => (
                <span key={sku} className="bg-primary/10 border border-primary/20 text-primary text-[10px] font-semibold px-2 py-0.5 rounded">
                  {sku}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Override parameters */}
      <div className="bg-surface-card border border-border-custom rounded-card p-5 space-y-4">
        <h3 className="text-xs font-bold text-text-primary uppercase tracking-widest">Data Config overrides</h3>
        <p className="text-xs text-text-secondary">Provide default parameters for nodes missing explicit database values.</p>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-xs text-text-secondary font-medium block">Default Unit Cost ($)</label>
            <input 
              type="number" 
              value={defaultCost} 
              onChange={(e) => setDefaultCost(e.target.value)}
              className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary transition" 
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-text-secondary font-medium block">Time Horizon (Days)</label>
            <input 
              type="number" 
              value={horizon} 
              onChange={(e) => setHorizon(e.target.value)}
              className="w-full bg-surface border border-border-custom rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary transition" 
            />
          </div>
        </div>
      </div>

      <button
        onClick={handleContinue}
        disabled={saving}
        className="w-full bg-primary hover:bg-primary/95 text-white font-semibold py-3 rounded-lg transition disabled:opacity-50 text-sm shadow-premium flex items-center justify-center gap-1.5"
      >
        {saving ? "Configuring…" : "Save & build topology"} <ArrowRight size={14} />
      </button>
    </div>
  );
}
