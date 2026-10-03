/**
 * API client — thin wrapper over fetch with auth token injection
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("sm_token");
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(body && !(body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
    ...(options.headers as Record<string, string> | undefined),
  };

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    ...options,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// Auth
export const api = {
  auth: {
    register: (email: string, password: string, name: string) =>
      request<{ access_token: string; user: User }>("POST", "/api/auth/register", { email, password, name }),
    login: (email: string, password: string) =>
      request<{ access_token: string; user: User }>("POST", "/api/auth/login", { email, password }),
    me: () => request<User>("GET", "/api/auth/me"),
  },

  projects: {
    list: () => request<Project[]>("GET", "/api/projects/"),
    get: (id: string) => request<Project>("GET", `/api/projects/${id}`),
    create: (name: string, mode: string = "simulation") =>
      request<Project>("POST", "/api/projects/", { name, mode }),
    update: (id: string, data: Partial<Project>) =>
      request<Project>("PATCH", `/api/projects/${id}`, data),
    saveTopology: (id: string, topology: Topology) =>
      request<Project>("PUT", `/api/projects/${id}/topology`, topology),
    delete: (id: string) => request<void>("DELETE", `/api/projects/${id}`),
    templates: () => request<TopologyTemplate[]>("GET", "/api/projects/templates"),
  },

  upload: {
    uploadFile: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return request<Dataset>("POST", "/api/upload/", fd);
    },
    get: (id: string) => request<Dataset>("GET", `/api/upload/${id}`),
    preview: (id: string, rows = 20) =>
      request<{ columns: string[]; rows: Record<string, unknown>[] }>(
        "GET", `/api/upload/${id}/preview?rows=${rows}`
      ),
  },

  simulations: {
    launch: (data: SimulationCreate) =>
      request<Simulation>("POST", "/api/simulate/", data),
    get: (id: string) => request<SimulationDetail>("GET", `/api/simulate/${id}`),
    listByProject: (projectId: string) =>
      request<Simulation[]>("GET", `/api/simulate/project/${projectId}`),
    whatIf: (simId: string, scenarioType: string, parameters?: Record<string, unknown>) =>
      request<WhatIfResult>("POST", "/api/simulate/whatif", {
        simulation_id: simId,
        scenario_type: scenarioType,
        parameters: parameters ?? {},
      }),
  },

  results: {
    get: (id: string) => request<SimulationDetail>("GET", `/api/results/${id}`),
    exportCsv: (id: string) =>
      fetch(`${API_BASE}/api/results/${id}/export/csv`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      }),
    exportPdf: (id: string) =>
      fetch(`${API_BASE}/api/results/${id}/export/pdf`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      }),
    analyze: (id: string) =>
      request<{ analysis: string }>("POST", `/api/results/${id}/analyze`),
    chat: (id: string, message: string, history: { role: string; content: string }[]) =>
      request<{ reply: string }>("POST", `/api/results/${id}/chat`, { message, history }),
  },

  dbConnector: {
    test: (cfg: DBConnectionConfig) =>
      request<DBTestResult>("POST", "/api/db-connector/test", cfg),
    preview: (cfg: DBConnectionConfig) =>
      request<DBPreviewResponse>("POST", "/api/db-connector/preview", cfg),
    importData: (cfg: DBConnectionConfig, projectId: string) =>
      request<DBImportResult>(
        "POST",
        `/api/db-connector/import?project_id=${encodeURIComponent(projectId)}`,
        cfg
      ),
    importMulti: (req: MultiTableImportRequest, projectId: string) =>
      request<MultiTableImportResult>(
        "POST",
        `/api/db-connector/import-multi?project_id=${encodeURIComponent(projectId)}`,
        req
      ),
  },

  liveControl: {
    infer: (data: LiveControlRequest) =>
      request<LiveControlResponse>("POST", "/api/live-control/infer", data),
    explain: (data: ExplainRequest) =>
      request<ExplainResponse>("POST", "/api/live-control/explain", data),
  },
};

// WebSocket helper
export function createSimulationSocket(
  simId: string,
  onMessage: (msg: WSMessage) => void
): WebSocket {
  const wsBase = (process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000").replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/simulation/${simId}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  name?: string;
  avatar_url?: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  topology?: Topology;
  dataset_id?: string;
  mode?: "simulation" | "live";
  created_at: string;
  updated_at: string;
}

export interface AgentNode {
  id: string;
  type: "supplier" | "distributor" | "manufacturer" | "retailer";
  name: string;
  capacity: number;
  lead_time_min: number;
  lead_time_max: number;
  holding_cost: number;
  stockout_penalty: number;
  policy: string;
  position_x: number;
  position_y: number;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  transport_cost: number;
}

export interface Topology {
  agents: AgentNode[];
  edges: TopologyEdge[];
}

export interface TopologyTemplate {
  id: string;
  name: string;
  description: string;
  agents: AgentNode[];
  edges: TopologyEdge[];
}

export interface Dataset {
  id: string;
  filename: string;
  original_name?: string;
  row_count?: number;
  columns?: Record<string, string>;
  eda_summary?: EDASummary;
  uploaded_at: string;
}

export interface EDASummary {
  row_count: number;
  column_count: number;
  demand_mean?: number;
  demand_std?: number;
  demand_min?: number;
  demand_max?: number;
  demand_median?: number;
  date_min?: string;
  date_max?: string;
  num_periods?: number;
  num_skus?: number;
  num_stores?: number;
  top_skus?: string[];
  multi_echelon?: boolean;
  num_retailers?: number;
  num_distributors?: number;
  num_suppliers?: number;
  retailer_names?: string[];
  distributor_names?: string[];
  supplier_names?: string[];
}

export interface CustomDisruption {
  node_id: string;
  start_day: number;
  end_day: number;
  disruption_type: 1 | 2 | 3;
  magnitude: number;
}

export interface SimulationCreate {
  project_id: string;
  algorithm: "mappo_gnn" | "mappo" | "ss_policy" | "base_stock" | "all";
  scenario: "normal" | "demand_spike" | "lead_time_delay" | "node_failure" | "combined";
  episodes: number;
  horizon?: number;
  holding_cost_mult?: number;
  stockout_penalty_mult?: number;
  custom_disruptions?: CustomDisruption[];
}

export interface Simulation {
  id: string;
  project_id: string;
  algorithm: string;
  scenario: string;
  episodes: number;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  kpis?: KPIs;
}

export interface KPIs {
  fill_rate: number;
  total_cost: number;
  avg_inventory: number;
  avg_lead_time: number;
  disruptions_handled: number;
  algorithm: string;
  scenario: string;
  inventory_turnover?: number;
  shortage_days?: number;
  holding_cost?: number;
  transport_cost?: number;
  ai_savings?: number;
}

export interface SimulationDetail extends Simulation {
  timeline?: TimelineStep[];
  agent_logs?: AgentLog[];
  llm_summary?: LLMSummary;
  comparison?: Record<string, unknown>;
}

export interface TimelineStep {
  step: number;
  cost: number;
  inventories: Record<string, number>;
  demands: Record<string, number>;
}

export interface AgentLog {
  agent_id: string;
  agent_name: string;
  avg_inventory: number;
  orders_placed: number;
  fill_rate: number;
  avg_cost_per_step: number;
}

export interface LLMSummary {
  global: string;
  recommendations: RecommendationCard[];
  scenario_note?: string;
  comparison?: Record<string, unknown>;
  comparison_summary?: string;
}

export interface RecommendationCard {
  agent_id: string;
  agent_name: string;
  action: string;
  reason: string;
  confidence: "High" | "Medium" | "Low";
  urgency: "Urgent" | "Normal" | "Monitor";
  quantity: number;
  cost_estimate: number;
}

export interface WhatIfResult {
  scenario_description: string;
  fill_rate_delta: number;
  cost_delta: number;
  recovery_days: number;
  recommendations: RecommendationCard[];
  llm_narrative: string;
}

export interface WSMessage {
  type: "progress" | "completed" | "failed" | "ping";
  simulation_id?: string;
  progress?: number;
  status?: string;
  kpis?: KPIs;
}

// ── DB Connector ──────────────────────────────────────────────────────────────

export interface DBConnectionConfig {
  db_type: "postgresql" | "mysql" | "sqlite";
  host?: string;
  port?: number;
  database: string;
  username?: string;
  password?: string;
  table_name?: string;
  inventory_col?: string;
  demand_col?: string;
  node_col?: string;
}

export interface DBTestResult {
  success: boolean;
  tables: string[];
  error?: string;
}

export interface DBPreviewResponse {
  columns: string[];
  rows: (string | number | null)[][];
  row_count: number;
}

export interface DBImportResult {
  success: boolean;
  filename: string;
  rows_imported: number;
  columns: string[];
}

export interface TableConfig {
  table_name: string;
  inventory_col?: string;
  demand_col?: string;
  node_col?: string;
}

export interface MultiTableImportRequest {
  db_type: "postgresql" | "mysql" | "sqlite";
  host?: string;
  port?: number;
  database: string;
  username?: string;
  password?: string;
  tables: TableConfig[];
}

export interface MultiTableImportResult {
  success: boolean;
  results: { table_name: string; filename: string; rows_imported: number; columns: string[]; data?: any[][] }[];
  errors:  { table_name: string; error: string }[];
  total_rows: number;
}

// ── Live Control ──────────────────────────────────────────────────────────────

export interface NodeInventoryInput {
  node_id: string;
  node_name?: string;
  node_type: "retailer" | "distributor" | "supplier";
  inventory_units: number;
  capacity_units?: number;
  demand_units?: number;
  unit_cost?: number;
  lead_time_days?: number;
}

export interface LiveControlRequest {
  project_id: string;
  nodes: NodeInventoryInput[];
}

export interface NodeRecommendation {
  node_id: string;
  node_name: string;
  node_type: string;
  inventory_units: number;
  inventory_pct: number;
  capacity_units: number;
  order_units: number;
  urgency: "critical" | "high" | "medium" | "low";
  estimated_cost: number;
  estimated_lead_time_days: number;
}

export interface LiveControlResponse {
  recommendations: NodeRecommendation[];
  total_order_units: number;
  total_estimated_cost: number;
  analysis: string;
  topology_note: string;
}

export interface FeatureExplanation {
  feature_name: string;
  shap_value: number;
}

export interface ExplainRequest {
  project_id: string;
  nodes: NodeInventoryInput[];
  target_node_id: string;
}

export interface ExplainResponse {
  node_id: string;
  base_value: number;
  positive_features: FeatureExplanation[];
  negative_features: FeatureExplanation[];
  summary: string;
}
