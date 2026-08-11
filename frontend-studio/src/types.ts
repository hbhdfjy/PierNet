export type StageStatus = "waiting" | "running" | "succeeded" | "failed" | "cancelled";

export type ProjectStatus = "draft" | "ready" | "running" | "failed" | "cancelled";

export interface StageSnapshot {
  id: string;
  title: string;
  status: StageStatus;
  progress: number | null;
  message: string;
  retryable: boolean;
  started_at: number | null;
  finished_at: number | null;
}

export interface ProjectSummary {
  project_id: string;
  name: string;
  goal: string;
  status: ProjectStatus;
  current_stage: string;
  created_at: number;
  updated_at: number;
}

export interface DataColumn {
  name: string;
  dtype: string;
  numeric: boolean;
  sample: unknown[];
}

export interface DataResource {
  filename: string;
  format: string;
  size_bytes: number;
  needs_mapping: boolean;
  samples?: number;
  input_shape?: number[];
  input_dim?: number;
  output_shape?: number[];
  output_dim?: number;
  input_names?: string[];
  output_names?: string[];
  input_stats?: NumericStats;
  output_stats?: NumericStats;
  preview?: Array<{ inputs: number[]; outputs: number[] }>;
}

export interface ExpertResource {
  filename: string;
  size_bytes: number;
  runtime: string;
  entrypoint: string;
  callable: string;
  file_count: number;
  validated: boolean;
}

export interface NumericStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface CompatibilityReport {
  compatible: boolean;
  sample_count: number;
  input_shape: number[];
  expected_output_shape: number[];
  actual_output_shape: number[];
  finite: boolean;
  sample_mse?: number;
  relative_rmse?: number;
  preview?: number[];
  message?: string;
}

export interface ProjectSnapshot extends ProjectSummary {
  stages: StageSnapshot[];
  data: DataResource | null;
  expert: ExpertResource | null;
  inspection: {
    data?: {
      kind: string;
      columns?: DataColumn[];
      suggested_input_fields?: string[];
      suggested_output_fields?: string[];
      needs_mapping: boolean;
    };
  } | null;
  compatibility: CompatibilityReport | null;
  artifacts: {
    recommended_prompt?: string;
    metrics?: Record<string, unknown>;
  } | null;
  result: InferenceResult | null;
  error: { code: string; message: string } | null;
  recommended_prompt: string | null;
  can_run: boolean;
  can_chat: boolean;
}

export interface InferenceResult {
  message: string;
  answer: string;
  routed: boolean;
  confidence: number;
  inputs: number[];
  output: number | number[] | number[][];
  chart: ChartSpec;
  latency_ms: number;
}

export interface ChatResponse extends InferenceResult {
  chat_id: string;
  project_id: string;
  created_at: number;
}

export type ChartSpec =
  | { kind: "metric"; value: number; label: string }
  | {
      kind: "line";
      x: number[];
      series: Array<{ name: string; values: number[] }>;
    }
  | {
      kind: "heatmap";
      rows: number;
      columns: number;
      values: number[][];
    };

export interface RunResponse {
  project_id: string;
  status: ProjectStatus;
  message: string;
}
