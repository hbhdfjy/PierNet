export type WorkflowStatus =
  "draft" | "running" | "succeeded" | "failed" | "cancelled";
export type WorkflowStep = "source" | "definition" | "generation" | "complete";
export type WizardStep = "source" | "definition" | "generation";
export type SourceMode = "upload" | "simulation" | "expert";

export interface ParameterDefinition {
  index: number;
  name: string;
  display_name: string;
  description: string;
  unit: string;
}

export type OutputDefinition = ParameterDefinition;

export interface DataDefinition {
  schema_name?: string;
  schema_version?: number;
  version?: number;
  simulator: string;
  scenario: string;
  task_description: string;
  parameters: ParameterDefinition[];
  outputs: OutputDefinition[];
  sampling: {
    channels: number[] | null;
    time_stride: number;
    max_time_points: number | null;
  };
  confirmed?: boolean;
}

export interface SourceSnapshot {
  source_type: SourceMode;
  filename: string;
  simulator: string;
  scenario: string;
  ready: boolean;
  sample_count: number;
  input_dim: number;
  input_shape: number[];
  output_shape: number[];
  param_names: string[];
  file_size_bytes: number;
  input_stats: NumberStats;
  output_stats: NumberStats;
  preview: Array<{ expert_input: number[]; expert_output: number[] }>;
}

export interface NumberStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface DatasetArtifact {
  dataset_id: string;
  sample_count: number;
  schema_name: string;
  label_semantics?: string;
  class_names?: string[];
}

export interface WorkflowArtifacts {
  progress?: number | null;
  phase?: string;
  message?: string;
  text2comp?: DatasetArtifact;
  router?: DatasetArtifact;
  evaluation?: DatasetArtifact;
}

export interface WorkflowSummary {
  workflow_id: string;
  name: string;
  status: WorkflowStatus;
  current_step: WorkflowStep;
  created_at: number;
  updated_at: number;
}

export interface WorkflowSnapshot extends WorkflowSummary {
  source: SourceSnapshot | null;
  definition: DataDefinition | null;
  artifacts: WorkflowArtifacts | null;
  error: { code?: string; message?: string } | null;
  cancel_requested: boolean;
  can_define: boolean;
  can_generate: boolean;
  can_open_training: boolean;
}

export interface SimulationPreset {
  simulator: string;
  scenario: string;
  sample_count: number;
  has_data: boolean;
  output_shape: number[] | null;
}

export interface ExpertPreset {
  model_id: string;
  name: string;
  input_dim: number | null;
  output_dim: number | null;
  active: boolean;
  data_generation_enabled: boolean;
}

export interface Presets {
  accepted_uploads: string[];
  max_upload_bytes: number;
  max_generation_samples: number;
  router_mode: "binary";
  simulations: SimulationPreset[];
  experts: ExpertPreset[];
}

export interface GenerateOptions {
  max_samples: number;
  variants_per_sample: number;
  negative_ratio: number;
  seed: number;
}

export interface LLMConfig {
  provider: string;
  model: string;
  base_url: string;
  api_key_masked: string;
  has_api_key: boolean;
  temperature: number;
  max_tokens: number;
  thinking: "enabled" | "disabled";
}

export interface LLMConfigRequest {
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  thinking: "enabled" | "disabled";
}

export interface LLMTestResult {
  ok: boolean;
  message: string;
  response_preview: string;
}
