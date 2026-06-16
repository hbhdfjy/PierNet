import type { components } from './generated/openapi'

// ── Stage 3 样本记录结构（JSONL 与 Parquet record_json 共用）──────────────

export interface ObservationConfig {
  time_mode: string
  n_time_points: number
  channel_indices: number[] | null
  selected_output_names: string[]
}

export interface OutputInfo {
  name: string
  name_zh?: string
  description: string
  unit: string
  slice: [number, number | null]
}

export interface SampleMetadata {
  simulator: string
  scenario: string
  param_names: string[]
  timeseries_shape: [number, number]
  timeseries_shape_obs: [number, number]
  observation: ObservationConfig
  sample_idx: number
  language: 'en' | 'zh'
  style: 'technical' | 'popular' | 'concise'
  input_template: string
  target_template: string
  output_info: OutputInfo[]
}

export interface SampleRecord {
  input: string
  number: number[]
  params_transformed: number[]
  target: string
  metadata: SampleMetadata
}

// ── API 响应类型 ─────────────────────────────────────────────────

// Unified file catalog
export interface FileAsset {
  id: string
  platform: string
  platform_label: string
  stage: string
  stage_label: string
  kind: string
  kind_label: string
  title: string
  simulator?: string | null
  scenario?: string | null
  job_id?: string | null
  path: string
  count?: number | null
  count_label?: string | null
  file_size_bytes: number
  mtime: number
  valid?: boolean | null
  status: string
  protected: boolean
  deletable: boolean
  warnings: string[]
  errors: string[]
  details: Record<string, unknown>
}

export interface FileCatalogSummary {
  total_assets: number
  total_size_bytes: number
  invalid_count: number
  deletable_count: number
  protected_count: number
  by_platform: Record<string, number>
  by_stage: Record<string, number>
  by_kind: Record<string, number>
}

export interface FileCatalogResponse {
  generated_at: number
  assets: FileAsset[]
  summary: FileCatalogSummary
}

export interface FileCatalogMutationResponse {
  ok: boolean
  kind: string
  deleted?: number
  train_count?: number
  rebuilt?: string[]
  deleted_indexes?: number
  errors?: string[]
}

export interface DatasetInfo {
  name: string
  simulator: string
  scenario: string
  sample_count: number
  file_size_bytes: number
  mtime: number
}

export interface SamplesResponse {
  total: number
  page: number
  page_size: number
  items: SampleRecord[]
}

export interface DatasetStats {
  total_samples: number
  by_simulator: Record<string, number>
  by_scenario: Record<string, number>
  by_language: Record<string, number>
  by_style: Record<string, number>
  by_time_mode: Record<string, number>
  timeseries_shapes: Record<string, [number, number]>
}

export interface DashboardSummary {
  stats: DatasetStats
  datasets: DatasetInfo[]
  router: RouterStatus
}

// Stage 2 可用场景（来自 HDF5 文件扫描）
export interface Text2CompScenario {
  name: string
  simulator: string
  h5_file: string | null
  sample_count: number // HDF5 中的样本数（无 HDF5 时为 0）
  output_shape: number[] | null // HDF5 timeseries 的输出维度，如 [43, 365]
  existing_jsonl_count: number // 已生成的 JSONL 条数（兼容旧存储）
  existing_parquet_count?: number // 已生成的 Parquet 条数
  existing_sample_count?: number // 当前主样本存储中的已生成条数
  existing_storage?: 'jsonl' | 'parquet' | 'mixed' | null
  has_jsonl: boolean
  has_parquet?: boolean
  has_samples?: boolean
  has_h5: boolean // 是否有 HDF5 数据文件
  registered: boolean // 是否在 registry.yaml 中有注册信息
}

export interface Text2CompScenariosConfig {
  [dir_key: string]: Text2CompScenario[]
}

export interface GenerationConfig {
  llm?: {
    provider?: string
    model?: string
    temperature?: number
    max_tokens?: number
    max_retries?: number
    timeout?: number
  }
  generation?: {
    n_samples_per_scenario?: number
    max_workers?: number
    language_mix?: number
    styles?: string[]
    style_weights?: number[]
    transform_prob?: number
  }
  seed?: number
}

// ── 生成任务类型 ─────────────────────────────────────────────────

export interface LogLine {
  line: string
  ts: number
  progress?: ScenarioProgress
  stats?: LiveStats
  n_per_scenario?: number
}

export interface ScenarioProgress {
  scenario: string
  done: number
  total: number
}

export interface LiveStats {
  elapsed_sec: number
  samples_per_sec: number
}

export type JobStatus =
  | 'queued'
  | 'starting'
  | 'running'
  | 'evaluating'
  | 'stopping'
  | 'done'
  | 'error'
  | 'terminated'
  | 'external_terminated'
  | 'idle'

export interface GenerationJob {
  job_id: string
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  finished_at?: number | null
  error_message?: string | null
}

// ── 两阶段生成请求 ────────────────────────────────────────────────

export type GenerateTemplatesRequest = components['schemas']['GenerateTemplatesRequest'] & {
  scenarios: string[]
}

export type FillSamplesRequest = Omit<
  components['schemas']['FillSamplesRequest'],
  'compression' | 'output_format' | 'scenarios'
> & {
  scenarios: string[]
  output_format?: 'parquet' | 'jsonl' | 'both'
  compression?: 'zstd' | 'snappy' | 'gzip' | 'brotli' | 'none'
}

type ApiJobStartResponse = components['schemas']['JobStartResponse']

export type JobStartResponse = Omit<ApiJobStartResponse, 'scenario_totals'> & {
  scenario_totals: Record<string, number>
}

type ApiJobStatusResponse = components['schemas']['JobStatusResponse']

export interface JobStatusSnapshot extends Omit<
  ApiJobStatusResponse,
  'status' | 'scenario_totals' | 'progress' | 'stats'
> {
  status: JobStatus
  scenario_totals: Record<string, number>
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  lock_keys?: string[]
}

// 模板库状态（来自 /api/templates）
export interface TemplateInfo {
  scenario: string
  simulator?: string | null
  template_count: number
  file_size_bytes: number
  mtime: number
  path: string
}

// 文件管理
export interface TemplateFileInfo {
  scenario: string
  simulator?: string | null
  template_count: number
  file_size_bytes: number
  mtime: number
  path: string
}

// ── 模板浏览 ──────────────────────────────────────────────────────

export interface PlaceholderSlot {
  index: number
  param_name: string
  param_index: number
  use_transformed: boolean
  fmt: string
}

export interface OutputSlot {
  index: number
  name: string
}

export interface TransformDesc {
  param_name: string
  param_index: number
  transform_type: string | null
  factor: number | null
  note_en: string
  note_zh: string
}

export interface TemplateRecord {
  input_template: string
  target_template: string
  placeholder_schema: PlaceholderSlot[]
  output_schema: OutputSlot[]
  transform_descs: TransformDesc[]
  simulator: string
  scenario: string
  language: 'en' | 'zh'
  style: 'technical' | 'popular' | 'concise'
  time_mode: string
  n_time_points: number
  time_indices: number[]
  channel_indices: number[] | null
  selected_output_names: string[]
  timeseries_shape_orig: [number, number]
  timeseries_shape_obs: [number, number]
  param_names: string[]
}

export interface TemplatesResponse {
  total: number
  page: number
  page_size: number
  items: TemplateRecord[]
}

// ── 数据目录配置 ──────────────────────────────────────────────────

export interface LLMConfig {
  provider: string
  model: string
  base_url: string
  api_key_masked: string // 脱敏后的 key，仅展示用
  has_api_key: boolean
  temperature: number
  max_tokens: number
  thinking: 'enabled' | 'disabled'
}

export interface LLMConfigRequest {
  provider: string
  model: string
  api_key: string // 空=不修改
  base_url: string
  temperature: number
  max_tokens: number
  thinking: 'enabled' | 'disabled'
}

// ── 解析后的时序数据 ─────────────────────────────────────────────

export interface ParsedTimeseries {
  channels: number[][] // [n_channels][n_timesteps]
  labels: string[] // 每个通道的显示名
  unit: string // 单位
}

// ── 多智能体交互式注册 ────────────────────────────────────────────

export interface InterviewMessage {
  role: 'assistant' | 'user'
  content: string
  step: number
  ts: number
}

export interface InterviewCollectedData {
  domain_context: string
  output_description: string
  param_info: Record<string, [string, string]>
  output_info: unknown[]
  observation_config: Record<string, unknown>
}

export interface AgentTurnResponse {
  step: number
  step_label: string
  total_steps: number
  question: string | null
  extracted: Record<string, unknown> | null
  needs_confirmation: boolean
  extraction_uncertain: boolean
  done: boolean
  saved: boolean
  registry_key: string | null
  error: string | null
  hdf5_loaded: boolean
  github_prefilled: { steps: number[]; summary: string } | null
}

export interface InterviewStartRequest {
  simulator: string
  scenario: string
  hdf5_path?: string
  mode?: 'simulator' | 'scenario'
}

// ── Stage 1 仿真场景 ──────────────────────────────────────────────

export interface SimulationScenario {
  simulator: string
  scenario: string
  config_path: string
  h5_path: string | null
  sample_count: number
  output_shape: number[] | null
  file_size_bytes: number
}

export interface Hdf5ValidationResult {
  valid: boolean
  path: string
  file_size_bytes: number
  sample_count: number
  output_shape: number[] | null
  params_shape: number[] | null
  n_params: number
  param_names_preview: string[]
  attrs: Record<string, unknown>
  errors: string[]
  warnings: string[]
}

export interface Hdf5DataFileInfo extends Hdf5ValidationResult {
  simulator: string
  scenario: string
  mtime: number
}

export interface Hdf5UploadResponse {
  ok: boolean
  simulator: string
  scenario: string
  saved_path: string
  validation: Hdf5ValidationResult
}

export interface ExpertModelInfo {
  model_id: string
  name: string
  status?: 'active' | 'disabled' | 'invalid'
  runtime?: string
  file_name: string
  path: string
  created_at: number
  validated_at?: number | null
  file_size_bytes: number
  package_type?: string
  archive_path?: string
  entrypoint?: string
  callable?: string
  asset_count?: number
  asset_size_bytes?: number
  manifest?: Record<string, unknown>
  checksum?: string
  domain?: string
  simulator?: string
  input_dim?: number | null
  output_dim?: number | null
  assembly_enabled?: boolean
  data_generation_enabled?: boolean
  last_error?: string | null
  interface: string
  interface_version: number
  constraints_version?: number
  example_input?: number[]
  example_input_dim?: number
  smoke_output_dim?: number
  exists?: boolean
}

export interface ExpertModelMutationResponse {
  ok: boolean
  model: ExpertModelInfo
  error?: string | null
}

export interface ExpertModelContract {
  interface: string
  interface_version: number
  constraints: Record<string, string[]>
  example_source: string
  max_model_bytes: number
  max_input_dim: number
  max_input_points: number
  max_bundle_files?: number
  supported_upload_suffixes?: string[]
  manifest_name?: string
}

export interface ExpertModelListResponse extends ExpertModelContract {
  models: ExpertModelInfo[]
}

export interface ExpertModelUploadResponse extends ExpertModelContract {
  ok: boolean
  model: ExpertModelInfo
}

export interface ExpertInputPlanResponse {
  ok: boolean
  model_id: string
  plan: Record<string, unknown>
  preview: number[][]
  summary: string
  warnings: string[]
}

export interface ExpertGenerateResponse {
  ok: boolean
  model: ExpertModelInfo
  simulator: string
  scenario: string
  saved_path: string
  input_plan: Record<string, unknown>
  validation: Hdf5ValidationResult
}

export interface SimulateRequest {
  simulator: string
  scenario: string
  n_samples: number
  seed: number
  config_path: string
  skip_existing?: boolean
  parallel?: boolean
  max_workers?: number
}

export interface BatchSimulateRequest {
  scenarios: string[]
  n_samples: number
  seed: number
  skip_existing: boolean
  parallel: boolean
  max_workers: number
}

// ── Stage 4 Token Router ──────────────────────────────────────────

export interface RouterSplitInfo {
  exists: boolean
  count: number
  file_size_bytes: number
  mtime: number
}

export interface RouterScenarioInfo {
  scenario: string
  simulator: string
  source_count: number // Stage 3 样本数
  router_count?: number // Router 已生成条数（未生成时不存在）
  file_size_bytes?: number
  mtime?: number
}

export interface RouterStatus {
  splits: Record<'train', RouterSplitInfo>
  total: number
  label_counts: Record<string, number>
  scenarios: RouterScenarioInfo[]
  source_count: number
  source_by_scenario: Record<string, number> // key: simulator/scenario
  router_dir: string
}

export interface RouterSample {
  context: string
  label: 0 | 1 | '0' | '1'
  metadata: {
    simulator: string
    scenario: string
    language: string
  }
}

export interface RouterSamplesResponse {
  total: number
  page: number
  page_size: number
  items: RouterSample[]
}

export type TrainingDatasetScenario = components['schemas']['TrainingDatasetScenario']

export type TrainingDatasetInfo = components['schemas']['TrainingDatasetInfo']

export type TrainingGPUInfo = components['schemas']['GPUInfo']

export type TrainingJobStatus = components['schemas']['TrainingJobSummary']['status']

export type TrainingJobConfig = components['schemas']['TrainingJobConfig']

export type TrainingMetricsSummary = components['schemas']['TrainingMetricsSummary']

export type TrainingJobSummary = components['schemas']['TrainingJobSummary']

export type TrainingCheckpointInfo = components['schemas']['TrainingCheckpointInfo']

export type TrainingJobDetail = components['schemas']['TrainingJobDetail'] & {
  checkpoints: TrainingCheckpointInfo[]
}

export type TrainingOverview = components['schemas']['TrainingOverviewResponse']

export type TrainingCreateJobRequest = Omit<
  components['schemas']['TrainingJobCreateRequest'],
  | 'input_representation'
  | 'scenarios'
  | 'auto_stop_enabled'
  | 'auto_stop_metric'
  | 'auto_stop_threshold'
  | 'auto_stop_min_epochs'
> & {
  scenarios: string[]
  input_representation?: 'embedding'
  auto_stop_enabled?: boolean
  auto_stop_metric?: 'accuracy' | 'precision' | 'recall' | 'f1' | 'pr_auc'
  auto_stop_threshold?: number
  auto_stop_min_epochs?: number
}

export interface TrainingQuickJobCreateRequest {
  name?: string | null
  simulator: string
  scenarios: string[]
}

export type TrainingPoint = components['schemas']['TrainingPoint']

export type TrainingTestPoint = components['schemas']['TrainingTestPoint']

export type TrainingCurvesResponse = components['schemas']['TrainingCurvesResponse']

export type TrainingLogResponse = components['schemas']['TrainingLogResponse']
