// ── Stage 2 JSONL 样本类型（与 Python 端完全对应）──────────────────

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

export interface ScenarioConfig {
  name: string
  scenario: string
  output_file: string
  n_samples: number
}

export interface ScenariosConfig {
  [simulator: string]: ScenarioConfig[]
}

// Stage 2 可用场景（来自 HDF5 文件扫描）
export interface Text2CompScenario {
  name: string
  simulator: string
  h5_file: string | null
  sample_count: number           // HDF5 中的样本数（无 HDF5 时为 0）
  output_shape: number[] | null  // HDF5 timeseries 的输出维度，如 [43, 365]
  existing_jsonl_count: number   // 已生成的 JSONL 条数
  has_jsonl: boolean
  has_h5: boolean                // 是否有 HDF5 数据文件
  registered: boolean            // 是否在 registry.yaml 中有注册信息
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

export type JobStatus = 'running' | 'done' | 'error' | 'terminated' | 'idle'

export interface GenerationJob {
  job_id: string
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
}

export interface RegisterRequest {
  scenarios: string[]          // 空=全部
  fields: string[]             // 空=全部
  overwrite: boolean
  simulator_level: boolean
  config: string
  output: string
}

// ── 两阶段生成请求 ────────────────────────────────────────────────

export interface GenerateTemplatesRequest {
  scenarios: string[]
  n_templates: number
  skip_existing: boolean
  append_existing: boolean
  config: string
  language_mix?: number
  transform_prob?: number
  max_workers?: number
}

export interface FillSamplesRequest {
  scenarios: string[]
  n_samples: number
  templates_dir: string   // 空=默认 data/templates/
  output_dir: string      // 空=默认 data/text2comp/
  skip_existing: boolean
  config: string
  seed?: number
}

export interface JobStartResponse {
  job_id: string
  status: string
  scenario_totals: Record<string, number>
}

// 模板库状态（来自 /api/templates）
export interface TemplateInfo {
  scenario: string
  template_count: number
  file_size_bytes: number
  mtime: number
  path: string
}

// 文件管理
export interface TemplateFileInfo {
  scenario: string
  template_count: number
  file_size_bytes: number
  mtime: number
  path: string
}

export interface SampleFileInfo {
  scenario: string
  sample_count: number
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
  slice_start: number
  slice_end: number | null
  row_level: boolean
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
  channel_level: string
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

export interface DataDirEntry {
  key: string
  path: string
  simulator: string
  file_suffix: string
  transient_simulator: string
  transient_keywords: string[]
}

export interface LLMConfig {
  provider: string
  model: string
  base_url: string
  api_key_masked: string   // 脱敏后的 key，仅展示用
  has_api_key: boolean
  temperature: number
  max_tokens: number
}

export interface LLMConfigRequest {
  provider: string
  model: string
  api_key: string          // 空=不修改
  base_url: string
  temperature: number
  max_tokens: number
}

// ── 解析后的时序数据 ─────────────────────────────────────────────

export interface ParsedTimeseries {
  channels: number[][]     // [n_channels][n_timesteps]
  labels: string[]         // 每个通道的显示名
  unit: string             // 单位
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

export interface InterviewState {
  session_id: string
  simulator: string
  scenario: string
  step: number
  status: 'interviewing' | 'confirming' | 'done' | 'error'
  history: InterviewMessage[]
  collected_data: InterviewCollectedData
  pending_extraction: Record<string, unknown> | null
  hdf5_loaded: boolean
  timeseries_shape: number[] | null
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
  target_samples: number
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

export interface SimulationHistoryRecord {
  job_id: string
  simulator: string
  scenario: string
  n_samples: number
  skip_existing: boolean
  started_at: number
  finished_at: number | null
  status: string
  elapsed_sec: number | null
  final_sample_count: number | null
}
