const API_ROOT = '/api'

export type DatasetScenario = {
  scenario: string
  simulator: string
  router_count: number
}

export type TrainingDataset = {
  dataset_id: string | null
  display_name: string | null
  simulator: string
  total_count: number
  scenarios: DatasetScenario[]
}

export type SimulationPreset = {
  simulator: string
  scenario: string
  sample_count: number
  has_data: boolean
  output_shape: number[]
}

export type DataDefinition = {
  simulator: string
  scenario: string
  task_description: string
  parameters: Array<Record<string, unknown>>
  outputs: Array<Record<string, unknown>>
  sampling: Record<string, unknown>
}

export type WorkflowSnapshot = {
  workflow_id: string
  name: string
  status: 'draft' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  current_step: 'source' | 'definition' | 'generation' | 'complete'
  source?: {
    ready?: boolean
    sample_count?: number
    input_dim?: number
    output_shape?: number[]
    simulator?: string
    scenario?: string
    progress?: number
    message?: string
  } | null
  definition?: DataDefinition | null
  artifacts?: {
    progress?: number
    phase?: string
    message?: string
    router?: { dataset_id: string; sample_count: number }
    text2comp?: { dataset_id: string; sample_count: number }
  } | null
  error?: { code?: string; message?: string } | null
}

export type Metrics = {
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1?: number | null
  pr_auc?: number | null
}

export type Text2CompMetrics = {
  mse?: number | null
  mae?: number | null
  normalized_rmse?: number | null
  r2?: number | null
}

export type TrainingJob = {
  job_id: string
  name: string
  status:
    | 'queued'
    | 'starting'
    | 'running'
    | 'evaluating'
    | 'stopping'
    | 'done'
    | 'error'
    | 'terminated'
    | 'external_terminated'
  simulator: string
  scenarios: string[]
  gpu_id: number
  latest_epoch?: number | null
  latest_step?: number | null
  steps_per_epoch?: number | null
  eta_seconds?: number | null
  error_message?: string | null
  pipeline_stage?: string | null
  router_status?: string | null
  text2comp_status?: string | null
  text2comp_model_path?: string | null
  text2comp_error_message?: string | null
  router_metrics?: Metrics | null
  latest_metrics?: Metrics | null
  text2comp_metrics?: Text2CompMetrics | null
  config: {
    epochs?: number
    simple_text2comp_epochs?: number | null
  }
}

export type AssemblyProfile = {
  model_id: string
  name: string
  simulator: string
  source_job_id?: string | null
}

export type RegisterLoadResponse = {
  status: string
  profile: AssemblyProfile
  loaded: { status: string; message: string }
}

export type AssemblyTestResponse = {
  router_prediction: string
  router_class_name?: string
  first_cot_result: string
  final_answer?: string
  llm_response?: string
  expert_used?: boolean
  expert_output?: string
  latency_ms: number
}

type ErrorPayload = { detail?: string | { message?: string }; message?: string }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    credentials: 'same-origin',
    ...init,
    headers: {
      ...(typeof init?.body === 'string' ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  const payload = (await response.json().catch(() => null)) as ErrorPayload | null
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string' ? detail : detail?.message || payload?.message
    throw new Error(message || `请求失败 (${response.status})`)
  }
  return payload as T
}

export const workflowApi = {
  createSession: () => request<{ session_id: string }>('/new-synth/session', { method: 'POST' }),
  presets: () => request<{ simulations: SimulationPreset[] }>('/new-synth/presets'),
  simpleDatasets: () => request<TrainingDataset[]>('/training/simple-datasets'),
  createWorkflow: (name: string) =>
    request<WorkflowSnapshot>('/new-synth/workflows', { method: 'POST', body: JSON.stringify({ name }) }),
  workflow: (workflowId: string) => request<WorkflowSnapshot>(`/new-synth/workflows/${workflowId}`),
  uploadSource: (workflowId: string, file: File) =>
    request<WorkflowSnapshot>(`/new-synth/workflows/${workflowId}/source/upload`, {
      method: 'POST',
      headers: { 'X-File-Name': encodeURIComponent(file.name) },
      body: file,
    }),
  useSimulation: (workflowId: string, preset: SimulationPreset) =>
    request<WorkflowSnapshot>(`/new-synth/workflows/${workflowId}/source/simulation`, {
      method: 'POST',
      body: JSON.stringify({
        simulator: preset.simulator,
        scenario: preset.scenario,
        n_samples: Math.max(100, Math.min(1000, preset.sample_count)),
        seed: 42,
        reuse_existing: preset.has_data,
      }),
    }),
  saveDefinition: (workflowId: string, definition: DataDefinition) =>
    request<WorkflowSnapshot>(`/new-synth/workflows/${workflowId}/definition`, {
      method: 'PUT',
      body: JSON.stringify(definition),
    }),
  generate: (workflowId: string, maxSamples: number, variantsPerSample: number) =>
    request<{ workflow_id: string; status: string; message: string }>(`/new-synth/workflows/${workflowId}/generate`, {
      method: 'POST',
      body: JSON.stringify({
        max_samples: maxSamples,
        variants_per_sample: variantsPerSample,
        negative_ratio: 1,
        seed: 42,
      }),
    }),
  createQuickJob: (dataset: TrainingDataset) =>
    request<TrainingJob>('/training/quick-jobs', {
      method: 'POST',
      body: JSON.stringify({
        dataset_id: dataset.dataset_id,
        name: `${dataset.display_name || dataset.simulator} · 对话工作流训练`,
        simulator: dataset.simulator,
        scenarios: dataset.scenarios.map(item => item.scenario),
        gpu_id: null,
        resume_from: null,
        seed: 42,
      }),
    }),
  trainingJob: (jobId: string) => request<TrainingJob>(`/training/jobs/${encodeURIComponent(jobId)}`),
  registerAndLoad: (jobId: string) =>
    request<RegisterLoadResponse>(`/training/jobs/${encodeURIComponent(jobId)}/register-load`, {
      method: 'POST',
      body: JSON.stringify({ gpu_id: null }),
    }),
  testAssembly: (profileId: string, testInput: string) =>
    request<AssemblyTestResponse>('/assembly/test', {
      method: 'POST',
      body: JSON.stringify({
        config: { assembly_profile_id: profileId },
        test_input: testInput,
      }),
    }),
}

export function wait(milliseconds: number): Promise<void> {
  return new Promise(resolve => window.setTimeout(resolve, milliseconds))
}

export async function waitForWorkflow(
  workflowId: string,
  onUpdate: (snapshot: WorkflowSnapshot) => void,
): Promise<WorkflowSnapshot> {
  for (;;) {
    const snapshot = await workflowApi.workflow(workflowId)
    onUpdate(snapshot)
    if (snapshot.status === 'failed' || snapshot.status === 'cancelled') {
      throw new Error(snapshot.error?.message || '数据准备任务未完成')
    }
    if (snapshot.status !== 'running') return snapshot
    await wait(1000)
  }
}
