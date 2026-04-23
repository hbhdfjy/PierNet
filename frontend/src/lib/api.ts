import type {
  DatasetInfo, SamplesResponse, DatasetStats, DashboardSummary,
  ScenariosConfig, GenerationConfig,
  Text2CompScenariosConfig, RegisterRequest,
  AgentTurnResponse, InterviewStartRequest, InterviewState,
  GenerateTemplatesRequest, FillSamplesRequest, TemplateInfo,
  LLMConfig, LLMConfigRequest,
  TemplateFileInfo, SampleFileInfo, JobStartResponse,
  TemplatesResponse,
  SimulationScenario, SimulateRequest, BatchSimulateRequest, SimulationHistoryRecord,
  RouterStatus, RouterSamplesResponse,
  TrainingOverview, TrainingDatasetInfo, TrainingGPUInfo, TrainingJobSummary,
  TrainingCreateJobRequest, TrainingJobDetail, TrainingCurvesResponse, TrainingLogResponse,
} from './types'

const BASE = '/api'

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${path} 失败 (${res.status}): ${text}`)
  }
  return res.json()
}

export const api = {
  // ── 数据集 ──────────────────────────────────────────────────────
  getDatasets: (): Promise<DatasetInfo[]> =>
    get('/datasets'),

  getSamples: (
    scenario: string,
    page = 0,
    pageSize = 20,
    language?: string,
    style?: string,
  ): Promise<SamplesResponse> => {
    const p: Record<string, string | number> = { scenario, page, page_size: pageSize }
    if (language) p.language = language
    if (style) p.style = style
    return get('/samples', p)
  },

  getStats: (): Promise<DatasetStats> =>
    get('/stats'),

  getDashboardSummary: (): Promise<DashboardSummary> =>
    get('/dashboard/summary'),

  // ── 配置 ────────────────────────────────────────────────────────
  getConfig: (): Promise<GenerationConfig> =>
    get('/config'),

  getLLMConfig: (): Promise<LLMConfig> =>
    get('/llm-config'),

  saveLLMConfig: async (req: LLMConfigRequest): Promise<void> => {
    const res = await fetch(`${BASE}/llm-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`保存失败 (${res.status}): ${text}`)
    }
  },

  testLLMConfig: async (req: LLMConfigRequest): Promise<{ ok: boolean; message: string; response_preview: string }> => {
    const res = await fetch(`${BASE}/llm-config/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`测试请求失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  getScenarios: (): Promise<ScenariosConfig> =>
    get('/config/scenarios'),

  getText2CompScenarios: (): Promise<Text2CompScenariosConfig> =>
    get('/config/text2comp-scenarios'),

  // ── 生成任务（SSE 流 + 终止）──────────────────────────────────
  stopGeneration: async (jobId: string): Promise<void> => {
    const res = await fetch(`${BASE}/generate/${jobId}`, { method: 'DELETE' })
    if (!res.ok && res.status !== 404) {
      throw new Error(`终止失败 (${res.status})`)
    }
  },

  openGenerationStream: (jobId: string): EventSource => {
    return new EventSource(`${BASE}/generate/${jobId}/stream`)
  },

  // ── 注册 ────────────────────────────────────────────────────────
  getRegistry: (): Promise<Record<string, unknown>> =>
    get('/registry'),

  updateRegistryEntry: async (key: string, body: Record<string, unknown>): Promise<void> => {
    const res = await fetch(`${BASE}/registry/${encodeURIComponent(key)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(`保存失败 (${res.status})`)
  },

  deleteRegistryEntry: async (key: string): Promise<void> => {
    const res = await fetch(`${BASE}/registry/${encodeURIComponent(key)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`删除失败 (${res.status})`)
  },

  // ── 多智能体交互式注册 ─────────────────────────────────────────
  startInterview: async (req: InterviewStartRequest): Promise<AgentTurnResponse & { session_id: string }> => {
    const res = await fetch(`${BASE}/interview/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动面试失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  sendInterviewMessage: async (sessionId: string, message: string): Promise<AgentTurnResponse> => {
    const res = await fetch(`${BASE}/interview/${sessionId}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`消息发送失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  confirmInterviewStep: async (
    sessionId: string,
    confirmed: boolean,
    editedData?: Record<string, unknown>,
  ): Promise<AgentTurnResponse> => {
    const res = await fetch(`${BASE}/interview/${sessionId}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmed, edited_data: editedData ?? null }),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`确认失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  getInterviewState: (sessionId: string): Promise<InterviewState> =>
    get(`/interview/${sessionId}/state`),

  cancelInterview: async (sessionId: string): Promise<void> => {
    await fetch(`${BASE}/interview/${sessionId}`, { method: 'DELETE' })
  },

  // ── 两阶段生成 ──────────────────────────────────────────────────
  getTemplatesStatus: (): Promise<TemplateInfo[]> =>
    get('/templates'),

  startGenerateTemplates: async (req: GenerateTemplatesRequest): Promise<JobStartResponse> => {
    const res = await fetch(`${BASE}/generate-templates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动模板生成失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  startFillSamples: async (req: FillSamplesRequest): Promise<JobStartResponse> => {
    const res = await fetch(`${BASE}/fill-samples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动样本填充失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  // ── 文件管理 ────────────────────────────────────────────────────
  listTemplateFiles: (): Promise<TemplateFileInfo[]> =>
    get('/files/templates'),

  getTemplateItems: (
    scenario: string,
    page = 0,
    pageSize = 20,
    language?: string,
    style?: string,
  ): Promise<TemplatesResponse> => {
    const p: Record<string, string | number> = { page, page_size: pageSize }
    if (language) p.language = language
    if (style) p.style = style
    return get(`/files/templates/${encodeURIComponent(scenario)}/items`, p)
  },

  listSampleFiles: (): Promise<SampleFileInfo[]> =>
    get('/files/samples'),

  trimTemplateFile: async (scenario: string, n: number): Promise<{ before: number; after: number }> => {
    const res = await fetch(`${BASE}/files/templates/${encodeURIComponent(scenario)}/trim?n=${n}`, { method: 'POST' })
    if (!res.ok) throw new Error(`截断失败 (${res.status})`)
    return res.json()
  },

  deleteTemplateFile: async (scenario: string): Promise<void> => {
    const res = await fetch(`${BASE}/files/templates/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`删除失败 (${res.status})`)
  },

  deleteSampleFile: async (scenario: string): Promise<void> => {
    const res = await fetch(`${BASE}/files/samples/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`删除失败 (${res.status})`)
  },

  clearAllTemplates: async (): Promise<void> => {
    const res = await fetch(`${BASE}/files/templates`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`清空失败 (${res.status})`)
  },

  clearAllSamples: async (): Promise<void> => {
    const res = await fetch(`${BASE}/files/samples`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`清空失败 (${res.status})`)
  },

  // ── Stage 1 物理仿真 ─────────────────────────────────────────────
  getSimulationScenarios: (refresh = false): Promise<SimulationScenario[]> =>
    get('/simulation/scenarios', refresh ? { refresh: 1 } : undefined),

  startSimulation: async (req: SimulateRequest): Promise<JobStartResponse> => {
    const res = await fetch(`${BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动仿真失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  startBatchSimulation: async (req: BatchSimulateRequest): Promise<JobStartResponse> => {
    const res = await fetch(`${BASE}/simulate/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动批量仿真失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  getSimulationHistory: (limit = 50): Promise<SimulationHistoryRecord[]> =>
    get('/simulation/history', { limit }),

  clearSimulationHistory: async (): Promise<void> => {
    const res = await fetch(`${BASE}/simulation/history`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`清空历史失败 (${res.status})`)
  },

  // ── Stage 4 Token Router ─────────────────────────────────────────
  getRouterStatus: (): Promise<RouterStatus> =>
    get('/router/status'),

  buildRouterData: async (
    seed = 42,
    scenarios: string[] = [],
    negRatio = 1,
  ): Promise<{ job_id: string; status: string }> => {
    const params = new URLSearchParams({
      seed: String(seed),
      neg_ratio: String(negRatio),
    })
    if (scenarios.length > 0) params.set('scenarios', scenarios.join(','))
    const res = await fetch(`${BASE}/router/build?${params}`, { method: 'POST' })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动路由数据生成失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  deleteRouterScenario: async (scenario: string): Promise<void> => {
    const res = await fetch(`${BASE}/router/scenario/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`删除失败 (${res.status})`)
  },

  deleteAllRouterData: async (): Promise<void> => {
    const res = await fetch(`${BASE}/router/all`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`清空失败 (${res.status})`)
  },

  getRouterSamples: (
    split: 'train' = 'train',
    page = 0,
    pageSize = 20,
    label = -1,
    scenario = '',
  ): Promise<RouterSamplesResponse> => {
    const p: Record<string, string | number> = { split, page, page_size: pageSize, label }
    if (scenario) p.scenario = scenario
    return get('/router/samples', p)
  },

  startRegister: async (req: RegisterRequest): Promise<{ job_id: string }> => {
    const res = await fetch(`${BASE}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动注册失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  // ── 训练平台 ─────────────────────────────────────────────────────
  getTrainingOverview: (): Promise<TrainingOverview> =>
    get('/training/overview'),

  getTrainingDatasets: (): Promise<TrainingDatasetInfo[]> =>
    get('/training/datasets'),

  getTrainingGPUs: (): Promise<TrainingGPUInfo[]> =>
    get('/training/gpus'),

  getTrainingJobs: (): Promise<TrainingJobSummary[]> =>
    get('/training/jobs'),

  createTrainingJob: async (req: TrainingCreateJobRequest): Promise<TrainingJobSummary> => {
    const res = await fetch(`${BASE}/training/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`启动训练失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  getTrainingJob: (jobId: string): Promise<TrainingJobDetail> =>
    get(`/training/jobs/${encodeURIComponent(jobId)}`),

  stopTrainingJob: async (jobId: string): Promise<TrainingJobSummary> => {
    const res = await fetch(`${BASE}/training/jobs/${encodeURIComponent(jobId)}/stop`, {
      method: 'POST',
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`终止训练失败 (${res.status}): ${text}`)
    }
    return res.json()
  },

  deleteTrainingJob: async (jobId: string): Promise<void> => {
    const res = await fetch(`${BASE}/training/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`删除训练任务失败 (${res.status}): ${text}`)
    }
  },

  getTrainingCurves: (jobId: string, maxPoints = 2000): Promise<TrainingCurvesResponse> =>
    get(`/training/jobs/${encodeURIComponent(jobId)}/curves`, { max_points: maxPoints }),

  getTrainingLogs: (jobId: string, limit = 300): Promise<TrainingLogResponse> =>
    get(`/training/jobs/${encodeURIComponent(jobId)}/logs`, { limit }),
}
