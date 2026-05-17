import type {
  DatasetInfo,
  SamplesResponse,
  DatasetStats,
  DashboardSummary,
  ScenariosConfig,
  GenerationConfig,
  Text2CompScenariosConfig,
  RegisterRequest,
  AgentTurnResponse,
  InterviewStartRequest,
  InterviewState,
  GenerateTemplatesRequest,
  FillSamplesRequest,
  TemplateInfo,
  LLMConfig,
  LLMConfigRequest,
  TemplateFileInfo,
  SampleFileInfo,
  JobStartResponse,
  JobStatusSnapshot,
  TemplatesResponse,
  SimulationScenario,
  SimulateRequest,
  BatchSimulateRequest,
  SimulationHistoryRecord,
  Hdf5DataFileInfo,
  Hdf5UploadResponse,
  FileCatalogResponse,
  FileCatalogMutationResponse,
  RouterStatus,
  RouterSamplesResponse,
  TrainingOverview,
  TrainingDatasetInfo,
  TrainingGPUInfo,
  TrainingJobSummary,
  TrainingCreateJobRequest,
  TrainingJobDetail,
  TrainingCurvesResponse,
  TrainingLogResponse,
} from './types'

const BASE = '/api'

export const PIERN_AUTH_TOKEN_KEY = 'piern-auth-token'

export function readAuthToken(): string {
  try {
    return localStorage.getItem(PIERN_AUTH_TOKEN_KEY) || ''
  } catch {
    return ''
  }
}

export function writeAuthToken(value: string): string {
  const token = value.trim()
  try {
    if (token) localStorage.setItem(PIERN_AUTH_TOKEN_KEY, token)
    else localStorage.removeItem(PIERN_AUTH_TOKEN_KEY)
  } catch {
    /* ignore */
  }
  return token
}

function withAuthHeaders(init?: RequestInit): RequestInit {
  const token = readAuthToken()
  if (!token) return init ?? {}
  const headers = new Headers(init?.headers)
  headers.set('Authorization', `Bearer ${token}`)
  return { ...(init ?? {}), headers }
}

function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return window.fetch(input, withAuthHeaders(init))
}

export interface ApiErrorPayload {
  code: string
  message: string
  details?: Record<string, unknown>
  request_id?: string | null
}

export class ApiRequestError extends Error {
  status: number
  code: string
  details: Record<string, unknown>
  requestId: string | null

  constructor(
    message: string,
    args: { status: number; code: string; details?: Record<string, unknown>; requestId?: string | null },
  ) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = args.status
    this.code = args.code
    this.details = args.details ?? {}
    this.requestId = args.requestId ?? null
  }
}

function isApiErrorPayload(value: unknown): value is ApiErrorPayload {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return typeof item.code === 'string' && typeof item.message === 'string'
}

function errorMessageFromPayload(payload: unknown): ApiErrorPayload | null {
  if (isApiErrorPayload(payload)) return payload
  if (payload && typeof payload === 'object') {
    const detail = (payload as Record<string, unknown>).detail
    if (isApiErrorPayload(detail)) return detail
    if (typeof detail === 'string') {
      return { code: 'HTTP_ERROR', message: detail, details: {} }
    }
  }
  return null
}

async function parseErrorResponse(res: Response, fallback: string): Promise<ApiRequestError> {
  const requestId = res.headers.get('x-request-id')
  const contentType = res.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    const payload = await res.json().catch(() => null)
    const parsed = errorMessageFromPayload(payload)
    if (parsed) {
      return new ApiRequestError(`${fallback} (${res.status}): ${parsed.message}`, {
        status: res.status,
        code: parsed.code,
        details: parsed.details,
        requestId: parsed.request_id ?? requestId,
      })
    }
  }
  const text = await res.text().catch(() => '')
  return new ApiRequestError(`${fallback} (${res.status})${text ? `: ${text}` : ''}`, {
    status: res.status,
    code: res.status >= 500 ? 'INTERNAL_ERROR' : 'HTTP_ERROR',
    details: {},
    requestId,
  })
}

async function ensureOk(res: Response, fallback: string): Promise<void> {
  if (!res.ok) throw await parseErrorResponse(res, fallback)
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await apiFetch(url.toString())
  await ensureOk(res, `API ${path} 失败`)
  return res.json()
}

export const api = {
  // ── 数据集 ──────────────────────────────────────────────────────
  getDatasets: (): Promise<DatasetInfo[]> => get('/datasets'),

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

  getStats: (): Promise<DatasetStats> => get('/stats'),

  getDashboardSummary: (): Promise<DashboardSummary> => get('/dashboard/summary'),

  // ── 配置 ────────────────────────────────────────────────────────
  getConfig: (): Promise<GenerationConfig> => get('/config'),

  getLLMConfig: (): Promise<LLMConfig> => get('/llm-config'),

  saveLLMConfig: async (req: LLMConfigRequest): Promise<void> => {
    const res = await apiFetch(`${BASE}/llm-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '保存失败')
  },

  testLLMConfig: async (req: LLMConfigRequest): Promise<{ ok: boolean; message: string; response_preview: string }> => {
    const res = await apiFetch(`${BASE}/llm-config/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '测试请求失败')
    return res.json()
  },

  getScenarios: (): Promise<ScenariosConfig> => get('/config/scenarios'),

  getText2CompScenarios: (): Promise<Text2CompScenariosConfig> => get('/config/text2comp-scenarios'),

  // ── 生成任务（SSE 流 + 终止）──────────────────────────────────
  stopGeneration: async (jobId: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/generate/${jobId}`, { method: 'DELETE' })
    if (!res.ok && res.status !== 404) await ensureOk(res, '终止失败')
  },

  openGenerationStream: (jobId: string): EventSource => {
    return new EventSource(`${BASE}/generate/${jobId}/stream`)
  },

  getGenerationStatus: (jobId: string): Promise<JobStatusSnapshot> =>
    get(`/generate/${encodeURIComponent(jobId)}/status`),

  listGenerationJobs: (filters?: { job_type?: string; status?: string }): Promise<JobStatusSnapshot[]> => {
    const params: Record<string, string> = {}
    if (filters?.job_type) params.job_type = filters.job_type
    if (filters?.status) params.status = filters.status
    return get('/generate/jobs', params)
  },

  // ── 注册 ────────────────────────────────────────────────────────
  getRegistry: (): Promise<Record<string, unknown>> => get('/registry'),

  updateRegistryEntry: async (key: string, body: Record<string, unknown>): Promise<void> => {
    const res = await apiFetch(`${BASE}/registry/${encodeURIComponent(key)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    await ensureOk(res, '保存失败')
  },

  deleteRegistryEntry: async (key: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/registry/${encodeURIComponent(key)}`, { method: 'DELETE' })
    await ensureOk(res, '删除失败')
  },

  // ── 多智能体交互式注册 ─────────────────────────────────────────
  startInterview: async (req: InterviewStartRequest): Promise<AgentTurnResponse & { session_id: string }> => {
    const res = await apiFetch(`${BASE}/interview/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动面试失败')
    return res.json()
  },

  sendInterviewMessage: async (sessionId: string, message: string): Promise<AgentTurnResponse> => {
    const res = await apiFetch(`${BASE}/interview/${sessionId}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })
    await ensureOk(res, '消息发送失败')
    return res.json()
  },

  confirmInterviewStep: async (
    sessionId: string,
    confirmed: boolean,
    editedData?: Record<string, unknown>,
  ): Promise<AgentTurnResponse> => {
    const res = await apiFetch(`${BASE}/interview/${sessionId}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmed, edited_data: editedData ?? null }),
    })
    await ensureOk(res, '确认失败')
    return res.json()
  },

  getInterviewState: (sessionId: string): Promise<InterviewState> => get(`/interview/${sessionId}/state`),

  cancelInterview: async (sessionId: string): Promise<void> => {
    await apiFetch(`${BASE}/interview/${sessionId}`, { method: 'DELETE' })
  },

  // ── 两阶段生成 ──────────────────────────────────────────────────
  getTemplatesStatus: (): Promise<TemplateInfo[]> => get('/templates'),

  startGenerateTemplates: async (req: GenerateTemplatesRequest): Promise<JobStartResponse> => {
    const res = await apiFetch(`${BASE}/generate-templates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动模板生成失败')
    return res.json()
  },

  startFillSamples: async (req: FillSamplesRequest): Promise<JobStartResponse> => {
    const res = await apiFetch(`${BASE}/fill-samples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动样本填充失败')
    return res.json()
  },

  // ── 文件管理 ────────────────────────────────────────────────────
  listTemplateFiles: (): Promise<TemplateFileInfo[]> => get('/files/templates'),

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

  listSampleFiles: (): Promise<SampleFileInfo[]> => get('/files/samples'),

  trimTemplateFile: async (scenario: string, n: number): Promise<{ before: number; after: number }> => {
    const res = await apiFetch(`${BASE}/files/templates/${encodeURIComponent(scenario)}/trim?n=${n}`, {
      method: 'POST',
    })
    await ensureOk(res, '截断失败')
    return res.json()
  },

  deleteTemplateFile: async (scenario: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/files/templates/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    await ensureOk(res, '删除失败')
  },

  deleteSampleFile: async (scenario: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/files/samples/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    await ensureOk(res, '删除失败')
  },

  clearAllTemplates: async (): Promise<void> => {
    const res = await apiFetch(`${BASE}/files/templates`, { method: 'DELETE' })
    await ensureOk(res, '清空失败')
  },

  clearAllSamples: async (): Promise<void> => {
    const res = await apiFetch(`${BASE}/files/samples`, { method: 'DELETE' })
    await ensureOk(res, '清空失败')
  },

  // ── Stage 1 物理仿真 ─────────────────────────────────────────────
  // Unified file catalog
  getFileCatalog: (): Promise<FileCatalogResponse> => get('/files/catalog'),

  deleteFileCatalogAsset: async (assetId: string): Promise<FileCatalogMutationResponse> => {
    const res = await apiFetch(`${BASE}/files/catalog/assets/${encodeURIComponent(assetId)}`, { method: 'DELETE' })
    await ensureOk(res, '删除文件资产失败')
    return res.json()
  },

  clearFileCatalogGroup: async (kind: 'templates' | 'samples' | 'router'): Promise<FileCatalogMutationResponse> => {
    const res = await apiFetch(`${BASE}/files/catalog/groups/${encodeURIComponent(kind)}`, { method: 'DELETE' })
    await ensureOk(res, '清空文件分组失败')
    return res.json()
  },

  rebuildFileCatalogIndexes: async (
    scope: 'all' | 'templates' | 'samples' | 'router' = 'all',
  ): Promise<FileCatalogMutationResponse> => {
    const res = await apiFetch(`${BASE}/files/catalog/rebuild?scope=${encodeURIComponent(scope)}`, { method: 'POST' })
    await ensureOk(res, '重建文件索引失败')
    return res.json()
  },

  getSimulationScenarios: (refresh = false): Promise<SimulationScenario[]> =>
    get('/simulation/scenarios', refresh ? { refresh: 1 } : undefined),

  listHdf5DataFiles: (): Promise<Hdf5DataFileInfo[]> => get('/simulation/data-files'),

  uploadHdf5Data: async (args: {
    simulator: string
    scenario: string
    file: File
    overwrite: boolean
  }): Promise<Hdf5UploadResponse> => {
    const url = new URL(`${BASE}/simulation/upload`, window.location.origin)
    url.searchParams.set('simulator', args.simulator)
    url.searchParams.set('scenario', args.scenario)
    url.searchParams.set('overwrite', String(args.overwrite))
    const res = await apiFetch(url.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: args.file,
    })
    await ensureOk(res, '上传失败')
    return res.json()
  },

  startSimulation: async (req: SimulateRequest): Promise<JobStartResponse> => {
    const res = await apiFetch(`${BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动仿真失败')
    return res.json()
  },

  startBatchSimulation: async (req: BatchSimulateRequest): Promise<JobStartResponse> => {
    const res = await apiFetch(`${BASE}/simulate/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动批量仿真失败')
    return res.json()
  },

  getSimulationHistory: (limit = 50): Promise<SimulationHistoryRecord[]> => get('/simulation/history', { limit }),

  clearSimulationHistory: async (): Promise<void> => {
    const res = await apiFetch(`${BASE}/simulation/history`, { method: 'DELETE' })
    await ensureOk(res, '清空历史失败')
  },

  // ── Stage 4 Token Router ─────────────────────────────────────────
  getRouterStatus: (): Promise<RouterStatus> => get('/router/status'),

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
    const res = await apiFetch(`${BASE}/router/build?${params}`, { method: 'POST' })
    await ensureOk(res, '启动路由数据生成失败')
    return res.json()
  },

  deleteRouterScenario: async (scenario: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/router/scenario/${encodeURIComponent(scenario)}`, { method: 'DELETE' })
    await ensureOk(res, '删除失败')
  },

  deleteAllRouterData: async (): Promise<void> => {
    const res = await apiFetch(`${BASE}/router/all`, { method: 'DELETE' })
    await ensureOk(res, '清空失败')
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
    const res = await apiFetch(`${BASE}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动注册失败')
    return res.json()
  },

  // ── 训练平台 ─────────────────────────────────────────────────────
  getTrainingOverview: (): Promise<TrainingOverview> => get('/training/overview'),

  getTrainingDatasets: (): Promise<TrainingDatasetInfo[]> => get('/training/datasets'),

  getTrainingGPUs: (): Promise<TrainingGPUInfo[]> => get('/training/gpus'),

  getTrainingJobs: (): Promise<TrainingJobSummary[]> => get('/training/jobs'),

  createTrainingJob: async (req: TrainingCreateJobRequest): Promise<TrainingJobSummary> => {
    const res = await apiFetch(`${BASE}/training/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    await ensureOk(res, '启动训练失败')
    return res.json()
  },

  getTrainingJob: (jobId: string): Promise<TrainingJobDetail> => get(`/training/jobs/${encodeURIComponent(jobId)}`),

  stopTrainingJob: async (jobId: string): Promise<TrainingJobSummary> => {
    const res = await apiFetch(`${BASE}/training/jobs/${encodeURIComponent(jobId)}/stop`, {
      method: 'POST',
    })
    await ensureOk(res, '终止训练失败')
    return res.json()
  },

  deleteTrainingJob: async (jobId: string): Promise<void> => {
    const res = await apiFetch(`${BASE}/training/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    })
    await ensureOk(res, '删除训练任务失败')
  },

  getTrainingCurves: (jobId: string, maxPoints = 2000): Promise<TrainingCurvesResponse> =>
    get(`/training/jobs/${encodeURIComponent(jobId)}/curves`, { max_points: maxPoints }),

  getTrainingLogs: (jobId: string, limit = 300): Promise<TrainingLogResponse> =>
    get(`/training/jobs/${encodeURIComponent(jobId)}/logs`, { limit }),
}
