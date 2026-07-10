import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Cpu,
  Layers3,
  Loader2,
  PlayCircle,
  Power,
  PowerOff,
  RefreshCw,
  Route,
} from 'lucide-react'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import type { ExpertModelInfo, TrainingJobSummary } from '../../lib/types'
import { TrainingSectionTitle as SectionTitle } from '../components/common'

type AssemblyStatus = Awaited<ReturnType<typeof api.getAssemblyStatus>>
type ExecutorMode = 'fno' | 'uploaded'
type AssemblyMode = 'profile' | 'training_job'

type AssemblyProfile = NonNullable<AssemblyStatus['assembly_profiles']>[number]

type TrainingBundle = {
  id: string
  label: string
  createdAt: string
  task: string
  llm: string
  router: string
  text2comp: string
  fno: string
}

const DIFF_SORP_TEST_INPUT =
  '请通过分析两帧归一化输入数据，基于传质动力学的理论框架，预测吸附系统在下一时间步的状态演变。\n[0.99644, 0.93487, 0.86565, 0.79567, 0.72738, 0.66152, 0.59858, 0.53898, 0.48310, 0.43119]'

const TRAINING_BUNDLES: readonly TrainingBundle[] = []

const TRAINING_BUNDLE_PATHS: Record<
  string,
  { llm: string; router: string; text2comp: string; fno: string; text2compOutputDim: number; fnoInputDim: number }
> = {}

type SelectableModel = {
  name: string
  path?: string
  trained?: boolean
  output_dim?: number | null
  input_dim?: number | null
  model_id?: string
  description?: string
}

type TrainingAssemblySource = {
  id: string
  kind: 'bundle' | 'job'
  name: string
  label: string
  createdAt: string
  simulator: string
  scenarios: string[]
  llm: SelectableModel | null
  router: SelectableModel | null
  text2comp: SelectableModel | null
  fno: SelectableModel | null
  uploadedExpertName?: string | null
  uploadedExpertInputDim?: number | null
  defaultPrompt?: string
  sourceJob?: TrainingJobSummary
}

function pickFirst<T extends SelectableModel>(items: T[] | undefined, predicate?: (item: T) => boolean): T | null {
  const list = items ?? []
  return list.find(item => predicate?.(item)) ?? list.find(item => item.trained) ?? list[0] ?? null
}

function isUsableTrainingJob(job: TrainingJobSummary): boolean {
  return Boolean(
    job.config?.simple_pipeline_enabled === true && job.run_dir && job.status === 'done' && job.text2comp_model_path,
  )
}

function routerPathFromTrainingJob(job: TrainingJobSummary): string {
  const suffix = job.status === 'done' ? 'router_final.pt' : 'router_latest.pt'
  return `${job.run_dir.replace(/\/$/, '')}/${suffix}`
}

function routerModelFromTrainingJob(job: TrainingJobSummary | null): SelectableModel | null {
  if (!job) return null
  const metric = job.latest_metrics?.f1 != null ? ` · F1 ${job.latest_metrics.f1.toFixed(4)}` : ''
  return {
    name: `${job.name || job.job_id} Router`,
    path: routerPathFromTrainingJob(job),
    trained: true,
    description: `${job.simulator.toUpperCase()} · ${job.scenarios.length} 个场景${metric}`,
  }
}

function text2compModelFromTrainingJob(job: TrainingJobSummary | null): SelectableModel | null {
  if (!job?.text2comp_model_path) return null
  const outputDim = job.text2comp_output_dim ?? job.uploaded_expert_input_dim
  const source = job.text2comp_target_source ? ` · 数据字段 ${job.text2comp_target_source}` : ''
  return {
    name: `${job.name || job.job_id} Text2Comp`,
    path: job.text2comp_model_path,
    trained: true,
    output_dim: outputDim,
    description: `输出维度 ${outputDim ?? '--'}${source}`,
  }
}

function findScannedModel<T extends SelectableModel>(
  items: T[] | undefined,
  name: string,
  path: string,
): T | SelectableModel {
  const list = items ?? []
  return (
    list.find(item => item.path === path) ??
    list.find(item => item.name === name) ??
    list.find(item => item.name.includes(name) || path.includes(item.name)) ?? {
      name,
      path,
      trained: true,
    }
  )
}

function trainingSourceFromBundle(bundle: TrainingBundle, status: AssemblyStatus | undefined): TrainingAssemblySource {
  const paths = TRAINING_BUNDLE_PATHS[bundle.id]
  return {
    id: `bundle:${bundle.id}`,
    kind: 'bundle',
    name: bundle.label,
    label: bundle.label,
    createdAt: bundle.createdAt,
    simulator: bundle.task,
    scenarios: [bundle.task],
    llm: findScannedModel(status?.llms, bundle.llm, paths.llm),
    router: findScannedModel(status?.routers, bundle.router, paths.router),
    text2comp: {
      ...findScannedModel(status?.text2comps, bundle.text2comp, paths.text2comp),
      output_dim: paths.text2compOutputDim,
      description: `输出维度 ${paths.text2compOutputDim}`,
    },
    fno: {
      ...findScannedModel(status?.fno_experts, bundle.fno, paths.fno),
      input_dim: paths.fnoInputDim,
      description: `输入维度 ${paths.fnoInputDim} · FNO Expert`,
    },
    defaultPrompt: bundle.id === 'diff-sorp-20260103-0124' ? DIFF_SORP_TEST_INPUT : undefined,
  }
}

function trainingSourceFromJob(job: TrainingJobSummary): TrainingAssemblySource {
  return {
    id: `job:${job.job_id}`,
    kind: 'job',
    name: job.name || job.job_id,
    label: job.name || job.job_id,
    createdAt: new Date((job.ended_at ?? job.created_at) * 1000).toISOString().slice(0, 10),
    simulator: job.simulator,
    scenarios: job.scenarios,
    llm: null,
    router: routerModelFromTrainingJob(job),
    text2comp: text2compModelFromTrainingJob(job),
    fno: null,
    uploadedExpertName: job.uploaded_expert_name,
    uploadedExpertInputDim: job.uploaded_expert_input_dim,
    sourceJob: job,
  }
}

function pickLLM(status: AssemblyStatus | undefined) {
  const llms = (status?.llms ?? []).filter(item => {
    const name = item.name.toLowerCase()
    return !['embedding', 'reranker', 'guard', 'vl', 'omni'].some(token => name.includes(token))
  })
  return (
    llms.find(item => item.name.includes('Instruct')) ??
    llms.find(item => item.name.includes('0.6B')) ??
    llms.find(item => item.downloaded) ??
    status?.llms?.find(item => item.downloaded) ??
    status?.llms?.[0] ??
    null
  )
}

function pickGpu(status: AssemblyStatus | undefined) {
  const gpus = status?.gpus ?? []
  if (!gpus.length) return null
  return [...gpus].sort((a, b) => {
    const availability = Number(b.available) - Number(a.available)
    if (availability !== 0) return availability
    return (b.memory_free_mb ?? 0) - (a.memory_free_mb ?? 0)
  })[0]
}

function compatibleFnoExperts(
  status: AssemblyStatus | undefined,
  text2comp: SelectableModel | null,
): SelectableModel[] {
  const expectedDim = text2comp?.output_dim
  return (status?.fno_experts ?? []).filter(expert => {
    if (!expert.trained) return false
    if (expectedDim == null || expert.input_dim == null) return true
    return expert.input_dim === expectedDim
  })
}

function compatibleUploadedExperts(
  status: AssemblyStatus | undefined,
  text2comp: SelectableModel | null,
): ExpertModelInfo[] {
  const expectedDim = text2comp?.output_dim
  return (status?.custom_experts ?? []).filter(expert => {
    if (expert.status !== 'active' || expert.assembly_enabled === false || expert.exists === false) return false
    if (expectedDim == null || expert.input_dim == null) return true
    return expert.input_dim === expectedDim
  })
}

function pathName(path?: string | null): string {
  if (!path) return '未配置'
  return path.split('/').filter(Boolean).pop() || path
}

function samePath(left?: string | null, right?: string | null): boolean {
  return Boolean(left && right && left === right)
}

function pathListIncludes(paths: string[] | undefined, expected?: string | null): boolean {
  return Boolean(expected && (paths ?? []).includes(expected))
}

type ParsedModflowAnswer = {
  isModflow: boolean
  raw: string
  matrix: number[][] | null
  trendLines: string[]
}

function splitTrendLines(text: string): string[] {
  return text
    .replace(/\s+(?=\d+\.\s*(?:井|#)\d+[:：])/g, '\n')
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
}

function parseModflowAnswer(answer?: string): ParsedModflowAnswer {
  const raw = (answer || '').trim()
  const trigger = 'MODFLOW地下水专家输出：'
  const trendMarker = '中文趋势总结：'
  if (!raw.includes(trigger)) {
    return { isModflow: false, raw, matrix: null, trendLines: [] }
  }

  const body = raw.slice(raw.indexOf(trigger) + trigger.length).trim()
  const trendIndex = body.indexOf(trendMarker)
  const matrixSource = (trendIndex >= 0 ? body.slice(0, trendIndex) : body).trim()
  const trendText = trendIndex >= 0 ? body.slice(trendIndex + trendMarker.length).trim() : ''
  const matrixMatch = matrixSource.match(/\[\[[\s\S]*\]\]/)
  let matrix: number[][] | null = null
  if (matrixMatch) {
    try {
      const parsed = JSON.parse(matrixMatch[0])
      if (
        Array.isArray(parsed) &&
        parsed.every(row => Array.isArray(row) && row.every(value => typeof value === 'number'))
      ) {
        matrix = parsed
      }
    } catch {
      matrix = null
    }
  }

  return {
    isModflow: true,
    raw,
    matrix,
    trendLines: splitTrendLines(trendText),
  }
}

function formatResultNumber(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : '--'
}

function numericSummary(values: number[]): string {
  if (values.length === 0) return ''
  const first = values[0]
  const last = values[values.length - 1]
  let min = first
  let max = first
  values.forEach(value => {
    if (value < min) min = value
    if (value > max) max = value
  })
  const direction =
    Math.abs(last - first) < 1e-6 ? '末端与起始水平接近' : last > first ? '末端高于起始水平' : '末端低于起始水平'
  return `专家模型返回了一组数值预测，共 ${values.length} 个数值点，范围约 ${formatResultNumber(min)} 到 ${formatResultNumber(max)}，${direction}。`
}

function formatPlainModflowAnswer(parsed: ParsedModflowAnswer): string {
  if (!parsed.matrix) return formatConversationalAnswer(parsed.raw)
  const lines = ['已完成 MODFLOW 地下水预测。']
  lines.push('', '预测数值（hydraulic_head）：')
  parsed.matrix.forEach((row, index) => {
    lines.push(`井${index + 1}：${row.map(formatResultNumber).join('，')}`)
  })
  if (parsed.trendLines.length > 0) {
    lines.push('', '中文趋势总结：', ...parsed.trendLines)
  } else {
    const values = parsed.matrix.flat()
    const summary = numericSummary(values)
    if (summary) lines.push('', summary)
  }
  return lines.join('\n')
}

function stripDenseNumericLines(text: string): { text: string; values: number[] } {
  const values: number[] = []
  const numberPattern = /-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi
  const kept = text
    .split('\n')
    .filter(line => {
      const matches = line.match(numberPattern) ?? []
      const trimmed = line.trim()
      const isReadableValueLine = /^第\s*\d+\s*-\s*\d+\s*点[:：]/.test(trimmed)
      const denseArrayLine =
        !isReadableValueLine &&
        matches.length >= 4 &&
        (trimmed.startsWith('[') || trimmed.endsWith(']') || /^[\d\s.,，+\-eE]+[,.，]?$/.test(trimmed))
      if (denseArrayLine) {
        matches.forEach(item => values.push(Number(item)))
        return false
      }
      return true
    })
    .map(line => line.trimEnd())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  return { text: kept, values }
}

function formatConversationalAnswer(answer?: string): string {
  const raw = (answer || '').trim()
  if (!raw) return ''
  const { text, values } = stripDenseNumericLines(raw)
  const normalized = text
    .replace(/^\s*[\w-]*\s*(?:FNO|Expert|专家)?\s*输出[:：]\s*$/i, '')
    .replace(/^\s*[\u4e00-\u9fffA-Za-z0-9_-]*专家输出[:：]\s*$/i, '')
    .trim()
  const summary = numericSummary(values)
  if (normalized && summary) return `${normalized}\n\n${summary}`
  if (normalized) return normalized

  if (summary) return `已完成专家模型预测。\n\n${summary}`
  return raw
}

function AssemblyResultAnswer({ answer }: { answer?: string }) {
  const parsed = useMemo(() => parseModflowAnswer(answer), [answer])
  if (!parsed.raw) {
    return <div className="text-sm text-slate-400">没有返回最终答案。</div>
  }

  const displayText = parsed.isModflow ? formatPlainModflowAnswer(parsed) : formatConversationalAnswer(parsed.raw)
  return <div className="training-simple-chat-result__answer">{displayText}</div>
}

function profileToCards(profile: AssemblyProfile, loaded: boolean) {
  const isFnoProfile = profile.executor === 'fno_profile' || profile.executor === 'standard_fno_profile'
  return [
    {
      key: 'llm',
      title: 'LLM',
      icon: Brain,
      model: { name: pathName(profile.llm_path), description: isFnoProfile ? 'Qwen model' : 'Qwen base / embedding' },
      loaded,
      tone: 'violet',
    },
    {
      key: 'router',
      title: 'Router',
      icon: Route,
      model: { name: pathName(profile.router_path), description: `${profile.simulator || 'modflow'} route policy` },
      loaded,
      tone: 'amber',
    },
    {
      key: 'text2comp',
      title: 'Text2Comp',
      icon: Layers3,
      model: {
        name: pathName(profile.text2comp_path),
        description: `输入 ${profile.feature_dim || '--'} · 参数 ${profile.param_count || '--'}`,
      },
      loaded,
      tone: 'sky',
    },
    {
      key: 'expert',
      title: isFnoProfile ? 'FNO Expert' : 'DNN Expert',
      icon: Activity,
      model: {
        name: pathName(profile.expert_path),
        description: isFnoProfile
          ? `输入 ${profile.feature_dim || '--'} · FNO Expert`
          : `输出 ${(profile.output_shape ?? []).join(' x ') || '--'}`,
      },
      loaded,
      tone: 'emerald',
    },
  ]
}

function modelNote(model: SelectableModel | null): string {
  if (!model) return '未找到可用模型'
  if (model.description) return model.description
  if (model.output_dim != null) return `输出维度 ${model.output_dim}`
  if (model.input_dim != null) return `输入维度 ${model.input_dim}`
  return model.trained === false ? '未训练' : '可用'
}

function statusText(loaded: boolean): string {
  return loaded ? '已加载' : '待加载'
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

export default function TrainingSimpleAssemblyPage() {
  const { mutate } = useSWRConfig()
  const { data: status, error } = useSWR<AssemblyStatus>('assembly-status', api.getAssemblyStatus, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const { data: trainingJobs } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const [executorMode, setExecutorMode] = useState<ExecutorMode>('uploaded')
  const [assemblyMode, setAssemblyMode] = useState<AssemblyMode>('profile')
  const [profileId, setProfileId] = useState('')
  const [trainingJobId, setTrainingJobId] = useState('')
  const [fnoPath, setFnoPath] = useState('')
  const [uploadedExpertId, setUploadedExpertId] = useState('')
  const [gpuId, setGpuId] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [testInput, setTestInput] = useState('')
  const [testResult, setTestResult] = useState<{
    final_answer?: string
    router_class_name?: string
    latency_ms?: number
  } | null>(null)

  const assemblyProfiles = useMemo(
    () => (status?.assembly_profiles ?? []).filter(item => item.trained && item.chat_enabled),
    [status?.assembly_profiles],
  )
  const assemblyProfile = useMemo(
    () => assemblyProfiles.find(item => item.model_id === profileId) ?? assemblyProfiles[0] ?? null,
    [assemblyProfiles, profileId],
  )
  const usableTrainingJobs = useMemo(
    () =>
      (trainingJobs ?? [])
        .filter(isUsableTrainingJob)
        .sort((a, b) => (b.ended_at ?? b.created_at) - (a.ended_at ?? a.created_at)),
    [trainingJobs],
  )
  const trainingSources = useMemo(
    () => [
      ...TRAINING_BUNDLES.map(bundle => trainingSourceFromBundle(bundle, status)),
      ...usableTrainingJobs.map(trainingSourceFromJob),
    ],
    [status, usableTrainingJobs],
  )
  const selectedTrainingSource = useMemo(
    () => trainingSources.find(item => item.id === trainingJobId) ?? trainingSources[0] ?? null,
    [trainingJobId, trainingSources],
  )
  const defaultLLM = useMemo(() => pickLLM(status), [status])
  const llm = assemblyMode === 'training_job' ? selectedTrainingSource?.llm || defaultLLM : defaultLLM
  const trainingJobRouter = selectedTrainingSource?.router ?? null
  const trainingJobText2Comp = selectedTrainingSource?.text2comp ?? null
  const registryRouter = useMemo(() => pickFirst(status?.routers, item => item.trained), [status?.routers])
  const router = assemblyMode === 'training_job' ? trainingJobRouter : registryRouter
  const registryText2Comp = useMemo(() => pickFirst(status?.text2comps, item => item.trained), [status?.text2comps])
  const text2comp = assemblyMode === 'training_job' ? trainingJobText2Comp : registryText2Comp
  const fnoModels = useMemo(() => compatibleFnoExperts(status, text2comp), [status, text2comp])
  const selectedBundleFno = selectedTrainingSource?.kind === 'bundle' ? selectedTrainingSource.fno : null
  const fno = useMemo(
    () => selectedBundleFno ?? fnoModels.find(item => item.path === fnoPath) ?? fnoModels[0] ?? null,
    [fnoModels, fnoPath, selectedBundleFno],
  )
  const uploadedExperts = useMemo(() => compatibleUploadedExperts(status, text2comp), [status, text2comp])
  const uploaded = useMemo(
    () => uploadedExperts.find(item => item.model_id === uploadedExpertId) ?? uploadedExperts[0] ?? null,
    [uploadedExpertId, uploadedExperts],
  )
  const bestGpu = useMemo(() => pickGpu(status), [status])
  const selectedGpu = useMemo(
    () => (status?.gpus ?? []).find(gpu => gpu.index === gpuId) ?? bestGpu,
    [bestGpu, gpuId, status?.gpus],
  )
  const expert = executorMode === 'uploaded' ? uploaded : fno
  const trainingExpert = selectedTrainingSource?.kind === 'bundle' ? fno : expert
  const profileLoaded = Boolean(status?.loaded_models?.assembly_profile?.loaded)
  const selectedProfileLoaded = Boolean(
    profileLoaded &&
    assemblyProfile?.model_id &&
    status?.loaded_models?.assembly_profile?.model_id === assemblyProfile.model_id,
  )
  const loadedModels = status?.loaded_models
  const standardLlmLoaded = Boolean(loadedModels?.llm?.loaded && samePath(loadedModels?.llm?.path, llm?.path))
  const standardRouterLoaded = Boolean(
    loadedModels?.router?.loaded && samePath(loadedModels?.router?.path, trainingJobRouter?.path),
  )
  const standardText2CompLoaded = Boolean(
    loadedModels?.text2comp?.loaded && pathListIncludes(loadedModels?.text2comp?.paths, text2comp?.path),
  )
  const standardExpertLoaded =
    selectedTrainingSource?.kind === 'bundle' || executorMode === 'fno'
      ? Boolean(loadedModels?.fno?.loaded && pathListIncludes(loadedModels?.fno?.paths, fno?.path))
      : Boolean(
          loadedModels?.uploaded_expert?.loaded &&
          uploaded?.model_id &&
          loadedModels?.uploaded_expert?.model_id === uploaded.model_id,
        )
  const standardChainLoaded = Boolean(
    assemblyMode === 'training_job' &&
    !profileLoaded &&
    standardLlmLoaded &&
    standardRouterLoaded &&
    standardText2CompLoaded &&
    standardExpertLoaded,
  )
  const isLoaded = assemblyMode === 'profile' ? selectedProfileLoaded : standardChainLoaded
  const canLoadTrainingJob = Boolean(
    llm &&
    selectedGpu &&
    selectedTrainingSource &&
    trainingJobRouter &&
    text2comp &&
    (selectedTrainingSource.kind === 'bundle' ? fno : executorMode === 'fno' ? fno : uploaded),
  )
  const canLoad = Boolean(selectedGpu && (assemblyMode === 'profile' ? assemblyProfile : canLoadTrainingJob))

  useEffect(() => {
    if (gpuId == null || (status?.gpus ?? []).some(gpu => gpu.index === gpuId)) return
    setGpuId(null)
  }, [gpuId, status?.gpus])

  useEffect(() => {
    if (profileId && assemblyProfiles.some(item => item.model_id === profileId)) return
    setProfileId(assemblyProfiles[0]?.model_id ?? '')
  }, [assemblyProfiles, profileId])

  useEffect(() => {
    if (trainingJobId && trainingSources.some(item => item.id === trainingJobId)) return
    setTrainingJobId(trainingSources[0]?.id ?? '')
  }, [trainingJobId, trainingSources])

  useEffect(() => {
    if (fnoPath && fnoModels.some(item => item.path === fnoPath)) return
    setFnoPath(fnoModels[0]?.path ?? '')
  }, [fnoModels, fnoPath])

  useEffect(() => {
    if (uploadedExpertId && uploadedExperts.some(item => item.model_id === uploadedExpertId)) return
    setUploadedExpertId(uploadedExperts[0]?.model_id ?? '')
  }, [uploadedExpertId, uploadedExperts])

  useEffect(() => {
    if (assemblyMode !== 'profile') return
    const prompt = assemblyProfile?.demo_prompt?.trim()
    if (!prompt) return
    setTestInput(prompt)
    setTestResult(null)
  }, [assemblyMode, assemblyProfile?.demo_prompt, assemblyProfile?.model_id])

  useEffect(() => {
    if (assemblyMode !== 'training_job') return
    if (selectedTrainingSource?.kind === 'bundle') {
      setExecutorMode('fno')
      if (selectedTrainingSource.defaultPrompt && !testInput.trim()) {
        setTestInput(selectedTrainingSource.defaultPrompt)
      }
      return
    }
    if (executorMode === 'uploaded' && uploadedExperts.length === 0 && fno) setExecutorMode('fno')
    if (executorMode === 'fno' && !fno && uploadedExperts.length > 0) setExecutorMode('uploaded')
  }, [assemblyMode, executorMode, fno, selectedTrainingSource, testInput, uploadedExperts.length])

  useEffect(() => {
    if (!status) return
    if (!assemblyProfile && assemblyMode === 'profile') {
      setAssemblyMode('training_job')
      return
    }
    if (assemblyProfile && !selectedTrainingSource && assemblyMode === 'training_job') {
      setAssemblyMode('profile')
    }
  }, [assemblyMode, assemblyProfile, selectedTrainingSource, status])

  const refresh = async () => {
    await Promise.all([mutate('assembly-status'), mutate('training-jobs')])
  }

  const load = async () => {
    if (!selectedGpu) {
      setActionError('缺少 GPU，无法加载。')
      return
    }
    let selectedRouter: SelectableModel | null = null
    if (assemblyMode === 'profile') {
      if (!assemblyProfile) {
        setActionError('没有可用的已注册拼装模型。')
        return
      }
    } else {
      if (!selectedTrainingSource || !trainingJobRouter) {
        setActionError('请先选择一个可拼装的简洁训练任务或训练组合。')
        return
      }
      selectedRouter = trainingJobRouter
      if (!llm || !text2comp) {
        setActionError('缺少 LLM 或本次训练产出的 Text2Comp，无法把训练任务拼装为完整链路。')
        return
      }
      if ((selectedTrainingSource.kind === 'bundle' || executorMode === 'fno') && !fno) {
        setActionError('没有可用 FNO Expert。')
        return
      }
      if (selectedTrainingSource.kind !== 'bundle' && executorMode === 'uploaded' && !uploaded) {
        setActionError('没有可用 Uploaded Expert，或输入维度不匹配。')
        return
      }
    }
    setBusy(true)
    setActionError(null)
    setTestResult(null)
    try {
      if (assemblyMode === 'profile' && assemblyProfile) {
        await api.loadAssemblyModels({
          assembly_profile_id: assemblyProfile.model_id,
          llm_path: assemblyProfile.llm_path,
          llm_gpu_id: selectedGpu.index,
          force_split: assemblyProfile.force_split,
          auto_sync: true,
        })
      } else {
        await api.loadAssemblyModels({
          llm_path: llm?.path,
          llm_gpu_id: selectedGpu.index,
          router_path: selectedRouter?.path,
          text2comp_path: text2comp?.path,
          fno_path: selectedTrainingSource?.kind === 'bundle' || executorMode === 'fno' ? fno?.path : undefined,
          expert_executor: selectedTrainingSource?.kind === 'bundle' ? 'fno' : executorMode,
          uploaded_expert_id:
            selectedTrainingSource?.kind !== 'bundle' && executorMode === 'uploaded' ? uploaded?.model_id : undefined,
          force_split: selectedTrainingSource?.kind === 'bundle',
          auto_sync: true,
        })
      }
      await refresh()
    } catch (err) {
      setActionError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const unload = async () => {
    setBusy(true)
    setActionError(null)
    setTestResult(null)
    try {
      await api.unloadAssemblyModels()
      await refresh()
    } catch (err) {
      setActionError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const runTest = async () => {
    if (!testInput.trim()) return
    setBusy(true)
    setActionError(null)
    setTestResult(null)
    try {
      const result = await api.testAssembly({
        config: {
          main_llm_path: assemblyMode === 'profile' ? assemblyProfile?.llm_path : llm?.path,
          assembly_profile_id: assemblyMode === 'profile' ? assemblyProfile?.model_id : undefined,
          gpu_config: { llm_gpu_ids: selectedGpu ? [selectedGpu.index] : [] },
        },
        test_input: testInput,
      })
      setTestResult({
        final_answer: result.final_answer || result.first_cot_result?.split('\n').pop() || '',
        router_class_name: result.router_class_name || result.router_prediction,
        latency_ms: result.latency_ms,
      })
    } catch (err) {
      setActionError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  const componentCards = [
    {
      key: 'llm',
      title: 'LLM',
      icon: Brain,
      model: llm,
      loaded: standardLlmLoaded,
      tone: 'violet',
    },
    {
      key: 'router',
      title: 'Router',
      icon: Route,
      model: router,
      loaded: standardRouterLoaded,
      tone: 'amber',
    },
    {
      key: 'text2comp',
      title: 'Text2Comp',
      icon: Layers3,
      model: text2comp,
      loaded: standardText2CompLoaded,
      tone: 'sky',
    },
    {
      key: 'expert',
      title: selectedTrainingSource?.kind === 'bundle' || executorMode === 'fno' ? 'FNO Expert' : 'Uploaded Expert',
      icon: Activity,
      model: trainingExpert,
      loaded: standardExpertLoaded,
      tone: 'emerald',
    },
  ]
  const modelCards =
    assemblyMode === 'profile' && assemblyProfile
      ? profileToCards(assemblyProfile, selectedProfileLoaded)
      : componentCards
  const loadButtonLabel = assemblyMode === 'training_job' ? '加载训练组合' : '一键加载'

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <h1 className="training-simple-hero__title">模型拼装</h1>
              <p className="training-simple-hero__meta">
                {assemblyMode === 'profile'
                  ? assemblyProfile?.name || '选择已注册模型'
                  : selectedTrainingSource?.label || '选择训练任务'}
              </p>
            </div>
            {isLoaded ? (
              <button type="button" className="btn-ghost training-simple-hero__action" onClick={unload} disabled={busy}>
                {busy ? <Loader2 size={15} className="animate-spin" /> : <PowerOff size={15} />}
                卸载模型
              </button>
            ) : (
              <button
                type="button"
                className="btn-primary training-simple-hero__action"
                onClick={load}
                disabled={!canLoad || busy}
              >
                {busy ? <Loader2 size={15} className="animate-spin" /> : <Power size={15} />}
                {loadButtonLabel}
              </button>
            )}
          </section>

          {(error || actionError) && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {actionError ?? `无法加载模型拼装状态：${error?.message}`}
            </div>
          )}

          <div className="training-simple-bottom-grid">
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Layers3 size={16} className="text-violet-300" />
                <SectionTitle title="拼装来源" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-segmented" aria-label="拼装来源">
                  <button
                    type="button"
                    className={assemblyMode === 'profile' ? 'is-active' : ''}
                    onClick={() => setAssemblyMode('profile')}
                    disabled={!assemblyProfile}
                  >
                    已注册拼装模型
                  </button>
                  <button
                    type="button"
                    className={assemblyMode === 'training_job' ? 'is-active' : ''}
                    onClick={() => setAssemblyMode('training_job')}
                    disabled={!selectedTrainingSource}
                  >
                    简洁训练任务
                  </button>
                </div>
                {assemblyMode === 'profile' && assemblyProfile && (
                  <div className="space-y-2">
                    <label className="training-label" htmlFor="assembly-profile-select">
                      完整拼装模型
                    </label>
                    <select
                      id="assembly-profile-select"
                      className="select"
                      value={assemblyProfile.model_id}
                      onChange={event => setProfileId(event.target.value)}
                    >
                      {assemblyProfiles.map(profile => (
                        <option key={profile.model_id} value={profile.model_id}>
                          {profile.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                {assemblyMode === 'training_job' && (
                  <div className="space-y-2">
                    <label className="training-label" htmlFor="assembly-training-job-select">
                      简洁训练任务
                    </label>
                    <select
                      id="assembly-training-job-select"
                      className="select"
                      value={selectedTrainingSource?.id ?? ''}
                      onChange={event => setTrainingJobId(event.target.value)}
                      disabled={trainingSources.length === 0}
                    >
                      {trainingSources.length === 0 ? (
                        <option value="">暂无可拼装训练任务</option>
                      ) : (
                        trainingSources.map(source => (
                          <option key={source.id} value={source.id}>
                            {source.label} · {source.simulator.toUpperCase()} · {source.scenarios.length} 个场景
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                )}
                <div className="training-simple-chain">
                  {modelCards.map(item => {
                    const Icon = item.icon
                    return (
                      <div key={item.key} className="training-simple-chain__item">
                        <div className="training-simple-chain__label">
                          <Icon size={14} />
                          <span>{item.title}</span>
                          {item.loaded && <CheckCircle2 size={14} />}
                        </div>
                        <strong title={item.model?.name}>{item.model?.name ?? '未找到模型'}</strong>
                        <small title={`${modelNote(item.model)} · ${statusText(item.loaded)}`}>
                          {modelNote(item.model)} · {statusText(item.loaded)}
                        </small>
                      </div>
                    )
                  })}
                </div>
                {assemblyMode === 'training_job' && selectedTrainingSource?.kind !== 'bundle' && (
                  <>
                    <div className="training-simple-segmented" aria-label="专家类型">
                      <button
                        type="button"
                        className={executorMode === 'fno' ? 'is-active' : ''}
                        onClick={() => setExecutorMode('fno')}
                        disabled={!fno}
                      >
                        FNO Expert
                      </button>
                      <button
                        type="button"
                        className={executorMode === 'uploaded' ? 'is-active' : ''}
                        onClick={() => setExecutorMode('uploaded')}
                        disabled={uploadedExperts.length === 0}
                      >
                        Uploaded Expert
                      </button>
                    </div>
                    {executorMode === 'fno' ? (
                      <div className="space-y-2">
                        <label className="training-label" htmlFor="assembly-fno-select">
                          FNO Expert
                        </label>
                        <select
                          id="assembly-fno-select"
                          className="select"
                          value={fno?.path ?? ''}
                          onChange={event => {
                            setFnoPath(event.target.value)
                            setExecutorMode('fno')
                          }}
                          disabled={fnoModels.length === 0}
                        >
                          {fnoModels.length === 0 ? (
                            <option value="">暂无匹配 FNO Expert</option>
                          ) : (
                            fnoModels.map(item => (
                              <option key={item.path} value={item.path}>
                                {item.name}
                              </option>
                            ))
                          )}
                        </select>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <label className="training-label" htmlFor="assembly-uploaded-select">
                          Uploaded Expert
                        </label>
                        <select
                          id="assembly-uploaded-select"
                          className="select"
                          value={uploaded?.model_id ?? ''}
                          onChange={event => {
                            setUploadedExpertId(event.target.value)
                            setExecutorMode('uploaded')
                          }}
                          disabled={uploadedExperts.length === 0}
                        >
                          {uploadedExperts.length === 0 ? (
                            <option value="">暂无匹配 Uploaded Expert</option>
                          ) : (
                            uploadedExperts.map(item => (
                              <option key={item.model_id} value={item.model_id}>
                                {item.name}
                              </option>
                            ))
                          )}
                        </select>
                      </div>
                    )}
                  </>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Cpu size={16} className="text-emerald-300" />
                <SectionTitle title="资源与测试" />
                <button
                  type="button"
                  className="training-icon-button ml-auto"
                  onClick={() => refresh()}
                  aria-label="刷新拼装状态"
                  title="刷新拼装状态"
                >
                  <RefreshCw size={14} />
                </button>
              </div>
              <div className="training-card__body space-y-3">
                <div className="space-y-2">
                  <label className="training-label" htmlFor="assembly-gpu-select">
                    运行资源
                  </label>
                  <select
                    id="assembly-gpu-select"
                    className="select"
                    value={gpuId == null ? 'auto' : String(gpuId)}
                    onChange={event => setGpuId(event.target.value === 'auto' ? null : Number(event.target.value))}
                  >
                    <option value="auto">{bestGpu ? `自动分配 · GPU ${bestGpu.index}` : '自动分配 · 等待 GPU'}</option>
                    {(status?.gpus ?? []).map(gpu => (
                      <option key={gpu.index} value={gpu.index}>
                        GPU {gpu.index} · {gpu.available ? '可用' : '占用中，可排队'}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <textarea
                    className="input min-h-[6rem] resize-y"
                    value={testInput}
                    onChange={event => setTestInput(event.target.value)}
                    placeholder="输入一句任务描述，用于验证已装载链路"
                    aria-label="测试输入"
                  />
                </div>
                {isLoaded && (
                  <div className="training-simple-job__actions">
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={runTest}
                      disabled={!testInput.trim() || busy}
                    >
                      {busy ? <Loader2 size={14} className="animate-spin" /> : <PlayCircle size={14} />}
                      运行测试
                    </button>
                  </div>
                )}
              </div>
            </section>
          </div>

          {!canLoad && (
            <div className="training-simple-job__notice">
              <AlertTriangle size={14} />
              <span>当前缺少一键拼装所需模型，请先注册完整拼装模型，或训练/上传组件模型。</span>
            </div>
          )}

          {testResult && (
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <PlayCircle size={16} className="text-emerald-300" />
                <SectionTitle title="模型回复" />
              </div>
              <div className="training-card__body">
                <div className="training-simple-chat-result">
                  <AssemblyResultAnswer answer={testResult.final_answer} />
                  <div className="training-simple-chat-result__meta">
                    <span>Router：{testResult.router_class_name || '--'}</span>
                    <span>耗时：{testResult.latency_ms != null ? `${testResult.latency_ms.toFixed(2)} ms` : '--'}</span>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
