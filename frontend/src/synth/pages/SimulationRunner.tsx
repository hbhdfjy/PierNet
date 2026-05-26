import { useState, useEffect, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeed } from '../../lib/seedContext'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type {
  ExpertGenerateResponse,
  ExpertInputPlanResponse,
  Hdf5DataFileInfo,
  JobStatus,
  SimulationScenario,
} from '../../lib/types'
import {
  Zap,
  RefreshCw,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Database,
  SkipForward,
  Layers,
  Play,
  BarChart2,
  Bot,
  FileCode2,
  UploadCloud,
  Wand2,
} from 'lucide-react'
import { cn, formatBytes } from '../../lib/utils'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import {
  SYNTH_SAMPLE_COUNT_MAX,
  SYNTH_SAMPLE_COUNT_MIN,
  SYNTH_WORKERS_MAX,
  SYNTH_WORKERS_MIN,
  normalizeSynthSampleCount,
  normalizeSynthWorkers,
} from '../generationLimits'
import { simulationScenarioKey } from '../simulationScenario'
import { isRestartableJobStatus, isTerminalJobStatus, useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

// ── 模拟器元数据 ──────────────────────────────────────────────────

const SIM_META: Record<
  string,
  {
    label: string
    shortLabel: string
    color: string
    bg: string
    border: string
    dot: string
  }
> = {
  modflow: {
    label: 'MODFLOW',
    shortLabel: 'MF',
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/25',
    dot: 'bg-blue-500',
  },
  simpeg: {
    label: 'SimPEG',
    shortLabel: 'SP',
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/25',
    dot: 'bg-purple-500',
  },
  power_flow: {
    label: 'Power Flow',
    shortLabel: 'PF',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/25',
    dot: 'bg-orange-500',
  },
  transient: {
    label: 'Transient',
    shortLabel: 'TR',
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/25',
    dot: 'bg-red-500',
  },
  gcam: {
    label: 'PyPSA/GCAM',
    shortLabel: 'GC',
    color: 'text-teal-400',
    bg: 'bg-teal-500/10',
    border: 'border-teal-500/25',
    dot: 'bg-teal-500',
  },
  expert_model: {
    label: 'Expert Model',
    shortLabel: 'EX',
    color: 'text-fuchsia-300',
    bg: 'bg-fuchsia-500/10',
    border: 'border-fuchsia-500/25',
    dot: 'bg-fuchsia-400',
  },
}

const fallbackMeta = {
  label: '未知',
  shortLabel: 'NA',
  color: 'text-slate-400',
  bg: 'bg-slate-500/10',
  border: 'border-slate-500/25',
  dot: 'bg-slate-500',
}

// ── 子组件 ────────────────────────────────────────────────────────

function SimBadge({ simulator, className }: { simulator: string; className?: string }) {
  const m = SIM_META[simulator] ?? fallbackMeta
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium border',
        m.bg,
        m.border,
        m.color,
        className,
      )}
    >
      <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', m.dot)} />
      {m.label}
    </span>
  )
}

// 单场景卡片（多选模式）
function ScenarioRow({
  s,
  checked,
  onToggle,
  nSamples,
}: {
  s: SimulationScenario
  checked: boolean
  onToggle: () => void
  nSamples: number
}) {
  const progress = s.sample_count > 0 && nSamples > 0 ? Math.min(1, s.sample_count / nSamples) : 0
  const isComplete = s.sample_count >= nSamples && nSamples > 0

  return (
    <button
      onClick={onToggle}
      className={cn(
        'w-full text-left px-3 py-2 rounded-lg border transition-all duration-150 relative overflow-hidden',
        checked
          ? 'bg-amber-500/10 border-amber-500/30 text-white'
          : 'bg-slate-800/40 border-slate-700/30 text-slate-400 hover:border-slate-600/50 hover:text-slate-200',
      )}
    >
      {/* 进度背景条 */}
      {progress > 0 && (
        <div
          className={cn(
            'absolute inset-y-0 left-0 opacity-10 transition-all duration-500',
            isComplete ? 'bg-emerald-400' : 'bg-amber-400',
          )}
          style={{ width: `${progress * 100}%` }}
        />
      )}
      <div className="relative flex items-center gap-2">
        {/* 复选框 */}
        <div
          className={cn(
            'w-4 h-4 rounded flex-shrink-0 flex items-center justify-center border transition-all',
            checked ? 'bg-amber-500 border-amber-400' : 'bg-slate-700/60 border-slate-600/60',
          )}
        >
          {checked && <CheckCircle2 size={10} className="text-white" />}
        </div>
        {/* 场景名 */}
        <span className="flex-1 font-mono text-xs font-medium truncate">{s.scenario}</span>
        {/* 右侧信息 */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {s.sample_count > 0 && (
            <span
              className={cn('text-xs tabular-nums font-medium', isComplete ? 'text-emerald-400' : 'text-slate-400')}
            >
              {s.sample_count.toLocaleString()}
            </span>
          )}
          {s.output_shape && (
            <span className="text-xs text-slate-600 font-mono hidden sm:inline">({s.output_shape.join('×')})</span>
          )}
        </div>
      </div>
    </button>
  )
}

// 数据预览卡片（右栏顶部）
function DataOverviewCards({ scenarios }: { scenarios: SimulationScenario[] }) {
  const totalSamples = scenarios.reduce((s, x) => s + x.sample_count, 0)
  const withData = scenarios.filter(s => s.h5_path)
  const totalSize = scenarios.reduce((s, x) => s + x.file_size_bytes, 0)

  const bySimulator: Record<string, number> = {}
  for (const s of scenarios) {
    bySimulator[s.simulator] = (bySimulator[s.simulator] ?? 0) + s.sample_count
  }

  const cards = [
    {
      id: 'total',
      label: '总样本数',
      value: totalSamples.toLocaleString(),
      sub: `${withData.length} 个文件`,
      icon: Database,
      color: 'text-amber-400',
    },
    {
      id: 'size',
      label: '总数据量',
      value: formatBytes(totalSize),
      sub: `${scenarios.length} 个场景`,
      icon: BarChart2,
      color: 'text-sky-400',
    },
    ...Object.entries(SIM_META).map(([sim, meta]) => ({
      id: sim,
      label: meta.label,
      value: (bySimulator[sim] ?? 0).toLocaleString(),
      sub: `${scenarios.filter(s => s.simulator === sim).length} 场景`,
      icon: Layers,
      color: meta.color,
    })),
  ]

  return (
    <div className="data-overview-grid">
      {cards.map(({ id, label, value, sub, icon: Icon, color }) => (
        <div key={id} className="card data-overview-card px-3 py-2.5">
          <div className="flex items-center gap-1.5 mb-1">
            <Icon size={12} className={color} />
            <span className="label min-w-0 truncate text-xs">{label}</span>
          </div>
          <div className={cn('text-xl font-bold tabular-nums', color)}>{value}</div>
          <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
        </div>
      ))}
    </div>
  )
}

const FALLBACK_EXPERT_CONSTRAINTS: Record<string, string[]> = {
  interface: ['必须定义可调用函数 predict(inputs)。', 'inputs 必须按 list[float] 处理，所有数值必须是有限 float。'],
  output: ['predict 必须返回有限 float 或一维 finite float 数组。'],
}

const FALLBACK_EXPERT_EXAMPLE = `# expert_model.py
EXAMPLE_INPUT = [0.0]

def predict(inputs):
    x = float(inputs[0])
    return [x, x * x]`

function ExpertModelPanel({ onGenerated }: { onGenerated: () => void }) {
  const { data, isLoading, mutate } = useSWR('simulation-expert-models', () => api.listExpertModels(), {
    refreshInterval: 30000,
    revalidateOnMount: true,
  })
  const models = useMemo(() => data?.models ?? [], [data?.models])
  const [modelId, setModelId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [modelName, setModelName] = useState('')
  const [scenario, setScenario] = useState('expert_demo')
  const [prompt, setPrompt] = useState('有100个点，每个点从0开始依次加10')
  const [overwrite, setOverwrite] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [plan, setPlan] = useState<ExpertInputPlanResponse | null>(null)
  const [result, setResult] = useState<ExpertGenerateResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId && models.length > 0) setModelId(models[0].model_id)
  }, [modelId, models])

  const selectedModel = models.find(model => model.model_id === modelId) ?? null
  const constraintGroups = data?.constraints ?? FALLBACK_EXPERT_CONSTRAINTS
  const exampleSource = data?.example_source ?? FALLBACK_EXPERT_EXAMPLE
  const planCount = typeof plan?.plan.count === 'number' ? plan.plan.count : null
  const planDim = typeof plan?.plan.input_dim === 'number' ? plan.plan.input_dim : null

  const handleUpload = async () => {
    if (!file) {
      setError('请选择 .py 专家模型文件')
      return
    }
    setError(null)
    setUploading(true)
    try {
      const response = await api.uploadExpertModel({ name: modelName.trim() || file.name, file })
      setModelId(response.model.model_id)
      setModelName('')
      setFile(null)
      await mutate()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '上传专家模型失败')
    } finally {
      setUploading(false)
    }
  }

  const handlePlan = async () => {
    if (!modelId) {
      setError('请先上传或选择专家模型')
      return
    }
    setError(null)
    setPlanning(true)
    try {
      const response = await api.planExpertInputs({ modelId, prompt })
      setPlan(response)
      setResult(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '解析专家模型输入失败')
    } finally {
      setPlanning(false)
    }
  }

  const handleGenerate = async () => {
    if (!modelId) {
      setError('请先上传或选择专家模型')
      return
    }
    setError(null)
    setGenerating(true)
    try {
      const response = await api.generateExpertData({ modelId, scenario, prompt, overwrite })
      setResult(response)
      setPlan({
        ok: true,
        model_id: modelId,
        plan: response.input_plan,
        preview: Array.isArray(response.input_plan.preview) ? (response.input_plan.preview as number[][]) : [],
        summary: typeof response.input_plan.summary === 'string' ? response.input_plan.summary : '',
        warnings: Array.isArray(response.input_plan.warnings) ? (response.input_plan.warnings as string[]) : [],
      })
      onGenerated()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '生成专家模型数据失败')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="card px-4 py-3 space-y-3 animate-fade-in">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-fuchsia-500/15 border border-fuchsia-500/25 flex items-center justify-center flex-shrink-0">
            <Bot size={14} className="text-fuchsia-300" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-100 text-base">专家模型</span>
              <span className="badge bg-fuchsia-500/10 text-fuchsia-200 border border-fuchsia-500/20 text-xs py-0.5">
                {models.length}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-mono truncate">{data?.interface ?? 'predict(inputs) -> float'}</p>
          </div>
        </div>
        {isLoading && <RefreshCw size={13} className="animate-spin text-fuchsia-300 flex-shrink-0" />}
      </div>

      <div className="rounded-lg border border-fuchsia-500/15 bg-fuchsia-500/5 p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="label text-fuchsia-200">接口约束</span>
          <span className="text-xs text-slate-500">符合约束的模型可直接上传并用于生成 HDF5</span>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_0.9fr] gap-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
            {Object.entries(constraintGroups).map(([group, items]) => (
              <div key={group} className="min-w-0">
                <div className="text-xs font-mono text-fuchsia-200/80 mb-1">{group}</div>
                <ul className="space-y-1 text-xs text-slate-400">
                  {items.map(item => (
                    <li key={item} className="leading-5">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <pre className="rounded-md border border-slate-700/40 bg-slate-950/45 px-3 py-2 text-xs text-slate-300 overflow-x-auto font-mono">
            {exampleSource}
          </pre>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(260px,0.82fr)_minmax(360px,1.18fr)] gap-3">
        <div className="rounded-lg border border-slate-700/35 bg-slate-900/25 p-3 space-y-3">
          <div className="flex items-center gap-2">
            <input
              id="expert-model-file"
              type="file"
              accept=".py,text/x-python"
              className="hidden"
              onChange={e => {
                const nextFile = e.target.files?.[0] ?? null
                setFile(nextFile)
                if (nextFile && !modelName) setModelName(nextFile.name.replace(/\.py$/i, ''))
              }}
            />
            <label className="btn-ghost py-1.5 px-2 text-xs cursor-pointer flex-shrink-0" htmlFor="expert-model-file">
              <UploadCloud size={12} />
              选择文件
            </label>
            <span className="text-xs text-slate-500 truncate min-w-0">{file ? file.name : '未选择文件'}</span>
          </div>
          <div className="grid grid-cols-[1fr_auto] gap-2">
            <input
              className="input text-xs py-1.5 px-2"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              placeholder="模型名称"
            />
            <button className="btn py-1.5 px-3 text-xs" onClick={handleUpload} disabled={uploading || !file}>
              {uploading ? <RefreshCw size={12} className="animate-spin" /> : <FileCode2 size={12} />}
              上传
            </button>
          </div>
          <div>
            <label className="label block mb-1">当前模型</label>
            <select
              className="input w-full text-xs py-1.5 px-2"
              value={modelId}
              onChange={e => setModelId(e.target.value)}
            >
              {models.length === 0 && <option value="">暂无模型</option>}
              {models.map(model => (
                <option key={model.model_id} value={model.model_id}>
                  {model.name}
                </option>
              ))}
            </select>
            {selectedModel && (
              <div className="flex items-center justify-between gap-2 mt-1.5 text-xs text-slate-500">
                <span className="font-mono truncate">{selectedModel.model_id}</span>
                <span className="tabular-nums flex-shrink-0">{formatBytes(selectedModel.file_size_bytes)}</span>
              </div>
            )}
            {selectedModel?.example_input_dim && (
              <div className="flex items-center justify-between gap-2 mt-1 text-xs text-slate-600">
                <span>校验输入维度 {selectedModel.example_input_dim}</span>
                {selectedModel.smoke_output_dim && <span>输出维度 {selectedModel.smoke_output_dim}</span>}
              </div>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-slate-700/35 bg-slate-900/25 p-3 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-[minmax(150px,0.34fr)_1fr] gap-2">
            <div>
              <label className="label block mb-1">场景名</label>
              <input
                className="input w-full text-xs py-1.5 px-2 font-mono"
                value={scenario}
                onChange={e => setScenario(e.target.value)}
                placeholder="expert_demo"
              />
            </div>
            <div>
              <label className="label block mb-1">输入设定</label>
              <input
                className="input w-full text-xs py-1.5 px-2"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="有100个点，每个点从0开始依次加10"
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button className="btn-ghost py-1.5 px-2 text-xs" onClick={handlePlan} disabled={planning || !modelId}>
              {planning ? <RefreshCw size={12} className="animate-spin" /> : <Wand2 size={12} />}
              预览输入
            </button>
            <button
              className="btn py-1.5 px-3 text-xs bg-fuchsia-600 hover:bg-fuchsia-500"
              onClick={handleGenerate}
              disabled={generating || !modelId || !scenario.trim() || !prompt.trim()}
            >
              {generating ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
              生成 HDF5
            </button>
            <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={e => setOverwrite(e.target.checked)}
                className="accent-fuchsia-500"
              />
              覆盖同名
            </label>
          </div>

          {plan && (
            <div className="rounded-lg border border-fuchsia-500/15 bg-fuchsia-500/5 px-3 py-2 text-xs">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                {planCount !== null && <span className="badge bg-slate-800/70 text-slate-300">{planCount} 点</span>}
                {planDim !== null && <span className="badge bg-slate-800/70 text-slate-300">{planDim} 维输入</span>}
                <span className="text-fuchsia-200 truncate">{plan.summary}</span>
              </div>
              <div className="font-mono text-slate-500 truncate">
                {plan.preview.map(row => `[${row.map(v => Number(v).toLocaleString()).join(', ')}]`).join('  ')}
              </div>
            </div>
          )}

          {result && (
            <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-300">
              已生成 {result.validation.sample_count.toLocaleString()} 条，保存到{' '}
              <span className="font-mono text-emerald-200">{result.saved_path}</span>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/8 px-3 py-2 text-xs text-red-300">
              <AlertCircle size={12} className="mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function SimulationRunner() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('simulate')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 720,
    minWidth: 360,
    maxWidth: 920,
    storageKey: 'PierNet_simulate_sidebar_width_v2',
  })

  const {
    data: scenarios,
    isLoading,
    mutate: refreshScenarios,
  } = useSWR<SimulationScenario[]>('simulation-scenarios', () => api.getSimulationScenarios(), {
    refreshInterval: 20000,
    revalidateOnMount: true,
  })
  const { data: hdf5Files, mutate: refreshHdf5Files } = useSWR<Hdf5DataFileInfo[]>(
    'simulation-data-files',
    () => api.listHdf5DataFiles(),
    {
      refreshInterval: 20000,
      revalidateOnMount: true,
    },
  )

  const stage1DataRows = useMemo<SimulationScenario[]>(() => {
    const rows = [...(scenarios ?? [])]
    const seen = new Set(rows.map(simulationScenarioKey))
    for (const item of hdf5Files ?? []) {
      const key = `${item.simulator}/${item.scenario}`
      if (seen.has(key)) continue
      rows.push({
        simulator: item.simulator,
        scenario: item.scenario,
        config_path: '',
        h5_path: item.path,
        sample_count: item.sample_count,
        output_shape: item.output_shape,
        file_size_bytes: item.file_size_bytes,
      })
      seen.add(key)
    }
    return rows
  }, [hdf5Files, scenarios])

  const refreshStage1Data = useCallback(() => {
    refreshScenarios(() => api.getSimulationScenarios(true), { revalidate: true })
    refreshHdf5Files(() => api.listHdf5DataFiles(), { revalidate: true })
  }, [refreshHdf5Files, refreshScenarios])

  // ── 状态 ──
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nSamples, setNSamples] = useState(100)
  const [nSamplesInput, setNSamplesInput] = useState('100')
  const { seed } = useSeed()
  const [skipExisting, setSkipExisting] = useState(false)
  const [parallel, setParallel] = useState(false)
  const [maxWorkers, setMaxWorkers] = useState(4)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tableOpen, setTableOpen] = useState(false)
  const [filterSim, setFilterSim] = useState<string | null>(null)

  // 按模拟器分组
  const grouped = useMemo(() => {
    const g: Record<string, SimulationScenario[]> = {}
    for (const s of scenarios ?? []) {
      if (!g[s.simulator]) g[s.simulator] = []
      g[s.simulator].push(s)
    }
    return g
  }, [scenarios])

  // 全选/清空
  const allNames = useMemo(() => (scenarios ?? []).map(simulationScenarioKey), [scenarios])
  const filteredNames = useMemo(() => {
    if (!filterSim) return allNames
    return (scenarios ?? []).filter(s => s.simulator === filterSim).map(simulationScenarioKey)
  }, [scenarios, filterSim, allNames])

  const toggle = (name: string) =>
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const selectAll = () => setSelected(new Set(filteredNames))
  const clearAll = () => setSelected(new Set())
  const selectIncomplete = () => {
    const incomplete = (scenarios ?? [])
      .filter(s => (!filterSim || s.simulator === filterSim) && s.sample_count < nSamples)
      .map(simulationScenarioKey)
    setSelected(new Set(incomplete))
  }

  const canLaunch = isRestartableJobStatus(monitor.status)

  const handleLaunch = async () => {
    if (selected.size === 0) {
      setError('请至少选择一个场景')
      return
    }
    setError(null)
    setLaunching(true)
    try {
      let result
      if (selected.size === 1) {
        const selectedKey = [...selected][0]
        const sc = scenarios?.find(s => simulationScenarioKey(s) === selectedKey)
        if (!sc) throw new Error('场景未找到')
        result = await api.startSimulation({
          simulator: sc.simulator,
          scenario: sc.scenario,
          n_samples: nSamples,
          seed,
          config_path: sc.config_path,
          skip_existing: skipExisting,
          parallel,
          max_workers: maxWorkers,
        })
      } else {
        result = await api.startBatchSimulation({
          scenarios: [...selected],
          n_samples: nSamples,
          seed,
          skip_existing: skipExisting,
          parallel,
          max_workers: maxWorkers,
        })
      }
      monitor.start(result.job_id, result.scenario_totals, result.status as JobStatus)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  // 每个场景完成后立即刷新场景列表（批量仿真时实时更新样本数）
  useEffect(() => {
    if (monitor.scenarioDoneCount > 0) {
      refreshStage1Data()
    }
  }, [monitor.scenarioDoneCount, refreshStage1Data])

  // 全部完成后也刷新一次
  useEffect(() => {
    if (isTerminalJobStatus(monitor.status)) {
      refreshStage1Data()
    }
  }, [monitor.status, refreshStage1Data])

  const selectedScenarios = useMemo(
    () => (scenarios ?? []).filter(s => selected.has(simulationScenarioKey(s))),
    [scenarios, selected],
  )

  return (
    <div className="workbench-shell">
      {/* ── 左栏 ── */}
      <div className="workbench-sidebar" style={{ width: sidebarWidth }}>
        {/* 页头 */}
        <div className="workbench-sidebar-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
                <Zap size={14} className="text-amber-400" />
              </div>
              <h1 className="text-lg font-bold text-white">仿真运行</h1>
              <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/20 text-xs">阶段 1</span>
            </div>
            <button
              className="btn-ghost py-1 px-2 text-xs"
              onClick={() => {
                refreshStage1Data()
              }}
              title="刷新"
            >
              <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
            </button>
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">物理仿真 → HDF5 数据集</p>
        </div>

        {/* 场景筛选标签 */}
        <div className="flex-shrink-0 flex items-center gap-1 px-4 pt-2 pb-1.5 border-b border-slate-700/20 overflow-x-auto">
          <button
            onClick={() => setFilterSim(null)}
            className={cn(
              'flex-shrink-0 px-2 py-0.5 rounded-md text-xs font-medium transition-all border',
              !filterSim
                ? 'bg-slate-600/40 text-slate-200 border-slate-500/40'
                : 'text-slate-500 border-transparent hover:text-slate-300',
            )}
          >
            全部 {scenarios ? `(${scenarios.length})` : ''}
          </button>
          {Object.entries(SIM_META).map(([sim, meta]) => {
            const count = grouped[sim]?.length ?? 0
            if (count === 0) return null
            return (
              <button
                key={sim}
                onClick={() => setFilterSim(filterSim === sim ? null : sim)}
                className={cn(
                  'flex-shrink-0 flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium transition-all border',
                  filterSim === sim
                    ? cn(meta.bg, meta.border, meta.color)
                    : 'text-slate-500 border-transparent hover:text-slate-300',
                )}
              >
                <span className={cn('w-1.5 h-1.5 rounded-full', meta.dot)} />
                {meta.shortLabel} ({count})
              </button>
            )
          })}
        </div>

        {/* 场景列表工具栏 */}
        <div className="flex-shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700/20">
          <div className="flex items-center gap-1">
            {selected.size > 0 && (
              <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/20 text-xs py-0.5">
                {selected.size} 已选
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            <button className="btn-ghost py-0.5 px-2 text-sm" onClick={selectAll}>
              全选
            </button>
            <button
              className="btn-ghost py-0.5 px-2 text-sm"
              onClick={selectIncomplete}
              title="选择样本数不足配置数的场景"
            >
              <SkipForward size={10} className="mr-0.5" />
              未满
            </button>
            <button className="btn-ghost py-0.5 px-2 text-sm" onClick={clearAll}>
              清空
            </button>
          </div>
        </div>

        {/* 场景列表（可滚动）*/}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0">
          {isLoading && (
            <div className="flex items-center gap-2 text-slate-500 text-xs py-3 px-1">
              <RefreshCw size={11} className="animate-spin text-amber-500" /> 扫描配置目录…
            </div>
          )}
          {!isLoading && Object.keys(grouped).length === 0 && (
            <div className="flex flex-col items-center gap-2 py-10 text-center">
              <Zap size={18} className="text-slate-700" />
              <p className="text-slate-500 text-xs">未找到任何场景配置</p>
            </div>
          )}
          {Object.entries(grouped).map(([sim, list]) => {
            const meta = SIM_META[sim] ?? fallbackMeta
            const visible = filterSim ? filterSim === sim : true
            if (!visible) return null
            return (
              <div key={sim}>
                <div className="flex items-center gap-1.5 mb-1.5 px-1">
                  <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', meta.dot)} />
                  <span className={cn('text-sm font-semibold', meta.color)}>{meta.label}</span>
                  <span className="text-sm text-slate-500">{list.length} 个</span>
                </div>
                <div className="space-y-1">
                  {list.map(s => (
                    <ScenarioRow
                      key={simulationScenarioKey(s)}
                      s={s}
                      checked={selected.has(simulationScenarioKey(s))}
                      onToggle={() => toggle(simulationScenarioKey(s))}
                      nSamples={nSamples}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* 底部参数 + 按钮 */}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/30">
          {/* 参数行 */}
          <div className="px-4 py-3 space-y-3">
            <div>
              <label className="label block mb-1">样本数 / 场景</label>
              <input
                type="number"
                className="input w-full text-xs py-1.5 px-3"
                value={nSamplesInput}
                min={SYNTH_SAMPLE_COUNT_MIN}
                max={SYNTH_SAMPLE_COUNT_MAX}
                onChange={e => {
                  setNSamplesInput(e.target.value)
                  const n = Number(e.target.value)
                  if (!isNaN(n)) setNSamples(normalizeSynthSampleCount(n))
                }}
                onBlur={() => {
                  const n = Number(nSamplesInput)
                  const v = normalizeSynthSampleCount(n)
                  setNSamples(v)
                  setNSamplesInput(String(v))
                }}
              />
            </div>

            {/* skip-existing 开关 */}
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => setSkipExisting(v => !v)}>
              <div
                className={cn(
                  'relative w-8 h-4 rounded-full transition-all duration-200 flex-shrink-0',
                  skipExisting ? 'bg-amber-500' : 'bg-slate-700',
                )}
              >
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all duration-200"
                  style={{ left: skipExisting ? '18px' : '2px' }}
                />
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-sm text-slate-300 font-medium">跳过已完成</span>
                <span className="text-sm text-slate-600 ml-1.5">
                  {skipExisting ? '已达目标则跳过，未满则重新生成' : '忽略已有样本重新生成'}
                </span>
              </div>
              <SkipForward size={12} className={skipExisting ? 'text-amber-400' : 'text-slate-600'} />
            </div>

            {/* 并行生成 */}
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  'relative w-8 h-4 rounded-full transition-all duration-200 flex-shrink-0 cursor-pointer',
                  parallel ? 'bg-sky-500' : 'bg-slate-700',
                )}
                onClick={() => setParallel(v => !v)}
              >
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all duration-200"
                  style={{ left: parallel ? '18px' : '2px' }}
                />
              </div>
              <span
                className="text-sm text-slate-300 font-medium flex-1 cursor-pointer"
                onClick={() => setParallel(v => !v)}
              >
                多核并行
              </span>
              {parallel && (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-slate-500">核数</span>
                  <input
                    type="number"
                    className="input w-14 text-xs py-0.5 px-2 text-center"
                    value={maxWorkers}
                    min={SYNTH_WORKERS_MIN}
                    max={SYNTH_WORKERS_MAX}
                    onClick={e => e.stopPropagation()}
                    onChange={e => {
                      const n = Number(e.target.value)
                      if (!isNaN(n)) setMaxWorkers(normalizeSynthWorkers(n, 4))
                    }}
                  />
                </div>
              )}
            </div>

            {/* 已选场景摘要 */}
            {selectedScenarios.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {selectedScenarios.slice(0, 4).map(s => {
                  const m = SIM_META[s.simulator] ?? fallbackMeta
                  return (
                    <span
                      key={simulationScenarioKey(s)}
                      className={cn(
                        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs border',
                        m.bg,
                        m.border,
                        m.color,
                      )}
                    >
                      {s.scenario}
                    </span>
                  )
                })}
                {selectedScenarios.length > 4 && (
                  <span className="text-xs text-slate-500 self-center">+{selectedScenarios.length - 4}</span>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {canLaunch && (
            <div className="px-4 pb-4 space-y-1.5">
              <button
                className={cn(
                  'btn w-full py-2.5 text-sm justify-center shadow-lg',
                  selected.size > 0 && !launching
                    ? 'bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-white'
                    : 'bg-slate-700/60 text-slate-500 cursor-not-allowed',
                )}
                onClick={handleLaunch}
                disabled={launching || selected.size === 0}
              >
                {launching ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" /> 启动中…
                  </>
                ) : selected.size > 1 ? (
                  <>
                    <Play size={14} /> 批量仿真（{selected.size} 个场景）
                  </>
                ) : (
                  <>
                    <Zap size={14} /> 开始仿真{selected.size === 1 ? `（${[...selected][0]}）` : ''}
                  </>
                )}
              </button>
              {isTerminalJobStatus(monitor.status) && (
                <button className="btn-ghost w-full py-1.5 justify-center text-xs" onClick={monitor.reset}>
                  重新配置
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <ResizeHandle onMouseDown={onResizeStart} color="amber" />

      {/* ── 右栏 ── */}
      <div className="workbench-main-scroll">
        <ExpertModelPanel
          onGenerated={() => {
            refreshStage1Data()
          }}
        />

        {/* 数据总览卡片 */}
        {stage1DataRows.length > 0 && <DataOverviewCards scenarios={stage1DataRows} />}

        {/* Job 监控面板 */}
        <JobMonitorPanel
          status={monitor.status}
          logs={monitor.logs}
          progress={monitor.progress}
          stats={monitor.stats}
          autoScroll={monitor.autoScroll}
          onAutoScrollChange={monitor.setAutoScroll}
          onStop={monitor.stop}
          jobId={monitor.jobId}
          jobIds={monitor.jobIds}
          stageLabel="物理仿真"
          stageColor="text-amber-400"
          accentColor="amber"
          onDone={() => navigate('/synth/register')}
          doneLabel="去注册场景"
        />

        {monitor.status === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Zap size={24} className="text-amber-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未启动仿真</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景 → 配置参数 → 点击启动</p>
            </div>
          </div>
        )}

        {/* HDF5 详细状态表格（折叠）*/}
        <div className="card overflow-hidden animate-fade-in">
          <button
            onClick={() => setTableOpen(o => !o)}
            className="w-full card-header accordion-card-header justify-between transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <Database size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-base">HDF5 文件详情</span>
              {scenarios && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs py-0.5">
                  {stage1DataRows.filter(s => s.h5_path).length} / {stage1DataRows.length}
                </span>
              )}
            </div>
            {tableOpen ? (
              <ChevronUp size={13} className="text-slate-500" />
            ) : (
              <ChevronDown size={13} className="text-slate-500" />
            )}
          </button>

          {tableOpen && (
            <div className="list-table-scroll">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700/40">
                    <th className="px-3 py-2 text-left label">模拟器</th>
                    <th className="px-3 py-2 text-left label">场景</th>
                    <th className="px-3 py-2 text-right label">样本数</th>
                    <th className="px-3 py-2 text-left label">形状</th>
                    <th className="px-3 py-2 text-right label">大小</th>
                    <th className="px-3 py-2 text-left label">路径</th>
                  </tr>
                </thead>
                <tbody>
                  {stage1DataRows.map(s => {
                    return (
                      <tr
                        key={`${s.simulator}/${s.scenario}`}
                        className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors"
                      >
                        <td className="px-3 py-1.5">
                          <SimBadge simulator={s.simulator} />
                        </td>
                        <td className="px-3 py-1.5 font-mono text-slate-300 max-w-[140px] truncate">{s.scenario}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums">
                          {s.sample_count > 0 ? (
                            <span className="text-sky-400">{s.sample_count.toLocaleString()}</span>
                          ) : (
                            <span className="text-slate-700">—</span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-slate-600">
                          {s.output_shape ? `(${s.output_shape.join('×')})` : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">
                          {s.file_size_bytes > 0 ? formatBytes(s.file_size_bytes) : '—'}
                        </td>
                        <td className="px-3 py-1.5 max-w-[160px] truncate">
                          {s.h5_path ? (
                            <span className="text-emerald-500/60 font-mono">{s.h5_path.split('/').pop()}</span>
                          ) : (
                            <span className="text-slate-700">未生成</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
