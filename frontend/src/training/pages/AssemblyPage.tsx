import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Workflow,
  CheckCircle2,
  ArrowRight,
  Zap,
  Settings,
  Box,
  Layers,
  Activity,
  HardDrive,
  Server,
  Loader2,
  Power,
  PowerOff,
  FileText,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Save,
  Info,
  X,
} from 'lucide-react'
import { api } from '../../lib/api'

// ===== 接口定义 =====

interface GPUInfo {
  index: number
  name: string
  memory_used_mb: number
  memory_free_mb: number
  memory_total_mb: number
  available: boolean
}

interface LLMInfo {
  name: string
  path: string
  size: string
  downloaded: boolean
}

interface RouterModel {
  name: string
  path: string
  num_classes: number
  class_names: string[]
  description: string
  trained: boolean
  gpu_id?: number
  router_type?: string
}

interface Text2CompModel {
  name: string
  simulator: string
  output_dim: number
  path: string
  trained: boolean
  gpu_id?: number
}

interface FNOExpert {
  name: string
  simulator: string
  input_dim: number
  output_shape: number[]
  path: string
  trained: boolean
  gpu_id?: number
}

interface UploadedExpert {
  model_id: string
  name: string
  simulator?: string
  domain?: string
  input_dim?: number | null
  output_dim?: number | null
  runtime?: string
  status?: string
  path: string
  trained?: boolean
  assembly_enabled?: boolean
  data_generation_enabled?: boolean
  last_error?: string | null
}

interface LoadedModels {
  llm: { loaded: boolean; gpu_id?: number; path?: string | null }
  router: { loaded: boolean; gpu_id?: number; path?: string | null }
  text2comp: { loaded: boolean; gpu_id?: number; paths?: string[] }
  fno: { loaded: boolean; gpu_id?: number; paths?: string[] }
  uploaded_expert?: { loaded: boolean; model_id?: string | null; path?: string | null; executor?: string | null }
}

interface AssemblyStatus {
  llms: LLMInfo[]
  routers: RouterModel[]
  text2comps: Text2CompModel[]
  fno_experts: FNOExpert[]
  custom_experts: UploadedExpert[]
  gpus: GPUInfo[]
  loaded_models: LoadedModels
  gpu_available: boolean
  architecture_note?: string
}

interface DomainInfo {
  simulator: string
  domain_context: string
  scenarios: Record<string, string>
  output_description: string
}

// ===== GPU卡片组件 =====

function GPUCard({ gpu, isLoaded, compact }: { gpu: GPUInfo; isLoaded?: boolean; compact?: boolean }) {
  const usedPercent = (gpu.memory_used_mb / gpu.memory_total_mb) * 100

  if (compact) {
    return (
      <div
        className={`rounded-lg border p-2 ${isLoaded ? 'border-emerald-500/40 bg-emerald-500/10' : 'border-slate-700/40 bg-slate-900/30'}`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <HardDrive size={12} className={isLoaded ? 'text-emerald-400' : 'text-slate-400'} />
            <span className="text-xs font-medium text-slate-100">GPU{gpu.index}</span>
          </div>
          {isLoaded && <span className="text-xs text-emerald-300">已加载</span>}
        </div>
        <div className="h-1 rounded-full bg-slate-700/50 overflow-hidden mt-1">
          <div
            className={`h-full rounded-full ${isLoaded ? 'bg-emerald-500' : 'bg-sky-500'}`}
            style={{ width: `${usedPercent}%` }}
          />
        </div>
        <div className="text-xs text-slate-500 mt-0.5">{gpu.memory_free_mb}MB空闲</div>
      </div>
    )
  }

  return (
    <div
      className={`rounded-lg border p-3 ${isLoaded ? 'border-emerald-500/40 bg-emerald-500/10' : 'border-slate-700/40 bg-slate-900/30'}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <HardDrive size={14} className={isLoaded ? 'text-emerald-400' : 'text-slate-400'} />
          <span className="text-sm font-medium text-slate-100">GPU {gpu.index}</span>
        </div>
        {isLoaded && (
          <span className="badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300 text-xs">已加载</span>
        )}
      </div>
      <div className="text-xs text-slate-400 mb-1">{gpu.name}</div>
      <div className="h-2 rounded-full bg-slate-700/50 overflow-hidden mb-1">
        <div
          className={`h-full rounded-full transition-all ${isLoaded ? 'bg-emerald-500' : 'bg-sky-500'}`}
          style={{ width: `${usedPercent}%` }}
        />
      </div>
      <div className="text-xs text-slate-500">
        {gpu.memory_used_mb}MB / {gpu.memory_total_mb}MB ({usedPercent.toFixed(1)}%)
      </div>
    </div>
  )
}

// ===== Router信息卡片 =====

function RouterInfoCard({ router }: { router: RouterModel }) {
  return (
    <div className="assembly-router-info">
      <div className="min-w-0">
        <div className="flex min-w-0 items-center gap-2">
          <Activity size={16} className="shrink-0 text-amber-400" />
          <span className="truncate font-semibold text-slate-100" title={router.name}>
            {router.name}
          </span>
        </div>
        <div className="mt-1 text-xs leading-5 text-slate-400">{router.description}</div>
      </div>
      <div className="assembly-router-info__meta">
        {router.router_type && <span>{router.router_type}</span>}
        <span>{router.num_classes} 个分类</span>
      </div>
      <div className="assembly-router-info__classes">
        {router.class_names.map((cls, idx) => (
          <span
            key={cls}
            className={`rounded px-2 py-0.5 text-xs ${
              cls === 'normal'
                ? 'bg-slate-600/40 text-slate-300'
                : cls === 'diff_reaction'
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : cls === 'diff_sorp'
                    ? 'bg-sky-500/20 text-sky-300'
                    : cls === 'burgers'
                      ? 'bg-violet-500/20 text-violet-300'
                      : 'bg-slate-600/40 text-slate-300'
            }`}
          >
            {idx}: {cls}
          </span>
        ))}
      </div>
    </div>
  )
}

// ===== 模型选择下拉框组件 =====

function ModelSelector({
  title,
  icon: Icon,
  models,
  selected,
  onSelect,
  color,
  loadedInfo,
  placeholder,
  multiple = false,
}: {
  title: string
  icon: React.ElementType
  models: {
    name: string
    description?: string
    trained: boolean
    gpu_id?: number
    output_dim?: number
    simulator?: string
  }[]
  selected: string | string[] | null
  onSelect: (name: string | string[]) => void
  color: string
  loadedInfo?: { loaded: boolean; gpu_id?: number }
  placeholder?: string
  multiple?: boolean
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const colorClasses: Record<string, string> = {
    amber: 'border-amber-500/40 bg-amber-500/15',
    sky: 'border-sky-500/40 bg-sky-500/15',
    emerald: 'border-emerald-500/40 bg-emerald-500/15',
    violet: 'border-violet-500/40 bg-violet-500/15',
  }
  const iconColors: Record<string, string> = {
    amber: 'text-amber-400',
    sky: 'text-sky-400',
    emerald: 'text-emerald-400',
    violet: 'text-violet-400',
  }

  const selectedArray = multiple ? ((selected || []) as string[]) : [selected as string]
  const selectedSet = new Set(selectedArray.filter(s => s))

  const handleSelect = (name: string) => {
    if (multiple) {
      const newSet = new Set(selectedSet)
      if (newSet.has(name)) {
        newSet.delete(name)
      } else {
        newSet.add(name)
      }
      onSelect(Array.from(newSet))
    } else {
      onSelect(name)
      setIsOpen(false)
    }
  }

  const displayText = multiple
    ? selectedArray.length > 0
      ? `${selectedArray.length} 个已选择`
      : placeholder || '-- 请选择模型 --'
    : selectedArray[0] || placeholder || '-- 请选择模型 --'

  return (
    <div className={`relative rounded-lg border p-3 ${colorClasses[color] || ''}`} ref={dropdownRef}>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon size={16} className={iconColors[color] || ''} />
          <span className="font-semibold text-slate-100">{title}</span>
        </div>
        {loadedInfo?.loaded && (
          <span className="badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300 text-xs">
            GPU {loadedInfo.gpu_id}
          </span>
        )}
      </div>

      {/* 下拉框按钮 */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex w-full cursor-pointer items-center justify-between rounded-md border bg-slate-900/50 px-3 py-2 text-sm text-slate-200
          focus:outline-none
          ${!selected || (multiple && selectedArray.length === 0) ? 'text-slate-400' : ''}`}
      >
        <span className="truncate">{displayText}</span>
        <ChevronDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* 下拉选项 */}
      {isOpen && (
        <div className="absolute left-0 right-0 z-10 mt-1 max-h-60 min-w-[200px] overflow-auto rounded-lg border border-slate-600 bg-slate-900 shadow-lg">
          {models.length === 0 ? (
            <div className="p-3 text-xs text-slate-400">无可用模型</div>
          ) : (
            models.map(m => (
              <button
                key={m.name}
                onClick={() => handleSelect(m.name)}
                disabled={!m.trained && !multiple}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-slate-800 transition-colors flex items-center justify-between
                  ${selectedSet.has(m.name) ? 'bg-slate-800/50 text-slate-100' : 'text-slate-300'}
                  ${!m.trained && !multiple ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <div>
                  <div className="font-medium">{m.name}</div>
                  {m.simulator && <div className="text-xs text-slate-400">simulator: {m.simulator}</div>}
                  {m.output_dim && <div className="text-xs text-slate-400">dim: {m.output_dim}</div>}
                </div>
                {selectedSet.has(m.name) && <CheckCircle2 size={14} className="text-emerald-400" />}
                {!m.trained && <span className="text-xs text-amber-400">未训练</span>}
              </button>
            ))
          )}
        </div>
      )}

      {/* 已选择的模型列表（多选模式） */}
      {multiple && selectedArray.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {selectedArray.map(name => (
            <span
              key={name}
              className="px-2 py-1 rounded-lg bg-slate-800/50 text-xs text-slate-200 flex items-center gap-1"
            >
              {name}
              <button onClick={() => handleSelect(name)} className="hover:text-red-400">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default function AssemblyPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<AssemblyStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedLLM, setSelectedLLM] = useState<string | null>(null)
  const [selectedRouter, setSelectedRouter] = useState<string | null>(null)
  const [selectedText2Comp, setSelectedText2Comp] = useState<string | null>(null)
  const [selectedFNO, setSelectedFNO] = useState<string | null>(null)
  const [expertExecutor, setExpertExecutor] = useState<'fno' | 'uploaded'>('fno')
  const [selectedUploadedExpert, setSelectedUploadedExpert] = useState<string | null>(null)
  const [selectedGPU, setSelectedGPU] = useState<number>(0)
  const [loadingModels, setLoadingModels] = useState(false)

  // Prompt管理状态
  const [showPromptPanel, setShowPromptPanel] = useState(false)
  const [domains, setDomains] = useState<DomainInfo[]>([])
  const [selectedDomain, setSelectedDomain] = useState('')
  const [promptLanguage, setPromptLanguage] = useState('zh')
  const [promptText, setPromptText] = useState('')
  const [promptSaving, setPromptSaving] = useState(false)

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(() => {
      if (!loadingModels) fetchGPUs()
    }, 5000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    api.getDomains().then(setDomains).catch(console.error)
    api
      .getPrompt()
      .then(data => setPromptText(data.piern_system_prompt))
      .catch(console.error)
  }, [])

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const data = await api.getAssemblyStatus()
      setStatus(data)
      const loadedLLM = data.loaded_models?.llm?.path
        ? data.llms?.find((l: LLMInfo) => l.path === data.loaded_models.llm.path)
        : null
      const loadedRouter = data.loaded_models?.router?.path
        ? data.routers?.find((r: RouterModel) => r.path === data.loaded_models.router.path)
        : null
      const loadedText2CompPath = data.loaded_models?.text2comp?.paths?.[0]
      const loadedText2Comp = loadedText2CompPath
        ? data.text2comps?.find((t: Text2CompModel) => t.path === loadedText2CompPath)
        : null
      const loadedFNOPath = data.loaded_models?.fno?.paths?.[0]
      const loadedFNO = loadedFNOPath ? data.fno_experts?.find((f: FNOExpert) => f.path === loadedFNOPath) : null
      const loadedUploadedId = data.loaded_models?.uploaded_expert?.model_id
      const loadedUploadedExpert = loadedUploadedId
        ? data.custom_experts?.find((expert: UploadedExpert) => expert.model_id === loadedUploadedId)
        : null

      if (loadedLLM) {
        setSelectedLLM(loadedLLM.name)
      } else if (data.llms?.length > 0 && !selectedLLM) {
        // 默认选择对话模型，避开Guard/Reranker/Embedding等专用模型
        const chatModels = data.llms.filter(
          (l: LLMInfo) =>
            l.name.includes('Instruct') ||
            l.name.includes('0.6B') ||
            (l.name.includes('4B') &&
              !l.name.includes('Guard') &&
              !l.name.includes('Reranker') &&
              !l.name.includes('Embedding')),
        )
        if (chatModels.length > 0) setSelectedLLM(chatModels[0].name)
        else setSelectedLLM(data.llms[0].name)
      }
      if (loadedRouter) {
        setSelectedRouter(loadedRouter.name)
      } else if (data.routers?.length > 0 && !selectedRouter) {
        const trainedRouter = data.routers.find((r: RouterModel) => r.trained)
        if (trainedRouter) setSelectedRouter(trainedRouter.name)
      }
      if (loadedText2Comp) {
        setSelectedText2Comp(loadedText2Comp.name)
      } else if (data.text2comps?.length > 0 && !selectedText2Comp) {
        const trained = data.text2comps.find((t: Text2CompModel) => t.trained)
        if (trained) setSelectedText2Comp(trained.name)
      }
      if (loadedFNO) {
        setSelectedFNO(loadedFNO.name)
      } else if (data.fno_experts?.length > 0 && !selectedFNO) {
        const trained = data.fno_experts.find((f: FNOExpert) => f.trained)
        if (trained) setSelectedFNO(trained.name)
      }
      if (data.loaded_models?.uploaded_expert?.executor === 'uploaded') setExpertExecutor('uploaded')
      if (loadedUploadedExpert) {
        setSelectedUploadedExpert(loadedUploadedExpert.name)
      } else if (data.custom_experts?.length > 0 && !selectedUploadedExpert) {
        setSelectedUploadedExpert(data.custom_experts[0].name)
      }
    } catch (e) {
      console.error('Failed to fetch assembly status:', e)
    } finally {
      setLoading(false)
    }
  }

  const fetchGPUs = async () => {
    try {
      const gpus = await api.getAssemblyGPUs()
      if (status) setStatus({ ...status, gpus })
    } catch (e) {
      console.error('Failed to fetch GPU status:', e)
    }
  }

  const handleLoadAll = async () => {
    if (!selectedLLM || !status) return
    const llmInfo = status.llms.find(l => l.name === selectedLLM)
    if (!llmInfo) return
    const routerInfo = status.routers.find(r => r.name === selectedRouter)
    const text2CompInfo = status.text2comps.find(t => t.name === selectedText2Comp)
    const fnoInfo = status.fno_experts.find(f => f.name === selectedFNO)
    const uploadedInfo = status.custom_experts?.find(expert => expert.name === selectedUploadedExpert)
    if (expertExecutor === 'uploaded' && !uploadedInfo) {
      alert('请选择 Uploaded Expert')
      return
    }

    setLoadingModels(true)
    try {
      const result = await api.loadAssemblyModels({
        llm_path: llmInfo.path,
        llm_gpu_id: selectedGPU,
        router_gpu_id: undefined,
        router_path: routerInfo?.path,
        text2comp_path: text2CompInfo?.path,
        fno_path: expertExecutor === 'fno' ? fnoInfo?.path : undefined,
        expert_executor: expertExecutor,
        uploaded_expert_id: expertExecutor === 'uploaded' ? uploadedInfo?.model_id : undefined,
        force_split: false,
        auto_sync: true,
      })
      console.log('Load result:', result)
      await fetchStatus()
    } catch (e) {
      console.error('Load failed:', e)
      alert(`加载失败: ${e}`)
    } finally {
      setLoadingModels(false)
    }
  }

  const handleUnload = async () => {
    setLoadingModels(true)
    try {
      await api.unloadAssemblyModels()
      // 清除所有选择状态
      setSelectedLLM(null)
      setSelectedRouter(null)
      setSelectedText2Comp(null)
      setSelectedFNO(null)
      setExpertExecutor('fno')
      await fetchStatus()
    } catch (e) {
      console.error('Unload failed:', e)
    } finally {
      setLoadingModels(false)
    }
  }

  const handleGeneratePrompt = async () => {
    if (!selectedDomain) return
    try {
      const res = await api.generatePrompt({ simulator: selectedDomain, language: promptLanguage })
      setPromptText(res.prompt)
    } catch (e) {
      console.error('Generate prompt failed:', e)
      alert(`生成失败: ${e}`)
    }
  }

  const handleSavePrompt = async () => {
    if (!promptText.trim()) return
    setPromptSaving(true)
    try {
      await api.updatePrompt({ piern_system_prompt: promptText })
      alert('Prompt已保存')
    } catch (e) {
      console.error('Save prompt failed:', e)
      alert(`保存失败: ${e}`)
    } finally {
      setPromptSaving(false)
    }
  }

  const isLoaded = status?.loaded_models?.llm?.loaded || false
  const currentRouter = status?.routers?.find(r => r.name === selectedRouter)
  const selectedText2CompInfo = status?.text2comps?.find(t => t.name === selectedText2Comp)
  const selectedUploadedExpertInfo = status?.custom_experts?.find(expert => expert.name === selectedUploadedExpert)
  const dimMismatch =
    expertExecutor === 'uploaded' &&
    selectedText2CompInfo &&
    selectedUploadedExpertInfo &&
    selectedText2CompInfo.output_dim !== selectedUploadedExpertInfo.input_dim

  if (loading) {
    return (
      <div className="training-page">
        <div className="training-page__body flex items-center justify-center">
          <div className="flex items-center gap-3">
            <Loader2 size={20} className="animate-spin text-slate-400" />
            <span className="text-slate-400">加载模型装配状态...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-4">
          {/* Header */}
          <section className="training-hero training-hero--compact">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="training-eyebrow">模型拼装</div>
                <h1 className="mt-2 text-2xl font-semibold text-white">配置 Piern 推理链路</h1>
                <p className="mt-1 text-sm text-slate-400">选择并装载 Router、Text2Comp 与 Expert。</p>
                {status?.architecture_note && <p className="mt-1 text-xs text-slate-500">{status.architecture_note}</p>}
              </div>
              <button className="btn-ghost shrink-0" onClick={() => navigate('/training')}>
                <Settings size={14} />
                返回概览
              </button>
            </div>
          </section>

          {/* GPU Status - 全部8个GPU */}
          <section className="rounded-lg border border-slate-700/40 bg-slate-900/30 p-3">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Server size={18} className="text-sky-400" />
                <span className="font-semibold text-slate-100">GPU 状态 ({status?.gpus?.length || 0}个)</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-400" htmlFor="assembly-gpu">
                  运行 GPU
                </label>
                <select
                  id="assembly-gpu"
                  value={selectedGPU}
                  onChange={e => setSelectedGPU(Number(e.target.value))}
                  className="select w-auto min-w-[12rem] py-1.5 text-xs"
                >
                  {status?.gpus?.map(gpu => (
                    <option key={gpu.index} value={gpu.index}>
                      GPU {gpu.index} ({gpu.memory_free_mb}MB空闲)
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
              {status?.gpus?.map(gpu => (
                <GPUCard key={gpu.index} gpu={gpu} isLoaded={status.loaded_models?.llm?.gpu_id === gpu.index} compact />
              ))}
            </div>
          </section>

          <div className="assembly-overview-grid">
            <section className="assembly-panel">
              <div className="assembly-panel__header">
                <Workflow size={16} className="text-violet-400" />
                <span>推理链路</span>
              </div>
              <div className="assembly-flow">
                {[
                  'LLM',
                  'Router',
                  'Text2Comp',
                  expertExecutor === 'uploaded' ? 'Uploaded Expert' : 'FNO',
                  'Output',
                ].map((name, idx) => {
                  const icons = [Box, Box, Layers, Activity, Zap]
                  const Icon = icons[idx]
                  const loadedKeys = ['llm', 'router', 'text2comp', 'fno'] as const
                  const isLd =
                    idx === 4
                      ? true
                      : idx === 3 && expertExecutor === 'uploaded'
                        ? Boolean(status?.loaded_models?.uploaded_expert?.loaded)
                        : Boolean(status?.loaded_models?.[loadedKeys[idx] as (typeof loadedKeys)[number]]?.loaded)
                  const colors = ['amber', 'amber', 'sky', 'emerald', 'violet']
                  const color = colors[idx]
                  return (
                    <div key={name} className="assembly-flow__segment">
                      <div className="assembly-flow__step">
                        <span
                          className={`assembly-flow__icon ${isLd ? 'border-emerald-500/40 bg-emerald-500/15' : `border-${color}-500/40 bg-${color}-500/15`}`}
                        >
                          <Icon size={18} className={isLd ? 'text-emerald-400' : `text-${color}-400`} />
                        </span>
                        <span title={name}>{name}</span>
                      </div>
                      {idx < 4 && <ArrowRight size={15} className="shrink-0 text-slate-500" />}
                    </div>
                  )
                })}
              </div>
            </section>

            {selectedRouter && currentRouter && (
              <section className="assembly-panel assembly-panel--router">
                <div className="assembly-panel__header">
                  <Info size={16} className="text-amber-400" />
                  <span>Router 分类信息</span>
                </div>
                <RouterInfoCard router={currentRouter} />
                <div className="mt-2 text-xs text-slate-400">
                  可路由至 {currentRouter.class_names.filter(c => c !== 'normal').join('、')} 等专家模型
                </div>
              </section>
            )}
          </div>

          {/* Load/Unload Controls */}
          <section className="rounded-lg border border-violet-500/20 bg-violet-500/8 p-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Zap size={18} className="text-violet-400" />
                <span className="font-semibold text-slate-100">装载设置</span>
              </div>
              <div className="flex items-center gap-3">
                {!isLoaded ? (
                  <button
                    className="btn-primary"
                    onClick={handleLoadAll}
                    disabled={loadingModels || !selectedLLM || Boolean(dimMismatch)}
                  >
                    {loadingModels ? <Loader2 size={14} className="animate-spin" /> : <Power size={14} />}
                    {loadingModels ? '加载中...' : '一键加载所有模型'}
                  </button>
                ) : (
                  <button
                    className="btn-secondary border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20"
                    onClick={handleUnload}
                    disabled={loadingModels}
                  >
                    {loadingModels ? <Loader2 size={14} className="animate-spin" /> : <PowerOff size={14} />}
                    {loadingModels ? '卸载中...' : '卸载所有模型'}
                  </button>
                )}
              </div>
            </div>
            <div className="assembly-control-grid">
              <div>
                <div className="mb-2 text-xs font-semibold text-slate-400">专家执行器</div>
                <div className="training-segmented">
                  {(['fno', 'uploaded'] as const).map(mode => (
                    <button
                      key={mode}
                      className={`training-segmented__button ${expertExecutor === mode ? 'training-segmented__button--active' : ''}`}
                      onClick={() => setExpertExecutor(mode)}
                    >
                      {mode === 'fno' ? 'FNO Expert' : 'Uploaded Expert'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="assembly-selection-summary">
                <span title={selectedLLM || undefined}>LLM · {selectedLLM || '未选择'}</span>
                <span title={selectedRouter || undefined}>Router · {selectedRouter || '未选择'}</span>
                <span title={selectedText2Comp || undefined}>Text2Comp · {selectedText2Comp || '未选择'}</span>
                <span title={(expertExecutor === 'uploaded' ? selectedUploadedExpert : selectedFNO) || undefined}>
                  Expert ·{' '}
                  {expertExecutor === 'uploaded' ? selectedUploadedExpert || '未选择' : selectedFNO || '未选择'}
                </span>
                <span>GPU · {selectedGPU}</span>
              </div>
            </div>
            {expertExecutor === 'uploaded' && dimMismatch && (
              <div className="mt-3 rounded-lg border border-rose-500/20 bg-rose-500/8 px-3 py-2 text-xs text-rose-300">
                Text2Comp 输出维度 {selectedText2CompInfo?.output_dim} 与 Uploaded Expert 输入维度{' '}
                {selectedUploadedExpertInfo?.input_dim} 不匹配。
              </div>
            )}
          </section>

          {/* Model Selection - 多选下拉框 */}
          <section className="grid gap-4 lg:grid-cols-4">
            <ModelSelector
              title="LLM 模型"
              icon={Box}
              models={status?.llms?.map(l => ({ name: l.name, description: l.size, trained: l.downloaded })) || []}
              selected={selectedLLM}
              onSelect={name => setSelectedLLM(name as string)}
              color="violet"
              placeholder="选择语言模型"
              loadedInfo={status?.loaded_models?.llm}
            />
            <ModelSelector
              title="Router"
              icon={Box}
              models={status?.routers || []}
              selected={selectedRouter}
              onSelect={name => setSelectedRouter(name as string)}
              color="amber"
              placeholder="选择路由模型"
              loadedInfo={status?.loaded_models?.router}
            />
            <ModelSelector
              title="Text2Comp"
              icon={Layers}
              models={status?.text2comps?.map(t => ({ ...t, description: `dim:${t.output_dim}` })) || []}
              selected={selectedText2Comp}
              onSelect={name => setSelectedText2Comp(name as string)}
              color="sky"
              placeholder="选择文生计算模型"
              loadedInfo={status?.loaded_models?.text2comp}
            />
            {expertExecutor === 'fno' ? (
              <ModelSelector
                title="FNO Expert"
                icon={Activity}
                models={status?.fno_experts?.map(f => ({ ...f, description: `in:${f.input_dim}` })) || []}
                selected={selectedFNO}
                onSelect={name => setSelectedFNO(name as string)}
                color="emerald"
                placeholder="选择FNO专家"
                loadedInfo={status?.loaded_models?.fno}
              />
            ) : (
              <ModelSelector
                title="Uploaded Expert"
                icon={Activity}
                models={
                  status?.custom_experts?.map(expert => ({
                    name: expert.name,
                    description: `in:${expert.input_dim ?? '--'} out:${expert.output_dim ?? '--'}`,
                    trained:
                      expert.status === 'active' && expert.assembly_enabled !== false && expert.trained !== false,
                    output_dim: expert.input_dim ?? undefined,
                    simulator: expert.simulator,
                  })) || []
                }
                selected={selectedUploadedExpert}
                onSelect={name => setSelectedUploadedExpert(name as string)}
                color="emerald"
                placeholder="选择上传专家"
                loadedInfo={{
                  loaded: Boolean(status?.loaded_models?.uploaded_expert?.loaded),
                  gpu_id: status?.loaded_models?.text2comp?.gpu_id,
                }}
              />
            )}
          </section>

          {/* Prompt 管理 */}
          <section className="rounded-lg border border-slate-700/40 bg-slate-900/30 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <FileText size={18} className="text-violet-400" />
                <span className="font-semibold text-slate-100">Prompt 管理</span>
              </div>
              <button
                className="text-xs text-slate-400 hover:text-slate-300"
                onClick={() => setShowPromptPanel(!showPromptPanel)}
              >
                {showPromptPanel ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            {showPromptPanel && (
              <div className="space-y-4">
                {/* 流程说明 */}
                <div className="rounded-lg border border-sky-500/20 bg-sky-500/8 p-3 text-xs text-slate-300">
                  <div className="flex items-center gap-2 mb-2">
                    <Info size={14} className="text-sky-400" />
                    <span className="font-medium">Prompt生成流程</span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-400">
                    <span>1. 选择 Simulator</span>
                    <ArrowRight size={12} />
                    <span>2. 从 DOMAIN_REGISTRY 获取 domain_context</span>
                    <ArrowRight size={12} />
                    <span>3. 自动生成领域专属 Prompt</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-slate-400 mb-1.5 block">Simulator</label>
                    <select
                      className="select w-full"
                      value={selectedDomain}
                      onChange={e => setSelectedDomain(e.target.value)}
                    >
                      <option value="">-- 选择 Simulator --</option>
                      {domains.map(d => (
                        <option key={d.simulator} value={d.simulator}>
                          {d.simulator}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-slate-400 mb-1.5 block">语言</label>
                    <div className="flex gap-2">
                      <button
                        className={`px-3 py-1.5 rounded-lg text-xs transition-all ${promptLanguage === 'zh' ? 'bg-violet-500/15 border border-violet-500/40 text-violet-300' : 'bg-slate-800/40 border border-slate-700/40 text-slate-500 hover:text-slate-300'}`}
                        onClick={() => setPromptLanguage('zh')}
                      >
                        中文
                      </button>
                      <button
                        className={`px-3 py-1.5 rounded-lg text-xs transition-all ${promptLanguage === 'en' ? 'bg-violet-500/15 border border-violet-500/40 text-violet-300' : 'bg-slate-800/40 border border-slate-700/40 text-slate-500 hover:text-slate-300'}`}
                        onClick={() => setPromptLanguage('en')}
                      >
                        英文
                      </button>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-xs text-slate-400 mb-1.5 block">
                    System Prompt <span className="ml-2 text-slate-600">可直接编辑</span>
                  </label>
                  <textarea
                    className="w-full resize-y whitespace-pre-wrap rounded-lg border border-slate-700/40 bg-slate-900/30 p-3 font-mono text-sm text-slate-200 focus:border-violet-500/40 focus:outline-none"
                    rows={8}
                    value={promptText}
                    onChange={e => setPromptText(e.target.value)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="text-xs text-slate-500">当前长度: {promptText.length} 字符</div>
                  <div className="flex gap-2">
                    <button className="btn-secondary" onClick={handleGeneratePrompt} disabled={!selectedDomain}>
                      <Sparkles size={14} />从 Domain 生成
                    </button>
                    <button
                      className="btn-primary"
                      onClick={handleSavePrompt}
                      disabled={!promptText.trim() || promptSaving}
                    >
                      {promptSaving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                      {promptSaving ? '保存中...' : '保存 Prompt'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
