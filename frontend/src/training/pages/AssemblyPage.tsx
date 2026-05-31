import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Workflow,
  PlayCircle,
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
  Upload,
  Info,
  Eye,
  EyeOff,
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

interface LoadedModels {
  llm: { loaded: boolean; gpu_id?: number }
  router: { loaded: boolean; gpu_id?: number }
  text2comp: { loaded: boolean; gpu_id?: number }
  fno: { loaded: boolean; gpu_id?: number }
}

interface AssemblyStatus {
  llms: LLMInfo[]
  routers: RouterModel[]
  text2comps: Text2CompModel[]
  fno_experts: FNOExpert[]
  gpus: GPUInfo[]
  loaded_models: LoadedModels
  gpu_available: boolean
  architecture_note?: string
}

interface TestResult {
  router_prediction: string
  router_class_name: string
  final_answer: string
  full_response: string
  expert_used?: string
  expert_output?: number[]
  latency_ms: number
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
      className={`rounded-xl border p-3 ${isLoaded ? 'border-emerald-500/40 bg-emerald-500/10' : 'border-slate-700/40 bg-slate-900/30'}`}
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
    <div className="rounded-xl border border-amber-500/40 bg-amber-500/15 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Activity size={16} className="text-amber-400" />
        <span className="font-semibold text-slate-100">{router.name}</span>
      </div>
      <div className="text-xs text-slate-400 mb-2">{router.description}</div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs text-slate-400">分类数:</span>
        <span className="text-xs font-medium text-amber-300">{router.num_classes}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {router.class_names.map((cls, idx) => (
          <span
            key={cls}
            className={`px-2 py-0.5 rounded text-xs ${
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
  selected: string | string[]
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

  const selectedArray = multiple ? (selected as string[]) : [selected as string]
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
    <div className={`rounded-2xl border p-4 ${colorClasses[color] || ''} relative`} ref={dropdownRef}>
      <div className="flex items-center justify-between mb-3">
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
        className={`w-full rounded-xl border bg-slate-900/50 px-3 py-2.5 text-sm text-slate-200
          focus:outline-none cursor-pointer flex items-center justify-between
          ${!selected || (multiple && selectedArray.length === 0) ? 'text-slate-400' : ''}`}
      >
        <span className="truncate">{displayText}</span>
        <ChevronDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* 下拉选项 */}
      {isOpen && (
        <div className="absolute z-10 mt-1 left-0 right-0 min-w-[200px] rounded-xl border border-slate-600 bg-slate-900 shadow-lg overflow-auto max-h-60">
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

// ===== 专家模型路径上传组件 =====

function ExpertUploadSection({
  onUpload,
  uploading,
}: {
  onUpload: (path: string, type: 'text2comp' | 'fno', simulator: string, outputDim?: number) => void
  uploading: boolean
}) {
  const [showUpload, setShowUpload] = useState(false)
  const [uploadType, setUploadType] = useState<'text2comp' | 'fno'>('text2comp')
  const [modelPath, setModelPath] = useState('')
  const [simulator, setSimulator] = useState('')
  const [outputDim, setOutputDim] = useState(32)

  const handleUploadClick = () => {
    if (modelPath && simulator) {
      onUpload(modelPath, uploadType, simulator, outputDim)
      setShowUpload(false)
      setModelPath('')
      setSimulator('')
    }
  }

  return (
    <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <Upload size={18} className="text-violet-400" />
          <span className="font-semibold text-slate-100">自定义专家模型</span>
        </div>
        <button onClick={() => setShowUpload(!showUpload)} className="text-xs text-slate-400 hover:text-slate-300">
          {showUpload ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {showUpload && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 mb-1 block">模型类型</label>
              <select
                value={uploadType}
                onChange={e => setUploadType(e.target.value as 'text2comp' | 'fno')}
                className="select w-full"
              >
                <option value="text2comp">Text2Comp</option>
                <option value="fno">FNO Expert</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Simulator名称</label>
              <input
                type="text"
                value={simulator}
                onChange={e => setSimulator(e.target.value)}
                placeholder="如: diff-sorp"
                className="input w-full"
              />
            </div>
          </div>

          <div>
            <label className="text-xs text-slate-400 mb-1 block">模型路径 (.pt文件)</label>
            <input
              type="text"
              value={modelPath}
              onChange={e => setModelPath(e.target.value)}
              placeholder="/path/to/model.pt"
              className="input w-full"
            />
          </div>

          {uploadType === 'text2comp' && (
            <div>
              <label className="text-xs text-slate-400 mb-1 block">输出维度 (output_dim)</label>
              <input
                type="number"
                value={outputDim}
                onChange={e => setOutputDim(Number(e.target.value))}
                className="input w-full"
              />
            </div>
          )}

          <button
            onClick={handleUploadClick}
            disabled={!modelPath || !simulator || uploading}
            className="btn-primary w-full"
          >
            {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
            {uploading ? '上传中...' : '添加模型'}
          </button>
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
  const [selectedGPU, setSelectedGPU] = useState<number>(0)
  const [testInput, setTestInput] = useState('')
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const [testing, setTesting] = useState(false)
  const [loadingModels, setLoadingModels] = useState(false)
  const [showFullResponse, setShowFullResponse] = useState(false)
  const [uploadingModel, setUploadingModel] = useState(false)

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
      if (data.llms?.length > 0 && !selectedLLM) {
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
      if (data.routers?.length > 0 && !selectedRouter) {
        const trainedRouter = data.routers.find((r: RouterModel) => r.trained)
        if (trainedRouter) setSelectedRouter(trainedRouter.name)
      }
      if (data.text2comps?.length > 0 && !selectedText2Comp) {
        const trained = data.text2comps.find((t: Text2CompModel) => t.trained)
        if (trained) setSelectedText2Comp(trained.name)
      }
      if (data.fno_experts?.length > 0 && !selectedFNO) {
        const trained = data.fno_experts.find((f: FNOExpert) => f.trained)
        if (trained) setSelectedFNO(trained.name)
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

    setLoadingModels(true)
    try {
      const result = await api.loadAssemblyModels({
        llm_path: llmInfo.path,
        llm_gpu_id: selectedGPU,
        router_gpu_id: undefined,
        force_split: false,
        auto_sync: true,
        experts: selectedText2Comp
          ? [
              {
                simulator: status?.text2comps.find(t => t.name === selectedText2Comp)?.simulator || selectedText2Comp,
              },
            ]
          : [],
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
      await fetchStatus()
    } catch (e) {
      console.error('Unload failed:', e)
    } finally {
      setLoadingModels(false)
    }
  }

  const handleUploadExpert = async (path: string, type: 'text2comp' | 'fno', simulator: string, outputDim?: number) => {
    setUploadingModel(true)
    try {
      // 这里需要后端API支持，暂时用console.log
      console.log('Upload expert:', { path, type, simulator, outputDim })
      alert('模型路径已记录，需要后端API支持注册新模型')
      await fetchStatus()
    } catch (e) {
      console.error('Upload failed:', e)
      alert(`上传失败: ${e}`)
    } finally {
      setUploadingModel(false)
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

  const runTest = async () => {
    if (!testInput.trim()) return
    setTesting(true)
    setTestResult(null)
    setShowFullResponse(false)
    try {
      const result = await api.testAssembly({
        config: {
          main_llm_path: status?.llms.find(l => l.name === selectedLLM)?.path,
          gpu_config: { llm_gpu_ids: [selectedGPU] },
        },
        test_input: testInput,
      })
      setTestResult({
        router_prediction: result.router_prediction || '',
        router_class_name: result.router_class_name || result.router_prediction,
        final_answer: result.final_answer || result.first_cot_result?.split('\n').pop() || '',
        full_response: result.first_cot_result || '',
        expert_used: result.expert_used,
        expert_output: result.expert_output,
        latency_ms: result.latency_ms,
      })
    } catch (e) {
      console.error('Test failed:', e)
      alert(`测试失败: ${e}`)
    } finally {
      setTesting(false)
    }
  }

  const isLoaded = status?.loaded_models?.llm?.loaded || false
  const canTest = isLoaded && testInput.trim()
  const currentRouter = status?.routers?.find(r => r.name === selectedRouter)

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
        <div className="space-y-5 p-5">
          {/* Header */}
          <section className="training-hero">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
              <div>
                <div className="training-eyebrow">模型拼装</div>
                <h1 className="mt-3 text-[1.8rem] font-semibold tracking-tight text-white xl:text-[2.1rem]">
                  PiERN Pipeline Builder
                </h1>
                <p className="mt-2 text-slate-400 text-sm">Router → Text2Comp → FNO Expert 多模型组合拼装界面</p>
                {status?.architecture_note && <p className="mt-1 text-xs text-slate-500">{status.architecture_note}</p>}
              </div>
              <button className="btn-secondary" onClick={() => navigate('/training')}>
                <Settings size={14} />
                返回概览
              </button>
            </div>
          </section>

          {/* GPU Status - 全部8个GPU */}
          <section className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Server size={18} className="text-sky-400" />
                <span className="font-semibold text-slate-100">GPU 状态 ({status?.gpus?.length || 0}个)</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-400">选择GPU:</label>
                <select
                  value={selectedGPU}
                  onChange={e => setSelectedGPU(Number(e.target.value))}
                  className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-sm text-slate-200"
                >
                  {status?.gpus?.map(gpu => (
                    <option key={gpu.index} value={gpu.index}>
                      GPU {gpu.index} ({gpu.memory_free_mb}MB空闲)
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="grid gap-2 grid-cols-4 md:grid-cols-8">
              {status?.gpus?.map(gpu => (
                <GPUCard key={gpu.index} gpu={gpu} isLoaded={status.loaded_models?.llm?.gpu_id === gpu.index} compact />
              ))}
            </div>
          </section>

          {/* Architecture Diagram */}
          <section className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
            <div className="flex items-center justify-center gap-3 py-3 overflow-x-auto">
              {['LLM', 'Router', 'Text2Comp', 'FNO', 'Output'].map((name, idx) => {
                const icons = [Box, Box, Layers, Activity, Zap]
                const Icon = icons[idx]
                const loadedKey = ['llm', 'router', 'text2comp', 'fno', null] as const
                const key = loadedKey[idx]
                const isLd = key === null ? true : status?.loaded_models?.[key]?.loaded
                const colors = ['amber', 'amber', 'sky', 'emerald', 'violet']
                const color = colors[idx]
                return (
                  <div key={name} className="flex flex-col items-center gap-1 min-w-[60px]">
                    <div
                      className={`rounded-xl border p-2 ${isLd ? `border-emerald-500/40 bg-emerald-500/15` : `border-${color}-500/40 bg-${color}-500/15`}`}
                    >
                      <Icon size={20} className={isLd ? 'text-emerald-400' : `text-${color}-400`} />
                    </div>
                    <span className="text-xs text-slate-300">{name}</span>
                    {idx < 4 && <ArrowRight size={16} className="text-slate-500" />}
                  </div>
                )
              })}
            </div>
          </section>

          {/* Router信息显示 */}
          {selectedRouter && currentRouter && (
            <section className="rounded-2xl border border-amber-500/20 bg-amber-500/8 p-4">
              <div className="flex items-center gap-3 mb-3">
                <Info size={16} className="text-amber-400" />
                <span className="font-semibold text-slate-100">Router 分类信息</span>
              </div>
              <RouterInfoCard router={currentRouter} />
              <div className="mt-2 text-xs text-slate-400">
                该Router可将输入路由到 {currentRouter.class_names.filter(c => c !== 'normal').join('、')} 等专家模型
              </div>
            </section>
          )}

          {/* Load/Unload Controls */}
          <section className="rounded-2xl border border-violet-500/20 bg-violet-500/8 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Zap size={18} className="text-violet-400" />
                <span className="font-semibold text-slate-100">模型加载控制</span>
              </div>
              <div className="flex items-center gap-3">
                {!isLoaded ? (
                  <button className="btn-primary" onClick={handleLoadAll} disabled={loadingModels || !selectedLLM}>
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
            <div className="mt-3 text-xs text-slate-400">
              LLM: {selectedLLM || '未选择'} | Text2Comp: {selectedText2Comp || '未选择'} | FNO:{' '}
              {selectedFNO || '未选择'} → GPU {selectedGPU}
            </div>
          </section>

          {/* Model Selection - 多选下拉框 */}
          <section className="grid gap-4 lg:grid-cols-4">
            <ModelSelector
              title="LLM 模型"
              icon={Box}
              models={status?.llms?.map(l => ({ name: l.name, description: l.size, trained: l.downloaded })) || []}
              selected={selectedLLM || ''}
              onSelect={name => setSelectedLLM(String(name))}
              color="violet"
              placeholder="选择语言模型"
              loadedInfo={status?.loaded_models?.llm}
            />
            <ModelSelector
              title="Router"
              icon={Box}
              models={status?.routers || []}
              selected={selectedRouter || ''}
              onSelect={name => setSelectedRouter(String(name))}
              color="amber"
              placeholder="选择路由模型"
              loadedInfo={status?.loaded_models?.router}
            />
            <ModelSelector
              title="Text2Comp"
              icon={Layers}
              models={status?.text2comps?.map(t => ({ ...t, description: `dim:${t.output_dim}` })) || []}
              selected={selectedText2Comp || ''}
              onSelect={name => setSelectedText2Comp(String(name))}
              color="sky"
              placeholder="选择文生计算模型"
              loadedInfo={status?.loaded_models?.text2comp}
            />
            <ModelSelector
              title="FNO Expert"
              icon={Activity}
              models={status?.fno_experts?.map(f => ({ ...f, description: `in:${f.input_dim}` })) || []}
              selected={selectedFNO || ''}
              onSelect={name => setSelectedFNO(String(name))}
              color="emerald"
              placeholder="选择FNO专家"
              loadedInfo={status?.loaded_models?.fno}
            />
          </section>

          {/* 自定义专家模型上传 */}
          <ExpertUploadSection onUpload={handleUploadExpert} uploading={uploadingModel} />

          {/* Prompt 管理 */}
          <section className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
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
                <div className="rounded-xl border border-sky-500/20 bg-sky-500/8 p-3 text-xs text-slate-300">
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
                    className="w-full rounded-xl border border-slate-700/40 bg-slate-900/30 p-3 text-sm text-slate-200 font-mono whitespace-pre-wrap resize-y focus:border-violet-500/40 focus:outline-none"
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

          {/* Test Input */}
          <section className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
            <div className="flex items-center gap-3 mb-4">
              <PlayCircle size={18} className="text-violet-400" />
              <span className="font-semibold text-slate-100">推理测试</span>
              {isLoaded && (
                <span className="badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300">模型已加载</span>
              )}
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">用户输入文本</label>
                <textarea
                  className="w-full rounded-xl border border-slate-700/40 bg-slate-900/30 p-3 text-sm text-slate-200 focus:border-violet-500/40 focus:outline-none"
                  rows={3}
                  placeholder="例如：求解1维扩散-吸附问题..."
                  value={testInput}
                  onChange={e => setTestInput(e.target.value)}
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="text-xs text-slate-500">{canTest ? '模型已加载，可测试' : '请先加载模型'}</div>
                <button className="btn-primary" disabled={!canTest || testing} onClick={runTest}>
                  <PlayCircle size={14} />
                  {testing ? '运行中...' : '开始测试'}
                </button>
              </div>
            </div>
          </section>

          {/* Test Results - 改进显示 */}
          {testResult && (
            <section className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-4">
              <div className="flex items-center gap-3 mb-4">
                <Workflow size={18} className="text-emerald-400" />
                <span className="font-semibold text-slate-100">测试结果</span>
              </div>

              <div className="grid gap-3 lg:grid-cols-3">
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/8 p-3">
                  <div className="text-xs text-amber-400 mb-1">Router判断</div>
                  <div className="text-lg font-semibold text-slate-100">{testResult.router_class_name}</div>
                </div>
                {testResult.expert_used && (
                  <div className="rounded-xl border border-sky-500/20 bg-sky-500/8 p-3">
                    <div className="text-xs text-sky-400 mb-1">使用专家</div>
                    <div className="text-lg font-semibold text-slate-100">{testResult.expert_used}</div>
                  </div>
                )}
                <div className="rounded-xl border border-violet-500/20 bg-violet-500/8 p-3">
                  <div className="text-xs text-violet-400 mb-1">推理延迟</div>
                  <div className="text-lg font-semibold text-slate-100">{testResult.latency_ms.toFixed(2)} ms</div>
                </div>
              </div>

              {/* 最终答案 */}
              <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/8 p-3">
                <div className="text-xs text-emerald-400 mb-2">最终答案</div>
                <div className="text-sm text-slate-200 whitespace-pre-wrap">{testResult.final_answer}</div>
              </div>

              {/* 查看完整响应 */}
              <div className="mt-3 flex items-center justify-between">
                <button
                  className="text-xs text-slate-400 hover:text-slate-300 flex items-center gap-1"
                  onClick={() => setShowFullResponse(!showFullResponse)}
                >
                  {showFullResponse ? <EyeOff size={14} /> : <Eye size={14} />}
                  {showFullResponse ? '隐藏完整响应' : '查看完整响应'}
                </button>
              </div>
              {showFullResponse && (
                <div className="mt-2 rounded-xl border border-slate-700/40 bg-slate-900/50 p-3 max-h-60 overflow-auto">
                  <div className="text-xs text-slate-400 mb-2">完整LLM响应</div>
                  <div className="text-sm text-slate-200 whitespace-pre-wrap font-mono">{testResult.full_response}</div>
                </div>
              )}
            </section>
          )}

          {/* Current Selection Summary */}
          <section className="rounded-2xl border border-violet-500/20 bg-violet-500/8 p-4">
            <div className="flex items-center gap-3 mb-3">
              <Zap size={16} className="text-violet-400" />
              <span className="font-medium text-slate-100">当前选择</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-xs text-slate-400">LLM</div>
                <div className="text-slate-200 mt-1 truncate">{selectedLLM || '未选择'}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">Router</div>
                <div className="text-slate-200 mt-1 truncate">{selectedRouter || '未选择'}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">Text2Comp</div>
                <div className="text-slate-200 mt-1 truncate">{selectedText2Comp || '未选择'}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">FNO</div>
                <div className="text-slate-200 mt-1 truncate">{selectedFNO || '未选择'}</div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
