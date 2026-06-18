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
import type { ExpertModelInfo } from '../../lib/types'
import { TrainingSectionTitle as SectionTitle, TrainingUsageBar as UsageBar } from '../components/common'
import { gpuUsageLabel } from '../shared'

type AssemblyStatus = Awaited<ReturnType<typeof api.getAssemblyStatus>>
type ExecutorMode = 'fno' | 'uploaded'
type ResourceMode = 'auto' | 'manual'

type SelectableModel = {
  name: string
  path?: string
  trained?: boolean
  output_dim?: number | null
  input_dim?: number | null
  model_id?: string
  description?: string
}

function pickFirst<T extends SelectableModel>(items: T[] | undefined, predicate?: (item: T) => boolean): T | null {
  const list = items ?? []
  return list.find(item => predicate?.(item)) ?? list.find(item => item.trained) ?? list[0] ?? null
}

function pickLLM(status: AssemblyStatus | undefined) {
  const llms = status?.llms ?? []
  return (
    llms.find(item => item.name.includes('Instruct') || item.name.includes('0.6B')) ??
    llms.find(item => item.downloaded) ??
    llms[0] ??
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

function compatibleUploadedExperts(
  status: AssemblyStatus | undefined,
  text2comp: SelectableModel | null,
): ExpertModelInfo[] {
  const expectedDim = text2comp?.output_dim
  return (status?.custom_experts ?? []).filter(expert => {
    if (expert.status !== 'active' || expert.assembly_enabled === false) return false
    if (expectedDim == null || expert.input_dim == null) return true
    return expert.input_dim === expectedDim
  })
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
  const [executorMode, setExecutorMode] = useState<ExecutorMode>('fno')
  const [resourceMode, setResourceMode] = useState<ResourceMode>('auto')
  const [gpuId, setGpuId] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [testInput, setTestInput] = useState('')
  const [testResult, setTestResult] = useState<{
    final_answer?: string
    router_class_name?: string
    latency_ms?: number
  } | null>(null)

  const llm = useMemo(() => pickLLM(status), [status])
  const router = useMemo(() => pickFirst(status?.routers, item => item.trained), [status?.routers])
  const text2comp = useMemo(() => pickFirst(status?.text2comps, item => item.trained), [status?.text2comps])
  const fno = useMemo(() => pickFirst(status?.fno_experts, item => item.trained), [status?.fno_experts])
  const uploadedExperts = useMemo(() => compatibleUploadedExperts(status, text2comp), [status, text2comp])
  const uploaded = useMemo(() => uploadedExperts[0] ?? null, [uploadedExperts])
  const bestGpu = useMemo(() => pickGpu(status), [status])
  const selectedGpu = useMemo(
    () => (status?.gpus ?? []).find(gpu => gpu.index === gpuId) ?? bestGpu,
    [bestGpu, gpuId, status?.gpus],
  )
  const expert = executorMode === 'uploaded' ? uploaded : fno
  const isLoaded = Boolean(status?.loaded_models?.llm?.loaded)
  const canLoad = Boolean(llm && selectedGpu && (executorMode === 'fno' ? fno : uploaded))

  useEffect(() => {
    if (gpuId != null && (status?.gpus ?? []).some(gpu => gpu.index === gpuId)) return
    setGpuId(bestGpu?.index ?? null)
  }, [bestGpu?.index, gpuId, status?.gpus])

  useEffect(() => {
    if (executorMode === 'uploaded' && uploadedExperts.length === 0 && fno) setExecutorMode('fno')
  }, [executorMode, fno, uploadedExperts.length])

  const refresh = async () => {
    await mutate('assembly-status')
  }

  const load = async () => {
    if (!llm || !selectedGpu) {
      setActionError('缺少 LLM 或 GPU，无法加载。')
      return
    }
    if (executorMode === 'fno' && !fno) {
      setActionError('没有可用 FNO Expert。')
      return
    }
    if (executorMode === 'uploaded' && !uploaded) {
      setActionError('没有可用 Uploaded Expert，或输入维度不匹配。')
      return
    }
    setBusy(true)
    setActionError(null)
    setTestResult(null)
    try {
      await api.loadAssemblyModels({
        llm_path: llm.path,
        llm_gpu_id: resourceMode === 'auto' ? selectedGpu.index : selectedGpu.index,
        router_path: router?.path,
        text2comp_path: text2comp?.path,
        fno_path: executorMode === 'fno' ? fno?.path : undefined,
        expert_executor: executorMode,
        uploaded_expert_id: executorMode === 'uploaded' ? uploaded?.model_id : undefined,
        auto_sync: true,
      })
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
          main_llm_path: llm?.path,
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

  const modelCards = [
    {
      key: 'llm',
      title: 'LLM',
      icon: Brain,
      model: llm,
      loaded: Boolean(status?.loaded_models?.llm?.loaded),
      tone: 'violet',
    },
    {
      key: 'router',
      title: 'Router',
      icon: Route,
      model: router,
      loaded: Boolean(status?.loaded_models?.router?.loaded),
      tone: 'amber',
    },
    {
      key: 'text2comp',
      title: 'Text2Comp',
      icon: Layers3,
      model: text2comp,
      loaded: Boolean(status?.loaded_models?.text2comp?.loaded),
      tone: 'sky',
    },
    {
      key: 'expert',
      title: executorMode === 'uploaded' ? 'Uploaded Expert' : 'FNO Expert',
      icon: Activity,
      model: expert,
      loaded: Boolean(status?.loaded_models?.fno?.loaded || status?.loaded_models?.uploaded_expert?.loaded),
      tone: 'emerald',
    },
  ]

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <div className="training-eyebrow">简洁训练 / 模型拼装</div>
              <h1 className="training-simple-hero__title">装载 PierNet 推理链路</h1>
              <p className="training-copy">自动选择可用模型，完成 LLM、Router、Text2Comp 和 Expert 的粗粒度拼装。</p>
              <div className="mt-2 flex flex-wrap gap-2 text-[12px] text-slate-400">
                <span className="training-chip">LLM {status?.llms.length ?? 0}</span>
                <span className="training-chip">Router {status?.routers.length ?? 0}</span>
                <span className="training-chip">Text2Comp {status?.text2comps.length ?? 0}</span>
                <span className="training-chip">
                  Expert {(status?.fno_experts.length ?? 0) + (status?.custom_experts.length ?? 0)}
                </span>
              </div>
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
                一键加载
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
                <SectionTitle title="自动拼装组件" copy="使用当前可用的推荐模型" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-assembly-grid">
                  {modelCards.map(item => {
                    const Icon = item.icon
                    return (
                      <div
                        key={item.key}
                        className={`training-simple-assembly-card training-simple-assembly-card--${item.tone}`}
                      >
                        <div className="training-simple-assembly-card__head">
                          <span>
                            <Icon size={16} />
                          </span>
                          <strong>{item.title}</strong>
                          {item.loaded && <CheckCircle2 size={15} />}
                        </div>
                        <div className="training-simple-assembly-card__name">{item.model?.name ?? '未找到模型'}</div>
                        <div className="training-simple-assembly-card__note">{modelNote(item.model)}</div>
                        <div className="training-simple-assembly-card__status">{statusText(item.loaded)}</div>
                      </div>
                    )
                  })}
                </div>
                <div className="training-simple-option-grid">
                  <button
                    type="button"
                    className={`training-simple-option ${executorMode === 'fno' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setExecutorMode('fno')}
                    disabled={!fno}
                  >
                    <Activity size={16} />
                    <span>
                      <strong>FNO Expert</strong>
                      <small>{fno ? fno.name : '暂无可用模型'}</small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`training-simple-option ${executorMode === 'uploaded' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setExecutorMode('uploaded')}
                    disabled={uploadedExperts.length === 0}
                  >
                    <Activity size={16} />
                    <span>
                      <strong>Uploaded Expert</strong>
                      <small>{uploaded ? uploaded.name : '暂无匹配模型'}</small>
                    </span>
                  </button>
                </div>
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Cpu size={16} className="text-emerald-300" />
                <SectionTitle title="装载资源与测试" copy="选择资源后加载链路并运行一次测试" />
                <button type="button" className="btn-ghost ml-auto" onClick={() => refresh()}>
                  <RefreshCw size={14} />
                  刷新
                </button>
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-option-grid">
                  <button
                    type="button"
                    className={`training-simple-option ${resourceMode === 'auto' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setResourceMode('auto')}
                  >
                    <CheckCircle2 size={16} />
                    <span>
                      <strong>自动分配</strong>
                      <small>
                        {bestGpu ? `GPU ${bestGpu.index} · ${bestGpu.available ? '可用' : '排队'}` : '等待 GPU 信息'}
                      </small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`training-simple-option ${resourceMode === 'manual' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setResourceMode('manual')}
                  >
                    <Cpu size={16} />
                    <span>
                      <strong>指定资源</strong>
                      <small>{selectedGpu ? `GPU ${selectedGpu.index}` : '等待 GPU 信息'}</small>
                    </span>
                  </button>
                </div>
                {resourceMode === 'manual' && (
                  <div className="training-simple-gpu-list training-scroll">
                    {(status?.gpus ?? []).map(gpu => {
                      const total = gpu.memory_total_mb || 0
                      const used = gpu.memory_used_mb || 0
                      const ratio = total > 0 ? (used / total) * 100 : 0
                      return (
                        <button
                          key={gpu.index}
                          type="button"
                          className={`training-simple-gpu ${selectedGpu?.index === gpu.index ? 'training-simple-gpu--active' : ''}`}
                          onClick={() => setGpuId(gpu.index)}
                        >
                          <span>
                            <strong>GPU {gpu.index}</strong>
                            <small>{gpu.available ? '可用' : '占用中，可排队'}</small>
                          </span>
                          <span className="training-simple-gpu__meter">
                            <span>{gpuUsageLabel(used, total)}</span>
                            <UsageBar value={ratio} />
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
                <textarea
                  className="input min-h-[6rem] resize-y"
                  value={testInput}
                  onChange={event => setTestInput(event.target.value)}
                  placeholder="输入一句任务描述，用于验证已装载链路"
                />
                <div className="training-simple-job__actions">
                  {isLoaded ? (
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={runTest}
                      disabled={!testInput.trim() || busy}
                    >
                      {busy ? <Loader2 size={14} className="animate-spin" /> : <PlayCircle size={14} />}
                      运行测试
                    </button>
                  ) : (
                    <button type="button" className="btn-primary" onClick={load} disabled={!canLoad || busy}>
                      {busy ? <Loader2 size={14} className="animate-spin" /> : <Power size={14} />}
                      一键加载
                    </button>
                  )}
                  {isLoaded && (
                    <button type="button" className="btn-ghost" onClick={unload} disabled={busy}>
                      <PowerOff size={14} />
                      卸载
                    </button>
                  )}
                </div>
              </div>
            </section>
          </div>

          {!canLoad && (
            <div className="training-simple-job__notice">
              <AlertTriangle size={14} />
              <span>当前缺少一键拼装所需模型，请先在对应模块训练或上传可用模型。</span>
            </div>
          )}

          {testResult && (
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <PlayCircle size={16} className="text-emerald-300" />
                <SectionTitle title="测试结果" copy="只展示粗粒度推理结果" />
              </div>
              <div className="training-card__body">
                <div className="training-simple-result-grid">
                  <div>
                    <span>Router 判断</span>
                    <strong>{testResult.router_class_name || '--'}</strong>
                  </div>
                  <div>
                    <span>延迟</span>
                    <strong>{testResult.latency_ms != null ? `${testResult.latency_ms.toFixed(2)} ms` : '--'}</strong>
                  </div>
                </div>
                <div className="mt-3 rounded-lg border border-emerald-500/20 bg-emerald-500/8 p-3 text-sm leading-6 text-slate-200">
                  {testResult.final_answer || '没有返回最终答案。'}
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
