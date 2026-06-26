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
import { TrainingSectionTitle as SectionTitle, TrainingUsageBar as UsageBar } from '../components/common'
import { gpuUsageLabel } from '../shared'

type AssemblyStatus = Awaited<ReturnType<typeof api.getAssemblyStatus>>
type ExecutorMode = 'fno' | 'uploaded'
type ResourceMode = 'auto' | 'manual'
type AssemblyMode = 'profile' | 'training_job'

type AssemblyProfile = NonNullable<AssemblyStatus['assembly_profiles']>[number]

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

function isUsableTrainingJob(job: TrainingJobSummary): boolean {
  return (
    Boolean(job.run_dir) &&
    (job.status === 'done' || job.status === 'terminated' || job.status === 'external_terminated') &&
    (job.latest_epoch ?? 0) > 0
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

function pathName(path?: string | null): string {
  if (!path) return '未配置'
  return path.split('/').filter(Boolean).pop() || path
}

function profileToCards(profile: AssemblyProfile, loaded: boolean) {
  return [
    {
      key: 'llm',
      title: 'LLM',
      icon: Brain,
      model: { name: pathName(profile.llm_path), description: 'Qwen base / embedding' },
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
      title: 'DNN Expert',
      icon: Activity,
      model: {
        name: pathName(profile.expert_path),
        description: `输出 ${(profile.output_shape ?? []).join(' x ') || '--'}`,
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
  const [executorMode, setExecutorMode] = useState<ExecutorMode>('fno')
  const [resourceMode, setResourceMode] = useState<ResourceMode>('auto')
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
  const selectedTrainingJob = useMemo(
    () => usableTrainingJobs.find(item => item.job_id === trainingJobId) ?? usableTrainingJobs[0] ?? null,
    [trainingJobId, usableTrainingJobs],
  )
  const llm = useMemo(() => pickLLM(status), [status])
  const trainingJobRouter = useMemo(() => routerModelFromTrainingJob(selectedTrainingJob), [selectedTrainingJob])
  const registryRouter = useMemo(() => pickFirst(status?.routers, item => item.trained), [status?.routers])
  const router = assemblyMode === 'training_job' ? trainingJobRouter : registryRouter
  const text2comp = useMemo(() => pickFirst(status?.text2comps, item => item.trained), [status?.text2comps])
  const fnoModels = useMemo(() => (status?.fno_experts ?? []).filter(item => item.trained), [status?.fno_experts])
  const fno = useMemo(() => fnoModels.find(item => item.path === fnoPath) ?? fnoModels[0] ?? null, [fnoModels, fnoPath])
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
  const profileLoaded = Boolean(status?.loaded_models?.assembly_profile?.loaded)
  const isLoaded = Boolean(profileLoaded || status?.loaded_models?.llm?.loaded)
  const canLoadTrainingJob = Boolean(
    llm && selectedGpu && trainingJobRouter && text2comp && (executorMode === 'fno' ? fno : uploaded),
  )
  const canLoad = Boolean(selectedGpu && (assemblyMode === 'profile' ? assemblyProfile : canLoadTrainingJob))

  useEffect(() => {
    if (gpuId != null && (status?.gpus ?? []).some(gpu => gpu.index === gpuId)) return
    setGpuId(bestGpu?.index ?? null)
  }, [bestGpu?.index, gpuId, status?.gpus])

  useEffect(() => {
    if (profileId && assemblyProfiles.some(item => item.model_id === profileId)) return
    setProfileId(assemblyProfiles[0]?.model_id ?? '')
  }, [assemblyProfiles, profileId])

  useEffect(() => {
    if (trainingJobId && usableTrainingJobs.some(item => item.job_id === trainingJobId)) return
    setTrainingJobId(usableTrainingJobs[0]?.job_id ?? '')
  }, [trainingJobId, usableTrainingJobs])

  useEffect(() => {
    if (fnoPath && fnoModels.some(item => item.path === fnoPath)) return
    setFnoPath(fnoModels[0]?.path ?? '')
  }, [fnoModels, fnoPath])

  useEffect(() => {
    if (uploadedExpertId && uploadedExperts.some(item => item.model_id === uploadedExpertId)) return
    setUploadedExpertId(uploadedExperts[0]?.model_id ?? '')
  }, [uploadedExpertId, uploadedExperts])

  useEffect(() => {
    if (executorMode === 'uploaded' && uploadedExperts.length === 0 && fno) setExecutorMode('fno')
    if (executorMode === 'fno' && !fno && uploadedExperts.length > 0) setExecutorMode('uploaded')
  }, [executorMode, fno, uploadedExperts.length])

  useEffect(() => {
    if (!status) return
    if (!assemblyProfile && assemblyMode === 'profile') {
      setAssemblyMode('training_job')
      return
    }
    if (assemblyProfile && !selectedTrainingJob && assemblyMode === 'training_job') {
      setAssemblyMode('profile')
    }
  }, [assemblyMode, assemblyProfile, selectedTrainingJob, status])

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
      if (!selectedTrainingJob || !trainingJobRouter) {
        setActionError('请先选择一个已有 checkpoint 的简洁训练任务。')
        return
      }
      selectedRouter = trainingJobRouter
      if (!llm || !text2comp) {
        setActionError('缺少 LLM 或 Text2Comp，无法把训练任务拼装为完整链路。')
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
          auto_sync: true,
        })
      } else {
        await api.loadAssemblyModels({
          llm_path: llm?.path,
          llm_gpu_id: resourceMode === 'auto' ? selectedGpu.index : selectedGpu.index,
          router_path: selectedRouter?.path,
          text2comp_path: text2comp?.path,
          fno_path: executorMode === 'fno' ? fno?.path : undefined,
          expert_executor: executorMode,
          uploaded_expert_id: executorMode === 'uploaded' ? uploaded?.model_id : undefined,
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
          assembly_profile_id:
            status?.loaded_models?.assembly_profile?.model_id ||
            (assemblyMode === 'profile' ? assemblyProfile?.model_id : undefined),
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
  const modelCards =
    assemblyMode === 'profile' && assemblyProfile ? profileToCards(assemblyProfile, profileLoaded) : componentCards

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <div className="training-eyebrow">简洁训练 / 模型拼装</div>
              <h1 className="training-simple-hero__title">装载 PierNet 推理链路</h1>
              <p className="training-copy">
                选择已注册完整模型，或选择一个简洁训练任务作为 Router 来源，再切换 Expert。
              </p>
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
                <SectionTitle title="拼装来源" copy="选择完整模型或简洁训练任务" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-option-grid">
                  <button
                    type="button"
                    className={`training-simple-option ${assemblyMode === 'profile' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setAssemblyMode('profile')}
                    disabled={!assemblyProfile}
                  >
                    <Layers3 size={16} />
                    <span>
                      <strong>已注册拼装模型</strong>
                      <small>{assemblyProfile ? assemblyProfile.name : '暂无完整模型'}</small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`training-simple-option ${assemblyMode === 'training_job' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setAssemblyMode('training_job')}
                    disabled={!selectedTrainingJob}
                  >
                    <Cpu size={16} />
                    <span>
                      <strong>简洁训练任务</strong>
                      <small>{selectedTrainingJob ? selectedTrainingJob.name : '暂无可拼装任务'}</small>
                    </span>
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
                    <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/8 p-3 text-xs leading-5 text-slate-300">
                      {assemblyProfile.description}
                    </div>
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
                      value={selectedTrainingJob?.job_id ?? ''}
                      onChange={event => setTrainingJobId(event.target.value)}
                      disabled={usableTrainingJobs.length === 0}
                    >
                      {usableTrainingJobs.length === 0 ? (
                        <option value="">暂无可拼装训练任务</option>
                      ) : (
                        usableTrainingJobs.map(job => (
                          <option key={job.job_id} value={job.job_id}>
                            {job.name} · {job.simulator.toUpperCase()} · {job.scenarios.length} 个场景
                          </option>
                        ))
                      )}
                    </select>
                    <div className="rounded-lg border border-sky-500/25 bg-sky-500/8 p-3 text-xs leading-5 text-slate-300">
                      {selectedTrainingJob
                        ? `使用 ${pathName(routerPathFromTrainingJob(selectedTrainingJob))} 作为 Router；LLM 和 Text2Comp 由平台自动补齐，Expert 可在下方切换。`
                        : '完成一次简洁训练后，这里会显示可用于拼装的 Router 任务。'}
                    </div>
                  </div>
                )}
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
                {assemblyMode === 'training_job' && (
                  <>
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
                    <div className="training-simple-option-grid">
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
                            <option value="">暂无 FNO Expert</option>
                          ) : (
                            fnoModels.map(item => (
                              <option key={item.path} value={item.path}>
                                {item.name}
                              </option>
                            ))
                          )}
                        </select>
                      </div>
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
                    </div>
                  </>
                )}
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
              <span>当前缺少一键拼装所需模型，请先注册完整拼装模型，或训练/上传组件模型。</span>
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
