import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  Check,
  Clock3,
  Cpu,
  Database,
  Layers3,
  MousePointerClick,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Trash2,
} from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import { useSeed } from '../../lib/seedContext'
import type { TrainingDatasetInfo, TrainingGPUInfo, TrainingJobSummary } from '../../lib/types'
import { ConfirmDialog, StatusBadge } from '../../shared/ui'
import { TrainingSectionTitle as SectionTitle, TrainingUsageBar as UsageBar } from '../components/common'
import {
  formatBytes,
  formatCount,
  formatDateTime,
  formatDuration,
  formatMetric,
  gpuUsageLabel,
  isTrainingJobActive,
  isTrainingJobDeletable,
  isTrainingJobStoppable,
  normalizeTrainingSeed,
  statusBadgeClass,
  statusLabel,
  trainingSimpleJobProgressPath,
} from '../shared'

type ResourceMode = 'auto' | 'manual'
type ResumeMode = 'fresh' | 'resume'

function selectedStats(dataset: TrainingDatasetInfo | null | undefined, selectedScenarios: string[]) {
  if (!dataset) return { count: 0, size: 0 }
  const selected = new Set(selectedScenarios)
  return dataset.scenarios.reduce(
    (acc, scenario) => {
      if (selected.has(scenario.scenario)) {
        acc.count += scenario.router_count
        acc.size += scenario.file_size_bytes
      }
      return acc
    },
    { count: 0, size: 0 },
  )
}

function sameScenarioSet(left: string[], right: string[]): boolean {
  if (left.length === 0 || left.length !== right.length) return false
  const a = [...left].sort()
  const b = [...right].sort()
  return a.every((item, index) => item === b[index])
}

function actionErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

function simpleJobNotice(job: Pick<TrainingJobSummary, 'error_message' | 'exit_reason' | 'status' | 'stop_requested'>) {
  const summary = (job.error_message || '').trim()
  const shortSummary = summary.length > 160 ? `${summary.slice(0, 160)}...` : summary
  const platformStop =
    job.exit_reason === 'platform_stop' || job.exit_reason === 'platform_stop_requested' || Boolean(job.stop_requested)

  if (job.status === 'stopping' && platformStop) return { message: '已发送停止请求，正在保存当前进度。' }
  if (job.status === 'terminated')
    return {
      message: platformStop ? '任务已按平台请求停止。' : `任务已停止${shortSummary ? `：${shortSummary}` : '。'}`,
    }
  if (job.status === 'external_terminated')
    return { message: `训练进程已退出${shortSummary ? `：${shortSummary}` : '。'}` }
  if (job.status === 'error') return { message: `训练失败${shortSummary ? `：${shortSummary}` : '。'}` }
  if (shortSummary) return { message: shortSummary }
  return null
}

function gpuFreeMemory(gpu: TrainingGPUInfo): number {
  return Math.max(0, gpu.memory_total_mib - gpu.memory_used_mib)
}

function bestGpuLabel(gpus: TrainingGPUInfo[] | undefined): string {
  if (!gpus?.length) return '等待 GPU 信息'
  const selected = [...gpus].sort((a, b) => {
    const availability = Number(b.available) - Number(a.available)
    if (availability !== 0) return availability
    const memory = gpuFreeMemory(b) - gpuFreeMemory(a)
    if (memory !== 0) return memory
    return a.utilization_gpu - b.utilization_gpu
  })[0]
  return `GPU ${selected.index} · ${selected.available ? '可用' : '排队'} · ${gpuUsageLabel(selected.memory_used_mib, selected.memory_total_mib)}`
}

export default function TrainingSimpleJobPage() {
  const navigate = useNavigate()
  const { mutate } = useSWRConfig()
  const { seed } = useSeed()
  const selectedSimulatorRef = useRef<string | null>(null)
  const {
    data: datasets,
    error: datasetError,
    isLoading,
  } = useSWR('training-datasets', api.getTrainingDatasets, {
    refreshInterval: 10000,
    revalidateOnFocus: false,
  })
  const { data: gpus } = useSWR<TrainingGPUInfo[]>('training-gpus', api.getTrainingGPUs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const { data: jobs } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const [simulator, setSimulator] = useState('')
  const [jobName, setJobName] = useState('')
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([])
  const [resourceMode, setResourceMode] = useState<ResourceMode>('auto')
  const [gpuId, setGpuId] = useState<number | null>(null)
  const [resumeMode, setResumeMode] = useState<ResumeMode>('fresh')
  const [resumeFrom, setResumeFrom] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [busyJobId, setBusyJobId] = useState<string | null>(null)
  const [deleteJob, setDeleteJob] = useState<TrainingJobSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  const dataset = useMemo(
    () => datasets?.find(item => item.simulator === simulator) ?? datasets?.[0] ?? null,
    [datasets, simulator],
  )
  const stats = useMemo(() => selectedStats(dataset, selectedScenarios), [dataset, selectedScenarios])
  const checkpointCandidates = useMemo(
    () =>
      (jobs ?? [])
        .filter(job => job.status === 'done')
        .filter(job => job.simulator === (dataset?.simulator ?? simulator))
        .filter(job => sameScenarioSet(job.scenarios, selectedScenarios))
        .map(job => ({
          job,
          label: `${job.name} · ${formatDateTime(job.ended_at ?? job.created_at)}`,
          value: `${job.run_dir}/router_latest.pt`,
        })),
    [dataset?.simulator, jobs, selectedScenarios, simulator],
  )
  const visibleJobs = useMemo(() => (jobs ?? []).slice(0, 8), [jobs])
  const canSubmit = Boolean(
    dataset && selectedScenarios.length > 0 && !submitting && (resourceMode === 'auto' || gpuId !== null),
  )

  useEffect(() => {
    if (!datasets) return
    if (datasets.length === 0) {
      selectedSimulatorRef.current = null
      setSelectedScenarios([])
      return
    }
    if (!simulator || !datasets.some(item => item.simulator === simulator)) {
      setSimulator(datasets[0].simulator)
    }
  }, [datasets, simulator])

  useEffect(() => {
    if (!dataset) {
      selectedSimulatorRef.current = null
      setSelectedScenarios([])
      return
    }
    const previousSimulator = selectedSimulatorRef.current
    selectedSimulatorRef.current = dataset.simulator
    const nextScenarios = dataset.scenarios.map(item => item.scenario)
    setSelectedScenarios(prev => {
      const available = new Set(nextScenarios)
      const kept = prev.filter(item => available.has(item))
      return previousSimulator === dataset.simulator ? kept : nextScenarios
    })
  }, [dataset])

  useEffect(() => {
    if (gpuId != null && (gpus ?? []).some(gpu => gpu.index === gpuId)) return
    setGpuId(gpus?.[0]?.index ?? null)
  }, [gpuId, gpus])

  useEffect(() => {
    if (!checkpointCandidates.some(option => option.value === resumeFrom)) {
      setResumeFrom('')
      if (resumeMode === 'resume') setResumeMode('fresh')
    }
  }, [checkpointCandidates, resumeFrom, resumeMode])

  const refreshAll = async () => {
    await Promise.all([
      mutate('training-overview'),
      mutate('training-datasets'),
      mutate('training-gpus'),
      mutate('training-jobs'),
    ])
  }

  const selectDataset = (nextSimulator: string) => {
    setSimulator(nextSimulator)
    const nextDataset = datasets?.find(item => item.simulator === nextSimulator)
    setSelectedScenarios(nextDataset?.scenarios.map(item => item.scenario) ?? [])
    setResumeMode('fresh')
    setResumeFrom('')
  }

  const toggleScenario = (scenario: string) => {
    setSelectedScenarios(current =>
      current.includes(scenario) ? current.filter(item => item !== scenario) : [...current, scenario],
    )
  }

  const applyResumeJob = (job: TrainingJobSummary) => {
    setSimulator(job.simulator)
    setSelectedScenarios(job.scenarios)
    setResumeMode('resume')
    setResumeFrom(`${job.run_dir}/router_latest.pt`)
    setJobName(`${job.name}-继续`)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const stopJob = async (job: TrainingJobSummary) => {
    setBusyJobId(job.job_id)
    setError(null)
    try {
      await api.stopTrainingJob(job.job_id)
      await refreshAll()
    } catch (err) {
      setError(`终止失败：${actionErrorMessage(err)}`)
    } finally {
      setBusyJobId(null)
    }
  }

  const confirmDeleteJob = async () => {
    if (!deleteJob) return
    setBusyJobId(deleteJob.job_id)
    setError(null)
    try {
      await api.deleteTrainingJob(deleteJob.job_id)
      await refreshAll()
      setDeleteJob(null)
    } catch (err) {
      setError(`删除失败：${actionErrorMessage(err)}`)
    } finally {
      setBusyJobId(null)
    }
  }

  const submit = async () => {
    if (!dataset) {
      setError('当前没有可用训练数据。')
      return
    }
    if (selectedScenarios.length === 0) {
      setError('请至少选择一个训练场景。')
      return
    }
    if (resourceMode === 'manual' && gpuId === null) {
      setError('请选择训练资源，或切换为自动分配。')
      return
    }
    if (resumeMode === 'resume' && !resumeFrom) {
      setError('请选择要继续训练的历史结果。')
      return
    }

    setSubmitting(true)
    setError(null)
    try {
      const job = await api.createQuickTrainingJob({
        name: jobName.trim() || null,
        simulator: dataset.simulator,
        scenarios: selectedScenarios,
        gpu_id: resourceMode === 'manual' ? gpuId : null,
        resume_from: resumeMode === 'resume' ? resumeFrom : null,
        seed: normalizeTrainingSeed(seed),
      })
      await refreshAll()
      navigate(trainingSimpleJobProgressPath(job.job_id))
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动简洁训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <div className="training-eyebrow">简洁训练</div>
              <h1 className="training-simple-hero__title">选择范围后启动 Router 训练</h1>
              <p className="training-copy">选择训练范围、资源策略和历史结果，启动 Router 训练闭环。</p>
              <div className="mt-2 flex flex-wrap gap-2 text-[12px] text-slate-400">
                <span className="training-chip">
                  场景 {selectedScenarios.length}/{dataset?.scenarios.length ?? 0}
                </span>
                <span className="training-chip">样本 {formatCount(stats.count)}</span>
                <span className="training-chip">
                  资源 {resourceMode === 'auto' ? '自动分配' : gpuId != null ? `GPU ${gpuId}` : '未选择'}
                </span>
              </div>
            </div>
            <button
              type="button"
              className="btn-primary training-simple-hero__action"
              onClick={submit}
              disabled={!canSubmit}
            >
              <PlayCircle size={15} />
              {submitting ? '启动中...' : '开始训练'}
            </button>
          </section>

          {(error || datasetError) && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {error ?? `无法加载训练数据：${datasetError?.message}`}
            </div>
          )}

          <div className="training-simple-grid">
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <SectionTitle title="大场景" copy="选择本次训练的数据范围" />
              </div>
              <div className="training-card__body">
                {isLoading && !datasets ? (
                  <div className="grid gap-2">
                    {[0, 1, 2].map(item => (
                      <div key={item} className="skeleton h-20 rounded-xl" />
                    ))}
                  </div>
                ) : datasets?.length ? (
                  <div className="training-simple-dataset-grid">
                    {datasets.map(item => {
                      const active = item.simulator === dataset?.simulator
                      return (
                        <button
                          key={item.simulator}
                          type="button"
                          className={`training-simple-dataset ${active ? 'training-simple-dataset--active' : ''}`}
                          onClick={() => selectDataset(item.simulator)}
                        >
                          <div className="training-simple-dataset__top">
                            <div className="training-simple-dataset__name">{item.simulator}</div>
                            {active && <Check size={15} />}
                          </div>
                          <div className="training-simple-dataset__meta">
                            {item.scenarios.length} 个子场景 · {formatCount(item.total_count)} 条
                          </div>
                        </button>
                      )
                    })}
                  </div>
                ) : (
                  <div className="training-surface text-sm text-slate-400">当前没有可用训练数据。</div>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel training-simple-panel--scenarios">
              <div className="card-header">
                <Layers3 size={16} className="text-emerald-300" />
                <SectionTitle title="子场景" copy="勾选需要训练的场景" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-summary">
                  <div>
                    <div className="training-label">已选择</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">
                      {selectedScenarios.length}/{dataset?.scenarios.length ?? 0} 个场景
                    </div>
                  </div>
                  <div>
                    <div className="training-label">样本</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">{formatCount(stats.count)}</div>
                  </div>
                  <div>
                    <div className="training-label">数据量</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">{formatBytes(stats.size)}</div>
                  </div>
                </div>

                {dataset && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-[13px] text-slate-400">{dataset.simulator.toUpperCase()}</div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="btn-ghost"
                        onClick={() => setSelectedScenarios(dataset.scenarios.map(item => item.scenario))}
                      >
                        全选
                      </button>
                      <button type="button" className="btn-ghost" onClick={() => setSelectedScenarios([])}>
                        清空
                      </button>
                    </div>
                  </div>
                )}

                <div className="training-simple-scenario-grid training-scroll">
                  {dataset?.scenarios.map(scenario => {
                    const checked = selectedScenarios.includes(scenario.scenario)
                    return (
                      <button
                        key={scenario.scenario}
                        type="button"
                        onClick={() => toggleScenario(scenario.scenario)}
                        className={`training-simple-scenario ${checked ? 'training-simple-scenario--active' : ''}`}
                        title={`${scenario.scenario} · ${formatCount(scenario.router_count)} 条 · ${formatBytes(scenario.file_size_bytes)}`}
                      >
                        <span className="training-simple-scenario__check">{checked && <Check size={13} />}</span>
                        <span className="min-w-0 flex-1">
                          <span className="training-simple-scenario__name">{scenario.scenario}</span>
                          <span className="training-simple-scenario__meta">
                            {formatCount(scenario.router_count)} · {formatBytes(scenario.file_size_bytes)}
                          </span>
                        </span>
                      </button>
                    )
                  }) ?? <div className="training-surface text-sm text-slate-400">请选择大场景。</div>}
                </div>
              </div>
            </section>
          </div>

          <div className="training-simple-bottom-grid">
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <MousePointerClick size={16} className="text-emerald-300" />
                <SectionTitle title="训练方式" copy="只选择粗粒度策略" />
              </div>
              <div className="training-card__body space-y-3">
                <input
                  className="input"
                  value={jobName}
                  onChange={event => setJobName(event.target.value)}
                  placeholder="任务名称，可留空"
                />
                <div className="training-simple-option-grid">
                  <button
                    type="button"
                    className={`training-simple-option ${resumeMode === 'fresh' ? 'training-simple-option--active' : ''}`}
                    onClick={() => {
                      setResumeMode('fresh')
                      setResumeFrom('')
                    }}
                  >
                    <PlayCircle size={16} />
                    <span>
                      <strong>重新训练</strong>
                      <small>从当前数据范围启动新任务</small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`training-simple-option ${resumeMode === 'resume' ? 'training-simple-option--active' : ''}`}
                    onClick={() => {
                      setResumeMode('resume')
                      setResumeFrom(checkpointCandidates[0]?.value ?? '')
                    }}
                    disabled={checkpointCandidates.length === 0}
                  >
                    <RotateCcw size={16} />
                    <span>
                      <strong>继续训练</strong>
                      <small>{checkpointCandidates.length > 0 ? '使用匹配的历史结果' : '暂无匹配结果'}</small>
                    </span>
                  </button>
                </div>
                {resumeMode === 'resume' && checkpointCandidates.length > 0 && (
                  <select className="select" value={resumeFrom} onChange={event => setResumeFrom(event.target.value)}>
                    {checkpointCandidates.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Cpu size={16} className="text-sky-300" />
                <SectionTitle title="训练资源" copy="自动分配，也可指定一张 GPU" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-option-grid">
                  <button
                    type="button"
                    className={`training-simple-option ${resourceMode === 'auto' ? 'training-simple-option--active' : ''}`}
                    onClick={() => setResourceMode('auto')}
                  >
                    <Check size={16} />
                    <span>
                      <strong>自动分配</strong>
                      <small>{bestGpuLabel(gpus)}</small>
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
                      <small>{gpuId != null ? `GPU ${gpuId}` : '等待 GPU 信息'}</small>
                    </span>
                  </button>
                </div>
                {resourceMode === 'manual' && (
                  <div className="training-simple-gpu-list training-scroll">
                    {(gpus ?? []).map(gpu => {
                      const memoryRatio =
                        gpu.memory_total_mib > 0 ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0
                      return (
                        <button
                          key={gpu.index}
                          type="button"
                          className={`training-simple-gpu ${gpuId === gpu.index ? 'training-simple-gpu--active' : ''}`}
                          onClick={() => setGpuId(gpu.index)}
                        >
                          <span>
                            <strong>GPU {gpu.index}</strong>
                            <small>{gpu.available ? '可用' : (gpu.reason ?? '占用中，可排队')}</small>
                          </span>
                          <span className="training-simple-gpu__meter">
                            <span>{gpuUsageLabel(gpu.memory_used_mib, gpu.memory_total_mib)}</span>
                            <UsageBar value={memoryRatio} />
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </section>
          </div>

          <div className="training-simple-footer">
            <div className="flex min-w-0 items-center gap-2 text-[13px] text-slate-400">
              <MousePointerClick size={15} className="text-sky-300" />
              <span className="truncate">启动后进入简洁进度页，平台按默认训练策略执行。</span>
            </div>
            <button type="button" className="btn-primary" onClick={submit} disabled={!canSubmit}>
              <PlayCircle size={15} />
              {submitting ? '启动中...' : '开始训练'}
            </button>
          </div>

          <section className="training-card training-card--compact training-simple-panel">
            <div className="card-header">
              <Clock3 size={16} className="text-violet-300" />
              <SectionTitle title="最近训练" copy="在简洁界面完成进度查看、终止、删除和继续训练" />
              <button type="button" className="btn-ghost ml-auto" onClick={() => refreshAll()}>
                <RefreshCw size={14} />
                刷新
              </button>
            </div>
            <div className="training-card__body">
              {visibleJobs.length ? (
                <div className="training-simple-job-grid">
                  {visibleJobs.map(job => {
                    const active = isTrainingJobActive(job.status)
                    const notice = simpleJobNotice(job)
                    return (
                      <div key={job.job_id} className="training-simple-job">
                        <div className="training-simple-job__head">
                          <div className="min-w-0">
                            <div className="training-simple-job__title">{job.name}</div>
                            <div className="training-simple-job__meta">
                              {job.simulator.toUpperCase()} · GPU {job.gpu_id} · {job.scenarios.length} 个场景
                            </div>
                          </div>
                          <StatusBadge className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</StatusBadge>
                        </div>
                        <div className="training-meta-grid">
                          <div className="training-surface--dense">
                            <div className="training-label">创建时间</div>
                            <div className="mt-1 truncate text-[13px] text-slate-100">
                              {formatDateTime(job.created_at)}
                            </div>
                          </div>
                          <div className="training-surface--dense">
                            <div className="training-label">最近 F1</div>
                            <div className="mt-1 truncate text-[13px] text-slate-100">
                              {formatMetric(job.latest_metrics?.f1, 4)}
                            </div>
                          </div>
                          <div className="training-surface--dense">
                            <div className="training-label">预计剩余</div>
                            <div className="mt-1 truncate text-[13px] text-slate-100">
                              {formatDuration(job.eta_seconds)}
                            </div>
                          </div>
                        </div>
                        {notice && (
                          <div className="training-simple-job__notice">
                            <AlertTriangle size={14} />
                            <span>{notice.message}</span>
                          </div>
                        )}
                        <div className="training-simple-job__actions">
                          <Link to={trainingSimpleJobProgressPath(job.job_id)} className="btn-primary">
                            <PlayCircle size={14} />
                            {active ? '查看进度' : '查看结果'}
                          </Link>
                          {isTrainingJobStoppable(job.status) && (
                            <button
                              type="button"
                              className="btn-danger"
                              onClick={() => stopJob(job)}
                              disabled={busyJobId === job.job_id || job.status === 'stopping'}
                            >
                              <PauseCircle size={14} />
                              {busyJobId === job.job_id ? '处理中...' : job.status === 'queued' ? '取消排队' : '终止'}
                            </button>
                          )}
                          {job.status === 'done' && (
                            <button type="button" className="btn-ghost" onClick={() => applyResumeJob(job)}>
                              <RotateCcw size={14} />
                              继续训练
                            </button>
                          )}
                          {isTrainingJobDeletable(job.status) && (
                            <button type="button" className="btn-ghost" onClick={() => setDeleteJob(job)}>
                              <Trash2 size={14} />
                              删除
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="training-surface text-sm text-slate-400">当前没有训练任务。</div>
              )}
            </div>
          </section>
        </div>
      </div>

      <ConfirmDialog
        open={Boolean(deleteJob)}
        title="删除训练任务"
        description={
          <>
            将删除 <span className="font-semibold text-slate-100">{deleteJob?.name}</span> 的任务记录和训练产物。
          </>
        }
        confirmLabel="删除"
        danger
        busy={busyJobId === deleteJob?.job_id}
        onCancel={() => setDeleteJob(null)}
        onConfirm={confirmDeleteJob}
      />
    </div>
  )
}
