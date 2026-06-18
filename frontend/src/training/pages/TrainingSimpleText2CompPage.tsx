import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Brain, Check, Clock3, Cpu, PauseCircle, PlayCircle, RefreshCw, Trash2 } from 'lucide-react'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import type { Text2CompDatasetInfo, Text2CompGPUInfo, Text2CompJobSummary, Text2CompOverview } from '../../lib/types'
import { ConfirmDialog, StatusBadge } from '../../shared/ui'
import { TrainingSectionTitle as SectionTitle, TrainingUsageBar as UsageBar } from '../components/common'
import { formatBytes, formatCount, formatDateTime, formatDuration, formatMetric, gpuUsageLabel } from '../shared'

type ResourceMode = 'auto' | 'manual'

type Text2CompStatus = Text2CompJobSummary['status']

function statusLabel(status: Text2CompStatus): string {
  switch (status) {
    case 'queued':
      return '排队中'
    case 'starting':
      return '启动中'
    case 'running':
      return '训练中'
    case 'evaluating':
      return '评估中'
    case 'done':
      return '已完成'
    case 'terminated':
      return '已终止'
    case 'error':
      return '失败'
    default:
      return status
  }
}

function statusClass(status: Text2CompStatus): string {
  if (status === 'done') return 'badge bg-emerald-500/8 text-emerald-300 border border-emerald-500/20'
  if (status === 'running' || status === 'starting' || status === 'evaluating') {
    return 'badge bg-sky-500/15 text-sky-300 border border-sky-500/20'
  }
  if (status === 'error') return 'badge bg-rose-500/8 text-rose-300 border border-rose-500/20'
  if (status === 'terminated') return 'badge bg-amber-500/12 text-amber-300 border border-amber-500/20'
  return 'badge bg-slate-800/70 text-slate-300 border border-slate-700/40'
}

function isActive(status: Text2CompStatus): boolean {
  return status === 'queued' || status === 'starting' || status === 'running' || status === 'evaluating'
}

function gpuFreeMemory(gpu: Text2CompGPUInfo): number {
  return Math.max(0, gpu.memory_total_mib - gpu.memory_used_mib)
}

function pickBestGpu(gpus: Text2CompGPUInfo[] | undefined): Text2CompGPUInfo | null {
  if (!gpus?.length) return null
  return [...gpus].sort((a, b) => {
    const availability = Number(b.available) - Number(a.available)
    if (availability !== 0) return availability
    return gpuFreeMemory(b) - gpuFreeMemory(a)
  })[0]
}

function datasetLabel(dataset: Text2CompDatasetInfo | null): string {
  if (!dataset) return '未选择数据集'
  return `${dataset.scenario} · ${formatCount(dataset.n_samples ?? 0)} 样本`
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

export default function TrainingSimpleText2CompPage() {
  const { mutate } = useSWRConfig()
  const { data: status, error } = useSWR<Text2CompOverview>('text2comp-overview', api.getText2CompOverview, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const [expertName, setExpertName] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [resourceMode, setResourceMode] = useState<ResourceMode>('auto')
  const [gpuId, setGpuId] = useState<number | null>(null)
  const [jobName, setJobName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [busyJobId, setBusyJobId] = useState<string | null>(null)
  const [deleteJob, setDeleteJob] = useState<Text2CompJobSummary | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const selectedExpert = useMemo(
    () => status?.expert_models.find(item => item.name === expertName) ?? status?.expert_models[0] ?? null,
    [expertName, status?.expert_models],
  )
  const matchingDatasets = useMemo(
    () => (status?.datasets ?? []).filter(item => !selectedExpert || item.simulator === selectedExpert.name),
    [selectedExpert, status?.datasets],
  )
  const selectedDataset = useMemo(
    () => matchingDatasets.find(item => item.path === datasetPath) ?? matchingDatasets[0] ?? null,
    [datasetPath, matchingDatasets],
  )
  const bestGpu = useMemo(() => pickBestGpu(status?.gpus), [status?.gpus])
  const selectedGpu = useMemo(
    () => (status?.gpus ?? []).find(gpu => gpu.index === gpuId) ?? bestGpu,
    [bestGpu, gpuId, status?.gpus],
  )
  const canSubmit = Boolean(selectedExpert && selectedDataset && selectedGpu && !submitting)

  useEffect(() => {
    if (!selectedExpert) return
    setExpertName(selectedExpert.name)
  }, [selectedExpert])

  useEffect(() => {
    if (!selectedDataset) return
    setDatasetPath(selectedDataset.path)
  }, [selectedDataset])

  useEffect(() => {
    if (gpuId != null && (status?.gpus ?? []).some(gpu => gpu.index === gpuId)) return
    setGpuId(bestGpu?.index ?? null)
  }, [bestGpu?.index, gpuId, status?.gpus])

  const refresh = async () => {
    await Promise.all([mutate('text2comp-overview'), mutate('text2comp-jobs')])
  }

  const submit = async () => {
    if (!selectedExpert || !selectedDataset || !selectedGpu) {
      setActionError('请选择专家域、训练数据和训练资源。')
      return
    }
    setSubmitting(true)
    setActionError(null)
    try {
      const result = await api.startText2CompTrain({
        name: jobName.trim() || null,
        simulator: selectedExpert.name,
        scenario: selectedDataset.scenario || selectedExpert.name,
        train_path: selectedDataset.path,
        output_dim: selectedExpert.output_dim,
        epochs: 100,
        batch_size: 8,
        learning_rate: 0.00001,
        weight_decay: 0.01,
        gpu_id: resourceMode === 'auto' ? selectedGpu.index : selectedGpu.index,
      })
      if (!result.ok) throw new Error(result.error || '启动文生计算训练失败')
      setJobName('')
      await refresh()
    } catch (err) {
      setActionError(errorMessage(err))
    } finally {
      setSubmitting(false)
    }
  }

  const stopJob = async (job: Text2CompJobSummary) => {
    setBusyJobId(job.job_id)
    setActionError(null)
    try {
      await api.stopText2CompJob(job.job_id)
      await refresh()
    } catch (err) {
      setActionError(`停止失败：${errorMessage(err)}`)
    } finally {
      setBusyJobId(null)
    }
  }

  const confirmDelete = async () => {
    if (!deleteJob) return
    setBusyJobId(deleteJob.job_id)
    setActionError(null)
    try {
      await api.deleteText2CompJob(deleteJob.job_id)
      setDeleteJob(null)
      await refresh()
    } catch (err) {
      setActionError(`删除失败：${errorMessage(err)}`)
    } finally {
      setBusyJobId(null)
    }
  }

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <div className="training-eyebrow">简洁训练 / 文生计算</div>
              <h1 className="training-simple-hero__title">训练 Text2Comp 模块</h1>
              <p className="training-copy">选择专家域、训练数据和资源，平台使用默认策略完成文生计算训练。</p>
              <div className="mt-2 flex flex-wrap gap-2 text-[12px] text-slate-400">
                <span className="training-chip">专家域 {formatCount(status?.expert_models.length ?? 0)}</span>
                <span className="training-chip">数据集 {formatCount(status?.datasets.length ?? 0)}</span>
                <span className="training-chip">活跃 {formatCount(status?.running_job_count ?? 0)}</span>
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

          {(error || actionError) && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {actionError ?? `无法加载文生计算状态：${error?.message}`}
            </div>
          )}

          <div className="training-simple-bottom-grid">
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Brain size={16} className="text-sky-300" />
                <SectionTitle title="专家域与数据" copy="选择需要训练的 Text2Comp 数据" />
              </div>
              <div className="training-card__body space-y-3">
                <input
                  className="input"
                  value={jobName}
                  onChange={event => setJobName(event.target.value)}
                  placeholder="任务名称，可留空"
                />
                <div className="training-simple-option-grid">
                  {(status?.expert_models ?? []).map(expert => (
                    <button
                      key={expert.name}
                      type="button"
                      className={`training-simple-option ${selectedExpert?.name === expert.name ? 'training-simple-option--active' : ''}`}
                      onClick={() => {
                        setExpertName(expert.name)
                        setDatasetPath('')
                      }}
                    >
                      <Brain size={16} />
                      <span>
                        <strong>{expert.name}</strong>
                        <small>
                          {expert.domain} · 输出 {expert.output_dim}
                        </small>
                      </span>
                    </button>
                  ))}
                </div>
                <div className="training-simple-dataset-grid">
                  {matchingDatasets.map(dataset => (
                    <button
                      key={dataset.path}
                      type="button"
                      className={`training-simple-dataset ${selectedDataset?.path === dataset.path ? 'training-simple-dataset--active' : ''}`}
                      onClick={() => setDatasetPath(dataset.path)}
                    >
                      <div className="training-simple-dataset__top">
                        <div className="training-simple-dataset__name">{dataset.scenario}</div>
                        {selectedDataset?.path === dataset.path && <Check size={15} />}
                      </div>
                      <div className="training-simple-dataset__meta">
                        {formatCount(dataset.n_samples ?? 0)} 样本 · {formatBytes(dataset.file_size_bytes ?? 0)}
                      </div>
                    </button>
                  ))}
                  {matchingDatasets.length === 0 && (
                    <div className="training-surface text-sm text-slate-400">当前专家域没有可用训练数据。</div>
                  )}
                </div>
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Cpu size={16} className="text-emerald-300" />
                <SectionTitle title="训练资源" copy="自动选择资源，也可以指定 GPU" />
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
                      const memoryRatio =
                        gpu.memory_total_mib > 0 ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0
                      return (
                        <button
                          key={gpu.index}
                          type="button"
                          className={`training-simple-gpu ${selectedGpu?.index === gpu.index ? 'training-simple-gpu--active' : ''}`}
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
                <div className="training-simple-footer">
                  <div className="flex min-w-0 items-center gap-2 text-[13px] text-slate-400">
                    <Brain size={15} className="text-sky-300" />
                    <span className="truncate">
                      {selectedExpert?.name ?? '未选择专家域'} · {datasetLabel(selectedDataset)}
                    </span>
                  </div>
                  <button type="button" className="btn-primary" onClick={submit} disabled={!canSubmit}>
                    <PlayCircle size={15} />
                    {submitting ? '启动中...' : '开始训练'}
                  </button>
                </div>
              </div>
            </section>
          </div>

          <section className="training-card training-card--compact training-simple-panel">
            <div className="card-header">
              <Clock3 size={16} className="text-violet-300" />
              <SectionTitle title="最近文生计算任务" copy="只展示状态、损失和基础操作" />
              <button type="button" className="btn-ghost ml-auto" onClick={() => refresh()}>
                <RefreshCw size={14} />
                刷新
              </button>
            </div>
            <div className="training-card__body">
              {status?.jobs?.length ? (
                <div className="training-simple-job-grid">
                  {status.jobs.slice(0, 8).map(job => (
                    <div key={job.job_id} className="training-simple-job">
                      <div className="training-simple-job__head">
                        <div className="min-w-0">
                          <div className="training-simple-job__title">{job.name}</div>
                          <div className="training-simple-job__meta">
                            {job.simulator} · GPU {job.gpu_id} · {formatDateTime(job.created_at)}
                          </div>
                        </div>
                        <StatusBadge className={statusClass(job.status)}>{statusLabel(job.status)}</StatusBadge>
                      </div>
                      <div className="training-meta-grid">
                        <div className="training-surface--dense">
                          <div className="training-label">最近损失</div>
                          <div className="mt-1 truncate text-[13px] text-slate-100">
                            {formatMetric(job.avg_loss, 6)}
                          </div>
                        </div>
                        <div className="training-surface--dense">
                          <div className="training-label">进度</div>
                          <div className="mt-1 truncate text-[13px] text-slate-100">
                            {job.latest_epoch != null ? `第 ${job.latest_epoch} 轮` : '--'}
                          </div>
                        </div>
                        <div className="training-surface--dense">
                          <div className="training-label">预计剩余</div>
                          <div className="mt-1 truncate text-[13px] text-slate-100">
                            {formatDuration(job.eta_seconds)}
                          </div>
                        </div>
                      </div>
                      {job.error_message && (
                        <div className="training-simple-job__notice">
                          <AlertTriangle size={14} />
                          <span>{job.error_message}</span>
                        </div>
                      )}
                      <div className="training-simple-job__actions">
                        {isActive(job.status) && (
                          <button
                            type="button"
                            className="btn-danger"
                            onClick={() => stopJob(job)}
                            disabled={busyJobId === job.job_id}
                          >
                            <PauseCircle size={14} />
                            {busyJobId === job.job_id ? '处理中...' : '终止'}
                          </button>
                        )}
                        {!isActive(job.status) && (
                          <button
                            type="button"
                            className="btn-ghost"
                            onClick={() => setDeleteJob(job)}
                            disabled={busyJobId === job.job_id}
                          >
                            <Trash2 size={14} />
                            删除
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="training-surface text-sm text-slate-400">当前没有文生计算训练任务。</div>
              )}
            </div>
          </section>
        </div>
      </div>

      <ConfirmDialog
        open={Boolean(deleteJob)}
        title="删除文生计算任务"
        description={
          <>
            将删除 <span className="font-semibold text-slate-100">{deleteJob?.name}</span> 的任务记录和训练产物。
          </>
        }
        confirmLabel="删除"
        danger
        busy={busyJobId === deleteJob?.job_id}
        onCancel={() => setDeleteJob(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
