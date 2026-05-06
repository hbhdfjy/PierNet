import { AlertTriangle, Gauge, PauseCircle, PlayCircle, RefreshCcw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import type { TrainingJobSummary } from '../../lib/types'
import {
  formatCount,
  formatDateTime,
  formatDuration,
  formatMetric,
  statusBadgeClass,
  statusLabel,
} from '../shared'

function KpiCard({ label, value, note, icon }: { label: string; value: string; note: string; icon: React.ReactNode }) {
  return (
    <div className="training-kpi">
      <div className="flex items-start justify-between gap-3">
        <span className="training-kpi__label">{label}</span>
        <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
          {icon}
        </span>
      </div>
      <div className="training-kpi__value">{value}</div>
      <div className="training-kpi__note">{note}</div>
    </div>
  )
}

function actionErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return String(error || '未知错误')
}

function JobRow({ job, expanded = false }: { job: TrainingJobSummary; expanded?: boolean }) {
  const { mutate } = useSWRConfig()
  const [isStopping, setIsStopping] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const stoppable = ['starting', 'running', 'evaluating'].includes(job.status)
  const stopping = job.status === 'stopping'
  const deletable = !stoppable && !stopping
  const scenarioText = job.scenarios.join(', ')

  const refreshAll = async () => {
    await Promise.all([mutate('training-jobs'), mutate('training-overview'), mutate('training-gpus')])
  }

  const stopJob = async () => {
    if (isStopping || isDeleting) return
    setActionError(null)
    setIsStopping(true)
    try {
      await api.stopTrainingJob(job.job_id)
      await refreshAll()
    } catch (error) {
      setActionError(`终止失败：${actionErrorMessage(error)}`)
    } finally {
      setIsStopping(false)
    }
  }

  const deleteJob = async () => {
    if (isDeleting) return
    const ok = window.confirm(`删除历史任务 ${job.name} (${job.job_id})？\n\n会彻底删除任务记录、运行目录、权重、曲线和日志。共享预处理缓存会保留。`)
    if (!ok) return
    setActionError(null)
    setIsDeleting(true)
    try {
      await api.deleteTrainingJob(job.job_id)
      await refreshAll()
    } catch (error) {
      setActionError(`删除失败：${actionErrorMessage(error)}`)
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className={expanded ? 'training-card p-3.5' : 'training-card p-3'}>
      <div className="flex h-full flex-col justify-between gap-3">
        <div className="grid gap-3 xl:grid-cols-[minmax(13rem,0.95fr)_minmax(12rem,0.9fr)_minmax(14rem,1.1fr)_auto] xl:items-center">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="pretty-tooltip min-w-0" data-tooltip={job.name}>
                <div className="truncate text-[15px] font-semibold text-slate-100">{job.name}</div>
              </div>
              <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
            </div>
            <div className="pretty-tooltip mt-1 min-w-0" data-tooltip={job.job_id}>
              <div className="mono truncate text-[11px] text-slate-500">{job.job_id}</div>
            </div>
          </div>

          <div className="min-w-0">
            <div className="training-label">训练数据</div>
            <div className="mt-1 text-[13px] font-medium text-slate-200">
              {job.simulator.toUpperCase()} · GPU {job.gpu_id} · {job.scenarios.length} 个场景
            </div>
          </div>

          <div className="pretty-tooltip min-w-0" data-tooltip={scenarioText}>
            <div className="training-label">子场景</div>
            <div className="training-mono-note mt-1">{scenarioText || '—'}</div>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            {(stoppable || stopping) && (
              <button type="button" className="btn-danger" onClick={stopJob} disabled={isStopping || stopping}>
                <PauseCircle size={14} />
                {isStopping || stopping ? '终止中...' : '终止'}
              </button>
            )}
            {deletable && (
              <button type="button" className="btn-ghost" onClick={deleteJob} disabled={isDeleting}>
                <Trash2 size={14} />
                {isDeleting ? '删除中...' : '删除'}
              </button>
            )}
            <Link to={`/training/jobs/${job.job_id}`} className={isDeleting ? 'btn-primary pointer-events-none opacity-60' : 'btn-primary'}>
              <PlayCircle size={14} />
              查看详情
            </Link>
          </div>
        </div>

        <div className="training-meta-grid">
          <div className="training-surface--dense">
            <div className="training-label">创建时间</div>
            <div className="mt-1 truncate text-[13px] text-slate-100">{formatDateTime(job.created_at)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">轮次 / 步数</div>
            <div className="mt-1 truncate text-[13px] text-slate-100">{job.latest_epoch ?? '—'} / {job.latest_step ?? '—'}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">训练损失</div>
            <div className="mt-1 truncate text-[13px] text-slate-100">{formatMetric(job.avg_loss, 6)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">最近 F1</div>
            <div className="mt-1 truncate text-[13px] text-slate-100">{formatMetric(job.latest_metrics?.f1, 4)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">预计剩余</div>
            <div className="mt-1 truncate text-[13px] text-slate-100">{formatDuration(job.eta_seconds)}</div>
          </div>
        </div>

        {actionError && (
          <div className="flex items-start gap-2 rounded-xl border border-amber-400/25 bg-amber-400/8 px-3 py-2 text-sm text-amber-200">
            <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
            <span>{actionError}</span>
          </div>
        )}

        {job.error_message && (
          <div className="flex items-start gap-2 rounded-xl border border-rose-500/20 bg-rose-500/8 px-3 py-2 text-sm text-rose-300">
            <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
            <span>{job.error_message}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default function TrainingJobsPage() {
  const { data, error, isLoading, mutate } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const runningCount = data?.filter(job => ['starting', 'running', 'evaluating', 'stopping'].includes(job.status)).length ?? 0
  const doneCount = data?.filter(job => job.status === 'done').length ?? 0
  const errorCount = data?.filter(job => job.status === 'error' || job.status === 'external_terminated').length ?? 0

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="training-eyebrow">任务管理</div>
                <h1 className="mt-2 text-[1.55rem] font-semibold tracking-tight text-white xl:text-[1.75rem]">训练任务</h1>
              </div>
              <div className="flex flex-wrap items-center gap-2.5">
                <button type="button" className="btn-ghost" onClick={() => mutate()}>
                  <RefreshCcw size={14} />
                  刷新
                </button>
                <Link to="/training/new" className="btn-primary">
                  <PlayCircle size={14} />
                  新建训练
                </Link>
              </div>
            </div>

            <div className="mt-4 training-kpi-grid">
              <KpiCard label="总任务数" value={formatCount(data?.length ?? 0)} note="注册表中的全部任务" icon={<Gauge size={16} />} />
              <KpiCard label="运行中" value={formatCount(runningCount)} note="启动中 / 训练中 / 评测中 / 停止中" icon={<PlayCircle size={16} />} />
              <KpiCard label="已完成" value={formatCount(doneCount)} note="包含 权重 和测试结果" icon={<RefreshCcw size={16} />} />
              <KpiCard label="失败" value={formatCount(errorCount)} note="可进入详情页查看错误" icon={<AlertTriangle size={16} />} />
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              无法加载训练任务：{error.message}
            </div>
          )}

          <section className="training-card">
            <div className="card-header">
              <Gauge size={16} className="text-sky-300" />
              <div>
                <div className="training-panel-title">任务列表</div>
                <div className="training-panel-copy">运行中任务可终止，历史任务可删除</div>
              </div>
            </div>
            <div className="training-card__body training-scroll training-job-list-scroll">
              {isLoading && !data ? (
                <div className="space-y-2.5">
                  {[0, 1, 2].map(item => <div key={item} className="skeleton h-40 rounded-xl" />)}
                </div>
              ) : data?.length ? (
                <div className="space-y-2.5">
                  {data.map(job => <JobRow key={job.job_id} job={job} expanded={data.length <= 2} />)}
                </div>
              ) : (
                <div className="training-card p-6 text-center text-sm text-slate-400">
                  当前没有训练任务。可以直接去“新建训练”启动第一个任务。
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
