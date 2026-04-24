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
        <span className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
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
  const deletable = !stoppable

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
    const ok = window.confirm(`删除历史任务 ${job.name} (${job.job_id})？\n\n会彻底删除任务记录、run 目录、checkpoint、曲线和日志。共享 prepared cache 会保留。`)
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
    <div className={expanded ? 'training-card p-5 min-h-[260px]' : 'training-card p-4'}>
      <div className="flex h-full flex-col justify-between gap-5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <div className="text-[17px] font-semibold text-slate-100">{job.name}</div>
              <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
            </div>
            <div className="mono mt-1 text-[12px] text-slate-500">{job.job_id}</div>
            <div className="mt-2 training-note">{job.simulator.toUpperCase()} · GPU {job.gpu_id} · {job.scenarios.length} 个子场景</div>
            <div className="mt-1 training-mono-note">{job.scenarios.join(', ')}</div>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            {stoppable && (
              <button type="button" className="btn-danger" onClick={stopJob} disabled={isStopping}>
                <PauseCircle size={14} />
                {isStopping ? '终止中...' : '终止'}
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
            <div className="mt-1 text-[15px] text-slate-100">{formatDateTime(job.created_at)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">Epoch / Step</div>
            <div className="mt-1 text-[15px] text-slate-100">{job.latest_epoch ?? '—'} / {job.latest_step ?? '—'}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">训练损失</div>
            <div className="mt-1 text-[15px] text-slate-100">{formatMetric(job.avg_loss, 6)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">最近 F1</div>
            <div className="mt-1 text-[15px] text-slate-100">{formatMetric(job.latest_metrics?.f1, 4)}</div>
          </div>
          <div className="training-surface--dense">
            <div className="training-label">预计剩余</div>
            <div className="mt-1 text-[15px] text-slate-100">{formatDuration(job.eta_seconds)}</div>
          </div>
        </div>

        {actionError && (
          <div className="flex items-start gap-2 rounded-2xl border border-amber-400/25 bg-amber-400/8 px-4 py-3 text-sm text-amber-200">
            <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
            <span>{actionError}</span>
          </div>
        )}

        {job.error_message && (
          <div className="flex items-start gap-2 rounded-2xl border border-rose-500/20 bg-rose-500/8 px-4 py-3 text-sm text-rose-300">
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

  const runningCount = data?.filter(job => ['starting', 'running', 'evaluating'].includes(job.status)).length ?? 0
  const doneCount = data?.filter(job => job.status === 'done').length ?? 0
  const errorCount = data?.filter(job => job.status === 'error').length ?? 0

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-5 p-5">
          <section className="training-hero">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
              <div>
                <div className="training-eyebrow">任务管理</div>
                <h1 className="mt-3 text-[1.8rem] font-semibold tracking-tight text-white xl:text-[2.1rem]">训练任务</h1>
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

            <div className="mt-5 training-kpi-grid">
              <KpiCard label="总任务数" value={formatCount(data?.length ?? 0)} note="注册表中的全部任务" icon={<Gauge size={16} />} />
              <KpiCard label="运行中" value={formatCount(runningCount)} note="starting / running / evaluating" icon={<PlayCircle size={16} />} />
              <KpiCard label="已完成" value={formatCount(doneCount)} note="包含 checkpoint 和测试结果" icon={<RefreshCcw size={16} />} />
              <KpiCard label="失败" value={formatCount(errorCount)} note="可进入详情页查看错误" icon={<AlertTriangle size={16} />} />
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              无法加载训练任务：{error.message}
            </div>
          )}

          <section className="training-card min-h-[420px]">
            <div className="card-header">
              <Gauge size={16} className="text-sky-300" />
              <div>
                <div className="training-panel-title">任务列表</div>
                <div className="training-panel-copy">运行中任务可终止，历史任务可删除</div>
              </div>
            </div>
            <div className="training-card__body training-scroll list-scroll-xl">
              {isLoading && !data ? (
                <div className="space-y-3">
                  {[0, 1, 2].map(item => <div key={item} className="skeleton h-40 rounded-3xl" />)}
                </div>
              ) : data?.length ? (
                <div className="space-y-3">
                  {data.map(job => <JobRow key={job.job_id} job={job} expanded={data.length <= 2} />)}
                </div>
              ) : (
                <div className="training-card p-8 text-center text-sm text-slate-400">
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
