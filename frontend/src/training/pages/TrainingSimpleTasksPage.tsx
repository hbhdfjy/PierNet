import { useMemo, useState } from 'react'
import { AlertTriangle, Clock3, PauseCircle, PlayCircle, RefreshCw, Trash2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import type { TrainingJobSummary } from '../../lib/types'
import { ConfirmDialog, StatusBadge } from '../../shared/ui'
import { TrainingSectionTitle as SectionTitle } from '../components/common'
import {
  formatDateTime,
  formatDuration,
  formatMetric,
  isTrainingJobActive,
  isTrainingJobDeletable,
  isTrainingJobStoppable,
  statusBadgeClass,
  statusLabel,
  trainingJobNotice,
  trainingSimpleJobProgressPath,
} from '../shared'

function actionErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

export default function TrainingSimpleTasksPage() {
  const { mutate } = useSWRConfig()
  const [busyJobId, setBusyJobId] = useState<string | null>(null)
  const [deleteJob, setDeleteJob] = useState<TrainingJobSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { data: jobs, isLoading } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const simpleJobs = useMemo(() => (jobs ?? []).filter(job => job.config?.simple_pipeline_enabled === true), [jobs])
  const orderedJobs = useMemo(
    () =>
      [...simpleJobs].sort((left, right) => {
        const activeDelta = Number(isTrainingJobActive(right.status)) - Number(isTrainingJobActive(left.status))
        if (activeDelta !== 0) return activeDelta
        return (right.created_at ?? 0) - (left.created_at ?? 0)
      }),
    [simpleJobs],
  )
  const runningCount = orderedJobs.filter(job => isTrainingJobActive(job.status)).length
  const completedCount = orderedJobs.filter(job => job.status === 'done').length

  const refreshAll = async () => {
    await Promise.all([mutate('training-jobs'), mutate('training-overview'), mutate('training-gpus')])
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

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <h1 className="training-simple-hero__title">训练任务</h1>
              <p className="training-simple-hero__meta">
                <span>运行中 {runningCount}</span>
                <span>已完成 {completedCount}</span>
                <span>总任务 {orderedJobs.length}</span>
              </p>
            </div>
            <Link to="/training/simple" className="btn-primary training-simple-hero__action">
              <PlayCircle size={15} />
              新建训练
            </Link>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              <AlertTriangle size={15} className="mr-2 inline" />
              {error}
            </div>
          )}

          <section className="training-card training-card--compact training-simple-panel">
            <div className="card-header">
              <Clock3 size={16} className="text-sky-300" />
              <SectionTitle title="任务列表" />
              <button
                type="button"
                className="training-icon-button ml-auto"
                onClick={() => refreshAll()}
                aria-label="刷新任务"
                title="刷新任务"
              >
                <RefreshCw size={14} />
              </button>
            </div>
            <div className="training-card__body">
              {isLoading && !jobs ? (
                <div className="grid gap-3">
                  {[0, 1, 2].map(item => (
                    <div key={item} className="skeleton h-28 rounded-xl" />
                  ))}
                </div>
              ) : orderedJobs.length ? (
                <div className="training-simple-job-grid">
                  {orderedJobs.map(job => {
                    const active = isTrainingJobActive(job.status)
                    const notice = trainingJobNotice(job)
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
                        <div className="training-simple-job__facts">
                          <span>创建 {formatDateTime(job.created_at)}</span>
                          <span>F1 {formatMetric(job.latest_metrics?.f1, 4)}</span>
                          <span>剩余 {formatDuration(job.eta_seconds)}</span>
                        </div>
                        {notice && (
                          <div
                            className={`training-simple-job__notice ${
                              notice.tone === 'rose' ? 'training-simple-job__notice--rose' : ''
                            }`}
                          >
                            <AlertTriangle size={14} />
                            <span>{notice.message}</span>
                          </div>
                        )}
                        <div className="training-simple-job__actions">
                          <Link to={trainingSimpleJobProgressPath(job.job_id)} className="btn-primary">
                            <PlayCircle size={14} />
                            {active ? '训练详情' : '查看详情'}
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
