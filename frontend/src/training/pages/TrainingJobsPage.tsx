import { AlertTriangle, Gauge, PauseCircle, PlayCircle, RefreshCcw } from 'lucide-react'
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

function JobRow({ job }: { job: TrainingJobSummary }) {
  const { mutate } = useSWRConfig()
  const stoppable = ['starting', 'running', 'evaluating'].includes(job.status)

  const stopJob = async () => {
    await api.stopTrainingJob(job.job_id)
    await Promise.all([mutate('training-jobs'), mutate('training-overview'), mutate('training-gpus')])
  }

  return (
    <div className="training-card p-3.5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-[16px] font-semibold text-slate-100">{job.name}</div>
          <div className="mono mt-1 text-[12px] text-slate-500">{job.job_id}</div>
          <div className="mt-1 text-[15px] text-slate-400">
            {job.simulator.toUpperCase()} · GPU {job.gpu_id} · {job.scenarios.length} 个子场景
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
          {stoppable && (
            <button type="button" className="btn-danger" onClick={stopJob}>
              <PauseCircle size={14} />
              终止
            </button>
          )}
          <Link to={`/training/jobs/${job.job_id}`} className="btn-ghost">
            <PlayCircle size={14} />
            查看详情
          </Link>
        </div>
      </div>

      <div className="mt-2.5 grid gap-2.5 sm:grid-cols-2 xl:grid-cols-5">
        <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-2.5">
          <div className="training-label">创建时间</div>
          <div className="mt-1 text-[16px] text-slate-100">{formatDateTime(job.created_at)}</div>
        </div>
        <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-2.5">
          <div className="training-label">Epoch / Step</div>
          <div className="mt-1 text-[16px] text-slate-100">
            {job.latest_epoch ?? '—'} / {job.latest_step ?? '—'}
          </div>
        </div>
        <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-2.5">
          <div className="training-label">训练损失</div>
          <div className="mt-1 text-[16px] text-slate-100">{formatMetric(job.avg_loss, 6)}</div>
        </div>
        <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-2.5">
          <div className="training-label">最近测试 F1</div>
          <div className="mt-1 text-[16px] text-slate-100">{formatMetric(job.latest_metrics?.f1, 4)}</div>
        </div>
        <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-2.5">
          <div className="training-label">预计剩余</div>
          <div className="mt-1 text-[16px] text-slate-100">{formatDuration(job.eta_seconds)}</div>
        </div>
      </div>

      {job.error_message && (
        <div className="mt-4 flex items-start gap-2 rounded-2xl border border-rose-500/20 bg-rose-500/8 px-4 py-3 text-sm text-rose-300">
          <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
          <span>{job.error_message}</span>
        </div>
      )}
    </div>
  )
}

export default function TrainingJobsPage() {
  const { data, error, isLoading, mutate } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const runningCount = data?.filter(job => ['starting', 'running', 'evaluating'].includes(job.status)).length ?? 0

  return (
    <div className="training-page">
      <div className="page-header">
        <div>
          <div className="training-label uppercase tracking-[0.22em]">Training Jobs</div>
          <h1 className="mt-2 training-title">训练任务管理</h1>
          <p className="mt-2 max-w-3xl training-copy">
            这里汇总所有受管训练任务，可以终止运行中的任务，并进入详情页查看训练曲线、测试曲线和日志。
          </p>
        </div>
        <div className="flex items-center gap-3">
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

      <div className="training-page__body">
        <div className="training-page__grid">
          <div className="grid gap-3 md:grid-cols-3">
          <div className="training-card p-3.5">
            <div className="training-label uppercase tracking-[0.18em]">总任务数</div>
            <div className="mt-1.5 text-[30px] font-semibold text-slate-100">{formatCount(data?.length ?? 0)}</div>
          </div>
          <div className="training-card p-3.5">
            <div className="training-label uppercase tracking-[0.18em]">运行中</div>
            <div className="mt-1.5 flex items-center gap-2 text-[30px] font-semibold text-slate-100">
              <Gauge size={18} className="text-sky-300" />
              {formatCount(runningCount)}
            </div>
          </div>
          <div className="training-card p-3.5">
            <div className="training-label uppercase tracking-[0.18em]">已完成</div>
            <div className="mt-1.5 text-[30px] font-semibold text-slate-100">
              {formatCount(data?.filter(job => job.status === 'done').length ?? 0)}
            </div>
            </div>
          </div>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              无法加载训练任务：{error.message}
            </div>
          )}

          <div className="training-scroll">
            {isLoading && !data ? (
              [0, 1, 2].map(item => <div key={item} className="skeleton h-48 rounded-3xl" />)
            ) : data?.length ? (
              data.map(job => <JobRow key={job.job_id} job={job} />)
            ) : (
              <div className="training-card p-8 text-center text-sm text-slate-400">
                当前没有训练任务。可以直接去“新建训练”页面启动第一个任务。
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
