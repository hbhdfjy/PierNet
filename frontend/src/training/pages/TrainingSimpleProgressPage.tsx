import { AlertTriangle, CheckCircle2, Clock3, Loader2, PauseCircle, PlayCircle, RefreshCw, Trash2 } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { useState } from 'react'
import { api } from '../../lib/api'
import type { TrainingJobDetail, TrainingJobStatus, TrainingLogResponse } from '../../lib/types'
import { ConfirmDialog } from '../../shared/ui'
import { TrainingUsageBar, type TrainingProgressTone } from '../components/common'
import {
  formatMetric,
  isTrainingJobDeletable,
  isTrainingJobStoppable,
  statusLabel,
  trainingJobRefreshInterval,
} from '../shared'

type SimpleStage = {
  label: string
  copy: string
  percent: number
  tone: TrainingProgressTone
  active: boolean
  icon: 'waiting' | 'running' | 'done' | 'warning'
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value))
}

function logsContain(lines: string[], fragment: string): boolean {
  return lines.some(line => line.includes(fragment))
}

function finiteTrainingPercent(job: TrainingJobDetail): number | null {
  const epochs = Number(job.config.epochs || 0)
  if (!Number.isFinite(epochs) || epochs <= 0) return null
  const currentEpoch = Number(job.latest_epoch || 0)
  const currentStep = Number(job.latest_step || 0)
  const stepsPerEpoch = Number(job.steps_per_epoch || 0)
  const epochProgress = stepsPerEpoch > 0 ? currentStep / stepsPerEpoch : 0
  return clampPercent(52 + ((currentEpoch + epochProgress) / epochs) * 36)
}

function stageIcon(stage: SimpleStage) {
  if (stage.icon === 'done') return <CheckCircle2 size={22} />
  if (stage.icon === 'warning') return <AlertTriangle size={22} />
  if (stage.icon === 'waiting') return <Clock3 size={22} />
  return stage.active ? <Loader2 size={22} className="training-simple-progress__spin" /> : <PlayCircle size={22} />
}

function stageFromJob(job: TrainingJobDetail | undefined, lines: string[]): SimpleStage {
  if (!job) {
    return {
      label: '正在读取训练任务',
      copy: '正在获取任务状态。',
      percent: 4,
      tone: 'sky',
      active: true,
      icon: 'waiting',
    }
  }

  if (job.status === 'done') {
    return {
      label: 'Router 训练完成',
      copy: '训练结果已经保存。',
      percent: 100,
      tone: 'emerald',
      active: false,
      icon: 'done',
    }
  }

  if (job.status === 'error' || job.status === 'external_terminated') {
    return {
      label: '训练失败',
      copy: job.error_message || '训练过程没有正常完成。',
      percent: 100,
      tone: 'amber',
      active: false,
      icon: 'warning',
    }
  }

  if (job.status === 'terminated') {
    return {
      label: '训练已停止',
      copy: '任务已经停止。',
      percent: 100,
      tone: 'amber',
      active: false,
      icon: 'warning',
    }
  }

  if (job.status === 'queued') {
    return {
      label: '等待训练开始',
      copy: '任务已经提交，正在等待训练资源。',
      percent: 8,
      tone: 'sky',
      active: true,
      icon: 'waiting',
    }
  }

  if (job.status === 'stopping') {
    return {
      label: '正在安全停止',
      copy: '正在保存当前进度。',
      percent: 92,
      tone: 'amber',
      active: true,
      icon: 'running',
    }
  }

  if (job.status === 'evaluating') {
    return {
      label: '正在评估 Router',
      copy: '正在计算当前训练效果。',
      percent: 88,
      tone: 'violet',
      active: true,
      icon: 'running',
    }
  }

  if (job.status === 'starting') {
    if (
      logsContain(lines, 'phase=encoder') ||
      logsContain(lines, 'phase=dataloader') ||
      logsContain(lines, 'phase=model')
    ) {
      return {
        label: '正在准备 Router 训练',
        copy: '正在初始化训练环境。',
        percent: 42,
        tone: 'sky',
        active: true,
        icon: 'running',
      }
    }
    if (logsContain(lines, 'phase=prepare')) {
      return {
        label: '正在处理 Router 训练数据',
        copy: '正在整理场景数据和训练样本。',
        percent: 26,
        tone: 'sky',
        active: true,
        icon: 'running',
      }
    }
    return {
      label: '正在启动 Router 训练',
      copy: '训练任务正在初始化。',
      percent: 18,
      tone: 'sky',
      active: true,
      icon: 'running',
    }
  }

  if (job.status === 'running') {
    return {
      label: '正在训练 Router',
      copy: '训练正在进行中。',
      percent: finiteTrainingPercent(job) ?? 72,
      tone: 'emerald',
      active: true,
      icon: 'running',
    }
  }

  return {
    label: statusLabel(job.status as TrainingJobStatus),
    copy: '训练任务正在更新状态。',
    percent: 50,
    tone: 'sky',
    active: true,
    icon: 'running',
  }
}

function actionErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

export default function TrainingSimpleProgressPage() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()
  const { mutate } = useSWRConfig()
  const [busy, setBusy] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const { data: job, error: jobError } = useSWR<TrainingJobDetail>(
    jobId ? `training-job-${jobId}` : null,
    () => api.getTrainingJob(jobId),
    {
      refreshInterval: current => trainingJobRefreshInterval(current?.status),
      revalidateOnFocus: false,
    },
  )
  const refreshInterval = trainingJobRefreshInterval(job?.status)
  const { data: logs } = useSWR<TrainingLogResponse>(
    jobId ? `training-logs-${jobId}` : null,
    () => api.getTrainingLogs(jobId, 120),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const stage = stageFromJob(job, logs?.lines ?? [])
  const canStop = job ? isTrainingJobStoppable(job.status) : false
  const canDelete = job ? isTrainingJobDeletable(job.status) : false

  const refreshAll = async () => {
    await Promise.all([mutate('training-jobs'), mutate('training-overview'), mutate('training-gpus')])
  }

  const stopJob = async () => {
    if (!job) return
    setBusy(true)
    setActionError(null)
    try {
      await api.stopTrainingJob(job.job_id)
      await refreshAll()
    } catch (error) {
      setActionError(`终止失败：${actionErrorMessage(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const deleteJob = async () => {
    if (!job) return
    setBusy(true)
    setActionError(null)
    try {
      await api.deleteTrainingJob(job.job_id)
      await refreshAll()
      navigate('/training/simple')
    } catch (error) {
      setActionError(`删除失败：${actionErrorMessage(error)}`)
      setBusy(false)
    }
  }

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-progress-page">
        <div className="training-simple-progress-wrap">
          <section className="training-simple-progress-card">
            <div className={`training-simple-progress__icon training-simple-progress__icon--${stage.tone}`}>
              {stageIcon(stage)}
            </div>
            <div className="training-eyebrow">简洁训练进度</div>
            <h1 className="training-simple-progress__title">{stage.label}</h1>
            <p className="training-simple-progress__copy">{stage.copy}</p>
            <div className="training-simple-progress__bar">
              <TrainingUsageBar value={stage.percent} tone={stage.tone} className="" />
              <div className="training-simple-progress__percent">{Math.round(stage.percent)}%</div>
            </div>
            {(job?.latest_metrics || job?.avg_loss != null) && (
              <div className="training-simple-result-grid">
                <div>
                  <span>最近 F1</span>
                  <strong>{formatMetric(job.latest_metrics?.f1, 4)}</strong>
                </div>
                <div>
                  <span>训练损失</span>
                  <strong>{formatMetric(job.avg_loss, 6)}</strong>
                </div>
              </div>
            )}
            {(jobError || actionError) && (
              <div className="training-simple-progress__error">
                {actionError ?? `加载训练进度失败：${jobError?.message}`}
              </div>
            )}
            <div className="training-simple-progress__meta">
              <span>{job?.name ?? jobId}</span>
              <span>{job ? `${job.simulator.toUpperCase()} · ${job.scenarios.length} 个场景` : '读取中'}</span>
            </div>
            <div className="training-simple-progress__actions">
              <Link to="/training/simple" className="btn-ghost">
                <RefreshCw size={14} />
                返回简洁训练
              </Link>
              {canStop && (
                <button
                  type="button"
                  className="btn-danger"
                  onClick={stopJob}
                  disabled={busy || job?.status === 'stopping'}
                >
                  <PauseCircle size={14} />
                  {busy ? '处理中...' : job?.status === 'queued' ? '取消排队' : '终止'}
                </button>
              )}
              {canDelete && (
                <button type="button" className="btn-ghost" onClick={() => setDeleteOpen(true)} disabled={busy}>
                  <Trash2 size={14} />
                  删除任务
                </button>
              )}
            </div>
          </section>
        </div>
      </div>

      <ConfirmDialog
        open={deleteOpen}
        title="删除训练任务"
        description={
          <>
            将删除 <span className="font-semibold text-slate-100">{job?.name}</span> 的任务记录和训练产物。
          </>
        }
        confirmLabel="删除"
        danger
        busy={busy}
        onCancel={() => setDeleteOpen(false)}
        onConfirm={deleteJob}
      />
    </div>
  )
}
