import { AlertTriangle, CheckCircle2, Clock3, Loader2, PlayCircle, RefreshCw } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { TrainingJobDetail, TrainingJobStatus, TrainingLogResponse } from '../../lib/types'
import { TrainingUsageBar, type TrainingProgressTone } from '../components/common'
import { statusLabel, trainingJobDetailPath, trainingJobRefreshInterval } from '../shared'

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
      copy: '模型权重和训练结果已经保存。',
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
      copy: '正在保存当前 checkpoint。',
      percent: 92,
      tone: 'amber',
      active: true,
      icon: 'running',
    }
  }

  if (job.status === 'evaluating') {
    return {
      label: '正在评估 Router',
      copy: '正在根据测试集计算当前训练效果。',
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
        copy: '正在加载嵌入模型并初始化训练环境。',
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

export default function TrainingSimpleProgressPage() {
  const { jobId = '' } = useParams()
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

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-progress-page">
        <div className="training-simple-progress-wrap">
          <section className="training-simple-progress-card">
            <div className={`training-simple-progress__icon training-simple-progress__icon--${stage.tone}`}>
              {stageIcon(stage)}
            </div>
            <div className="training-eyebrow">一键训练进度</div>
            <h1 className="training-simple-progress__title">{stage.label}</h1>
            <p className="training-simple-progress__copy">{stage.copy}</p>
            <div className="training-simple-progress__bar">
              <TrainingUsageBar value={stage.percent} tone={stage.tone} className="" />
              <div className="training-simple-progress__percent">{Math.round(stage.percent)}%</div>
            </div>
            {jobError && <div className="training-simple-progress__error">加载训练进度失败：{jobError.message}</div>}
            <div className="training-simple-progress__meta">
              <span>{job?.name ?? jobId}</span>
              <span>{job ? `${job.simulator.toUpperCase()} · ${job.scenarios.length} 个场景` : '读取中'}</span>
            </div>
            <div className="training-simple-progress__actions">
              <Link to="/training/simple" className="btn-ghost">
                <RefreshCw size={14} />
                返回一键训练
              </Link>
              {job && !stage.active && (
                <Link to={trainingJobDetailPath(job.job_id)} className="btn-primary">
                  <PlayCircle size={14} />
                  查看结果详情
                </Link>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
