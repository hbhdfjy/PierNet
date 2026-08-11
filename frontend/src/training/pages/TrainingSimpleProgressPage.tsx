import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Layers3,
  Loader2,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { useState } from 'react'
import { api } from '../../lib/api'
import type { TrainingJobDetail, TrainingJobStatus, TrainingLogResponse } from '../../lib/types'
import { ConfirmDialog } from '../../shared/ui'
import { TrainingUsageBar, type TrainingProgressTone } from '../components/common'
import {
  formatDuration,
  isTrainingJobDeletable,
  isTrainingJobStoppable,
  statusLabel,
  trainingJobDetailPath,
  trainingJobRefreshInterval,
} from '../shared'

type ProgressStage = {
  label: string
  step: string
  detail: string
  percent: number
  tone: TrainingProgressTone
  active: boolean
  icon: 'waiting' | 'running' | 'done' | 'warning'
  stageIndex: number
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
  return clampPercent(42 + ((currentEpoch + epochProgress) / epochs) * 18)
}

function estimateEta(job: TrainingJobDetail | undefined, stage: ProgressStage): string {
  if (!job) return '正在读取任务'
  if (job.eta_seconds != null) return formatDuration(job.eta_seconds)
  if (!stage.active) return '—'
  if (job.status === 'queued') return '等待资源'
  const epochs = Number(job.config.epochs || 0)
  const stepsPerEpoch = Number(job.steps_per_epoch || 0)
  const stepsPerSec = Number(job.steps_per_sec || 0)
  if (epochs > 0 && stepsPerEpoch > 0 && stepsPerSec > 0) {
    const done = Number(job.latest_epoch || 0) * stepsPerEpoch + Number(job.latest_step || 0)
    const total = epochs * stepsPerEpoch
    return formatDuration(Math.max(0, total - done) / stepsPerSec)
  }
  return '估算中'
}

function stageIcon(stage: ProgressStage) {
  if (stage.icon === 'done') return <CheckCircle2 size={22} />
  if (stage.icon === 'warning') return <AlertTriangle size={22} />
  if (stage.icon === 'waiting') return <Clock3 size={22} />
  return stage.active ? <Loader2 size={22} className="training-simple-progress__spin" /> : <PlayCircle size={22} />
}

function stageFromJob(job: TrainingJobDetail | undefined, lines: string[]): ProgressStage {
  if (!job) {
    return {
      label: '正在读取训练任务',
      step: '读取任务记录',
      detail: '正在获取任务状态、训练配置和最近日志。',
      percent: 4,
      tone: 'sky',
      active: true,
      icon: 'waiting',
      stageIndex: 0,
    }
  }

  const pipelineStage = job.pipeline_stage
  const text2compActive = pipelineStage === 'text2comp'

  if (job.status === 'done') {
    return {
      label: '分阶段训练完成',
      step: 'Router 和 Text2Comp 已保存',
      detail: job.text2comp_model_path
        ? 'Router checkpoint 和 Text2Comp 模型已经写入任务目录，可以进入模型拼装。'
        : 'Router checkpoint 已写入任务目录。',
      percent: 100,
      tone: 'emerald',
      active: false,
      icon: 'done',
      stageIndex: 7,
    }
  }

  if (job.status === 'error' || job.status === 'external_terminated') {
    return {
      label: '训练失败',
      step: '等待处理异常',
      detail:
        job.text2comp_error_message || job.error_message || '训练过程没有正常完成，请在复杂训练任务中查看完整日志。',
      percent: 100,
      tone: 'amber',
      active: false,
      icon: 'warning',
      stageIndex: 7,
    }
  }

  if (job.status === 'terminated') {
    return {
      label: '训练已停止',
      step: '停止训练进程',
      detail: '任务已经停止，已保留可用的任务记录和中间产物。',
      percent: 100,
      tone: 'amber',
      active: false,
      icon: 'warning',
      stageIndex: 7,
    }
  }

  if (job.status === 'queued') {
    return {
      label: '等待训练开始',
      step: '分配 worker 和 GPU',
      detail: '任务已提交，正在等待可用训练资源。',
      percent: 10,
      tone: 'sky',
      active: true,
      icon: 'waiting',
      stageIndex: 1,
    }
  }

  if (job.status === 'stopping') {
    return {
      label: '正在安全停止',
      step: text2compActive ? '停止 Text2Comp 训练' : '停止 Router 训练',
      detail: '已发送停止请求，正在等待 checkpoint 或任务状态安全落盘。',
      percent: 96,
      tone: 'amber',
      active: true,
      icon: 'running',
      stageIndex: 6,
    }
  }

  if (job.status === 'evaluating') {
    return {
      label: text2compActive ? '正在评估 Text2Comp' : '正在评估 Router',
      step: '计算验证指标',
      detail: text2compActive
        ? 'Router 已完成，正在评估文生计算模型并保存权重。'
        : '正在评估 Router 训练结果，更新指标并保存最新权重。',
      percent: text2compActive ? 86 : 58,
      tone: 'violet',
      active: true,
      icon: 'running',
      stageIndex: text2compActive ? 5 : 3,
    }
  }

  if (job.status === 'starting') {
    if (text2compActive) {
      return {
        label: '正在准备 Text2Comp',
        step: '装载 Qwen 与回归头',
        detail: 'Router 已完成，正在读取训练数据中的目标输出并初始化文生计算模型。',
        percent: 68,
        tone: 'sky',
        active: true,
        icon: 'running',
        stageIndex: 4,
      }
    }
    if (logsContain(lines, 'phase=prepare')) {
      return {
        label: '正在准备 Router 数据',
        step: '读取场景并生成训练缓存',
        detail: '正在按所选场景整理 Router 数据、过滤样本并写入 prepared cache。',
        percent: 20,
        tone: 'sky',
        active: true,
        icon: 'running',
        stageIndex: 1,
      }
    }
    if (logsContain(lines, 'phase=dataloader')) {
      return {
        label: '正在构建 Router 输入',
        step: '创建 dataloader',
        detail: '正在组织 batch、切分训练/验证集，并准备数据加载器。',
        percent: 28,
        tone: 'sky',
        active: true,
        icon: 'running',
        stageIndex: 1,
      }
    }
    if (logsContain(lines, 'phase=encoder')) {
      return {
        label: '正在构建 Router 输入',
        step: '生成文本向量表示',
        detail: '正在调用嵌入模型处理上下文，将文本样本转换为训练特征。',
        percent: 34,
        tone: 'sky',
        active: true,
        icon: 'running',
        stageIndex: 1,
      }
    }
    if (logsContain(lines, 'phase=model')) {
      return {
        label: '正在初始化 Router',
        step: '装载训练头和优化器',
        detail: '正在创建模型参数、优化器和训练状态，马上进入迭代训练。',
        percent: 40,
        tone: 'sky',
        active: true,
        icon: 'running',
        stageIndex: 2,
      }
    }
    return {
      label: '正在启动训练',
      step: '检查任务配置',
      detail: '正在写入任务目录、确认场景范围，并准备先训练 Router。',
      percent: 10,
      tone: 'sky',
      active: true,
      icon: 'running',
      stageIndex: 0,
    }
  }

  if (job.status === 'running') {
    if (text2compActive) {
      const epoch = Number(job.latest_epoch ?? 0)
      const step = Number(job.latest_step ?? 0)
      const totalEpochs = Number(job.config.simple_text2comp_epochs || 0)
      return {
        label: '正在训练 Text2Comp',
        step: totalEpochs > 0 ? `第 ${epoch + 1}/${totalEpochs} 轮 · step ${step}` : `训练 step ${step}`,
        detail: 'Text2Comp 正在学习训练数据中的目标输出。',
        percent: job.text2comp_model_path ? 92 : 78,
        tone: 'emerald',
        active: true,
        icon: 'running',
        stageIndex: 5,
      }
    }
    const epoch = Number(job.latest_epoch ?? 0)
    const step = Number(job.latest_step ?? 0)
    const totalEpochs = Number(job.config.epochs || 0)
    return {
      label: '正在训练 Router',
      step: totalEpochs > 0 ? `第 ${epoch + 1}/${totalEpochs} 轮 · step ${step}` : `训练 step ${step}`,
      detail: '正在基于所选场景更新 Router 参数，并持续记录损失、指标和剩余时间。',
      percent: finiteTrainingPercent(job) ?? 50,
      tone: 'emerald',
      active: true,
      icon: 'running',
      stageIndex: 2,
    }
  }

  return {
    label: statusLabel(job.status as TrainingJobStatus),
    step: '同步任务状态',
    detail: '训练任务正在更新状态。',
    percent: 50,
    tone: 'sky',
    active: true,
    icon: 'running',
    stageIndex: 0,
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

  const isSimpleJob = !job || job.config?.simple_pipeline_enabled === true
  const stage = stageFromJob(job, logs?.lines ?? [])
  const eta = estimateEta(job, stage)
  const canStop = job && isSimpleJob ? isTrainingJobStoppable(job.status) : false
  const canDelete = job && isSimpleJob ? isTrainingJobDeletable(job.status) : false

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
      navigate('/training/simple/tasks')
    } catch (error) {
      setActionError(`删除失败：${actionErrorMessage(error)}`)
      setBusy(false)
    }
  }

  if (job && !isSimpleJob) {
    return (
      <div className="training-page">
        <div className="training-page__body training-simple-progress-page">
          <div className="training-simple-progress-wrap">
            <section className="training-simple-progress-card">
              <div className="training-simple-progress__icon training-simple-progress__icon--amber">
                <AlertTriangle size={22} />
              </div>
              <div className="training-eyebrow">训练详情</div>
              <h1 className="training-simple-progress__title">这不是简洁训练任务</h1>
              <p className="training-simple-progress__copy">
                当前任务来自完整训练平台，请进入普通训练详情页查看日志、指标和操作。
              </p>
              <div className="training-simple-progress__actions">
                <Link to="/training/simple/tasks" className="btn-ghost">
                  <RefreshCw size={14} />
                  返回任务
                </Link>
                <Link to={trainingJobDetailPath(job.job_id)} className="btn-primary">
                  <PlayCircle size={14} />
                  打开普通训练详情
                </Link>
              </div>
            </section>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-progress-page">
        <div className="training-simple-progress-wrap">
          <section className="training-simple-progress-card">
            <div className={`training-simple-progress__icon training-simple-progress__icon--${stage.tone}`}>
              {stageIcon(stage)}
            </div>
            <h1 className="training-simple-progress__title">{stage.label}</h1>
            <p className="training-simple-progress__copy">{stage.step}</p>

            <div className="training-simple-progress__bar" aria-label="模型训练进度">
              <TrainingUsageBar value={stage.percent} tone={stage.tone} className="" />
              <div className="training-simple-progress__percent">{Math.round(stage.percent)}%</div>
            </div>

            <div className="training-simple-progress__summary">
              <span>{job ? `${job.name} · ${job.scenarios.length} 个场景` : '正在读取任务'}</span>
              <span>{stage.active ? `剩余 ${eta}` : statusLabel(job?.status as TrainingJobStatus)}</span>
            </div>

            {(jobError || actionError) && (
              <div className="training-simple-progress__error">
                {actionError ?? `加载训练进度失败：${jobError?.message}`}
              </div>
            )}
            <div className="training-simple-progress__actions">
              <Link to="/training/simple/tasks" className="btn-ghost">
                <RefreshCw size={14} />
                返回任务
              </Link>
              {job?.status === 'done' && (
                <Link to="/training/simple/assembly" className="btn-primary">
                  <Layers3 size={14} />
                  模型拼装
                </Link>
              )}
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
