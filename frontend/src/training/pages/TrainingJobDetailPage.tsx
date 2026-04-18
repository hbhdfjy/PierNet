import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  FileText,
  PauseCircle,
  RadioTower,
  RefreshCcw,
  Save,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../../lib/api'
import type {
  TrainingCheckpointInfo,
  TrainingCurvesResponse,
  TrainingJobDetail,
  TrainingLogResponse,
  TrainingPoint,
} from '../../lib/types'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatMetric,
  shortPath,
  statusBadgeClass,
  statusLabel,
} from '../shared'

type TrainingAxisMode = 'step' | 'epoch'

function ChartCard({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string
  subtitle: string
  actions?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="training-card overflow-hidden">
      <div className="card-header justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 size={16} className="text-sky-300" />
          <div>
            <div className="text-[17px] font-semibold text-slate-100">{title}</div>
            <div className="text-[15px] text-slate-400">{subtitle}</div>
          </div>
        </div>
        {actions}
      </div>
      <div className="h-[304px] p-3.5">{children}</div>
    </div>
  )
}

function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center rounded-2xl border border-slate-700/40 bg-slate-900/30 text-[15px] text-slate-400">
      {message}
    </div>
  )
}

function CheckpointList({ checkpoints }: { checkpoints: TrainingCheckpointInfo[] }) {
  if (checkpoints.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-[15px] text-slate-400">
        当前还没有 checkpoint。
      </div>
    )
  }

  return (
    <div className="space-y-2.5">
      {checkpoints.map(item => (
        <div key={item.path} className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-3.5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="mono text-[15px] font-semibold text-slate-100">{item.name}</div>
              <div className="mt-1 text-[13px] text-slate-400" title={item.path}>
                {shortPath(item.path, 88)}
              </div>
            </div>
            <Save size={16} className="text-emerald-300" />
          </div>
          <div className="mt-2.5 grid grid-cols-3 gap-2.5 text-[13px] text-slate-400">
            <div>
              <div className="label">Epoch</div>
              <div className="mt-1 text-[15px] text-slate-200">{item.epoch ?? '—'}</div>
            </div>
            <div>
              <div className="label">大小</div>
              <div className="mt-1 text-[15px] text-slate-200">{formatBytes(item.size_bytes)}</div>
            </div>
            <div>
              <div className="label">时间</div>
              <div className="mt-1 text-[15px] text-slate-200">{formatDateTime(item.mtime)}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function buildEpochSeries(points: TrainingPoint[]): TrainingPoint[] {
  const lastPointByEpoch = new Map<number, TrainingPoint>()
  for (const point of points) {
    lastPointByEpoch.set(point.epoch, point)
  }
  return Array.from(lastPointByEpoch.values()).sort((a, b) => a.epoch - b.epoch)
}

export default function TrainingJobDetailPage() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()
  const { mutate } = useSWRConfig()
  const [trainingAxisMode, setTrainingAxisMode] = useState<TrainingAxisMode>('step')

  const { data: job, error: jobError } = useSWR<TrainingJobDetail>(
    jobId ? `training-job-${jobId}` : null,
    () => api.getTrainingJob(jobId),
    {
      refreshInterval: current => {
        if (!current) return 5000
        return ['starting', 'running', 'evaluating'].includes(current.status) ? 5000 : 0
      },
      revalidateOnFocus: false,
    },
  )

  const refreshInterval = useMemo(() => {
    if (!job) return 5000
    return ['starting', 'running', 'evaluating'].includes(job.status) ? 5000 : 0
  }, [job])

  const { data: curves } = useSWR<TrainingCurvesResponse>(
    jobId ? `training-curves-${jobId}` : null,
    () => api.getTrainingCurves(jobId, 2000),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const { data: logs } = useSWR<TrainingLogResponse>(
    jobId ? `training-logs-${jobId}` : null,
    () => api.getTrainingLogs(jobId, 400),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const trainingChart = useMemo(() => {
    const raw = curves?.training_points ?? []
    const epochSeries = curves?.training_epoch_points ?? buildEpochSeries(raw)
    return {
      xKey: trainingAxisMode === 'step' ? 'global_step' : 'epoch',
      data: trainingAxisMode === 'step' ? raw : epochSeries,
      subtitleSuffix: trainingAxisMode === 'step' ? 'step' : 'epoch',
    }
  }, [curves?.training_epoch_points, curves?.training_points, trainingAxisMode])

  const scenarioMetricData = useMemo(() => {
    const scenarioNames = new Set<string>()
    const rows = new Map<number, Record<string, number>>()

    for (const point of curves?.test_points ?? []) {
      const row = rows.get(point.epoch) ?? { epoch: point.epoch }
      for (const [scenario, metrics] of Object.entries(point.per_scenario)) {
        scenarioNames.add(scenario)
        row[scenario] = Number(metrics.f1 ?? 0)
      }
      rows.set(point.epoch, row)
    }

    return {
      scenarioNames: Array.from(scenarioNames),
      data: Array.from(rows.values()).sort((a, b) => Number(a.epoch) - Number(b.epoch)),
    }
  }, [curves?.test_points])

  const stopJob = async () => {
    await api.stopTrainingJob(jobId)
    await Promise.all([
      mutate(`training-job-${jobId}`),
      mutate(`training-curves-${jobId}`),
      mutate(`training-logs-${jobId}`),
      mutate('training-jobs'),
      mutate('training-overview'),
      mutate('training-gpus'),
    ])
  }

  if (!jobId) {
    return (
      <div className="training-page">
        <div className="training-page__body">
          <div className="training-card p-8 text-[15px] text-slate-400">缺少训练任务 ID。</div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="page-header">
        <div>
          <div className="training-label uppercase tracking-[0.22em]">Training Job Detail</div>
          <h1 className="mt-2 training-title">{job?.name ?? jobId}</h1>
          <div className="mono mt-1 text-[13px] text-slate-500">{jobId}</div>
          <p className="mt-2 max-w-3xl training-copy">
            查看训练进度、测试结果、checkpoint 和原始日志。训练曲线支持按 step 或 epoch 切换横轴。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button type="button" className="btn-ghost" onClick={() => navigate('/training/jobs')}>
            <ArrowLeft size={14} />
            返回任务列表
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              mutate(`training-job-${jobId}`)
              mutate(`training-curves-${jobId}`)
              mutate(`training-logs-${jobId}`)
            }}
          >
            <RefreshCcw size={14} />
            刷新
          </button>
          {job && ['starting', 'running', 'evaluating'].includes(job.status) && (
            <button type="button" className="btn-danger" onClick={stopJob}>
              <PauseCircle size={14} />
              终止训练
            </button>
          )}
        </div>
      </div>

      {jobError && (
        <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
          加载训练任务失败：{jobError.message}
        </div>
      )}

      <div className="training-page__body">
        {job && (
          <div className="training-scroll">
            <div className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-5">
              <div className="training-card p-3.5">
                <div className="training-label uppercase tracking-[0.18em]">状态</div>
                <div className="mt-2.5">
                  <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
                </div>
                <div className="mt-2 text-[15px] text-slate-400">GPU {job.gpu_id} · PID {job.pid ?? '—'}</div>
              </div>
              <div className="training-card p-3.5">
                <div className="training-label uppercase tracking-[0.18em]">Epoch / Step</div>
                <div className="mt-1.5 text-[30px] font-semibold text-slate-100">
                  {job.latest_epoch ?? '—'} / {job.latest_step ?? '—'}
                </div>
                <div className="mt-1 text-[15px] text-slate-400">global step {job.global_step ?? '—'}</div>
              </div>
              <div className="training-card p-3.5">
                <div className="training-label uppercase tracking-[0.18em]">训练损失</div>
                <div className="mt-1.5 text-[30px] font-semibold text-slate-100">{formatMetric(job.avg_loss, 6)}</div>
                <div className="mt-1 text-[15px] text-slate-400">{formatMetric(job.steps_per_sec, 2)} step/s</div>
              </div>
              <div className="training-card p-3.5">
                <div className="training-label uppercase tracking-[0.18em]">最近测试 F1</div>
                <div className="mt-1.5 text-[30px] font-semibold text-slate-100">
                  {formatMetric(job.latest_metrics?.f1, 4)}
                </div>
                <div className="mt-1 text-[15px] text-slate-400">
                  PR-AUC {formatMetric(job.latest_metrics?.pr_auc, 4)}
                </div>
              </div>
              <div className="training-card p-3.5">
                <div className="training-label uppercase tracking-[0.18em]">预计剩余</div>
                <div className="mt-1.5 text-[30px] font-semibold text-slate-100">{formatDuration(job.eta_seconds)}</div>
                <div className="mt-1 text-[15px] text-slate-400">创建于 {formatDateTime(job.created_at)}</div>
              </div>
            </div>

            <div className="grid gap-3 xl:grid-cols-[1.05fr_0.95fr]">
              <section className="training-card">
                <div className="card-header">
                  <RadioTower size={16} className="text-violet-300" />
                  <div>
                    <div className="text-[17px] font-semibold text-slate-100">任务摘要</div>
                    <div className="text-[15px] text-slate-400">配置、路径和最近状态</div>
                  </div>
                </div>
                <div className="grid gap-2.5 p-3 md:grid-cols-2">
                  <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-3.5">
                    <div className="training-label">任务名称</div>
                    <div className="mt-1 text-[16px] font-semibold text-slate-100">{job.name}</div>
                    <div className="mt-3 training-label">训练数据</div>
                    <div className="mt-1 text-[16px] font-semibold text-slate-100">
                      {job.simulator.toUpperCase()} · {job.scenarios.join(', ')}
                    </div>
                    <div className="mt-2 text-[15px] text-slate-400">
                      test ratio {formatMetric(job.config.test_ratio, 2)} · eval interval {job.config.eval_interval}
                      {' · '}epochs {job.config.epochs === 0 ? '∞' : job.config.epochs}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-3.5">
                    <div className="training-label">训练参数</div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-[15px] text-slate-400">
                      <div>
                        epochs: <span className="mono text-slate-200">{job.config.epochs === 0 ? '∞' : job.config.epochs}</span>
                      </div>
                      <div>
                        batch: <span className="mono text-slate-200">{job.config.batch_size}</span>
                      </div>
                      <div>
                        test batch: <span className="mono text-slate-200">{job.config.test_batch_size}</span>
                      </div>
                      <div>
                        workers: <span className="mono text-slate-200">{job.config.num_workers}</span>
                      </div>
                      <div>
                        lr: <span className="mono text-slate-200">{job.config.learning_rate}</span>
                      </div>
                      <div>
                        wd: <span className="mono text-slate-200">{job.config.weight_decay}</span>
                      </div>
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-700/30 bg-slate-900/30 p-3.5 md:col-span-2">
                    <div className="training-label">运行目录</div>
                    <div className="mono mt-2 text-[13px] text-slate-200" title={job.run_dir}>
                      {shortPath(job.run_dir, 110)}
                    </div>
                    <div className="training-label mt-4">日志文件</div>
                    <div className="mono mt-2 text-[13px] text-slate-200" title={job.log_path}>
                      {shortPath(job.log_path, 110)}
                    </div>
                    {job.error_message && (
                      <div className="mt-4 flex items-start gap-2 rounded-2xl border border-rose-500/20 bg-rose-500/8 px-4 py-3 text-sm text-rose-300">
                        <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
                        <span>{job.error_message}</span>
                      </div>
                    )}
                  </div>
                </div>
              </section>

              <section className="training-card min-h-0">
                <div className="card-header">
                  <Save size={16} className="text-emerald-300" />
                  <div>
                    <div className="text-[17px] font-semibold text-slate-100">Checkpoint</div>
                    <div className="text-[15px] text-slate-400">当前 run 已保存的模型文件</div>
                  </div>
                </div>
                <div className="training-card__body training-scroll">
                  <CheckpointList checkpoints={curves?.checkpoints ?? job.checkpoints} />
                </div>
              </section>
            </div>

            <div className="grid gap-3 xl:grid-cols-2">
              <ChartCard
                title="训练损失曲线"
                subtitle={`avg_loss vs ${trainingChart.subtitleSuffix}`}
                actions={
                  <div className="training-segmented">
                    <button
                      type="button"
                      className={`training-segmented__button ${trainingAxisMode === 'step' ? 'training-segmented__button--active' : ''}`}
                      onClick={() => setTrainingAxisMode('step')}
                    >
                      Step
                    </button>
                    <button
                      type="button"
                      className={`training-segmented__button ${trainingAxisMode === 'epoch' ? 'training-segmented__button--active' : ''}`}
                      onClick={() => setTrainingAxisMode('epoch')}
                    >
                      Epoch
                    </button>
                  </div>
                }
              >
                {trainingChart.data.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingChart.data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(51,65,85,0.3)" strokeDasharray="4 4" />
                      <XAxis dataKey={trainingChart.xKey} stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <YAxis stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <Tooltip
                        contentStyle={{
                          background: 'rgba(15, 23, 42, 0.94)',
                          border: '1px solid rgba(71, 85, 105, 0.55)',
                          borderRadius: 16,
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="avg_loss" name="avg_loss" stroke="#38bdf8" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ChartEmpty message="当前还没有训练曲线点。" />
                )}
              </ChartCard>

              <ChartCard title="训练速度曲线" subtitle={`steps_per_sec vs ${trainingChart.subtitleSuffix}`}>
                {trainingChart.data.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingChart.data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(51,65,85,0.3)" strokeDasharray="4 4" />
                      <XAxis dataKey={trainingChart.xKey} stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <YAxis stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <Tooltip
                        contentStyle={{
                          background: 'rgba(15, 23, 42, 0.94)',
                          border: '1px solid rgba(71, 85, 105, 0.55)',
                          borderRadius: 16,
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="steps_per_sec"
                        name="steps_per_sec"
                        stroke="#34d399"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ChartEmpty message="当前还没有训练曲线点。" />
                )}
              </ChartCard>
            </div>

            <div className="grid gap-3 xl:grid-cols-2">
              <ChartCard title="测试指标曲线" subtitle="每次测试后的整体指标">
                {curves?.test_points?.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={curves.test_points} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(51,65,85,0.3)" strokeDasharray="4 4" />
                      <XAxis dataKey="epoch" stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <YAxis domain={[0, 1]} stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <Tooltip
                        contentStyle={{
                          background: 'rgba(15, 23, 42, 0.94)',
                          border: '1px solid rgba(71, 85, 105, 0.55)',
                          borderRadius: 16,
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="precision" stroke="#38bdf8" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="recall" stroke="#f59e0b" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="f1" stroke="#34d399" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="pr_auc" stroke="#a78bfa" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ChartEmpty message="当前还没有测试点，需要等到 eval_interval 触发测试。" />
                )}
              </ChartCard>

              <ChartCard title="分场景 F1 曲线" subtitle="每个子场景独立观察">
                {scenarioMetricData.scenarioNames.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scenarioMetricData.data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(51,65,85,0.3)" strokeDasharray="4 4" />
                      <XAxis dataKey="epoch" type="number" stroke="rgba(148,163,184,0.9)" fontSize={13} allowDecimals={false} />
                      <YAxis domain={[0, 1]} stroke="rgba(148,163,184,0.9)" fontSize={13} />
                      <Tooltip
                        contentStyle={{
                          background: 'rgba(15, 23, 42, 0.94)',
                          border: '1px solid rgba(71, 85, 105, 0.55)',
                          borderRadius: 16,
                        }}
                      />
                      <Legend />
                      {scenarioMetricData.scenarioNames.map((scenario, index) => {
                        const colors = ['#38bdf8', '#34d399', '#f59e0b', '#f472b6', '#a78bfa', '#fb7185']
                        return (
                          <Line
                            key={scenario}
                            dataKey={scenario}
                            name={scenario}
                            stroke={colors[index % colors.length]}
                            dot={false}
                            strokeWidth={2}
                          />
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ChartEmpty message="当前还没有分场景测试曲线。" />
                )}
              </ChartCard>
            </div>

            <section className="training-card min-h-0">
              <div className="card-header">
                <FileText size={16} className="text-amber-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">训练日志</div>
                  <div className="text-[15px] text-slate-400">最近 400 行原始输出</div>
                </div>
              </div>
              <div className="training-card__body min-h-0">
                <div className="rounded-2xl border border-slate-700/40 bg-slate-950/80 p-4">
                  <pre className="h-[360px] overflow-auto whitespace-pre-wrap break-words text-[13px] leading-6 text-slate-300">
                    {(logs?.lines ?? []).join('\n') || '暂无日志输出。'}
                  </pre>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
