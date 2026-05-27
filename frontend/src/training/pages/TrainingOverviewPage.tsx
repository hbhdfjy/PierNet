import { Activity, CheckCircle2, Cpu, Database, Gauge, Layers3, PlayCircle, TimerReset } from 'lucide-react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { TrainingOverview } from '../../lib/types'
import {
  formatBytes,
  formatCount,
  formatDateTime,
  formatMetric,
  gpuUsageLabel,
  statusBadgeClass,
  statusLabel,
  trainingJobDetailPath,
} from '../shared'

type KpiTone = 'sky' | 'emerald' | 'violet' | 'amber'

function KpiCard({
  label,
  value,
  note,
  icon,
  tone = 'sky',
}: {
  label: string
  value: string
  note: string
  icon: React.ReactNode
  tone?: KpiTone
}) {
  return (
    <div className={`training-kpi training-overview-kpi training-overview-kpi--${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="training-kpi__label">{label}</span>
          <div className="training-kpi__value">{value}</div>
        </div>
        <span className="training-kpi__icon">{icon}</span>
      </div>
      <div className="training-kpi__note">{note}</div>
    </div>
  )
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div className="min-w-0">
      <div className="training-panel-title">{title}</div>
      <div className="training-panel-copy">{copy}</div>
    </div>
  )
}

function UsageBar({ value, tone = 'sky' }: { value: number; tone?: KpiTone }) {
  const pct = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
  return (
    <div className={`training-progress training-progress--${tone} mt-2`}>
      <div className="training-progress__fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function TrainingOverviewPage() {
  const { data, error, isLoading } = useSWR<TrainingOverview>('training-overview', api.getTrainingOverview, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const totalDatasets = data?.datasets.reduce((sum, item) => sum + item.total_count, 0) ?? 0
  const availableGpus = data?.gpus.filter(gpu => gpu.available).length ?? 0
  const busyGpus = (data?.gpus.length ?? 0) - availableGpus
  const latestJob = data?.jobs[0]
  const totalGpuMemory = data?.gpus.reduce((sum, gpu) => sum + gpu.memory_total_mib, 0) ?? 0
  const usedGpuMemory = data?.gpus.reduce((sum, gpu) => sum + gpu.memory_used_mib, 0) ?? 0
  const memoryRatio = totalGpuMemory > 0 ? (usedGpuMemory / totalGpuMemory) * 100 : 0

  return (
    <div className="training-page">
      <div className="training-page__body training-overview-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-overview-hero">
            <div className="training-overview-hero__top">
              <div className="training-overview-hero__copy">
                <div className="training-eyebrow">Token Router 训练</div>
                <h1 className="training-overview-hero__title">训练总览</h1>
                <p className="training-copy">
                  查看可训练数据、GPU 可用性和最近训练任务，快速判断是否可以启动新的 Token Router 训练。
                </p>
              </div>
              <div className="training-overview-hero__actions">
                <Link to="/training/new" className="btn-primary">
                  <PlayCircle size={15} />
                  新建训练
                </Link>
                <Link to="/training/jobs" className="btn-ghost">
                  <Layers3 size={15} />
                  任务列表
                </Link>
              </div>
            </div>

            <div className="training-overview-statusbar">
              <div className="training-overview-statusbar__item">
                <CheckCircle2 size={13} />
                <span>{availableGpus > 0 ? `${availableGpus} 张 GPU 可用` : '暂无可用 GPU'}</span>
              </div>
              <div className="training-overview-statusbar__item">
                <Database size={13} />
                <span>{formatCount(totalDatasets)} 条 Router 样本</span>
              </div>
              <div className="training-overview-statusbar__item">
                <Activity size={13} />
                <span>
                  {latestJob
                    ? `最近任务 ${statusLabel(latestJob.status)} · ${formatDateTime(latestJob.created_at)}`
                    : '暂无训练任务'}
                </span>
              </div>
            </div>

            <div className="training-overview-hero__kpis training-kpi-grid">
              <KpiCard
                label="样本"
                value={formatCount(totalDatasets)}
                note={`${data?.datasets.length ?? 0} 个大场景`}
                icon={<Database size={16} />}
                tone="sky"
              />
              <KpiCard
                label="GPU"
                value={`${availableGpus}/${data?.gpus.length ?? 0}`}
                note={busyGpus > 0 ? `${busyGpus} 张占用中` : '全部空闲可调度'}
                icon={<Gauge size={16} />}
                tone="emerald"
              />
              <KpiCard
                label="活跃中"
                value={formatCount(data?.running_job_count ?? 0)}
                note="排队 / 启动 / 训练 / 评测 / 停止"
                icon={<Activity size={16} />}
                tone="violet"
              />
              <KpiCard
                label="已完成"
                value={formatCount(data?.completed_job_count ?? 0)}
                note="已有权重和测试结果"
                icon={<TimerReset size={16} />}
                tone="amber"
              />
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              无法加载训练平台数据：{error.message}
            </div>
          )}

          <div className="training-overview-grid training-overview-grid--balanced">
            <section className="training-card training-card--compact training-overview-panel training-overview-panel--datasets">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <SectionTitle title="训练数据" copy="按大场景聚合的 Router 训练集" />
              </div>
              <div className="training-card__body training-scroll training-overview-scroll">
                {isLoading && !data ? (
                  <div className="space-y-2">
                    {[0, 1, 2].map(item => (
                      <div key={item} className="skeleton h-20 rounded-xl" />
                    ))}
                  </div>
                ) : data?.datasets.length ? (
                  <div className="training-overview-list">
                    {data.datasets.map(dataset => (
                      <div key={dataset.simulator} className="training-dataset-card">
                        <div className="training-dataset-card__head">
                          <div className="min-w-0">
                            <div className="training-dataset-card__name">{dataset.simulator}</div>
                            <div className="training-dataset-card__meta">
                              {dataset.scenarios.length} 个子场景 · {formatCount(dataset.total_count)} 条
                            </div>
                          </div>
                          <Link to="/training/new" className="btn-ghost py-1.5 text-xs">
                            用于训练
                          </Link>
                        </div>
                        <div className="training-chip-grid training-chip-grid--compact">
                          {dataset.scenarios.map(scenario => (
                            <div
                              key={scenario.scenario}
                              className="training-chip training-chip--dataset"
                              title={`${scenario.scenario} · ${formatCount(scenario.router_count)} 条 · ${formatBytes(scenario.file_size_bytes)}`}
                            >
                              <span className="truncate text-slate-200">{scenario.scenario}</span>
                              <span className="mono text-[11px] text-slate-500">
                                {formatCount(scenario.router_count)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="training-surface text-sm text-slate-400">当前没有可用于训练的 router 数据。</div>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-overview-panel">
              <div className="card-header">
                <Cpu size={16} className="text-emerald-300" />
                <SectionTitle title="GPU 状态" copy="当前可用的 GPU 卡" />
              </div>
              <div className="training-card__body training-scroll training-overview-scroll">
                {data?.gpus?.length ? (
                  <div className="training-overview-list">
                    <div className="training-gpu-summary">
                      <div>
                        <div className="training-label">总显存占用</div>
                        <div className="mt-1 text-[13px] text-slate-200">
                          {gpuUsageLabel(usedGpuMemory, totalGpuMemory)} · {memoryRatio.toFixed(1)}%
                        </div>
                      </div>
                      <UsageBar value={memoryRatio} tone="emerald" />
                    </div>
                    {data.gpus.map(gpu => {
                      const gpuMemoryRatio =
                        gpu.memory_total_mib > 0 ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0
                      return (
                        <div key={gpu.index} className="training-gpu-card">
                          <div className="training-gpu-card__head">
                            <div className="min-w-0">
                              <div className="training-gpu-card__title">GPU {gpu.index}</div>
                              <div className="training-gpu-card__name">{gpu.name}</div>
                            </div>
                            <span
                              className={
                                gpu.available
                                  ? 'badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300'
                                  : 'badge border border-amber-500/20 bg-amber-500/12 text-amber-300'
                              }
                            >
                              {gpu.available ? '可用' : (gpu.reason ?? '占用中')}
                            </span>
                          </div>
                          <div className="mt-2 training-stat-grid">
                            <div>
                              <div className="training-label">显存</div>
                              <div className="mt-0.5 text-[13px] text-slate-200">
                                {gpuUsageLabel(gpu.memory_used_mib, gpu.memory_total_mib)}
                              </div>
                              <UsageBar value={gpuMemoryRatio} tone="emerald" />
                            </div>
                            <div>
                              <div className="training-label">利用率</div>
                              <div className="mt-0.5 text-[13px] text-slate-200">{gpu.utilization_gpu}%</div>
                              <UsageBar value={gpu.utilization_gpu} tone="sky" />
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className="training-surface text-sm text-slate-400">没有 GPU 信息。</div>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-overview-panel">
              <div className="card-header">
                <Layers3 size={16} className="text-violet-300" />
                <SectionTitle title="最近任务" copy="最近 5 个训练任务" />
              </div>
              <div className="training-card__body training-scroll training-overview-scroll">
                {data?.jobs.length ? (
                  <div className="training-overview-list">
                    {data.jobs.slice(0, 5).map(job => (
                      <Link
                        key={job.job_id}
                        to={trainingJobDetailPath(job.job_id)}
                        className="training-recent-job card-hover"
                      >
                        <div className="training-recent-job__top">
                          <div className="min-w-0">
                            <div className="training-recent-job__name">{job.name}</div>
                            <div className="training-recent-job__id mono">{job.job_id}</div>
                          </div>
                          <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
                        </div>
                        <div className="training-recent-job__note">
                          GPU {job.gpu_id} · {job.simulator} · {job.scenarios.length} 个子场景
                        </div>
                        <div className="training-recent-job__metrics">
                          <div>
                            <div className="training-label">创建时间</div>
                            <div className="mt-0.5 text-[13px] text-slate-200">{formatDateTime(job.created_at)}</div>
                          </div>
                          <div>
                            <div className="training-label">最近 F1</div>
                            <div className="mt-0.5 text-[13px] text-slate-200">
                              {formatMetric(job.latest_metrics?.f1, 4)}
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="training-surface text-sm text-slate-400">当前没有训练任务。</div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
