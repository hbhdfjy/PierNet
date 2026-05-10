import { Activity, Database, Gauge, Layers3, PlayCircle, TimerReset } from 'lucide-react'
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
} from '../shared'

function KpiCard({ label, value, note, icon }: { label: string; value: string; note: string; icon: React.ReactNode }) {
  return (
    <div className="training-kpi training-kpi--compact">
      <div className="flex items-center justify-between gap-3">
        <div>
          <span className="training-kpi__label">{label}</span>
          <div className="training-kpi__value">{value}</div>
        </div>
        <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
          {icon}
        </span>
      </div>
      <div className="training-kpi__note">{note}</div>
    </div>
  )
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div>
      <div className="training-panel-title">{title}</div>
      <div className="training-panel-copy">{copy}</div>
    </div>
  )
}

function UsageBar({ value }: { value: number }) {
  const pct = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
  return (
    <div className="training-progress mt-2">
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

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="training-eyebrow">Token Router 训练</div>
                <h1 className="mt-2 text-[1.65rem] font-semibold tracking-tight text-white xl:text-[1.9rem]">
                  训练总览
                </h1>
              </div>
              <div className="flex flex-wrap items-center gap-2.5">
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

            <div className="mt-4 training-kpi-grid">
              <KpiCard
                label="样本"
                value={formatCount(totalDatasets)}
                note={`${data?.datasets.length ?? 0} 个大场景`}
                icon={<Database size={16} />}
              />
              <KpiCard
                label="GPU"
                value={`${availableGpus}/${data?.gpus.length ?? 0}`}
                note="当前可用卡数"
                icon={<Gauge size={16} />}
              />
              <KpiCard
                label="运行中"
                value={formatCount(data?.running_job_count ?? 0)}
                note="启动中 / 训练中 / 评测中 / 停止中"
                icon={<Activity size={16} />}
              />
              <KpiCard
                label="已完成"
                value={formatCount(data?.completed_job_count ?? 0)}
                note="已有权重和测试结果"
                icon={<TimerReset size={16} />}
              />
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              无法加载训练平台数据：{error.message}
            </div>
          )}

          <div className="training-overview-grid">
            <section className="training-card training-card--compact">
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
                  <div className="space-y-2">
                    {data.datasets.map(dataset => (
                      <div key={dataset.simulator} className="training-surface--compact">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-[14px] font-semibold uppercase tracking-[0.14em] text-sky-300">
                              {dataset.simulator}
                            </div>
                            <div className="mt-0.5 text-[13px] text-slate-400">
                              {dataset.scenarios.length} 个子场景 · {formatCount(dataset.total_count)} 条
                            </div>
                          </div>
                          <Link to="/training/new" className="btn-ghost">
                            用于训练
                          </Link>
                        </div>
                        <div className="mt-2 training-chip-grid">
                          {dataset.scenarios.map(scenario => (
                            <div
                              key={scenario.scenario}
                              className="training-chip"
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

            <section className="training-card training-card--compact">
              <div className="card-header">
                <Gauge size={16} className="text-emerald-300" />
                <SectionTitle title="GPU 状态" copy="当前可用的 GPU 卡" />
              </div>
              <div className="training-card__body training-scroll training-overview-scroll">
                {data?.gpus?.length ? (
                  <div className="space-y-2">
                    {data.gpus.map(gpu => {
                      const memoryRatio =
                        gpu.memory_total_mib > 0 ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0
                      return (
                        <div key={gpu.index} className="training-surface--compact">
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <div className="text-[15px] font-semibold text-slate-100">GPU {gpu.index}</div>
                              <div className="mt-0.5 truncate text-[13px] text-slate-400">{gpu.name}</div>
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
                              <UsageBar value={memoryRatio} />
                            </div>
                            <div>
                              <div className="training-label">利用率</div>
                              <div className="mt-0.5 text-[13px] text-slate-200">{gpu.utilization_gpu}%</div>
                              <UsageBar value={gpu.utilization_gpu} />
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

            <section className="training-card training-card--compact">
              <div className="card-header">
                <Layers3 size={16} className="text-violet-300" />
                <SectionTitle title="最近任务" copy="最近 5 个训练任务" />
              </div>
              <div className="training-card__body training-scroll training-overview-scroll">
                {data?.jobs.length ? (
                  <div className="space-y-2">
                    {data.jobs.slice(0, 5).map(job => (
                      <Link key={job.job_id} to={`/training/jobs/${job.job_id}`} className="card-hover block p-3">
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-[15px] font-semibold text-slate-100">{job.name}</div>
                            <div className="mono mt-1 text-[12px] text-slate-500">{job.job_id}</div>
                          </div>
                          <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
                        </div>
                        <div className="mt-1.5 training-note">
                          GPU {job.gpu_id} · {job.simulator} · {job.scenarios.length} 个子场景
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-[13px] text-slate-400">
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
