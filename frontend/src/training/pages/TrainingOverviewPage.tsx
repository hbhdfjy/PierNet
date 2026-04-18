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

function StatCard({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: React.ElementType
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="training-card p-3.5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="training-label uppercase tracking-[0.18em]">{label}</div>
          <div className="mt-1.5 text-[30px] font-semibold text-slate-100">{value}</div>
          <div className="mt-1 text-[15px] text-slate-400">{hint}</div>
        </div>
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-sky-500/20 bg-sky-500/15 text-sky-300">
          <Icon size={18} />
        </div>
      </div>
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
      <div className="page-header">
        <div>
          <div className="training-label uppercase tracking-[0.22em]">Training Platform</div>
          <h1 className="mt-2 training-title">Token Router 训练总览</h1>
          <p className="mt-2 max-w-3xl training-copy">
            这里集中展示训练数据、GPU 可用状态和最近训练任务。训练平台目前只支持单 GPU 任务调度。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/training/new" className="btn-primary">
            <PlayCircle size={15} />
            新建训练
          </Link>
          <Link to="/training/jobs" className="btn-ghost">
            <Layers3 size={15} />
            查看任务
          </Link>
        </div>
      </div>

      {error && (
        <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
          无法加载训练平台数据：{error.message}
        </div>
      )}

      <div className="training-page__body">
        <div className="training-page__grid xl:grid-cols-[1.18fr_0.82fr]">
          <section className="training-stack min-h-0">
        <div className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                icon={Database}
                label="训练样本"
                value={formatCount(totalDatasets)}
                hint={`${data?.datasets.length ?? 0} 个大场景分组`}
              />
              <StatCard
                icon={Gauge}
                label="可用 GPU"
                value={`${availableGpus}/${data?.gpus.length ?? 0}`}
                hint="当前版本仅允许单卡启动训练"
              />
              <StatCard
                icon={Activity}
                label="运行中任务"
                value={formatCount(data?.running_job_count ?? 0)}
                hint="starting / running / evaluating"
              />
              <StatCard
                icon={TimerReset}
                label="已完成任务"
                value={formatCount(data?.completed_job_count ?? 0)}
                hint="已有测试结果和 checkpoint"
              />
            </div>

            <section className="training-card min-h-0 flex-1">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">训练数据</div>
                  <div className="text-[15px] text-slate-400">基于当前 router manifest 聚合</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                {isLoading && !data ? (
                  <div className="space-y-2.5">
                    {[0, 1, 2].map(item => (
                      <div key={item} className="skeleton h-20 rounded-2xl" />
                    ))}
                  </div>
                ) : data?.datasets.length ? (
                  <div className="space-y-3">
                    {data.datasets.map(dataset => (
                      <div key={dataset.simulator} className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-3.5">
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-[16px] font-semibold uppercase tracking-[0.14em] text-sky-300">
                              {dataset.simulator}
                            </div>
                            <div className="mt-1 text-[15px] text-slate-400">
                              {dataset.scenarios.length} 个子场景，{formatCount(dataset.total_count)} 条 router 样本
                            </div>
                          </div>
                          <Link to="/training/new" className="btn-ghost">
                            用于训练
                          </Link>
                        </div>
                        <div className="mt-3 grid gap-2.5 md:grid-cols-2">
                          {dataset.scenarios.map(scenario => (
                            <div key={scenario.scenario} className="rounded-2xl border border-slate-700/30 bg-slate-800/30 px-3 py-2">
                              <div className="text-[16px] font-medium text-slate-100">{scenario.scenario}</div>
                              <div className="mt-1 text-[15px] text-slate-400">
                                {formatCount(scenario.router_count)} 条 · {formatBytes(scenario.file_size_bytes)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    当前没有可用于训练的 router 数据。
                  </div>
                )}
              </div>
            </section>
          </section>

          <div className="training-stack min-h-0">
            <section className="training-card min-h-0">
              <div className="card-header">
                <Gauge size={16} className="text-emerald-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">GPU 状态</div>
                  <div className="text-[15px] text-slate-400">只显示训练平台可见的单卡分配信息</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                {data?.gpus?.length ? (
                  data.gpus.map(gpu => (
                    <div key={gpu.index} className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-3">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="text-[16px] font-semibold text-slate-100">GPU {gpu.index}</div>
                          <div className="mt-1 text-[15px] text-slate-400">{gpu.name}</div>
                        </div>
                        <span
                          className={
                            gpu.available
                              ? 'badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300'
                              : 'badge border border-amber-500/20 bg-amber-500/12 text-amber-300'
                          }
                        >
                          {gpu.available ? '可用' : gpu.reason ?? '占用中'}
                        </span>
                      </div>
                      <div className="mt-2.5 grid grid-cols-2 gap-2.5 text-[15px] text-slate-400">
                        <div>
                          <div className="training-label">显存</div>
                          <div className="mt-1 text-[16px] text-slate-200">
                            {gpuUsageLabel(gpu.memory_used_mib, gpu.memory_total_mib)}
                          </div>
                        </div>
                        <div>
                          <div className="training-label">利用率</div>
                          <div className="mt-1 text-[16px] text-slate-200">{gpu.utilization_gpu}%</div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    没有 GPU 信息。
                  </div>
                )}
              </div>
            </section>

            <section className="training-card min-h-0 flex-1">
              <div className="card-header">
                <Layers3 size={16} className="text-violet-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">最近任务</div>
                  <div className="text-[15px] text-slate-400">最近 5 个训练任务快照</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                {data?.jobs.length ? (
                  data.jobs.slice(0, 5).map(job => (
                    <Link key={job.job_id} to={`/training/jobs/${job.job_id}`} className="card-hover block p-3.5">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-[16px] font-semibold text-slate-100">{job.name}</div>
                          <div className="mono mt-1 text-[12px] text-slate-500">{job.job_id}</div>
                          <div className="mt-1 text-[15px] text-slate-400">
                            GPU {job.gpu_id} · {job.simulator} · {job.scenarios.length} 个子场景
                          </div>
                        </div>
                        <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
                      </div>
                      <div className="mt-2.5 grid grid-cols-2 gap-2.5 text-[15px] text-slate-400">
                        <div>
                          <div className="training-label">创建时间</div>
                          <div className="mt-1 text-[16px] text-slate-200">{formatDateTime(job.created_at)}</div>
                        </div>
                        <div>
                          <div className="training-label">最近 F1</div>
                          <div className="mt-1 text-[16px] text-slate-200">{formatMetric(job.latest_metrics?.f1, 4)}</div>
                        </div>
                      </div>
                    </Link>
                  ))
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    当前没有训练任务。
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
