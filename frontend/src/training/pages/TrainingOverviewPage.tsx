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

function HeroMetric({
  label,
  value,
  note,
  tone,
  icon,
}: {
  label: string
  value: string
  note: string
  tone: string
  icon: React.ReactNode
}) {
  return (
    <div className="rounded-[1.25rem] border border-slate-700/45 bg-slate-950/28 px-4 py-4 backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div className={tone}>{icon}</div>
        <span className="training-label">{label}</span>
      </div>
      <div className="mt-4 text-[1.8rem] font-semibold tracking-tight text-white">{value}</div>
      <div className="mt-1 text-[13px] text-slate-500">{note}</div>
    </div>
  )
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div>
      <h2 className="text-[1.05rem] font-semibold tracking-tight text-slate-100">{title}</h2>
      <p className="mt-1 text-[14px] text-slate-400">{copy}</p>
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
        <div className="space-y-7 p-6">
          <section className="training-card relative overflow-hidden border border-slate-700/45 bg-slate-950/30">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.16),transparent_28%),radial-gradient(circle_at_85%_18%,rgba(56,189,248,0.14),transparent_26%)]" />
            <div className="relative p-6 xl:p-7">
              <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
                <div className="max-w-3xl">
                  <div className="mb-3 flex flex-wrap items-center gap-2">
                    <span className="badge border border-emerald-400/20 bg-emerald-500/10 text-emerald-300">Training</span>
                    <span className="badge border border-slate-700/50 bg-slate-900/40 text-slate-400">Token Router</span>
                  </div>
                  <h1 className="text-[2.1rem] font-semibold tracking-tight text-white xl:text-[2.6rem]">{"\u8bad\u7ec3\u603b\u89c8"}</h1>
                  <p className="mt-3 max-w-2xl text-[15px] leading-7 text-slate-300">
                    {"\u76f4\u63a5\u67e5\u770b\u8bad\u7ec3\u6570\u636e\u3001GPU \u53ef\u7528\u72b6\u6001\u548c\u6700\u8fd1\u4efb\u52a1\u3002\u5f53\u524d\u7248\u672c\u53ea\u652f\u6301\u5355 GPU \u8c03\u5ea6\u3002"}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <Link to="/training/new" className="btn-primary">
                    <PlayCircle size={15} />
                    {"\u65b0\u5efa\u8bad\u7ec3"}
                  </Link>
                  <Link to="/training/jobs" className="btn-ghost">
                    <Layers3 size={15} />
                    {"\u4efb\u52a1\u5217\u8868"}
                  </Link>
                </div>
              </div>

              <div className="mt-7 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <HeroMetric
                  label={"\u6837\u672c"}
                  value={formatCount(totalDatasets)}
                  note={`${data?.datasets.length ?? 0} \u4e2a\u5927\u573a\u666f`}
                  tone="flex h-10 w-10 items-center justify-center rounded-2xl border border-sky-500/20 bg-sky-500/15 text-sky-300"
                  icon={<Database size={17} />}
                />
                <HeroMetric
                  label={"GPU"}
                  value={`${availableGpus}/${data?.gpus.length ?? 0}`}
                  note={"\u5f53\u524d\u53ef\u7528\u5361\u6570"}
                  tone="flex h-10 w-10 items-center justify-center rounded-2xl border border-emerald-500/20 bg-emerald-500/12 text-emerald-300"
                  icon={<Gauge size={17} />}
                />
                <HeroMetric
                  label={"\u8fd0\u884c\u4e2d"}
                  value={formatCount(data?.running_job_count ?? 0)}
                  note={"starting / running / evaluating"}
                  tone="flex h-10 w-10 items-center justify-center rounded-2xl border border-amber-500/20 bg-amber-500/12 text-amber-300"
                  icon={<Activity size={17} />}
                />
                <HeroMetric
                  label={"\u5df2\u5b8c\u6210"}
                  value={formatCount(data?.completed_job_count ?? 0)}
                  note={"\u5df2\u6709 checkpoint \u548c\u6d4b\u8bd5\u7ed3\u679c"}
                  tone="flex h-10 w-10 items-center justify-center rounded-2xl border border-violet-500/20 bg-violet-500/12 text-violet-300"
                  icon={<TimerReset size={17} />}
                />
              </div>
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {"\u65e0\u6cd5\u52a0\u8f7d\u8bad\u7ec3\u5e73\u53f0\u6570\u636e\uff1a"}{error.message}
            </div>
          )}

          <div className="grid gap-4 xl:grid-cols-[1.12fr_0.88fr]">
            <section className="training-card min-h-0">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <SectionTitle title={"\u8bad\u7ec3\u6570\u636e"} copy={"\u57fa\u4e8e Router manifest \u805a\u5408"} />
              </div>
              <div className="training-card__body training-scroll list-scroll-xl">
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
                            <div className="text-[15px] font-semibold uppercase tracking-[0.14em] text-sky-300">
                              {dataset.simulator}
                            </div>
                            <div className="mt-1 text-[14px] text-slate-400">
                              {dataset.scenarios.length} {"\u4e2a\u5b50\u573a\u666f"} / {formatCount(dataset.total_count)} {"\u6761"}
                            </div>
                          </div>
                          <Link to="/training/new" className="btn-ghost">
                            {"\u7528\u4e8e\u8bad\u7ec3"}
                          </Link>
                        </div>
                        <div className="mt-3 grid gap-2.5 md:grid-cols-2">
                          {dataset.scenarios.map(scenario => (
                            <div key={scenario.scenario} className="rounded-2xl border border-slate-700/30 bg-slate-800/30 px-3 py-2.5">
                              <div className="text-[15px] font-medium text-slate-100">{scenario.scenario}</div>
                              <div className="mt-1 text-[14px] text-slate-400">
                                {formatCount(scenario.router_count)} {"\u6761"} / {formatBytes(scenario.file_size_bytes)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    {"\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u4e8e\u8bad\u7ec3\u7684 router \u6570\u636e\u3002"}
                  </div>
                )}
              </div>
            </section>

            <div className="training-stack min-h-0">
              <section className="training-card min-h-0">
                <div className="card-header">
                  <Gauge size={16} className="text-emerald-300" />
                  <SectionTitle title={"GPU \u72b6\u6001"} copy={"\u53ea\u5c55\u793a\u8bad\u7ec3\u5e73\u53f0\u53ef\u89c1\u4fe1\u606f"} />
                </div>
                <div className="training-card__body training-scroll list-scroll-lg">
                  {data?.gpus?.length ? (
                    data.gpus.map(gpu => (
                      <div key={gpu.index} className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-3">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <div className="text-[15px] font-semibold text-slate-100">GPU {gpu.index}</div>
                            <div className="mt-1 text-[14px] text-slate-400">{gpu.name}</div>
                          </div>
                          <span
                            className={
                              gpu.available
                                ? 'badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300'
                                : 'badge border border-amber-500/20 bg-amber-500/12 text-amber-300'
                            }
                          >
                            {gpu.available ? '\u53ef\u7528' : gpu.reason ?? '\u5360\u7528\u4e2d'}
                          </span>
                        </div>
                        <div className="mt-2.5 grid grid-cols-2 gap-2.5 text-[14px] text-slate-400">
                          <div>
                            <div className="training-label">{"\u663e\u5b58"}</div>
                            <div className="mt-1 text-[15px] text-slate-200">
                              {gpuUsageLabel(gpu.memory_used_mib, gpu.memory_total_mib)}
                            </div>
                          </div>
                          <div>
                            <div className="training-label">{"\u5229\u7528\u7387"}</div>
                            <div className="mt-1 text-[15px] text-slate-200">{gpu.utilization_gpu}%</div>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                      {"\u6ca1\u6709 GPU \u4fe1\u606f\u3002"}
                    </div>
                  )}
                </div>
              </section>

              <section className="training-card min-h-0 flex-1">
                <div className="card-header">
                  <Layers3 size={16} className="text-violet-300" />
                  <SectionTitle title={"\u6700\u8fd1\u4efb\u52a1"} copy={"\u6700\u8fd1 5 \u4e2a\u8bad\u7ec3\u8fd0\u884c"} />
                </div>
                <div className="training-card__body training-scroll list-scroll-lg">
                  {data?.jobs.length ? (
                    data.jobs.slice(0, 5).map(job => (
                      <Link key={job.job_id} to={`/training/jobs/${job.job_id}`} className="card-hover block p-3.5">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-[15px] font-semibold text-slate-100">{job.name}</div>
                            <div className="mono mt-1 text-[12px] text-slate-500">{job.job_id}</div>
                            <div className="mt-1 text-[14px] text-slate-400">
                              GPU {job.gpu_id} / {job.simulator} / {job.scenarios.length} {"\u4e2a\u5b50\u573a\u666f"}
                            </div>
                          </div>
                          <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
                        </div>
                        <div className="mt-2.5 grid grid-cols-2 gap-2.5 text-[14px] text-slate-400">
                          <div>
                            <div className="training-label">{"\u521b\u5efa\u65f6\u95f4"}</div>
                            <div className="mt-1 text-[15px] text-slate-200">{formatDateTime(job.created_at)}</div>
                          </div>
                          <div>
                            <div className="training-label">{"\u6700\u8fd1 F1"}</div>
                            <div className="mt-1 text-[15px] text-slate-200">{formatMetric(job.latest_metrics?.f1, 4)}</div>
                          </div>
                        </div>
                      </Link>
                    ))
                  ) : (
                    <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                      {"\u5f53\u524d\u6ca1\u6709\u8bad\u7ec3\u4efb\u52a1\u3002"}
                    </div>
                  )}
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
