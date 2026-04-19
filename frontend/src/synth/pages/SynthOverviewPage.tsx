import { BarChart2, Database, FileText, GitBranch, PlayCircle, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { DashboardSummary, SimulationScenario, TemplateFileInfo, SampleFileInfo } from '../../lib/types'
import { formatBytes, SIMULATOR_LABELS } from '../../lib/utils'

function formatCount(value: number): string {
  return new Intl.NumberFormat('zh-CN').format(value)
}

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

export default function SynthOverviewPage() {
  const { data: summary, error, isLoading } = useSWR<DashboardSummary>(
    'synth-dashboard-summary',
    api.getDashboardSummary,
    { refreshInterval: 5000, revalidateOnFocus: false },
  )
  const { data: simulations } = useSWR<SimulationScenario[]>(
    'synth-simulation-scenarios',
    () => api.getSimulationScenarios(),
    { refreshInterval: 10000, revalidateOnFocus: false },
  )
  const { data: templates } = useSWR<TemplateFileInfo[]>(
    'synth-template-files',
    api.listTemplateFiles,
    { refreshInterval: 10000, revalidateOnFocus: false },
  )
  const { data: samples } = useSWR<SampleFileInfo[]>(
    'synth-sample-files',
    api.listSampleFiles,
    { refreshInterval: 10000, revalidateOnFocus: false },
  )

  const simulationCount = simulations?.length ?? 0
  const simulationSamples = simulations?.reduce((sum, item) => sum + item.sample_count, 0) ?? 0
  const templateCount = templates?.reduce((sum, item) => sum + item.template_count, 0) ?? 0
  const templateScenarios = templates?.length ?? 0
  const sampleCount = summary?.stats.total_samples ?? samples?.reduce((sum, item) => sum + item.sample_count, 0) ?? 0
  const routerCount = summary?.router.total ?? 0
  const routerScenarioCount = summary?.router.scenarios.length ?? 0

  const topScenarios = [...(summary?.datasets ?? [])]
    .sort((a, b) => b.sample_count - a.sample_count)
    .slice(0, 6)

  const routerScenarios = [...(summary?.router.scenarios ?? [])]
    .sort((a, b) => (b.router_count ?? 0) - (a.router_count ?? 0))
    .slice(0, 6)

  return (
    <div className="training-page">
      <div className="page-header">
        <div>
          <div className="training-label uppercase tracking-[0.22em]">Synthesis Platform</div>
          <h1 className="mt-2 training-title">数据合成总览</h1>
          <p className="mt-2 max-w-3xl training-copy">
            集中展示 Stage 1 到 Stage 4 的关键统计、数据规模和快捷入口。数据合成平台现在以
            <span className="mono px-1 text-slate-200">/synth</span>
            作为独立入口。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/synth/simulate" className="btn-primary">
            <PlayCircle size={15} />
            进入仿真
          </Link>
          <Link to="/training" className="btn-ghost">
            <GitBranch size={15} />
            前往训练平台
          </Link>
        </div>
      </div>

      {error && (
        <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
          无法加载数据合成总览：{error.message}
        </div>
      )}

      <div className="training-page__body">
        <div className="training-page__grid xl:grid-cols-[1.15fr_0.85fr]">
          <section className="training-stack min-h-0">
            <div className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                icon={Database}
                label="Stage 1"
                value={formatCount(simulationSamples)}
                hint={`${simulationCount} 个仿真场景`}
              />
              <StatCard
                icon={Sparkles}
                label="Stage 2"
                value={formatCount(templateCount)}
                hint={`${templateScenarios} 个模板库文件`}
              />
              <StatCard
                icon={FileText}
                label="Stage 3"
                value={formatCount(sampleCount)}
                hint={`${summary?.datasets.length ?? 0} 个样本场景`}
              />
              <StatCard
                icon={GitBranch}
                label="Stage 4"
                value={formatCount(routerCount)}
                hint={`${routerScenarioCount} 个 router 场景`}
              />
            </div>

            <section className="training-card min-h-0 flex-1">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">Stage 3 样本场景</div>
                  <div className="text-[15px] text-slate-400">按样本规模排序的前 6 个场景</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                {isLoading && !summary ? (
                  <div className="space-y-2.5">
                    {[0, 1, 2].map(item => (
                      <div key={item} className="skeleton h-20 rounded-2xl" />
                    ))}
                  </div>
                ) : topScenarios.length ? (
                  <div className="space-y-3">
                    {topScenarios.map((scenario) => (
                      <Link
                        key={scenario.name}
                        to="/synth/samples"
                        className="card-hover block p-3.5"
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-[16px] font-semibold text-slate-100">{scenario.name}</div>
                            <div className="mt-1 text-[15px] text-slate-400">
                              {SIMULATOR_LABELS[scenario.simulator] ?? scenario.simulator}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[16px] font-semibold text-slate-100">
                              {formatCount(scenario.sample_count)}
                            </div>
                            <div className="mt-1 text-[15px] text-slate-400">
                              {formatBytes(scenario.file_size_bytes)}
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    当前没有 Stage 3 样本数据。
                  </div>
                )}
              </div>
            </section>
          </section>

          <div className="training-stack min-h-0">
            <section className="training-card min-h-0">
              <div className="card-header">
                <GitBranch size={16} className="text-emerald-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">Stage 4 Router 数据</div>
                  <div className="text-[15px] text-slate-400">按 router 样本规模排序的前 6 个场景</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                {routerScenarios.length ? (
                  <div className="space-y-3">
                    {routerScenarios.map((scenario) => (
                      <Link
                        key={scenario.scenario}
                        to="/synth/router-viewer"
                        className="card-hover block p-3.5"
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-[16px] font-semibold text-slate-100">{scenario.scenario}</div>
                            <div className="mt-1 text-[15px] text-slate-400">
                              {SIMULATOR_LABELS[scenario.simulator] ?? scenario.simulator}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[16px] font-semibold text-slate-100">
                              {formatCount(scenario.router_count ?? 0)}
                            </div>
                            <div className="mt-1 text-[15px] text-slate-400">
                              {formatBytes(scenario.file_size_bytes ?? 0)}
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 px-4 py-5 text-sm text-slate-400">
                    当前没有 Stage 4 router 数据。
                  </div>
                )}
              </div>
            </section>

            <section className="training-card min-h-0 flex-1">
              <div className="card-header">
                <BarChart2 size={16} className="text-violet-300" />
                <div>
                  <div className="text-[17px] font-semibold text-slate-100">快捷入口</div>
                  <div className="text-[15px] text-slate-400">常用工作流页面</div>
                </div>
              </div>
              <div className="training-card__body training-scroll">
                <div className="grid gap-3">
                  <Link to="/synth/simulate" className="card-hover block p-3.5">
                    <div className="text-[16px] font-semibold text-slate-100">物理仿真</div>
                    <div className="mt-1 text-[15px] text-slate-400">Stage 1 仿真运行和历史记录</div>
                  </Link>
                  <Link to="/synth/templates" className="card-hover block p-3.5">
                    <div className="text-[16px] font-semibold text-slate-100">模板生成</div>
                    <div className="mt-1 text-[15px] text-slate-400">Stage 2 模板配置和批量生成</div>
                  </Link>
                  <Link to="/synth/fill" className="card-hover block p-3.5">
                    <div className="text-[16px] font-semibold text-slate-100">样本填充</div>
                    <div className="mt-1 text-[15px] text-slate-400">Stage 3 数值填充和样本写出</div>
                  </Link>
                  <Link to="/synth/router" className="card-hover block p-3.5">
                    <div className="text-[16px] font-semibold text-slate-100">构建路由</div>
                    <div className="mt-1 text-[15px] text-slate-400">Stage 4 router 数据生成和管理</div>
                  </Link>
                  <Link to="/synth/stats" className="card-hover block p-3.5">
                    <div className="text-[16px] font-semibold text-slate-100">详细统计</div>
                    <div className="mt-1 text-[15px] text-slate-400">保留原来的数据统计页面</div>
                  </Link>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
