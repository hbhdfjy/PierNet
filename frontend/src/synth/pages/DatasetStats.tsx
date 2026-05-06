import useSWR from 'swr'
import { api } from '../../lib/api'
import type { DashboardSummary } from '../../lib/types'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import {
  RefreshCw,
  Database,
  Layers,
  Globe,
  BarChart2,
  FileText,
  GitBranch,
  Activity,
  FolderOpen,
  TrendingUp,
} from 'lucide-react'
import EmptyState from '../components/ui/EmptyState'
import { cn } from '../../lib/utils'
import {
  formatBytes,
  LANGUAGE_LABELS,
  STYLE_LABELS,
  SIMULATOR_LABELS,
  getSimulatorBadgeClass,
} from '../../lib/utils'

const PALETTE = ['#38bdf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#22d3ee', '#fb923c', '#4ade80']

const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#0f172a',
    border: '1px solid rgba(51,65,85,0.7)',
    borderRadius: 16,
    fontSize: 13,
    boxShadow: '0 14px 40px rgba(0,0,0,0.5)',
    padding: '10px 14px',
  },
  labelStyle: { color: '#e2e8f0', fontWeight: 600, marginBottom: 4 },
  itemStyle: { color: '#94a3b8' },
}

function formatCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 100_000 ? 0 : 1)}K`
  return value.toString()
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div>
      <div className="training-panel-title">{title}</div>
      <div className="training-panel-copy">{copy}</div>
    </div>
  )
}

function KpiCard({
  label,
  value,
  note,
  icon,
}: {
  label: string
  value: string
  note: string
  icon: React.ReactNode
}) {
  return (
    <div className="training-kpi">
      <div className="flex items-start justify-between gap-3">
        <span className="training-kpi__label">{label}</span>
        <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
          {icon}
        </span>
      </div>
      <div className="training-kpi__value">{value}</div>
      <div className="training-kpi__note">{note}</div>
    </div>
  )
}

function SectionBlock({
  icon,
  title,
  copy,
  children,
}: {
  icon: React.ReactNode
  title: string
  copy: string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/45 bg-slate-900/45 text-slate-300">
          {icon}
        </div>
        <SectionTitle title={title} copy={copy} />
      </div>
      {children}
    </section>
  )
}

function OverviewHero({
  totalSamples,
  datasetCount,
  scenarioCount,
  simulatorCount,
  routerTotal,
  shapeCount,
  onRefresh,
  loading,
}: {
  totalSamples: number
  datasetCount: number
  scenarioCount: number
  simulatorCount: number
  routerTotal: number
  shapeCount: number
  onRefresh: () => void
  loading: boolean
}) {
  return (
    <section className="training-hero training-hero--compact">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        <div className="max-w-3xl">
          <div className="training-eyebrow">
            <span>数据平台</span>
            <span className="text-slate-500">/</span>
            <span>数据集</span>
          </div>
          <h1 className="mt-2 text-[1.55rem] font-semibold tracking-tight text-white xl:text-[1.75rem]">数据总览</h1>
          <p className="mt-1.5 max-w-2xl training-copy">
            直接查看样本规模、内容分布、文件结构和 Router 训练数据。
          </p>
        </div>
        <button className="btn-ghost self-start" onClick={onRefresh}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          刷新
        </button>
      </div>

      <div className="mt-4 training-kpi-grid">
        <KpiCard label="样本" value={totalSamples.toLocaleString()} note={`${datasetCount} 个 JSONL`} icon={<Database size={16} />} />
        <KpiCard label="场景" value={scenarioCount.toString()} note={`${simulatorCount} 个仿真器`} icon={<Layers size={16} />} />
        <KpiCard label="Router" value={formatCompact(routerTotal)} note="训练输入规模" icon={<GitBranch size={16} />} />
        <KpiCard label="结构" value={shapeCount.toString()} note="时序形状" icon={<BarChart2 size={16} />} />
      </div>
    </section>
  )
}

function PieCard({
  title,
  icon,
  data,
  className,
}: {
  title: string
  icon: React.ReactNode
  data: Record<string, number>
  className?: string
}) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return null

  const total = entries.reduce((sum, [, value]) => sum + value, 0)
  const chartData = entries.map(([name, value]) => ({ name, value }))

  return (
    <div className={cn('training-card overflow-hidden', className)}>
      <div className="card-header">
        <span className="text-slate-400">{icon}</span>
        <SectionTitle title={title} copy={`${entries.length} 项`} />
      </div>
      <div className="training-card__body">
        <div className="flex items-center gap-3.5">
          <div className="relative flex-shrink-0">
            <ResponsiveContainer width={108} height={108}>
              <PieChart>
                <Pie data={chartData} cx="50%" cy="50%" innerRadius={24} outerRadius={48} dataKey="value" paddingAngle={2} strokeWidth={0}>
                  {chartData.map((_, index) => (
                    <Cell key={index} fill={PALETTE[index % PALETTE.length]} />
                  ))}
                </Pie>
                <Tooltip
                  {...TOOLTIP_STYLE}
                  formatter={(value: number) => [`${value.toLocaleString()} (${((value / total) * 100).toFixed(1)}%)`, '']}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="rounded-full bg-slate-950/75 px-3 py-1 text-center backdrop-blur">
                <div className="font-mono text-sm font-semibold text-white">{formatCompact(total)}</div>
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">total</div>
              </div>
            </div>
          </div>
          <div className="min-w-0 flex-1 space-y-2.5">
            {entries.map(([name, value], index) => {
              const pct = ((value / total) * 100).toFixed(0)
              return (
                <div key={name}>
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ background: PALETTE[index % PALETTE.length] }} />
                    <span className="flex-1 truncate text-sm font-medium text-slate-300">
                      {LANGUAGE_LABELS[name] ?? STYLE_LABELS[name] ?? SIMULATOR_LABELS[name] ?? name}
                    </span>
                    <span className="flex-shrink-0 font-mono text-sm tabular-nums text-slate-400">{value.toLocaleString()}</span>
                  </div>
                  <div className="ml-4 h-1.5 overflow-hidden rounded-full bg-slate-800/70">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: PALETTE[index % PALETTE.length] }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function ScenarioBar({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 22)
  if (entries.length === 0) return null

  const maxVal = Math.max(...entries.map(([, value]) => value))
  const minVal = Math.min(...entries.map(([, value]) => value))
  const useLog = maxVal / Math.max(minVal, 1) > 100

  return (
    <div className="training-card overflow-hidden">
      <div className="card-header">
        <Layers size={16} className="text-slate-400" />
        <SectionTitle title="按场景分布" copy={`${entries.length} 个场景${useLog ? ' / 对数轴' : ''}`} />
      </div>
      <div className="training-card__body">
        <div className="list-scroll-lg space-y-3 pr-1">
          {entries.map(([name, count], index) => {
            const pct = useLog
              ? (Math.log10(Math.max(count, 1)) / Math.log10(Math.max(maxVal, 1))) * 100
              : (count / maxVal) * 100
            return (
              <div
                key={name}
                className="group rounded-xl border border-slate-800/50 bg-slate-900/20 px-3 py-2.5 transition-colors hover:border-slate-700/60 hover:bg-slate-900/35"
              >
                <div className="mb-2 flex items-center gap-3">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full border border-slate-700/40 bg-slate-900/60 font-mono text-[11px] text-slate-500">
                    {index + 1}
                  </span>
                  <span className="flex-1 truncate font-mono text-sm text-slate-300">{name}</span>
                  <span className="font-mono text-sm tabular-nums text-sky-300">{count.toLocaleString()}</span>
                </div>
                <div className="relative h-2.5 overflow-hidden rounded-full bg-slate-800/70">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.max(pct, 1)}%`,
                      background: 'linear-gradient(90deg, rgba(14,165,233,0.88), rgba(56,189,248,1))',
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function ShapeTable({ shapes }: { shapes: Record<string, [number, number]> }) {
  const entries = Object.entries(shapes)
  if (entries.length === 0) return null

  return (
    <div className="training-card overflow-hidden">
      <div className="card-header">
        <TrendingUp size={16} className="text-slate-400" />
        <SectionTitle title="时序形状" copy="按仿真器聚合" />
      </div>
      <div className="list-table-scroll">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/40 bg-slate-800/30">
              <th className="px-5 py-3 text-left label">仿真器</th>
              <th className="px-5 py-3 text-right label">通道数</th>
              <th className="px-5 py-3 text-right label">时间点</th>
              <th className="px-5 py-3 text-right label">形状</th>
              <th className="px-5 py-3 text-right label">总元素</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sim, [channels, timesteps]], index) => (
              <tr key={sim} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', index % 2 === 0 ? '' : 'bg-slate-800/10')}>
                <td className="px-5 py-3">
                  <span className={cn('badge border', getSimulatorBadgeClass(sim))}>{SIMULATOR_LABELS[sim] ?? sim}</span>
                </td>
                <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{channels}</td>
                <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{timesteps}</td>
                <td className="px-5 py-3 text-right font-mono text-slate-500">({channels}, {timesteps})</td>
                <td className="px-5 py-3 text-right font-mono tabular-nums font-semibold text-sky-400">{(channels * timesteps).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function RouterStatCard({
  label,
  value,
  note,
  color,
  icon,
}: {
  label: string
  value: string
  note: string
  color: string
  icon: React.ReactNode
}) {
  return (
    <div className="training-surface">
      <div className="flex items-start justify-between gap-3">
        <div className="training-kpi__label">{label}</div>
        <span className={cn('flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/35', color)}>
          {icon}
        </span>
      </div>
      <div className={cn('mt-2 font-mono text-[1.45rem] font-semibold tracking-tight', color)}>{value}</div>
      <div className="mt-0.5 text-[12px] text-slate-400">{note}</div>
    </div>
  )
}

export default function DatasetStats() {
  const statsSwrOptions = { revalidateOnFocus: false }
  const { data: summary, isLoading: loading, mutate: refreshSummary } =
    useSWR<DashboardSummary>('dashboard-summary', () => api.getDashboardSummary(), statsSwrOptions)

  const stats = summary?.stats
  const datasets = summary?.datasets ?? []
  const routerStatus = summary?.router

  const datasetCount = datasets.length
  const scenarioCount = stats ? Object.keys(stats.by_scenario).length : 0
  const simulatorCount = stats ? Object.keys(stats.by_simulator).length : 0
  const shapeCount = stats ? Object.keys(stats.timeseries_shapes).length : 0

  const routerTotal = routerStatus?.total ?? 0
  const routerPositive = routerStatus?.label_counts['1'] ?? 0
  const routerNegative = routerStatus?.label_counts['0'] ?? 0
  const routerPositiveRate = routerTotal > 0 ? ((routerPositive / routerTotal) * 100).toFixed(1) : '0.0'
  const routerNegativeRate = routerTotal > 0 ? ((routerNegative / routerTotal) * 100).toFixed(1) : '0.0'

  return (
    <div className="page-shell">
      <div className="page-content space-y-4 p-4">
        {loading && !stats && (
          <div className="flex h-48 items-center justify-center gap-2.5 text-slate-500">
            <RefreshCw size={16} className="animate-spin" />
            <span>加载中…</span>
          </div>
        )}

        {stats && (
          <>
            <OverviewHero
              totalSamples={stats.total_samples}
              datasetCount={datasetCount}
              scenarioCount={scenarioCount}
              simulatorCount={simulatorCount}
              routerTotal={routerTotal}
              shapeCount={shapeCount}
              onRefresh={() => { refreshSummary() }}
              loading={loading}
            />

            <SectionBlock icon={<Globe size={17} />} title="内容分布" copy="场景规模、语言、风格和时间采样。">
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.22fr_0.78fr]">
                <ScenarioBar data={stats.by_scenario} />
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <PieCard title="语言分布" icon={<Globe size={15} />} data={stats.by_language} />
                  <PieCard title="写作风格" icon={<FileText size={15} />} data={stats.by_style} />
                  <PieCard title="时间采样" icon={<TrendingUp size={15} />} data={stats.by_time_mode} className="md:col-span-2" />
                </div>
              </div>
            </SectionBlock>

            <SectionBlock icon={<FolderOpen size={17} />} title="文件与结构" copy="时序形状和 JSONL 落盘状态。">
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[0.92fr_1.08fr]">
                <ShapeTable shapes={stats.timeseries_shapes} />
                <div className="training-card overflow-hidden">
                  <div className="card-header">
                    <Database size={16} className="text-slate-400" />
                    <SectionTitle title="JSONL 文件" copy={`${datasets.length} 个文件`} />
                  </div>
                  <div className="list-table-scroll">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700/40 bg-slate-800/30">
                          <th className="px-5 py-3 text-left label">场景</th>
                          <th className="px-5 py-3 text-left label">仿真器</th>
                          <th className="px-5 py-3 text-right label">样本数</th>
                          <th className="px-5 py-3 text-right label">文件大小</th>
                          <th className="px-5 py-3 text-right label">修改时间</th>
                        </tr>
                      </thead>
                      <tbody>
                        {datasets.map((dataset, index) => (
                          <tr key={dataset.name} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', index % 2 === 0 ? '' : 'bg-slate-800/10')}>
                            <td className="px-5 py-3 font-mono text-slate-200">{dataset.name}</td>
                            <td className="px-5 py-3">
                              <span className={cn('badge border', getSimulatorBadgeClass(dataset.simulator))}>{SIMULATOR_LABELS[dataset.simulator] ?? dataset.simulator}</span>
                            </td>
                            <td className="px-5 py-3 text-right font-mono tabular-nums text-sky-400">{dataset.sample_count.toLocaleString()}</td>
                            <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{formatBytes(dataset.file_size_bytes)}</td>
                            <td className="px-5 py-3 text-right text-slate-500">{new Date(dataset.mtime * 1000).toLocaleString('zh-CN')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </SectionBlock>
          </>
        )}

        {routerStatus && routerStatus.scenarios.length > 0 && (
          <SectionBlock icon={<GitBranch size={17} />} title="路由训练数据" copy="源样本规模、正负分布和按场景落盘明细。">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <RouterStatCard label="源样本数" value={(routerStatus.source_count ?? 0).toLocaleString()} note="与 Router 源数据一一对应" color="text-sky-400" icon={<Database size={18} />} />
              <RouterStatCard label="正样本" value={routerPositive.toLocaleString()} note={`${routerPositiveRate}% / label = 1`} color="text-emerald-400" icon={<Activity size={18} />} />
              <RouterStatCard label="负样本" value={routerNegative.toLocaleString()} note={`${routerNegativeRate}% / label = 0`} color="text-amber-400" icon={<FileText size={18} />} />
            </div>

            <div className="training-card overflow-hidden">
              <div className="card-header">
                <GitBranch size={16} className="text-rose-400" />
                <SectionTitle title="按场景落盘明细" copy={`${routerStatus.scenarios.length} 个场景`} />
              </div>
              <div className="list-table-scroll-compact">
                <table className="w-full table-fixed text-sm">
                  <thead>
                    <tr className="border-b border-slate-700/40 bg-slate-800/30">
                      <th className="px-5 py-3 text-left label">场景</th>
                      <th className="px-5 py-3 text-left label">仿真器</th>
                      <th className="px-5 py-3 text-right label">Router 样本数</th>
                      <th className="px-5 py-3 text-right label">源数据</th>
                      <th className="px-5 py-3 text-right label">文件大小</th>
                      <th className="px-5 py-3 text-right label">修改时间</th>
                    </tr>
                  </thead>
                  <tbody>
                    {routerStatus.scenarios.map((scenario, index) => (
                      <tr key={scenario.scenario} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', index % 2 === 0 ? '' : 'bg-slate-800/10')}>
                        <td className="px-5 py-3 font-mono text-slate-200">{scenario.scenario}</td>
                        <td className="px-5 py-3">
                          <span className={cn('badge border', getSimulatorBadgeClass(scenario.simulator))}>{SIMULATOR_LABELS[scenario.simulator] ?? scenario.simulator}</span>
                        </td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-rose-400">{(scenario.router_count ?? 0).toLocaleString()}</td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{(routerStatus.source_by_scenario[scenario.scenario] ?? 0).toLocaleString()}</td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{formatBytes(scenario.file_size_bytes ?? 0)}</td>
                        <td className="px-5 py-3 text-right text-slate-500">{scenario.mtime ? new Date(scenario.mtime * 1000).toLocaleString('zh-CN') : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </SectionBlock>
        )}

        {!loading && (!stats || stats.total_samples === 0) && (
          <EmptyState
            icon={Database}
            title="暂无数据"
            description="请先在数据合成流程中产出样本数据，数据总览才会显示统计结果。"
          />
        )}
      </div>
    </div>
  )
}
