import useSWR from 'swr'
import { api } from '../../lib/api'
import type { DashboardSummary } from '../../lib/types'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
} from 'recharts'
import {
  formatBytes, LANGUAGE_LABELS, STYLE_LABELS,
  SIMULATOR_LABELS, getSimulatorBadgeClass,
} from '../../lib/utils'
import {
  RefreshCw,
  Database,
  Layers,
  TrendingUp,
  Globe,
  BarChart2,
  FileText,
  GitBranch,
  Activity,
  FolderOpen,
} from 'lucide-react'
import EmptyState from '../components/ui/EmptyState'
import { cn } from '../../lib/utils'

const PALETTE = ['#38bdf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#22d3ee', '#fb923c', '#4ade80']

const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#0f172a',
    border: '1px solid rgba(51,65,85,0.7)',
    borderRadius: 12,
    fontSize: 13,
    boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
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
    <section className="card relative overflow-hidden border border-slate-700/50 bg-slate-950/30">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_28%),radial-gradient(circle_at_82%_18%,rgba(52,211,153,0.14),transparent_26%),linear-gradient(180deg,rgba(15,23,42,0.12),rgba(15,23,42,0.34))]" />
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-sky-400/0 via-sky-300/70 to-emerald-300/0" />
      <div className="relative p-6 xl:p-7">
        <div className="flex items-start justify-between gap-4">
          <div className="max-w-3xl">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span className="badge border border-sky-400/20 bg-sky-500/10 text-sky-300">PiERN</span>
              <span className="badge border border-emerald-400/20 bg-emerald-500/10 text-emerald-300">{"\u6570\u636e\u5408\u6210\u5e73\u53f0"}</span>
              <span className="badge border border-slate-700/50 bg-slate-900/50 text-slate-400">{"\u9996\u9875\u603b\u89c8"}</span>
            </div>
            <h1 className="text-[2rem] font-semibold tracking-tight text-white xl:text-[2.35rem]">{"\u6570\u636e\u603b\u89c8"}</h1>
            <p className="mt-3 max-w-2xl text-[15px] leading-7 text-slate-300/90 xl:text-[16px]">
              {"\u628a\u6837\u672c\u89c4\u6a21\u3001\u5185\u5bb9\u5206\u5e03\u3001\u6587\u4ef6\u7ed3\u6784\u548c Router \u8bad\u7ec3\u6570\u636e\u653e\u5728\u540c\u4e00\u4e2a\u5165\u53e3\u91cc\uff0c\u6253\u5f00\u5c31\u80fd\u76f4\u63a5\u770b\u5230\u6570\u636e\u5f53\u524d\u72b6\u6001\u3002"}
            </p>
            <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-slate-400">
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/50 bg-slate-900/35 px-3 py-1.5">
                <Layers size={14} className="text-slate-400" />
                <span>{scenarioCount} {"\u4e2a\u573a\u666f"}</span>
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/50 bg-slate-900/35 px-3 py-1.5">
                <Database size={14} className="text-slate-400" />
                <span>{datasetCount} JSONL</span>
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/50 bg-slate-900/35 px-3 py-1.5">
                <BarChart2 size={14} className="text-slate-400" />
                <span>{simulatorCount} Simulator</span>
              </span>
            </div>
          </div>
          <button className="btn-ghost self-start" onClick={onRefresh}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            {"\u5237\u65b0"}
          </button>
        </div>

        <div className="mt-7 grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[1.5rem] border border-slate-700/50 bg-slate-950/35 px-6 py-6 shadow-[0_18px_48px_rgba(2,6,23,0.22)] backdrop-blur">
            <div className="label text-sky-300/80">{"\u5f53\u524d\u6837\u672c\u89c4\u6a21"}</div>
            <div className="mt-3 font-mono text-[2.75rem] font-semibold leading-none tracking-tight text-white xl:text-[3.25rem]">
              {totalSamples.toLocaleString()}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-400">
              <span>{"\u76ee\u524d\u5df2\u805a\u5408"} {datasetCount} {"\u4e2a\u6837\u672c\u6587\u4ef6"}</span>
              <span className="h-1 w-1 rounded-full bg-slate-600" />
              <span>{"\u53ef\u76f4\u63a5\u7528\u4e8e\u67e5\u770b\u3001\u586b\u5145\u548c\u8bad\u7ec3"}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <HeroMetric
              label={"\u573a\u666f"}
              value={scenarioCount}
              accent="text-emerald-300"
              glow="bg-emerald-400/15"
              note={`${simulatorCount} Simulator`}
              icon={<Layers size={16} />}
            />
            <HeroMetric
              label={"Router"}
              value={routerTotal}
              accent="text-rose-300"
              glow="bg-rose-400/15"
              note={"\u8def\u7531\u8bad\u7ec3\u6570\u636e"}
              icon={<GitBranch size={16} />}
            />
            <HeroMetric
              label={"JSONL"}
              value={datasetCount}
              accent="text-sky-300"
              glow="bg-sky-400/15"
              note={"\u6837\u672c\u6587\u4ef6"}
              icon={<Database size={16} />}
            />
            <HeroMetric
              label={"\u7ed3\u6784"}
              value={shapeCount}
              accent="text-violet-300"
              glow="bg-violet-400/15"
              note={"\u65f6\u5e8f\u5f62\u72b6"}
              icon={<TrendingUp size={16} />}
            />
          </div>
        </div>
      </div>
    </section>
  )
}

function HeroMetric({
  label,
  value,
  accent,
  glow,
  note,
  icon,
}: {
  label: string
  value: string | number
  accent: string
  glow: string
  note: string
  icon: React.ReactNode
}) {
  return (
    <div className="rounded-[1.25rem] border border-slate-700/45 bg-slate-950/30 px-4 py-4 backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div className={cn('flex h-9 w-9 items-center justify-center rounded-xl text-white', glow)}>
          <span className={accent}>{icon}</span>
        </div>
        <span className="label">{label}</span>
      </div>
      <div className={cn('mt-4 font-mono text-[1.65rem] font-semibold leading-none tabular-nums', accent)}>
        {typeof value === 'number' ? formatCompact(value) : value}
      </div>
      <div className="mt-2 text-xs text-slate-500">{note}</div>
    </div>
  )
}

function SectionBlock({
  icon,
  title,
  description,
  children,
}: {
  icon: React.ReactNode
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2 xl:flex-row xl:items-end xl:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-700/45 bg-slate-900/45 text-slate-300">
            {icon}
          </div>
          <div>
            <h2 className="text-[1.15rem] font-semibold tracking-tight text-slate-100">{title}</h2>
            <p className="mt-1 max-w-3xl text-[14px] leading-6 text-slate-400">{description}</p>
          </div>
        </div>
      </div>
      {children}
    </section>
  )
}

function StatCard({
  label,
  value,
  sub,
  color,
  dotColor,
  icon,
}: {
  label: string
  value: string | number
  sub: string
  color: string
  dotColor: string
  icon: React.ReactNode
}) {
  return (
    <div className="card-hover group relative overflow-hidden px-5 py-4">
      <div className={cn('absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-current to-transparent opacity-70', color)} />
      <div className="mb-3 flex items-start justify-between">
        <div className={cn('flex h-10 w-10 items-center justify-center rounded-2xl', `${dotColor}/15`)}>
          <span className={color}>{icon}</span>
        </div>
        <div className={cn('mt-2 h-2.5 w-2.5 rounded-full', dotColor)} />
      </div>
      <div className={cn('mb-1 font-mono text-[1.9rem] font-semibold tabular-nums tracking-tight', color)}>{value}</div>
      <div className="text-[12px] font-semibold uppercase tracking-[0.16em] text-slate-500">{label}</div>
      <div className="mt-1 text-sm text-slate-400">{sub}</div>
    </div>
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

  const total = entries.reduce((s, [, v]) => s + v, 0)
  const chartData = entries.map(([k, v]) => ({ name: k, value: v }))

  return (
    <div className={cn('card relative overflow-hidden', className)}>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.08),transparent_32%)]" />
      <div className="relative">
        <div className="card-header gap-2">
          <span className="text-slate-400">{icon}</span>
          <span className="text-base font-semibold text-slate-200">{title}</span>
          <span className="ml-auto badge border border-slate-600/30 bg-slate-700/60 text-slate-400">
            {entries.length} {"\u9879"}
          </span>
        </div>
        <div className="flex items-center gap-5 p-5">
          <div className="relative flex-shrink-0">
            <ResponsiveContainer width={124} height={124}>
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={28}
                  outerRadius={54}
                  dataKey="value"
                  paddingAngle={2}
                  strokeWidth={0}
                >
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                  ))}
                </Pie>
                <Tooltip
                  {...TOOLTIP_STYLE}
                  formatter={(v: number) => [`${v.toLocaleString()} (${((v / total) * 100).toFixed(1)}%)`, '']}
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
          <div className="min-w-0 flex-1 space-y-3">
            {entries.map(([k, v], i) => {
              const pct = ((v / total) * 100).toFixed(0)
              return (
                <div key={k}>
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ background: PALETTE[i % PALETTE.length] }} />
                    <span className="flex-1 truncate text-sm font-medium text-slate-300">
                      {LANGUAGE_LABELS[k] ?? STYLE_LABELS[k] ?? SIMULATOR_LABELS[k] ?? k}
                    </span>
                    <span className="flex-shrink-0 font-mono text-sm tabular-nums text-slate-400">{v.toLocaleString()}</span>
                  </div>
                  <div className="ml-4 h-1.5 overflow-hidden rounded-full bg-slate-800/70">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: PALETTE[i % PALETTE.length] }} />
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

  const maxVal = Math.max(...entries.map(([, v]) => v))
  const minVal = Math.min(...entries.map(([, v]) => v))
  const useLog = maxVal / Math.max(minVal, 1) > 100

  return (
    <div className="card relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.08),transparent_30%)]" />
      <div className="relative">
        <div className="card-header">
          <Layers size={15} className="text-slate-400" />
          <span className="text-base font-semibold text-slate-200">{"\u6309\u573a\u666f\u5206\u5e03"}</span>
          <div className="ml-auto flex items-center gap-2">
            {useLog && (
              <span className="badge border border-amber-500/20 bg-amber-500/10 text-amber-400 text-xs">{"\u5bf9\u6570\u8f74"}</span>
            )}
            <span className="badge border border-slate-600/30 bg-slate-700/60 text-slate-400">{entries.length} {"\u4e2a\u573a\u666f"}</span>
          </div>
        </div>
        <div className="p-5">
          <div className="list-scroll-lg space-y-3 pr-1">
            {entries.map(([name, count], index) => {
              const pct = useLog
                ? (Math.log10(Math.max(count, 1)) / Math.log10(Math.max(maxVal, 1))) * 100
                : (count / maxVal) * 100
              return (
                <div key={name} className="group rounded-2xl border border-slate-800/50 bg-slate-900/20 px-3 py-3 transition-colors hover:border-slate-700/60 hover:bg-slate-900/35">
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
          {useLog && <p className="mt-3 text-center text-xs text-slate-600">{"* \u6570\u636e\u91cf\u8de8\u5ea6\u8d85\u8fc7 100 \u500d\uff0c\u6761\u5f62\u957f\u5ea6\u6309\u5bf9\u6570\u6bd4\u4f8b\u663e\u793a"}</p>}
        </div>
      </div>
    </div>
  )
}

function ShapeTable({ shapes }: { shapes: Record<string, [number, number]> }) {
  const entries = Object.entries(shapes)
  if (entries.length === 0) return null

  return (
    <div className="card overflow-hidden">
      <div className="card-header">
        <TrendingUp size={15} className="text-slate-400" />
        <span className="text-base font-semibold text-slate-200">{"\u65f6\u5e8f\u5f62\u72b6"}</span>
      </div>
      <div className="list-table-scroll">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/40 bg-slate-800/30">
              <th className="px-5 py-3 text-left label">Simulator</th>
              <th className="px-5 py-3 text-right label">{"\u901a\u9053\u6570"}</th>
              <th className="px-5 py-3 text-right label">{"\u65f6\u95f4\u70b9"}</th>
              <th className="px-5 py-3 text-right label">{"\u5f62\u72b6"}</th>
              <th className="px-5 py-3 text-right label">{"\u603b\u5143\u7d20"}</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sim, [ch, ts]], i) => (
              <tr key={sim} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', i % 2 === 0 ? '' : 'bg-slate-800/10')}>
                <td className="px-5 py-3">
                  <span className={cn('badge border', getSimulatorBadgeClass(sim))}>{SIMULATOR_LABELS[sim] ?? sim}</span>
                </td>
                <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{ch}</td>
                <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{ts}</td>
                <td className="px-5 py-3 text-right font-mono text-slate-500">({ch}, {ts})</td>
                <td className="px-5 py-3 text-right font-mono tabular-nums font-semibold text-sky-400">{(ch * ts).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
      <div className="page-content p-6 space-y-8">
        {loading && !stats && (
          <div className="flex h-48 items-center justify-center gap-2.5 text-slate-500">
            <RefreshCw size={16} className="animate-spin" />
            <span>{"\u52a0\u8f7d\u4e2d\u2026"}</span>
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

            <SectionBlock
              icon={<Globe size={17} />}
              title={"\u5185\u5bb9\u5206\u5e03"}
              description={"\u628a\u573a\u666f\u89c4\u6a21\u3001\u8bed\u8a00\u5206\u5e03\u3001\u5199\u4f5c\u98ce\u683c\u548c\u65f6\u95f4\u91c7\u6837\u6536\u5728\u4e00\u5c4f\u91cc\uff0c\u76f4\u63a5\u770b\u51fa\u6570\u636e\u7684\u5185\u5bb9\u6c14\u8d28\u3002"}
            >
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.22fr_0.78fr]">
                <ScenarioBar data={stats.by_scenario} />
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <PieCard title={"\u8bed\u8a00\u5206\u5e03"} icon={<Globe size={15} />} data={stats.by_language} />
                  <PieCard title={"\u5199\u4f5c\u98ce\u683c"} icon={<FileText size={15} />} data={stats.by_style} />
                  <PieCard title={"\u65f6\u95f4\u91c7\u6837"} icon={<TrendingUp size={15} />} data={stats.by_time_mode} className="md:col-span-2" />
                </div>
              </div>
            </SectionBlock>

            <SectionBlock
              icon={<FolderOpen size={17} />}
              title={"\u6587\u4ef6\u4e0e\u7ed3\u6784"}
              description={"\u5c06\u65f6\u5e8f\u7ed3\u6784\u548c JSONL \u6587\u4ef6\u653e\u5728\u540c\u4e00\u7ec4\uff0c\u67e5\u770b\u65f6\u53ea\u9700\u5173\u5fc3\u7ed3\u6784\u548c\u843d\u76d8\u72b6\u6001\u3002"}
            >
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[0.92fr_1.08fr]">
                <ShapeTable shapes={stats.timeseries_shapes} />
                <div className="card overflow-hidden">
                  <div className="card-header">
                    <Database size={15} className="text-slate-400" />
                    <span className="text-base font-semibold text-slate-200">JSONL {"\u6587\u4ef6"}</span>
                  </div>
                  <div className="list-table-scroll">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700/40 bg-slate-800/30">
                          <th className="px-5 py-3 text-left label">{"\u573a\u666f"}</th>
                          <th className="px-5 py-3 text-left label">Simulator</th>
                          <th className="px-5 py-3 text-right label">{"\u6837\u672c\u6570"}</th>
                          <th className="px-5 py-3 text-right label">{"\u6587\u4ef6\u5927\u5c0f"}</th>
                          <th className="px-5 py-3 text-right label">{"\u4fee\u6539\u65f6\u95f4"}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {datasets.map((d, i) => (
                          <tr key={d.name} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', i % 2 === 0 ? '' : 'bg-slate-800/10')}>
                            <td className="px-5 py-3 font-mono text-slate-200">{d.name}</td>
                            <td className="px-5 py-3">
                              <span className={cn('badge border', getSimulatorBadgeClass(d.simulator))}>{SIMULATOR_LABELS[d.simulator] ?? d.simulator}</span>
                            </td>
                            <td className="px-5 py-3 text-right font-mono tabular-nums text-sky-400">{d.sample_count.toLocaleString()}</td>
                            <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{formatBytes(d.file_size_bytes)}</td>
                            <td className="px-5 py-3 text-right text-slate-500">{new Date(d.mtime * 1000).toLocaleString('zh-CN')}</td>
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
          <SectionBlock
            icon={<GitBranch size={17} />}
            title={"\u8def\u7531\u8bad\u7ec3\u6570\u636e"}
            description={"\u628a Router \u8bad\u7ec3\u771f\u6b63\u9700\u8981\u7684\u4e1c\u897f\u6536\u5230\u4e00\u8d77\uff1a\u6e90\u6837\u672c\u89c4\u6a21\u3001\u6b63\u8d1f\u5206\u5e03\u548c\u6309\u573a\u666f\u7684\u843d\u76d8\u660e\u7ec6\u3002"}
          >
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <StatCard label={"\u6e90\u6837\u672c\u6570"} value={(routerStatus.source_count ?? 0).toLocaleString()} sub={"\u4e0e Router \u6e90\u6570\u636e\u4e00\u4e00\u5bf9\u5e94"} color="text-sky-400" dotColor="bg-sky-500" icon={<Database size={18} />} />
              <StatCard label={"\u6b63\u6837\u672c"} value={routerPositive.toLocaleString()} sub={`${routerPositiveRate}% / label = 1`} color="text-emerald-400" dotColor="bg-emerald-500" icon={<Activity size={18} />} />
              <StatCard label={"\u8d1f\u6837\u672c"} value={routerNegative.toLocaleString()} sub={`${routerNegativeRate}% / label = 0`} color="text-amber-400" dotColor="bg-amber-500" icon={<FileText size={18} />} />
            </div>

            <div className="card overflow-hidden">
              <div className="card-header">
                <GitBranch size={15} className="text-rose-400" />
                <span className="text-base font-semibold text-slate-200">{"\u6309\u573a\u666f\u843d\u76d8\u660e\u7ec6"}</span>
              </div>
              <div className="list-table-scroll-compact">
                <table className="w-full table-fixed text-sm">
                  <thead>
                    <tr className="border-b border-slate-700/40 bg-slate-800/30">
                      <th className="px-5 py-3 text-left label">{"\u573a\u666f"}</th>
                      <th className="px-5 py-3 text-left label">Simulator</th>
                      <th className="px-5 py-3 text-right label">{"Router \u6837\u672c\u6570"}</th>
                      <th className="px-5 py-3 text-right label">{"\u6e90\u6570\u636e"}</th>
                      <th className="px-5 py-3 text-right label">{"\u6587\u4ef6\u5927\u5c0f"}</th>
                      <th className="px-5 py-3 text-right label">{"\u4fee\u6539\u65f6\u95f4"}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {routerStatus.scenarios.map((sc, i) => (
                      <tr key={sc.scenario} className={cn('border-b border-slate-800/40 transition-colors hover:bg-slate-700/20', i % 2 === 0 ? '' : 'bg-slate-800/10')}>
                        <td className="px-5 py-3 font-mono text-slate-200">{sc.scenario}</td>
                        <td className="px-5 py-3">
                          <span className={cn('badge border', getSimulatorBadgeClass(sc.simulator))}>{SIMULATOR_LABELS[sc.simulator] ?? sc.simulator}</span>
                        </td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-rose-400">{(sc.router_count ?? 0).toLocaleString()}</td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{(routerStatus.source_by_scenario[sc.scenario] ?? 0).toLocaleString()}</td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums text-slate-400">{formatBytes(sc.file_size_bytes ?? 0)}</td>
                        <td className="px-5 py-3 text-right text-slate-500">{sc.mtime ? new Date(sc.mtime * 1000).toLocaleString('zh-CN') : '-'}</td>
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
            title={"\u6682\u65e0\u6570\u636e"}
            description={"\u8bf7\u5148\u5728\u6570\u636e\u5408\u6210\u6d41\u7a0b\u4e2d\u4ea7\u51fa\u6837\u672c\u6570\u636e\uff0c\u6570\u636e\u603b\u89c8\u624d\u4f1a\u663e\u793a\u7edf\u8ba1\u7ed3\u679c\u3002"}
          />
        )}
      </div>
    </div>
  )
}
