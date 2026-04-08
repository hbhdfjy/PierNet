import useSWR from 'swr'
import { api } from '../lib/api'
import type { DatasetStats, DatasetInfo } from '../lib/types'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import {
  formatBytes, LANGUAGE_LABELS, STYLE_LABELS,
  SIMULATOR_LABELS, SIMULATOR_BADGE, getSimulatorBadgeClass,
} from '../lib/utils'
import { RefreshCw, Database, Layers, TrendingUp, Globe, BarChart2, FileText } from 'lucide-react'
import EmptyState from '../components/ui/EmptyState'
import { cn } from '../lib/utils'

// ── 颜色 ──────────────────────────────────────────────────────────

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

// ── 汇总卡片 ──────────────────────────────────────────────────────

function StatCard({
  label, value, sub, color, dotColor, icon,
}: {
  label: string; value: string | number; sub: string
  color: string; dotColor: string; icon: React.ReactNode
}) {
  return (
    <div className="card-hover px-5 py-4 cursor-default group">
      <div className="flex items-start justify-between mb-3">
        <div className={cn('w-9 h-9 rounded-xl flex items-center justify-center', dotColor + '/15')}>
          <span className={color}>{icon}</span>
        </div>
        <div className={cn('w-2 h-2 rounded-full mt-1.5', dotColor)} />
      </div>
      <div className={cn('text-2xl font-bold font-mono tabular-nums mb-1', color)}>{value}</div>
      <div className="text-xs text-slate-500 font-medium">{label}</div>
      <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
    </div>
  )
}

// ── 饼图卡片 ──────────────────────────────────────────────────────

function PieCard({ title, icon, data }: {
  title: string; icon: React.ReactNode; data: Record<string, number>
}) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const total = entries.reduce((s, [, v]) => s + v, 0)
  const chartData = entries.map(([k, v]) => ({ name: k, value: v }))

  return (
    <div className="card overflow-hidden">
      <div className="card-header gap-2">
        <span className="text-slate-400">{icon}</span>
        <span className="font-semibold text-slate-200">{title}</span>
        <span className="ml-auto badge bg-slate-700/60 text-slate-400 border border-slate-600/30">
          {entries.length} 项
        </span>
      </div>
      <div className="p-5 flex gap-5 items-center">
        {/* 饼图 */}
        <div className="flex-shrink-0">
          <ResponsiveContainer width={120} height={120}>
            <PieChart>
              <Pie
                data={chartData} cx="50%" cy="50%"
                innerRadius={28} outerRadius={54}
                dataKey="value" paddingAngle={2} strokeWidth={0}
              >
                {chartData.map((_, i) => (
                  <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                ))}
              </Pie>
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(v: number) => [`${v.toLocaleString()}  (${((v / total) * 100).toFixed(1)}%)`, '']}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {/* 图例 */}
        <div className="flex-1 space-y-2.5 min-w-0">
          {entries.map(([k, v], i) => {
            const pct = ((v / total) * 100).toFixed(0)
            return (
              <div key={k}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: PALETTE[i % PALETTE.length] }} />
                  <span className="text-sm text-slate-300 truncate flex-1">
                    {LANGUAGE_LABELS[k] ?? STYLE_LABELS[k] ?? SIMULATOR_LABELS[k] ?? k}
                  </span>
                  <span className="text-sm font-mono tabular-nums text-slate-400 flex-shrink-0">
                    {v.toLocaleString()}
                  </span>
                </div>
                {/* 进度条 */}
                <div className="h-1 bg-slate-700/40 rounded-full overflow-hidden ml-4">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${pct}%`, background: PALETTE[i % PALETTE.length] }}
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

// ── 场景横向柱状图 ────────────────────────────────────────────────

function ScenarioBar({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 22)
  const chartData = entries.map(([k, v]) => ({
    name: k.length > 24 ? k.slice(0, 22) + '…' : k,
    count: v,
  }))
  const maxVal = Math.max(...entries.map(([, v]) => v))

  return (
    <div className="card overflow-hidden">
      <div className="card-header">
        <Layers size={15} className="text-slate-400" />
        <span className="font-semibold text-slate-200">按场景分布</span>
        <span className="ml-auto badge bg-slate-700/60 text-slate-400 border border-slate-600/30">
          {entries.length} 个场景
        </span>
      </div>
      <div className="p-5">
        <ResponsiveContainer width="100%" height={Math.max(240, entries.length * 32)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 48, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(51,65,85,0.4)" horizontal={false} />
            <XAxis
              type="number" domain={[0, maxVal * 1.1]}
              tick={{ fill: '#64748b', fontSize: 12 }}
              axisLine={false} tickLine={false}
              tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
            />
            <YAxis
              type="category" dataKey="name" width={180}
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              axisLine={false} tickLine={false}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v: number) => [v.toLocaleString(), '样本数']}
            />
            <Bar dataKey="count" radius={[0, 6, 6, 0]} opacity={0.9}
              fill="url(#barGrad)"
              label={{ position: 'right', fill: '#64748b', fontSize: 12 }}
            >
              <defs>
                <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#0ea5e9" stopOpacity={0.7} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.95} />
                </linearGradient>
              </defs>
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ── 时序形状表 ────────────────────────────────────────────────────

function ShapeTable({ shapes }: { shapes: Record<string, [number, number]> }) {
  const entries = Object.entries(shapes)
  if (entries.length === 0) return null

  return (
    <div className="card overflow-hidden">
      <div className="card-header">
        <TrendingUp size={15} className="text-slate-400" />
        <span className="font-semibold text-slate-200">时序形状（观测后）</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/40 bg-slate-800/30">
              <th className="px-5 py-3 text-left label">场景</th>
              <th className="px-5 py-3 text-right label">通道数</th>
              <th className="px-5 py-3 text-right label">时间点</th>
              <th className="px-5 py-3 text-right label">总元素</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sc, [ch, ts]], i) => (
              <tr key={sc} className={cn(
                'border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors',
                i % 2 === 0 ? '' : 'bg-slate-800/10',
              )}>
                <td className="px-5 py-3 font-mono text-slate-300 text-sm">{sc}</td>
                <td className="px-5 py-3 text-right tabular-nums text-slate-400">{ch}</td>
                <td className="px-5 py-3 text-right tabular-nums text-slate-400">{ts}</td>
                <td className="px-5 py-3 text-right tabular-nums font-semibold text-sky-400">
                  {(ch * ts).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function DatasetStats() {
  const { data: stats, isLoading: sLoading, mutate: refreshStats } =
    useSWR<DatasetStats>('stats', () => api.getStats(), { refreshInterval: 30000 })
  const { data: datasets, isLoading: dLoading, mutate: refreshDatasets } =
    useSWR<DatasetInfo[]>('datasets', () => api.getDatasets(), { refreshInterval: 30000 })

  const loading = sLoading || dLoading

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {/* 页头 */}
      <div className="page-header flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-sky-500/15 border border-sky-500/25 flex items-center justify-center flex-shrink-0">
            <BarChart2 size={16} className="text-sky-400" />
          </div>
          <div>
            <h1 className="font-semibold text-white leading-none">数据集统计</h1>
            <p className="text-xs text-slate-500 mt-0.5">Stage 3 生成样本的分布概览</p>
          </div>
        </div>
        <button
          className="btn-ghost"
          onClick={() => { refreshStats(); refreshDatasets() }}
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          刷新
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* 加载中 */}
        {loading && !stats && (
          <div className="flex items-center justify-center gap-2.5 h-48 text-slate-500">
            <RefreshCw size={16} className="animate-spin" />
            <span>加载中…</span>
          </div>
        )}

        {stats && (
          <>
            {/* 汇总卡片 */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                label="总样本数" value={stats.total_samples.toLocaleString()} sub="条训练样本"
                color="text-sky-400" dotColor="bg-sky-500" icon={<Database size={18} />}
              />
              <StatCard
                label="场景数" value={Object.keys(stats.by_scenario).length} sub="个仿真场景"
                color="text-emerald-400" dotColor="bg-emerald-500" icon={<Layers size={18} />}
              />
              <StatCard
                label="Simulator" value={Object.keys(stats.by_simulator).length} sub="种仿真器"
                color="text-amber-400" dotColor="bg-amber-500" icon={<BarChart2 size={18} />}
              />
              <StatCard
                label="时序形状" value={Object.keys(stats.timeseries_shapes).length} sub="种输出形状"
                color="text-purple-400" dotColor="bg-purple-500" icon={<TrendingUp size={18} />}
              />
            </div>

            {/* 分布饼图 */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              <PieCard title="语言分布" icon={<Globe size={15} />} data={stats.by_language} />
              <PieCard title="写作风格" icon={<FileText size={15} />} data={stats.by_style} />
              <PieCard title="时间采样" icon={<TrendingUp size={15} />} data={stats.by_time_mode} />
              {Object.keys(stats.by_simulator).length > 0 && (
                <PieCard title="Simulator" icon={<Layers size={15} />} data={stats.by_simulator} />
              )}
            </div>

            {/* 场景分布柱状图 */}
            {Object.keys(stats.by_scenario).length > 0 && (
              <ScenarioBar data={stats.by_scenario} />
            )}

            {/* 时序形状表 */}
            {Object.keys(stats.timeseries_shapes).length > 0 && (
              <ShapeTable shapes={stats.timeseries_shapes} />
            )}
          </>
        )}

        {/* JSONL 文件列表 */}
        {datasets && datasets.length > 0 && (
          <div className="card overflow-hidden">
            <div className="card-header">
              <Database size={15} className="text-slate-400" />
              <span className="font-semibold text-slate-200">JSONL 文件</span>
              <span className="ml-auto badge bg-slate-700/60 text-slate-400 border border-slate-600/30">
                {datasets.length} 个文件
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/40 bg-slate-800/30">
                    <th className="px-5 py-3 text-left label">场景</th>
                    <th className="px-5 py-3 text-left label">Simulator</th>
                    <th className="px-5 py-3 text-right label">样本数</th>
                    <th className="px-5 py-3 text-right label">文件大小</th>
                    <th className="px-5 py-3 text-right label">修改时间</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((d, i) => (
                    <tr key={d.name} className={cn(
                      'border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors',
                      i % 2 === 0 ? '' : 'bg-slate-800/10',
                    )}>
                      <td className="px-5 py-3 font-mono text-slate-200">{d.name}</td>
                      <td className="px-5 py-3">
                        <span className={cn('badge border', getSimulatorBadgeClass(d.simulator))}>
                          {SIMULATOR_LABELS[d.simulator] ?? d.simulator}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-right tabular-nums text-sky-400 font-semibold">
                        {d.sample_count.toLocaleString()}
                      </td>
                      <td className="px-5 py-3 text-right tabular-nums text-slate-400">
                        {formatBytes(d.file_size_bytes)}
                      </td>
                      <td className="px-5 py-3 text-right text-slate-500">
                        {new Date(d.mtime * 1000).toLocaleString('zh-CN')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!loading && (!stats || stats.total_samples === 0) && (
          <EmptyState
            icon={Database}
            title="暂无数据"
            description="请先在「注册数据集」页面注册 Simulator，然后运行 Stage 3 生成训练样本"
          />
        )}

      </div>
    </div>
  )
}
