import useSWR from 'swr'
import { api } from '../lib/api'
import type { DatasetStats, DatasetInfo } from '../lib/types'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import { formatBytes, LANGUAGE_LABELS, STYLE_LABELS, SIMULATOR_LABELS } from '../lib/utils'
import { RefreshCw, Database, Layers, TrendingUp, Globe, BarChart2 } from 'lucide-react'
import EmptyState from '../components/ui/EmptyState'

const COLORS = ['#38bdf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#22d3ee', '#fb923c']

const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#1e293b',
    border: '1px solid rgba(51,65,85,0.8)',
    borderRadius: 10,
    fontSize: 12,
    boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
  },
  labelStyle: { color: '#e2e8f0' },
  itemStyle: { color: '#94a3b8' },
}

function PieCard({ title, icon, data }: { title: string; icon: React.ReactNode; data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const total = entries.reduce((s, [, v]) => s + v, 0)
  const chartData = entries.map(([k, v]) => ({ name: k, value: v }))

  return (
    <div className="card">
      <div className="card-header">
        <span className="text-slate-400">{icon}</span>
        <span className="text-sm font-medium text-slate-200">{title}</span>
      </div>
      <div className="p-4 flex gap-4 items-center">
        <ResponsiveContainer width={130} height={130}>
          <PieChart>
            <Pie data={chartData} cx="50%" cy="50%" innerRadius={32} outerRadius={58}
              dataKey="value" paddingAngle={3} strokeWidth={0}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.9} />
              ))}
            </Pie>
            <Tooltip {...TOOLTIP_STYLE}
              formatter={(v: number) => [`${v}  (${((v / total) * 100).toFixed(1)}%)`, '']}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex-1 space-y-2 min-w-0">
          {entries.map(([k, v], i) => (
            <div key={k} className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: COLORS[i % COLORS.length] }} />
              <span className="text-sm text-slate-300 truncate flex-1">
                {LANGUAGE_LABELS[k] ?? STYLE_LABELS[k] ?? SIMULATOR_LABELS[k] ?? k}
              </span>
              <span className="text-sm text-slate-400 font-mono tabular-nums">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ScenarioBar({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 20)
  const chartData = entries.map(([k, v]) => ({
    name: k.length > 22 ? k.slice(0, 20) + '…' : k,
    count: v,
  }))

  return (
    <div className="card">
      <div className="card-header">
        <Layers size={14} className="text-slate-400" />
        <span className="text-sm font-medium text-slate-200">按场景分布</span>
        <span className="badge bg-slate-700 text-slate-400 ml-auto">{entries.length} 个场景</span>
      </div>
      <div className="p-4">
        <ResponsiveContainer width="100%" height={Math.max(220, entries.length * 30)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 4, right: 32, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(51,65,85,0.5)" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={170}
              tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Bar dataKey="count" fill="#38bdf8" radius={[0, 5, 5, 0]} opacity={0.85}
              label={{ position: 'right', fill: '#64748b', fontSize: 11 }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function ShapeTable({ shapes }: { shapes: Record<string, [number, number]> }) {
  const entries = Object.entries(shapes)
  if (entries.length === 0) return null
  return (
    <div className="card">
      <div className="card-header">
        <TrendingUp size={14} className="text-slate-400" />
        <span className="text-sm font-medium text-slate-200">时序形状（观测后）</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/40">
              <th className="px-4 py-2.5 text-left label">场景</th>
              <th className="px-4 py-2.5 text-right label">通道数</th>
              <th className="px-4 py-2.5 text-right label">时间点</th>
              <th className="px-4 py-2.5 text-right label">总元素</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sc, [ch, ts]]) => (
              <tr key={sc} className="border-b border-slate-800/60 hover:bg-slate-700/20 transition-colors">
                <td className="px-4 py-2.5 font-mono text-slate-300">{sc}</td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-400">{ch}</td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-400">{ts}</td>
                <td className="px-4 py-2.5 text-right tabular-nums text-sky-400 font-semibold">{(ch * ts).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

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
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-sky-500/15 border border-sky-500/25 flex items-center justify-center flex-shrink-0">
            <BarChart2 size={13} className="text-sky-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white leading-none">数据集统计</h1>
            <p className="text-xs text-slate-500 mt-0.5">Stage 3 生成样本的分布概览</p>
          </div>
        </div>
        <button
          className="btn-ghost py-1 px-2"
          onClick={() => { refreshStats(); refreshDatasets() }}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          刷新
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-5">

      {loading && !stats && (
        <div className="flex items-center justify-center gap-2 h-48 text-slate-500">
          <RefreshCw size={14} className="animate-spin" />
          <span className="text-xs">加载中…</span>
        </div>
      )}

      {stats && (
        <>
          {/* 汇总卡片 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
            {[
              { label: '总样本数', value: stats.total_samples.toLocaleString(), color: 'text-sky-400', sub: '条训练样本', dotColor: 'bg-sky-500' },
              { label: '场景数',   value: Object.keys(stats.by_scenario).length,  color: 'text-emerald-400', sub: '个仿真场景', dotColor: 'bg-emerald-500' },
              { label: 'Simulator', value: Object.keys(stats.by_simulator).length, color: 'text-amber-400', sub: '种仿真器', dotColor: 'bg-amber-500' },
              { label: '时序形状', value: Object.keys(stats.timeseries_shapes).length, color: 'text-purple-400', sub: '种输出形状', dotColor: 'bg-purple-500' },
            ].map(({ label, value, color, sub, dotColor }) => (
              <div key={label} className="card-hover px-4 py-3.5 cursor-default">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
                  <span className="text-xs text-slate-500">{label}</span>
                </div>
                <div className={`text-xl font-bold font-mono tabular-nums ${color}`}>{value}</div>
                <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
              </div>
            ))}
          </div>

          {/* 分布图 */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-5">
            <PieCard title="语言分布" icon={<Globe size={13} />} data={stats.by_language} />
            <PieCard title="写作风格" icon={<span className="text-xs">✍️</span>} data={stats.by_style} />
            <PieCard title="时间采样模式" icon={<TrendingUp size={13} />} data={stats.by_time_mode} />
            {Object.keys(stats.by_simulator).length > 0 && (
              <PieCard title="Simulator 分布" icon={<Layers size={13} />} data={stats.by_simulator} />
            )}
          </div>

          {Object.keys(stats.by_scenario).length > 0 && (
            <div className="mb-5"><ScenarioBar data={stats.by_scenario} /></div>
          )}
          {Object.keys(stats.timeseries_shapes).length > 0 && (
            <div className="mb-5"><ShapeTable shapes={stats.timeseries_shapes} /></div>
          )}
        </>
      )}

      {/* 文件列表 */}
      {datasets && datasets.length > 0 && (
        <div className="card">
          <div className="card-header">
            <Database size={14} className="text-slate-400" />
            <span className="text-sm font-medium text-slate-200">JSONL 文件</span>
            <span className="badge bg-slate-700 text-slate-400 ml-auto">{datasets.length} 个文件</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/40">
                <th className="px-4 py-2.5 text-left label">场景</th>
                <th className="px-4 py-2.5 text-left label">Simulator</th>
                <th className="px-4 py-2.5 text-right label">样本数</th>
                <th className="px-4 py-2.5 text-right label">文件大小</th>
                <th className="px-4 py-2.5 text-right label">修改时间</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.name} className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-slate-200">{d.name}</td>
                  <td className="px-4 py-2.5">
                    <span className="badge bg-slate-700/60 text-slate-300">{d.simulator}</span>
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-sky-400 font-semibold">{d.sample_count}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-slate-400">{formatBytes(d.file_size_bytes)}</td>
                  <td className="px-4 py-2.5 text-right text-slate-500">
                    {new Date(d.mtime * 1000).toLocaleString('zh-CN')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && (!stats || stats.total_samples === 0) && (
        <EmptyState
          icon={Database}
          title="暂无数据"
          description="请先在「注册数据集」页面注册 Simulator，然后「启动生成」运行 Stage 2"
        />
      )}
      </div>
    </div>
  )
}
