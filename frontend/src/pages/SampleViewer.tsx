import { useState, useMemo } from 'react'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { DatasetInfo, SamplesResponse } from '../lib/types'
import SampleCard from '../components/sample/SampleCard'
import EmptyState from '../components/ui/EmptyState'
import { ChevronLeft, ChevronRight, Filter, Database, RefreshCw } from 'lucide-react'
import { cn, SIMULATOR_BADGE, SIMULATOR_LABELS } from '../lib/utils'

const PAGE_SIZE = 10

export default function SampleViewer() {
  const [scenario, setScenario] = useState<string>('')
  const [page, setPage] = useState(0)
  const [language, setLanguage] = useState('')
  const [style, setStyle] = useState('')

  const { data: datasets, isLoading: dLoading } =
    useSWR<DatasetInfo[]>('datasets', () => api.getDatasets())

  const grouped = useMemo(() => {
    const g: Record<string, DatasetInfo[]> = {}
    for (const d of datasets ?? []) {
      const sim = d.simulator || 'unknown'
      if (!g[sim]) g[sim] = []
      g[sim].push(d)
    }
    return g
  }, [datasets])

  const selectedScenario = scenario || datasets?.[0]?.name || ''

  const { data: samplesData, isLoading: sLoading, mutate } = useSWR<SamplesResponse>(
    selectedScenario ? ['samples', selectedScenario, page, language, style] : null,
    () => api.getSamples(selectedScenario, page, PAGE_SIZE, language || undefined, style || undefined),
    { revalidateOnFocus: false },
  )

  const totalPages = samplesData ? Math.ceil(samplesData.total / PAGE_SIZE) : 0

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* ── 左侧场景列表 ── */}
      <div className="w-60 flex-shrink-0 border-r border-slate-700/40 overflow-y-auto bg-slate-900/60 flex flex-col">
        <div className="px-5 py-4 border-b border-slate-700/40 flex-shrink-0">
          <div className="label">数据集</div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {dLoading && (
            <div className="flex items-center gap-2 px-4 py-4 text-slate-500 text-sm">
              <RefreshCw size={13} className="animate-spin" /> 加载中…
            </div>
          )}
          {Object.entries(grouped).map(([sim, items]) => {
            const badge = SIMULATOR_BADGE[sim]
            return (
              <div key={sim}>
                <div className={cn('flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold border-b border-slate-800/60', badge?.text ?? 'text-slate-500')}>
                  <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', badge?.dot ?? 'bg-slate-500')} />
                  {SIMULATOR_LABELS[sim] ?? sim}
                </div>
                {items.map(d => (
                  <button
                    key={d.name}
                    onClick={() => { setScenario(d.name); setPage(0) }}
                    className={cn(
                      'w-full text-left px-5 py-2.5 text-sm transition-all duration-150 border-b border-slate-800/40',
                      selectedScenario === d.name
                        ? 'bg-sky-500/10 text-sky-300 border-l-2 border-l-sky-500'
                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border-l-2 border-l-transparent',
                    )}
                  >
                    <div className="font-medium truncate text-xs">{d.name}</div>
                    <div className="text-slate-600 mt-0.5 text-xs tabular-nums">
                      {d.sample_count.toLocaleString()} 条
                    </div>
                  </button>
                ))}
              </div>
            )
          })}
          {!dLoading && (!datasets || datasets.length === 0) && (
            <div className="px-4 py-8 text-slate-600 text-sm flex flex-col items-center gap-2">
              <Database size={22} className="opacity-30" />
              <span>暂无数据集</span>
            </div>
          )}
        </div>
      </div>

      {/* ── 右侧主区域 ── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* 工具栏 */}
        <div className="flex items-center gap-2.5 px-4 py-2.5 border-b border-slate-700/40 bg-slate-900/30 flex-shrink-0">
          <Filter size={13} className="text-slate-500 flex-shrink-0" />
          <select
            className="select text-sm py-1 w-28"
            value={language}
            onChange={(e) => { setLanguage(e.target.value); setPage(0) }}
          >
            <option value="">全部语言</option>
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
          <select
            className="select text-sm py-1 w-28"
            value={style}
            onChange={(e) => { setStyle(e.target.value); setPage(0) }}
          >
            <option value="">全部风格</option>
            <option value="technical">专业技术</option>
            <option value="popular">科普</option>
            <option value="concise">简洁</option>
          </select>
          <div className="flex-1" />
          {samplesData && (
            <span className="badge bg-slate-700/60 text-slate-400 border border-slate-600/30 tabular-nums">
              {samplesData.total} 条 · 第 {page + 1}/{Math.max(totalPages, 1)} 页
            </span>
          )}
          <button className="btn-ghost py-1" onClick={() => mutate()}>
            <RefreshCw size={12} className={sLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* 样本列表 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {sLoading && (
            <div className="flex items-center justify-center gap-2 h-32 text-slate-500">
              <RefreshCw size={16} className="animate-spin" />
              <span className="text-sm">加载中…</span>
            </div>
          )}
          {!sLoading && samplesData?.items.map((sample, i) => (
            <SampleCard key={`${selectedScenario}-${page}-${i}`} sample={sample} index={page * PAGE_SIZE + i} />
          ))}
          {!sLoading && (!samplesData || samplesData.items.length === 0) && (
            <EmptyState
              icon={Database}
              title={selectedScenario ? '该场景暂无样本' : '请在左侧选择场景'}
              size="sm"
            />
          )}
        </div>

        {/* 分页 */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-1.5 px-4 py-3 border-t border-slate-700/40 flex-shrink-0 bg-slate-900/20">
            <button
              className="btn-ghost py-1 text-sm"
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
            >
              <ChevronLeft size={14} /> 上一页
            </button>
            <div className="flex gap-1">
              {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                const p = totalPages <= 7 ? i : (
                  page < 4 ? i :
                  page > totalPages - 4 ? totalPages - 7 + i :
                  page - 3 + i
                )
                return (
                  <button
                    key={p}
                    onClick={() => setPage(p)}
                    className={cn(
                      'w-8 h-8 rounded-lg text-sm font-mono tabular-nums transition-all duration-150',
                      p === page
                        ? 'bg-sky-600 text-white shadow-sm shadow-sky-900/30'
                        : 'text-slate-400 hover:bg-slate-700/60 hover:text-slate-200',
                    )}
                  >
                    {p + 1}
                  </button>
                )
              })}
            </div>
            <button
              className="btn-ghost py-1 text-sm"
              disabled={page >= totalPages - 1}
              onClick={() => setPage(p => p + 1)}
            >
              下一页 <ChevronRight size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
