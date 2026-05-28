import { useState, useMemo } from 'react'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { DatasetInfo, SamplesResponse } from '../../lib/types'
import SampleCard from '../components/sample/SampleCard'
import { EmptyState } from '../../shared/ui'
import { ChevronLeft, ChevronRight, Filter, Database, RefreshCw, FileText } from 'lucide-react'
import { cn, SIMULATOR_BADGE, SIMULATOR_LABELS } from '../../lib/utils'

const PAGE_SIZE = 10

function datasetKey(item: Pick<DatasetInfo, 'simulator' | 'scenario' | 'name'>): string {
  return `${item.simulator || 'unknown'}::${item.scenario || item.name}`
}

export default function SampleViewer() {
  const [datasetId, setDatasetId] = useState('')
  const [page, setPage] = useState(0)
  const [language, setLanguage] = useState('')
  const [style, setStyle] = useState('')

  const { data: datasets, isLoading: dLoading } = useSWR<DatasetInfo[]>('datasets', () => api.getDatasets())

  const grouped = useMemo(() => {
    const g: Record<string, DatasetInfo[]> = {}
    for (const d of datasets ?? []) {
      const sim = d.simulator || 'unknown'
      if (!g[sim]) g[sim] = []
      g[sim].push(d)
    }
    return g
  }, [datasets])

  const selectedDataset = useMemo(() => {
    const items = datasets ?? []
    return items.find(item => datasetKey(item) === datasetId) ?? items[0] ?? null
  }, [datasets, datasetId])
  const selectedDatasetKey = selectedDataset ? datasetKey(selectedDataset) : ''
  const selectedScenario = selectedDataset?.scenario ?? ''

  const {
    data: samplesData,
    isLoading: sLoading,
    mutate,
  } = useSWR<SamplesResponse>(
    selectedDataset ? ['samples', selectedDataset.simulator, selectedDataset.scenario, page, language, style] : null,
    () =>
      api.getSamples(
        selectedDataset!.scenario,
        page,
        PAGE_SIZE,
        language || undefined,
        style || undefined,
        selectedDataset!.simulator || undefined,
      ),
    { revalidateOnFocus: false },
  )

  const totalPages = samplesData ? Math.ceil(samplesData.total / PAGE_SIZE) : 0

  return (
    <div className="page-shell">
      {/* 页头 */}
      <div className="page-header flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center flex-shrink-0">
            <FileText size={13} className="text-emerald-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">样本浏览</h1>
            <p className="text-sm text-slate-500 mt-0.5">阶段 3 生成的训练样本</p>
          </div>
        </div>
        <button className="btn-ghost" onClick={() => mutate()}>
          <RefreshCw size={14} className={sLoading ? 'animate-spin' : ''} />
          刷新
        </button>
      </div>
      <div className="flex-1 flex overflow-hidden">
        {/* ── 左侧场景列表 ── */}
        <div className="page-rail w-52">
          <div className="px-4 py-3 border-b border-slate-700/40 flex-shrink-0">
            <div className="label">数据集</div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {dLoading && (
              <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
                <RefreshCw size={12} className="animate-spin" /> 加载中…
              </div>
            )}
            {Object.entries(grouped).map(([sim, items]) => {
              const badge = SIMULATOR_BADGE[sim]
              return (
                <div key={sim}>
                  <div
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider border-b border-slate-800/60 bg-slate-900/30',
                      badge?.text ?? 'text-slate-500',
                    )}
                  >
                    <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', badge?.dot ?? 'bg-slate-500')} />
                    {SIMULATOR_LABELS[sim] ?? sim}
                  </div>
                  {items.map(d => {
                    const itemKey = datasetKey(d)
                    return (
                      <button
                        key={itemKey}
                        onClick={() => {
                          setDatasetId(itemKey)
                          setPage(0)
                        }}
                        className={cn(
                          'w-full text-left px-3 py-2 text-sm transition-all duration-150 border-b border-slate-800/30',
                          selectedDatasetKey === itemKey
                            ? 'bg-emerald-500/8 text-emerald-300 border-l-2 border-l-emerald-500'
                            : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200 border-l-2 border-l-transparent',
                        )}
                      >
                        <div className="font-medium truncate">{d.name}</div>
                        <div className="text-slate-600 mt-0.5 tabular-nums">{d.sample_count.toLocaleString()} 条</div>
                      </button>
                    )
                  })}
                </div>
              )
            })}
            {!dLoading && (!datasets || datasets.length === 0) && (
              <div className="px-4 py-8 text-slate-600 text-xs flex flex-col items-center gap-2">
                <Database size={20} className="opacity-30" />
                <span>暂无数据集</span>
              </div>
            )}
          </div>
        </div>

        {/* ── 右侧主区域 ── */}
        <div className="page-content">
          {/* 工具栏 */}
          <div className="toolbar-strip flex items-center gap-2 px-3 py-2">
            <Filter size={12} className="text-slate-600 flex-shrink-0" />
            <select
              className="select text-xs py-1 px-2 w-24 h-7"
              value={language}
              onChange={e => {
                setLanguage(e.target.value)
                setPage(0)
              }}
            >
              <option value="">全部语言</option>
              <option value="en">English</option>
              <option value="zh">中文</option>
            </select>
            <select
              className="select text-xs py-1 px-2 w-24 h-7"
              value={style}
              onChange={e => {
                setStyle(e.target.value)
                setPage(0)
              }}
            >
              <option value="">全部风格</option>
              <option value="technical">专业技术</option>
              <option value="popular">科普</option>
              <option value="concise">简洁</option>
            </select>
            <div className="flex-1" />
            {samplesData && (
              <span className="text-xs text-slate-500 tabular-nums">
                共 {samplesData.total} 条 · 第 {page + 1}/{Math.max(totalPages, 1)} 页
              </span>
            )}
            <button className="btn-ghost py-1 px-1.5" onClick={() => mutate()}>
              <RefreshCw size={11} className={sLoading ? 'animate-spin' : ''} />
            </button>
          </div>

          {/* 样本列表 */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {sLoading && (
              <div className="flex items-center justify-center gap-2 h-32 text-slate-500">
                <RefreshCw size={14} className="animate-spin" />
                <span className="text-xs">加载中…</span>
              </div>
            )}
            {!sLoading &&
              samplesData?.items.map((sample, i) => (
                <SampleCard key={`${selectedDatasetKey}-${page}-${i}`} sample={sample} index={page * PAGE_SIZE + i} />
              ))}
            {!sLoading && (!samplesData || samplesData.items.length === 0) && (
              <EmptyState icon={Database} title={selectedScenario ? '该场景暂无样本' : '请在左侧选择场景'} size="sm" />
            )}
          </div>

          {/* 分页 */}
          {totalPages > 1 && (
            <div className="pagination-strip flex items-center justify-center gap-1.5 px-4 py-3 border-t border-slate-700/40 flex-shrink-0 bg-slate-900/20">
              <button className="btn-ghost py-1 text-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>
                <ChevronLeft size={14} /> 上一页
              </button>
              <div className="pagination-pages flex gap-1">
                {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                  const p =
                    totalPages <= 7 ? i : page < 4 ? i : page > totalPages - 4 ? totalPages - 7 + i : page - 3 + i
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
    </div>
  )
}
