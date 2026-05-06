import { useState, useMemo } from 'react'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { RouterStatus, RouterSample, RouterScenarioInfo } from '../../lib/types'
import { GitBranch, RefreshCw, Database, ChevronLeft, ChevronRight, Check } from 'lucide-react'
import { cn, SIMULATOR_BADGE, SIMULATOR_LABELS } from '../../lib/utils'
import EmptyState from '../components/ui/EmptyState'

const PAGE_SIZE = 10

// ── 样本卡片 ─────────────────────────────────────────────────────

function SampleCard({ sample, index }: { sample: RouterSample; index: number }) {
  const isPos = sample.label === 1

  return (
    <div className={cn(
      'card overflow-hidden border-l-2',
      isPos ? 'border-l-emerald-500' : 'border-l-slate-600',
    )}>
      <div className="accordion-card-header px-4 py-2.5 flex items-center gap-3 border-b border-slate-700/40">
        <span className={cn(
          'badge border text-xs font-bold',
          isPos
            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
            : 'bg-slate-700/40 text-slate-400 border-slate-600/30',
        )}>
          {isPos ? 'label=1  \u4e13\u5bb6' : 'label=0  LLM'}
        </span>
        <span className="text-xs text-slate-500 font-mono">{sample.metadata.language}</span>
        <span className="ml-auto text-xs text-slate-600 font-mono">#{index}</span>
      </div>
      <div className="px-4 py-3">
        <div className="text-xs text-slate-500 mb-1.5 font-medium">context</div>
        <div className="text-sm text-slate-300 leading-relaxed font-mono bg-slate-800/40 rounded-lg px-3 py-2.5 break-words [overflow-wrap:anywhere] whitespace-pre-wrap">
          {sample.context}
        </div>
      </div>
    </div>
  )
}
// ── 主页面 ────────────────────────────────────────────────────────

export default function RouterViewer() {
  const [scenario, setScenario] = useState('')
  const [labelFilter, setLabelFilter] = useState<-1 | 0 | 1>(-1)
  const [page, setPage] = useState(0)

  const { data: status, isLoading: sLoading, mutate: refresh } =
    useSWR<RouterStatus>('router-status', () => api.getRouterStatus())

  // 按 simulator 分组（只显示有合法 router 数据的场景）
  // router_count 合法范围：(0, source_count * 20]，超出视为脏数据
  const grouped = useMemo(() => {
    const g: Record<string, RouterScenarioInfo[]> = {}
    for (const s of status?.scenarios ?? []) {
      const rc = s.router_count ?? 0
      if (rc === 0) continue
      if (rc > s.source_count * 20) continue  // 异常大，跳过（脏数据）
      if (!g[s.simulator]) g[s.simulator] = []
      g[s.simulator].push(s)
    }
    return g
  }, [status?.scenarios])

  const hasRouterData = (status?.total ?? 0) > 0
  const activeScenario = scenario || Object.values(grouped).flat()[0]?.scenario || ''

  const { data: samplesResp, isLoading: samplesLoading } = useSWR(
    activeScenario ? ['router-samples', activeScenario, labelFilter, page] : null,
    () => api.getRouterSamples('train', page, PAGE_SIZE, labelFilter, activeScenario),
    { revalidateOnFocus: false },
  )

  const totalPages = samplesResp ? Math.ceil(samplesResp.total / PAGE_SIZE) : 0

  return (
    <div className="page-shell">

      {/* 页头 */}
      <div className="page-header flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-rose-500/15 border border-rose-500/25 flex items-center justify-center flex-shrink-0">
            <GitBranch size={13} className="text-rose-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">路由样本浏览</h1>
            <p className="text-sm text-slate-500 mt-0.5">阶段 4 Token 路由训练数据</p>
          </div>
        </div>
        <button className="btn-ghost" onClick={() => refresh()}>
          <RefreshCw size={14} className={sLoading ? 'animate-spin' : ''} />
          刷新
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">

        {/* ── 左侧场景列表 ── */}
        <div className="page-rail w-52">
          <div className="px-4 py-3 border-b border-slate-700/40 flex-shrink-0">
            <div className="label">场景</div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {sLoading && (
              <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
                <RefreshCw size={12} className="animate-spin" /> 加载中…
              </div>
            )}
            {!sLoading && Object.keys(grouped).length === 0 && (
              <div className="px-3 py-4 text-xs text-slate-500">尚未生成路由数据</div>
            )}
            {Object.entries(grouped).map(([sim, items]) => {
              const badge = SIMULATOR_BADGE[sim]
              return (
                <div key={sim}>
                  <div className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider border-b border-slate-800/60 bg-slate-900/30',
                    badge?.text ?? 'text-slate-500',
                  )}>
                    <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', badge?.dot ?? 'bg-slate-500')} />
                    {SIMULATOR_LABELS[sim] ?? sim}
                  </div>
                  {items.map(item => (
                    <button
                      key={item.scenario}
                      onClick={() => { setScenario(item.scenario); setPage(0) }}
                      className={cn(
                        'w-full text-left px-3 py-2 text-sm transition-all duration-150 border-b border-slate-800/30',
                        activeScenario === item.scenario
                          ? 'bg-rose-500/8 text-rose-300 font-medium'
                          : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200',
                      )}
                    >
                      <div className="truncate">{item.scenario}</div>
                      <div className="text-xs mt-0.5 font-mono text-slate-600">
                        {item.router_count != null
                          ? `${item.router_count.toLocaleString()} 条`
                          : `源 ${item.source_count.toLocaleString()} 条`}
                      </div>
                    </button>
                  ))}
                </div>
              )
            })}
          </div>
        </div>

        {/* ── 右侧内容 ── */}
        <div className="page-content">

          {!hasRouterData && !sLoading && (
            <div className="flex-1 flex items-center justify-center">
              <EmptyState
                icon={GitBranch}
                title="暂无路由数据"
                description="请先在「路由数据生成」页面生成训练数据"
              />
            </div>
          )}

          {hasRouterData && (
            <>
              {/* 工具栏 */}
              <div className="toolbar-strip flex items-center gap-3 px-4 py-2.5 flex-wrap">
                {/* Label 筛选 */}
                <div className="flex rounded-lg overflow-hidden border border-slate-700/50">
                  {([[-1, '全部'], [1, 'label=1 专家'], [0, 'label=0 LLM']] as const).map(([v, lbl]) => (
                    <button key={v}
                      className={cn('px-3 py-1.5 text-xs font-medium transition-colors',
                        labelFilter === v
                          ? 'bg-slate-700 text-white'
                          : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50')}
                      onClick={() => { setLabelFilter(v as -1 | 0 | 1); setPage(0) }}
                    >{lbl}</button>
                  ))}
                </div>
                <span className="ml-auto text-xs text-slate-500">
                  共 <span className="text-slate-300 font-mono">{samplesResp?.total.toLocaleString() ?? '—'}</span> 条
                </span>
              </div>

              {/* 样本列表 */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {samplesLoading && (
                  <div className="flex items-center justify-center gap-2 h-32 text-slate-500">
                    <RefreshCw size={14} className="animate-spin" /><span>加载中…</span>
                  </div>
                )}
                {!samplesLoading && samplesResp?.items.map((s, i) => (
                  <SampleCard key={i} sample={s} index={page * PAGE_SIZE + i} />
                ))}
                {!samplesLoading && samplesResp?.items.length === 0 && (
                  <div className="text-center text-slate-500 py-8 text-sm">暂无数据</div>
                )}
              </div>

              {/* 分页 */}
              {totalPages > 1 && (
                <div className="flex-shrink-0 px-4 py-3 border-t border-slate-700/40 flex items-center justify-between">
                  <span className="text-xs text-slate-500">第 {page + 1} / {totalPages} 页</span>
                  <div className="flex gap-2">
                    <button className="btn-ghost py-1.5 px-3 text-xs"
                      disabled={page === 0} onClick={() => setPage(p => p - 1)}>
                      <ChevronLeft size={13} />上一页
                    </button>
                    <button className="btn-ghost py-1.5 px-3 text-xs"
                      disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>
                      下一页<ChevronRight size={13} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
