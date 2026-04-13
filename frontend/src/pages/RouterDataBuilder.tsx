import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeed } from '../lib/seedContext'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { RouterStatus, RouterScenarioInfo } from '../lib/types'
import {
  GitBranch, RefreshCw, Settings, Layers,
  AlertCircle, Check, Database, FolderOpen, Trash2, ChevronDown, ChevronUp,
} from 'lucide-react'
import { cn, formatBytes, SIMULATOR_BADGE, SIMULATOR_LABELS } from '../lib/utils'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

// ── 场景按钮（对齐 ScenarioButton 风格）────────────────────────────

function RouterScenarioButton({
  item, active, onClick,
}: {
  item: RouterScenarioInfo
  active: boolean
  onClick: () => void
}) {
  const c = SIMULATOR_BADGE[item.simulator]
  // router_count 合法范围：(0, source_count * 20]，超出视为脏数据
  const rc = item.router_count ?? 0
  const hasRouter = rc > 0 && rc <= item.source_count * 20
  // 进度背景：以 source_count（1:1 时的正样本数）为基准，
  // 实际生成条数 / source_count 即可感知"至少覆盖了多少源样本"
  const pct = item.source_count > 0 && hasRouter
    ? Math.min(100, (rc / item.source_count) * 100)
    : 0

  return (
    <button
      onClick={onClick}
      className={cn(
        'relative text-left px-3 py-3 rounded-xl border transition-all duration-150 overflow-hidden w-full',
        active
          ? 'bg-rose-500/15 border-rose-500/40 shadow-sm shadow-rose-900/20'
          : `${c?.bg ?? 'bg-slate-800/50'} ${c?.border ?? 'border-slate-700/50'} hover:border-opacity-70 cursor-pointer`,
      )}
    >
      {/* 已生成进度底色 */}
      {hasRouter && pct > 0 && !active && (
        <div className="absolute inset-y-0 left-0 bg-rose-500/8 pointer-events-none"
          style={{ width: `${pct}%` }} />
      )}
      <div className="relative">
        <div className="flex items-center justify-between gap-1 mb-1">
          <span className={cn('font-semibold text-sm truncate',
            active ? 'text-rose-200' : 'text-slate-100')}>
            {item.scenario}
          </span>
          {active && <Check size={14} className="text-rose-400 flex-shrink-0" />}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className={cn('text-xs flex items-center gap-1',
            active ? 'text-rose-400/70' : (c?.text ?? 'text-slate-400'))}>
            <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', c?.dot ?? 'bg-slate-500')} />
            {SIMULATOR_LABELS[item.simulator] ?? item.simulator}
          </span>
          <span className="text-xs text-slate-600 tabular-nums flex items-center gap-0.5">
            <Database size={10} />{item.source_count.toLocaleString()}
          </span>
          {hasRouter && (
            <span className="text-xs text-rose-500/70 tabular-nums">
              ✓{(item.router_count ?? 0).toLocaleString()}
            </span>
          )}
        </div>
      </div>
    </button>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function RouterDataBuilder() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('router')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 520,
    minWidth: 320,
    maxWidth: 640,
    storageKey: 'piern_router_sidebar_width',
  })

  const { data: status, isLoading, mutate: refreshStatus } =
    useSWR<RouterStatus>('router-status', () => api.getRouterStatus(), { refreshInterval: 10000 })

  // 参数
  const { seed } = useSeed()
  const [negRatio, setNegRatio] = useState(1)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 文件管理
  const [filesOpen, setFilesOpen] = useState(false)
  const [deletingScenario, setDeletingScenario] = useState<string | null>(null)
  const [clearingAll, setClearingAll] = useState(false)

  // 场景多选
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const toggle = (name: string) => setSelected(prev => {
    const next = new Set(prev)
    if (next.has(name)) next.delete(name); else next.add(name)
    return next
  })

  // 按 simulator 分组
  const grouped = useMemo(() => {
    const g: Record<string, RouterScenarioInfo[]> = {}
    for (const s of status?.scenarios ?? []) {
      if (!g[s.simulator]) g[s.simulator] = []
      g[s.simulator].push(s)
    }
    return g
  }, [status?.scenarios])

  const allScenarios = status?.scenarios ?? []
  const hasScenarios = allScenarios.length > 0
  const hasRouterData = status && status.total > 0

  useEffect(() => {
    if (monitor.status === 'done' || monitor.status === 'error') {
      refreshStatus()
    }
  }, [monitor.status, refreshStatus])

  async function handleBuild() {
    if (selected.size === 0) { setError('请至少选择一个场景'); return }
    setError(null)
    setLaunching(true)
    try {
      const res = await api.buildRouterData(seed, Array.from(selected), negRatio)
      monitor.start(res.job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLaunching(false)
    }
  }

  async function handleDeleteScenario(scenario: string) {
    setDeletingScenario(scenario)
    try {
      await api.deleteRouterScenario(scenario)
      refreshStatus()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '删除失败')
    } finally {
      setDeletingScenario(null)
    }
  }

  async function handleClearAll() {
    if (!confirm('确认清空所有路由数据文件？此操作不可撤销。')) return
    setClearingAll(true)
    try {
      await api.deleteAllRouterData()
      refreshStatus()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '清空失败')
    } finally {
      setClearingAll(false)
    }
  }

  const canLaunch = !monitor.status || monitor.status === 'idle' || monitor.status === 'done'
    || monitor.status === 'error' || monitor.status === 'terminated'

  const posCount = status?.label_counts['1'] ?? 0
  const trainTotal = status?.splits.train.count ?? 0

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* ── 左栏 ── */}
      <div
        className="flex flex-col overflow-hidden border-r border-slate-700/40 flex-shrink-0"
        style={{ width: sidebarWidth }}
      >
        {/* 页头（固定）*/}
        <div className="flex-shrink-0 px-4 pt-4 pb-3 border-b border-slate-700/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/30 flex items-center justify-center flex-shrink-0">
                <GitBranch size={14} className="text-rose-400" />
              </div>
              <h1 className="text-lg font-bold text-white">路由数据生成</h1>
              <span className="badge bg-rose-500/15 text-rose-300 border border-rose-500/20 text-xs">Stage 4</span>
            </div>
            {hasRouterData && (
              <div className="text-xs text-slate-500">
                {status!.total.toLocaleString()} 条
              </div>
            )}
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">从 Stage 3 样本构建 Token 路由二分类训练数据</p>
        </div>

        {/* 场景选择（flex-1，独立滚动）*/}
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {/* 工具栏（固定）*/}
          <div className="flex-shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700/20">
            <div className="flex items-center gap-2">
              <Layers size={12} className="text-slate-400" />
              <span className="font-medium text-slate-300 text-sm">选择场景</span>
              {selected.size > 0 && (
                <span className="badge bg-rose-500/15 text-rose-300 border border-rose-500/20 text-xs py-0.5">
                  {selected.size} 已选
                </span>
              )}
            </div>
            <div className="flex items-center gap-0.5">
              {(['全选', '清空'] as const).map((label, i) => (
                <button key={label} className="btn-ghost py-0.5 px-2 text-sm"
                  onClick={[
                    () => setSelected(new Set(allScenarios.map(s => s.scenario))),
                    () => setSelected(new Set()),
                  ][i]}>
                  {label}
                </button>
              ))}
              <button className="btn-ghost py-0.5 px-1.5" onClick={() => refreshStatus()}>
                <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>

          {/* 场景列表（可滚动）*/}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {isLoading && (
              <div className="flex items-center gap-2 text-slate-500 text-xs py-2">
                <RefreshCw size={11} className="animate-spin text-rose-500" /> 加载中…
              </div>
            )}
            {!isLoading && !hasScenarios && (
              <div className="flex flex-col items-center gap-3 py-8 text-center">
                <Layers size={20} className="text-slate-700" />
                <div>
                  <p className="text-slate-500 text-xs font-medium">未找到 Stage 3 数据</p>
                  <p className="text-slate-600 text-sm mt-1">请先完成 Stage 3 样本填充</p>
                </div>
              </div>
            )}
            {Object.entries(grouped).map(([sim, items]) => (
              <div key={sim}>
                <div className="label mb-2 text-sm">{SIMULATOR_LABELS[sim] ?? sim}</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {items.map(item => (
                    <RouterScenarioButton
                      key={item.scenario}
                      item={item}
                      active={selected.has(item.scenario)}
                      onClick={() => toggle(item.scenario)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 底部：参数 + 按钮（固定）*/}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/20">
          <div className="px-4 py-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Settings size={12} className="text-slate-500" />
              <span className="text-sm font-medium text-slate-400">参数</span>
            </div>
            {/* 负样本倍数 */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="label text-xs">负样本倍数</span>
                <span className="text-xs text-slate-500 tabular-nums">
                  1:{negRatio}
                  <span className="text-slate-600 ml-1">
                    （每场景 1 万条 → {(10000 * (1 + negRatio)).toLocaleString()} 条）
                  </span>
                </span>
              </div>
              <input type="range" className="w-full accent-rose-500 h-1"
                min={1} max={10} value={negRatio}
                onChange={e => setNegRatio(parseInt(e.target.value))} />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>1:1</span><span>1:5</span><span>1:10</span>
              </div>
            </div>
          </div>

          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {canLaunch && (
            <div className="px-4 pb-4 space-y-1.5">
              <button
                className="btn-primary w-full py-2.5 text-sm justify-center shadow-lg"
                style={{ background: selected.size > 0 && !launching ? 'linear-gradient(135deg, #e11d48, #be123c)' : undefined }}
                onClick={handleBuild}
                disabled={launching || selected.size === 0}
              >
                {launching
                  ? <><RefreshCw size={14} className="animate-spin" /> 启动中…</>
                  : <><GitBranch size={14} /> 生成路由数据{selected.size > 0 ? `（${selected.size} 个场景）` : ''}</>
                }
              </button>
              {(monitor.status === 'done' || monitor.status === 'error' || monitor.status === 'terminated') && (
                <button className="btn-ghost w-full py-1.5 justify-center text-xs" onClick={monitor.reset}>
                  重新配置
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <ResizeHandle onMouseDown={onResizeStart} color="rose" />

      {/* ── 右栏 ── */}
      <div className="flex-1 flex flex-col overflow-y-auto p-4 space-y-3 min-w-0">

        {/* Job 监控 */}
        <JobMonitorPanel
          status={monitor.status}
          logs={monitor.logs}
          progress={monitor.progress}
          stats={monitor.stats}
          autoScroll={monitor.autoScroll}
          onAutoScrollChange={monitor.setAutoScroll}
          onStop={monitor.stop}
          stageLabel="路由数据生成"
          stageColor="text-rose-400"
          accentColor="rose"
          onDone={() => navigate('/router-viewer')}
          doneLabel="查看路由样本"
        />

        {monitor.status === 'idle' && !hasRouterData && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <GitBranch size={24} className="text-rose-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未生成路由数据</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景，点击「生成路由数据」</p>
            </div>
          </div>
        )}

        {/* 汇总 */}
        {hasRouterData && (
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: '总样本数', count: status!.splits.train.count, color: 'text-sky-400' },
              { label: '正样本', count: posCount, color: 'text-emerald-400' },
            ].map(({ label, count, color }) => (
              <div key={label} className="card px-4 py-3">
                <div className="text-xs text-slate-500 mb-1 font-medium uppercase tracking-wide">{label}</div>
                <div className={cn('text-2xl font-bold font-mono tabular-nums', color)}>
                  {count.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 文件管理 */}
        <div className="card overflow-hidden">
          <button
            onClick={() => setFilesOpen(o => !o)}
            className="w-full card-header justify-between hover:bg-slate-700/20 transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <FolderOpen size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-base">路由数据文件管理</span>
              {(status?.scenarios?.filter(s => (s.router_count ?? 0) > 0).length ?? 0) > 0 && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">
                  {status!.scenarios.filter(s => (s.router_count ?? 0) > 0).length} 个场景
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button className="btn-ghost py-0.5 px-1.5 text-xs"
                onClick={e => { e.stopPropagation(); refreshStatus() }}>
                <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
              </button>
              {filesOpen ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
            </div>
          </button>

          {filesOpen && (
            <div className="p-3 space-y-2">
              {(() => {
                const scenariosWithData = status?.scenarios?.filter(s => (s.router_count ?? 0) > 0) ?? []
                if (scenariosWithData.length === 0) {
                  return <p className="text-slate-500 text-xs text-center py-3">暂无路由数据文件</p>
                }
                return (
                  <>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-700/40">
                          <th className="px-3 py-1.5 text-left label">场景</th>
                          <th className="px-3 py-1.5 text-right label">样本数</th>
                          <th className="px-3 py-1.5 text-right label">大小</th>
                          <th className="px-3 py-1.5 text-right label">操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scenariosWithData.map(s => (
                          <tr key={s.scenario} className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors">
                            <td className="px-3 py-1.5 font-mono text-slate-300">{s.scenario}</td>
                            <td className="px-3 py-1.5 text-right tabular-nums text-rose-400">
                              {(s.router_count ?? 0).toLocaleString()}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">
                              {formatBytes(s.file_size_bytes ?? 0)}
                            </td>
                            <td className="px-3 py-1.5 text-right">
                              <button
                                className="btn-ghost py-0.5 px-1.5 text-red-400 hover:text-red-300"
                                onClick={() => handleDeleteScenario(s.scenario)}
                                disabled={deletingScenario === s.scenario}
                              >
                                {deletingScenario === s.scenario
                                  ? <RefreshCw size={10} className="animate-spin" />
                                  : <Trash2 size={10} />}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="flex justify-end pt-0.5">
                      <button
                        className="btn-ghost py-1 px-2.5 text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                        onClick={handleClearAll}
                        disabled={clearingAll}
                      >
                        {clearingAll ? <RefreshCw size={11} className="animate-spin" /> : <Trash2 size={11} />}
                        清空全部
                      </button>
                    </div>
                  </>
                )
              })()}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
