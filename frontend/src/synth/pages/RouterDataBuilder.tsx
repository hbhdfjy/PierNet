import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeed } from '../../lib/seedContext'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { RouterStatus, RouterScenarioInfo } from '../../lib/types'
import { GitBranch, RefreshCw, Settings, Layers, AlertCircle, Check, Database, FolderOpen } from 'lucide-react'
import { cn, SIMULATOR_BADGE, SIMULATOR_LABELS } from '../../lib/utils'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { isRestartableJobStatus, isTerminalJobStatus, useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

// ── 场景按钮（对齐 ScenarioButton 风格）────────────────────────────

function RouterScenarioButton({
  item,
  active,
  onClick,
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
  const pct = item.source_count > 0 && hasRouter ? Math.min(100, (rc / item.source_count) * 100) : 0

  return (
    <button
      onClick={onClick}
      className={cn(
        'router-scenario-button scenario-button relative w-full overflow-hidden border text-left transition-all duration-150',
        active
          ? 'bg-rose-500/15 border-rose-500/40 shadow-sm shadow-rose-900/20'
          : `${c?.bg ?? 'bg-slate-800/50'} ${c?.border ?? 'border-slate-700/50'} hover:border-opacity-70 cursor-pointer`,
      )}
    >
      {/* 已生成进度底色 */}
      {hasRouter && pct > 0 && !active && (
        <div className="scenario-button__progress bg-rose-500/8" style={{ width: `${pct}%` }} />
      )}
      <div className="scenario-button__content">
        <div className="scenario-button__title-row">
          <span className={cn('scenario-button__title', active ? 'text-rose-200' : 'text-slate-100')}>
            {item.scenario}
          </span>
          {active && <Check size={14} className="text-rose-400 flex-shrink-0" />}
        </div>
        <div className="scenario-button__meta">
          <span
            className={cn('scenario-button__meta-item', active ? 'text-rose-400/70' : (c?.text ?? 'text-slate-400'))}
          >
            <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', c?.dot ?? 'bg-slate-500')} />
            {SIMULATOR_LABELS[item.simulator] ?? item.simulator}
          </span>
          <span className="scenario-button__meta-item tabular-nums text-slate-600">
            <Database size={10} />
            {item.source_count.toLocaleString()}
          </span>
          {hasRouter && (
            <span className="scenario-button__meta-item tabular-nums text-rose-500/70">
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
    defaultWidth: 720,
    minWidth: 360,
    maxWidth: 920,
    storageKey: 'piern_router_sidebar_width_v2',
  })

  const {
    data: status,
    isLoading,
    mutate: refreshStatus,
  } = useSWR<RouterStatus>('router-status', () => api.getRouterStatus(), { refreshInterval: 10000 })

  // 参数
  const { seed } = useSeed()
  const [negRatio, setNegRatio] = useState(1)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 文件管理

  // 场景多选
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const toggle = (name: string) =>
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
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
    if (isTerminalJobStatus(monitor.status)) {
      refreshStatus()
    }
  }, [monitor.status, refreshStatus])

  async function handleBuild() {
    if (selected.size === 0) {
      setError('请至少选择一个场景')
      return
    }
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

  const canLaunch = isRestartableJobStatus(monitor.status)

  return (
    <div className="workbench-shell">
      {/* ── 左栏 ── */}
      <div className="workbench-sidebar" style={{ width: sidebarWidth }}>
        {/* 页头（固定）*/}
        <div className="workbench-sidebar-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/30 flex items-center justify-center flex-shrink-0">
                <GitBranch size={14} className="text-rose-400" />
              </div>
              <h1 className="text-lg font-bold text-white">路由数据生成</h1>
              <span className="badge bg-rose-500/15 text-rose-300 border border-rose-500/20 text-xs">阶段 4</span>
            </div>
            {hasRouterData && <div className="text-xs text-slate-500">{status!.total.toLocaleString()} 条</div>}
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">从阶段 3 样本构建 Token 路由二分类训练数据</p>
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
                <button
                  key={label}
                  className="btn-ghost py-0.5 px-2 text-sm"
                  onClick={
                    [() => setSelected(new Set(allScenarios.map(s => s.scenario))), () => setSelected(new Set())][i]
                  }
                >
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
                  <p className="text-slate-500 text-xs font-medium">未找到阶段 3 数据</p>
                  <p className="text-slate-600 text-sm mt-1">请先完成阶段 3 样本填充</p>
                </div>
              </div>
            )}
            {Object.entries(grouped).map(([sim, items]) => (
              <div key={sim}>
                <div className="workbench-group-label">{SIMULATOR_LABELS[sim] ?? sim}</div>
                <div className="scenario-grid grid gap-1.5">
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
              <input
                type="range"
                className="w-full accent-rose-500 h-1"
                min={1}
                max={10}
                value={negRatio}
                onChange={e => setNegRatio(parseInt(e.target.value))}
              />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>1:1</span>
                <span>1:5</span>
                <span>1:10</span>
              </div>
            </div>
            <div className="rounded-xl border border-slate-700/40 bg-slate-900/35 px-3 py-2.5 text-sm text-slate-400">
              {'Chat Template \u56fa\u5b9a\u4e3a '}
              <span className="font-mono text-sky-300">qwen</span>
            </div>
            <div className="rounded-xl border border-slate-700/40 bg-slate-900/35 px-3 py-2.5 text-sm text-slate-400">
              {'Embedding Backbone \u56fa\u5b9a\u4e3a '}
              <span
                className="pretty-tooltip inline-block max-w-full align-bottom"
                data-tooltip="由 PIERN_QWEN_EMBEDDING_MODEL 配置，默认读取 ~/Qwen/Qwen2.5-0.5B-Instruct"
              >
                <span className="block truncate font-mono text-sky-300">PIERN_QWEN_EMBEDDING_MODEL</span>
              </span>
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
                style={{
                  background: selected.size > 0 && !launching ? 'linear-gradient(135deg, #e11d48, #be123c)' : undefined,
                }}
                onClick={handleBuild}
                disabled={launching || selected.size === 0}
              >
                {launching ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" /> 启动中…
                  </>
                ) : (
                  <>
                    <GitBranch size={14} /> 生成路由数据{selected.size > 0 ? `（${selected.size} 个场景）` : ''}
                  </>
                )}
              </button>
              {isTerminalJobStatus(monitor.status) && (
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
      <div className="workbench-main-scroll">
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

        {monitor.status === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <GitBranch size={24} className="text-rose-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未启动任务</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景，点击「生成路由数据」</p>
            </div>
          </div>
        )}

        {/* File management moved to /files */}
        <div className="card overflow-hidden">
          <div className="card-header justify-between py-3">
            <div className="flex items-center gap-2">
              <FolderOpen size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-base">Router files</span>
            </div>
            <button className="btn-ghost py-1.5 text-xs" onClick={() => navigate('/files')}>
              打开文件管理
            </button>
          </div>
          <div className="p-4">
            <div className="rounded-2xl border border-slate-700/35 bg-slate-900/30 p-4">
              <div className="font-semibold text-slate-100">Centralized file manager</div>
              <p className="mt-1 text-sm leading-6 text-slate-400">
                Router scenario files, train.jsonl, and clear operations now live in the unified file manager.
              </p>
              <button className="btn-ghost mt-3 text-xs text-rose-300" onClick={() => navigate('/files')}>
                打开统一文件管理
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
