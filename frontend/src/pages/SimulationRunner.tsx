import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeed } from '../lib/seedContext'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { SimulationScenario, SimulationHistoryRecord } from '../lib/types'
import {
  Zap, RefreshCw, AlertCircle, ChevronDown, ChevronUp,
  History, CheckCircle2, XCircle, Database,
  SkipForward, Layers, Play, Square, BarChart2,
} from 'lucide-react'
import { cn, formatBytes, formatElapsed } from '../lib/utils'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

// ── 模拟器元数据 ──────────────────────────────────────────────────

const SIM_META: Record<string, {
  label: string; shortLabel: string
  color: string; bg: string; border: string; dot: string
}> = {
  modflow:      { label: 'MODFLOW',      shortLabel: 'MF',  color: 'text-blue-400',   bg: 'bg-blue-500/10',   border: 'border-blue-500/25',   dot: 'bg-blue-500'   },
  simpeg:       { label: 'SimPEG',       shortLabel: 'SP',  color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/25', dot: 'bg-purple-500' },
  power_flow:   { label: 'Power Flow',   shortLabel: 'PF',  color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/25', dot: 'bg-orange-500' },
  transient:    { label: 'Transient',    shortLabel: 'TR',  color: 'text-red-400',    bg: 'bg-red-500/10',    border: 'border-red-500/25',    dot: 'bg-red-500'    },
  gcam:         { label: 'PyPSA/GCAM',   shortLabel: 'GC',  color: 'text-teal-400',   bg: 'bg-teal-500/10',   border: 'border-teal-500/25',   dot: 'bg-teal-500'   },
}

const fallbackMeta = { label: '未知', shortLabel: '?', color: 'text-slate-400', bg: 'bg-slate-500/10', border: 'border-slate-500/25', dot: 'bg-slate-500' }

// ── 子组件 ────────────────────────────────────────────────────────

function SimBadge({ simulator, className }: { simulator: string; className?: string }) {
  const m = SIM_META[simulator] ?? fallbackMeta
  return (
    <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium border', m.bg, m.border, m.color, className)}>
      <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', m.dot)} />
      {m.label}
    </span>
  )
}

// 单场景卡片（多选模式）
function ScenarioRow({
  s, checked, onToggle, nSamples,
}: {
  s: SimulationScenario
  checked: boolean
  onToggle: () => void
  nSamples: number
}) {
  const m = SIM_META[s.simulator] ?? fallbackMeta
  const progress = s.sample_count > 0 && nSamples > 0
    ? Math.min(1, s.sample_count / nSamples)
    : 0
  const isComplete = s.sample_count >= nSamples && nSamples > 0

  return (
    <button
      onClick={onToggle}
      className={cn(
        'w-full text-left px-3 py-2 rounded-lg border transition-all duration-150 relative overflow-hidden',
        checked
          ? 'bg-amber-500/10 border-amber-500/30 text-white'
          : 'bg-slate-800/40 border-slate-700/30 text-slate-400 hover:border-slate-600/50 hover:text-slate-200',
      )}
    >
      {/* 进度背景条 */}
      {progress > 0 && (
        <div
          className={cn(
            'absolute inset-y-0 left-0 opacity-10 transition-all duration-500',
            isComplete ? 'bg-emerald-400' : 'bg-amber-400',
          )}
          style={{ width: `${progress * 100}%` }}
        />
      )}
      <div className="relative flex items-center gap-2">
        {/* 复选框 */}
        <div className={cn(
          'w-4 h-4 rounded flex-shrink-0 flex items-center justify-center border transition-all',
          checked
            ? 'bg-amber-500 border-amber-400'
            : 'bg-slate-700/60 border-slate-600/60',
        )}>
          {checked && <CheckCircle2 size={10} className="text-white" />}
        </div>
        {/* 场景名 */}
        <span className="flex-1 font-mono text-xs font-medium truncate">{s.scenario}</span>
        {/* 右侧信息 */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {s.sample_count > 0 && (
            <span className={cn(
              'text-xs tabular-nums font-medium',
              isComplete ? 'text-emerald-400' : 'text-slate-400',
            )}>
              {s.sample_count.toLocaleString()}
            </span>
          )}
          {s.output_shape && (
            <span className="text-xs text-slate-600 font-mono hidden sm:inline">
              ({s.output_shape.join('×')})
            </span>
          )}
        </div>
      </div>
    </button>
  )
}

// 历史记录行
function HistoryRow({ r }: { r: SimulationHistoryRecord }) {
  const m = SIM_META[r.simulator] ?? fallbackMeta
  const statusIcon = r.status === 'done'
    ? <CheckCircle2 size={12} className="text-emerald-400 flex-shrink-0" />
    : r.status === 'error'
    ? <XCircle size={12} className="text-red-400 flex-shrink-0" />
    : r.status === 'running'
    ? <RefreshCw size={12} className="text-sky-400 animate-spin flex-shrink-0" />
    : <Square size={12} className="text-amber-400 flex-shrink-0" />

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 hover:bg-slate-700/20 rounded-lg transition-colors text-xs">
      {statusIcon}
      <span className={cn('font-medium flex-shrink-0', m.color)}>{m.shortLabel}</span>
      <span className="font-mono text-slate-300 truncate flex-1">{r.scenario}</span>
      <span className="text-slate-500 tabular-nums flex-shrink-0">{r.n_samples.toLocaleString()}</span>
      {r.elapsed_sec != null && (
        <span className="text-slate-600 tabular-nums flex-shrink-0">{formatElapsed(r.elapsed_sec)}</span>
      )}
      {r.final_sample_count != null && (
        <span className="text-emerald-500/70 tabular-nums flex-shrink-0">→{r.final_sample_count.toLocaleString()}</span>
      )}
    </div>
  )
}

// 数据预览卡片（右栏顶部）
function DataOverviewCards({ scenarios }: { scenarios: SimulationScenario[] }) {
  const totalSamples = scenarios.reduce((s, x) => s + x.sample_count, 0)
  const withData = scenarios.filter(s => s.h5_path)
  const totalSize = scenarios.reduce((s, x) => s + x.file_size_bytes, 0)

  const bySimulator: Record<string, number> = {}
  for (const s of scenarios) {
    bySimulator[s.simulator] = (bySimulator[s.simulator] ?? 0) + s.sample_count
  }

  const cards = [
    { id: 'total',    label: '总样本数', value: totalSamples.toLocaleString(), sub: `${withData.length} 个文件`, icon: Database, color: 'text-amber-400' },
    { id: 'size',     label: '总数据量', value: formatBytes(totalSize),        sub: `${scenarios.length} 个场景`, icon: BarChart2, color: 'text-sky-400' },
    ...Object.entries(SIM_META).map(([sim, meta]) => ({
      id: sim,
      label: meta.label,
      value: (bySimulator[sim] ?? 0).toLocaleString(),
      sub: `${scenarios.filter(s => s.simulator === sim).length} 场景`,
      icon: Layers,
      color: meta.color,
    })),
  ]

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      {cards.map(({ id, label, value, sub, icon: Icon, color }) => (
        <div key={id} className="card px-3 py-2.5">
          <div className="flex items-center gap-1.5 mb-1">
            <Icon size={12} className={color} />
            <span className="label text-xs">{label}</span>
          </div>
          <div className={cn('text-base font-bold tabular-nums', color)}>{value}</div>
          <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
        </div>
      ))}
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function SimulationRunner() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('simulate')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 520,
    minWidth: 300,
    maxWidth: 580,
    storageKey: 'piern_simulate_sidebar_width',
  })

  const { data: scenarios, isLoading, mutate: refreshScenarios } =
    useSWR<SimulationScenario[]>('simulation-scenarios', () => api.getSimulationScenarios(), {
      refreshInterval: 20000,
      revalidateOnMount: true,
    })

  const { data: history, mutate: refreshHistory } =
    useSWR<SimulationHistoryRecord[]>('simulation-history', () => api.getSimulationHistory(30), {
      refreshInterval: 5000,
    })

  // ── 状态 ──
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nSamples, setNSamples] = useState(100)
  const [nSamplesInput, setNSamplesInput] = useState('100')
  const { seed } = useSeed()
  const [skipExisting, setSkipExisting] = useState(false)
  const [parallel, setParallel] = useState(false)
  const [maxWorkers, setMaxWorkers] = useState(4)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [tableOpen, setTableOpen] = useState(false)
  const [filterSim, setFilterSim] = useState<string | null>(null)

  // 按模拟器分组
  const grouped = useMemo(() => {
    const g: Record<string, SimulationScenario[]> = {}
    for (const s of scenarios ?? []) {
      if (!g[s.simulator]) g[s.simulator] = []
      g[s.simulator].push(s)
    }
    return g
  }, [scenarios])

  // 全选/清空
  const allNames = useMemo(() => (scenarios ?? []).map(s => s.scenario), [scenarios])
  const filteredNames = useMemo(() => {
    if (!filterSim) return allNames
    return (scenarios ?? []).filter(s => s.simulator === filterSim).map(s => s.scenario)
  }, [scenarios, filterSim, allNames])

  const toggle = (name: string) => setSelected(prev => {
    const next = new Set(prev)
    if (next.has(name)) next.delete(name); else next.add(name)
    return next
  })

  const selectAll = () => setSelected(new Set(filteredNames))
  const clearAll = () => setSelected(new Set())
  const selectIncomplete = () => {
    const incomplete = (scenarios ?? [])
      .filter(s => (!filterSim || s.simulator === filterSim) && s.sample_count < nSamples)
      .map(s => s.scenario)
    setSelected(new Set(incomplete))
  }

  const canLaunch = !monitor.status || ['idle', 'done', 'error', 'terminated'].includes(monitor.status)

  const handleLaunch = async () => {
    if (selected.size === 0) { setError('请至少选择一个场景'); return }
    setError(null)
    setLaunching(true)
    try {
      let result
      if (selected.size === 1) {
        const sc = scenarios?.find(s => s.scenario === [...selected][0])
        if (!sc) throw new Error('场景未找到')
        result = await api.startSimulation({
          simulator: sc.simulator,
          scenario: sc.scenario,
          n_samples: nSamples,
          seed,
          config_path: sc.config_path,
          skip_existing: skipExisting,
          parallel,
          max_workers: maxWorkers,
        })
      } else {
        result = await api.startBatchSimulation({
          scenarios: [...selected],
          n_samples: nSamples,
          seed,
          skip_existing: skipExisting,
          parallel,
          max_workers: maxWorkers,
        })
      }
      monitor.start(result.job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  // 每个场景完成后立即刷新场景列表（批量仿真时实时更新样本数）
  useEffect(() => {
    if (monitor.scenarioDoneCount > 0) {
      refreshScenarios(() => api.getSimulationScenarios(true), { revalidate: true })
      refreshHistory()
    }
  }, [monitor.scenarioDoneCount, refreshScenarios, refreshHistory])

  // 全部完成后也刷新一次
  useEffect(() => {
    if (monitor.status === 'done' || monitor.status === 'error') {
      refreshScenarios(() => api.getSimulationScenarios(true), { revalidate: true })
      refreshHistory()
    }
  }, [monitor.status, refreshScenarios, refreshHistory])

  const selectedScenarios = useMemo(
    () => (scenarios ?? []).filter(s => selected.has(s.scenario)),
    [scenarios, selected],
  )

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* ── 左栏 ── */}
      <div
        className="flex flex-col overflow-hidden border-r border-slate-700/40 flex-shrink-0"
        style={{ width: sidebarWidth }}
      >
        {/* 页头 */}
        <div className="flex-shrink-0 px-4 pt-4 pb-3 border-b border-slate-700/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
                <Zap size={14} className="text-amber-400" />
              </div>
              <h1 className="text-base font-bold text-white">仿真运行</h1>
              <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/20 text-xs">Stage 1</span>
            </div>
            <button
              className="btn-ghost py-1 px-2 text-xs"
              onClick={() => {
                refreshScenarios(() => api.getSimulationScenarios(true), { revalidate: true })
                refreshHistory()
              }}
              title="刷新"
            >
              <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
            </button>
          </div>
          <p className="text-slate-500 text-xs mt-1.5 ml-9">物理仿真 → HDF5 数据集</p>
        </div>

        {/* 场景筛选标签 */}
        <div className="flex-shrink-0 flex items-center gap-1 px-4 pt-2 pb-1.5 border-b border-slate-700/20 overflow-x-auto">
          <button
            onClick={() => setFilterSim(null)}
            className={cn(
              'flex-shrink-0 px-2 py-0.5 rounded-md text-xs font-medium transition-all border',
              !filterSim
                ? 'bg-slate-600/40 text-slate-200 border-slate-500/40'
                : 'text-slate-500 border-transparent hover:text-slate-300',
            )}
          >
            全部 {scenarios ? `(${scenarios.length})` : ''}
          </button>
          {Object.entries(SIM_META).map(([sim, meta]) => {
            const count = grouped[sim]?.length ?? 0
            if (count === 0) return null
            return (
              <button
                key={sim}
                onClick={() => setFilterSim(filterSim === sim ? null : sim)}
                className={cn(
                  'flex-shrink-0 flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium transition-all border',
                  filterSim === sim
                    ? cn(meta.bg, meta.border, meta.color)
                    : 'text-slate-500 border-transparent hover:text-slate-300',
                )}
              >
                <span className={cn('w-1.5 h-1.5 rounded-full', meta.dot)} />
                {meta.shortLabel} ({count})
              </button>
            )
          })}
        </div>

        {/* 场景列表工具栏 */}
        <div className="flex-shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700/20">
          <div className="flex items-center gap-1">
            {selected.size > 0 && (
              <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/20 text-xs py-0.5">
                {selected.size} 已选
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            <button className="btn-ghost py-0.5 px-2 text-xs" onClick={selectAll}>全选</button>
            <button className="btn-ghost py-0.5 px-2 text-xs" onClick={selectIncomplete} title="选择样本数不足目标的场景">
              <SkipForward size={10} className="mr-0.5" />未满
            </button>
            <button className="btn-ghost py-0.5 px-2 text-xs" onClick={clearAll}>清空</button>
          </div>
        </div>

        {/* 场景列表（可滚动）*/}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0">
          {isLoading && (
            <div className="flex items-center gap-2 text-slate-500 text-xs py-3 px-1">
              <RefreshCw size={11} className="animate-spin text-amber-500" /> 扫描配置目录…
            </div>
          )}
          {!isLoading && Object.keys(grouped).length === 0 && (
            <div className="flex flex-col items-center gap-2 py-10 text-center">
              <Zap size={18} className="text-slate-700" />
              <p className="text-slate-500 text-xs">未找到任何场景配置</p>
            </div>
          )}
          {Object.entries(grouped).map(([sim, list]) => {
            const meta = SIM_META[sim] ?? fallbackMeta
            const visible = filterSim ? filterSim === sim : true
            if (!visible) return null
            return (
              <div key={sim}>
                <div className="flex items-center gap-1.5 mb-1.5 px-1">
                  <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', meta.dot)} />
                  <span className={cn('text-xs font-semibold', meta.color)}>{meta.label}</span>
                  <span className="text-xs text-slate-600">{list.length} 个</span>
                </div>
                <div className="space-y-1">
                  {list.map(s => (
                    <ScenarioRow
                      key={s.scenario}
                      s={s}
                      checked={selected.has(s.scenario)}
                      onToggle={() => toggle(s.scenario)}
                      nSamples={nSamples}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* 底部参数 + 按钮 */}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/30">

          {/* 参数行 */}
          <div className="px-4 py-3 space-y-3">
            <div>
              <label className="label block mb-1">样本数 / 场景</label>
              <input
                type="number"
                className="input w-full text-xs py-1.5 px-3"
                value={nSamplesInput}
                min={1} max={100000}
                onChange={e => {
                  setNSamplesInput(e.target.value)
                  const n = parseInt(e.target.value, 10)
                  if (!isNaN(n) && n >= 1) setNSamples(n)
                }}
                onBlur={() => {
                  const n = parseInt(nSamplesInput, 10)
                  const v = isNaN(n) ? 1 : Math.max(1, Math.min(100000, n))
                  setNSamples(v); setNSamplesInput(String(v))
                }}
              />
            </div>

            {/* skip-existing 开关 */}
            <div
              className="flex items-center gap-2 cursor-pointer"
              onClick={() => setSkipExisting(v => !v)}
            >
              <div className={cn(
                'relative w-8 h-4 rounded-full transition-all duration-200 flex-shrink-0',
                skipExisting ? 'bg-amber-500' : 'bg-slate-700',
              )}>
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all duration-200"
                  style={{ left: skipExisting ? '18px' : '2px' }}
                />
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-xs text-slate-300 font-medium">断点续跑</span>
                <span className="text-xs text-slate-600 ml-1.5">
                  {skipExisting ? '已有样本时补齐到目标数' : '忽略已有样本重新生成'}
                </span>
              </div>
              <SkipForward size={12} className={skipExisting ? 'text-amber-400' : 'text-slate-600'} />
            </div>

            {/* 并行生成 */}
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  'relative w-8 h-4 rounded-full transition-all duration-200 flex-shrink-0 cursor-pointer',
                  parallel ? 'bg-sky-500' : 'bg-slate-700',
                )}
                onClick={() => setParallel(v => !v)}
              >
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all duration-200"
                  style={{ left: parallel ? '18px' : '2px' }}
                />
              </div>
              <span
                className="text-xs text-slate-300 font-medium flex-1 cursor-pointer"
                onClick={() => setParallel(v => !v)}
              >
                多核并行
              </span>
              {parallel && (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-slate-500">核数</span>
                  <input
                    type="number"
                    className="input w-14 text-xs py-0.5 px-2 text-center"
                    value={maxWorkers}
                    min={1} max={32}
                    onClick={e => e.stopPropagation()}
                    onChange={e => {
                      const n = parseInt(e.target.value, 10)
                      if (!isNaN(n) && n >= 1 && n <= 32) setMaxWorkers(n)
                    }}
                  />
                </div>
              )}
            </div>

            {/* 已选场景摘要 */}
            {selectedScenarios.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {selectedScenarios.slice(0, 4).map(s => {
                  const m = SIM_META[s.simulator] ?? fallbackMeta
                  return (
                    <span key={s.scenario} className={cn(
                      'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs border',
                      m.bg, m.border, m.color,
                    )}>
                      {s.scenario}
                    </span>
                  )
                })}
                {selectedScenarios.length > 4 && (
                  <span className="text-xs text-slate-500 self-center">+{selectedScenarios.length - 4}</span>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-xs">{error}</span>
            </div>
          )}

          {canLaunch && (
            <div className="px-4 pb-4 space-y-1.5">
              <button
                className={cn(
                  'btn w-full py-2.5 text-sm justify-center shadow-lg',
                  selected.size > 0 && !launching
                    ? 'bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-white'
                    : 'bg-slate-700/60 text-slate-500 cursor-not-allowed',
                )}
                onClick={handleLaunch}
                disabled={launching || selected.size === 0}
              >
                {launching
                  ? <><RefreshCw size={14} className="animate-spin" /> 启动中…</>
                  : selected.size > 1
                  ? <><Play size={14} /> 批量仿真（{selected.size} 个场景）</>
                  : <><Zap size={14} /> 开始仿真{selected.size === 1 ? `（${[...selected][0]}）` : ''}</>
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

      <ResizeHandle onMouseDown={onResizeStart} color="amber" />

      {/* ── 右栏 ── */}
      <div className="flex-1 flex flex-col overflow-y-auto p-4 space-y-3 min-w-0">

        {/* 数据总览卡片 */}
        {scenarios && scenarios.length > 0 && (
          <DataOverviewCards scenarios={scenarios} />
        )}

        {/* Job 监控面板 */}
        <JobMonitorPanel
          status={monitor.status}
          logs={monitor.logs}
          progress={monitor.progress}
          stats={monitor.stats}
          autoScroll={monitor.autoScroll}
          onAutoScrollChange={monitor.setAutoScroll}
          onStop={monitor.stop}
          jobId={monitor.jobId}
          jobIds={monitor.jobIds}
          stageLabel="物理仿真"
          stageColor="text-amber-400"
          accentColor="amber"
          onDone={() => navigate('/register')}
          doneLabel="去生成模板"
        />

        {monitor.status === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Zap size={24} className="text-amber-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">尚未启动仿真</p>
              <p className="text-slate-600 text-xs mt-1">在左侧选择场景 → 配置参数 → 点击启动</p>
            </div>
          </div>
        )}

        {/* HDF5 详细状态表格（折叠）*/}
        <div className="card overflow-hidden animate-fade-in">
          <button
            onClick={() => setTableOpen(o => !o)}
            className="w-full card-header justify-between hover:bg-slate-700/20 transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <Database size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-sm">HDF5 文件详情</span>
              {scenarios && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs py-0.5">
                  {scenarios.filter(s => s.h5_path).length} / {scenarios.length}
                </span>
              )}
            </div>
            {tableOpen ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
          </button>

          {tableOpen && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700/40">
                    <th className="px-3 py-2 text-left label">模拟器</th>
                    <th className="px-3 py-2 text-left label">场景</th>
                    <th className="px-3 py-2 text-right label">已有</th>
                    <th className="px-3 py-2 text-right label">目标</th>
                    <th className="px-3 py-2 text-left label">形状</th>
                    <th className="px-3 py-2 text-right label">大小</th>
                    <th className="px-3 py-2 text-left label">路径</th>
                  </tr>
                </thead>
                <tbody>
                  {(scenarios ?? []).map(s => {
                    const meta = SIM_META[s.simulator] ?? fallbackMeta
                    const isComplete = s.target_samples > 0 && s.sample_count >= s.target_samples
                    return (
                      <tr key={`${s.simulator}/${s.scenario}`} className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors">
                        <td className="px-3 py-1.5">
                          <SimBadge simulator={s.simulator} />
                        </td>
                        <td className="px-3 py-1.5 font-mono text-slate-300 max-w-[140px] truncate">{s.scenario}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums">
                          {s.sample_count > 0
                            ? <span className={isComplete ? 'text-emerald-400 font-medium' : 'text-amber-400'}>
                                {s.sample_count.toLocaleString()}
                              </span>
                            : <span className="text-slate-700">—</span>
                          }
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                          {s.target_samples > 0 ? s.target_samples.toLocaleString() : '—'}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-slate-600">
                          {s.output_shape ? `(${s.output_shape.join('×')})` : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">
                          {s.file_size_bytes > 0 ? formatBytes(s.file_size_bytes) : '—'}
                        </td>
                        <td className="px-3 py-1.5 max-w-[160px] truncate">
                          {s.h5_path
                            ? <span className="text-emerald-500/60 font-mono">{s.h5_path.split('/').pop()}</span>
                            : <span className="text-slate-700">未生成</span>
                          }
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 历史记录（折叠）*/}
        <div className="card overflow-hidden">
          <button
            onClick={() => setHistoryOpen(o => !o)}
            className="w-full card-header justify-between hover:bg-slate-700/20 transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <History size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-sm">历史运行记录</span>
              {history && history.length > 0 && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs py-0.5">
                  {history.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {historyOpen && history && history.length > 0 && (
                <button
                  className="btn-ghost py-0.5 px-2 text-xs text-red-400/70 hover:text-red-400"
                  onClick={async e => {
                    e.stopPropagation()
                    await api.clearSimulationHistory()
                    refreshHistory()
                  }}
                >
                  清空
                </button>
              )}
              {historyOpen ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
            </div>
          </button>

          {historyOpen && (
            <div className="py-1">
              {(!history || history.length === 0) ? (
                <p className="text-slate-600 text-xs text-center py-4">暂无历史记录（重启后清空）</p>
              ) : (
                <div className="space-y-0.5 px-2 pb-2">
                  {history.map((r, i) => <HistoryRow key={`${r.job_id}-${i}`} r={r} />)}
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
