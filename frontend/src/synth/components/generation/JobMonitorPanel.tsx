import { useRef, useEffect, useState } from 'react'
import { Virtuoso, type VirtuosoHandle } from 'react-virtuoso'
import type { LogLine, ScenarioProgress, LiveStats, JobStatus } from '../../../lib/types'
import { formatElapsed } from '../../../lib/utils'
import { cn } from '../../../lib/utils'
import {
  Square,
  ArrowDownToLine,
  CheckCircle,
  XCircle,
  Loader2,
  Zap,
  Clock,
  ChevronRight,
  Terminal,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

interface Props {
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  autoScroll: boolean
  onAutoScrollChange: (v: boolean) => void
  onStop: () => void
  onDone?: () => void
  doneLabel?: string
  doneIcon?: React.ReactNode
  jobId?: string | null
  jobIds?: string[]
  stageLabel?: string
  stageColor?: string
  accentColor?: string
}

// 日志行着色
const LOG_COLOR = (line: string): string => {
  if (line.includes('[ERROR]') || line.includes('✗') || line.includes('失败')) return 'text-red-400'
  if (line.includes('✓') || line.includes('已保存') || line.includes('已写入') || line.includes('完成'))
    return 'text-emerald-400'
  if (line.includes('[WARNING]') || line.includes('警告')) return 'text-amber-400'
  if (line.startsWith('  →') || line.includes('Step') || line.includes('[注册]')) return 'text-sky-400/80'
  if (line.includes('[跳过]')) return 'text-slate-600'
  return 'text-slate-400'
}

// accent 颜色映射
const ACCENT = {
  violet: { bar: 'from-violet-500 to-violet-400', dot: 'bg-violet-400', ring: '#8b5cf6', glow: 'shadow-glow-violet' },
  emerald: {
    bar: 'from-emerald-500 to-emerald-400',
    dot: 'bg-emerald-400',
    ring: '#10b981',
    glow: 'shadow-glow-emerald',
  },
  sky: { bar: 'from-sky-500 to-sky-400', dot: 'bg-sky-400', ring: '#38bdf8', glow: 'shadow-glow-sky' },
  amber: { bar: 'from-amber-500 to-amber-400', dot: 'bg-amber-400', ring: '#f59e0b', glow: 'shadow-glow-amber' },
  rose: { bar: 'from-rose-500 to-rose-400', dot: 'bg-rose-400', ring: '#f43f5e', glow: 'shadow-glow-rose' },
} as const
type AccentKey = keyof typeof ACCENT

// ── 场景进度卡片 ──────────────────────────────────────────────────

function ScenarioCard({
  scenario,
  done,
  total,
  isRunning,
  accent,
}: {
  scenario: string
  done: number
  total: number
  isRunning: boolean
  accent: string
}) {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0
  const isComplete = done >= total && total > 0
  const a = ACCENT[accent as AccentKey] ?? ACCENT.sky

  return (
    <div
      className={cn(
        'rounded-xl border p-3 transition-all duration-300',
        isComplete
          ? 'bg-emerald-500/5 border-emerald-500/20'
          : isRunning && pct > 0
            ? 'bg-slate-800/50 border-slate-600/40'
            : 'bg-slate-800/30 border-slate-700/30',
      )}
    >
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2 min-w-0">
          {isComplete ? (
            <CheckCircle size={12} className="text-emerald-400 flex-shrink-0" />
          ) : isRunning && pct > 0 ? (
            <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0 animate-pulse', a.dot)} />
          ) : (
            <span className="w-1.5 h-1.5 rounded-full bg-slate-700 flex-shrink-0" />
          )}
          <span className="font-mono text-xs text-slate-300 truncate">{scenario}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          {isComplete ? (
            <span className="text-xs text-emerald-400 font-medium">完成</span>
          ) : (
            <span className="text-xs text-slate-500 tabular-nums">
              <span className={cn('font-medium', pct > 0 ? 'text-slate-300' : 'text-slate-600')}>
                {done.toLocaleString()}
              </span>
              {total > 0 && <span className="text-slate-700"> / {total.toLocaleString()}</span>}
            </span>
          )}
          {total > 0 && (
            <span
              className={cn(
                'text-xs tabular-nums font-mono w-9 text-right',
                isComplete ? 'text-emerald-400' : 'text-slate-500',
              )}
            >
              {pct.toFixed(0)}%
            </span>
          )}
        </div>
      </div>
      {/* 进度条 */}
      <div className="h-1 bg-slate-700/40 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full bg-gradient-to-r transition-all duration-700 ease-out',
            isComplete ? 'from-emerald-500 to-emerald-400' : a.bar,
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── 总体进度环 ────────────────────────────────────────────────────

function OverallRing({
  totalDone,
  totalTarget,
  status,
  accent,
}: {
  totalDone: number
  totalTarget: number
  status: JobStatus
  accent: string
}) {
  const pct = totalTarget > 0 ? Math.min(100, (totalDone / totalTarget) * 100) : 0
  const r = 34
  const circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ
  const a = ACCENT[accent as AccentKey] ?? ACCENT.sky
  const isDone = status === 'done'
  const isError = status === 'error'

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-20 h-20">
        <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r={r} fill="none" stroke="rgba(51,65,85,0.5)" strokeWidth="5" />
          <circle
            cx="40"
            cy="40"
            r={r}
            fill="none"
            stroke={isDone ? '#10b981' : isError ? '#ef4444' : a.ring}
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(0.4,0,0.2,1)' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {isDone ? (
            <CheckCircle size={20} className="text-emerald-400" />
          ) : isError ? (
            <XCircle size={20} className="text-red-400" />
          ) : (
            <>
              <span className="text-base font-bold text-white tabular-nums leading-none">{pct.toFixed(0)}</span>
              <span className="text-xs text-slate-500 mt-0.5">%</span>
            </>
          )}
        </div>
      </div>
      <div className="text-center">
        <div className="text-xs font-medium text-slate-300 tabular-nums">
          {totalDone.toLocaleString()}
          {totalTarget > 0 && <span className="text-slate-600"> / {totalTarget.toLocaleString()}</span>}
        </div>
        <div className="text-xs text-slate-500 mt-0.5">
          {
            {
              running: '处理中',
              done: '已完成',
              error: '出错',
              terminated: '已终止',
              external_terminated: '外部终止',
              idle: '',
            }[status]
          }
        </div>
      </div>
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────

export default function JobMonitorPanel({
  status,
  logs,
  progress,
  stats,
  autoScroll,
  onAutoScrollChange,
  onStop,
  onDone,
  doneLabel = '继续',
  doneIcon,
  jobId,
  jobIds,
  stageLabel,
  stageColor = 'text-slate-400',
  accentColor = 'sky',
}: Props) {
  const displayIds = jobIds && jobIds.length > 0 ? jobIds : jobId ? [jobId] : []
  const virtuosoRef = useRef<VirtuosoHandle>(null)
  const [logsOpen, setLogsOpen] = useState(false)

  useEffect(() => {
    if (autoScroll && logs.length > 0) {
      virtuosoRef.current?.scrollToIndex({ index: logs.length - 1, behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const totalDone = Object.values(progress).reduce((s, p) => s + p.done, 0)
  const totalTarget = Object.values(progress).reduce((s, p) => s + p.total, 0)
  const isRunning = status === 'running'
  const isDone = status === 'done'
  const scenarioList = Object.entries(progress)

  if (status === 'idle') return null

  const a = ACCENT[accentColor as AccentKey] ?? ACCENT.sky

  return (
    <div className="space-y-2.5 animate-fade-in">
      {/* ── 主监控卡片 ── */}
      <div className="card overflow-hidden">
        {/* 顶部状态栏 */}
        <div
          className={cn(
            'flex items-center gap-3 px-4 py-2.5',
            isDone ? 'bg-emerald-500/5' : isRunning ? 'bg-slate-800/30' : 'bg-slate-800/20',
          )}
          style={{ borderBottom: '1px solid hsl(var(--border) / 0.4)' }}
        >
          {/* 状态图标 + 文字 */}
          <div
            className={cn(
              'flex items-center gap-2',
              {
                running: 'text-sky-400',
                done: 'text-emerald-400',
                error: 'text-red-400',
                terminated: 'text-amber-400',
                external_terminated: 'text-amber-400',
                idle: 'text-slate-400',
              }[status],
            )}
          >
            {isRunning && <Loader2 size={13} className="animate-spin" />}
            {isDone && <CheckCircle size={13} />}
            {(status === 'error' || status === 'terminated' || status === 'external_terminated') && (
              <XCircle size={13} />
            )}
            <span className="font-medium text-sm">
              {
                {
                  running: '运行中',
                  done: '已完成',
                  error: '出错',
                  terminated: '已终止',
                  external_terminated: '外部终止',
                  idle: '',
                }[status]
              }
            </span>
          </div>

          {stageLabel && (
            <>
              <div className="w-px h-3.5 bg-slate-700" />
              <span className={cn('text-xs', stageColor)}>{stageLabel}</span>
            </>
          )}

          {displayIds.length > 0 && (
            <div className="flex gap-1">
              {displayIds.map(id => (
                <span key={id} className="text-xs text-slate-600 font-mono">
                  #{id.slice(0, 8)}
                </span>
              ))}
            </div>
          )}

          {/* 实时指标 */}
          {isRunning && (
            <div className="flex items-center gap-3 text-xs ml-0.5">
              <div className="flex items-center gap-1 text-sky-400/80">
                <Zap size={11} />
                <span className="font-mono tabular-nums">{stats.samples_per_sec.toFixed(1)}</span>
                <span className="text-slate-600 text-xs">条/s</span>
              </div>
              <div className="flex items-center gap-1 text-slate-500">
                <Clock size={11} />
                <span className="font-mono tabular-nums">{formatElapsed(stats.elapsed_sec)}</span>
              </div>
            </div>
          )}

          <div className="flex-1" />

          {isRunning && (
            <button
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium
                         bg-red-500/10 text-red-400 border border-red-500/20
                         hover:bg-red-500/20 hover:border-red-500/30 transition-all duration-150"
              onClick={onStop}
            >
              <Square size={11} />
              终止
            </button>
          )}
          {isDone && onDone && (
            <button
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium
                         bg-emerald-500/10 text-emerald-400 border border-emerald-500/20
                         hover:bg-emerald-500/20 transition-all duration-150"
              onClick={onDone}
            >
              {doneLabel} {doneIcon ?? <ChevronRight size={12} />}
            </button>
          )}
        </div>

        {/* 进度内容 */}
        <div className="p-4">
          {scenarioList.length === 0 && isRunning && (
            <div className="flex flex-col items-center gap-3 py-5">
              <div className="flex gap-1">
                {[0, 1, 2, 3, 4].map(i => (
                  <div
                    key={i}
                    className={cn('w-1.5 h-1.5 rounded-full animate-bounce', a.dot)}
                    style={{ animationDelay: `${i * 0.12}s`, animationDuration: '0.8s' }}
                  />
                ))}
              </div>
              <span className="text-xs text-slate-500">等待任务初始化…</span>
            </div>
          )}

          {scenarioList.length > 0 &&
            (scenarioList.length <= 6 ? (
              <div className="flex gap-5 items-start">
                <div className="flex-1 grid grid-cols-1 gap-2">
                  {scenarioList.map(([sc, p]) => (
                    <ScenarioCard
                      key={sc}
                      scenario={sc}
                      done={p.done}
                      total={p.total}
                      isRunning={isRunning}
                      accent={accentColor}
                    />
                  ))}
                </div>
                {totalTarget > 0 && (
                  <div className="flex-shrink-0">
                    <OverallRing totalDone={totalDone} totalTarget={totalTarget} status={status} accent={accentColor} />
                  </div>
                )}
              </div>
            ) : (
              <div className="flex gap-5 items-start">
                <div className="flex-1 list-scroll-lg space-y-1.5">
                  {scenarioList.map(([sc, p]) => {
                    const pct = p.total > 0 ? Math.min(100, (p.done / p.total) * 100) : 0
                    const isComplete = p.done >= p.total && p.total > 0
                    return (
                      <div key={sc}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-mono text-slate-400 truncate max-w-[55%]">{sc}</span>
                          <span className="text-slate-500 tabular-nums">
                            {isComplete ? (
                              <span className="text-emerald-400">✓</span>
                            ) : (
                              <>
                                <span className="text-slate-300">{p.done.toLocaleString()}</span>
                                {p.total > 0 && <span className="text-slate-700">/{p.total.toLocaleString()}</span>}
                              </>
                            )}
                            <span className="ml-1.5">{pct.toFixed(0)}%</span>
                          </span>
                        </div>
                        <div className="h-1 bg-slate-700/40 rounded-full overflow-hidden">
                          <div
                            className={cn(
                              'h-full rounded-full bg-gradient-to-r transition-all duration-700',
                              isComplete ? 'from-emerald-500 to-emerald-400' : a.bar,
                            )}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
                {totalTarget > 0 && (
                  <div className="flex-shrink-0">
                    <OverallRing totalDone={totalDone} totalTarget={totalTarget} status={status} accent={accentColor} />
                  </div>
                )}
              </div>
            ))}
        </div>
      </div>

      {/* ── 完成横幅 ── */}
      {isDone && (
        <div
          className="flex items-center gap-3 px-4 py-2.5 rounded-xl
                        bg-emerald-500/8 border border-emerald-500/20 animate-fade-in"
        >
          <CheckCircle size={15} className="text-emerald-400 flex-shrink-0" />
          <span className="text-sm text-emerald-300 flex-1">
            完成！共处理 <span className="font-semibold tabular-nums">{totalDone.toLocaleString()}</span> 条。
          </span>
          {onDone && (
            <button
              className="flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300
                         font-medium transition-colors flex-shrink-0"
              onClick={onDone}
            >
              {doneLabel} {doneIcon ?? <ChevronRight size={12} />}
            </button>
          )}
        </div>
      )}

      {/* ── 日志（折叠）── */}
      {logs.length > 0 && (
        <div className="card overflow-hidden">
          <button
            onClick={() => setLogsOpen(o => !o)}
            className="w-full flex items-center gap-2 px-4 py-2.5 transition-colors text-left"
            style={{ color: 'hsl(var(--text-faint))' }}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'hsl(var(--surface2))')}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            <Terminal size={12} />
            <span className="text-xs">详细日志</span>
            <span className="text-xs text-slate-600 tabular-nums">{logs.length} 行</span>
            <div className="flex-1" />
            <button
              className={cn(
                'p-1 rounded transition-colors',
                autoScroll ? 'text-sky-400' : 'text-slate-600 hover:text-slate-400',
              )}
              onClick={e => {
                e.stopPropagation()
                onAutoScrollChange(!autoScroll)
              }}
              title="自动滚动"
            >
              <ArrowDownToLine size={11} />
            </button>
            {logsOpen ? (
              <ChevronUp size={12} className="text-slate-600" />
            ) : (
              <ChevronDown size={12} className="text-slate-600" />
            )}
          </button>
          {logsOpen && (
            <div style={{ height: '220px', borderTop: '1px solid hsl(var(--border) / 0.4)' }}>
              <Virtuoso
                ref={virtuosoRef}
                style={{ height: '100%' }}
                data={logs}
                followOutput={autoScroll ? 'smooth' : false}
                itemContent={(_, log) => (
                  <div className={cn('px-4 py-0.5 font-mono text-xs leading-5 select-text', LOG_COLOR(log.line))}>
                    {log.line}
                  </div>
                )}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
