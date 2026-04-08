import { useRef, useEffect, useState } from 'react'
import { Virtuoso, type VirtuosoHandle } from 'react-virtuoso'
import type { LogLine, ScenarioProgress, LiveStats, JobStatus } from '../../lib/types'
import { formatElapsed } from '../../lib/utils'
import { cn } from '../../lib/utils'
import {
  Square, ArrowDownToLine, CheckCircle, XCircle,
  Loader2, Zap, Clock, ChevronRight, Terminal, ChevronDown, ChevronUp,
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
  jobIds?: string[]       // 多任务时所有 job ID
  stageLabel?: string
  stageColor?: string
  accentColor?: string   // 进度条颜色，如 'violet' | 'emerald'
}

const LOG_COLOR = (line: string) => {
  if (line.includes('[ERROR]') || line.includes('✗') || line.includes('失败')) return 'text-red-400'
  if (line.includes('✓') || line.includes('已保存') || line.includes('已写入') || line.includes('完成')) return 'text-emerald-400'
  if (line.includes('[WARNING]') || line.includes('警告')) return 'text-amber-400'
  if (line.startsWith('  →') || line.includes('Step') || line.includes('[注册]')) return 'text-sky-300/80'
  if (line.includes('[跳过]')) return 'text-slate-600'
  return 'text-slate-400'
}

// ── 单个场景的动画进度卡片 ────────────────────────────────────────

function ScenarioCard({
  scenario, done, total, isRunning, accent,
}: {
  scenario: string
  done: number
  total: number
  isRunning: boolean
  accent: string
}) {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0
  const isComplete = done >= total && total > 0

  const barColor = {
    violet:  'from-violet-500 to-violet-400',
    emerald: 'from-emerald-500 to-emerald-400',
    sky:     'from-sky-500 to-sky-400',
  }[accent] ?? 'from-sky-500 to-sky-400'

  const dotColor = {
    violet:  'bg-violet-400',
    emerald: 'bg-emerald-400',
    sky:     'bg-sky-400',
  }[accent] ?? 'bg-sky-400'

  return (
    <div className={cn(
      'rounded-xl border p-4 transition-all duration-300',
      isComplete
        ? 'bg-emerald-500/5 border-emerald-500/20'
        : isRunning && pct > 0
        ? 'bg-slate-800/60 border-slate-600/50'
        : 'bg-slate-800/30 border-slate-700/40',
    )}>
      {/* 场景名 + 状态 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          {/* 状态指示点 */}
          {isComplete ? (
            <CheckCircle size={14} className="text-emerald-400 flex-shrink-0" />
          ) : isRunning && pct > 0 ? (
            <span className={cn('w-2 h-2 rounded-full flex-shrink-0 animate-pulse', dotColor)} />
          ) : (
            <span className="w-2 h-2 rounded-full bg-slate-600 flex-shrink-0" />
          )}
          <span className="font-mono text-sm text-slate-200 truncate">{scenario}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-3">
          {isComplete ? (
            <span className="text-xs text-emerald-400 font-medium">完成</span>
          ) : (
            <span className="text-xs text-slate-500 tabular-nums">
              <span className={cn('font-semibold', pct > 0 ? 'text-slate-200' : 'text-slate-500')}>
                {done.toLocaleString()}
              </span>
              {total > 0 && <span className="text-slate-700"> / {total.toLocaleString()}</span>}
            </span>
          )}
          {total > 0 && (
            <span className={cn(
              'text-xs tabular-nums font-mono w-10 text-right',
              isComplete ? 'text-emerald-400' : 'text-slate-400',
            )}>
              {pct.toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      {/* 进度条 */}
      <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full bg-gradient-to-r transition-all duration-700 ease-out',
            isComplete ? 'from-emerald-500 to-emerald-400' : barColor,
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── 总体进度环 ────────────────────────────────────────────────────

function OverallRing({
  totalDone, totalTarget, status, accent,
}: {
  totalDone: number
  totalTarget: number
  status: JobStatus
  accent: string
}) {
  const pct = totalTarget > 0 ? Math.min(100, (totalDone / totalTarget) * 100) : 0
  const r = 36
  const circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ

  const strokeColor = {
    violet:  '#8b5cf6',
    emerald: '#10b981',
    sky:     '#38bdf8',
  }[accent] ?? '#38bdf8'

  const isDone = status === 'done'
  const isError = status === 'error'

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-24 h-24">
        <svg className="w-24 h-24 -rotate-90" viewBox="0 0 88 88">
          {/* 背景圆 */}
          <circle cx="44" cy="44" r={r} fill="none" stroke="rgba(51,65,85,0.6)" strokeWidth="6" />
          {/* 进度圆 */}
          <circle
            cx="44" cy="44" r={r}
            fill="none"
            stroke={isDone ? '#10b981' : isError ? '#ef4444' : strokeColor}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 0.6s ease-out' }}
          />
        </svg>
        {/* 中心文字 */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {isDone ? (
            <CheckCircle size={22} className="text-emerald-400" />
          ) : isError ? (
            <XCircle size={22} className="text-red-400" />
          ) : (
            <>
              <span className="text-lg font-bold text-white tabular-nums leading-none">
                {pct.toFixed(0)}
              </span>
              <span className="text-xs text-slate-500">%</span>
            </>
          )}
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm font-medium text-slate-300 tabular-nums">
          {totalDone.toLocaleString()}
          {totalTarget > 0 && <span className="text-slate-600"> / {totalTarget.toLocaleString()}</span>}
        </div>
        <div className="text-xs text-slate-500 mt-0.5">
          {{ running: '处理中', done: '已完成', error: '出错', terminated: '已终止', idle: '' }[status]}
        </div>
      </div>
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────

export default function JobMonitorPanel({
  status, logs, progress, stats,
  autoScroll, onAutoScrollChange,
  onStop, onDone, doneLabel = '继续', doneIcon,
  jobId, jobIds, stageLabel, stageColor = 'text-slate-400',
  accentColor = 'sky',
}: Props) {
  const displayIds = jobIds && jobIds.length > 0 ? jobIds : (jobId ? [jobId] : [])
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

  return (
    <div className="space-y-3 mt-1">

      {/* ── 主监控卡片 ── */}
      <div className="card overflow-hidden">
        {/* 顶部状态栏 */}
        <div className={cn(
          'flex items-center gap-3 px-4 py-3 border-b border-slate-700/30',
          isDone ? 'bg-emerald-500/5' : isRunning ? 'bg-slate-800/40' : 'bg-slate-800/20',
        )}>
          <div className={cn('flex items-center gap-2', {
            running: 'text-sky-400', done: 'text-emerald-400',
            error: 'text-red-400', terminated: 'text-amber-400', idle: 'text-slate-400',
          }[status])}>
            {isRunning && <Loader2 size={14} className="animate-spin" />}
            {isDone    && <CheckCircle size={14} />}
            {status === 'error'      && <XCircle size={14} />}
            {status === 'terminated' && <XCircle size={14} />}
            <span className="font-medium text-sm">
              {{ running: '运行中', done: '已完成', error: '出错', terminated: '已终止', idle: '' }[status]}
            </span>
          </div>

          {stageLabel && (
            <>
              <div className="w-px h-4 bg-slate-700" />
              <span className={cn('text-sm', stageColor)}>{stageLabel}</span>
            </>
          )}
          {displayIds.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {displayIds.map(id => (
                <span key={id} className="text-xs text-slate-600 font-mono">#{id}</span>
              ))}
            </div>
          )}

          {/* 实时指标 */}
          {isRunning && (
            <div className="flex items-center gap-4 text-sm ml-1">
              <div className="flex items-center gap-1.5 text-sky-400">
                <Zap size={12} />
                <span className="font-mono tabular-nums">{stats.samples_per_sec.toFixed(1)}</span>
                <span className="text-slate-500 text-xs">条/s</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-400">
                <Clock size={12} />
                <span className="font-mono tabular-nums">{formatElapsed(stats.elapsed_sec)}</span>
              </div>
            </div>
          )}

          <div className="flex-1" />

          {isRunning && (
            <button className="btn-danger py-1.5 text-sm" onClick={onStop}>
              <Square size={13} /> 终止
            </button>
          )}
          {isDone && onDone && (
            <button className="btn-ghost text-sm text-emerald-400 py-1.5" onClick={onDone}>
              {doneLabel} {doneIcon ?? <ChevronRight size={14} />}
            </button>
          )}
        </div>

        {/* 进度内容 */}
        <div className="p-4">
          {scenarioList.length === 0 && isRunning && (
            /* 刚启动还没收到 init 事件时显示等待动画 */
            <div className="flex flex-col items-center gap-4 py-6">
              <div className="flex gap-1.5">
                {[0, 1, 2, 3, 4].map(i => (
                  <div
                    key={i}
                    className={cn(
                      'w-2 h-2 rounded-full animate-bounce',
                      accentColor === 'violet' ? 'bg-violet-500' :
                      accentColor === 'emerald' ? 'bg-emerald-500' : 'bg-sky-500',
                    )}
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
              <span className="text-slate-500 text-sm">等待任务开始…</span>
            </div>
          )}
          {scenarioList.length > 0 && (
            scenarioList.length <= 6 ? (
              /* ≤6 个场景：大卡片 + 总进度环并排 */
              <div className="flex gap-5 items-start">
                <div className="flex-1 grid grid-cols-1 gap-2.5">
                  {scenarioList.map(([sc, p]) => (
                    <ScenarioCard
                      key={sc} scenario={sc}
                      done={p.done} total={p.total}
                      isRunning={isRunning}
                      accent={accentColor}
                    />
                  ))}
                </div>
                {totalTarget > 0 && (
                  <div className="flex-shrink-0 pt-1">
                    <OverallRing
                      totalDone={totalDone} totalTarget={totalTarget}
                      status={status} accent={accentColor}
                    />
                  </div>
                )}
              </div>
            ) : (
              /* >6 个场景：紧凑进度条列表 + 总进度环 */
              <div className="flex gap-5 items-start">
                <div className="flex-1 space-y-2">
                  {scenarioList.map(([sc, p]) => {
                    const pct = p.total > 0 ? Math.min(100, (p.done / p.total) * 100) : 0
                    const isComplete = p.done >= p.total && p.total > 0
                    const barColor = ({
                      violet:  'from-violet-500 to-violet-400',
                      emerald: 'from-emerald-500 to-emerald-400',
                      sky:     'from-sky-500 to-sky-400',
                    } as Record<string, string>)[accentColor] ?? 'from-sky-500 to-sky-400'
                    return (
                      <div key={sc}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-mono text-slate-300 truncate max-w-[55%]">{sc}</span>
                          <span className="text-slate-500 tabular-nums">
                            {isComplete
                              ? <span className="text-emerald-400">✓</span>
                              : <><span className="text-slate-300">{p.done.toLocaleString()}</span>{p.total > 0 && <span className="text-slate-700">/{p.total.toLocaleString()}</span>}</>
                            }
                            <span className="ml-1.5">{pct.toFixed(0)}%</span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className={cn('h-full rounded-full bg-gradient-to-r transition-all duration-700', isComplete ? 'from-emerald-500 to-emerald-400' : barColor)}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
                {totalTarget > 0 && (
                  <div className="flex-shrink-0 pt-1">
                    <OverallRing
                      totalDone={totalDone} totalTarget={totalTarget}
                      status={status} accent={accentColor}
                    />
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </div>

      {/* ── 完成横幅 ── */}
      {isDone && (
        <div className="card bg-emerald-500/5 border-emerald-500/20 px-4 py-3 flex items-center gap-3">
          <CheckCircle size={17} className="text-emerald-400 flex-shrink-0" />
          <span className="text-sm text-emerald-300">
            完成！共处理 <span className="font-semibold tabular-nums">{totalDone.toLocaleString()}</span> 条。
          </span>
          {onDone && (
            <button className="btn-ghost text-sm text-emerald-400 ml-auto py-1" onClick={onDone}>
              {doneLabel} {doneIcon ?? <ChevronRight size={14} />}
            </button>
          )}
        </div>
      )}

      {/* ── 日志（折叠）── */}
      {logs.length > 0 && (
        <div className="card overflow-hidden">
          <button
            onClick={() => setLogsOpen(o => !o)}
            className="w-full flex items-center gap-2 px-4 py-2.5 hover:bg-slate-700/20 transition-colors text-left"
          >
            <Terminal size={13} className="text-slate-500" />
            <span className="text-sm text-slate-500">详细日志</span>
            <span className="text-xs text-slate-600 tabular-nums">{logs.length} 行</span>
            <div className="flex-1" />
            <button
              className={cn('p-1 rounded transition-colors', autoScroll && 'text-sky-400')}
              onClick={e => { e.stopPropagation(); onAutoScrollChange(!autoScroll) }}
              title="自动滚动"
            >
              <ArrowDownToLine size={12} />
            </button>
            {logsOpen ? <ChevronUp size={13} className="text-slate-600" /> : <ChevronDown size={13} className="text-slate-600" />}
          </button>
          {logsOpen && (
            <div style={{ height: '240px' }} className="border-t border-slate-700/30">
              <Virtuoso
                ref={virtuosoRef}
                style={{ height: '100%' }}
                data={logs}
                followOutput={autoScroll ? 'smooth' : false}
                itemContent={(_, log) => (
                  <div className={cn('px-4 py-0.5 font-mono text-sm leading-relaxed select-text', LOG_COLOR(log.line))}>
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
