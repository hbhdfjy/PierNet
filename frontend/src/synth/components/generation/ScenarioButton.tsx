import { Check, Database, Sparkles, AlertCircle, Tag } from 'lucide-react'
import { cn, SIMULATOR_BADGE } from '../../../lib/utils'
import type { Text2CompScenario } from '../../../lib/types'

interface Props {
  s: Text2CompScenario
  active: boolean
  onClick: () => void
  templateCount?: number
  disabled?: boolean
  tone?: 'sky' | 'violet' | 'emerald'
}

const TONE = {
  sky: {
    focus: 'focus-visible:ring-sky-500/40',
    active: 'bg-sky-500/12 border-sky-500/35 shadow-sm',
    title: 'text-sky-300',
    icon: 'text-sky-400',
    meta: 'text-sky-400/70',
    count: 'text-sky-500/80',
  },
  violet: {
    focus: 'focus-visible:ring-violet-500/40',
    active: 'bg-violet-500/12 border-violet-500/35 shadow-sm',
    title: 'text-violet-300',
    icon: 'text-violet-400',
    meta: 'text-violet-400/70',
    count: 'text-violet-500/80',
  },
  emerald: {
    focus: 'focus-visible:ring-emerald-500/40',
    active: 'bg-emerald-500/12 border-emerald-500/35 shadow-sm',
    title: 'text-emerald-300',
    icon: 'text-emerald-400',
    meta: 'text-emerald-400/70',
    count: 'text-emerald-500/80',
  },
} as const

export default function ScenarioButton({ s, active, onClick, templateCount, disabled, tone = 'sky' }: Props) {
  const pct = s.sample_count > 0 ? Math.min(100, (s.existing_jsonl_count / s.sample_count) * 100) : 0
  const c = SIMULATOR_BADGE[s.simulator]
  const noData = !s.has_h5
  const unregistered = !s.registered
  const palette = TONE[tone]

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={noData ? '尚无 HDF5 数据，需先运行 阶段 1 物理仿真' : undefined}
      className={cn(
        'relative text-left px-3 py-2.5 rounded-xl border transition-all duration-150 overflow-hidden w-full',
        'focus:outline-none focus-visible:ring-2',
        palette.focus,
        disabled
          ? 'opacity-35 cursor-not-allowed bg-slate-800/20 border-slate-700/20'
          : noData
            ? 'bg-slate-800/20 border-slate-700/25 border-dashed cursor-pointer opacity-60 hover:opacity-80'
            : active
              ? palette.active
              : cn(
                  c?.bg ?? 'bg-slate-800/40',
                  c?.border ?? 'border-slate-700/40',
                  'hover:border-opacity-80 cursor-pointer hover:bg-opacity-70',
                ),
      )}
    >
      {/* 已生成进度底色 */}
      {s.has_jsonl && pct > 0 && !active && !disabled && !noData && (
        <div
          className="absolute inset-y-0 left-0 bg-emerald-500/6 pointer-events-none transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      )}

      <div className="relative">
        {/* 场景名 + 状态 */}
        <div className="flex items-center justify-between gap-1 mb-1.5">
          <span
            className={cn(
              'font-medium text-sm truncate leading-none',
              active ? palette.title : noData ? 'text-slate-500' : 'text-slate-200',
            )}
          >
            {s.name}
          </span>
          {active && <Check size={11} className={cn('flex-shrink-0', palette.icon)} />}
          {noData && !active && <AlertCircle size={10} className="text-slate-600 flex-shrink-0" />}
        </div>

        {/* 元数据行 */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* simulator */}
          <span
            className={cn('text-xs flex items-center gap-1', active ? palette.meta : (c?.text ?? 'text-slate-500'))}
          >
            <span className={cn('w-1 h-1 rounded-full flex-shrink-0', c?.dot ?? 'bg-slate-600')} />
            {s.simulator}
          </span>

          {/* 样本数 */}
          {s.has_h5 ? (
            <span className="text-xs text-slate-600 tabular-nums flex items-center gap-0.5">
              <Database size={9} />
              {s.sample_count.toLocaleString()}
            </span>
          ) : (
            <span className="text-xs text-amber-600/70 flex items-center gap-0.5">
              <Database size={9} />
              无数据
            </span>
          )}

          {/* 输出形状 */}
          {s.output_shape && <span className="text-xs text-slate-600 font-mono">{s.output_shape.join('×')}</span>}

          {/* 注册状态 */}
          {s.registered ? (
            <span className="text-xs text-emerald-600/60 flex items-center gap-0.5">
              <Tag size={9} />
              已注册
            </span>
          ) : (
            <span className="text-xs text-red-500/50 flex items-center gap-0.5">
              <Tag size={9} />
              未注册
            </span>
          )}

          {/* 模板数 */}
          {templateCount !== undefined && templateCount > 0 && (
            <span className={cn('text-xs tabular-nums flex items-center gap-0.5', palette.count)}>
              <Sparkles size={9} />
              {templateCount.toLocaleString()}
            </span>
          )}

          {/* 已有 JSONL */}
          {s.has_jsonl && <span className="text-xs text-emerald-500/70 tabular-nums">✓{s.existing_jsonl_count}</span>}
        </div>

        {/* 提示文字 */}
        {noData && <p className="text-xs text-slate-600 mt-1 leading-tight">需先运行阶段 1 生成 HDF5</p>}
        {!noData && unregistered && (
          <p className="text-xs text-amber-600/60 mt-1 leading-tight">未注册元数据，生成可能失败</p>
        )}
      </div>
    </button>
  )
}
