import { Check, Database, Sparkles, AlertCircle, Tag } from 'lucide-react'
import { cn, SIMULATOR_BADGE } from '../../lib/utils'
import type { Text2CompScenario } from '../../lib/types'

interface Props {
  s: Text2CompScenario
  active: boolean
  onClick: () => void
  templateCount?: number
  disabled?: boolean
}

export default function ScenarioButton({ s, active, onClick, templateCount, disabled }: Props) {
  const pct = s.sample_count > 0
    ? Math.min(100, (s.existing_jsonl_count / s.sample_count) * 100)
    : 0
  const c = SIMULATOR_BADGE[s.simulator]
  const noData = !s.has_h5
  const unregistered = !s.registered

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={noData ? '尚无 HDF5 数据，需先运行 Stage 1 物理仿真' : undefined}
      className={cn(
        'relative text-left px-3 py-3 rounded-xl border transition-all duration-150 overflow-hidden w-full',
        disabled
          ? 'opacity-35 cursor-not-allowed bg-slate-800/30 border-slate-700/30'
          : noData
          ? 'bg-slate-800/20 border-slate-700/30 border-dashed cursor-pointer opacity-70 hover:opacity-90'
          : active
          ? 'bg-sky-500/15 border-sky-500/40 shadow-sm shadow-sky-900/20'
          : `${c?.bg ?? 'bg-slate-800/50'} ${c?.border ?? 'border-slate-700/50'} hover:border-opacity-70 cursor-pointer`,
      )}
    >
      {/* 已生成进度底色 */}
      {s.has_jsonl && pct > 0 && !active && !disabled && !noData && (
        <div className="absolute inset-y-0 left-0 bg-emerald-500/8 pointer-events-none"
          style={{ width: `${pct}%` }} />
      )}
      <div className="relative">
        <div className="flex items-center justify-between gap-1 mb-1">
          <span className={cn('font-semibold text-sm truncate',
            active ? 'text-sky-200' : noData ? 'text-slate-500' : 'text-slate-100')}>
            {s.name}
          </span>
          {active && <Check size={14} className="text-sky-400 flex-shrink-0" />}
          {noData && !active && <AlertCircle size={12} className="text-slate-600 flex-shrink-0" />}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className={cn('text-xs flex items-center gap-1', active ? 'text-sky-400/70' : (c?.text ?? 'text-slate-400'))}>
            <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', c?.dot ?? 'bg-slate-500')} />
            {s.simulator}
          </span>
          {s.has_h5 ? (
            <span className="text-xs text-slate-600 tabular-nums flex items-center gap-0.5">
              <Database size={10} />{s.sample_count.toLocaleString()}
            </span>
          ) : (
            <span className="text-xs text-amber-600/80 flex items-center gap-0.5">
              <Database size={10} />无数据
            </span>
          )}
          {s.output_shape && (
            <span className="text-xs text-slate-500 font-mono">
              ({s.output_shape.join('×')})
            </span>
          )}
          {s.registered ? (
            <span className="text-xs text-emerald-600/70 flex items-center gap-0.5">
              <Tag size={10} />已注册
            </span>
          ) : (
            <span className="text-xs text-red-500/60 flex items-center gap-0.5">
              <Tag size={10} />未注册
            </span>
          )}
          {templateCount !== undefined && templateCount > 0 && (
            <span className="text-xs text-sky-500 tabular-nums flex items-center gap-0.5">
              <Sparkles size={10} />{templateCount.toLocaleString()}
            </span>
          )}
          {s.has_jsonl && (
            <span className="text-xs text-emerald-600 tabular-nums">✓{s.existing_jsonl_count}</span>
          )}
        </div>
        {noData && (
          <p className="text-xs text-slate-600 mt-1 leading-tight">需先运行 Stage 1 生成 HDF5</p>
        )}
        {!noData && unregistered && (
          <p className="text-xs text-amber-600/70 mt-1 leading-tight">未注册元数据，生成可能失败</p>
        )}
      </div>
    </button>
  )
}
