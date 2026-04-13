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
        'relative text-left px-3 py-2.5 rounded-xl border transition-all duration-150 overflow-hidden w-full',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40',
        disabled
          ? 'opacity-35 cursor-not-allowed bg-slate-800/20 border-slate-700/20'
          : noData
          ? 'bg-slate-800/20 border-slate-700/25 border-dashed cursor-pointer opacity-60 hover:opacity-80'
          : active
          ? 'bg-sky-500/12 border-sky-500/35 shadow-sm'
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
          <span className={cn(
            'font-medium text-xs truncate leading-none',
            active ? 'text-sky-300' : noData ? 'text-slate-500' : 'text-slate-200',
          )}>
            {s.name}
          </span>
          {active && <Check size={11} className="text-sky-400 flex-shrink-0" />}
          {noData && !active && <AlertCircle size={10} className="text-slate-600 flex-shrink-0" />}
        </div>

        {/* 元数据行 */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* simulator */}
          <span className={cn(
            'text-[10px] flex items-center gap-1',
            active ? 'text-sky-400/70' : (c?.text ?? 'text-slate-500'),
          )}>
            <span className={cn('w-1 h-1 rounded-full flex-shrink-0', c?.dot ?? 'bg-slate-600')} />
            {s.simulator}
          </span>

          {/* 样本数 */}
          {s.has_h5 ? (
            <span className="text-[10px] text-slate-600 tabular-nums flex items-center gap-0.5">
              <Database size={9} />{s.sample_count.toLocaleString()}
            </span>
          ) : (
            <span className="text-[10px] text-amber-600/70 flex items-center gap-0.5">
              <Database size={9} />无数据
            </span>
          )}

          {/* 输出形状 */}
          {s.output_shape && (
            <span className="text-[10px] text-slate-600 font-mono">
              {s.output_shape.join('×')}
            </span>
          )}

          {/* 注册状态 */}
          {s.registered ? (
            <span className="text-[10px] text-emerald-600/60 flex items-center gap-0.5">
              <Tag size={9} />已注册
            </span>
          ) : (
            <span className="text-[10px] text-red-500/50 flex items-center gap-0.5">
              <Tag size={9} />未注册
            </span>
          )}

          {/* 模板数 */}
          {templateCount !== undefined && templateCount > 0 && (
            <span className="text-[10px] text-sky-500/80 tabular-nums flex items-center gap-0.5">
              <Sparkles size={9} />{templateCount.toLocaleString()}
            </span>
          )}

          {/* 已有 JSONL */}
          {s.has_jsonl && (
            <span className="text-[10px] text-emerald-500/70 tabular-nums">
              ✓{s.existing_jsonl_count}
            </span>
          )}
        </div>

        {/* 提示文字 */}
        {noData && (
          <p className="text-[10px] text-slate-600 mt-1 leading-tight">需先运行 Stage 1 生成 HDF5</p>
        )}
        {!noData && unregistered && (
          <p className="text-[10px] text-amber-600/60 mt-1 leading-tight">未注册元数据，生成可能失败</p>
        )}
      </div>
    </button>
  )
}
