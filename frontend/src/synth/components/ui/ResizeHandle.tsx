import { cn } from '../../../lib/utils'

interface Props {
  onMouseDown: (e: React.MouseEvent) => void
  color?: 'amber' | 'violet' | 'emerald' | 'rose' | 'sky'
}

const COLORS = {
  amber: {
    hover: 'hover:bg-amber-500/20',
    line: 'group-hover:bg-amber-500/40',
    knob: 'group-hover:bg-amber-500/50 group-hover:border-amber-400/40',
  },
  violet: {
    hover: 'hover:bg-violet-500/20',
    line: 'group-hover:bg-violet-500/40',
    knob: 'group-hover:bg-violet-500/50 group-hover:border-violet-400/40',
  },
  emerald: {
    hover: 'hover:bg-emerald-500/20',
    line: 'group-hover:bg-emerald-500/40',
    knob: 'group-hover:bg-emerald-500/50 group-hover:border-emerald-400/40',
  },
  rose: {
    hover: 'hover:bg-rose-500/20',
    line: 'group-hover:bg-rose-500/40',
    knob: 'group-hover:bg-rose-500/50 group-hover:border-rose-400/40',
  },
  sky: {
    hover: 'hover:bg-sky-500/20',
    line: 'group-hover:bg-sky-500/40',
    knob: 'group-hover:bg-sky-500/50 group-hover:border-sky-400/40',
  },
} as const

export default function ResizeHandle({ onMouseDown, color = 'sky' }: Props) {
  const c = COLORS[color]
  return (
    <div
      onMouseDown={onMouseDown}
      className={cn('w-1 flex-shrink-0 cursor-col-resize group relative transition-colors duration-150', c.hover)}
    >
      {/* 分隔线 */}
      <div className={cn('absolute inset-y-0 left-0 w-px bg-slate-700/50 transition-colors duration-150', c.line)} />
      {/* 抓取点 */}
      <div
        className={cn(
          'absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
          'w-2.5 h-7 rounded-full bg-slate-700/60 border border-slate-600/50',
          'flex flex-col items-center justify-center gap-0.5',
          'opacity-0 group-hover:opacity-100 transition-all duration-150',
          c.knob,
        )}
      >
        {[0, 1, 2].map(i => (
          <div key={i} className="w-px h-px rounded-full bg-white/60" />
        ))}
      </div>
    </div>
  )
}
