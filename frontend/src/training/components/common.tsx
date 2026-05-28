import { cn } from '../../lib/utils'
import { PanelTitle } from '../../shared/ui'

export type TrainingProgressTone = 'sky' | 'emerald' | 'violet' | 'amber'

export const TrainingSectionTitle = PanelTitle

export function TrainingUsageBar({
  value,
  tone = 'sky',
  className = 'mt-1.5',
}: {
  value: number
  tone?: TrainingProgressTone
  className?: string
}) {
  const pct = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
  return (
    <div className={cn('training-progress', `training-progress--${tone}`, className)}>
      <div className="training-progress__fill" style={{ width: `${pct}%` }} />
    </div>
  )
}
