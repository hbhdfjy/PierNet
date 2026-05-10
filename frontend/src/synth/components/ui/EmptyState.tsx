import { type LucideIcon } from 'lucide-react'
import { cn } from '../../../lib/utils'

interface Props {
  icon: LucideIcon
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

export default function EmptyState({ icon: Icon, title, description, action, className, size = 'md' }: Props) {
  const s = {
    sm: { iconSize: 20, iconBox: 'w-10 h-10 rounded-xl', py: 'py-8', title: 'text-sm' },
    md: { iconSize: 28, iconBox: 'w-14 h-14 rounded-2xl', py: 'py-14', title: 'text-base' },
    lg: { iconSize: 36, iconBox: 'w-18 h-18 rounded-2xl', py: 'py-20', title: 'text-lg' },
  }[size]

  return (
    <div className={cn(`empty-state ${s.py}`, className)}>
      <div className={cn('empty-state__icon', s.iconBox)}>
        <Icon size={s.iconSize} className="text-slate-500" />
      </div>
      <div>
        <p className={cn('empty-state__title font-medium', s.title)}>{title}</p>
        {description && <p className="empty-state__description text-xs mt-1 max-w-xs leading-relaxed">{description}</p>}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  )
}
