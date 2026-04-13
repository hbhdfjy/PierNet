import { type LucideIcon } from 'lucide-react'
import { cn } from '../../lib/utils'

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
    sm: { iconSize: 20, iconBox: 'w-10 h-10 rounded-xl', py: 'py-8',  gap: 'gap-2.5', title: 'text-sm'  },
    md: { iconSize: 28, iconBox: 'w-14 h-14 rounded-2xl', py: 'py-14', gap: 'gap-3',   title: 'text-base' },
    lg: { iconSize: 36, iconBox: 'w-18 h-18 rounded-2xl', py: 'py-20', gap: 'gap-4',   title: 'text-lg'  },
  }[size]

  return (
    <div className={cn(
      `flex flex-col items-center justify-center text-center ${s.py} ${s.gap}`,
      className,
    )}>
      <div className={cn(
        s.iconBox,
        'flex items-center justify-center',
        'bg-slate-800/50 border border-slate-700/30',
      )}>
        <Icon size={s.iconSize} className="text-slate-600" />
      </div>
      <div>
        <p className={cn('font-medium text-slate-400', s.title)}>{title}</p>
        {description && (
          <p className="text-xs text-slate-600 mt-1 max-w-xs leading-relaxed">{description}</p>
        )}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  )
}
