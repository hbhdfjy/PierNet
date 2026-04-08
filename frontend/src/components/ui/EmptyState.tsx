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
  const sizes = {
    sm: { icon: 24, title: 'text-sm', desc: 'text-xs', gap: 'gap-2', py: 'py-8' },
    md: { icon: 36, title: 'text-base', desc: 'text-sm', gap: 'gap-3', py: 'py-12' },
    lg: { icon: 48, title: 'text-lg', desc: 'text-base', gap: 'gap-4', py: 'py-16' },
  }
  const s = sizes[size]
  return (
    <div className={cn(`flex flex-col items-center justify-center ${s.py} ${s.gap}`, className)}>
      <Icon size={s.icon} className="text-slate-600 opacity-60" />
      <div className="text-center">
        <p className={cn('font-medium text-slate-400', s.title)}>{title}</p>
        {description && <p className={cn('text-slate-600 mt-1', s.desc)}>{description}</p>}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  )
}
