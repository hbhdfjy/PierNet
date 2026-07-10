import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { cn } from '../lib/utils'

export function StatusBadge({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold',
        className,
      )}
    >
      {children}
    </span>
  )
}

export function MetricTile({
  label,
  value,
  note,
  icon,
}: {
  label: string
  value: string
  note?: string
  icon?: ReactNode
}) {
  return (
    <div className="training-kpi">
      <div className="flex items-start justify-between gap-3">
        <span className="training-kpi__label">{label}</span>
        {icon && <span className="training-kpi__icon">{icon}</span>}
      </div>
      <div className="training-kpi__value">{value}</div>
      {note && <div className="training-kpi__note">{note}</div>}
    </div>
  )
}

export function PanelTitle({ title, copy, className }: { title: string; copy?: string; className?: string }) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="training-panel-title">{title}</div>
      {copy && <div className="training-panel-copy">{copy}</div>}
    </div>
  )
}

export function MetaTile({
  label,
  value,
  className,
  valueClassName,
  tooltip = true,
}: {
  label: string
  value: string
  className?: string
  valueClassName?: string
  tooltip?: boolean
}) {
  const valueNode = (
    <div className={cn('truncate font-mono text-[13px] font-semibold text-slate-100', valueClassName)}>{value}</div>
  )

  return (
    <div className={cn('rounded-lg border border-slate-700/35 bg-slate-900/30 p-2.5', className)}>
      <div className="label mb-1 text-[11px]">{label}</div>
      {tooltip ? (
        <div className="pretty-tooltip min-w-0" data-tooltip={value}>
          {valueNode}
        </div>
      ) : (
        valueNode
      )}
    </div>
  )
}

type EmptyStateSize = 'sm' | 'md' | 'lg'

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
  size = 'md',
}: {
  icon: LucideIcon
  title: string
  description?: string
  action?: ReactNode
  className?: string
  size?: EmptyStateSize
}) {
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
        {description && <p className="empty-state__description mt-1 max-w-xs text-xs leading-relaxed">{description}</p>}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  )
}

export function TruncatedText({
  children,
  value,
  className,
  tooltipClassName,
}: {
  children?: ReactNode
  value: string
  className?: string
  tooltipClassName?: string
}) {
  return (
    <div className={cn('pretty-tooltip min-w-0', tooltipClassName)} data-tooltip={value}>
      <div className={cn('truncate', className)}>{children ?? (value || '—')}</div>
    </div>
  )
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = '确认',
  cancelLabel = '取消',
  danger = false,
  busy = false,
  onCancel,
  onConfirm,
}: {
  open: boolean
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
  busy?: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/70 px-4 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-slate-700/70 bg-slate-900 p-4 shadow-2xl shadow-slate-950/60">
        <h2 className="text-base font-semibold text-slate-100">{title}</h2>
        <div className="mt-2 text-sm leading-6 text-slate-300">{description}</div>
        <div className="mt-4 flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </button>
          <button type="button" className={danger ? 'btn-danger' : 'btn-primary'} onClick={onConfirm} disabled={busy}>
            {busy ? '处理中...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
