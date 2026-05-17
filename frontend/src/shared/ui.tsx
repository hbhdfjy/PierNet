import type { ReactNode } from 'react'
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
