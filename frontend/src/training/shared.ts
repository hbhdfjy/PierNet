import type { TrainingJobStatus } from '../lib/types'

export function formatCount(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return new Intl.NumberFormat('zh-CN').format(value)
}

export function formatBytes(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size >= 100 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`
}

export function formatDateTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', {
    hour12: false,
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.floor(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h}小时 ${m}分钟`
  if (m > 0) return `${m}分钟 ${s}秒`
  return `${s}秒`
}

export function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

export function statusLabel(status: TrainingJobStatus): string {
  switch (status) {
    case 'starting':
      return '启动中'
    case 'running':
      return '训练中'
    case 'evaluating':
      return '测试中'
    case 'stopping':
      return '停止中'
    case 'done':
      return '已完成'
    case 'error':
      return '失败'
    case 'terminated':
      return '已终止'
    case 'external_terminated':
      return '外部终止'
    case 'queued':
      return '排队中'
    default:
      return status
  }
}

export function statusBadgeClass(status: TrainingJobStatus): string {
  switch (status) {
    case 'done':
      return 'badge bg-emerald-500/8 text-emerald-300 border border-emerald-500/20'
    case 'running':
    case 'evaluating':
    case 'starting':
      return 'badge bg-sky-500/15 text-sky-300 border border-sky-500/20'
    case 'stopping':
      return 'badge bg-amber-500/14 text-amber-300 border border-amber-500/25'
    case 'terminated':
      return 'badge bg-amber-500/12 text-amber-300 border border-amber-500/20'
    case 'external_terminated':
      return 'badge bg-orange-500/12 text-orange-300 border border-orange-500/25'
    case 'error':
      return 'badge bg-rose-500/8 text-rose-300 border border-rose-500/20'
    default:
      return 'badge bg-slate-800/70 text-slate-300 border border-slate-700/40'
  }
}

export function gpuUsageLabel(used: number, total: number): string {
  if (!total) return '—'
  return `${used} / ${total} MiB`
}

export function shortPath(value: string, max = 72): string {
  if (value.length <= max) return value
  return `${value.slice(0, 22)}…${value.slice(-Math.max(18, max - 24))}`
}
