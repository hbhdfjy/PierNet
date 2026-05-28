import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

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

export function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false })
}

export function formatElapsed(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}m ${s}s`
}

// ── Simulator 颜色（唯一定义，所有页面从这里 import）──────────────

export const SIMULATOR_COLORS: Record<string, string> = {
  modflow: '#3b82f6',
  simpeg: '#8b5cf6',
  power_flow: '#f59e0b',
  transient: '#ef4444',
  gcam: '#10b981',
}

// Tailwind 类名形式（用于 badge/card 背景）
export const SIMULATOR_BADGE: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  modflow: { bg: 'bg-blue-500/15', text: 'text-blue-300', border: 'border-blue-500/30', dot: 'bg-blue-500' },
  simpeg: { bg: 'bg-purple-500/15', text: 'text-purple-300', border: 'border-purple-500/30', dot: 'bg-purple-500' },
  power_flow: { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30', dot: 'bg-amber-500' },
  transient: { bg: 'bg-red-500/15', text: 'text-red-300', border: 'border-red-500/30', dot: 'bg-red-500' },
  gcam: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/30', dot: 'bg-emerald-500' },
}

export function getSimulatorBadgeClass(simulator: string): string {
  const c = SIMULATOR_BADGE[simulator]
  if (!c) return 'bg-slate-700/50 text-slate-300 border-slate-600/40'
  return `${c.bg} ${c.text} ${c.border}`
}

export const LANGUAGE_LABELS: Record<string, string> = {
  en: 'English',
  zh: '中文',
}

export const LANGUAGE_BADGE: Record<string, string> = {
  en: 'bg-sky-500/10 text-sky-400 border-sky-500/20',
  zh: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
}

export const STYLE_LABELS: Record<string, string> = {
  technical: '专业技术',
  popular: '科普',
  concise: '简洁',
}

export const STYLE_BADGE: Record<string, string> = {
  technical: 'bg-teal-500/10 text-teal-400 border-teal-500/20',
  popular: 'bg-pink-500/10 text-pink-400 border-pink-500/20',
  concise: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
}

export const SIMULATOR_LABELS: Record<string, string> = {
  modflow: 'MODFLOW（地下水）',
  simpeg: 'SimPEG（地球物理）',
  power_flow: 'pandapower（稳态潮流）',
  transient: 'ANDES（暂态稳定）',
  gcam: 'PyPSA（能源-气候）',
}

// 多通道时序图颜色
const LINE_PALETTE = [
  '#38bdf8',
  '#f87171',
  '#34d399',
  '#fbbf24',
  '#a78bfa',
  '#22d3ee',
  '#fb923c',
  '#84cc16',
  '#ec4899',
  '#6366f1',
]
export function getLineColor(index: number): string {
  return LINE_PALETTE[index % LINE_PALETTE.length]
}
