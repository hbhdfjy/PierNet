import { BarChart3, Save } from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'
import { Tooltip, XAxis, YAxis } from 'recharts'
import type { TrainingCheckpointInfo } from '../../../lib/types'
import { TruncatedText } from '../../../shared/ui'
import { TrainingSectionTitle } from '../common'
import { formatBytes, formatDateTime } from '../../shared'
import { formatChartTooltipLabel, formatChartTooltipValue, tooltipActualValue } from './chartUtils'
import type { ChartHoverSnapshot, ChartTooltipEntry } from './chartUtils'

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    background: 'rgba(15, 23, 42, 0.96)',
    border: '1px solid rgba(71, 85, 105, 0.55)',
    borderRadius: 16,
    boxShadow: '0 16px 40px rgba(2, 6, 23, 0.42)',
    padding: '10px 12px',
  },
  labelStyle: {
    color: '#e2e8f0',
    fontWeight: 600,
  },
  itemStyle: {
    color: '#cbd5e1',
  },
}

const AXIS_STYLE = {
  stroke: 'rgba(148,163,184,0.86)',
  fontSize: 12,
  tickLine: false,
  axisLine: false,
  tickMargin: 8,
}
const AXIS_TICK_STYLE = {
  fill: 'rgba(148,163,184,0.9)',
  fontSize: 12,
}

type ChartHoverPanelVariant = 'default' | 'emphasis'

function nodeTitle(value: ReactNode): string | undefined {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value)
  }
  return undefined
}

export function ChartTooltip({ axisLabel }: { axisLabel: string }) {
  return (
    <Tooltip
      {...CHART_TOOLTIP_STYLE}
      allowEscapeViewBox={{ x: true, y: true }}
      wrapperStyle={{ outline: 'none', pointerEvents: 'none', zIndex: 30 }}
      cursor={{ stroke: 'rgba(148,163,184,0.4)', strokeDasharray: '4 4', strokeWidth: 1.2 }}
      content={({ active, label, payload }) => {
        const entries =
          (payload as ChartTooltipEntry[] | undefined)?.filter(
            item => item.value !== null && item.value !== undefined,
          ) ?? []
        if (!active || entries.length === 0) {
          return null
        }

        return (
          <div
            style={{
              background: 'rgba(15, 23, 42, 0.97)',
              border: '1px solid rgba(71, 85, 105, 0.55)',
              borderRadius: 16,
              boxShadow: '0 16px 40px rgba(2, 6, 23, 0.42)',
              padding: '10px 12px',
              minWidth: 164,
            }}
          >
            <div style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 12, marginBottom: 8 }}>
              {formatChartTooltipLabel(axisLabel, label)}
            </div>
            <div style={{ display: 'grid', gap: 6 }}>
              {entries.map(item => (
                <div
                  key={String(item.dataKey ?? item.name)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '10px minmax(0, 1fr) auto',
                    gap: 8,
                    alignItems: 'center',
                  }}
                >
                  <span
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 999,
                      background: item.color ?? '#38bdf8',
                      boxShadow: '0 0 0 2px rgba(15, 23, 42, 0.96)',
                    }}
                  />
                  <span style={{ color: '#cbd5e1', fontSize: 12, minWidth: 0 }}>
                    {item.name ?? item.dataKey ?? 'value'}
                  </span>
                  <span style={{ color: '#f8fafc', fontSize: 12, fontWeight: 600, marginLeft: 8 }}>
                    {formatChartTooltipValue(tooltipActualValue(item))}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )
      }}
    />
  )
}

export function ChartHoverPanel({
  snapshot,
  variant = 'default',
}: {
  snapshot: ChartHoverSnapshot | null
  variant?: ChartHoverPanelVariant
}) {
  const isEmphasis = variant === 'emphasis'
  const hasPosition = typeof snapshot?.x === 'number' && typeof snapshot?.y === 'number'
  const maxWidth = isEmphasis ? '19rem' : '15rem'
  const positionHeight = isEmphasis ? '24rem' : '14rem'
  const panelStyle: CSSProperties = hasPosition
    ? {
        left: `clamp(0.75rem, ${(snapshot?.x ?? 0) + 14}px, max(0.75rem, calc(100% - ${maxWidth})))`,
        top: `clamp(0.75rem, ${(snapshot?.y ?? 0) + 14}px, max(0.75rem, calc(100% - ${positionHeight})))`,
      }
    : {
        right: '1rem',
        top: '1rem',
      }

  return (
    <div
      className={`pointer-events-none absolute z-50 rounded-2xl border backdrop-blur-xl transition-[left,top] duration-75 ${
        isEmphasis
          ? 'min-w-[240px] max-w-[300px] border-sky-400/35 bg-slate-950/98 px-4 py-3.5 shadow-[0_24px_52px_rgba(2,6,23,0.62)]'
          : 'min-w-[180px] max-w-[240px] border-slate-700/55 bg-slate-950/96 px-3.5 py-3 shadow-[0_18px_40px_rgba(2,6,23,0.46)]'
      }`}
      style={panelStyle}
    >
      <div
        className={`font-semibold uppercase tracking-[0.16em] ${isEmphasis ? 'text-[10px] text-sky-300/80' : 'text-[11px] text-slate-500'}`}
      >
        {snapshot ? snapshot.label : isEmphasis ? '当前点' : '悬停数值'}
      </div>
      {snapshot ? (
        <div className={`mt-2 ${isEmphasis ? 'grid gap-2.5' : 'grid gap-2'}`}>
          {snapshot.rows.map(row => (
            <div
              key={row.key}
              className={`grid items-center ${
                isEmphasis
                  ? 'grid-cols-[12px_minmax(0,1fr)_auto] gap-2 rounded-lg border px-2.5 py-1.5'
                  : 'grid-cols-[10px_minmax(0,1fr)_auto] gap-2'
              }`}
              style={
                isEmphasis
                  ? {
                      borderColor: `${row.color}33`,
                      background: `linear-gradient(90deg, ${row.color}18, rgba(15,23,42,0.16) 42%)`,
                    }
                  : undefined
              }
            >
              <span
                className={
                  isEmphasis
                    ? 'h-3 w-3 rounded-full shadow-[0_0_0_3px_rgba(15,23,42,0.92)]'
                    : 'h-2.5 w-2.5 rounded-full'
                }
                style={{ backgroundColor: row.color }}
              />
              <span
                className={`truncate ${isEmphasis ? 'text-[12.5px] font-medium text-slate-200' : 'text-[12px] text-slate-300'}`}
              >
                {row.label}
              </span>
              <span
                className={`mono font-semibold ${isEmphasis ? 'text-[13px] text-white' : 'text-[12px] text-slate-100'}`}
              >
                {row.value}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className={`mt-2 ${isEmphasis ? 'text-[12.5px] text-slate-300' : 'text-[12px] text-slate-400'}`}>
          将指针移到图表上查看数值。
        </div>
      )}
    </div>
  )
}

export function ChartXAxis({
  dataKey,
  type = 'category',
  allowDecimals = true,
}: {
  dataKey: string
  type?: 'category' | 'number'
  allowDecimals?: boolean
}) {
  return (
    <XAxis
      dataKey={dataKey}
      type={type}
      allowDecimals={allowDecimals}
      minTickGap={20}
      tick={AXIS_TICK_STYLE}
      {...AXIS_STYLE}
    />
  )
}

export function ChartYAxis({
  domain,
  tickCount,
  tickFormatter,
  width = 56,
  allowDataOverflow,
}: {
  domain?: [number, number]
  tickCount?: number
  tickFormatter?: (value: number) => string
  width?: number
  allowDataOverflow?: boolean
}) {
  return (
    <YAxis
      width={width}
      domain={domain}
      tickCount={tickCount}
      tickFormatter={tickFormatter}
      allowDataOverflow={allowDataOverflow}
      tick={AXIS_TICK_STYLE}
      {...AXIS_STYLE}
    />
  )
}

export const SectionTitle = TrainingSectionTitle

export function MetaField({
  label,
  value,
  mono = false,
  title,
}: {
  label: string
  value: ReactNode
  mono?: boolean
  title?: string
}) {
  const valueTitle = title ?? nodeTitle(value)
  return (
    <div>
      <div className="training-label">{label}</div>
      <div className="pretty-tooltip mt-1 min-w-0" data-tooltip={valueTitle}>
        <span className={`block truncate text-[13px] leading-5 ${mono ? 'mono text-slate-200' : 'text-slate-100'}`}>
          {value}
        </span>
      </div>
    </div>
  )
}

export function ChartCard({
  title,
  subtitle,
  actions,
  overlay,
  children,
}: {
  title: string
  subtitle: string
  actions?: ReactNode
  overlay?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="training-card training-card--chart">
      <div className="card-header justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
            <BarChart3 size={16} />
          </span>
          <SectionTitle title={title} copy={subtitle} />
        </div>
        {actions}
      </div>
      <div className="relative h-[300px] overflow-visible p-3.5">
        {overlay}
        {children}
      </div>
    </div>
  )
}

export function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center rounded-xl border border-slate-700/40 bg-slate-900/25 px-6 text-center text-[13px] text-slate-400">
      {message}
    </div>
  )
}

export function CheckpointList({ checkpoints }: { checkpoints: TrainingCheckpointInfo[] }) {
  if (checkpoints.length === 0) {
    return <div className="training-surface text-[14px] text-slate-400">当前还没有权重文件。</div>
  }

  return (
    <div className="checkpoint-list">
      {checkpoints.map(item => (
        <div key={item.path} className="checkpoint-card training-surface--dense">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="mono truncate text-[14px] font-semibold text-slate-100">{item.name}</div>
              <div className="pretty-tooltip mt-1 text-[11px] text-slate-500" data-tooltip={item.path}>
                <TruncatedText value={item.path} className="text-[11px] text-slate-500" />
              </div>
            </div>
            <span className="flex h-7 w-7 items-center justify-center rounded-xl border border-emerald-500/20 bg-emerald-500/8 text-emerald-300">
              <Save size={14} />
            </span>
          </div>
          <div className="checkpoint-card__meta mt-2 grid gap-2 text-[12px] text-slate-400">
            <MetaField label="轮次" value={item.epoch ?? '—'} />
            <MetaField label="大小" value={formatBytes(item.size_bytes)} />
            <MetaField label="时间" value={formatDateTime(item.mtime)} />
          </div>
        </div>
      ))}
    </div>
  )
}
