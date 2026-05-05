import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  FileText,
  PauseCircle,
  RadioTower,
  RefreshCcw,
  Save,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../../lib/api'
import type {
  TrainingCheckpointInfo,
  TrainingCurvesResponse,
  TrainingJobDetail,
  TrainingLogResponse,
  TrainingPoint,
} from '../../lib/types'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatMetric,
  shortPath,
  statusBadgeClass,
  statusLabel,
} from '../shared'

type TrainingAxisMode = 'step' | 'epoch'

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

const CHART_MARGIN = { top: 10, right: 12, left: 6, bottom: 0 }
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
const LEGEND_STYLE = {
  fontSize: 12,
  color: '#94a3b8',
  paddingTop: 10,
}

type ChartTooltipEntry = {
  color?: string
  dataKey?: string | number
  name?: string
  payload?: Record<string, unknown>
  value?: unknown
}

type ChartHoverSnapshot = {
  label: string
  rows: Array<{
    color: string
    key: string
    label: string
    value: string
  }>
}

type ChartMouseState = {
  activeLabel?: unknown
  activePayload?: ChartTooltipEntry[]
}

type ChartHoverPanelVariant = 'default' | 'emphasis'

type NumericDomainOptions = {
  maxClamp?: number
  minClamp?: number
  minPad?: number
  padRatio?: number
  lowerQuantile?: number
  upperQuantile?: number
}

/*
function formatChartTooltipValue(value: unknown) {
  if (value === null || value === undefined || value === '') {
    return '—'
  }
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return String(value)
  }
  const abs = Math.abs(value)
  const maximumFractionDigits = abs >= 1000 ? 2 : abs >= 100 ? 3 : abs >= 1 ? 4 : 6
  return value.toLocaleString('en-US', { maximumFractionDigits })
}
*/

function formatChartTooltipValue(value: unknown) {
  if (value === null || value === undefined || value === '') {
    return '--'
  }
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return String(value)
  }
  const abs = Math.abs(value)
  const maximumFractionDigits = abs >= 1000 ? 2 : abs >= 100 ? 3 : abs >= 1 ? 4 : 6
  return value.toLocaleString('en-US', { maximumFractionDigits })
}

function formatChartTooltipLabel(axisLabel: string, label: unknown) {
  if (label === null || label === undefined || label === '') {
    return axisLabel
  }
  if (typeof label === 'number') {
    return `${axisLabel}: ${label.toLocaleString('en-US')}`
  }
  return `${axisLabel}: ${String(label)}`
}

function tooltipActualValue(item: ChartTooltipEntry): unknown {
  const payload = item.payload
  if (!payload) {
    return item.value
  }
  const nameKey = typeof item.name === 'string' ? item.name : undefined
  if (nameKey && Object.prototype.hasOwnProperty.call(payload, nameKey)) {
    return payload[nameKey]
  }
  const dataKey = item.dataKey == null ? undefined : String(item.dataKey)
  if (dataKey && Object.prototype.hasOwnProperty.call(payload, dataKey)) {
    return payload[dataKey]
  }
  return item.value
}

function buildActiveDot(fill: string) {
  return {
    r: 4.5,
    fill,
    stroke: 'rgba(15,23,42,0.96)',
    strokeWidth: 2,
  }
}

function quantile(sortedValues: number[], ratio: number) {
  if (sortedValues.length === 0) return undefined
  const index = (sortedValues.length - 1) * ratio
  const lower = Math.floor(index)
  const upper = Math.ceil(index)
  if (lower === upper) {
    return sortedValues[lower]
  }
  const weight = index - lower
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight
}

function buildNumericDomain(values: Array<number | null | undefined>, options: NumericDomainOptions = {}): [number, number] | undefined {
  const {
    padRatio = 0.08,
    minPad = 0.0001,
    minClamp,
    maxClamp,
    lowerQuantile = 0,
    upperQuantile = 1,
  } = options

  const numericValues = values
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    .sort((a, b) => a - b)
  if (numericValues.length === 0) {
    return undefined
  }

  let minValue = quantile(numericValues, lowerQuantile) ?? numericValues[0]
  let maxValue = quantile(numericValues, upperQuantile) ?? numericValues[numericValues.length - 1]
  if (maxValue < minValue) {
    ;[minValue, maxValue] = [maxValue, minValue]
  }
  const span = maxValue - minValue
  const padding = span === 0
    ? Math.max(Math.abs(maxValue) * padRatio, minPad)
    : Math.max(span * padRatio, minPad)

  minValue -= padding
  maxValue += padding

  if (typeof minClamp === 'number') {
    minValue = Math.max(minClamp, minValue)
  }
  if (typeof maxClamp === 'number') {
    maxValue = Math.min(maxClamp, maxValue)
  }

  if (maxValue <= minValue) {
    const fallbackPad = Math.max(Math.abs(maxValue || minValue || 1) * padRatio, minPad)
    minValue -= fallbackPad
    maxValue += fallbackPad
    if (typeof minClamp === 'number') {
      minValue = Math.max(minClamp, minValue)
    }
    if (typeof maxClamp === 'number') {
      maxValue = Math.min(maxClamp, maxValue)
    }
  }

  return [minValue, maxValue]
}

function buildUnitMetricDomain(values: Array<number | null | undefined>): [number, number] | undefined {
  const numericValues = values
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    .sort((a, b) => a - b)
  if (numericValues.length === 0) {
    return undefined
  }

  const minValue = numericValues[0]
  const maxValue = numericValues[numericValues.length - 1]
  const highScoreBand = minValue >= 0.9
  const span = Math.max(maxValue - minValue, 0)
  const minVisibleSpan = highScoreBand ? 0.003 : 0.03
  const pad = Math.max(span * 0.25, highScoreBand ? 0.0005 : 0.005)
  let lower = minValue - pad
  let upper = maxValue + pad

  if (highScoreBand) {
    lower = Math.max(0.9, lower)
    upper = Math.max(upper, maxValue >= 0.995 ? 1.001 : 1)
  }

  if (upper - lower < minVisibleSpan) {
    const center = (upper + lower) / 2
    lower = center - minVisibleSpan / 2
    upper = center + minVisibleSpan / 2
  }

  lower = Math.max(0, lower)
  upper = Math.min(highScoreBand ? 1.001 : 1, upper)

  if (upper <= lower) {
    upper = Math.min(1.001, lower + minVisibleSpan)
  }

  return [lower, upper]
}

function normalizeToDomain(value: number | null | undefined, domain: [number, number]): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }
  const span = domain[1] - domain[0]
  if (span <= 0) {
    return 0.5
  }
  return (value - domain[0]) / span
}

function formatUnitDomainTick(value: number, domain: [number, number]) {
  const actual = domain[0] + value * (domain[1] - domain[0])
  return actual.toFixed(actual >= 0.99 ? 4 : 3)
}

function buildChartHoverSnapshot(axisLabel: string, state?: ChartMouseState | null): ChartHoverSnapshot | null {
  const entries = state?.activePayload?.filter(item => item.value !== null && item.value !== undefined) ?? []
  if (!entries.length) {
    return null
  }

  return {
    label: formatChartTooltipLabel(axisLabel, state?.activeLabel),
    rows: entries.map(item => ({
      color: item.color ?? '#38bdf8',
      key: String(item.dataKey ?? item.name ?? 'value'),
      label: String(item.name ?? item.dataKey ?? 'value'),
      value: formatChartTooltipValue(tooltipActualValue(item)),
    })),
  }
}

function ChartTooltip({ axisLabel }: { axisLabel: string }) {
  return (
    <Tooltip
      {...CHART_TOOLTIP_STYLE}
      allowEscapeViewBox={{ x: true, y: true }}
      wrapperStyle={{ outline: 'none', pointerEvents: 'none', zIndex: 30 }}
      cursor={{ stroke: 'rgba(148,163,184,0.4)', strokeDasharray: '4 4', strokeWidth: 1.2 }}
      content={({ active, label, payload }) => {
        const entries = (payload as ChartTooltipEntry[] | undefined)?.filter(item => item.value !== null && item.value !== undefined) ?? []
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

function ChartHoverPanel({
  snapshot,
  variant = 'default',
}: {
  snapshot: ChartHoverSnapshot | null
  variant?: ChartHoverPanelVariant
}) {
  const isEmphasis = variant === 'emphasis'
  return (
    <div
      className={`pointer-events-none rounded-2xl border backdrop-blur ${
        isEmphasis
          ? 'min-w-[240px] max-w-[300px] border-sky-400/25 bg-slate-950/92 px-4 py-3.5 shadow-[0_24px_44px_rgba(14,165,233,0.16)]'
          : 'min-w-[180px] max-w-[240px] border-slate-700/50 bg-slate-950/88 px-3.5 py-3 shadow-[0_18px_36px_rgba(2,6,23,0.34)]'
      }`}
    >
      <div className={`font-semibold uppercase tracking-[0.16em] ${isEmphasis ? 'text-[10px] text-sky-300/80' : 'text-[11px] text-slate-500'}`}>
        {snapshot ? snapshot.label : isEmphasis ? '当前点' : '悬停数值'}
      </div>
      {snapshot ? (
        <div className={`mt-2 ${isEmphasis ? 'grid gap-2.5' : 'grid gap-2'}`}>
          {snapshot.rows.map(row => (
            <div
              key={row.key}
              className={`grid items-center ${
                isEmphasis
                  ? 'grid-cols-[12px_minmax(0,1fr)_auto] gap-2.5 rounded-xl border px-3 py-2'
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
                className={isEmphasis ? 'h-3 w-3 rounded-full shadow-[0_0_0_3px_rgba(15,23,42,0.92)]' : 'h-2.5 w-2.5 rounded-full'}
                style={{ backgroundColor: row.color }}
              />
              <span className={`truncate ${isEmphasis ? 'text-[12.5px] font-medium text-slate-200' : 'text-[12px] text-slate-300'}`}>{row.label}</span>
              <span className={`mono font-semibold ${isEmphasis ? 'text-[13px] text-white' : 'text-[12px] text-slate-100'}`}>{row.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className={`mt-2 ${isEmphasis ? 'text-[12.5px] text-slate-300' : 'text-[12px] text-slate-400'}`}>将指针移到图表上查看数值。</div>
      )}
    </div>
  )
}

function ChartXAxis({ dataKey, type = 'category', allowDecimals = true }: { dataKey: string; type?: 'category' | 'number'; allowDecimals?: boolean }) {
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

function ChartYAxis({
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
  return <YAxis width={width} domain={domain} tickCount={tickCount} tickFormatter={tickFormatter} allowDataOverflow={allowDataOverflow} tick={AXIS_TICK_STYLE} {...AXIS_STYLE} />
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div>
      <div className="training-panel-title">{title}</div>
      <div className="training-panel-copy">{copy}</div>
    </div>
  )
}

function KpiCard({
  label,
  value,
  note,
  icon,
}: {
  label: string
  value: string
  note: string
  icon: React.ReactNode
}) {
  return (
    <div className="training-kpi">
      <div className="flex items-start justify-between gap-3">
        <span className="training-kpi__label">{label}</span>
        <span className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
          {icon}
        </span>
      </div>
      <div className="training-kpi__value">{value}</div>
      <div className="training-kpi__note">{note}</div>
    </div>
  )
}

function MetaField({
  label,
  value,
  mono = false,
  title,
}: {
  label: string
  value: React.ReactNode
  mono?: boolean
  title?: string
}) {
  return (
    <div>
      <div className="training-label">{label}</div>
      <div
        className={`mt-1 min-w-0 text-[15px] leading-6 ${mono ? 'mono break-all text-slate-200' : 'break-words text-slate-100'}`}
        title={title}
      >
        {value}
      </div>
    </div>
  )
}

function ChartCard({
  title,
  subtitle,
  actions,
  overlay,
  children,
}: {
  title: string
  subtitle: string
  actions?: React.ReactNode
  overlay?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="training-card training-card--chart">
      <div className="card-header justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-700/40 bg-slate-900/35 text-sky-300">
            <BarChart3 size={16} />
          </span>
          <SectionTitle title={title} copy={subtitle} />
        </div>
        {actions}
      </div>
      <div className="relative h-[320px] overflow-visible p-4">
        {overlay ? <div className="absolute right-4 top-4 z-20">{overlay}</div> : null}
        {children}
      </div>
    </div>
  )
}

function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center rounded-2xl border border-slate-700/40 bg-slate-900/25 px-6 text-center text-[14px] text-slate-400">
      {message}
    </div>
  )
}

function CheckpointList({ checkpoints }: { checkpoints: TrainingCheckpointInfo[] }) {
  if (checkpoints.length === 0) {
    return (
      <div className="training-surface text-[14px] text-slate-400">
        当前还没有权重文件。
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {checkpoints.map(item => (
        <div key={item.path} className="training-surface--dense">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="mono truncate text-[14px] font-semibold text-slate-100">{item.name}</div>
              <div className="mt-1 text-[11px] text-slate-500" title={item.path}>
                {shortPath(item.path, 96)}
              </div>
            </div>
            <span className="flex h-7 w-7 items-center justify-center rounded-xl border border-emerald-500/20 bg-emerald-500/8 text-emerald-300">
              <Save size={14} />
            </span>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-[12px] text-slate-400">
            <MetaField label="轮次" value={item.epoch ?? '—'} />
            <MetaField label="大小" value={formatBytes(item.size_bytes)} />
            <MetaField label="时间" value={formatDateTime(item.mtime)} />
          </div>
        </div>
      ))}
    </div>
  )
}

function buildEpochSeries(points: TrainingPoint[]): TrainingPoint[] {
  const lastPointByEpoch = new Map<number, TrainingPoint>()
  for (const point of points) {
    lastPointByEpoch.set(point.epoch, point)
  }
  return Array.from(lastPointByEpoch.values()).sort((a, b) => a.epoch - b.epoch)
}

export default function TrainingJobDetailPage() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()
  const { mutate } = useSWRConfig()
  const [trainingAxisMode, setTrainingAxisMode] = useState<TrainingAxisMode>('step')
  const [isStopping, setIsStopping] = useState(false)
  const [lossHover, setLossHover] = useState<ChartHoverSnapshot | null>(null)
  const [metricHover, setMetricHover] = useState<ChartHoverSnapshot | null>(null)
  const [scenarioHover, setScenarioHover] = useState<ChartHoverSnapshot | null>(null)

  const { data: job, error: jobError } = useSWR<TrainingJobDetail>(
    jobId ? `training-job-${jobId}` : null,
    () => api.getTrainingJob(jobId),
    {
      refreshInterval: current => {
        if (!current) return 2000
        if (current.status === 'starting' || current.status === 'stopping') return 2000
        return ['running', 'evaluating'].includes(current.status) ? 5000 : 0
      },
      revalidateOnFocus: false,
    },
  )

  const refreshInterval = useMemo(() => {
    if (!job) return 2000
    if (job.status === 'starting' || job.status === 'stopping') return 2000
    return ['running', 'evaluating'].includes(job.status) ? 5000 : 0
  }, [job])

  const { data: curves } = useSWR<TrainingCurvesResponse>(
    jobId ? `training-curves-${jobId}` : null,
    () => api.getTrainingCurves(jobId, 2000),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const { data: logs } = useSWR<TrainingLogResponse>(
    jobId ? `training-logs-${jobId}` : null,
    () => api.getTrainingLogs(jobId, 400),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const trainingChart = useMemo(() => {
    const raw = curves?.training_points ?? []
    const epochSeries = curves?.training_epoch_points ?? buildEpochSeries(raw)
    return {
      xKey: trainingAxisMode === 'step' ? 'global_step' : 'epoch',
      data: trainingAxisMode === 'step' ? raw : epochSeries,
      subtitleSuffix: trainingAxisMode === 'step' ? '步骤' : '轮次',
    }
  }, [curves?.training_epoch_points, curves?.training_points, trainingAxisMode])

  const lossDomain = useMemo(
    () =>
      buildNumericDomain(trainingChart.data.map(point => point.avg_loss), {
        minClamp: 0,
        minPad: 0.00005,
        padRatio: 0.06,
        lowerQuantile: 0.08,
        upperQuantile: 0.92,
      }),
    [trainingChart.data],
  )

  const testMetricDomain = useMemo(
    () =>
      buildUnitMetricDomain((curves?.test_points ?? []).flatMap(point => [point.precision, point.recall, point.f1, point.pr_auc])),
    [curves?.test_points],
  )
  const testMetricPlotData = useMemo(() => {
    const domain = testMetricDomain ?? [0, 1]
    return (curves?.test_points ?? []).map(point => ({
      ...point,
      f1_plot: normalizeToDomain(point.f1, domain),
      pr_auc_plot: normalizeToDomain(point.pr_auc, domain),
      precision_plot: normalizeToDomain(point.precision, domain),
      recall_plot: normalizeToDomain(point.recall, domain),
    }))
  }, [curves?.test_points, testMetricDomain])

  const scenarioMetricData = useMemo(() => {
    const scenarioNames = new Set<string>()
    const rows = new Map<number, Record<string, number>>()
    const values: number[] = []

    for (const point of curves?.test_points ?? []) {
      const row = rows.get(point.epoch) ?? { epoch: point.epoch }
      for (const [scenario, metrics] of Object.entries(point.per_scenario)) {
        scenarioNames.add(scenario)
        const rawValue = metrics.f1
        if (rawValue === null || rawValue === undefined) {
          continue
        }
        const value = Number(rawValue)
        if (!Number.isFinite(value)) {
          continue
        }
        row[scenario] = value
        values.push(value)
      }
      rows.set(point.epoch, row)
    }

    const domain = buildUnitMetricDomain(values) ?? [0.9, 1.001]
    const names = Array.from(scenarioNames)
    const data = Array.from(rows.values())
      .sort((a, b) => Number(a.epoch) - Number(b.epoch))
      .map(row => {
        const nextRow: Record<string, number> = { ...row }
        for (const scenario of names) {
          const normalized = normalizeToDomain(row[scenario], domain)
          if (normalized !== undefined) {
            nextRow[`${scenario}__plot`] = normalized
          }
        }
        return nextRow
      })

    return {
      scenarioNames: names,
      data,
      domain,
    }
  }, [curves?.test_points])

  const stopJob = async () => {
    if (isStopping || job?.status === 'stopping') return
    setIsStopping(true)
    try {
      await api.stopTrainingJob(jobId)
      await Promise.all([
        mutate(`training-job-${jobId}`),
        mutate(`training-curves-${jobId}`),
        mutate(`training-logs-${jobId}`),
        mutate('training-jobs'),
        mutate('training-overview'),
        mutate('training-gpus'),
      ])
    } finally {
      setIsStopping(false)
    }
  }

  const deleteJob = async () => {
    if (!job) return
    const ok = window.confirm(`删除历史任务 ${job.name} (${job.job_id})？\n\n会彻底删除任务记录、运行目录、权重、曲线和日志。共享预处理缓存会保留。`)
    if (!ok) return
    await api.deleteTrainingJob(job.job_id)
    await Promise.all([
      mutate('training-jobs'),
      mutate('training-overview'),
      mutate('training-gpus'),
      mutate(`training-job-${job.job_id}`),
      mutate(`training-curves-${job.job_id}`),
      mutate(`training-logs-${job.job_id}`),
    ])
    navigate('/training/jobs')
  }

  if (!jobId) {
    return (
      <div className="training-page">
        <div className="training-page__body">
          <div className="training-surface text-[15px] text-slate-400">缺少训练任务 ID。</div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-5 p-5">
          <section className="training-hero">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
              <div>
                <div className="training-eyebrow">任务详情</div>
                <h1 className="mt-3 text-[1.8rem] font-semibold tracking-tight text-white xl:text-[2.1rem]">
                  {job?.name ?? jobId}
                </h1>
                <div className="mono mt-2 text-[13px] text-slate-500">{jobId}</div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button type="button" className="btn-ghost" onClick={() => navigate('/training/jobs')}>
                  <ArrowLeft size={14} />
                  返回任务列表
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => {
                    mutate(`training-job-${jobId}`)
                    mutate(`training-curves-${jobId}`)
                    mutate(`training-logs-${jobId}`)
                  }}
                >
                  <RefreshCcw size={14} />
                  刷新
                </button>
                {job && ['starting', 'running', 'evaluating', 'stopping'].includes(job.status) ? (
                  <button type="button" className="btn-danger" onClick={stopJob} disabled={isStopping || job.status === 'stopping'}>
                    <PauseCircle size={14} />
                    {isStopping || job.status === 'stopping' ? '\u7ec8\u6b62\u4e2d...' : '\u7ec8\u6b62\u8bad\u7ec3'}
                  </button>
                ) : job ? (
                  <button type="button" className="btn-ghost" onClick={deleteJob}>
                    <Trash2 size={14} />
                    删除任务
                  </button>
                ) : null}
              </div>
            </div>

            {job && (
              <div className="mt-6 training-kpi-grid">
                <KpiCard
                  label="状态"
                  value={statusLabel(job.status)}
                  note={`GPU ${job.gpu_id} / PID ${job.pid ?? '—'}`}
                  icon={<RadioTower size={16} />}
                />
                <KpiCard
                  label="轮次 / 步数"
                  value={`${job.latest_epoch ?? '—'} / ${job.latest_step ?? '—'}`}
                  note={`全局步数 ${job.global_step ?? '—'}`}
                  icon={<BarChart3 size={16} />}
                />
                <KpiCard
                  label="损失"
                  value={formatMetric(job.avg_loss, 6)}
                  note={`${formatMetric(job.steps_per_sec, 2)} 步/秒`}
                  icon={<ActivityIcon />}
                />
                <KpiCard
                  label="最近 F1"
                  value={formatMetric(job.latest_metrics?.f1, 4)}
                  note={`PR-AUC ${formatMetric(job.latest_metrics?.pr_auc, 4)}`}
                  icon={<Save size={16} />}
                />
                <KpiCard
                  label="预计剩余"
                  value={formatDuration(job.eta_seconds)}
                  note={`创建于 ${formatDateTime(job.created_at)}`}
                  icon={<RefreshCcw size={16} />}
                />
              </div>
            )}
          </section>

          {jobError && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              加载训练任务失败：{jobError.message}
            </div>
          )}

          {job && (
            <>
              <div className="grid gap-4 xl:grid-cols-[1.08fr_0.92fr]">
                <section className="training-card">
                  <div className="card-header">
                    <RadioTower size={16} className="text-violet-300" />
                    <SectionTitle title="任务摘要" copy="配置与路径" />
                  </div>
                  <div className="training-card__body">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="training-surface">
                        <div className="training-meta-grid">
                          <MetaField label="任务名称" value={job.name} />
                          <MetaField label="状态" value={<span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>} />
                          <MetaField label="训练数据" value={`${job.simulator.toUpperCase()} / ${job.scenarios.join(', ')}`} />
                          <MetaField label="测试比例" value={formatMetric(job.config.test_ratio, 2)} />
                          <MetaField label="评测间隔" value={`${job.config.eval_interval} 轮`} />
                          <MetaField label="总轮数" value={job.config.epochs === 0 ? '∞' : job.config.epochs} />
                          <MetaField label="保留权重" value={`${job.config.keep_last_epochs ?? 5} 个轮次`} />
                        </div>
                      </div>
                      <div className="training-surface">
                        <div className="training-meta-grid">
                          <MetaField label="训练批大小" value={job.config.batch_size} mono />
                          <MetaField label="测试批大小" value={job.config.test_batch_size} mono />
                          <MetaField label="加载线程" value={job.config.num_workers} mono />
                          <MetaField label="学习率" value={job.config.learning_rate} mono />
                          <MetaField label="权重衰减" value={job.config.weight_decay} mono />
                          <MetaField label="恢复训练" value={job.config.resume_from ? '是' : '否'} />
                          <MetaField label="输入表示" value={job.config.input_representation ?? 'pretrained_embeddings'} mono />
                          <MetaField
                            label="嵌入模型"
                            value={shortPath(job.config.embedding_model || job.config.embedding_tokenizer || '—', 64)}
                            mono
                            title={job.config.embedding_model || job.config.embedding_tokenizer || undefined}
                          />
                        </div>
                      </div>
                      <div className="training-surface md:col-span-2">
                        <div className="grid gap-3 md:grid-cols-2">
                          <MetaField label="运行目录" value={shortPath(job.run_dir, 112)} mono title={job.run_dir} />
                          <MetaField label="日志文件" value={shortPath(job.log_path, 112)} mono title={job.log_path} />
                        </div>
                        {job.error_message && (
                          <div className="mt-4 flex items-start gap-2 rounded-2xl border border-rose-500/20 bg-rose-500/8 px-4 py-3 text-sm text-rose-300">
                            <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
                            <span>{job.error_message}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <section className="training-card min-h-0">
                  <div className="card-header">
                    <Save size={16} className="text-emerald-300" />
                    <SectionTitle title="权重文件" copy="已保存的模型权重" />
                  </div>
                <div className="training-card__body training-scroll list-scroll-lg">
                  <CheckpointList checkpoints={curves?.checkpoints ?? job.checkpoints} />
                </div>
              </section>
            </div>

              <div className="grid gap-4">
                <ChartCard
                  title="训练损失"
                  subtitle={`平均损失 / ${trainingChart.subtitleSuffix}`}
                  overlay={<ChartHoverPanel snapshot={lossHover} />}
                  actions={
                    <div className="training-segmented">
                      <button
                        type="button"
                        className={`training-segmented__button ${trainingAxisMode === 'step' ? 'training-segmented__button--active' : ''}`}
                        onClick={() => setTrainingAxisMode('step')}
                      >
                        步骤
                      </button>
                      <button
                        type="button"
                        className={`training-segmented__button ${trainingAxisMode === 'epoch' ? 'training-segmented__button--active' : ''}`}
                        onClick={() => setTrainingAxisMode('epoch')}
                      >
                        轮次
                      </button>
                    </div>
                  }
                >
                {trainingChart.data.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={trainingChart.data}
                      margin={CHART_MARGIN}
                      onMouseMove={(state: ChartMouseState) => setLossHover(buildChartHoverSnapshot(trainingAxisMode === 'step' ? '步骤' : '轮次', state))}
                      onMouseLeave={() => setLossHover(null)}
                    >
                      <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                      <ChartXAxis dataKey={trainingChart.xKey} type={trainingAxisMode === 'step' ? 'number' : 'category'} allowDecimals={trainingAxisMode === 'step'} />
                      <ChartYAxis
                        domain={lossDomain}
                        tickCount={5}
                        width={52}
                        tickFormatter={(value: number) => value.toFixed(value >= 1 ? 3 : 4)}
                        allowDataOverflow
                      />
                      <ChartTooltip axisLabel={trainingAxisMode === 'step' ? '步骤' : '轮次'} />
                      <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                      <Line
                        type="monotone"
                        dataKey="avg_loss"
                        name="平均损失"
                        stroke="#38bdf8"
                        dot={false}
                        activeDot={buildActiveDot('#38bdf8')}
                        strokeWidth={2.25}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ChartEmpty message="当前还没有训练曲线点。" />
                )}
                </ChartCard>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <ChartCard title="测试指标" subtitle="精确率 / 召回率 / F1 / PR-AUC">
                {curves?.test_points?.length ? (
                  <>
                    <div className="pointer-events-none absolute right-8 top-8 z-20">
                      <ChartHoverPanel snapshot={metricHover} />
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={testMetricPlotData}
                      margin={CHART_MARGIN}
                      onMouseMove={(state: ChartMouseState) => setMetricHover(buildChartHoverSnapshot('轮次', state))}
                      onMouseLeave={() => setMetricHover(null)}
                    >
                      <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                      <ChartXAxis dataKey="epoch" type="number" allowDecimals={false} />
                      <ChartYAxis
                        domain={[0, 1]}
                        tickCount={5}
                        width={56}
                        tickFormatter={(value: number) => formatUnitDomainTick(value, testMetricDomain ?? [0, 1])}
                        allowDataOverflow
                      />
                      <ChartTooltip axisLabel="轮次" />
                      <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                      <Line type="monotone" dataKey="precision_plot" name="精确率" stroke="#38bdf8" dot={false} activeDot={buildActiveDot('#38bdf8')} strokeWidth={2.1} isAnimationActive={false} />
                      <Line type="monotone" dataKey="recall_plot" name="召回率" stroke="#f59e0b" dot={false} activeDot={buildActiveDot('#f59e0b')} strokeWidth={2.1} isAnimationActive={false} />
                      <Line type="monotone" dataKey="f1_plot" name="F1" stroke="#34d399" dot={false} activeDot={buildActiveDot('#34d399')} strokeWidth={2.1} isAnimationActive={false} />
                      <Line type="monotone" dataKey="pr_auc_plot" name="PR-AUC" stroke="#a78bfa" dot={false} activeDot={buildActiveDot('#a78bfa')} strokeWidth={2.1} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  </>
                ) : (
                  <ChartEmpty message="当前还没有测试点，需要等到测试间隔触发。" />
                )}
                </ChartCard>

                <ChartCard title="分场景 F1" subtitle="每个子场景单独观察">
                {scenarioMetricData.scenarioNames.length ? (
                  <>
                    <div className="pointer-events-none absolute right-6 top-6 z-20">
                      <ChartHoverPanel snapshot={scenarioHover} variant="emphasis" />
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={scenarioMetricData.data}
                      margin={CHART_MARGIN}
                      onMouseMove={(state: ChartMouseState) => setScenarioHover(buildChartHoverSnapshot('轮次', state))}
                      onMouseLeave={() => setScenarioHover(null)}
                    >
                      <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                      <ChartXAxis dataKey="epoch" type="number" allowDecimals={false} />
                      <ChartYAxis
                        domain={[0, 1]}
                        tickCount={5}
                        width={56}
                        tickFormatter={(value: number) => formatUnitDomainTick(value, scenarioMetricData.domain)}
                        allowDataOverflow
                      />
                      <ChartTooltip axisLabel="轮次" />
                      <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                      {scenarioMetricData.scenarioNames.map((scenario, index) => {
                        const colors = ['#38bdf8', '#34d399', '#f59e0b', '#f472b6', '#a78bfa', '#fb7185']
                        return (
                          <Line
                            key={scenario}
                            dataKey={`${scenario}__plot`}
                            name={scenario}
                            stroke={colors[index % colors.length]}
                            dot={false}
                            activeDot={{ ...buildActiveDot(colors[index % colors.length]), r: 6 }}
                            strokeWidth={2.6}
                            isAnimationActive={false}
                          />
                        )
                      })}
                    </LineChart>
                    </ResponsiveContainer>
                  </>
                  ) : (
                    <ChartEmpty message="当前还没有分场景测试曲线。" />
                  )}
                </ChartCard>
              </div>

              <section className="training-card min-h-0">
                <div className="card-header">
                  <FileText size={16} className="text-amber-300" />
                  <SectionTitle title="训练日志" copy="最近 400 行输出" />
                </div>
                <div className="training-card__body min-h-0">
                  <div className="grid gap-3 md:grid-cols-[0.22fr_0.78fr]">
                    <div className="training-surface--dense">
                      <div className="training-panel-title">日志摘要</div>
                      <div className="mt-3 grid gap-2 sm:grid-cols-2">
                        <MetaField label="总行数" value={logs?.lines.length ?? 0} mono />
                        <MetaField label="状态" value={statusLabel(job.status)} />
                        <MetaField label="最近 epoch" value={job.latest_epoch ?? '—'} mono />
                        <MetaField label="最近 step" value={job.latest_step ?? '—'} mono />
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-700/40 bg-slate-950/72 p-3.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                          <div className="training-panel-title">终端输出</div>
                          <div className="training-panel-copy">自动刷新最新内容</div>
                        </div>
                        <div className="mono text-[11px] text-slate-500">{shortPath(job.log_path, 48)}</div>
                      </div>
                      <pre className="list-scroll-xl whitespace-pre-wrap break-words rounded-2xl border border-slate-800/70 bg-slate-950/65 px-3.5 py-3 text-[12px] leading-5 text-slate-300">
                        {(logs?.lines ?? []).join('\n') || '暂无日志输出。'}
                      </pre>
                    </div>
                  </div>
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function ActivityIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 12h3l2-5 4 10 2-5h5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
