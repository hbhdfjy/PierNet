import type { TrainingPoint } from '../../../lib/types'

export type TrainingAxisMode = 'step' | 'epoch'

export const CHART_MARGIN = { top: 10, right: 12, left: 6, bottom: 0 }
export const LEGEND_STYLE = {
  fontSize: 12,
  color: '#94a3b8',
  paddingTop: 10,
}

export type ChartTooltipEntry = {
  color?: string
  dataKey?: string | number
  name?: string
  payload?: Record<string, unknown>
  value?: unknown
}

export type ChartHoverSnapshot = {
  label: string
  x?: number
  y?: number
  rows: Array<{
    color: string
    key: string
    label: string
    value: string
  }>
}

export type ChartMouseState = {
  activeLabel?: unknown
  activePayload?: ChartTooltipEntry[]
  chartX?: number
  chartY?: number
}

type NumericDomainOptions = {
  maxClamp?: number
  minClamp?: number
  minPad?: number
  padRatio?: number
  lowerQuantile?: number
  upperQuantile?: number
}

export function formatChartTooltipValue(value: unknown) {
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

export function formatChartTooltipLabel(axisLabel: string, label: unknown) {
  if (label === null || label === undefined || label === '') {
    return axisLabel
  }
  if (typeof label === 'number') {
    return `${axisLabel}: ${label.toLocaleString('en-US')}`
  }
  return `${axisLabel}: ${String(label)}`
}

export function tooltipActualValue(item: ChartTooltipEntry): unknown {
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

export function buildActiveDot(fill: string) {
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

export function buildNumericDomain(
  values: Array<number | null | undefined>,
  options: NumericDomainOptions = {},
): [number, number] | undefined {
  const { padRatio = 0.08, minPad = 0.0001, minClamp, maxClamp, lowerQuantile = 0, upperQuantile = 1 } = options

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
  const padding = span === 0 ? Math.max(Math.abs(maxValue) * padRatio, minPad) : Math.max(span * padRatio, minPad)

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

export function buildUnitMetricDomain(values: Array<number | null | undefined>): [number, number] | undefined {
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

export function normalizeToDomain(value: number | null | undefined, domain: [number, number]): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }
  const span = domain[1] - domain[0]
  if (span <= 0) {
    return 0.5
  }
  return (value - domain[0]) / span
}

export function formatUnitDomainTick(value: number, domain: [number, number]) {
  const actual = domain[0] + value * (domain[1] - domain[0])
  return actual.toFixed(actual >= 0.99 ? 4 : 3)
}

export function compactPathName(value: string | null | undefined) {
  const trimmed = (value ?? '').trim()
  if (!trimmed) return '—'
  return trimmed.split('/').filter(Boolean).pop() ?? trimmed
}

export function inputRepresentationLabel(value: string | null | undefined) {
  if (!value || value === 'pretrained_embeddings') {
    return '预训练嵌入'
  }
  return value
}

export function buildChartHoverSnapshot(axisLabel: string, state?: ChartMouseState | null): ChartHoverSnapshot | null {
  const entries = state?.activePayload?.filter(item => item.value !== null && item.value !== undefined) ?? []
  if (!entries.length) {
    return null
  }

  return {
    label: formatChartTooltipLabel(axisLabel, state?.activeLabel),
    x: typeof state?.chartX === 'number' ? state.chartX : undefined,
    y: typeof state?.chartY === 'number' ? state.chartY : undefined,
    rows: entries.map(item => ({
      color: item.color ?? '#38bdf8',
      key: String(item.dataKey ?? item.name ?? 'value'),
      label: String(item.name ?? item.dataKey ?? 'value'),
      value: formatChartTooltipValue(tooltipActualValue(item)),
    })),
  }
}

export function buildEpochSeries(points: TrainingPoint[]): TrainingPoint[] {
  const lastPointByEpoch = new Map<number, TrainingPoint>()
  for (const point of points) {
    lastPointByEpoch.set(point.epoch, point)
  }
  return Array.from(lastPointByEpoch.values()).sort((a, b) => a.epoch - b.epoch)
}
