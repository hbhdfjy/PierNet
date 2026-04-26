import type { SampleMetadata, ParsedTimeseries } from './types'

/**
 * 从 target 字符串中提取嵌入的时序矩阵。
 *
 * target 格式：
 *   "...模拟结果为 [[10.37, 20.45, ...], [9.93, 18.82, ...], ...]。"
 *   或 power_flow 多输出：
 *   "...bus_voltages: {output_0}, voltage_angles: {output_1}, ..."
 *   已填充后变为嵌入多个 [[...]] 矩阵。
 *
 * 策略：找到第一个 [[...]] 匹配，JSON.parse 之。
 * 对于多输出（output_info 级别），找到所有 [[...]] 块。
 */
export function parseTimeseries(
  target: string,
  meta: SampleMetadata,
): ParsedTimeseries | null {
  // 找到所有 [[...]] 块（用括号深度匹配，不依赖正则）
  const blocks: number[][][] = []
  let i = 0
  while (i < target.length) {
    const start = target.indexOf('[[', i)
    if (start === -1) break
    let depth = 0, end = -1
    for (let j = start; j < target.length; j++) {
      if (target[j] === '[') depth++
      else if (target[j] === ']') { depth--; if (depth === 0) { end = j + 1; break } }
    }
    if (end === -1) break
    try {
      const parsed = JSON.parse(target.slice(start, end)) as number[][]
      blocks.push(parsed)
    } catch { /* ignore malformed */ }
    i = end
  }

  if (blocks.length === 0) return null

  const obs = meta.observation
  const outputInfo = meta.output_info as Array<{ name: string; name_zh?: string; unit?: string; slice?: [number, number | null] }>

  const parseByOutputInfoSlices = (matrix: number[][]): ParsedTimeseries | null => {
    if (outputInfo.length === 0 || !outputInfo.some((info) => info.slice != null)) return null
    const totalRows = outputInfo.reduce((sum, info) => {
      const [s, e] = info.slice ?? [0, null]
      return sum + Math.max(0, (e ?? matrix.length) - s)
    }, 0)
    if (matrix.length !== totalRows) return null

    const channels: number[][] = []
    const labels: string[] = []
    const units: string[] = []
    for (const info of outputInfo) {
      const [s, e] = info.slice ?? [0, null]
      const rows = matrix.slice(s, e ?? matrix.length)
      const nameZh = info.name_zh ?? info.name ?? 'output'
      const unit = info.unit ?? ''
      for (let r = 0; r < rows.length; r++) {
        channels.push(rows[r])
        labels.push(rows.length === 1
          ? (unit ? `${nameZh} (${unit})` : nameZh)
          : (unit ? `${nameZh}[${r + 1}] (${unit})` : `${nameZh}[${r + 1}]`)
        )
        units.push(unit)
      }
    }
    if (channels.length === 0) return null
    return { channels, labels, unit: units[0] ?? '' }
  }

  // 单矩阵路径：target 里只有 1 个 [[...]] 块（modflow, transient, power_flow 等）
  // 直接用 output_info 的 slice 信息为每段通道生成标签
  if (blocks.length === 1 || obs.channel_indices !== null) {
    const matrix = blocks[0]
    if (!matrix || matrix.length === 0) return null

    // 有明确通道索引：按索引生成标签（modflow/transient）
    const slicedByOutput = parseByOutputInfoSlices(matrix)
    if (slicedByOutput) return slicedByOutput

    if (obs.channel_indices !== null) {
      const chanIndices = obs.channel_indices
      const unit = outputInfo[0]?.unit ?? ''
      const nameZh = outputInfo[0]?.name_zh ?? outputInfo[0]?.name ?? 'output'
      const labels = chanIndices.map((idx) => unit ? `${nameZh}[${idx + 1}] (${unit})` : `${nameZh}[${idx + 1}]`)
      return { channels: matrix, labels, unit }
    }

    // 无通道索引：按 output_info 的 slice 分段标注（power_flow: V/θ/P 三段）
    if (outputInfo.length > 1) {
      const channels: number[][] = []
      const labels: string[] = []
      const units: string[] = []
      for (const info of outputInfo) {
        const [s, e] = info.slice ?? [0, null]
        const rows = matrix.slice(s, e ?? matrix.length)
        const nameZh = info.name_zh ?? info.name ?? 'output'
        const unit = info.unit ?? ''
        for (let r = 0; r < rows.length; r++) {
          channels.push(rows[r])
          labels.push(rows.length === 1
            ? (unit ? `${nameZh} (${unit})` : nameZh)
            : (unit ? `${nameZh}[${r + 1}] (${unit})` : `${nameZh}[${r + 1}]`)
          )
          units.push(unit)
        }
      }
      if (channels.length === 0) return null
      return { channels, labels, unit: units[0] ?? '' }
    }

    // 单 output_info 条目（simpeg 等）
    const unit = outputInfo[0]?.unit ?? ''
    const nameZh = outputInfo[0]?.name_zh ?? outputInfo[0]?.name ?? 'output'
    const labels = matrix.map((_, r) => matrix.length === 1
      ? (unit ? `${nameZh} (${unit})` : nameZh)
      : (unit ? `${nameZh}[${r + 1}] (${unit})` : `${nameZh}[${r + 1}]`)
    )
    return { channels: matrix, labels, unit }
  }

  // 多矩阵路径：target 里有多个 [[...]] 块
  // 情况A：每个块行数 == output_info 对应 slice 的行数（gcam：每块1行，对应1个变量）
  // 情况B：每个块都是完整时序（power_flow：4个占位符各填了完整43行），取第一个块按 slice 分段
  const firstBlock = blocks[0]
  const hasSlice = outputInfo.length > 0 && outputInfo[0].slice != null

  // 判断是否为 power_flow 类型：块数 > output_info 条数，或块的行数等于所有 slice 的总行数
  const totalSliceRows = hasSlice
    ? outputInfo.reduce((s, o) => s + ((o.slice?.[1] ?? firstBlock.length) - (o.slice?.[0] ?? 0)), 0)
    : 0
  const isPowerFlowStyle = hasSlice && firstBlock.length === totalSliceRows

  if (isPowerFlowStyle) {
    // 取第一个块，按 output_info slice 分段标注
    const channels: number[][] = []
    const labels: string[] = []
    const units: string[] = []
    for (const info of outputInfo) {
      const [s, e] = info.slice ?? [0, null]
      const rows = firstBlock.slice(s, e ?? firstBlock.length)
      const nameZh = info.name_zh ?? info.name ?? 'output'
      const unit = info.unit ?? ''
      for (let r = 0; r < rows.length; r++) {
        channels.push(rows[r])
        labels.push(rows.length === 1
          ? (unit ? `${nameZh} (${unit})` : nameZh)
          : (unit ? `${nameZh}[${r + 1}] (${unit})` : `${nameZh}[${r + 1}]`)
        )
        units.push(unit)
      }
    }
    if (channels.length === 0) return null
    return { channels, labels, unit: units[0] ?? '' }
  }

  // 每块对应一个 output_info 条目（gcam：5个块，每块1行）
  const channels: number[][] = []
  const labels: string[] = []
  const units: string[] = []

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]
    const info = outputInfo[i]
    if (!block || !info) continue
    const nameZh = info.name_zh ?? info.name ?? `output${i}`
    const unit = info.unit ?? ''
    for (let r = 0; r < block.length; r++) {
      channels.push(block[r])
      labels.push(block.length === 1
        ? (unit ? `${nameZh} (${unit})` : nameZh)
        : (unit ? `${nameZh}[${r + 1}] (${unit})` : `${nameZh}[${r + 1}]`)
      )
      units.push(unit)
    }
  }

  if (channels.length === 0) return null
  return { channels, labels, unit: units[0] ?? '' }
}

/** 将 channels 数据转换为 Recharts 需要的格式 */
export function toRechartsData(
  channels: number[][],
  labels: string[],
): Record<string, number>[] {
  if (channels.length === 0 || channels[0].length === 0) return []
  const n = channels[0].length
  return Array.from({ length: n }, (_, t) => {
    const point: Record<string, number> = { t }
    channels.forEach((ch, i) => {
      point[labels[i]] = ch[t] ?? 0
    })
    return point
  })
}
