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
  const outputInfo = meta.output_info

  // row-level 降采样（modflow, power_transient）：只有1个 output_info 条目
  // blocks[0] 就是 ts_obs，形状 [n_sel_channels][n_time_points]
  if (obs.channel_indices !== null || outputInfo.length === 1) {
    const matrix = blocks[0]
    if (!matrix || matrix.length === 0) return null

    const unit = outputInfo[0]?.unit ?? ''
    const nameZh = outputInfo[0]?.name_zh ?? outputInfo[0]?.name ?? 'output'
    const chanIndices = obs.channel_indices ?? matrix.map((_, i) => i)

    const labels = chanIndices.map((idx) => unit ? `${nameZh}[${idx + 1}] (${unit})` : `${nameZh}[${idx + 1}]`)

    return { channels: matrix, labels, unit }
  }

  // output_info 级别降采样（power_flow, gcam, simpeg）：
  // blocks 可能有多个，每个对应一个 output_info 条目
  const selected = obs.selected_output_names
  const channels: number[][] = []
  const labels: string[] = []
  const units: string[] = []

  for (let i = 0; i < selected.length; i++) {
    const info = outputInfo.find((o) => o.name === selected[i]) ?? outputInfo[i]
    const block = blocks[i]
    if (!block) continue

    // block 形状：[n_rows][n_time_points]，n_rows 由 slice 决定
    const [s, e] = info?.slice ?? [0, null]
    const n_rows = e !== null ? e - s : block.length

    for (let r = 0; r < Math.min(n_rows, block.length); r++) {
      channels.push(block[r])
      const nameZh = info?.name_zh ?? info?.name ?? selected[i]
      const unit = info?.unit ?? ''
      labels.push(n_rows === 1
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
