type ParsedModflowAnswer = {
  isModflow: boolean
  raw: string
  matrix: number[][] | null
  trendLines: string[]
}

function splitTrendLines(text: string): string[] {
  return text
    .replace(/\s+(?=\d+\.\s*(?:井|#)\d+[:：])/g, '\n')
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
}

function parseModflowAnswer(answer?: string): ParsedModflowAnswer {
  const raw = (answer || '').trim()
  const trigger = 'MODFLOW地下水专家输出：'
  const trendMarker = '中文趋势总结：'
  if (!raw.includes(trigger)) {
    return { isModflow: false, raw, matrix: null, trendLines: [] }
  }

  const body = raw.slice(raw.indexOf(trigger) + trigger.length).trim()
  const trendIndex = body.indexOf(trendMarker)
  const matrixSource = (trendIndex >= 0 ? body.slice(0, trendIndex) : body).trim()
  const trendText = trendIndex >= 0 ? body.slice(trendIndex + trendMarker.length).trim() : ''
  const matrixMatch = matrixSource.match(/\[\[[\s\S]*\]\]/)
  let matrix: number[][] | null = null
  if (matrixMatch) {
    try {
      const parsed = JSON.parse(matrixMatch[0])
      if (
        Array.isArray(parsed) &&
        parsed.every(row => Array.isArray(row) && row.every(value => typeof value === 'number'))
      ) {
        matrix = parsed
      }
    } catch {
      matrix = null
    }
  }

  return {
    isModflow: true,
    raw,
    matrix,
    trendLines: splitTrendLines(trendText),
  }
}

function formatResultNumber(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : '--'
}

function numericSummary(values: number[]): string {
  if (values.length === 0) return ''
  const first = values[0]
  const last = values[values.length - 1]
  let min = first
  let max = first
  values.forEach(value => {
    if (value < min) min = value
    if (value > max) max = value
  })
  const direction =
    Math.abs(last - first) < 1e-6 ? '末端与起始水平接近' : last > first ? '末端高于起始水平' : '末端低于起始水平'
  return `专家模型返回了一组数值预测，共 ${values.length} 个数值点，范围约 ${formatResultNumber(min)} 到 ${formatResultNumber(max)}，${direction}。`
}

function formatPlainModflowAnswer(parsed: ParsedModflowAnswer): string {
  if (!parsed.matrix) return formatConversationalAnswer(parsed.raw)
  const lines = ['已完成 MODFLOW 地下水预测。']
  lines.push('', '预测数值（hydraulic_head）：')
  parsed.matrix.forEach((row, index) => {
    lines.push(`井${index + 1}：${row.map(formatResultNumber).join('，')}`)
  })
  if (parsed.trendLines.length > 0) {
    lines.push('', '中文趋势总结：', ...parsed.trendLines)
  } else {
    const values = parsed.matrix.flat()
    const summary = numericSummary(values)
    if (summary) lines.push('', summary)
  }
  return lines.join('\n')
}

function stripDenseNumericLines(text: string): { text: string; values: number[] } {
  const values: number[] = []
  const numberPattern = /-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi
  const kept = text
    .split('\n')
    .filter(line => {
      const matches = line.match(numberPattern) ?? []
      const trimmed = line.trim()
      const isReadableValueLine = /^第\s*\d+\s*-\s*\d+\s*点[:：]/.test(trimmed)
      const denseArrayLine =
        !isReadableValueLine &&
        matches.length >= 4 &&
        (trimmed.startsWith('[') || trimmed.endsWith(']') || /^[\d\s.,，+\-eE]+[,.，]?$/.test(trimmed))
      if (denseArrayLine) {
        matches.forEach(item => values.push(Number(item)))
        return false
      }
      return true
    })
    .map(line => line.trimEnd())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  return { text: kept, values }
}

function formatConversationalAnswer(answer?: string): string {
  const raw = (answer || '').trim()
  if (!raw) return ''
  const { text, values } = stripDenseNumericLines(raw)
  const normalized = text
    .replace(/^\s*[\w-]*\s*(?:FNO|Expert|专家)?\s*输出[:：]\s*$/i, '')
    .replace(/^\s*[\u4e00-\u9fffA-Za-z0-9_-]*专家输出[:：]\s*$/i, '')
    .trim()
  const summary = numericSummary(values)
  if (normalized && summary) return `${normalized}\n\n${summary}`
  if (normalized) return normalized
  if (summary) return `已完成专家模型预测。\n\n${summary}`
  return raw
}

export function formatAssemblyAnswer(answer?: string): string {
  const parsed = parseModflowAnswer(answer)
  if (!parsed.raw) return '模型没有返回最终答案。'
  return parsed.isModflow ? formatPlainModflowAnswer(parsed) : formatConversationalAnswer(parsed.raw)
}
