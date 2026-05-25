export type IntegerFieldBounds = {
  min?: number
  max?: number
}

export function parseIntegerText(value: string, bounds: IntegerFieldBounds = {}): number | null {
  const trimmed = value.trim()
  if (trimmed === '') return null

  if (!/^[+-]?\d+$/.test(trimmed)) return null

  const parsed = Number(trimmed)
  if (!Number.isInteger(parsed)) return null
  if (bounds.min !== undefined && parsed < bounds.min) return null
  if (bounds.max !== undefined && parsed > bounds.max) return null
  return parsed
}

export function parseIntegerField(value: string, fallback: number, bounds: IntegerFieldBounds = {}): number {
  return parseIntegerText(value, bounds) ?? fallback
}

export function parseOptionalIntegerField(
  value: string,
  fallback: number | null,
  bounds: IntegerFieldBounds = {},
): number | null {
  if (value.trim() === '') return null
  return parseIntegerText(value, bounds) ?? fallback
}
