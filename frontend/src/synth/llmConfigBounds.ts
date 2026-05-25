export const LLM_TEMPERATURE_MIN = 0
export const LLM_TEMPERATURE_MAX = 2
export const LLM_MAX_TOKENS_MIN = 64
export const LLM_MAX_TOKENS_MAX = 8192

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function normalizeTemperature(value: number, fallback = 1.0): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return clamp(resolved, LLM_TEMPERATURE_MIN, LLM_TEMPERATURE_MAX)
}

export function normalizeMaxTokens(value: number, fallback = 1024): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return clamp(Math.floor(resolved), LLM_MAX_TOKENS_MIN, LLM_MAX_TOKENS_MAX)
}

export function parseMaxTokensInput(value: string, fallback = 1024): number {
  const parsed = Number(value)
  return normalizeMaxTokens(parsed, fallback)
}
