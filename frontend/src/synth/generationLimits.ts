export const SYNTH_SAMPLE_COUNT_MIN = 1
export const SYNTH_SAMPLE_COUNT_MAX = 1_000_000
export const SYNTH_WORKERS_MIN = 1
export const SYNTH_WORKERS_MAX = 64

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function normalizeSynthSampleCount(value: number, fallback = SYNTH_SAMPLE_COUNT_MIN): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return clamp(Math.floor(resolved), SYNTH_SAMPLE_COUNT_MIN, SYNTH_SAMPLE_COUNT_MAX)
}

export function normalizeSynthWorkers(value: number, fallback = SYNTH_WORKERS_MIN): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return clamp(Math.floor(resolved), SYNTH_WORKERS_MIN, SYNTH_WORKERS_MAX)
}
