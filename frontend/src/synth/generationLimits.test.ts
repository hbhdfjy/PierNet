import { describe, expect, it } from 'vitest'
import {
  SYNTH_SAMPLE_COUNT_MAX,
  SYNTH_WORKERS_MAX,
  normalizeSynthSampleCount,
  normalizeSynthWorkers,
} from './generationLimits'

describe('synth generation limits', () => {
  it('normalizes sample counts to backend schema bounds', () => {
    expect(normalizeSynthSampleCount(0)).toBe(1)
    expect(normalizeSynthSampleCount(1.9)).toBe(1)
    expect(normalizeSynthSampleCount(Number.NaN)).toBe(1)
    expect(normalizeSynthSampleCount(2_000_000)).toBe(SYNTH_SAMPLE_COUNT_MAX)
  })

  it('normalizes worker counts to backend schema bounds', () => {
    expect(normalizeSynthWorkers(0)).toBe(1)
    expect(normalizeSynthWorkers(1.9)).toBe(1)
    expect(normalizeSynthWorkers(Number.NaN, 8)).toBe(8)
    expect(normalizeSynthWorkers(128)).toBe(SYNTH_WORKERS_MAX)
  })
})
