import { beforeEach, describe, expect, it } from 'vitest'
import { SEED_MAX, normalizeSeed, parseSeedInput, readStoredSeed, writeStoredSeed } from './seedContext'

describe('seed context helpers', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('clamps seeds to backend-supported integer bounds', () => {
    expect(normalizeSeed(-1)).toBe(0)
    expect(normalizeSeed(1.9)).toBe(1)
    expect(normalizeSeed(Number.NaN)).toBe(42)
    expect(normalizeSeed(3_000_000_000)).toBe(SEED_MAX)
  })

  it('parses complete numeric seed input without parseInt truncation', () => {
    expect(parseSeedInput('1e3')).toBe(1000)
    expect(parseSeedInput('12abc')).toBeNull()
    expect(parseSeedInput('')).toBeNull()
  })

  it('stores and reads normalized seed values', () => {
    expect(writeStoredSeed(3_000_000_000)).toBe(SEED_MAX)
    expect(localStorage.getItem('PierNet-seed')).toBe(String(SEED_MAX))
    expect(readStoredSeed()).toBe(SEED_MAX)

    localStorage.setItem('PierNet-seed', '12abc')
    expect(readStoredSeed()).toBe(42)
  })
})
