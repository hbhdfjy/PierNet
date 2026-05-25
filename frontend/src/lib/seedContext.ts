import { createContext, useContext } from 'react'

export const SEED_MIN = 0
export const SEED_MAX = 2_147_483_647

export function normalizeSeed(value: number, fallback = 42): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return Math.min(SEED_MAX, Math.max(SEED_MIN, Math.floor(resolved)))
}

export function parseSeedInput(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) return null

  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed)) return null
  return normalizeSeed(parsed)
}

export interface SeedContextValue {
  seed: number
  setSeed: (v: number) => void
}

export const SeedContext = createContext<SeedContextValue>({
  seed: 42,
  setSeed: () => {},
})

export function useSeed(): SeedContextValue {
  return useContext(SeedContext)
}

export function readStoredSeed(): number {
  if (typeof localStorage === 'undefined') return 42
  return parseSeedInput(localStorage.getItem('PierNet-seed') ?? '42') ?? 42
}

export function writeStoredSeed(value: number): number {
  const seed = normalizeSeed(value)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('PierNet-seed', String(seed))
  }
  return seed
}
