import { createContext, useContext } from 'react'

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
  const value = parseInt(localStorage.getItem('piern-seed') ?? '42', 10)
  return Number.isFinite(value) ? Math.max(0, value) : 42
}

export function writeStoredSeed(value: number): number {
  const seed = Math.max(0, Math.floor(value))
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('piern-seed', String(seed))
  }
  return seed
}
