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
