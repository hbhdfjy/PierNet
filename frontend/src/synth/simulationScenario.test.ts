import { describe, expect, it } from 'vitest'
import { simulationScenarioKey } from './simulationScenario'

describe('simulation scenario helpers', () => {
  it('uses simulator and scenario as the stable stage 1 selector', () => {
    expect(simulationScenarioKey({ simulator: 'simpeg', scenario: 'shared' })).toBe('simpeg/shared')
    expect(simulationScenarioKey({ simulator: '', scenario: 'shared' })).toBe('unknown/shared')
  })
})
