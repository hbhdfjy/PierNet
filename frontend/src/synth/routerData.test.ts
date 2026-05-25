import { describe, expect, it } from 'vitest'
import {
  buildableRouterScenarios,
  filterRouterScenarioSelection,
  hasRouterBuildSource,
  hasUsableRouterData,
  routerLabelValue,
  routerProgressPercent,
  routerScenarioKey,
} from './routerData'

describe('router data helpers', () => {
  it('uses simulator and scenario as the stable scenario selector', () => {
    expect(routerScenarioKey({ simulator: 'gcam', scenario: 'shared' })).toBe('gcam/shared')
    expect(routerScenarioKey({ simulator: '', scenario: 'shared' })).toBe('unknown/shared')
  })

  it('shows router-only scenarios restored without source samples', () => {
    const item = { source_count: 0, router_count: 12 }

    expect(hasUsableRouterData(item)).toBe(true)
    expect(routerProgressPercent(item)).toBe(100)
  })

  it('only exposes Stage 3-backed scenarios for router builds', () => {
    const sourceBacked = { simulator: 'gcam', scenario: 'shared', source_count: 3, router_count: 6 }
    const routerOnly = { simulator: 'simpeg', scenario: 'legacy', source_count: 0, router_count: 4 }

    expect(hasRouterBuildSource(sourceBacked)).toBe(true)
    expect(hasRouterBuildSource(routerOnly)).toBe(false)
    expect(buildableRouterScenarios([sourceBacked, routerOnly])).toEqual([sourceBacked])
    expect(filterRouterScenarioSelection(new Set(['gcam/shared', 'simpeg/legacy']), [sourceBacked])).toEqual(
      new Set(['gcam/shared']),
    )
  })

  it('keeps the dirty-count guard when source samples are known', () => {
    expect(hasUsableRouterData({ source_count: 10, router_count: 201 })).toBe(false)
    expect(hasUsableRouterData({ source_count: 10, router_count: 200 })).toBe(true)
    expect(routerProgressPercent({ source_count: 10, router_count: 5 })).toBe(50)
  })

  it('drops stale scenario selections after router status refreshes', () => {
    const selected = new Set(['gcam/shared', 'simpeg/shared', 'stale/missing'])
    const next = filterRouterScenarioSelection(selected, [
      {
        simulator: 'gcam',
        scenario: 'shared',
      },
    ])

    expect([...next]).toEqual(['gcam/shared'])
  })

  it('normalizes legacy string router labels', () => {
    expect(routerLabelValue({ label: 1 })).toBe(1)
    expect(routerLabelValue({ label: '1' })).toBe(1)
    expect(routerLabelValue({ label: 0 })).toBe(0)
    expect(routerLabelValue({ label: '0' })).toBe(0)
    expect(routerLabelValue({ label: 'unknown' })).toBeNull()
  })
})
