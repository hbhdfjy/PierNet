import { describe, expect, it } from 'vitest'
import { normalizeTemplateCount, normalizeTemplateProbability, templateScenarioSimulatorMap } from './templateData'

describe('template data helpers', () => {
  it('normalizes template counts to backend-supported bounds', () => {
    expect(normalizeTemplateCount(25)).toBe(25)
    expect(normalizeTemplateCount(1.9)).toBe(1)
    expect(normalizeTemplateCount(0)).toBe(1)
    expect(normalizeTemplateCount(10001)).toBe(10000)
    expect(normalizeTemplateCount(Number.NaN, 50)).toBe(50)
  })

  it('normalizes template probabilities to backend-supported bounds', () => {
    expect(normalizeTemplateProbability(0.75)).toBe(0.75)
    expect(normalizeTemplateProbability(-0.1)).toBe(0)
    expect(normalizeTemplateProbability(1.5)).toBe(1)
    expect(normalizeTemplateProbability(Number.NaN, 0.25)).toBe(0.25)
  })

  it('maps unique template scenario names to their simulator', () => {
    expect(
      templateScenarioSimulatorMap({
        modflow: [{ name: 'coastal', simulator: 'modflow' }],
        simpeg: [{ name: 'magnetic', simulator: 'simpeg' }],
      }),
    ).toEqual({ coastal: 'modflow', magnetic: 'simpeg' })
  })

  it('marks duplicate scenario names across simulators as unknown instead of mislabelling them', () => {
    expect(
      templateScenarioSimulatorMap({
        modflow: [{ name: 'shared', simulator: 'modflow' }],
        simpeg: [{ name: 'shared', simulator: 'simpeg' }],
      }),
    ).toEqual({ shared: 'unknown' })
  })
})
