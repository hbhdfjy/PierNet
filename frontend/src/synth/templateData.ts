import type { Text2CompScenario } from '../lib/types'

export const TEMPLATE_COUNT_MIN = 1
export const TEMPLATE_COUNT_MAX = 10000
export const TEMPLATE_PROBABILITY_MIN = 0
export const TEMPLATE_PROBABILITY_MAX = 1

export function normalizeTemplateCount(value: number, fallback = TEMPLATE_COUNT_MIN): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return Math.min(TEMPLATE_COUNT_MAX, Math.max(TEMPLATE_COUNT_MIN, Math.floor(resolved)))
}

export function normalizeTemplateProbability(value: number, fallback = 0): number {
  const resolved = Number.isFinite(value) ? value : fallback
  return Math.min(TEMPLATE_PROBABILITY_MAX, Math.max(TEMPLATE_PROBABILITY_MIN, resolved))
}

type ScenarioIdentityConfig = Record<string, Array<Pick<Text2CompScenario, 'name' | 'simulator'>>>

export function templateScenarioSimulatorMap(config: ScenarioIdentityConfig | undefined): Record<string, string> {
  const result: Record<string, string> = {}
  if (!config) return result

  for (const items of Object.values(config)) {
    for (const item of items) {
      const scenario = item.name
      const simulator = item.simulator || 'unknown'
      if (result[scenario] && result[scenario] !== simulator) result[scenario] = 'unknown'
      else result[scenario] = simulator
    }
  }

  return result
}
