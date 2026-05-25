type ScenarioKeyInput = {
  simulator?: string | null
  scenario: string
}

export function simulationScenarioKey(item: ScenarioKeyInput): string {
  return `${item.simulator || 'unknown'}/${item.scenario}`
}
