import type { Text2CompScenario } from '../lib/types'

export function text2compScenarioKey(item: Pick<Text2CompScenario, 'simulator' | 'name'>): string {
  return `${item.simulator || 'unknown'}/${item.name}`
}

export function duplicateText2CompScenarioNames(items: Array<Pick<Text2CompScenario, 'name'>>): Set<string> {
  const counts = new Map<string, number>()
  for (const item of items) counts.set(item.name, (counts.get(item.name) ?? 0) + 1)
  return new Set([...counts.entries()].filter(([, count]) => count > 1).map(([name]) => name))
}

export function selectableText2CompScenarios(
  items: Text2CompScenario[],
  duplicateNames: Set<string>,
  predicate: (item: Text2CompScenario) => boolean,
): Text2CompScenario[] {
  return items.filter(item => !duplicateNames.has(item.name) && predicate(item))
}
