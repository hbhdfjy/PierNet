import type { RouterScenarioInfo } from '../lib/types'

export function routerScenarioKey(item: Pick<RouterScenarioInfo, 'simulator' | 'scenario'>): string {
  return `${item.simulator || 'unknown'}/${item.scenario}`
}

export function filterRouterScenarioSelection(
  selected: Iterable<string>,
  items: Array<Pick<RouterScenarioInfo, 'simulator' | 'scenario'>>,
): Set<string> {
  const available = new Set(items.map(routerScenarioKey))
  return new Set([...selected].filter(item => available.has(item)))
}

export function hasRouterBuildSource(item: Pick<RouterScenarioInfo, 'source_count'>): boolean {
  return Number(item.source_count ?? 0) > 0
}

export function buildableRouterScenarios<T extends Pick<RouterScenarioInfo, 'source_count'>>(items: T[]): T[] {
  return items.filter(hasRouterBuildSource)
}

export function hasUsableRouterData(item: Pick<RouterScenarioInfo, 'router_count' | 'source_count'>): boolean {
  const routerCount = item.router_count ?? 0
  if (routerCount <= 0) return false
  if (item.source_count <= 0) return true
  return routerCount <= item.source_count * 20
}

export function routerProgressPercent(item: Pick<RouterScenarioInfo, 'router_count' | 'source_count'>): number {
  if (!hasUsableRouterData(item)) return 0
  if (item.source_count <= 0) return 100
  return Math.min(100, ((item.router_count ?? 0) / item.source_count) * 100)
}

export function routerLabelValue(item: { label: unknown }): 0 | 1 | null {
  if (item.label === 1 || item.label === '1') return 1
  if (item.label === 0 || item.label === '0') return 0
  return null
}
