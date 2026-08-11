type SimulatorSource = {
  simulator: string
  scenario?: string
}

export type GoalRoute = {
  simulator: string
  label: string
  keywords: readonly string[]
  preferredScenario?: string
}

const ROUTES: readonly GoalRoute[] = [
  {
    simulator: 'modflow',
    label: '地下水 MODFLOW',
    keywords: ['地下水', '水位', '水头', '含水层', '渗透', '抽水', 'aquifer', 'groundwater', 'modflow'],
    preferredScenario: 'unified_aquifer',
  },
  {
    simulator: 'gcam',
    label: '能源气候 GCAM',
    keywords: ['碳定价', '碳排放', '能源', '气候', 'carbon', 'emission', 'climate', 'gcam'],
    preferredScenario: 'carbon_pricing',
  },
  {
    simulator: 'transient',
    label: '电力暂态',
    keywords: ['电力暂态', '暂态稳定', '故障', '切机', 'transient'],
    preferredScenario: 'ieee14_fault',
  },
  {
    simulator: 'power_flow',
    label: '电力潮流',
    keywords: ['电力潮流', '潮流计算', 'power flow'],
  },
  {
    simulator: 'mechanics_column_buckling',
    label: '柱屈曲力学',
    keywords: ['柱屈曲', '屈曲', 'buckling'],
  },
]

function normalizeSimulator(value: string): string {
  return value.trim().toLowerCase().replace(/-/g, '_')
}

export function inferGoalRoute(goal: string): GoalRoute | null {
  const normalized = goal.trim().toLowerCase()
  if (!normalized) return null
  return ROUTES.find(route => route.keywords.some(keyword => normalized.includes(keyword))) || null
}

export function recommendedSimulationKey(goal: string, sources: SimulatorSource[]): string | null {
  const route = inferGoalRoute(goal)
  if (!route) return null
  const matching = sources.filter(source => normalizeSimulator(source.simulator) === route.simulator)
  if (!matching.length) return null
  const preferred = matching.find(source => source.scenario === route.preferredScenario) || matching[0]
  return preferred.scenario ? `${preferred.simulator}/${preferred.scenario}` : preferred.simulator
}

export function goalSimulatorMismatch(goal: string, actualSimulator: string): string | null {
  const route = inferGoalRoute(goal)
  if (!route || route.simulator === normalizeSimulator(actualSimulator)) return null
  return `训练目标识别为“${route.label}”，但当前数据属于“${actualSimulator}”。请改用 ${route.simulator} 数据，避免训练出错误任务的模型。`
}
