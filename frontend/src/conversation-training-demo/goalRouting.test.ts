import { describe, expect, it } from 'vitest'

import { goalSimulatorMismatch, inferGoalRoute, recommendedSimulationKey } from './goalRouting'

describe('conversation training goal routing', () => {
  it('recognizes groundwater goals as MODFLOW', () => {
    expect(inferGoalRoute('训练一个地下水水位预测模型')?.simulator).toBe('modflow')
  })

  it('prefers the unified aquifer MODFLOW scenario', () => {
    expect(
      recommendedSimulationKey('预测含水层水头', [
        { simulator: 'gcam', scenario: 'carbon_pricing' },
        { simulator: 'modflow', scenario: 'coastal_seawater' },
        { simulator: 'modflow', scenario: 'unified_aquifer' },
      ]),
    ).toBe('modflow/unified_aquifer')
  })

  it('blocks a goal and dataset domain mismatch', () => {
    expect(goalSimulatorMismatch('训练地下水模型', 'gcam')).toContain('请改用 modflow 数据')
    expect(goalSimulatorMismatch('训练地下水模型', 'modflow')).toBeNull()
    expect(goalSimulatorMismatch('训练地下水模型', 'MODFLOW')).toBeNull()
    expect(goalSimulatorMismatch('训练一个预测模型', 'gcam')).toBeNull()
  })
})
