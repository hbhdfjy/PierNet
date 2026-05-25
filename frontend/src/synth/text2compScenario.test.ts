import { describe, expect, it } from 'vitest'
import type { Text2CompScenario } from '../lib/types'
import {
  duplicateText2CompScenarioNames,
  selectableText2CompScenarios,
  text2compScenarioKey,
} from './text2compScenario'

function scenario(simulator: string, name: string, hasH5 = true): Text2CompScenario {
  return {
    simulator,
    name,
    h5_file: hasH5 ? `${simulator}_${name}.h5` : null,
    sample_count: hasH5 ? 1 : 0,
    output_shape: null,
    existing_jsonl_count: 0,
    has_jsonl: false,
    has_h5: hasH5,
    registered: true,
  }
}

describe('text2comp scenario helpers', () => {
  it('detects ambiguous scenario names and keeps stable UI keys', () => {
    const items = [scenario('simpeg', 'shared'), scenario('modflow', 'shared'), scenario('gcam', 'unique')]
    const duplicates = duplicateText2CompScenarioNames(items)

    expect(text2compScenarioKey(items[0])).toBe('simpeg/shared')
    expect([...duplicates]).toEqual(['shared'])
    expect(selectableText2CompScenarios(items, duplicates, item => item.has_h5).map(item => item.name)).toEqual([
      'unique',
    ])
  })
})
