import { describe, expect, it } from 'vitest'
import type { Text2CompJobSummary, TrainingJobSummary } from '../lib/types'
import { buildPipelineGuide } from './pipelineGuide'

const baseInput = {
  h5Count: 2,
  registeredCount: 2,
  templateCount: 40,
  templateScenarioCount: 2,
  totalSamples: 120,
  routerTotal: 240,
  trainingJobs: [] as TrainingJobSummary[],
  text2compJobs: [] as Text2CompJobSummary[],
  assemblyProfileCount: 0,
  assemblyLoaded: false,
  lastInferenceTestAt: null,
}

describe('buildPipelineGuide', () => {
  it('marks the first missing step as current and later steps as blocked', () => {
    const steps = buildPipelineGuide('simple', baseInput)

    expect(steps.slice(0, 5).every(step => step.status === 'complete')).toBe(true)
    expect(steps[5].status).toBe('current')
    expect(steps[6].status).toBe('blocked')
    expect(steps[4].completed).toContain('二分类 Router 数据')
  })

  it('requires both Router and Text2Comp completion in complex mode', () => {
    const routerJob = {
      status: 'done',
      config: { simple_pipeline_enabled: false },
    } as TrainingJobSummary
    const text2compJob = { status: 'done' } as Text2CompJobSummary

    const routerOnly = buildPipelineGuide('complex', {
      ...baseInput,
      trainingJobs: [routerJob],
    })
    expect(routerOnly[5].complete).toBe(false)
    expect(routerOnly[5].missing).toContain('缺少已完成的 Text2Comp 训练任务')

    const complete = buildPipelineGuide('complex', {
      ...baseInput,
      trainingJobs: [routerJob],
      text2compJobs: [text2compJob],
      assemblyProfileCount: 1,
      lastInferenceTestAt: 1_700_000_000,
    })
    expect(complete.every(step => step.status === 'complete')).toBe(true)
  })
})
