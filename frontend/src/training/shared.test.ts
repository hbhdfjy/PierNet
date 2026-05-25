import { describe, expect, it } from 'vitest'
import {
  isTrainingJobActive,
  isTrainingJobDeletable,
  isTrainingJobStoppable,
  normalizeTrainingBatchSize,
  normalizeTrainingEvalInterval,
  normalizeTrainingFiniteEpochs,
  normalizeTrainingKeepLastEpochs,
  normalizeTrainingLearningRate,
  normalizeTrainingNumWorkers,
  normalizeTrainingSeed,
  normalizeTrainingTestBatchSize,
  normalizeTrainingTestRatio,
  normalizeTrainingWeightDecay,
  trainingJobRefreshInterval,
  trainingJobDetailPath,
} from './shared'

const activeStatuses = ['queued', 'starting', 'running', 'evaluating', 'stopping'] as const
const cancellableStatuses = ['queued', 'starting', 'running', 'evaluating'] as const
const terminalStatuses = ['done', 'error', 'terminated', 'external_terminated'] as const

describe('training action helpers', () => {
  it('treats queued, running, and stopping jobs as active', () => {
    for (const status of activeStatuses) {
      expect(isTrainingJobActive(status)).toBe(true)
      expect(isTrainingJobDeletable(status)).toBe(false)
    }
  })

  it('allows queued and running jobs to be stopped instead of deleted', () => {
    for (const status of cancellableStatuses) {
      expect(isTrainingJobStoppable(status)).toBe(true)
    }
  })

  it('blocks deletion while a stop is still pending', () => {
    expect(isTrainingJobStoppable('stopping')).toBe(false)
    expect(isTrainingJobDeletable('stopping')).toBe(false)
  })

  it('allows terminal jobs to be deleted', () => {
    for (const status of terminalStatuses) {
      expect(isTrainingJobActive(status)).toBe(false)
      expect(isTrainingJobStoppable(status)).toBe(false)
      expect(isTrainingJobDeletable(status)).toBe(true)
    }
  })

  it('normalizes test ratio to match backend schema bounds', () => {
    expect(normalizeTrainingTestRatio(0)).toBe(0)
    expect(normalizeTrainingTestRatio(0.9)).toBe(0.9)
    expect(normalizeTrainingTestRatio(-1)).toBe(0)
    expect(normalizeTrainingTestRatio(2)).toBe(0.9)
    expect(normalizeTrainingTestRatio(Number.NaN)).toBe(0.1)
  })

  it('normalizes create-job numeric fields to backend schema bounds', () => {
    expect(normalizeTrainingFiniteEpochs(0)).toBe(1)
    expect(normalizeTrainingFiniteEpochs(100001)).toBe(100000)
    expect(normalizeTrainingEvalInterval(0)).toBe(1)
    expect(normalizeTrainingEvalInterval(100001)).toBe(100000)
    expect(normalizeTrainingKeepLastEpochs(-1)).toBe(0)
    expect(normalizeTrainingKeepLastEpochs(201)).toBe(200)
    expect(normalizeTrainingSeed(3_000_000_000)).toBe(2_147_483_647)
    expect(normalizeTrainingBatchSize(1.9)).toBe(1)
    expect(normalizeTrainingBatchSize(9000)).toBe(8192)
    expect(normalizeTrainingTestBatchSize(0)).toBe(1)
    expect(normalizeTrainingTestBatchSize(9000)).toBe(8192)
    expect(normalizeTrainingLearningRate(Number.NaN)).toBe(2e-4)
    expect(normalizeTrainingLearningRate(2)).toBe(1)
    expect(normalizeTrainingWeightDecay(-1)).toBe(0)
    expect(normalizeTrainingWeightDecay(11)).toBe(10)
    expect(normalizeTrainingNumWorkers(-1)).toBe(0)
    expect(normalizeTrainingNumWorkers(129)).toBe(128)
  })

  it('polls queued jobs until the worker starts them', () => {
    expect(trainingJobRefreshInterval()).toBe(2000)
    expect(trainingJobRefreshInterval('queued')).toBe(2000)
    expect(trainingJobRefreshInterval('starting')).toBe(2000)
    expect(trainingJobRefreshInterval('stopping')).toBe(2000)
    expect(trainingJobRefreshInterval('running')).toBe(5000)
    expect(trainingJobRefreshInterval('evaluating')).toBe(5000)
    expect(trainingJobRefreshInterval('done')).toBe(0)
  })

  it('encodes job ids when building detail routes', () => {
    expect(trainingJobDetailPath('train/id?x=1')).toBe('/training/jobs/train%2Fid%3Fx%3D1')
  })
})
