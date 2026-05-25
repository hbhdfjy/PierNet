import { describe, expect, it } from 'vitest'
import {
  LIVE_JOB_STATUSES,
  currentJobIdFromNewestFirst,
  isLiveJobStatus,
  isRestartableJobStatus,
  isTerminalJobStatus,
  jobIdsNewestFirst,
  restoredJobStatusMap,
  stopFailureMessage,
} from './useJobMonitor'

describe('synthesis job monitor status helpers', () => {
  it('treats all backend terminal states as terminal and restartable', () => {
    for (const status of ['done', 'error', 'terminated', 'external_terminated'] as const) {
      expect(isTerminalJobStatus(status)).toBe(true)
      expect(isRestartableJobStatus(status)).toBe(true)
    }
  })

  it('treats every backend active state as live and non-restartable', () => {
    for (const status of LIVE_JOB_STATUSES) {
      expect(isLiveJobStatus(status)).toBe(true)
      expect(isTerminalJobStatus(status)).toBe(false)
      expect(isRestartableJobStatus(status)).toBe(false)
    }
    expect(isLiveJobStatus('done')).toBe(false)
  })

  it('does not allow a running job to be restarted', () => {
    expect(isTerminalJobStatus('running')).toBe(false)
    expect(isRestartableJobStatus('running')).toBe(false)
  })

  it('preserves live backend statuses while restoring monitored jobs', () => {
    expect(
      restoredJobStatusMap([
        { id: 'queued-job', status: 'queued' },
        { id: 'eval-job', status: 'evaluating' },
        { id: 'done-job', status: 'done' },
      ]),
    ).toEqual({
      'queued-job': 'queued',
      'eval-job': 'evaluating',
      'done-job': 'done',
    })
  })

  it('keeps every discovered active job in newest-first order', () => {
    expect(
      jobIdsNewestFirst([
        { job_id: 'old-job', started_at: 10 },
        { job_id: 'missing-start-time', started_at: null },
        { job_id: 'new-job', started_at: 20 },
      ]),
    ).toEqual(['new-job', 'old-job', 'missing-start-time'])
  })

  it('uses the newest restored job as the current job id', () => {
    expect(currentJobIdFromNewestFirst(['new-job', 'old-job'])).toBe('new-job')
    expect(currentJobIdFromNewestFirst([])).toBeNull()
  })

  it('summarizes partial stop failures by job id', () => {
    expect(
      stopFailureMessage([
        { id: 'stopped-job', ok: true },
        { id: 'failed-job', ok: false, error: new Error('backend refused stop') },
      ]),
    ).toBe('failed-job: backend refused stop')
    expect(stopFailureMessage([{ id: 'stopped-job', ok: true }])).toBeNull()
  })
})
