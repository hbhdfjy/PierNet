import { describe, expect, it } from 'vitest'
import { isRestartableJobStatus, isTerminalJobStatus } from './useJobMonitor'

describe('synthesis job monitor status helpers', () => {
  it('treats all backend terminal states as terminal and restartable', () => {
    for (const status of ['done', 'error', 'terminated', 'external_terminated'] as const) {
      expect(isTerminalJobStatus(status)).toBe(true)
      expect(isRestartableJobStatus(status)).toBe(true)
    }
  })

  it('does not allow a running job to be restarted', () => {
    expect(isTerminalJobStatus('running')).toBe(false)
    expect(isRestartableJobStatus('running')).toBe(false)
  })
})
