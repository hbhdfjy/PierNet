import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import JobMonitorPanel from './JobMonitorPanel'

describe('JobMonitorPanel', () => {
  it('shows stop failures without leaving an unhandled action', async () => {
    const onStop = vi.fn().mockRejectedValue(new Error('backend refused stop'))

    render(
      <JobMonitorPanel
        status="running"
        logs={[]}
        progress={{}}
        stats={{ elapsed_sec: 0, samples_per_sec: 0 }}
        autoScroll
        onAutoScrollChange={vi.fn()}
        onStop={onStop}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /终止/ }))

    await waitFor(() => expect(onStop).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('backend refused stop')).toBeTruthy()
  })
})
