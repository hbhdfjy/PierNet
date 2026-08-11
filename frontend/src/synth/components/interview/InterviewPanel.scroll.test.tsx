import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../../lib/api'
import InterviewPanel from './InterviewPanel'

vi.mock('../../../lib/api', () => ({
  api: {
    getRegistry: vi.fn(),
    startInterview: vi.fn(),
    cancelInterview: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getRegistry: Mock
  startInterview: Mock
  cancelInterview: Mock
}

function renderPanel() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <InterviewPanel onRegistryUpdate={vi.fn()} />
    </SWRConfig>,
  )
}

describe('InterviewPanel confirmation editor', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Element.prototype.scrollIntoView = vi.fn()
    mockApi.getRegistry.mockResolvedValue({})
    mockApi.cancelInterview.mockResolvedValue(undefined)
    mockApi.startInterview.mockResolvedValue({
      session_id: 'interview-1',
      step: 3,
      question: '请确认输出通道结构',
      extracted: {
        output_info: [
          { name: 'output_a', description: 'A', unit: 'm', slice: [0, 1] },
          { name: 'output_b', description: 'B', unit: 'm', slice: [1, 2] },
          { name: 'output_c', description: 'C', unit: 'm', slice: [2, 3] },
        ],
      },
      extraction_uncertain: false,
      needs_confirmation: true,
      done: false,
    })
  })

  it('keeps the long editor scrollable while confirmation actions remain outside it', async () => {
    renderPanel()

    fireEvent.change(screen.getByPlaceholderText('如 modflow、simpeg、fenics'), {
      target: { value: 'mechanics' },
    })
    fireEvent.click(screen.getByRole('button', { name: '开始注册仿真器' }))

    const scrollRegion = await screen.findByTestId('interview-extraction-scroll')
    expect(scrollRegion.className).toContain('interview-extraction-scroll')
    expect(scrollRegion.textContent).toContain('output_info — 3 个通道组')

    const confirmButton = await screen.findByRole('button', { name: '确认，下一步' })
    expect(scrollRegion.contains(confirmButton)).toBe(false)
    await waitFor(() => expect(mockApi.startInterview).toHaveBeenCalledTimes(1))
  })
})
