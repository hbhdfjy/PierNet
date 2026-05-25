import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import { parseMaxTokensInput } from '../llmConfigBounds'
import LLMConfig from './LLMConfig'

vi.mock('../../lib/api', () => ({
  api: {
    getLLMConfig: vi.fn(),
    saveLLMConfig: vi.fn(),
    testLLMConfig: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getLLMConfig: Mock
  saveLLMConfig: Mock
  testLLMConfig: Mock
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <LLMConfig />
    </SWRConfig>,
  )
}

describe('LLMConfig', () => {
  it('parses complete max token number input without parseInt truncation', () => {
    expect(parseMaxTokensInput('1e3')).toBe(1000)
    expect(parseMaxTokensInput('12abc', 2048)).toBe(2048)
    expect(parseMaxTokensInput('999999')).toBe(8192)
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getLLMConfig.mockResolvedValue({
      provider: 'siliconflow',
      model: '',
      base_url: '',
      api_key_masked: '',
      has_api_key: false,
      temperature: 1,
      max_tokens: 1024,
      thinking: 'disabled',
    })
    mockApi.saveLLMConfig.mockResolvedValue(undefined)
    mockApi.testLLMConfig.mockResolvedValue({ ok: true, message: 'ok', response_preview: 'ok' })
  })

  it('normalizes max tokens before saving and testing', async () => {
    renderPage()

    await waitFor(() => expect(mockApi.getLLMConfig).toHaveBeenCalled())
    fireEvent.change(screen.getByPlaceholderText('deepseek-ai/DeepSeek-V3'), { target: { value: 'Qwen/test' } })
    fireEvent.change(screen.getByLabelText('Max Tokens'), { target: { value: '999999' } })
    fireEvent.click(screen.getByRole('button', { name: /保存并测试/ }))

    await waitFor(() => expect(mockApi.saveLLMConfig).toHaveBeenCalled())
    expect(mockApi.saveLLMConfig.mock.calls[0][0]).toMatchObject({ max_tokens: 8192 })
    expect(mockApi.testLLMConfig.mock.calls[0][0]).toMatchObject({ max_tokens: 8192, api_key: '' })
  })

  it('requires a model before saving or testing', async () => {
    renderPage()

    await waitFor(() => expect(mockApi.getLLMConfig).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: /保存并测试/ }))

    await waitFor(() => expect(screen.getByText('请填写模型名称')).toBeTruthy())
    expect(mockApi.saveLLMConfig).not.toHaveBeenCalled()
    expect(mockApi.testLLMConfig).not.toHaveBeenCalled()

    fireEvent.click(screen.getByTitle('仅测试，不保存'))
    await waitFor(() => expect(screen.getByText('请填写模型名称')).toBeTruthy())
    expect(mockApi.testLLMConfig).not.toHaveBeenCalled()
  })
})
