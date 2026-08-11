import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import AssemblyPage from './AssemblyPage'

vi.mock('../../lib/api', () => ({
  api: {
    getAssemblyStatus: vi.fn(),
    getAssemblyGPUs: vi.fn(),
    loadAssemblyModels: vi.fn(),
    unloadAssemblyModels: vi.fn(),
    getDomains: vi.fn(),
    getPrompt: vi.fn(),
    updatePrompt: vi.fn(),
    generatePrompt: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getAssemblyStatus: Mock
  getAssemblyGPUs: Mock
  loadAssemblyModels: Mock
  unloadAssemblyModels: Mock
  getDomains: Mock
  getPrompt: Mock
  updatePrompt: Mock
  generatePrompt: Mock
}

function statusPayload() {
  return {
    llms: [{ name: 'Qwen3-0.6B', path: '/models/qwen', size: '1 GB', downloaded: true }],
    routers: [
      {
        name: 'router',
        path: '/artifacts/router.pt',
        num_classes: 2,
        class_names: ['normal', 'diff_sorp'],
        description: 'router',
        trained: true,
      },
    ],
    text2comps: [
      {
        name: 'text2comp',
        simulator: 'diff_sorp',
        output_dim: 128,
        path: '/artifacts/text2comp.pt',
        trained: true,
      },
    ],
    fno_experts: [],
    custom_experts: [
      {
        model_id: 'expert-128',
        name: 'uploaded_expert',
        simulator: 'diff_sorp',
        domain: 'custom',
        input_dim: 128,
        output_dim: 2,
        runtime: 'python',
        status: 'active',
        path: '/data/expert_models/files/expert-128/source',
        trained: true,
        assembly_enabled: true,
        data_generation_enabled: true,
      },
    ],
    gpus: [
      {
        index: 0,
        name: 'GPU',
        memory_used_mb: 0,
        memory_free_mb: 1000,
        memory_total_mb: 1000,
        available: true,
      },
    ],
    loaded_models: {
      llm: { loaded: false },
      router: { loaded: false },
      text2comp: { loaded: false },
      fno: { loaded: false },
      uploaded_expert: { loaded: false, executor: 'fno' },
    },
    gpu_available: true,
  }
}

function renderPage() {
  return render(
    <MemoryRouter>
      <AssemblyPage />
    </MemoryRouter>,
  )
}

describe('AssemblyPage uploaded experts', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getAssemblyStatus.mockResolvedValue(statusPayload())
    mockApi.getAssemblyGPUs.mockResolvedValue(statusPayload().gpus)
    mockApi.getDomains.mockResolvedValue([])
    mockApi.getPrompt.mockResolvedValue({ piern_system_prompt: '' })
    mockApi.loadAssemblyModels.mockResolvedValue({
      status: 'loaded',
      llm: '/models/qwen',
      llm_gpu_id: 0,
      router_gpu_id: 0,
      message: 'loaded',
      architecture: '',
      gpu_status: [],
    })
  })

  it('loads with uploaded expert executor and model id', async () => {
    renderPage()

    await waitFor(() => expect(mockApi.getAssemblyStatus).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: 'Uploaded Expert' }))
    expect(screen.getAllByText('uploaded_expert').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: /一键加载所有模型/ }))

    await waitFor(() => expect(mockApi.loadAssemblyModels).toHaveBeenCalled())
    expect(mockApi.loadAssemblyModels.mock.calls[0][0]).toMatchObject({
      expert_executor: 'uploaded',
      uploaded_expert_id: 'expert-128',
      text2comp_path: '/artifacts/text2comp.pt',
      fno_path: undefined,
    })
  })

  it('keeps model conversation controls out of the assembly page', async () => {
    renderPage()

    await waitFor(() => expect(mockApi.getAssemblyStatus).toHaveBeenCalled())
    expect(screen.queryByText('推理测试')).toBeNull()
    expect(screen.queryByText('测试结果')).toBeNull()
    expect(screen.queryByRole('button', { name: /开始测试/ })).toBeNull()
  })
})
