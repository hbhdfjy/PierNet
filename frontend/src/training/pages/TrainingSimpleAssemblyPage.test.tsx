import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import TrainingSimpleAssemblyPage from './TrainingSimpleAssemblyPage'

vi.mock('../../lib/api', () => ({
  api: {
    getAssemblyStatus: vi.fn(),
    getTrainingJobs: vi.fn(),
    loadAssemblyModels: vi.fn(),
    unloadAssemblyModels: vi.fn(),
    testAssembly: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getAssemblyStatus: Mock
  getTrainingJobs: Mock
  loadAssemblyModels: Mock
  unloadAssemblyModels: Mock
  testAssembly: Mock
}

function statusPayload() {
  return {
    llms: [{ name: 'Qwen3-0.6B', path: '/models/qwen', size: '1 GB', downloaded: true }],
    assembly_profiles: [],
    routers: [
      {
        name: 'registry-router',
        path: '/artifacts/router.pt',
        num_classes: 2,
        class_names: ['normal', 'gcam'],
        description: 'router',
        trained: true,
      },
    ],
    text2comps: [
      {
        name: 'registry-text2comp',
        simulator: 'gcam',
        output_dim: 4,
        path: '/artifacts/text2comp.pt',
        domain: 'PDE',
        description: 'text2comp',
        trained: true,
      },
    ],
    fno_experts: [],
    custom_experts: [
      {
        model_id: 'uploaded-expert-4',
        name: 'uploaded_expert',
        simulator: 'expert_model',
        domain: 'custom',
        input_dim: 4,
        output_dim: 4,
        runtime: 'python',
        status: 'active',
        path: '/data/expert_models/files/uploaded-expert-4/source',
        trained: true,
        assembly_enabled: true,
        data_generation_enabled: true,
      },
    ],
    gpus: [
      {
        index: 0,
        name: 'GPU 0',
        memory_used_mb: 0,
        memory_free_mb: 1000,
        memory_total_mb: 1000,
        available: true,
      },
    ],
    loaded_models: {
      assembly_profile: { loaded: false },
      llm: { loaded: false },
      router: { loaded: false },
      text2comp: { loaded: false },
      fno: { loaded: false },
      uploaded_expert: { loaded: false, executor: null },
    },
    gpu_available: true,
  }
}

function trainingJobsPayload() {
  return [
    {
      job_id: 'train-ready',
      name: 'Ready training job',
      simulator: 'gcam',
      status: 'done',
      scenarios: ['carbon_pricing'],
      created_at: 10,
      updated_at: 20,
      ended_at: 30,
      progress: 1,
      gpu_id: 0,
      run_dir: '/runs/train-ready',
      artifact_root: '/artifacts/token_router/gcam',
      command: [],
      checkpoints: [],
      latest_metrics: { f1: 0.91 },
      text2comp_model_path: '/artifacts/text2comp_models/train-ready/final_model.pt',
      uploaded_expert_name: 'uploaded_expert',
      uploaded_expert_input_dim: 4,
    },
  ]
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <TrainingSimpleAssemblyPage />
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingSimpleAssemblyPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getAssemblyStatus.mockResolvedValue(statusPayload())
    mockApi.getTrainingJobs.mockResolvedValue(trainingJobsPayload())
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

  it('assembles a completed simple training job with an uploaded expert', async () => {
    renderPage()

    await waitFor(() => expect(screen.getByText(/Ready training job/)).toBeTruthy())
    expect(screen.queryByText(/burgers/)).toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: /一键加载/ })[0])

    await waitFor(() => expect(mockApi.loadAssemblyModels).toHaveBeenCalled())
    expect(mockApi.loadAssemblyModels.mock.calls[0][0]).toMatchObject({
      llm_path: '/models/qwen',
      router_path: '/runs/train-ready/router_final.pt',
      text2comp_path: '/artifacts/text2comp_models/train-ready/final_model.pt',
      expert_executor: 'uploaded',
      uploaded_expert_id: 'uploaded-expert-4',
      fno_path: undefined,
    })
  })
})
