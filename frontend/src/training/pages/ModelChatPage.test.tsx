import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import ModelChatPage from './ModelChatPage'

vi.mock('../../lib/api', () => ({
  api: {
    getAssemblyStatus: vi.fn(),
    loadAssemblyModels: vi.fn(),
    testAssembly: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getAssemblyStatus: Mock
  loadAssemblyModels: Mock
  testAssembly: Mock
}

function statusPayload(loadedModelId: 'modflow-demo' | 'diff-sorp-demo' | null = 'modflow-demo') {
  const profiles = [
    {
      model_id: 'modflow-demo',
      name: 'MODFLOW 对话模型',
      executor: 'modflow_profile',
      simulator: 'modflow',
      root: '/profiles/modflow-demo',
      llm_path: '/models/qwen',
      router_path: '/profiles/router.pt',
      text2comp_path: '/profiles/text2comp.pt',
      expert_path: '/profiles/expert.pt',
      trained: true,
      chat_enabled: true,
      demo_prompt: '请预测 MODFLOW 地下水任务，K_mean=42。',
    },
    {
      model_id: 'diff-sorp-demo',
      name: 'diff-sorp 对话模型',
      executor: 'fno_profile',
      simulator: 'diff_sorp',
      root: '/profiles/diff-sorp-demo',
      llm_path: '/models/qwen',
      router_path: '/profiles/diff-sorp-router.pt',
      text2comp_path: '/profiles/diff-sorp-text2comp.pt',
      expert_path: '/profiles/diff-sorp-fno.pt',
      trained: true,
      chat_enabled: true,
      demo_prompt: '请预测 diff-sorp 下一时间步。',
    },
  ]
  const loadedProfile = profiles.find(profile => profile.model_id === loadedModelId)
  return {
    llms: [{ name: 'Qwen', path: '/models/qwen', size: '1 GB', downloaded: true }],
    assembly_profiles: profiles,
    routers: [],
    text2comps: [],
    fno_experts: [],
    custom_experts: [],
    gpus: [],
    loaded_models: {
      assembly_profile: loadedProfile
        ? {
            loaded: true,
            model_id: loadedProfile.model_id,
            name: loadedProfile.name,
            executor: loadedProfile.executor,
          }
        : { loaded: false },
      llm: loadedProfile ? { loaded: true, path: '/models/qwen', gpu_id: 0 } : { loaded: false },
      router: { loaded: false },
      text2comp: { loaded: false },
      fno: { loaded: false },
      uploaded_expert: { loaded: false },
    },
    gpu_available: true,
  }
}

function renderPage(assemblyPath = '/training/simple/assembly') {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ModelChatPage assemblyPath={assemblyPath} />
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('ModelChatPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getAssemblyStatus.mockResolvedValue(statusPayload())
    mockApi.loadAssemblyModels.mockResolvedValue({
      status: 'loaded',
      llm: '/models/qwen',
      llm_gpu_id: 0,
      router_gpu_id: 0,
      message: 'loaded',
      architecture: 'test',
      gpu_status: [],
    })
    mockApi.testAssembly.mockResolvedValue({
      router_prediction: 'modflow',
      router_class_name: 'modflow',
      final_answer:
        'MODFLOW地下水专家输出：\n[[10.759705543518066,13.775484085083008]]\n中文趋势总结：\n1. 井1：末段高于起始水平。',
      first_cot_result: '',
      expert_used: true,
      latency_ms: 38.72,
    })
  })

  it('injects the loaded profile prompt and sends it from the dedicated chat page', async () => {
    renderPage()

    const input = (await screen.findByLabelText('对话输入')) as HTMLTextAreaElement
    await waitFor(() => expect(input.value).toContain('K_mean=42'))
    fireEvent.click(screen.getByRole('button', { name: '发送' }))

    await waitFor(() => expect(mockApi.testAssembly).toHaveBeenCalled())
    expect(mockApi.testAssembly.mock.calls[0][0]).toMatchObject({
      config: {
        main_llm_path: '/models/qwen',
        assembly_profile_id: 'modflow-demo',
        gpu_config: { llm_gpu_ids: [0] },
      },
      test_input: '请预测 MODFLOW 地下水任务，K_mean=42。',
    })
    expect(await screen.findByText(/已完成 MODFLOW 地下水预测/)).toBeTruthy()
    expect(screen.getByText(/井1：10\.7597，13\.7755/)).toBeTruthy()
    expect(screen.getByText('Router：modflow')).toBeTruthy()
    expect(screen.getByText('Expert 已调用')).toBeTruthy()
  })

  it('directs users to assembly when no complete model is loaded', async () => {
    const status = statusPayload(null)
    status.assembly_profiles = []
    mockApi.getAssemblyStatus.mockResolvedValue(status)

    renderPage()

    const link = await screen.findByRole('link', { name: /前往模型拼装/ })
    expect(link.getAttribute('href')).toBe('/training/simple/assembly')
    expect((screen.getByLabelText('对话输入') as HTMLTextAreaElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '发送' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('injects the loaded uploaded expert prompt for a manual GCAM chain', async () => {
    const status = statusPayload(null)
    const manualStatus = {
      ...status,
      custom_experts: [
        {
          model_id: 'gcam-v2',
          name: 'GCAM 统一专家',
          simulator: 'gcam',
          domain: 'energy_climate',
          input_dim: 18,
          output_dim: 80,
          runtime: 'python',
          status: 'active',
          path: '/experts/gcam-v2',
          file_name: 'gcam-v2.zip',
          created_at: 1,
          file_size_bytes: 1,
          interface: 'predict',
          interface_version: 3,
          demo_prompt: '请使用 GCAM 专家完成 carbon_pricing 情景预测。',
        },
      ],
      loaded_models: {
        assembly_profile: { loaded: false },
        llm: { loaded: true, path: '/models/qwen', gpu_id: 0 },
        router: { loaded: true, path: '/models/router.pt', gpu_id: 0 },
        text2comp: { loaded: true, paths: ['/models/gcam-text2comp.pt'], gpu_id: 0 },
        fno: { loaded: false },
        uploaded_expert: { loaded: true, model_id: 'gcam-v2', path: '/experts/gcam-v2' },
      },
    }
    mockApi.getAssemblyStatus.mockResolvedValue(manualStatus)

    renderPage()

    const input = (await screen.findByLabelText('对话输入')) as HTMLTextAreaElement
    await waitFor(() => expect(input.value).toContain('carbon_pricing'))
    expect(screen.getByRole('heading', { name: 'GCAM 能源-气候模型' })).toBeTruthy()
    const modelPicker = screen.getByLabelText('已部署模型') as HTMLSelectElement
    expect(modelPicker.selectedOptions[0]?.textContent).toBe('GCAM 能源-气候模型')
  })

  it('lists every deployed model and switches the active profile from the chat page', async () => {
    let status = statusPayload()
    mockApi.getAssemblyStatus.mockImplementation(() => Promise.resolve(status))
    mockApi.loadAssemblyModels.mockImplementation(async () => {
      status = statusPayload('diff-sorp-demo')
      return {
        status: 'loaded',
        llm: '/models/qwen',
        llm_gpu_id: 0,
        router_gpu_id: 0,
        message: 'loaded',
        architecture: 'test',
        gpu_status: [],
      }
    })

    renderPage()

    const select = (await screen.findByLabelText('已部署模型')) as HTMLSelectElement
    expect(Array.from(select.options).map(option => option.textContent)).toEqual([
      'MODFLOW 对话模型',
      'diff-sorp 对话模型',
    ])

    fireEvent.change(select, { target: { value: 'diff-sorp-demo' } })

    await waitFor(() =>
      expect(mockApi.loadAssemblyModels).toHaveBeenCalledWith({
        assembly_profile_id: 'diff-sorp-demo',
        llm_gpu_id: 0,
        router_gpu_id: 0,
        force_split: false,
      }),
    )
    await waitFor(() => expect((screen.getByLabelText('对话输入') as HTMLTextAreaElement).value).toContain('diff-sorp'))
    expect(screen.getByRole('heading', { name: 'diff-sorp 对话模型' })).toBeTruthy()
    expect(screen.getByText('diff-sorp-router.pt')).toBeTruthy()
  })
})
