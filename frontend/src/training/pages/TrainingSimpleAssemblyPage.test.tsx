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
  },
}))

const mockApi = api as unknown as {
  getAssemblyStatus: Mock
  getTrainingJobs: Mock
  loadAssemblyModels: Mock
  unloadAssemblyModels: Mock
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
      {
        model_id: 'missing-uploaded-expert',
        name: 'missing_uploaded_expert',
        simulator: 'expert_model',
        domain: 'custom',
        input_dim: 4,
        output_dim: 4,
        runtime: 'python',
        status: 'active',
        path: '/missing/source',
        trained: true,
        assembly_enabled: true,
        data_generation_enabled: true,
        exists: false,
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
      config: { simple_pipeline_enabled: true },
      text2comp_model_path: '/artifacts/text2comp_models/train-ready/final_model.pt',
      uploaded_expert_name: 'uploaded_expert',
      uploaded_expert_input_dim: 4,
    },
  ]
}

function profilePayload() {
  return {
    model_id: 'modflow-demo',
    name: 'MODFLOW demo profile',
    description: 'registered profile',
    executor: 'modflow_profile',
    simulator: 'modflow',
    root: '/profiles/modflow-demo',
    manifest_path: '/profiles/modflow-demo/artifacts/manifest.json',
    llm_path: '/models/qwen',
    router_path: '/profiles/modflow-demo/artifacts/router_modflow.pt',
    text2comp_path: '/profiles/modflow-demo/artifacts/text2comp_modflow.pt',
    expert_path: '/profiles/modflow-demo/artifacts/expert_modflow_dnn.pt',
    trained: true,
    chat_enabled: true,
    demo_prompt_label: 'MODFLOW 演示',
    demo_prompt: '请预测 MODFLOW 地下水任务，K_mean=42，Q_pumping=-2600。',
    missing_paths: [],
  }
}

function diffSorpProfilePayload() {
  return {
    model_id: 'diff_sorp_demo',
    name: 'Piern diff-sorp 对话拼装模型',
    description: 'registered diff-sorp profile',
    executor: 'fno_profile',
    simulator: 'diff_sorp',
    root: '/root/data/PierNet',
    manifest_path: null,
    llm_path: '/root/eb-public/huggingface-models/Qwen/Qwen3-4B-Instruct-2507',
    router_path: '/root/data/PierNet/artifacts/token_router/0124_pde_router_best_model.pt',
    text2comp_path:
      '/root/data/PierNet/artifacts/text2comp_models/legacy/0103_raw_1d_diff-sorp_text2computation_best_model_7e-6.pt',
    expert_path: '/root/data/PierNet/artifacts/fno_models/legacy/1D_diff-sorp_NA_NA_FNO_2_1.pt',
    feature_dim: 128,
    output_shape: [64],
    trained: true,
    chat_enabled: true,
    force_split: true,
    demo_prompt_label: 'diff-sorp 标准样例',
    demo_prompt:
      '请通过分析两帧归一化输入数据，基于传质动力学的理论框架，预测吸附系统在下一时间步的状态演变。\n[0.99644, 0.93487, 0.86565, 0.79567, 0.72738, 0.66152, 0.59858, 0.53898, 0.48310, 0.43119]',
    missing_paths: [],
  }
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

    await screen.findByLabelText('简洁训练任务')
    expect(screen.queryByText(/burgers/)).toBeNull()
    const sourceSelect = screen.getByLabelText('简洁训练任务') as HTMLSelectElement
    fireEvent.change(sourceSelect, { target: { value: 'job:train-ready' } })
    await waitFor(() => expect(sourceSelect.value).toBe('job:train-ready'))
    await waitFor(() => expect(screen.queryAllByText(/Uploaded Expert/).length).toBeGreaterThan(0))
    expect(screen.queryByText(/missing_uploaded_expert/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /加载训练组合/ }))

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

  it('filters FNO experts that do not match the selected Text2Comp output dimension', async () => {
    mockApi.getAssemblyStatus.mockResolvedValue({
      ...statusPayload(),
      fno_experts: [
        {
          name: 'fno-128',
          simulator: 'diff_sorp',
          input_dim: 128,
          output_shape: [64],
          path: '/artifacts/fno-128.pt',
          trained: true,
          description: '128 dim FNO',
        },
        {
          name: 'fno-4',
          simulator: 'custom',
          input_dim: 4,
          output_shape: [4],
          path: '/artifacts/fno-4.pt',
          trained: true,
          description: '4 dim FNO',
        },
      ],
    })

    renderPage()

    await screen.findByLabelText('简洁训练任务')
    fireEvent.click(screen.getByRole('button', { name: /简洁训练任务/ }))
    fireEvent.click(screen.getByRole('button', { name: /FNO Expert/ }))
    const fnoSelect = screen.getByLabelText('FNO Expert') as HTMLSelectElement

    expect(Array.from(fnoSelect.options).map(option => option.textContent)).toEqual(['fno-4'])
    fireEvent.click(screen.getByRole('button', { name: /加载训练组合/ }))

    await waitFor(() => expect(mockApi.loadAssemblyModels).toHaveBeenCalled())
    expect(mockApi.loadAssemblyModels.mock.calls[0][0]).toMatchObject({
      expert_executor: 'fno',
      fno_path: '/artifacts/fno-4.pt',
    })
    expect(mockApi.loadAssemblyModels.mock.calls[0][0].fno_path).not.toBe('/artifacts/fno-128.pt')
  })

  it('does not expose non-simple jobs even when they have Text2Comp artifacts', async () => {
    mockApi.getTrainingJobs.mockResolvedValue([
      ...trainingJobsPayload(),
      {
        ...trainingJobsPayload()[0],
        job_id: 'complex-with-text2comp',
        name: 'Complex training with Text2Comp',
        config: {
          ...trainingJobsPayload()[0].config,
          simple_pipeline_enabled: false,
        },
        text2comp_model_path: '/artifacts/text2comp_models/complex/final_model.pt',
      },
    ])

    renderPage()

    await screen.findByLabelText('简洁训练任务')
    fireEvent.click(screen.getByRole('button', { name: /简洁训练任务/ }))
    const sourceSelect = screen.getByLabelText('简洁训练任务') as HTMLSelectElement
    const optionTexts = Array.from(sourceSelect.options).map(option => option.textContent ?? '')

    expect(optionTexts.some(text => text.includes('Ready training job'))).toBe(true)
    expect(optionTexts.some(text => text.includes('Complex training with Text2Comp'))).toBe(false)
  })

  it('does not expose registered diff-sorp as a simple training task', async () => {
    mockApi.getAssemblyStatus.mockResolvedValue({
      ...statusPayload(),
      assembly_profiles: [diffSorpProfilePayload()],
    })

    renderPage()

    await screen.findByLabelText('完整拼装模型')
    fireEvent.click(screen.getByRole('button', { name: /简洁训练任务/ }))

    const sourceSelect = (await screen.findByLabelText('简洁训练任务')) as HTMLSelectElement
    const optionTexts = Array.from(sourceSelect.options).map(option => option.textContent ?? '')
    expect(optionTexts.some(text => text.includes('diff-sorp'))).toBe(false)
    expect(optionTexts.some(text => text.includes('Ready training job'))).toBe(true)
  })

  it('loads diff-sorp from registered assembly profiles', async () => {
    mockApi.getAssemblyStatus.mockResolvedValue({
      ...statusPayload(),
      assembly_profiles: [profilePayload(), diffSorpProfilePayload()],
      loaded_models: {
        ...statusPayload().loaded_models,
        assembly_profile: {
          loaded: true,
          model_id: 'modflow-demo',
          name: 'MODFLOW demo profile',
          executor: 'modflow_profile',
        },
        llm: { loaded: true, path: '/models/qwen' },
      },
    })

    renderPage()

    const profileSelect = (await screen.findByLabelText('完整拼装模型')) as HTMLSelectElement
    fireEvent.change(profileSelect, { target: { value: 'diff_sorp_demo' } })
    await waitFor(() => expect(profileSelect.value).toBe('diff_sorp_demo'))
    fireEvent.click(screen.getByRole('button', { name: /一键加载/ }))

    await waitFor(() => expect(mockApi.loadAssemblyModels).toHaveBeenCalled())
    expect(mockApi.loadAssemblyModels.mock.calls[0][0]).toMatchObject({
      assembly_profile_id: 'diff_sorp_demo',
      llm_path: '/root/eb-public/huggingface-models/Qwen/Qwen3-4B-Instruct-2507',
      llm_gpu_id: 0,
      force_split: true,
    })
  })

  it('marks training-chain cards loaded only when the selected paths match', async () => {
    mockApi.getAssemblyStatus.mockResolvedValue({
      ...statusPayload(),
      loaded_models: {
        ...statusPayload().loaded_models,
        llm: { loaded: true, path: '/models/other-qwen' },
        router: { loaded: true, path: '/runs/other/router_final.pt' },
        text2comp: { loaded: true, paths: ['/artifacts/text2comp_models/other/final_model.pt'] },
        fno: { loaded: true, paths: ['/artifacts/fno-other.pt'] },
        uploaded_expert: { loaded: true, model_id: 'other-uploaded-expert', executor: 'uploaded' },
      },
    })

    renderPage()

    await screen.findByLabelText('简洁训练任务')
    fireEvent.click(screen.getByRole('button', { name: /简洁训练任务/ }))

    expect(screen.getAllByText(/· 待加载$/)).toHaveLength(4)
    expect(screen.queryByText(/· 已加载$/)).toBeNull()
  })

  it('keeps model conversation controls out of the assembly page', async () => {
    mockApi.getAssemblyStatus.mockResolvedValue({
      ...statusPayload(),
      loaded_models: {
        ...statusPayload().loaded_models,
        llm: { loaded: true, path: '/models/qwen' },
      },
    })

    renderPage()

    await screen.findByLabelText('简洁训练任务')
    fireEvent.click(screen.getByRole('button', { name: /简洁训练任务/ }))

    expect(screen.queryByRole('button', { name: /运行测试/ })).toBeNull()
    expect(screen.queryByLabelText('测试输入')).toBeNull()
    expect(screen.queryByText('模型回复')).toBeNull()
    expect(screen.getByRole('button', { name: /加载训练组合/ })).toBeTruthy()
  })
})
