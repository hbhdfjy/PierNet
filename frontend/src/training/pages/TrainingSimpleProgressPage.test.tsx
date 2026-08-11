import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import type { TrainingJobDetail } from '../../lib/types'
import TrainingSimpleProgressPage from './TrainingSimpleProgressPage'

vi.mock('../../lib/api', () => ({
  api: {
    getTrainingJob: vi.fn(),
    getTrainingLogs: vi.fn(),
    stopTrainingJob: vi.fn(),
    deleteTrainingJob: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getTrainingJob: Mock
  getTrainingLogs: Mock
  stopTrainingJob: Mock
  deleteTrainingJob: Mock
}

function job(overrides: Partial<TrainingJobDetail> = {}): TrainingJobDetail {
  return {
    job_id: 'train-1',
    name: 'Training job',
    status: 'running',
    simulator: 'modflow',
    scenarios: ['coastal'],
    gpu_id: 0,
    created_at: 10,
    started_at: 11,
    ended_at: null,
    pid: 123,
    artifact_root: '/artifacts',
    run_dir: '/artifacts/runs/train-1',
    log_path: '/logs/train-1.log',
    command: [],
    prepared_name: null,
    config: {
      epochs: 1,
      eval_interval: 1,
      keep_last_epochs: 5,
      seed: 42,
      batch_size: 256,
      test_batch_size: 256,
      learning_rate: 0.0002,
      weight_decay: 0.01,
      num_workers: 8,
      prepare_workers: null,
      test_ratio: 0.1,
      max_train_samples: null,
      max_test_samples: null,
      resume_from: null,
      input_representation: 'pretrained_embeddings',
      embedding_model: '',
      embedding_tokenizer: '',
      auto_stop_enabled: false,
      auto_stop_metric: 'f1',
      auto_stop_threshold: 0.98,
      auto_stop_min_epochs: 1,
      simple_pipeline_enabled: true,
      simple_quality_gate_enabled: true,
      simple_router_min_f1: 0.95,
      simple_text2comp_epochs: 1,
      simple_text2comp_max_samples: 1024,
    },
    latest_epoch: 0,
    latest_step: 1,
    steps_per_epoch: 10,
    global_step: 1,
    avg_loss: null,
    steps_per_sec: null,
    eta_seconds: null,
    latest_test_epoch: null,
    latest_metrics: null,
    error_message: null,
    stop_requested: false,
    stop_requested_at: null,
    exit_reason: null,
    pipeline_stage: 'router',
    router_status: 'running',
    text2comp_job_id: null,
    text2comp_status: null,
    text2comp_run_dir: null,
    text2comp_model_path: null,
    text2comp_dataset_path: null,
    text2comp_error_message: null,
    uploaded_expert_id: 'expert-a',
    uploaded_expert_name: 'expert_a',
    uploaded_expert_input_dim: 4,
    checkpoints: [],
    ...overrides,
  }
}

function renderPage(jobId = 'train-1') {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter
        initialEntries={[`/training/simple/jobs/${jobId}`]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/training/simple/jobs/:jobId" element={<TrainingSimpleProgressPage />} />
        </Routes>
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingSimpleProgressPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getTrainingLogs.mockResolvedValue({ path: '/logs/train-1.log', lines: [] })
  })

  it('shows simple training progress for simple pipeline jobs', async () => {
    mockApi.getTrainingJob.mockResolvedValue(job())

    renderPage()

    await waitFor(() => expect(screen.getByText(/正在训练 Router/)).toBeTruthy())
    expect(screen.getByRole('button', { name: /终止/ })).toBeTruthy()
  })

  it('rejects non-simple jobs and links to the normal training detail page', async () => {
    mockApi.getTrainingJob.mockResolvedValue(
      job({
        job_id: 'complex-1',
        name: 'Complex Router Training',
        config: { ...job().config, simple_pipeline_enabled: false },
      }),
    )

    renderPage('complex-1')

    await waitFor(() => expect(screen.getByText('这不是简洁训练任务')).toBeTruthy())
    expect(screen.queryByRole('button', { name: /终止/ })).toBeNull()
    expect(screen.getByRole('link', { name: /打开普通训练详情/ }).getAttribute('href')).toBe('/training/jobs/complex-1')
  })
})
