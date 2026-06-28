import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import type { TrainingJobSummary } from '../../lib/types'
import TrainingSimpleTasksPage from './TrainingSimpleTasksPage'

vi.mock('../../lib/api', () => ({
  api: {
    getTrainingJobs: vi.fn(),
    stopTrainingJob: vi.fn(),
    deleteTrainingJob: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getTrainingJobs: Mock
  stopTrainingJob: Mock
  deleteTrainingJob: Mock
}

function job(overrides: Partial<TrainingJobSummary> = {}): TrainingJobSummary {
  return {
    job_id: 'job-simple',
    name: 'Simple training',
    status: 'done',
    simulator: 'modflow',
    scenarios: ['coastal'],
    gpu_id: 0,
    created_at: 10,
    started_at: 11,
    ended_at: 12,
    pid: null,
    artifact_root: '/artifacts',
    run_dir: '/artifacts/runs/job-simple',
    log_path: '/logs/job-simple.log',
    config: {
      epochs: 0,
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
      auto_stop_enabled: true,
      auto_stop_metric: 'f1',
      auto_stop_threshold: 0.98,
      auto_stop_min_epochs: 1,
      simple_pipeline_enabled: true,
      simple_text2comp_epochs: 1,
      simple_text2comp_max_samples: 1024,
    },
    latest_epoch: null,
    latest_step: null,
    steps_per_epoch: null,
    global_step: null,
    avg_loss: null,
    steps_per_sec: null,
    eta_seconds: null,
    latest_test_epoch: null,
    latest_metrics: null,
    error_message: null,
    stop_requested: false,
    stop_requested_at: null,
    exit_reason: null,
    pipeline_stage: 'done',
    router_status: 'done',
    text2comp_job_id: 'text2comp-child',
    text2comp_status: 'done',
    text2comp_run_dir: '/text2comp/job',
    text2comp_model_path: '/text2comp/job/final_model.pt',
    text2comp_dataset_path: '/data/text2comp/job.jsonl',
    text2comp_error_message: null,
    uploaded_expert_id: 'expert-a',
    uploaded_expert_name: 'expert_a',
    uploaded_expert_input_dim: 4,
    ...overrides,
  }
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <TrainingSimpleTasksPage />
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingSimpleTasksPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.stopTrainingJob.mockResolvedValue({})
    mockApi.deleteTrainingJob.mockResolvedValue(undefined)
  })

  it('only shows simple pipeline training jobs', async () => {
    mockApi.getTrainingJobs.mockResolvedValue([
      job({ job_id: 'simple-1', name: 'Simple A', config: { ...job().config, simple_pipeline_enabled: true } }),
      job({
        job_id: 'complex-1',
        name: 'Complex Router Training',
        config: { ...job().config, simple_pipeline_enabled: false },
      }),
    ])

    renderPage()

    await waitFor(() => expect(screen.getByText('Simple A')).toBeTruthy())
    expect(screen.queryByText('Complex Router Training')).toBeNull()
    expect(screen.getByText('总任务 1')).toBeTruthy()
  })

  it('keeps stop actions scoped to visible simple jobs', async () => {
    mockApi.getTrainingJobs.mockResolvedValue([
      job({ job_id: 'simple-running', name: 'Simple Running', status: 'running', created_at: 20 }),
      job({
        job_id: 'complex-running',
        name: 'Complex Running',
        status: 'running',
        created_at: 30,
        config: { ...job().config, simple_pipeline_enabled: false },
      }),
    ])

    renderPage()

    await waitFor(() => expect(screen.getByText('Simple Running')).toBeTruthy())
    expect(screen.queryByText('Complex Running')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /终止/ }))

    await waitFor(() => expect(mockApi.stopTrainingJob).toHaveBeenCalledWith('simple-running'))
  })
})
