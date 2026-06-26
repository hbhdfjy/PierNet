import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import type { TrainingCurvesResponse, TrainingJobDetail, TrainingLogResponse } from '../../lib/types'
import TrainingJobDetailPage from './TrainingJobDetailPage'

vi.mock('../../lib/api', () => ({
  api: {
    getTrainingJob: vi.fn(),
    getTrainingCurves: vi.fn(),
    getTrainingLogs: vi.fn(),
    stopTrainingJob: vi.fn(),
    deleteTrainingJob: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getTrainingJob: Mock
  getTrainingCurves: Mock
  getTrainingLogs: Mock
  stopTrainingJob: Mock
  deleteTrainingJob: Mock
}

const baseJob: TrainingJobDetail = {
  job_id: 'train-test',
  name: 'train-test',
  status: 'running',
  simulator: 'modflow',
  scenarios: ['coastal'],
  gpu_id: 0,
  created_at: 1,
  started_at: 2,
  ended_at: null,
  pid: 123,
  artifact_root: '/tmp/artifacts',
  run_dir: '/tmp/artifacts/runs/train-test',
  log_path: '/tmp/runlogs/train-test.log',
  config: {
    epochs: 1,
    eval_interval: 1,
    keep_last_epochs: 5,
    seed: 42,
    batch_size: 2,
    test_batch_size: 2,
    learning_rate: 0.0002,
    weight_decay: 0.01,
    num_workers: 0,
    prepare_workers: 0,
    test_ratio: 0.1,
    max_train_samples: null,
    max_test_samples: null,
    resume_from: null,
    input_representation: 'pretrained_embeddings',
    embedding_model: '/models/qwen',
    embedding_tokenizer: '/models/qwen',
    auto_stop_enabled: false,
    auto_stop_metric: 'f1',
    auto_stop_threshold: 0.98,
    auto_stop_min_epochs: 1,
    simple_pipeline_enabled: false,
    simple_text2comp_epochs: null,
    simple_text2comp_max_samples: null,
  },
  command: [],
  checkpoints: [],
  prepared_name: 'prepared',
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
}

const emptyCurves: TrainingCurvesResponse = {
  job_id: 'train-test',
  training_points: [],
  training_epoch_points: [],
  test_points: [],
  checkpoints: [],
}

const emptyLogs: TrainingLogResponse = {
  job_id: 'train-test',
  lines: [],
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter
        initialEntries={['/training/jobs/train-test']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/training/jobs/:jobId" element={<TrainingJobDetailPage />} />
        </Routes>
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingJobDetailPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getTrainingJob.mockResolvedValue(baseJob)
    mockApi.getTrainingCurves.mockResolvedValue(emptyCurves)
    mockApi.getTrainingLogs.mockResolvedValue(emptyLogs)
    mockApi.stopTrainingJob.mockResolvedValue(baseJob)
    mockApi.deleteTrainingJob.mockResolvedValue(undefined)
  })

  it('shows stop API failures instead of dropping the rejection', async () => {
    mockApi.stopTrainingJob.mockRejectedValue(new Error('training job has no stop file'))

    renderPage()

    await screen.findByRole('button', { name: /终止训练/ })
    fireEvent.click(screen.getByRole('button', { name: /终止训练/ }))

    await waitFor(() => expect(mockApi.stopTrainingJob).toHaveBeenCalledWith('train-test'))
    expect(await screen.findByText(/终止失败：training job has no stop file/)).toBeTruthy()
  })
})
