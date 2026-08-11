import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { SWRConfig, useSWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import type { TrainingDatasetInfo } from '../../lib/types'
import TrainingNewJobPage from './TrainingNewJobPage'

vi.mock('../../lib/api', () => ({
  api: {
    getTrainingDatasets: vi.fn(),
    getTrainingGPUs: vi.fn(),
    getTrainingJobs: vi.fn(),
    createTrainingJob: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getTrainingDatasets: Mock
  getTrainingGPUs: Mock
  getTrainingJobs: Mock
  createTrainingJob: Mock
}

const datasets: TrainingDatasetInfo[] = [
  {
    simulator: 'modflow',
    source: 'legacy',
    total_count: 30,
    scenarios: [
      {
        scenario: 'coastal',
        simulator: 'modflow',
        router_count: 10,
        file_size_bytes: 100,
        mtime: 1,
        path: '/data/router/modflow/coastal.parquet',
      },
      {
        scenario: 'basin',
        simulator: 'modflow',
        router_count: 20,
        file_size_bytes: 200,
        mtime: 2,
        path: '/data/router/modflow/basin.parquet',
      },
    ],
  },
]

let refreshDatasets: ((next: TrainingDatasetInfo[]) => Promise<unknown>) | null = null

function DatasetCacheProbe() {
  const { mutate } = useSWRConfig()
  useEffect(() => {
    refreshDatasets = (next: TrainingDatasetInfo[]) => mutate('training-datasets', next, false)
    return () => {
      refreshDatasets = null
    }
  }, [mutate])
  return null
}

function renderPage(initialEntry = '/') {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter initialEntries={[initialEntry]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DatasetCacheProbe />
        <TrainingNewJobPage />
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingNewJobPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getTrainingDatasets.mockResolvedValue(datasets)
    mockApi.getTrainingGPUs.mockResolvedValue([
      {
        index: 0,
        name: 'GPU 0',
        memory_used_mib: 1,
        memory_total_mib: 1024,
        utilization_gpu: 0,
        available: true,
        locked_by_job_id: null,
        reason: null,
      },
    ])
    mockApi.getTrainingJobs.mockResolvedValue([])
    mockApi.createTrainingJob.mockResolvedValue({ job_id: 'train-test' })
  })

  it('submits backend-supported test ratio bounds without silently narrowing them', async () => {
    renderPage()

    await waitFor(() => expect(screen.getByText(/已选择 2 \/ 2/)).toBeTruthy())
    fireEvent.change(screen.getByLabelText('测试集比例'), { target: { value: '0.9' } })
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }))

    await waitFor(() => expect(mockApi.createTrainingJob).toHaveBeenCalled())
    expect(mockApi.createTrainingJob.mock.calls[0][0]).toMatchObject({ test_ratio: 0.9 })

    mockApi.createTrainingJob.mockClear()
    fireEvent.change(screen.getByLabelText('测试集比例'), { target: { value: '0' } })
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }))

    await waitFor(() => expect(mockApi.createTrainingJob).toHaveBeenCalled())
    expect(mockApi.createTrainingJob.mock.calls[0][0]).toMatchObject({ test_ratio: 0 })
  })

  it('allows submitting queued training to a busy visible GPU', async () => {
    mockApi.getTrainingGPUs.mockResolvedValue([
      {
        index: 0,
        name: 'GPU 0',
        memory_used_mib: 900,
        memory_total_mib: 1024,
        utilization_gpu: 80,
        available: false,
        locked_by_job_id: 'train-running',
        reason: 'locked by train-running',
      },
    ])

    renderPage()

    await waitFor(() => expect(screen.getByText(/已选择 2 \/ 2/)).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }))

    await waitFor(() => expect(mockApi.createTrainingJob).toHaveBeenCalled())
    expect(mockApi.createTrainingJob.mock.calls[0][0]).toMatchObject({ gpu_id: 0 })
  })

  it('normalizes numeric training parameters to backend schema bounds before submit', async () => {
    renderPage()

    await waitFor(() => expect(screen.getByText(/已选择 2 \/ 2/)).toBeTruthy())
    fireEvent.click(screen.getByLabelText('无限训练'))
    fireEvent.change(screen.getByLabelText('训练轮数'), { target: { value: '100001' } })
    fireEvent.change(screen.getByLabelText('测试间隔'), { target: { value: '100001' } })
    fireEvent.change(screen.getByLabelText('保留最近权重'), { target: { value: '201' } })
    fireEvent.change(screen.getByLabelText('训练批大小'), { target: { value: '9000' } })
    fireEvent.change(screen.getByLabelText('测试批大小'), { target: { value: '9000' } })
    fireEvent.change(screen.getByLabelText('学习率'), { target: { value: '2' } })
    fireEvent.change(screen.getByLabelText('权重衰减'), { target: { value: '11' } })
    fireEvent.change(screen.getByLabelText('数据加载线程'), { target: { value: '129' } })
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }))

    await waitFor(() => expect(mockApi.createTrainingJob).toHaveBeenCalled())
    expect(mockApi.createTrainingJob.mock.calls[0][0]).toMatchObject({
      epochs: 100000,
      eval_interval: 100000,
      keep_last_epochs: 200,
      batch_size: 8192,
      test_batch_size: 8192,
      learning_rate: 1,
      weight_decay: 10,
      num_workers: 128,
    })
  })

  it('keeps manual scenario selection when datasets refresh for the same simulator', async () => {
    renderPage()

    await waitFor(() => expect(screen.getByText(/已选择 2 \/ 2/)).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /basin/ }))

    await waitFor(() => expect(screen.getByText(/已选择 1 \/ 2/)).toBeTruthy())
    await act(async () => {
      await refreshDatasets?.([
        {
          ...datasets[0],
          scenarios: datasets[0].scenarios.map(item => ({ ...item, mtime: item.mtime + 10 })),
        },
      ])
    })

    await waitFor(() => expect(screen.getByText(/已选择 1 \/ 2/)).toBeTruthy())
  })

  it('opens and submits a new synthesis dataset selected by stable id', async () => {
    mockApi.getTrainingDatasets.mockResolvedValue([
      ...datasets,
      {
        dataset_id: 'router-stable-id',
        display_name: '结构预测数据 · Router',
        source: 'new_synth',
        simulator: 'mechanics',
        total_count: 8,
        scenarios: [
          {
            dataset_id: 'router-stable-id',
            scenario: 'column_buckling',
            simulator: 'mechanics',
            router_count: 8,
            file_size_bytes: 80,
            mtime: 3,
            path: '/data/new_synth/router.jsonl',
          },
        ],
      },
    ])

    renderPage('/?datasetId=router-stable-id')

    await waitFor(() => expect(screen.getByDisplayValue(/结构预测数据/)).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /启动训练/ }))

    await waitFor(() => expect(mockApi.createTrainingJob).toHaveBeenCalled())
    expect(mockApi.createTrainingJob.mock.calls[0][0]).toMatchObject({
      dataset_id: 'router-stable-id',
      simulator: 'mechanics',
      scenarios: ['column_buckling'],
    })
  })
})
