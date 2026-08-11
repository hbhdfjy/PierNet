import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SWRConfig } from 'swr'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import type { TrainingDatasetInfo } from '../../lib/types'
import TrainingSimpleJobPage from './TrainingSimpleJobPage'

vi.mock('../../lib/api', () => ({
  api: {
    getSimpleTrainingDatasets: vi.fn(),
    createQuickTrainingJob: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getSimpleTrainingDatasets: Mock
  createQuickTrainingJob: Mock
}

const datasets: TrainingDatasetInfo[] = [
  {
    display_name: 'MODFLOW 地下水',
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

function renderPage(initialEntry = '/') {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <MemoryRouter initialEntries={[initialEntry]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <TrainingSimpleJobPage />
      </MemoryRouter>
    </SWRConfig>,
  )
}

describe('TrainingSimpleJobPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getSimpleTrainingDatasets.mockResolvedValue(datasets)
    mockApi.createQuickTrainingJob.mockResolvedValue({ job_id: 'train-simple' })
  })

  it('submits selected data ranges to the quick training API without an expert model', async () => {
    renderPage()

    await screen.findByRole('button', { name: /modflow/i })
    expect(screen.queryByText(/专家模型/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /开始训练/ }))

    await waitFor(() => expect(mockApi.createQuickTrainingJob).toHaveBeenCalled())
    expect(mockApi.createQuickTrainingJob.mock.calls[0][0]).toMatchObject({
      simulator: 'modflow',
      scenarios: ['coastal', 'basin'],
    })
    expect(mockApi.createQuickTrainingJob.mock.calls[0][0]).not.toHaveProperty('uploaded_expert_id')
  })

  it('blocks quick training when no small scenario is selected', async () => {
    renderPage()

    await screen.findByRole('button', { name: /modflow/i })
    expect(screen.getByText('大场景')).toBeTruthy()
    expect(screen.getByText('小场景')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '清空' }))
    const submitButton = screen.getByRole('button', { name: /开始训练/ })
    expect(submitButton.hasAttribute('disabled')).toBe(true)
    fireEvent.click(submitButton)
    expect(mockApi.createQuickTrainingJob).not.toHaveBeenCalled()
  })

  it('selects a new synthesis dataset by stable id and submits that id', async () => {
    mockApi.getSimpleTrainingDatasets.mockResolvedValue([
      ...datasets,
      {
        dataset_id: 'router-stable-id',
        display_name: '结构预测数据',
        source: 'new_synth',
        simulator: 'mechanics',
        total_count: 400,
        scenarios: [
          {
            dataset_id: 'router-stable-id',
            scenario: 'column_buckling',
            simulator: 'mechanics',
            router_count: 400,
            file_size_bytes: 80,
            mtime: 3,
            path: '/data/new_synth/router.jsonl',
          },
        ],
      },
    ])

    renderPage('/?datasetId=router-stable-id')

    await screen.findByRole('button', { name: /结构预测数据/ })
    fireEvent.click(screen.getByRole('button', { name: /开始训练/ }))

    await waitFor(() => expect(mockApi.createQuickTrainingJob).toHaveBeenCalled())
    expect(mockApi.createQuickTrainingJob.mock.calls[0][0]).toMatchObject({
      dataset_id: 'router-stable-id',
      simulator: 'mechanics',
      scenarios: ['column_buckling'],
    })
  })
})
