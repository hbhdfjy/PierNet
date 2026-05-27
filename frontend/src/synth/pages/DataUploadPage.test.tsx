import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { SWRConfig } from 'swr'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import DataUploadPage from './DataUploadPage'

vi.mock('../../lib/api', () => ({
  api: {
    getSimulationScenarios: vi.fn(),
    listHdf5DataFiles: vi.fn(),
    uploadHdf5Data: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  getSimulationScenarios: Mock
  listHdf5DataFiles: Mock
  uploadHdf5Data: Mock
}

const validation = {
  valid: true,
  path: 'data/modflow/modflow_case.h5',
  file_size_bytes: 4,
  sample_count: 1,
  output_shape: [1, 1],
  params_shape: [1, 1],
  n_params: 1,
  param_names_preview: ['param_0'],
  attrs: {},
  errors: [],
  warnings: [],
}

function renderPage() {
  return render(
    <MemoryRouter>
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
        <DataUploadPage />
      </SWRConfig>
    </MemoryRouter>,
  )
}

describe('DataUploadPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.listHdf5DataFiles.mockResolvedValue([])
    mockApi.uploadHdf5Data.mockResolvedValue({
      ok: true,
      simulator: 'modflow',
      scenario: 'case',
      saved_path: 'data/modflow/modflow_case.h5',
      validation,
    })
    mockApi.getSimulationScenarios.mockResolvedValue([
      {
        simulator: 'modflow',
        scenario: 'case',
        config_path: 'configs/modflow/variants/case.yaml',
        h5_path: 'data/modflow/modflow_case.h5',
        sample_count: 1,
        output_shape: [1, 1],
        file_size_bytes: 4,
      },
    ])
  })

  it('refreshes the shared simulation-scenarios cache after upload', async () => {
    renderPage()

    const file = new File(['hdf5'], 'modflow_case.h5', { type: 'application/octet-stream' })
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => expect(screen.getByDisplayValue('case')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /上传并预检/ }))

    await waitFor(() => {
      expect(mockApi.uploadHdf5Data).toHaveBeenCalledWith({
        simulator: 'modflow',
        scenario: 'case',
        file,
        overwrite: false,
      })
    })
    expect(mockApi.getSimulationScenarios).toHaveBeenCalledWith(true)
  })
})
