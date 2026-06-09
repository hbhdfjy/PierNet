import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import { api } from '../../lib/api'
import ExpertModelManager from './ExpertModelManager'

vi.mock('../../lib/api', () => ({
  api: {
    listExpertModels: vi.fn(),
    updateExpertModel: vi.fn(),
    validateExpertModel: vi.fn(),
    deleteExpertModel: vi.fn(),
  },
}))

const mockApi = api as unknown as {
  listExpertModels: Mock
  updateExpertModel: Mock
  validateExpertModel: Mock
  deleteExpertModel: Mock
}

const model = {
  model_id: 'expert-123',
  name: 'uploaded_expert',
  status: 'active',
  runtime: 'python',
  file_name: 'adapter.py',
  path: '/data/expert_models/files/expert-123/source',
  created_at: 1,
  validated_at: 2,
  file_size_bytes: 128,
  package_type: 'zip',
  entrypoint: 'adapter.py',
  callable: 'predict',
  checksum: 'a'.repeat(64),
  domain: 'custom',
  simulator: 'diff_sorp',
  input_dim: 128,
  output_dim: 2,
  assembly_enabled: true,
  data_generation_enabled: true,
  interface: 'def predict(inputs: list[float]) -> float | list[float]',
  interface_version: 3,
  exists: true,
}

describe('ExpertModelManager', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.listExpertModels.mockResolvedValue({
      interface: model.interface,
      interface_version: 3,
      constraints: {},
      example_source: '',
      max_model_bytes: 1024,
      max_input_dim: 4096,
      max_input_points: 1000000,
      models: [model],
    })
  })

  it('renders registered uploaded expert details', async () => {
    render(<ExpertModelManager />)

    await waitFor(() => expect(mockApi.listExpertModels).toHaveBeenCalled())
    expect(screen.getAllByText('uploaded_expert').length).toBeGreaterThan(0)
    expect(screen.getByText('expert-123')).toBeTruthy()
    expect(screen.getByText('128 维')).toBeTruthy()
    expect(screen.getByText('diff_sorp')).toBeTruthy()
    expect(screen.getByText('adapter.py :: predict')).toBeTruthy()
  })
})
