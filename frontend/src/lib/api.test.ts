import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

function jsonOk(body: unknown = {}): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }),
  )
}

function emptyOk(): Promise<Response> {
  return Promise.resolve(new Response('', { status: 200 }))
}

function requestUrl(fetchMock: ReturnType<typeof vi.fn>, callIndex: number): URL {
  return new URL(String(fetchMock.mock.calls[callIndex][0]), window.location.origin)
}

function requestInit(fetchMock: ReturnType<typeof vi.fn>, callIndex: number): RequestInit | undefined {
  return fetchMock.mock.calls[callIndex][1] as RequestInit | undefined
}

describe('api client path encoding', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('encodes generation job ids in mutating and streaming URLs', async () => {
    const fetchMock = vi.fn().mockImplementation(() => emptyOk())
    vi.stubGlobal('fetch', fetchMock)

    await api.stopGeneration('job id?x=1')

    expect(fetchMock).toHaveBeenCalledWith('/api/generate/job%20id%3Fx%3D1', { method: 'DELETE' })

    class EventSourceStub {
      url: string

      constructor(url: string) {
        this.url = url
      }
    }
    vi.stubGlobal('EventSource', EventSourceStub)

    const stream = api.openGenerationStream('job id?x=1') as unknown as EventSourceStub

    expect(stream.url).toBe('/api/generate/job%20id%3Fx%3D1/stream')
  })

  it('encodes registry keys with slashes in mutating URLs', async () => {
    const fetchMock = vi.fn().mockImplementation(() => emptyOk())
    vi.stubGlobal('fetch', fetchMock)

    const body = { scenario_description: 'ok' }
    await api.updateRegistryEntry('modflow/coastal?x=1', body)
    await api.deleteRegistryEntry('modflow/coastal?x=1')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/registry/modflow%2Fcoastal%3Fx%3D1',
      expect.objectContaining({ method: 'PUT' }),
    )
    expect(requestInit(fetchMock, 0)?.body).toBe(JSON.stringify(body))
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/registry/modflow%2Fcoastal%3Fx%3D1', { method: 'DELETE' })
  })

  it('surfaces cancel interview API errors', async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify({ detail: { code: 'CANCEL_FAILED', message: 'cannot cancel' } }), {
          status: 500,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.cancelInterview('session id?x=1')).rejects.toMatchObject({
      code: 'CANCEL_FAILED',
      status: 500,
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/interview/session%20id%3Fx%3D1', { method: 'DELETE' })
  })

  it('encodes interview session ids in nested URLs', async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonOk({ ok: true }))
    vi.stubGlobal('fetch', fetchMock)

    await api.sendInterviewMessage('session id?x=1', 'hello')
    await api.confirmInterviewStep('session id?x=1', true)
    await api.cancelInterview('session id?x=1')

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/interview/session%20id%3Fx%3D1/message', expect.any(Object))
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/interview/session%20id%3Fx%3D1/confirm', expect.any(Object))
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/interview/session%20id%3Fx%3D1', { method: 'DELETE' })
  })

  it('encodes template scenario and file catalog asset path segments', async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonOk({ items: [], total: 0 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.getTemplateItems('shared scenario?x=1', 2, 50, 'zh/CN', 'formal?x=1')
    await api.deleteFileCatalogAsset('router_parquet:modflow/shared?x=1')

    const itemsUrl = requestUrl(fetchMock, 0)
    expect(itemsUrl.pathname).toBe('/api/files/templates/shared%20scenario%3Fx%3D1/items')
    expect(itemsUrl.searchParams.get('page')).toBe('2')
    expect(itemsUrl.searchParams.get('page_size')).toBe('50')
    expect(itemsUrl.searchParams.get('language')).toBe('zh/CN')
    expect(itemsUrl.searchParams.get('style')).toBe('formal?x=1')
    const assetUrl = requestUrl(fetchMock, 1)
    expect(assetUrl.pathname).toBe('/api/files/catalog/assets/router_parquet%3Amodflow%2Fshared%3Fx%3D1')
    expect(requestInit(fetchMock, 1)?.method).toBe('DELETE')
  })

  it('encodes upload and router build query parameters', async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonOk({ job_id: 'job', status: 'queued' }))
    vi.stubGlobal('fetch', fetchMock)

    const file = new File(['payload'], 'coastal.hdf5')
    await api.uploadHdf5Data({ simulator: 'modflow/v2', scenario: 'coastal?x=1', overwrite: true, file })
    await api.buildRouterData(7, ['gcam/shared?x=1', 'simpeg/mt3d'], 2, 3)

    const uploadUrl = requestUrl(fetchMock, 0)
    expect(uploadUrl.pathname).toBe('/api/simulation/upload')
    expect(uploadUrl.searchParams.get('simulator')).toBe('modflow/v2')
    expect(uploadUrl.searchParams.get('scenario')).toBe('coastal?x=1')
    expect(uploadUrl.searchParams.get('overwrite')).toBe('true')
    expect(requestInit(fetchMock, 0)?.body).toBe(file)

    const buildUrl = requestUrl(fetchMock, 1)
    expect(buildUrl.pathname).toBe('/api/router/build')
    expect(buildUrl.searchParams.get('seed')).toBe('7')
    expect(buildUrl.searchParams.get('neg_ratio')).toBe('2')
    expect(buildUrl.searchParams.get('max_workers')).toBe('3')
    expect(buildUrl.searchParams.getAll('scenarios')).toEqual(['gcam/shared?x=1', 'simpeg/mt3d'])
    expect(String(fetchMock.mock.calls[1][0])).toContain('scenarios=gcam%2Fshared%3Fx%3D1&scenarios=simpeg%2Fmt3d')
    expect(requestInit(fetchMock, 1)?.method).toBe('POST')
  })

  it('encodes training job ids in detail, mutation, curve, and log URLs', async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonOk({}))
    vi.stubGlobal('fetch', fetchMock)

    await api.getTrainingJob('train id/with?x=1')
    await api.stopTrainingJob('train id/with?x=1')
    await api.deleteTrainingJob('train id/with?x=1')
    await api.getTrainingCurves('train id/with?x=1', 123)
    await api.getTrainingLogs('train id/with?x=1', 45)

    const encodedJob = 'train%20id%2Fwith%3Fx%3D1'
    expect(requestUrl(fetchMock, 0).pathname).toBe('/api/training/jobs/' + encodedJob)
    expect(requestUrl(fetchMock, 1).pathname).toBe('/api/training/jobs/' + encodedJob + '/stop')
    expect(requestUrl(fetchMock, 2).pathname).toBe('/api/training/jobs/' + encodedJob)
    expect(requestUrl(fetchMock, 3).pathname).toBe('/api/training/jobs/' + encodedJob + '/curves')
    expect(requestUrl(fetchMock, 3).searchParams.get('max_points')).toBe('123')
    expect(requestUrl(fetchMock, 4).pathname).toBe('/api/training/jobs/' + encodedJob + '/logs')
    expect(requestUrl(fetchMock, 4).searchParams.get('limit')).toBe('45')
    expect(requestInit(fetchMock, 1)?.method).toBe('POST')
    expect(requestInit(fetchMock, 2)?.method).toBe('DELETE')
  })
})
