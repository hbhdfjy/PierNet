import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../../lib/api'
import type { LogLine, ScenarioProgress, LiveStats, JobStatus, JobStatusSnapshot } from '../../lib/types'

export interface JobMonitorState {
  jobIds: string[]
  jobId: string | null
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  autoScroll: boolean
  setAutoScroll: (v: boolean) => void
  start: (jobId: string, scenarioTotals?: Record<string, number>, initialStatus?: JobStatus) => void
  stop: () => Promise<void>
  reset: () => void
  scenarioDoneCount: number // 每完成一个场景递增，用于触发外部刷新
}

const JOBS_KEY = (stageKey: string) => `PierNet_jobs_${stageKey}`
const EXPECTED_JOB_TYPES: Record<string, string[]> = {
  templates: ['generate_templates'],
  fill: ['fill_samples'],
  simulate: ['simulate'],
  router: ['router'],
}

function expectedJobTypesForStage(stageKey: string): string[] | undefined {
  return EXPECTED_JOB_TYPES[stageKey]
}

// localStorage：关闭标签页/浏览器后仍保留
function loadStoredJobs(stageKey: string): string[] {
  try {
    const raw = localStorage.getItem(JOBS_KEY(stageKey))
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveStoredJobs(stageKey: string, ids: string[]) {
  try {
    if (ids.length > 0) localStorage.setItem(JOBS_KEY(stageKey), JSON.stringify(ids))
    else localStorage.removeItem(JOBS_KEY(stageKey))
  } catch {
    /* ignore */
  }
}

function isExpectedJobForStage(stageKey: string, jobType: string | null | undefined): boolean {
  const expected = expectedJobTypesForStage(stageKey)
  return !expected || (typeof jobType === 'string' && expected.includes(jobType))
}

export const LIVE_JOB_STATUSES = ['queued', 'starting', 'running', 'evaluating', 'stopping'] as const

export function isTerminalJobStatus(status: string | null | undefined): status is JobStatus {
  return status === 'done' || status === 'error' || status === 'terminated' || status === 'external_terminated'
}

export function isLiveJobStatus(status: string | null | undefined): status is JobStatus {
  return LIVE_JOB_STATUSES.some(item => item === status)
}

export function isRestartableJobStatus(status: JobStatus | null | undefined): boolean {
  return !status || status === 'idle' || isTerminalJobStatus(status)
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const next = Number(value)
  return Number.isFinite(next) ? next : fallback
}

export interface RestoredJobStatus {
  id: string
  status: JobStatus
}

export function restoredJobStatusMap(items: RestoredJobStatus[]): Record<string, JobStatus> {
  return Object.fromEntries(items.map(item => [item.id, item.status]))
}

export function jobIdsNewestFirst(jobs: Pick<JobStatusSnapshot, 'job_id' | 'started_at'>[]): string[] {
  return [...jobs].sort((a, b) => Number(b.started_at ?? 0) - Number(a.started_at ?? 0)).map(job => job.job_id)
}

export function currentJobIdFromNewestFirst(jobIds: string[]): string | null {
  return jobIds[0] ?? null
}

export type StopJobResult = { id: string; ok: true } | { id: string; ok: false; error: unknown }

function stopErrorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function stopFailureMessage(results: StopJobResult[]): string | null {
  const failed = results.filter((result): result is Extract<StopJobResult, { ok: false }> => !result.ok)
  if (failed.length === 0) return null
  return failed.map(result => `${result.id}: ${stopErrorText(result.error)}`).join('\n')
}

function normalizeProgress(scenario: string, progress: Partial<ScenarioProgress>): ScenarioProgress {
  const name = String(progress.scenario ?? scenario ?? '').trim()
  return {
    scenario: name,
    done: Math.max(0, toFiniteNumber(progress.done, 0)),
    total: Math.max(0, toFiniteNumber(progress.total, 0)),
  }
}

export function useJobMonitor(stageKey = 'default'): JobMonitorState {
  const [jobIds, setJobIds] = useState<string[]>([])
  const [jobStatuses, setJobStatuses] = useState<Record<string, JobStatus>>({})
  const [logs, setLogs] = useState<LogLine[]>([])
  const [progress, setProgress] = useState<Record<string, ScenarioProgress>>({})
  const [stats, setStats] = useState<LiveStats>({ elapsed_sec: 0, samples_per_sec: 0 })
  const [autoScroll, setAutoScroll] = useState(true)
  const [scenarioDoneCount, setScenarioDoneCount] = useState(0)

  const esMap = useRef<Map<string, EventSource>>(new Map())
  const lastLoggedProgressRef = useRef<Record<string, Record<string, number>>>({})
  const errorCheckInFlightRef = useRef<Set<string>>(new Set())

  const applyBackendSnapshot = useCallback((data: JobStatusSnapshot) => {
    setProgress(prev => {
      const next = { ...prev }
      for (const [scenario, total] of Object.entries(data.scenario_totals ?? {})) {
        const existing = next[scenario]
        next[scenario] = normalizeProgress(scenario, { scenario, done: existing?.done ?? 0, total })
      }
      for (const [scenario, item] of Object.entries(data.progress ?? {})) {
        const normalized = normalizeProgress(scenario, item)
        if (normalized.scenario) next[normalized.scenario] = normalized
      }
      return next
    })
    if (data.stats) {
      setStats({
        elapsed_sec: toFiniteNumber(data.stats.elapsed_sec, 0),
        samples_per_sec: toFiniteNumber(data.stats.samples_per_sec, 0),
      })
    }
  }, [])

  const status: JobStatus = (() => {
    const statuses = Object.values(jobStatuses)
    if (statuses.length === 0) return 'idle'
    if (statuses.some(s => s === 'stopping')) return 'stopping'
    if (statuses.some(s => s === 'evaluating')) return 'evaluating'
    if (statuses.some(s => s === 'running')) return 'running'
    if (statuses.some(s => s === 'starting')) return 'starting'
    if (statuses.some(s => s === 'queued')) return 'queued'
    if (statuses.some(s => s === 'error')) return 'error'
    if (statuses.some(s => s === 'external_terminated')) return 'external_terminated'
    if (statuses.some(s => s === 'terminated')) return 'terminated'
    if (statuses.every(s => s === 'done')) return 'done'
    return 'idle'
  })()

  // 用 ref 记录每个 job 是否已收到终止事件，避免 onerror 与 onmessage 的竞态
  const terminatedRef = useRef<Set<string>>(new Set())

  const connectOne = useCallback(
    (id: string, initialStatus: JobStatus = 'running') => {
      if (esMap.current.has(id)) return
      // 用传入的 initialStatus 初始化，不用 ?? 避免 React 批处理导致旧状态覆盖
      setJobStatuses(prev => {
        if (prev[id] !== undefined) return prev // 已有状态不覆盖（restoreJobs 已预设）
        return { ...prev, [id]: initialStatus }
      })

      const es = api.openGenerationStream(id)

      es.onmessage = e => {
        try {
          const event = JSON.parse(e.data)
          if (event.type === 'init') {
            const totals: Record<string, number> = event.scenario_totals ?? {}
            setProgress(prev => {
              const next = { ...prev }
              for (const [sc, total] of Object.entries(totals)) {
                next[sc] = normalizeProgress(sc, { scenario: sc, done: prev[sc]?.done ?? 0, total: total as number })
              }
              return next
            })
          } else if (event.type === 'log') {
            let shouldAppendLog = true
            if (event.progress) {
              const p = normalizeProgress(String(event.progress.scenario ?? ''), event.progress)
              if (p.scenario) {
                setProgress(prev => ({ ...prev, [p.scenario]: p }))
                const byJob = lastLoggedProgressRef.current[id] ?? {}
                if (byJob[p.scenario] === p.done) {
                  shouldAppendLog = false
                } else {
                  lastLoggedProgressRef.current[id] = { ...byJob, [p.scenario]: p.done }
                }
              }
            }
            if (event.stats) setStats(event.stats)
            if (shouldAppendLog) {
              setLogs(prev => {
                const next = [...prev, { line: event.line, ts: event.ts } as LogLine]
                return next.length > 5000 ? next.slice(-5000) : next
              })
            }
          } else if (event.type === 'queued' || event.type === 'started') {
            setLogs(prev => [
              ...prev,
              { line: event.message ?? (event.type === 'queued' ? '任务已进入队列' : '任务已开始执行'), ts: event.ts },
            ])
            setJobStatuses(prev => ({ ...prev, [id]: event.type === 'queued' ? 'queued' : 'running' }))
          } else if (event.type === 'done') {
            terminatedRef.current.add(id) // 先标记，再 setState，防止 onerror 竞态
            setJobStatuses(prev => ({ ...prev, [id]: 'done' }))
            es.close()
            esMap.current.delete(id)
          } else if (event.type === 'error') {
            terminatedRef.current.add(id)
            setJobStatuses(prev => ({ ...prev, [id]: 'error' }))
            setLogs(prev => [...prev, { line: `[ERROR] ${event.message ?? '未知错误'}`, ts: event.ts }])
            es.close()
            esMap.current.delete(id)
          } else if (event.type === 'terminated' || event.type === 'external_terminated') {
            terminatedRef.current.add(id)
            setJobStatuses(prev => ({ ...prev, [id]: event.type }))
            if (event.message) {
              setLogs(prev => [...prev, { line: `[终止] ${event.message}`, ts: event.ts }])
            }
            es.close()
            esMap.current.delete(id)
          } else if (event.type === 'scenario_done') {
            setScenarioDoneCount(n => n + 1)
          }
        } catch {
          /* ignore */
        }
      }

      es.onerror = () => {
        if (terminatedRef.current.has(id)) {
          es.close()
          esMap.current.delete(id)
          return
        }
        if (errorCheckInFlightRef.current.has(id)) return
        errorCheckInFlightRef.current.add(id)
        api
          .getGenerationStatus(id)
          .then(data => {
            applyBackendSnapshot(data)
            const snapshotStatus = data.status as JobStatus
            setJobStatuses(prev => ({ ...prev, [id]: snapshotStatus }))
            if (isTerminalJobStatus(snapshotStatus)) {
              terminatedRef.current.add(id)
              es.close()
              esMap.current.delete(id)
            }
          })
          .catch(() => {
            // Keep the EventSource open; the browser will retry transient drops.
          })
          .finally(() => {
            window.setTimeout(() => {
              errorCheckInFlightRef.current.delete(id)
            }, 1500)
          })
      }

      esMap.current.set(id, es)
    },
    [applyBackendSnapshot],
  )

  // mount 时：从 localStorage 恢复；没有本地记录时，从后端发现正在运行的同阶段任务
  useEffect(() => {
    const discoverRunningJobIds = async (): Promise<string[]> => {
      const expected = expectedJobTypesForStage(stageKey)
      if (!expected) return []
      try {
        const jobs = (await Promise.all(LIVE_JOB_STATUSES.map(status => api.listGenerationJobs({ status })))).flat()
        return jobIdsNewestFirst(jobs.filter(job => isExpectedJobForStage(stageKey, job.job_type)))
      } catch {
        return []
      }
    }

    const restoreJobs = async () => {
      const stored = loadStoredJobs(stageKey)
      const candidateIds = stored.length > 0 ? stored : await discoverRunningJobIds()
      if (candidateIds.length === 0) return

      const toConnect: { id: string; status: JobStatus }[] = []
      const toRestore: { id: string; status: JobStatus }[] = []
      const toDrop: string[] = []

      await Promise.all(
        candidateIds.map(async id => {
          try {
            const data = await api.getGenerationStatus(id)
            if (!isExpectedJobForStage(stageKey, data.job_type)) {
              toDrop.push(id)
              return
            }
            applyBackendSnapshot(data)
            const snapshotStatus = data.status as JobStatus
            if (isLiveJobStatus(snapshotStatus)) {
              toConnect.push({ id, status: snapshotStatus })
            } else if (isTerminalJobStatus(snapshotStatus)) {
              toRestore.push({ id, status: snapshotStatus })
            } else {
              toDrop.push(id)
            }
          } catch {
            toDrop.push(id)
          }
        }),
      )

      const validIds = [...toConnect.map(r => r.id), ...toRestore.map(r => r.id)]

      if (validIds.length > 0) {
        setJobIds(validIds)
        saveStoredJobs(stageKey, validIds)

        // done/error：先在 terminatedRef 标记，防止 connectOne 的 onerror 误判为 error
        toRestore.forEach(({ id }) => terminatedRef.current.add(id))

        // 设置后端快照中的真实状态，connectOne 不会覆盖已有状态。
        const restoredJobs = [...toConnect, ...toRestore]
        setJobStatuses(prev => ({ ...prev, ...restoredJobStatusMap(restoredJobs) }))

        // live：建立 SSE 连接继续跟踪。
        toConnect.forEach(({ id, status }) => connectOne(id, status))

        // done/error：连接 SSE 回放历史日志，拿完后自动断开。
        toRestore.forEach(({ id, status }) => connectOne(id, status))
      } else {
        saveStoredJobs(stageKey, [])
      }
    }

    restoreJobs()

    const eventSources = esMap.current
    return () => {
      eventSources.forEach(es => es.close())
      eventSources.clear()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = useCallback(
    (id: string, scenarioTotals?: Record<string, number>, initialStatus: JobStatus = 'running') => {
      // 清理旧的非 live 任务，保留仍在运行或排队的连接。
      const toClose: string[] = []
      esMap.current.forEach((_, oldId) => {
        if (!isLiveJobStatus(jobStatuses[oldId])) toClose.push(oldId)
      })
      toClose.forEach(oldId => {
        esMap.current.get(oldId)?.close()
        esMap.current.delete(oldId)
      })
      terminatedRef.current.delete(id)
      errorCheckInFlightRef.current.delete(id)
      setJobIds([id])
      setJobStatuses({ [id]: initialStatus })
      setLogs([])
      setProgress(
        Object.fromEntries(
          Object.entries(scenarioTotals ?? {}).map(([scenario, total]) => [scenario, { scenario, done: 0, total }]),
        ),
      )
      setStats({ elapsed_sec: 0, samples_per_sec: 0 })
      lastLoggedProgressRef.current = { [id]: {} }
      saveStoredJobs(stageKey, [id])
      connectOne(id)
    },
    [connectOne, stageKey, jobStatuses],
  )

  const stop = useCallback(async () => {
    const runningIds = Object.entries(jobStatuses)
      .filter(([, s]) => isLiveJobStatus(s))
      .map(([id]) => id)
    if (runningIds.length === 0) return

    const results: StopJobResult[] = await Promise.all(
      runningIds.map(async id => {
        try {
          await api.stopGeneration(id)
          return { id, ok: true as const }
        } catch (error) {
          return { id, ok: false as const, error }
        }
      }),
    )

    const stoppedIds = results.filter(result => result.ok).map(result => result.id)
    const stoppedSet = new Set(stoppedIds)

    if (stoppedIds.length > 0) {
      setJobStatuses(prev => {
        const next = { ...prev }
        stoppedIds.forEach(id => {
          next[id] = 'terminated'
        })
        return next
      })
      stoppedIds.forEach(id => {
        terminatedRef.current.add(id)
        errorCheckInFlightRef.current.delete(id)
        esMap.current.get(id)?.close()
        esMap.current.delete(id)
      })
    }

    const remainingIds = jobIds.filter(id => !stoppedSet.has(id))
    saveStoredJobs(stageKey, remainingIds)
    setJobIds(remainingIds)

    const failureMessage = stopFailureMessage(results)
    if (failureMessage) throw new Error(failureMessage)
  }, [jobIds, jobStatuses, stageKey])

  const reset = useCallback(() => {
    esMap.current.forEach(es => es.close())
    esMap.current.clear()
    terminatedRef.current.clear()
    errorCheckInFlightRef.current.clear()
    lastLoggedProgressRef.current = {}
    setJobIds([])
    setJobStatuses({})
    setLogs([])
    setProgress({})
    setStats({ elapsed_sec: 0, samples_per_sec: 0 })
    setScenarioDoneCount(0)
    saveStoredJobs(stageKey, [])
  }, [stageKey])

  return {
    jobIds,
    jobId: currentJobIdFromNewestFirst(jobIds),
    status,
    logs,
    progress,
    stats,
    autoScroll,
    setAutoScroll,
    start,
    stop,
    reset,
    scenarioDoneCount,
  }
}
