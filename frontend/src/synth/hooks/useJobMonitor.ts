import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../../lib/api'
import type { LogLine, ScenarioProgress, LiveStats, JobStatus } from '../../lib/types'

export interface JobMonitorState {
  jobIds: string[]
  jobId: string | null
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  autoScroll: boolean
  setAutoScroll: (v: boolean) => void
  start: (jobId: string, scenarioTotals?: Record<string, number>) => void
  stop: () => Promise<void>
  reset: () => void
  scenarioDoneCount: number  // 每完成一个场景递增，用于触发外部刷新
}

const JOBS_KEY = (stageKey: string) => `piern_jobs_${stageKey}`

// localStorage：关闭标签页/浏览器后仍保留
function loadStoredJobs(stageKey: string): string[] {
  try {
    const raw = localStorage.getItem(JOBS_KEY(stageKey))
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveStoredJobs(stageKey: string, ids: string[]) {
  try {
    if (ids.length > 0) localStorage.setItem(JOBS_KEY(stageKey), JSON.stringify(ids))
    else localStorage.removeItem(JOBS_KEY(stageKey))
  } catch { /* ignore */ }
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

  const status: JobStatus = (() => {
    const statuses = Object.values(jobStatuses)
    if (statuses.length === 0) return 'idle'
    if (statuses.some(s => s === 'running')) return 'running'
    if (statuses.some(s => s === 'error')) return 'error'
    if (statuses.some(s => s === 'terminated')) return 'terminated'
    if (statuses.every(s => s === 'done')) return 'done'
    return 'idle'
  })()

  const removeJob = useCallback((id: string) => {
    setJobIds(prev => {
      const next = prev.filter(j => j !== id)
      saveStoredJobs(stageKey, next)
      return next
    })
    setJobStatuses(prev => {
      const next = { ...prev }
      delete next[id]
      return next
    })
  }, [stageKey])

  // 用 ref 记录每个 job 是否已收到终止事件，避免 onerror 与 onmessage 的竞态
  const terminatedRef = useRef<Set<string>>(new Set())

  const connectOne = useCallback((id: string, initialStatus: JobStatus = 'running') => {
    if (esMap.current.has(id)) return
    // 用传入的 initialStatus 初始化，不用 ?? 避免 React 批处理导致旧状态覆盖
    setJobStatuses(prev => {
      if (prev[id] !== undefined) return prev   // 已有状态不覆盖（restoreJobs 已预设）
      return { ...prev, [id]: initialStatus }
    })

    const es = api.openGenerationStream(id)

    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        if (event.type === 'init') {
          const totals: Record<string, number> = event.scenario_totals ?? {}
          setProgress(prev => {
            const next = { ...prev }
            for (const [sc, total] of Object.entries(totals)) {
              next[sc] = { scenario: sc, done: prev[sc]?.done ?? 0, total: total as number }
            }
            return next
          })
        } else if (event.type === 'log') {
          let shouldAppendLog = true
          if (event.progress) {
            const p: ScenarioProgress = event.progress
            setProgress(prev => ({ ...prev, [p.scenario]: p }))
            const byJob = lastLoggedProgressRef.current[id] ?? {}
            if (byJob[p.scenario] === p.done) {
              shouldAppendLog = false
            } else {
              lastLoggedProgressRef.current[id] = { ...byJob, [p.scenario]: p.done }
            }
          }
          if (event.stats) setStats(event.stats)
          if (shouldAppendLog) {
            setLogs(prev => {
              const next = [...prev, { line: event.line, ts: event.ts } as LogLine]
              return next.length > 5000 ? next.slice(-5000) : next
            })
          }
        } else if (event.type === 'done') {
          terminatedRef.current.add(id)   // 先标记，再 setState，防止 onerror 竞态
          setProgress(prev => {
            const next = { ...prev }
            for (const [scenario, p] of Object.entries(next)) {
              if (p.total > 0 && p.done < p.total) {
                next[scenario] = { ...p, done: p.total }
              }
            }
            return next
          })
          setJobStatuses(prev => ({ ...prev, [id]: 'done' }))
          es.close()
          esMap.current.delete(id)
        } else if (event.type === 'error') {
          terminatedRef.current.add(id)
          setJobStatuses(prev => ({ ...prev, [id]: 'error' }))
          setLogs(prev => [...prev, { line: `[ERROR] ${event.message ?? '未知错误'}`, ts: event.ts }])
          es.close()
          esMap.current.delete(id)
        } else if (event.type === 'terminated') {
          terminatedRef.current.add(id)
          setJobStatuses(prev => ({ ...prev, [id]: 'terminated' }))
          es.close()
          esMap.current.delete(id)
        } else if (event.type === 'scenario_done') {
          setScenarioDoneCount(n => n + 1)
        }
      } catch { /* ignore */ }
    }

    es.onerror = () => {
      es.close()
      esMap.current.delete(id)
      // 已收到终止事件（done/error/terminated）的连接断开不算错误
      if (!terminatedRef.current.has(id)) {
        setJobStatuses(prev => {
          if (prev[id] === 'running') return { ...prev, [id]: 'error' }
          return prev
        })
      }
    }

    esMap.current.set(id, es)
  }, [stageKey])

  // mount 时：从 localStorage 恢复，查询后端状态决定如何恢复
  useEffect(() => {
    const stored = loadStoredJobs(stageKey)
    if (stored.length === 0) return

    const restoreJobs = async () => {
      const toConnect: string[] = []
      const toRestore: { id: string; status: 'done' | 'error' }[] = []
      const toDrop: string[] = []

      await Promise.all(stored.map(async (id) => {
        try {
          const res = await fetch(`/api/generate/${id}/status`)
          if (res.ok) {
            const data = await res.json()
            if (data.status === 'running') {
              toConnect.push(id)
            } else if (data.status === 'done' || data.status === 'error') {
              toRestore.push({ id, status: data.status as 'done' | 'error' })
            } else {
              toDrop.push(id)
            }
          } else {
            toDrop.push(id)  // 404 等
          }
        } catch {
          toDrop.push(id)
        }
      }))

      const validIds = [...toConnect, ...toRestore.map(r => r.id)]

      if (validIds.length > 0) {
        setJobIds(validIds)
        saveStoredJobs(stageKey, validIds)

        // done/error：先在 terminatedRef 标记，防止 connectOne 的 onerror 误判为 error
        toRestore.forEach(({ id }) => terminatedRef.current.add(id))

        // done/error：设置真实最终状态（connectOne 里检测到 prev[id] 已有值会跳过覆盖）
        const finalStatuses: Record<string, JobStatus> = {}
        toRestore.forEach(({ id, status }) => { finalStatuses[id] = status })
        setJobStatuses(prev => ({ ...prev, ...finalStatuses }))

        // running：建立 SSE 连接
        toConnect.forEach(id => connectOne(id))

        // done/error：连接 SSE 回放历史日志，拿完后自动断开
        toRestore.forEach(({ id }) => connectOne(id))
      } else {
        saveStoredJobs(stageKey, [])
      }
    }

    restoreJobs()

    return () => {
      esMap.current.forEach(es => es.close())
      esMap.current.clear()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = useCallback((id: string, scenarioTotals?: Record<string, number>) => {
    // 清理旧的已完成/出错任务，只保留 running 的
    const toClose: string[] = []
    esMap.current.forEach((_, oldId) => {
      if (jobStatuses[oldId] !== 'running') toClose.push(oldId)
    })
    toClose.forEach(oldId => {
      esMap.current.get(oldId)?.close()
      esMap.current.delete(oldId)
    })
    setJobIds([id])
    setJobStatuses({ [id]: 'running' })
    setLogs([])
    setProgress(Object.fromEntries(
      Object.entries(scenarioTotals ?? {}).map(([scenario, total]) => [scenario, { scenario, done: 0, total }]),
    ))
    setStats({ elapsed_sec: 0, samples_per_sec: 0 })
    lastLoggedProgressRef.current = { [id]: {} }
    saveStoredJobs(stageKey, [id])
    connectOne(id)
  }, [connectOne, stageKey, jobStatuses])

  const stop = useCallback(async () => {
    const runningIds = Object.entries(jobStatuses)
      .filter(([, s]) => s === 'running')
      .map(([id]) => id)
    await Promise.all(runningIds.map(id => api.stopGeneration(id)))
    setJobStatuses(prev => {
      const next = { ...prev }
      runningIds.forEach(id => { next[id] = 'terminated' })
      return next
    })
    esMap.current.forEach(es => es.close())
    esMap.current.clear()
    saveStoredJobs(stageKey, [])
    setJobIds([])
  }, [jobStatuses, stageKey])

  const reset = useCallback(() => {
    esMap.current.forEach(es => es.close())
    esMap.current.clear()
    terminatedRef.current.clear()
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
    jobId: jobIds.length > 0 ? jobIds[jobIds.length - 1] : null,
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
