import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import type { LogLine, ScenarioProgress, LiveStats, JobStatus } from '../lib/types'

export interface JobMonitorState {
  jobIds: string[]
  jobId: string | null
  status: JobStatus
  logs: LogLine[]
  progress: Record<string, ScenarioProgress>
  stats: LiveStats
  autoScroll: boolean
  setAutoScroll: (v: boolean) => void
  start: (jobId: string) => void
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

  const connectOne = useCallback((id: string, initialStatus: JobStatus = 'running') => {
    if (esMap.current.has(id)) return
    setJobStatuses(prev => ({ ...prev, [id]: prev[id] ?? initialStatus }))

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
          if (event.progress) {
            const p: ScenarioProgress = event.progress
            setProgress(prev => ({ ...prev, [p.scenario]: p }))
          }
          if (event.stats) setStats(event.stats)
          setLogs(prev => {
            const next = [...prev, { line: event.line, ts: event.ts } as LogLine]
            return next.length > 5000 ? next.slice(-5000) : next
          })
        } else if (event.type === 'done') {
          setJobStatuses(prev => ({ ...prev, [id]: 'done' }))
          es.close()
          esMap.current.delete(id)
          // 保留 localStorage，让关闭重开后仍能看到完成状态
        } else if (event.type === 'error') {
          setJobStatuses(prev => ({ ...prev, [id]: 'error' }))
          setLogs(prev => [...prev, { line: `[ERROR] ${event.message ?? '未知错误'}`, ts: event.ts }])
          es.close()
          esMap.current.delete(id)
        } else if (event.type === 'terminated') {
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
      // 只有 running 状态才标为 error（done/error 状态的历史回放断开不算错误）
      setJobStatuses(prev => {
        if (prev[id] === 'running') return { ...prev, [id]: 'error' }
        return prev
      })
    }

    // handlers 附加完毕后再存入 map，避免极窄窗口内事件丢失
    esMap.current.set(id, es)
  }, [removeJob, stageKey])

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

        // done/error：先标记真实最终状态，避免恢复期间 status 跳到 running 导致按钮消失
        toRestore.forEach(({ id, status }) => {
          setJobStatuses(prev => ({ ...prev, [id]: status }))
        })

        // running：建立 SSE 连接
        toConnect.forEach(id => connectOne(id))

        // done/error：连接 SSE 回放历史日志（后端有缓存），拿完后自动断开
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

  const start = useCallback((id: string) => {
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
    setProgress({})
    setStats({ elapsed_sec: 0, samples_per_sec: 0 })
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
