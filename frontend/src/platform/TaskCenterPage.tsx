import { useEffect, useMemo, useState } from 'react'
import { Activity, DatabaseZap, FileCheck2, RefreshCw, ShieldCheck, Square, Workflow } from 'lucide-react'
import { api } from '../lib/api'
import type { AuditEvent, IntegrityStatus, UnifiedJobSummary } from '../lib/types'
import { cn } from '../lib/utils'
import { StatusBadge, TruncatedText } from '../shared/ui'

const ACTIVE = new Set(['queued', 'starting', 'running', 'evaluating', 'stopping'])

function formatTime(value?: number | null): string {
  if (!value) return '—'
  return new Date(value * 1000).toLocaleString()
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: '排队中',
    starting: '启动中',
    running: '运行中',
    evaluating: '评估中',
    stopping: '停止中',
    done: '已完成',
    error: '失败',
    terminated: '已终止',
    external_terminated: '外部终止',
  }
  return labels[status] ?? status
}

function statusClass(status: string): string {
  if (status === 'done') return 'bg-emerald-500/12 text-emerald-300 ring-1 ring-emerald-400/25'
  if (status === 'error' || status === 'external_terminated')
    return 'bg-rose-500/12 text-rose-300 ring-1 ring-rose-400/25'
  if (status === 'terminated') return 'bg-amber-500/12 text-amber-300 ring-1 ring-amber-400/25'
  return 'bg-sky-500/12 text-sky-300 ring-1 ring-sky-400/25'
}

function platformLabel(platform: string): string {
  return platform === 'training' ? '训练' : '合成'
}

export default function TaskCenterPage() {
  const [jobs, setJobs] = useState<UnifiedJobSummary[]>([])
  const [audit, setAudit] = useState<AuditEvent[]>([])
  const [integrity, setIntegrity] = useState<IntegrityStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const activeJobs = useMemo(() => jobs.filter(job => ACTIVE.has(job.status)), [jobs])

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const [jobItems, auditPayload, integrityPayload] = await Promise.all([
        api.listUnifiedJobs({ limit: 200 }),
        api.listAuditEvents(20),
        api.getIntegrityStatus(),
      ])
      setJobs(jobItems)
      setAudit(auditPayload.items)
      setIntegrity(integrityPayload)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 5000)
    return () => window.clearInterval(timer)
  }, [])

  const stopJob = async (jobId: string) => {
    await api.stopUnifiedJob(jobId)
    await refresh()
  }

  const rebuildIntegrity = async () => {
    await api.rebuildIntegrityManifest()
    await refresh()
  }

  return (
    <div className="page-surface">
      <div className="page-heading-row">
        <div className="page-title-block">
          <div className="page-kicker">
            <Workflow size={15} />
            统一任务中心
          </div>
          <h1>任务、审计与数据完整性</h1>
          <p>跨平台查看任务状态、最近操作和迁移校验基础状态。</p>
        </div>
        <button type="button" className="btn-secondary" onClick={refresh} disabled={loading}>
          <RefreshCw size={15} className={cn(loading && 'animate-spin')} />
          刷新
        </button>
      </div>

      {error && <div className="alert-error">无法加载任务中心：{error}</div>}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="panel overflow-hidden">
          <div className="panel__header">
            <div className="panel-title">
              <Activity size={17} />
              <div>
                <h2>统一任务列表</h2>
                <p>
                  {activeJobs.length} 个活跃任务 · 最近 {jobs.length} 条
                </p>
              </div>
            </div>
          </div>
          <div className="divide-y divide-slate-700/45">
            {jobs.length === 0 ? (
              <div className="empty-block">暂无任务记录</div>
            ) : (
              jobs.map(job => (
                <div
                  key={`${job.platform}:${job.job_id}`}
                  className="grid gap-3 px-4 py-3 lg:grid-cols-[120px_minmax(0,1fr)_140px_150px_auto] lg:items-center"
                >
                  <div className="flex items-center gap-2">
                    <StatusBadge className="bg-slate-700/55 text-slate-200">{platformLabel(job.platform)}</StatusBadge>
                    <span className="mono text-xs text-slate-500">{job.job_type}</span>
                  </div>
                  <div className="min-w-0">
                    <div className="font-semibold text-slate-100">
                      <TruncatedText value={job.job_id}>{job.name || job.job_id}</TruncatedText>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">{formatTime(job.started_at ?? job.created_at)}</div>
                  </div>
                  <StatusBadge className={statusClass(job.status)}>{statusLabel(job.status)}</StatusBadge>
                  <div className="mono text-sm text-slate-300">
                    {job.finished_at ? formatTime(job.finished_at) : '—'}
                  </div>
                  <div className="flex justify-end">
                    {ACTIVE.has(job.status) ? (
                      <button type="button" className="btn-ghost" onClick={() => stopJob(job.job_id)}>
                        <Square size={13} />
                        停止
                      </button>
                    ) : (
                      <span className="text-xs text-slate-600">—</span>
                    )}
                  </div>
                  {job.error_message && <div className="lg:col-span-5 text-sm text-rose-300">{job.error_message}</div>}
                </div>
              ))
            )}
          </div>
        </section>

        <aside className="space-y-4">
          <section className="panel">
            <div className="panel__header">
              <div className="panel-title">
                <FileCheck2 size={17} />
                <div>
                  <h2>数据完整性</h2>
                  <p>{integrity?.manifest_exists ? '已建立清单' : '尚未建立清单'}</p>
                </div>
              </div>
            </div>
            <div className="space-y-3 p-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">状态</span>
                <StatusBadge className={integrity?.ok ? statusClass('done') : statusClass('error')}>
                  {integrity?.ok ? '正常' : '异常'}
                </StatusBadge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">扫描文件</span>
                <span className="mono text-slate-200">{integrity?.scanned_entries ?? 0}</span>
              </div>
              <button type="button" className="btn-secondary w-full justify-center" onClick={rebuildIntegrity}>
                <DatabaseZap size={14} />
                重建校验清单
              </button>
            </div>
          </section>

          <section className="panel">
            <div className="panel__header">
              <div className="panel-title">
                <ShieldCheck size={17} />
                <div>
                  <h2>最近审计</h2>
                  <p>最近 {audit.length} 次写操作</p>
                </div>
              </div>
            </div>
            <div className="max-h-[430px] divide-y divide-slate-700/45 overflow-y-auto">
              {audit.length === 0 ? (
                <div className="empty-block">暂无审计事件</div>
              ) : (
                audit.map(item => (
                  <div key={item.id} className="space-y-1 px-4 py-3">
                    <div className="flex items-center justify-between gap-3">
                      <TruncatedText value={item.action} className="font-semibold text-slate-200" />
                      <span className="mono shrink-0 text-xs text-slate-500">{item.status_code ?? '—'}</span>
                    </div>
                    <div className="mono text-xs text-slate-500">{formatTime(item.ts)}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  )
}
