import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Bot, CheckCircle2, Database, Power, PowerOff, RefreshCw, ShieldCheck, Trash2 } from 'lucide-react'
import { api } from '../../lib/api'
import type { ExpertModelInfo } from '../../lib/types'
import { cn, formatBytes } from '../../lib/utils'

function statusClass(status?: string) {
  if (status === 'active') return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300'
  if (status === 'disabled') return 'border-amber-500/25 bg-amber-500/10 text-amber-300'
  return 'border-rose-500/25 bg-rose-500/10 text-rose-300'
}

function boolLabel(value?: boolean) {
  return value ? '启用' : '关闭'
}

function dimLabel(value?: number | null) {
  return typeof value === 'number' && value > 0 ? `${value} 维` : '--'
}

export default function ExpertModelManager() {
  const [models, setModels] = useState<ExpertModelInfo[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(true)
  const [acting, setActing] = useState('')
  const [error, setError] = useState<string | null>(null)

  const selected = useMemo(
    () => models.find(model => model.model_id === selectedId) ?? models[0] ?? null,
    [models, selectedId],
  )

  const loadModels = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.listExpertModels()
      setModels(response.models)
      if (!selectedId && response.models.length > 0) setSelectedId(response.models[0].model_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载专家模型失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const runAction = async (key: string, action: () => Promise<void>) => {
    setActing(key)
    setError(null)
    try {
      await action()
      await loadModels()
    } catch (e) {
      setError(e instanceof Error ? e.message : '操作失败')
    } finally {
      setActing('')
    }
  }

  const toggleStatus = (model: ExpertModelInfo) =>
    runAction(`status:${model.model_id}`, async () => {
      await api.updateExpertModel(model.model_id, { status: model.status === 'active' ? 'disabled' : 'active' })
    })

  const toggleFlag = (model: ExpertModelInfo, field: 'assembly_enabled' | 'data_generation_enabled') =>
    runAction(`${field}:${model.model_id}`, async () => {
      await api.updateExpertModel(model.model_id, { [field]: !model[field] })
    })

  const revalidate = (model: ExpertModelInfo) =>
    runAction(`validate:${model.model_id}`, async () => {
      await api.validateExpertModel(model.model_id)
    })

  const deleteModel = (model: ExpertModelInfo) =>
    runAction(`delete:${model.model_id}`, async () => {
      if (!window.confirm(`删除专家模型 ${model.name}？`)) return
      await api.deleteExpertModel(model.model_id)
      setSelectedId('')
    })

  return (
    <div className="page-shell">
      <div className="page-content space-y-4 p-4">
        <section className="card px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-9 h-9 rounded-lg bg-fuchsia-500/12 border border-fuchsia-500/25 flex items-center justify-center">
                <Bot size={18} className="text-fuchsia-300" />
              </div>
              <div className="min-w-0">
                <h1 className="text-lg font-semibold text-white">专家模型管理</h1>
                <p className="text-xs text-slate-500 font-mono truncate">predict(inputs: list[float])</p>
              </div>
            </div>
            <button className="btn-ghost px-3 py-2 text-xs" onClick={loadModels} disabled={loading}>
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
              刷新
            </button>
          </div>
        </section>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-rose-500/20 bg-rose-500/8 px-3 py-2 text-sm text-rose-300">
            <AlertCircle size={15} className="mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-[minmax(320px,0.42fr)_1fr]">
          <section className="card overflow-hidden">
            <div className="border-b border-slate-700/35 px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-slate-100">Registry</span>
                <span className="badge bg-slate-800/70 text-slate-300">{models.length} 个模型</span>
              </div>
            </div>
            <div className="max-h-[calc(100vh-260px)] min-h-[260px] overflow-auto p-2">
              {loading && <div className="p-4 text-sm text-slate-500">加载中...</div>}
              {!loading && models.length === 0 && <div className="p-4 text-sm text-slate-500">暂无专家模型</div>}
              {models.map(model => {
                const active = selected?.model_id === model.model_id
                return (
                  <button
                    key={model.model_id}
                    className={cn(
                      'w-full rounded-lg border px-3 py-3 text-left transition-all mb-2',
                      active
                        ? 'border-fuchsia-500/35 bg-fuchsia-500/10'
                        : 'border-slate-700/35 bg-slate-900/30 hover:border-slate-600/60',
                    )}
                    onClick={() => setSelectedId(model.model_id)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-slate-100">{model.name}</div>
                        <div className="mt-1 truncate font-mono text-xs text-slate-500">{model.model_id}</div>
                      </div>
                      <span className={cn('badge text-xs', statusClass(model.status))}>
                        {model.status ?? 'invalid'}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5 text-xs text-slate-500">
                      <span className="badge bg-slate-800/60 text-slate-400">{model.package_type ?? 'package'}</span>
                      <span className="badge bg-slate-800/60 text-slate-400">in {dimLabel(model.input_dim)}</span>
                      <span className="badge bg-slate-800/60 text-slate-400">out {dimLabel(model.output_dim)}</span>
                    </div>
                  </button>
                )
              })}
            </div>
          </section>

          <section className="card min-h-[480px]">
            {!selected ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">选择一个专家模型</div>
            ) : (
              <div className="space-y-4 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="truncate text-xl font-semibold text-white">{selected.name}</h2>
                      <span className={cn('badge', statusClass(selected.status))}>{selected.status ?? 'invalid'}</span>
                    </div>
                    <div className="mt-1 truncate font-mono text-xs text-slate-500">{selected.path}</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button className="btn-ghost px-3 py-2 text-xs" onClick={() => toggleStatus(selected)}>
                      {selected.status === 'active' ? <PowerOff size={13} /> : <Power size={13} />}
                      {selected.status === 'active' ? '禁用' : '启用'}
                    </button>
                    <button className="btn-ghost px-3 py-2 text-xs" onClick={() => revalidate(selected)}>
                      <ShieldCheck size={13} />
                      重新校验
                    </button>
                    <button
                      className="btn-ghost px-3 py-2 text-xs text-rose-300 hover:text-rose-200"
                      onClick={() => deleteModel(selected)}
                      disabled={acting === `delete:${selected.model_id}`}
                    >
                      <Trash2 size={13} />
                      删除
                    </button>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-4">
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Runtime</div>
                    <div className="mt-1 font-mono text-sm text-slate-200">{selected.runtime ?? 'python'}</div>
                  </div>
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Input</div>
                    <div className="mt-1 font-mono text-sm text-slate-200">{dimLabel(selected.input_dim)}</div>
                  </div>
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Output</div>
                    <div className="mt-1 font-mono text-sm text-slate-200">{dimLabel(selected.output_dim)}</div>
                  </div>
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Size</div>
                    <div className="mt-1 font-mono text-sm text-slate-200">
                      {formatBytes(selected.file_size_bytes || 0)}
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <button
                    className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3 text-left hover:border-slate-600/60"
                    onClick={() => toggleFlag(selected, 'assembly_enabled')}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <CheckCircle2 size={15} className="text-fuchsia-300" />
                        Assembly
                      </div>
                      <span
                        className={cn(
                          'badge',
                          selected.assembly_enabled ? statusClass('active') : statusClass('disabled'),
                        )}
                      >
                        {boolLabel(selected.assembly_enabled)}
                      </span>
                    </div>
                  </button>
                  <button
                    className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3 text-left hover:border-slate-600/60"
                    onClick={() => toggleFlag(selected, 'data_generation_enabled')}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <Database size={15} className="text-sky-300" />
                        数据生成
                      </div>
                      <span
                        className={cn(
                          'badge',
                          selected.data_generation_enabled ? statusClass('active') : statusClass('disabled'),
                        )}
                      >
                        {boolLabel(selected.data_generation_enabled)}
                      </span>
                    </div>
                  </button>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Domain / Simulator</div>
                    <div className="mt-2 flex flex-wrap gap-2 font-mono text-xs text-slate-300">
                      <span>{selected.domain ?? 'custom'}</span>
                      <span className="text-slate-600">/</span>
                      <span>{selected.simulator ?? 'expert_model'}</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                    <div className="text-xs text-slate-500">Entrypoint</div>
                    <div className="mt-2 font-mono text-xs text-slate-300">
                      {selected.entrypoint ?? selected.file_name} :: {selected.callable ?? 'predict'}
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-700/35 bg-slate-950/25 p-3">
                  <div className="text-xs text-slate-500">Checksum</div>
                  <div className="mt-2 break-all font-mono text-xs text-slate-400">{selected.checksum || '--'}</div>
                </div>

                {selected.last_error && (
                  <div className="rounded-lg border border-rose-500/20 bg-rose-500/8 p-3 text-sm text-rose-300">
                    {selected.last_error}
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
