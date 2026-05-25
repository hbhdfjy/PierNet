import { useMemo, useState } from 'react'
import useSWR from 'swr'
import { FolderOpen, Lock, RefreshCw, Search, ShieldCheck, Trash2 } from 'lucide-react'
import { api } from '../lib/api'
import type { FileAsset, FileCatalogResponse } from '../lib/types'
import { cn, formatBytes } from '../lib/utils'

const ALL = 'all'

const PLATFORM_LABELS: Record<string, string> = {
  synth: '数据合成',
  training: '训练平台',
  system: '系统',
}

const STAGE_LABELS: Record<string, string> = {
  stage1: '阶段 1 HDF5 数据',
  stage2: '阶段 2 模板',
  stage3: '阶段 3 样本',
  stage4: '阶段 4 路由',
  training: '训练产物',
  system: '清单 / 索引',
}

const KIND_LABELS: Record<string, string> = {
  hdf5: 'HDF5 文件',
  template: '模板 JSONL',
  sample: '样本 JSONL（兼容）',
  sample_merged: '合并样本 JSONL',
  sample_parquet: '样本 Parquet',
  router_scenario: '路由场景 JSONL（兼容）',
  router_parquet: '路由 Parquet',
  router_cache: '路由 JSONL 缓存',
  router_train: '路由训练 JSONL',
  catalog_db: '目录数据库',
  training_job: '训练任务',
  training_prepared: '训练 prepared 缓存',
  training_checkpoint: '训练权重',
  manifest: '清单',
  index: '索引',
}

const KIND_CLEAR_OPTIONS = [
  { key: 'samples', label: '清空样本' },
  { key: 'router', label: '清空路由数据' },
] as const

type FileManagerContentProps = {
  initialPlatform?: string
  lockPlatform?: boolean
  title?: string
  copy?: string
}

function platformLabel(asset: FileAsset) {
  return PLATFORM_LABELS[asset.platform] ?? asset.platform_label ?? asset.platform
}

function stageLabel(asset: FileAsset) {
  return STAGE_LABELS[asset.stage] ?? asset.stage_label ?? asset.stage
}

function kindLabel(asset: FileAsset) {
  return KIND_LABELS[asset.kind] ?? asset.kind_label ?? asset.kind
}

function timeText(ts: number) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

function statusLabel(asset: FileAsset) {
  if (asset.valid === false || asset.status === 'invalid') return '无效'
  if (asset.protected) return '受保护'
  if (asset.deletable) return '可管理'
  return asset.status || '正常'
}

function statusClass(asset: FileAsset) {
  if (asset.valid === false || asset.status === 'invalid') return 'border-red-500/30 bg-red-500/10 text-red-300'
  if (asset.protected) return 'border-amber-500/25 bg-amber-500/10 text-amber-300'
  return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300'
}

function SelectFilter({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: Array<{ value: string; label: string; count?: number }>
  onChange: (value: string) => void
}) {
  return (
    <label className="space-y-1.5">
      <span className="label text-xs">{label}</span>
      <select className="select w-full" value={value} onChange={e => onChange(e.target.value)}>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
            {option.count != null ? ` (${option.count})` : ''}
          </option>
        ))}
      </select>
    </label>
  )
}

function JsonDetails({ value }: { value: unknown }) {
  if (value == null || value === '') return <span className="text-slate-600">-</span>
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return (
      <span className="pretty-tooltip block min-w-0" data-tooltip={String(value)}>
        <span className="file-meta-inline block truncate font-mono text-slate-300">{String(value)}</span>
      </span>
    )
  }
  return (
    <pre className="file-meta-code max-h-56 overflow-auto rounded-xl border border-slate-700/35 bg-slate-950/35 p-3 text-xs leading-5 text-slate-400">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

function DetailPanel({
  asset,
  busy,
  onDelete,
}: {
  asset: FileAsset | null
  busy: boolean
  onDelete: (asset: FileAsset) => void
}) {
  if (!asset) {
    return (
      <div className="training-card file-manager-detail-panel file-manager-detail-empty">
        选择一个文件资产查看详情。
      </div>
    )
  }

  const detailEntries = Object.entries(asset.details ?? {})

  return (
    <div className="training-card file-manager-detail-panel overflow-hidden">
      <div className="card-header items-start">
        <ShieldCheck size={17} className="mt-0.5 text-sky-400" />
        <div className="min-w-0 flex-1">
          <div className="training-panel-title truncate">{asset.title}</div>
          <div className="pretty-tooltip min-w-0" data-tooltip={asset.path || undefined}>
            <div className="training-panel-copy truncate font-mono">{asset.path || '无文件路径'}</div>
          </div>
        </div>
        <span className={cn('badge border text-xs', statusClass(asset))}>{statusLabel(asset)}</span>
      </div>

      <div className="file-manager-detail-body">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Metric label="平台" value={platformLabel(asset)} />
          <Metric label="阶段" value={stageLabel(asset)} />
          <Metric label="类型" value={kindLabel(asset)} />
          <Metric label="大小" value={formatBytes(asset.file_size_bytes)} />
          <Metric
            label="数量"
            value={asset.count != null ? `${asset.count.toLocaleString()} ${asset.count_label ?? ''}` : '-'}
          />
          <Metric label="修改时间" value={timeText(asset.mtime)} />
        </div>

        {(asset.errors.length > 0 || asset.warnings.length > 0) && (
          <div className="space-y-2">
            {asset.errors.map(msg => (
              <div
                key={`err-${msg}`}
                className="rounded-xl border border-red-500/25 bg-red-500/10 px-3 py-2 text-sm text-red-200"
              >
                {msg}
              </div>
            ))}
            {asset.warnings.map(msg => (
              <div
                key={`warn-${msg}`}
                className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
              >
                {msg}
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <button className="btn-danger" disabled={!asset.deletable || busy} onClick={() => onDelete(asset)}>
            {busy ? <RefreshCw size={13} className="animate-spin" /> : <Trash2 size={13} />}
            删除文件
          </button>
          {asset.protected && (
            <div className="inline-flex items-center gap-1.5 rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-1.5 text-sm text-amber-200">
              <Lock size={13} />
              受保护资产不能直接删除。
            </div>
          )}
        </div>

        {detailEntries.length > 0 && (
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-200">元数据</div>
            <div className="space-y-2">
              {detailEntries.map(([key, value]) => (
                <div key={key} className="file-meta-block rounded-xl border border-slate-700/35 bg-slate-900/25 p-3">
                  <div className="label mb-1 text-xs">{key}</div>
                  <JsonDetails value={value} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-700/35 bg-slate-900/28 p-2.5">
      <div className="label mb-1 text-[11px]">{label}</div>
      <div className="truncate font-mono text-[13px] font-semibold text-slate-100">{value}</div>
    </div>
  )
}

export function FileManagerContent({
  initialPlatform = ALL,
  lockPlatform = false,
  title = '统一文件管理',
  copy = '集中管理 HDF5、模板、样本、路由数据、训练产物、清单与索引。',
}: FileManagerContentProps = {}) {
  const { data, isLoading, mutate } = useSWR<FileCatalogResponse>('file-catalog', () => api.getFileCatalog(), {
    refreshInterval: 12000,
  })
  const scopedAssets = useMemo(() => {
    const assets = data?.assets ?? []
    return lockPlatform ? assets.filter(asset => asset.platform === initialPlatform) : assets
  }, [data?.assets, initialPlatform, lockPlatform])

  const [platform, setPlatform] = useState(initialPlatform)
  const [stage, setStage] = useState(ALL)
  const [kind, setKind] = useState(ALL)
  const [status, setStatus] = useState(ALL)
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [busyAssetId, setBusyAssetId] = useState<string | null>(null)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const platformOptions = useMemo(() => toOptions(scopedAssets, 'platform', platformLabel), [scopedAssets])
  const stageOptions = useMemo(() => toOptions(scopedAssets, 'stage', stageLabel), [scopedAssets])
  const kindOptions = useMemo(() => toOptions(scopedAssets, 'kind', kindLabel), [scopedAssets])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return scopedAssets.filter(asset => {
      if (!lockPlatform && platform !== ALL && asset.platform !== platform) return false
      if (stage !== ALL && asset.stage !== stage) return false
      if (kind !== ALL && asset.kind !== kind) return false
      if (status === 'invalid' && !(asset.valid === false || asset.status === 'invalid')) return false
      if (status === 'deletable' && !asset.deletable) return false
      if (status === 'protected' && !asset.protected) return false
      if (q) {
        const haystack = [
          asset.title,
          asset.path,
          asset.simulator,
          asset.scenario,
          kindLabel(asset),
          stageLabel(asset),
          asset.job_id,
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase()
        if (!haystack.includes(q)) return false
      }
      return true
    })
  }, [scopedAssets, lockPlatform, platform, stage, kind, status, query])

  const selectedAsset = filtered.find(asset => asset.id === selectedId) ?? filtered[0] ?? null
  const scopedSummary = useMemo(
    () => ({
      total_assets: scopedAssets.length,
      total_size_bytes: scopedAssets.reduce((sum, asset) => sum + asset.file_size_bytes, 0),
      deletable_count: scopedAssets.filter(asset => asset.deletable).length,
      protected_count: scopedAssets.filter(asset => asset.protected).length,
      invalid_count: scopedAssets.filter(asset => asset.valid === false || asset.status === 'invalid').length,
    }),
    [scopedAssets],
  )

  async function refresh() {
    setError(null)
    await mutate()
  }

  async function deleteAsset(asset: FileAsset) {
    if (!asset.deletable) return
    const ok = window.confirm(`删除 ${asset.title}?\n${asset.path}\n此操作不可撤销。`)
    if (!ok) return
    setBusyAssetId(asset.id)
    setError(null)
    try {
      await api.deleteFileCatalogAsset(asset.id)
      setSelectedId(null)
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    } finally {
      setBusyAssetId(null)
    }
  }

  async function clearGroup(group: 'samples' | 'router') {
    const label = KIND_CLEAR_OPTIONS.find(item => item.key === group)?.label ?? group
    const ok = window.confirm(`${label}? 此操作不可撤销。`)
    if (!ok) return
    setBusyAction(group)
    setError(null)
    try {
      await api.clearFileCatalogGroup(group)
      setSelectedId(null)
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : '清空失败')
    } finally {
      setBusyAction(null)
    }
  }

  async function rebuildIndexes() {
    setBusyAction('rebuild')
    setError(null)
    try {
      const res = await api.rebuildFileCatalogIndexes('all')
      if (res.errors && res.errors.length > 0) setError(res.errors.slice(0, 3).join('\n'))
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : '重建失败')
    } finally {
      setBusyAction(null)
    }
  }

  return (
    <div className="page-shell">
      <div className="page-content space-y-4 p-4">
        <section className="training-hero training-hero--compact">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="training-eyebrow">统一文件目录</div>
              <h1 className="mt-2 text-[1.65rem] font-semibold tracking-tight text-white xl:text-[1.9rem]">{title}</h1>
              <p className="mt-1 max-w-3xl text-[13px] leading-6 text-slate-400">{copy}</p>
            </div>
            <div className="grid min-w-[460px] grid-cols-5 gap-2 max-xl:min-w-0 max-xl:w-full max-md:grid-cols-2">
              <Metric label="文件" value={scopedSummary.total_assets.toLocaleString()} />
              <Metric label="存储" value={formatBytes(scopedSummary.total_size_bytes)} />
              <Metric label="可删除" value={scopedSummary.deletable_count.toLocaleString()} />
              <Metric label="受保护" value={scopedSummary.protected_count.toLocaleString()} />
              <Metric label="无效" value={scopedSummary.invalid_count.toLocaleString()} />
            </div>
          </div>
        </section>

        <section className="training-card overflow-hidden">
          <div className="card-header flex-wrap gap-3">
            <FolderOpen size={17} className="text-sky-400" />
            <div className="min-w-[180px]">
              <div className="training-panel-title">文件筛选</div>
              <div className="training-panel-copy">左侧列表支持独立滚动，右侧查看详情和操作。</div>
            </div>
            <div className="flex-1" />
            <button className="btn-ghost text-xs" onClick={refresh} disabled={isLoading}>
              <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />
              刷新
            </button>
            {!lockPlatform && (
              <button
                className="btn-ghost text-xs text-sky-300"
                onClick={rebuildIndexes}
                disabled={busyAction === 'rebuild'}
              >
                {busyAction === 'rebuild' ? (
                  <RefreshCw size={12} className="animate-spin" />
                ) : (
                  <ShieldCheck size={12} />
                )}
                重建索引
              </button>
            )}
            {!lockPlatform &&
              KIND_CLEAR_OPTIONS.map(option => (
                <button
                  key={option.key}
                  className="btn-ghost text-xs text-red-300"
                  disabled={busyAction === option.key}
                  onClick={() => clearGroup(option.key)}
                >
                  {busyAction === option.key ? <RefreshCw size={12} className="animate-spin" /> : <Trash2 size={12} />}
                  {option.label}
                </button>
              ))}
          </div>

          <div
            className={`grid grid-cols-1 gap-3 border-b border-slate-700/35 p-4 ${lockPlatform ? 'lg:grid-cols-[1.2fr_0.9fr_0.9fr_0.9fr]' : 'lg:grid-cols-[1.2fr_0.9fr_0.9fr_0.9fr_0.9fr]'}`}
          >
            <label className="space-y-1.5">
              <span className="label text-xs">搜索</span>
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  className="input pl-9"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="场景、路径或任务名"
                />
              </div>
            </label>
            {!lockPlatform && (
              <SelectFilter label="平台" value={platform} options={platformOptions} onChange={setPlatform} />
            )}
            <SelectFilter label="阶段" value={stage} options={stageOptions} onChange={setStage} />
            <SelectFilter label="类型" value={kind} options={kindOptions} onChange={setKind} />
            <SelectFilter
              label="状态"
              value={status}
              options={[
                { value: ALL, label: '全部' },
                { value: 'deletable', label: '可管理' },
                { value: 'protected', label: '受保护' },
                { value: 'invalid', label: '无效' },
              ]}
              onChange={setStatus}
            />
          </div>

          {error && (
            <div className="mx-4 mt-3 rounded-xl border border-red-500/25 bg-red-500/10 p-3 text-sm whitespace-pre-wrap text-red-200">
              {error}
            </div>
          )}

          <div className="file-manager-workspace">
            <div className="file-manager-list-panel">
              <table className="file-manager-table w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/40 bg-slate-800/40">
                    <th className="px-4 py-3 text-left label">文件</th>
                    <th className="px-4 py-3 text-left label">阶段</th>
                    <th className="px-4 py-3 text-left label">类型</th>
                    <th className="px-4 py-3 text-right label">数量</th>
                    <th className="px-4 py-3 text-right label">大小</th>
                    <th className="px-4 py-3 text-right label">状态</th>
                    <th className="px-4 py-3 text-right label">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map(asset => (
                    <tr
                      key={asset.id}
                      className={cn(
                        'cursor-pointer border-b border-slate-800/45 transition-colors hover:bg-slate-700/18',
                        selectedAsset?.id === asset.id && 'bg-sky-500/8',
                      )}
                      onClick={() => setSelectedId(asset.id)}
                    >
                      <td className="px-4 py-3">
                        <div
                          className="pretty-tooltip max-w-[22rem] truncate font-mono font-semibold text-slate-100"
                          data-tooltip={asset.title}
                        >
                          {asset.title}
                        </div>
                        <div className="mt-1 max-w-[520px] truncate font-mono text-xs text-slate-600">{asset.path}</div>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{stageLabel(asset)}</td>
                      <td className="px-4 py-3 text-slate-400">{kindLabel(asset)}</td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums text-sky-300">
                        {asset.count != null ? asset.count.toLocaleString() : '-'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-slate-400">
                        {formatBytes(asset.file_size_bytes)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={cn('badge border text-xs', statusClass(asset))}>{statusLabel(asset)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          className="btn-ghost py-1 px-2 text-red-300"
                          disabled={!asset.deletable || busyAssetId === asset.id}
                          onClick={e => {
                            e.stopPropagation()
                            deleteAsset(asset)
                          }}
                          title={asset.deletable ? '删除' : '不可删除'}
                        >
                          {busyAssetId === asset.id ? (
                            <RefreshCw size={12} className="animate-spin" />
                          ) : asset.protected ? (
                            <Lock size={12} />
                          ) : (
                            <Trash2 size={12} />
                          )}
                        </button>
                      </td>
                    </tr>
                  ))}
                  {!isLoading && filtered.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-5 py-14 text-center text-slate-500">
                        没有匹配的文件
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <DetailPanel asset={selectedAsset} busy={!!busyAssetId} onDelete={deleteAsset} />
          </div>
        </section>
      </div>
    </div>
  )
}

function toOptions(
  assets: FileAsset[],
  valueKey: 'platform' | 'stage' | 'kind',
  labelFn: (asset: FileAsset) => string,
) {
  const counts = new Map<string, { label: string; count: number }>()
  for (const asset of assets) {
    const key = asset[valueKey]
    const label = labelFn(asset)
    const current = counts.get(key)
    counts.set(key, { label, count: (current?.count ?? 0) + 1 })
  }
  return [
    { value: ALL, label: '全部', count: assets.length },
    ...Array.from(counts.entries()).map(([value, item]) => ({ value, label: item.label, count: item.count })),
  ]
}
