
import { useMemo, useState } from 'react'
import { NavLink } from 'react-router-dom'
import useSWR from 'swr'
import {
  ArrowLeft,
  Database,
  FolderOpen,
  Lock,
  Moon,
  RefreshCw,
  Scissors,
  Search,
  ShieldCheck,
  Sun,
  Trash2,
} from 'lucide-react'
import { api } from '../lib/api'
import type { FileAsset, FileCatalogResponse } from '../lib/types'
import { cn, formatBytes } from '../lib/utils'
import type { Theme } from '../shared/theme'

const ALL = 'all'

const PLATFORM_LABELS: Record<string, string> = {
  synth: 'Data Synth',
  training: 'Training',
  system: 'System',
}

const STAGE_LABELS: Record<string, string> = {
  stage1: 'Stage 1 HDF5',
  stage2: 'Stage 2 Templates',
  stage3: 'Stage 3 Samples',
  stage4: 'Stage 4 Router',
  training: 'Training Artifacts',
  system: 'Manifests / Indexes',
}

const KIND_LABELS: Record<string, string> = {
  hdf5: 'HDF5',
  template: 'Template JSONL',
  sample: 'Sample JSONL',
  sample_merged: 'Merged Samples',
  router_scenario: 'Router Scenario',
  router_train: 'Router Train',
  training_job: 'Training Job',
  manifest: 'Manifest',
  index: 'Index',
}

const KIND_CLEAR_OPTIONS = [
  { key: 'templates', label: 'Clear templates' },
  { key: 'samples', label: 'Clear samples' },
  { key: 'router', label: 'Clear router' },
] as const

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
  if (asset.valid === false || asset.status === 'invalid') return 'Invalid'
  if (asset.protected) return 'Protected'
  if (asset.deletable) return 'Manageable'
  return asset.status || 'OK'
}

function statusClass(asset: FileAsset) {
  if (asset.valid === false || asset.status === 'invalid') return 'border-red-500/30 bg-red-500/10 text-red-300'
  if (asset.protected) return 'border-amber-500/25 bg-amber-500/10 text-amber-300'
  return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300'
}

function SimpleNav({ to, label, icon: Icon, end = false }: { to: string; label: string; icon: React.ElementType; end?: boolean }) {
  return (
    <NavLink to={to} end={end} className={({ isActive }) => `nav-item ${isActive ? 'nav-item--active' : ''}`}>
      {({ isActive }) => (
        <>
          {isActive && <span className="nav-item__rail" />}
          <div className="nav-item__icon"><Icon size={14} /></div>
          <span className="nav-item__label">{label}</span>
        </>
      )}
    </NavLink>
  )
}

function SelectFilter({ label, value, options, onChange }: {
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
            {option.label}{option.count != null ? ` (${option.count})` : ''}
          </option>
        ))}
      </select>
    </label>
  )
}

function JsonDetails({ value }: { value: unknown }) {
  if (value == null || value === '') return <span className="text-slate-600">-</span>
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return <span className="font-mono text-slate-300">{String(value)}</span>
  }
  return (
    <pre className="max-h-56 overflow-auto rounded-xl border border-slate-700/35 bg-slate-950/35 p-3 text-xs leading-5 text-slate-400">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

function DetailPanel({
  asset,
  busy,
  trimValue,
  onTrimValueChange,
  onDelete,
  onTrim,
}: {
  asset: FileAsset | null
  busy: boolean
  trimValue: string
  onTrimValueChange: (value: string) => void
  onDelete: (asset: FileAsset) => void
  onTrim: (asset: FileAsset) => void
}) {
  if (!asset) {
    return <div className="training-card min-h-[360px] p-6 text-sm text-slate-500">Select a file asset to inspect details.</div>
  }

  const detailEntries = Object.entries(asset.details ?? {})

  return (
    <div className="training-card overflow-hidden">
      <div className="card-header items-start">
        <ShieldCheck size={17} className="mt-0.5 text-sky-400" />
        <div className="min-w-0 flex-1">
          <div className="training-panel-title truncate">{asset.title}</div>
          <div className="training-panel-copy break-all font-mono">{asset.path || 'No file path'}</div>
        </div>
        <span className={cn('badge border text-xs', statusClass(asset))}>{statusLabel(asset)}</span>
      </div>

      <div className="space-y-4 p-4">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Metric label="Platform" value={platformLabel(asset)} />
          <Metric label="Stage" value={stageLabel(asset)} />
          <Metric label="Kind" value={kindLabel(asset)} />
          <Metric label="Size" value={formatBytes(asset.file_size_bytes)} />
          <Metric label="Count" value={asset.count != null ? `${asset.count.toLocaleString()} ${asset.count_label ?? ''}` : '-'} />
          <Metric label="Modified" value={timeText(asset.mtime)} />
        </div>

        {(asset.errors.length > 0 || asset.warnings.length > 0) && (
          <div className="space-y-2">
            {asset.errors.map(msg => (
              <div key={`err-${msg}`} className="rounded-xl border border-red-500/25 bg-red-500/10 px-3 py-2 text-sm text-red-200">{msg}</div>
            ))}
            {asset.warnings.map(msg => (
              <div key={`warn-${msg}`} className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">{msg}</div>
            ))}
          </div>
        )}

        {asset.kind === 'template' && (
          <div className="rounded-2xl border border-slate-700/35 bg-slate-900/30 p-3">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
              <Scissors size={14} className="text-amber-300" />Trim template file
            </div>
            <div className="flex gap-2">
              <input
                className="input font-mono"
                type="number"
                min={1}
                placeholder={asset.count != null ? String(asset.count) : 'Rows to keep'}
                value={trimValue}
                onChange={e => onTrimValueChange(e.target.value)}
              />
              <button className="btn-ghost flex-shrink-0 text-amber-300" disabled={busy || !trimValue} onClick={() => onTrim(asset)}>
                Trim
              </button>
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <button className="btn-danger" disabled={!asset.deletable || busy} onClick={() => onDelete(asset)}>
            {busy ? <RefreshCw size={13} className="animate-spin" /> : <Trash2 size={13} />}
            Delete asset
          </button>
          {asset.protected && (
            <div className="inline-flex items-center gap-1.5 rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-1.5 text-sm text-amber-200">
              <Lock size={13} />Protected assets cannot be deleted directly.
            </div>
          )}
        </div>

        {detailEntries.length > 0 && (
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-200">Metadata</div>
            <div className="space-y-2">
              {detailEntries.map(([key, value]) => (
                <div key={key} className="rounded-2xl border border-slate-700/35 bg-slate-900/25 p-3">
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
    <div className="rounded-xl border border-slate-700/35 bg-slate-900/28 p-3">
      <div className="label mb-1 text-[11px]">{label}</div>
      <div className="truncate font-mono text-sm font-semibold text-slate-100">{value}</div>
    </div>
  )
}

export function FileManagerContent() {
  const { data, isLoading, mutate } = useSWR<FileCatalogResponse>('file-catalog', () => api.getFileCatalog(), { refreshInterval: 12000 })
  const assets = data?.assets ?? []

  const [platform, setPlatform] = useState(ALL)
  const [stage, setStage] = useState(ALL)
  const [kind, setKind] = useState(ALL)
  const [status, setStatus] = useState(ALL)
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [busyAssetId, setBusyAssetId] = useState<string | null>(null)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [trimValue, setTrimValue] = useState('')
  const [error, setError] = useState<string | null>(null)

  const platformOptions = useMemo(() => toOptions(assets, 'platform', platformLabel), [assets])
  const stageOptions = useMemo(() => toOptions(assets, 'stage', stageLabel), [assets])
  const kindOptions = useMemo(() => toOptions(assets, 'kind', kindLabel), [assets])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return assets.filter(asset => {
      if (platform !== ALL && asset.platform !== platform) return false
      if (stage !== ALL && asset.stage !== stage) return false
      if (kind !== ALL && asset.kind !== kind) return false
      if (status === 'invalid' && !(asset.valid === false || asset.status === 'invalid')) return false
      if (status === 'deletable' && !asset.deletable) return false
      if (status === 'protected' && !asset.protected) return false
      if (q) {
        const haystack = [asset.title, asset.path, asset.simulator, asset.scenario, kindLabel(asset), stageLabel(asset), asset.job_id]
          .filter(Boolean).join(' ').toLowerCase()
        if (!haystack.includes(q)) return false
      }
      return true
    })
  }, [assets, platform, stage, kind, status, query])

  const selectedAsset = assets.find(asset => asset.id === selectedId) ?? filtered[0] ?? null

  async function refresh() {
    setError(null)
    await mutate()
  }

  async function deleteAsset(asset: FileAsset) {
    if (!asset.deletable) return
    const ok = window.confirm(`Delete ${asset.title}?\n${asset.path}\nThis cannot be undone.`)
    if (!ok) return
    setBusyAssetId(asset.id)
    setError(null)
    try {
      await api.deleteFileCatalogAsset(asset.id)
      setSelectedId(null)
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed')
    } finally {
      setBusyAssetId(null)
    }
  }

  async function trimTemplate(asset: FileAsset) {
    const n = parseInt(trimValue, 10)
    if (!asset.scenario || Number.isNaN(n) || n < 1) {
      setError('Enter a valid trim count')
      return
    }
    setBusyAssetId(asset.id)
    setError(null)
    try {
      await api.trimTemplateFile(asset.scenario, n)
      setTrimValue('')
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Trim failed')
    } finally {
      setBusyAssetId(null)
    }
  }

  async function clearGroup(group: 'templates' | 'samples' | 'router') {
    const label = KIND_CLEAR_OPTIONS.find(item => item.key === group)?.label ?? group
    const ok = window.confirm(`${label}? This cannot be undone.`)
    if (!ok) return
    setBusyAction(group)
    setError(null)
    try {
      await api.clearFileCatalogGroup(group)
      setSelectedId(null)
      await mutate()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Clear failed')
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
      setError(e instanceof Error ? e.message : 'Rebuild failed')
    } finally {
      setBusyAction(null)
    }
  }

  return (
    <div className="page-shell">
          <div className="page-content p-5 space-y-4">
            <section className="rounded-[28px] border border-slate-700/35 bg-gradient-to-br from-slate-900/88 via-slate-900/62 to-sky-950/32 p-5 shadow-2xl shadow-slate-950/20">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/20 bg-sky-500/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-sky-300">
                    Unified File Catalog
                  </div>
                  <h1 className="mt-3 text-3xl font-black tracking-tight text-slate-50">Unified file manager</h1>
                  <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">
                    Centralized HDF5, template, sample, router, training artifact, manifest, and index management.
                  </p>
                </div>
                <div className="grid min-w-[520px] grid-cols-5 gap-2 max-xl:min-w-0 max-xl:w-full max-md:grid-cols-2">
                  <Metric label="Assets" value={(data?.summary.total_assets ?? 0).toLocaleString()} />
                  <Metric label="Storage" value={formatBytes(data?.summary.total_size_bytes ?? 0)} />
                  <Metric label="Deletable" value={(data?.summary.deletable_count ?? 0).toLocaleString()} />
                  <Metric label="Protected" value={(data?.summary.protected_count ?? 0).toLocaleString()} />
                  <Metric label="Invalid" value={(data?.summary.invalid_count ?? 0).toLocaleString()} />
                </div>
              </div>
            </section>

            <section className="training-card overflow-hidden">
              <div className="card-header flex-wrap gap-3">
                <FolderOpen size={17} className="text-sky-400" />
                <div className="min-w-[180px]">
                  <div className="training-panel-title">Asset filters</div>
                  <div className="training-panel-copy">The table body scrolls independently; details and actions are on the right.</div>
                </div>
                <div className="flex-1" />
                <button className="btn-ghost text-xs" onClick={refresh} disabled={isLoading}>
                  <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />Refresh
                </button>
                <button className="btn-ghost text-xs text-sky-300" onClick={rebuildIndexes} disabled={busyAction === 'rebuild'}>
                  {busyAction === 'rebuild' ? <RefreshCw size={12} className="animate-spin" /> : <ShieldCheck size={12} />}
                  Rebuild indexes
                </button>
                {KIND_CLEAR_OPTIONS.map(option => (
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

              <div className="grid grid-cols-1 gap-3 border-b border-slate-700/35 p-4 lg:grid-cols-[1.2fr_0.9fr_0.9fr_0.9fr_0.9fr]">
                <label className="space-y-1.5">
                  <span className="label text-xs">Search</span>
                  <div className="relative">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input className="input pl-9" value={query} onChange={e => setQuery(e.target.value)} placeholder="Scenario, path, job name" />
                  </div>
                </label>
                <SelectFilter label="Platform" value={platform} options={platformOptions} onChange={setPlatform} />
                <SelectFilter label="Stage" value={stage} options={stageOptions} onChange={setStage} />
                <SelectFilter label="Kind" value={kind} options={kindOptions} onChange={setKind} />
                <SelectFilter
                  label="Status"
                  value={status}
                  options={[
                    { value: ALL, label: 'All' },
                    { value: 'deletable', label: 'Manageable' },
                    { value: 'protected', label: 'Protected' },
                    { value: 'invalid', label: 'Invalid' },
                  ]}
                  onChange={setStatus}
                />
              </div>

              {error && (
                <div className="mx-4 mt-3 rounded-2xl border border-red-500/25 bg-red-500/10 p-3 text-sm whitespace-pre-wrap text-red-200">
                  {error}
                </div>
              )}

              <div className="grid min-h-[560px] grid-cols-1 gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_420px]">
                <div className="list-table-scroll max-h-[620px] rounded-2xl border border-slate-700/35">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700/40 bg-slate-800/40">
                        <th className="px-4 py-3 text-left label">Asset</th>
                        <th className="px-4 py-3 text-left label">Stage</th>
                        <th className="px-4 py-3 text-left label">Kind</th>
                        <th className="px-4 py-3 text-right label">Count</th>
                        <th className="px-4 py-3 text-right label">Size</th>
                        <th className="px-4 py-3 text-right label">Status</th>
                        <th className="px-4 py-3 text-right label">Action</th>
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
                            <div className="font-mono font-semibold text-slate-100">{asset.title}</div>
                            <div className="mt-1 max-w-[520px] truncate font-mono text-xs text-slate-600">{asset.path}</div>
                          </td>
                          <td className="px-4 py-3 text-slate-400">{stageLabel(asset)}</td>
                          <td className="px-4 py-3 text-slate-400">{kindLabel(asset)}</td>
                          <td className="px-4 py-3 text-right font-mono tabular-nums text-sky-300">
                            {asset.count != null ? asset.count.toLocaleString() : '-'}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-400">{formatBytes(asset.file_size_bytes)}</td>
                          <td className="px-4 py-3 text-right">
                            <span className={cn('badge border text-xs', statusClass(asset))}>{statusLabel(asset)}</span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <button
                              className="btn-ghost py-1 px-2 text-red-300"
                              disabled={!asset.deletable || busyAssetId === asset.id}
                              onClick={e => { e.stopPropagation(); deleteAsset(asset) }}
                              title={asset.deletable ? 'Delete' : 'Cannot delete'}
                            >
                              {busyAssetId === asset.id ? <RefreshCw size={12} className="animate-spin" /> : asset.protected ? <Lock size={12} /> : <Trash2 size={12} />}
                            </button>
                          </td>
                        </tr>
                      ))}
                      {!isLoading && filtered.length === 0 && (
                        <tr>
                          <td colSpan={7} className="px-5 py-14 text-center text-slate-500">No matching file assets</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <DetailPanel
                  asset={selectedAsset}
                  busy={!!busyAssetId}
                  trimValue={trimValue}
                  onTrimValueChange={setTrimValue}
                  onDelete={deleteAsset}
                  onTrim={trimTemplate}
                />
              </div>
            </section>
          </div>
    </div>
  )
}

export default function FileManagerPage({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar w-56 flex-shrink-0">
        <div className="app-brand">
          <div className="app-brand__mark-wrap">
            <div className="app-brand__mark">F</div>
            <span className="app-brand__status" />
          </div>
          <div className="min-w-0">
            <div className="app-brand__title">PiERN Files</div>
            <div className="app-brand__subtitle">Unified file manager</div>
          </div>
        </div>

        <nav className="app-nav">
          <div>
            <div className="app-section-label"><span className="label text-[11px] whitespace-nowrap">Files</span><div className="app-section-label__line" /></div>
            <div className="space-y-1">
              <SimpleNav to="/files" end icon={FolderOpen} label="Unified Files" />
            </div>
          </div>
          <div>
            <div className="app-section-label"><span className="label text-[11px] whitespace-nowrap">Platforms</span><div className="app-section-label__line" /></div>
            <div className="space-y-1">
              <SimpleNav to="/synth" icon={Database} label="Data Synth" />
              <SimpleNav to="/training" icon={ArrowLeft} label="Training" />
            </div>
          </div>
        </nav>

        <div className="app-sidebar__footer">
          <button type="button" onClick={toggleTheme} className="theme-toggle">
            {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </aside>

      <main className="app-main">
        <FileManagerContent />
      </main>
    </div>
  )
}

function toOptions(assets: FileAsset[], valueKey: 'platform' | 'stage' | 'kind', labelFn: (asset: FileAsset) => string) {
  const counts = new Map<string, { label: string; count: number }>()
  for (const asset of assets) {
    const key = asset[valueKey]
    const label = labelFn(asset)
    const current = counts.get(key)
    counts.set(key, { label, count: (current?.count ?? 0) + 1 })
  }
  return [
    { value: ALL, label: 'All', count: assets.length },
    ...Array.from(counts.entries()).map(([value, item]) => ({ value, label: item.label, count: item.count })),
  ]
}
