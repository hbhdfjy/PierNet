import { useEffect, useRef, useState, useCallback } from 'react'
import useSWR from 'swr'
import { Virtuoso, type VirtuosoHandle } from 'react-virtuoso'
import { api } from '../lib/api'
import type { Text2CompScenariosConfig, Text2CompScenario, LogLine, JobStatus } from '../lib/types'
import {
  BookOpen, CheckCircle, XCircle, Loader2, Play, Square,
  RefreshCw, ArrowDownToLine, ChevronRight, ChevronDown, ChevronUp,
  FileText, Terminal, Tag, Eye, Clock, Layers,
  Pencil, Trash2, Save, X, Check, MessageSquare,
  Plus, Zap, AlertCircle,
} from 'lucide-react'
import { cn } from '../lib/utils'
import InterviewPanel from '../components/interview/InterviewPanel'
import EmptyState from '../components/ui/EmptyState'
import { useResizable } from '../hooks/useResizable'

// ── 字段组说明 ────────────────────────────────────────────────────
const FIELD_GROUPS = [
  { id: 'domain',      label: 'domain',      desc: 'domain_context · output_description · param_info' },
  { id: 'output_info', label: 'output_info', desc: '输出通道结构' },
  { id: 'observation', label: 'observation', desc: 'observation_config' },
]

const STATUS_ICON: Record<JobStatus, React.ReactNode> = {
  idle:       <Play size={14} className="text-slate-400" />,
  running:    <Loader2 size={14} className="animate-spin text-sky-400" />,
  done:       <CheckCircle size={14} className="text-emerald-400" />,
  error:      <XCircle size={14} className="text-red-400" />,
  terminated: <XCircle size={14} className="text-amber-400" />,
}

// ── Registry 类型 ─────────────────────────────────────────────────
type OutputInfoItem = { name: string; name_zh?: string; description: string; unit: string; slice: [number, number | null] }
type TimeModeItem   = { name: string; desc_en: string; desc_zh?: string; indices: string }
type ObsConfig = {
  fixed_time_mode?: string; fixed_channels?: unknown; channel_level?: string
  channel_min?: number; channel_max?: number | null; channel_name_template?: string
  time_modes?: TimeModeItem[]; time_mode_weights?: number[]
}
type RegistryEntry = {
  domain_context?: string; output_description?: string
  param_info?: Record<string, [string, string]>
  output_info?: OutputInfoItem[]; observation_config?: ObsConfig
  [k: string]: unknown
}

function FieldBadge({ present, label }: { present: boolean; label: string }) {
  return (
    <span className={cn('badge border text-xs', present
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25'
      : 'bg-slate-700/30 text-slate-600 border-slate-700/30')}>
      {present ? '✓' : '○'} {label}
    </span>
  )
}

function EditableField({ field, label, icon, children, value, editing, editVal, saving, saveErr, onStartEdit, onChangeVal, onCommit, onCancel }: {
  field: string; label: string; icon: React.ReactNode; children: React.ReactNode; value: unknown
  editing: string | null; editVal: string; saving: boolean; saveErr: string | null
  onStartEdit: (field: string, value: unknown) => void; onChangeVal: (v: string) => void
  onCommit: () => void; onCancel: () => void
}) {
  const isEditing = editing === field
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <div className="label flex items-center gap-1.5">{icon}{label}</div>
        <div className="flex-1" />
        {!isEditing ? (
          <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500 hover:text-slate-200" onClick={() => onStartEdit(field, value)}>
            <Pencil size={12} /> 编辑
          </button>
        ) : (
          <div className="flex gap-1">
            <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" onClick={onCommit} disabled={saving}>
              {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
            </button>
            <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500" onClick={onCancel}><X size={12} /> 取消</button>
          </div>
        )}
      </div>
      {isEditing ? (
        <div className="space-y-1.5">
          <textarea className="input w-full font-mono text-sm resize-y leading-relaxed"
            rows={field === 'domain_context' ? 7 : 12} value={editVal}
            onChange={e => onChangeVal(e.target.value)} autoFocus />
          {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
        </div>
      ) : children}
    </div>
  )
}

function RegistryEntryCard({ entryKey, entry, onSave, onDelete }: {
  entryKey: string; entry: RegistryEntry
  onSave: (key: string, data: RegistryEntry) => Promise<void>
  onDelete: (key: string) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'domain' | 'output' | 'obs' | 'params' | 'raw'>('domain')
  const [editing, setEditing] = useState<string | null>(null)
  const [editVal, setEditVal] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveErr, setSaveErr] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const hasDomain = !!entry.domain_context
  const hasOutput = Array.isArray(entry.output_info)
  const hasObs    = !!entry.observation_config
  const hasParams = !!entry.param_info
  const [simulator, scenario] = entryKey.includes('/') ? entryKey.split('/', 2) : [entryKey, '']
  const obs = entry.observation_config
  const params = entry.param_info ?? {}
  const outputInfo = (entry.output_info ?? []) as OutputInfoItem[]

  const startEdit = (field: string, value: unknown) => {
    setEditing(field); setSaveErr(null)
    setEditVal(typeof value === 'string' ? value : JSON.stringify(value, null, 2))
  }
  const commitEdit = async () => {
    if (!editing) return
    setSaving(true); setSaveErr(null)
    try {
      const parsed = (editing === 'domain_context' || editing === 'output_description') ? editVal : JSON.parse(editVal)
      await onSave(entryKey, { ...entry, [editing]: parsed })
      setEditing(null)
    } catch (e) {
      setSaveErr(e instanceof SyntaxError ? 'JSON 格式错误' : String(e))
    } finally { setSaving(false) }
  }
  const cancelEdit = () => { setEditing(null); setSaveErr(null) }
  const efProps = { editing, editVal, saving, saveErr, onStartEdit: startEdit, onChangeVal: setEditVal, onCommit: commitEdit, onCancel: cancelEdit }

  return (
    <div className={cn('border rounded-2xl overflow-hidden transition-all duration-200',
      open ? 'border-sky-500/30 bg-slate-800/60' : 'border-slate-700/40 bg-slate-800/30 hover:border-slate-600/60')}>
      <div className="flex items-center gap-3 px-4 py-3.5">
        <button onClick={() => setOpen(o => !o)} className="flex items-center gap-3 flex-1 text-left min-w-0">
          <span className={cn('transition-transform duration-200 text-slate-500 flex-shrink-0', open && 'rotate-90')}><ChevronRight size={15} /></span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-base text-slate-200 font-medium">{scenario || simulator}</span>
              {scenario && <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">{simulator}</span>}
            </div>
          </div>
        </button>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <FieldBadge present={hasDomain} label="domain" />
          <FieldBadge present={hasOutput} label="output" />
          <FieldBadge present={hasObs}    label="obs" />
          {!confirmDelete ? (
            <button className="btn-ghost py-1 px-2 text-xs text-slate-600 hover:text-red-400 ml-1"
              onClick={e => { e.stopPropagation(); setConfirmDelete(true) }}><Trash2 size={13} /></button>
          ) : (
            <div className="flex items-center gap-1 ml-1">
              <span className="text-xs text-red-400">确认？</span>
              <button className="btn-ghost py-0.5 px-2 text-xs text-red-400" onClick={async () => { await onDelete(entryKey); setConfirmDelete(false) }}><Check size={12} /></button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500" onClick={() => setConfirmDelete(false)}><X size={12} /></button>
            </div>
          )}
        </div>
      </div>

      {open && (
        <div className="border-t border-slate-700/30">
          <div className="flex border-b border-slate-700/30 bg-slate-900/30 overflow-x-auto">
            {([
              { id: 'domain', icon: <FileText size={13} />, label: '领域描述', disabled: false },
              { id: 'output', icon: <Layers size={13} />,   label: '输出通道', disabled: !hasOutput },
              { id: 'obs',    icon: <Clock size={13} />,    label: '观测配置', disabled: !hasObs },
              { id: 'params', icon: <Tag size={13} />,      label: `参数 (${Object.keys(params).length})`, disabled: !hasParams },
              { id: 'raw',    icon: <FileText size={13} />, label: 'Raw JSON', disabled: false },
            ] as const).map(t => (
              <button key={t.id} disabled={t.disabled} onClick={() => { setTab(t.id); setEditing(null) }}
                className={cn('flex items-center gap-1.5 px-4 py-2.5 text-sm transition-colors border-b-2 whitespace-nowrap flex-shrink-0',
                  tab === t.id && !t.disabled ? 'border-sky-500 text-sky-300 bg-sky-500/5'
                  : t.disabled ? 'border-transparent text-slate-700 cursor-not-allowed'
                  : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-700/20')}>
                {t.icon}{t.label}
              </button>
            ))}
          </div>
          <div className="p-4">
            {tab === 'domain' && (
              <div className="space-y-4">
                <EditableField {...efProps} field="domain_context" label="domain_context" icon={<FileText size={12} />} value={entry.domain_context ?? ''}>
                  {entry.domain_context ? <p className="text-sm text-slate-300 leading-relaxed bg-slate-900/40 rounded-xl p-3.5 border border-slate-700/30">{entry.domain_context}</p> : <p className="text-sm text-slate-600 italic">（未填写）</p>}
                </EditableField>
                <EditableField {...efProps} field="output_description" label="output_description" icon={<Eye size={12} />} value={entry.output_description ?? ''}>
                  {entry.output_description ? <p className="text-sm text-slate-400 font-mono bg-slate-900/40 rounded-xl p-3 border border-slate-700/30">{entry.output_description}</p> : <p className="text-sm text-slate-600 italic">（未填写）</p>}
                </EditableField>
              </div>
            )}
            {tab === 'output' && (
              <EditableField {...efProps} field="output_info" label={`output_info — ${outputInfo.length} 个通道组`} icon={<Layers size={12} />} value={entry.output_info ?? []}>
                {hasOutput ? (
                  <div className="space-y-2">
                    {outputInfo.map((o, i) => (
                      <div key={i} className="flex items-start gap-3 bg-slate-900/40 rounded-xl p-3.5 border border-slate-700/30">
                        <span className="font-mono text-xs text-slate-600 bg-slate-800 rounded px-1.5 py-0.5 flex-shrink-0 mt-0.5">[{o.slice?.[0] ?? 0}:{o.slice?.[1] ?? '…'}]</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-sm text-sky-300 font-medium">{o.name}</span>
                            {o.name_zh && <span className="text-sm text-slate-400">{o.name_zh}</span>}
                            <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">{o.unit}</span>
                          </div>
                          <p className="text-sm text-slate-500 mt-0.5">{o.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-sm text-slate-600 italic">（未填写）</p>}
              </EditableField>
            )}
            {tab === 'obs' && (
              <EditableField {...efProps} field="observation_config" label="observation_config" icon={<Clock size={12} />} value={entry.observation_config ?? {}}>
                {hasObs && obs ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: '默认时间模式', value: obs.fixed_time_mode ?? '—' },
                        { label: '通道选择级别', value: obs.channel_level ?? '—' },
                        { label: '最少通道数',   value: obs.channel_min ?? '—' },
                        { label: '最多通道数',   value: obs.channel_max ?? '全选' },
                      ].map(({ label, value }) => (
                        <div key={label} className="bg-slate-900/40 rounded-xl p-3 border border-slate-700/30">
                          <div className="label mb-1">{label}</div>
                          <div className="text-sm text-slate-200 font-mono">{String(value)}</div>
                        </div>
                      ))}
                    </div>
                    {obs.time_modes && obs.time_modes.length > 0 && (
                      <div>
                        <div className="label mb-2 flex items-center gap-1.5"><Clock size={12} />时间模式</div>
                        <div className="space-y-2">
                          {obs.time_modes.map((m, i) => {
                            const weight = obs.time_mode_weights?.[i]
                            const isDefault = m.name === obs.fixed_time_mode
                            return (
                              <div key={i} className={cn('rounded-xl p-3 border', isDefault ? 'bg-sky-500/8 border-sky-500/25' : 'bg-slate-900/30 border-slate-700/30')}>
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="font-mono text-sm text-slate-200">{m.name}</span>
                                  {isDefault && <span className="badge bg-sky-500/15 text-sky-400 border border-sky-500/25 text-xs">默认</span>}
                                  {weight != null && <span className="text-xs text-slate-500">权重 {(weight * 100).toFixed(0)}%</span>}
                                </div>
                                <p className="text-sm text-slate-400 mt-0.5">{m.desc_en}</p>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : <p className="text-sm text-slate-600 italic">（未填写）</p>}
              </EditableField>
            )}
            {tab === 'params' && (
              <EditableField {...efProps} field="param_info" label={`param_info — ${Object.keys(params).length} 个参数`} icon={<Tag size={12} />} value={entry.param_info ?? {}}>
                {hasParams ? (
                  <div className="rounded-xl border border-slate-700/30 overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-900/50">
                        <tr className="border-b border-slate-700/30">
                          <th className="px-4 py-2.5 text-left label">参数名</th>
                          <th className="px-4 py-2.5 text-left label">物理含义</th>
                          <th className="px-4 py-2.5 text-left label">单位</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(params).map(([name, info]) => (
                          <tr key={name} className="border-b border-slate-800/40 hover:bg-slate-700/15 transition-colors">
                            <td className="px-4 py-2.5 font-mono text-sky-300/80">{name}</td>
                            <td className="px-4 py-2.5 text-slate-300">{Array.isArray(info) ? info[0] : String(info)}</td>
                            <td className="px-4 py-2.5 font-mono text-slate-500 text-xs">{Array.isArray(info) ? info[1] : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : <p className="text-sm text-slate-600 italic">（未填写）</p>}
              </EditableField>
            )}
            {tab === 'raw' && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <div className="label flex items-center gap-1.5"><FileText size={12} />完整 JSON</div>
                  <div className="flex-1" />
                  {editing !== '__raw__' ? (
                    <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500 hover:text-slate-200" onClick={() => startEdit('__raw__', entry)}><Pencil size={12} /> 编辑</button>
                  ) : (
                    <div className="flex gap-1">
                      <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                        onClick={async () => { setSaving(true); setSaveErr(null); try { await onSave(entryKey, JSON.parse(editVal)); setEditing(null) } catch (e) { setSaveErr(e instanceof SyntaxError ? 'JSON 格式错误' : String(e)) } finally { setSaving(false) } }}>
                        {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}保存
                      </button>
                      <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500" onClick={() => { setEditing(null); setSaveErr(null) }}><X size={12} />取消</button>
                    </div>
                  )}
                </div>
                {editing === '__raw__' ? (
                  <div className="space-y-1.5">
                    <textarea className="input w-full font-mono text-sm resize-y leading-relaxed" rows={20} value={editVal} onChange={e => setEditVal(e.target.value)} autoFocus />
                    {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
                  </div>
                ) : (
                  <pre className="text-xs font-mono text-slate-400 bg-slate-950/50 rounded-xl p-4 border border-slate-700/30 overflow-x-auto max-h-96 overflow-y-auto leading-relaxed">{JSON.stringify(entry, null, 2)}</pre>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Registry 面板 ─────────────────────────────────────────────────

function RegistryPanel({ mutateRegistry }: { mutateRegistry: () => void }) {
  const { data: registry, isLoading, mutate } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })
  const [search, setSearch] = useState('')

  const handleSave = async (key: string, data: RegistryEntry) => { await api.updateRegistryEntry(key, data as Record<string, unknown>); mutate(); mutateRegistry() }
  const handleDelete = async (key: string) => { await api.deleteRegistryEntry(key); mutate(); mutateRegistry() }

  if (isLoading) return <div className="flex-1 flex items-center justify-center gap-2 text-slate-500"><RefreshCw size={15} className="animate-spin" /><span>加载…</span></div>

  const entries = Object.entries((registry ?? {}) as Record<string, RegistryEntry>)
    .filter(([k]) => !search || k.toLowerCase().includes(search.toLowerCase()))
    .sort(([a], [b]) => a.localeCompare(b))
  const total = Object.keys(registry ?? {}).length

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-700/30 flex-shrink-0">
        <FileText size={15} className="text-slate-400 flex-shrink-0" />
        <span className="text-sm font-medium text-slate-200">registry.yaml</span>
        <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30">{total} 条</span>
        <div className="flex-1" />
        <input type="text" className="input py-1.5 w-40 text-sm" placeholder="搜索…" value={search} onChange={e => setSearch(e.target.value)} />
        <button className="btn-ghost py-1.5" onClick={() => mutate()}><RefreshCw size={13} /></button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {entries.length === 0 && <EmptyState icon={BookOpen} title={total === 0 ? 'registry.yaml 为空' : `没有匹配 "${search}" 的记录`} description={total === 0 ? '在左侧完成注册后，条目会显示在这里' : undefined} />}
        {entries.map(([key, entry]) => <RegistryEntryCard key={key} entryKey={key} entry={entry} onSave={handleSave} onDelete={handleDelete} />)}
      </div>
    </div>
  )
}

// ── 自动注册日志面板 ──────────────────────────────────────────────

function AutoRegisterPanel() {
  const { data: scenariosCfg, isLoading: scLoading } = useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () => api.getText2CompScenarios())
  const { mutate: mutateRegistry } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })

  const [selectedScenarios, setSelectedScenarios] = useState<Set<string>>(new Set())
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set(['domain', 'output_info', 'observation']))
  const [overwrite, setOverwrite] = useState(false)
  const [simulatorLevel, setSimulatorLevel] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<JobStatus>('idle')
  const [logs, setLogs] = useState<LogLine[]>([])
  const [registeredKeys, setRegisteredKeys] = useState<string[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(true)

  const esRef = useRef<EventSource | null>(null)
  const virtuosoRef = useRef<VirtuosoHandle>(null)

  const allScenarios: Text2CompScenario[] = scenariosCfg ? Object.values(scenariosCfg).flat() : []

  const connectStream = useCallback((id: string) => {
    esRef.current?.close()
    setLogs([]); setRegisteredKeys([]); setStatus('running'); setConfigOpen(false)
    const es = api.openGenerationStream(id)
    esRef.current = es
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        if (event.type === 'log') {
          setLogs(prev => { const next = [...prev, { line: event.line, ts: event.ts }]; return next.length > 2000 ? next.slice(-2000) : next })
          if (event.register_progress) setRegisteredKeys(prev => { const key = event.register_progress.key; return prev.includes(key) ? prev : [...prev, key] })
        } else if (event.type === 'done') { setStatus('done'); es.close(); mutateRegistry() }
        else if (event.type === 'error') { setStatus('error'); setLogs(prev => [...prev, { line: `[ERROR] ${event.message}`, ts: event.ts }]); es.close() }
      } catch { /* ignore */ }
    }
    es.onerror = () => { setStatus('error'); es.close() }
  }, [mutateRegistry])

  useEffect(() => { if (jobId) connectStream(jobId); return () => esRef.current?.close() }, [jobId, connectStream])
  useEffect(() => { if (autoScroll && logs.length > 0) virtuosoRef.current?.scrollToIndex({ index: logs.length - 1, behavior: 'smooth' }) }, [logs, autoScroll])

  const handleLaunch = async () => {
    setError(null); setLaunching(true)
    try {
      const result = await api.startRegister({ scenarios: Array.from(selectedScenarios), fields: Array.from(selectedFields), overwrite, simulator_level: simulatorLevel, config: 'configs/text2comp/default.yaml', output: 'configs/text2comp/registry.yaml' })
      setJobId(result.job_id)
    } catch (e: unknown) { setError(e instanceof Error ? e.message : '启动失败') }
    finally { setLaunching(false) }
  }

  const handleStop = async () => { if (!jobId) return; await api.stopGeneration(jobId); setStatus('terminated'); esRef.current?.close() }

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
      {/* 配置区（可折叠）*/}
      <div className="flex-shrink-0 border-b border-slate-700/40">
        <button onClick={() => setConfigOpen(o => !o)} className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-slate-400 hover:bg-slate-800/40 transition-colors">
          <BookOpen size={13} className="text-slate-500" />
          <span className="flex-1 text-left font-medium">自动注册配置</span>
          {status !== 'idle' && (
            <span className="flex items-center gap-1.5 mr-2">
              {STATUS_ICON[status]}
              <span className={cn('text-xs', { running: 'text-sky-400', done: 'text-emerald-400', error: 'text-red-400', terminated: 'text-amber-400', idle: '' }[status])}>
                {{ idle: '', running: '注册中…', done: `已完成 ${registeredKeys.length} 条`, error: '出错', terminated: '已终止' }[status]}
              </span>
            </span>
          )}
          {configOpen ? <ChevronUp size={13} className="text-slate-600" /> : <ChevronDown size={13} className="text-slate-600" />}
        </button>
        {configOpen && (
          <div className="px-4 pb-4 pt-1 grid grid-cols-2 gap-4 bg-slate-900/30 border-t border-slate-700/30">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="label text-xs">选择场景</span>
                <div className="flex gap-1">
                  <button className="btn-ghost py-0.5 text-xs" onClick={() => setSelectedScenarios(new Set(allScenarios.map(s => s.name)))}>全选</button>
                  <button className="btn-ghost py-0.5 text-xs" onClick={() => setSelectedScenarios(new Set())}>清空</button>
                </div>
              </div>
              <div className="max-h-32 overflow-y-auto space-y-0.5 bg-slate-900/40 rounded-lg p-2 border border-slate-700/30">
                {allScenarios.map(s => (
                  <label key={s.name} className="flex items-center gap-2 px-1 py-0.5 rounded cursor-pointer hover:bg-slate-700/30">
                    <input type="checkbox" className="accent-sky-500 flex-shrink-0" checked={selectedScenarios.has(s.name)} onChange={() => setSelectedScenarios(prev => { const next = new Set(prev); next.has(s.name) ? next.delete(s.name) : next.add(s.name); return next })} />
                    <span className={cn('text-xs truncate flex-1', selectedScenarios.has(s.name) ? 'text-slate-200' : 'text-slate-500')}>{s.name}</span>
                  </label>
                ))}
                {!scLoading && allScenarios.length === 0 && <div className="text-slate-600 text-xs py-1 text-center">未找到 HDF5 文件</div>}
              </div>
              <div className="text-xs text-slate-600 mt-1">{selectedScenarios.size === 0 ? '不选 = 全部场景' : `已选 ${selectedScenarios.size} 个`}</div>
            </div>
            <div className="space-y-3">
              <div>
                <div className="label mb-1.5 text-xs">字段组</div>
                <div className="space-y-1">
                  {FIELD_GROUPS.map(fg => (
                    <label key={fg.id} className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" className="accent-sky-500 flex-shrink-0" checked={selectedFields.has(fg.id)}
                        onChange={() => setSelectedFields(prev => { const next = new Set(prev); if (next.has(fg.id)) { if (next.size > 1) next.delete(fg.id) } else next.add(fg.id); return next })} />
                      <span className={cn('text-xs', selectedFields.has(fg.id) ? 'text-slate-300' : 'text-slate-500')}>{fg.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="space-y-1">
                {[{ key: 'overwrite', val: overwrite, set: setOverwrite, label: '覆盖已有字段' }, { key: 'simLevel', val: simulatorLevel, set: setSimulatorLevel, label: 'Simulator 级别' }].map(({ key, val, set, label }) => (
                  <label key={key} className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" className="accent-sky-500 flex-shrink-0" checked={val} onChange={e => set(e.target.checked)} />
                    <span className="text-xs text-slate-400">{label}</span>
                  </label>
                ))}
              </div>
              {error && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{error}</div>}
              {status !== 'running' ? (
                <button className="btn-primary w-full justify-center py-2 text-xs" onClick={handleLaunch} disabled={launching}>
                  {launching ? <><RefreshCw size={12} className="animate-spin" /> 启动…</> : <><Play size={12} /> 开始注册</>}
                </button>
              ) : (
                <button className="btn-danger w-full justify-center py-2 text-xs" onClick={handleStop}><Square size={12} /> 终止</button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 日志流 */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-slate-700/30 flex-shrink-0 bg-slate-900/20">
        <span className="text-xs text-slate-500 tabular-nums">{logs.length} 行</span>
        <button className={cn('btn-ghost py-0.5', autoScroll && 'text-sky-400 bg-sky-500/10')} onClick={() => setAutoScroll(!autoScroll)} title="自动滚动"><ArrowDownToLine size={13} /></button>
      </div>
      {logs.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm gap-2">
          {status === 'idle' ? <>展开上方配置，点击「开始注册」</> : <><Loader2 size={14} className="animate-spin" /> 等待输出…</>}
        </div>
      ) : (
        <Virtuoso ref={virtuosoRef} className="flex-1" data={logs} followOutput={autoScroll ? 'smooth' : false}
          itemContent={(_, log) => (
            <div className={cn('px-4 py-0.5 font-mono text-sm leading-relaxed select-text',
              log.line.includes('[ERROR]') || log.line.includes('✗') ? 'text-red-400' :
              log.line.includes('✓') || log.line.includes('已保存') || log.line.includes('完成') ? 'text-emerald-400' :
              log.line.includes('[注册]') ? 'text-sky-300' :
              log.line.includes('[跳过]') ? 'text-slate-600' :
              log.line.includes('→ 推断') ? 'text-amber-300' : 'text-slate-400')}>{log.line}</div>
          )} />
      )}
    </div>
  )
}
// ── 主页面 ────────────────────────────────────────────────────────

export default function RegisterSimulator() {
  const { mutate: mutateRegistry } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })

  return (
    <div className="page-shell">
      <InterviewPanel onRegistryUpdate={() => mutateRegistry()} />
    </div>
  )
}
