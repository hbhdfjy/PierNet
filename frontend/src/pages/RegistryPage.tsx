import { useState } from 'react'
import useSWR from 'swr'
import { api } from '../lib/api'
import {
  BookOpen, XCircle, Loader2, ChevronRight,
  FileText, Tag, Eye, Clock, Layers, Pencil, Trash2,
  Save, X, Check, RefreshCw, Plus,
} from 'lucide-react'
import { cn } from '../lib/utils'
import EmptyState from '../components/ui/EmptyState'

type OutputInfoItem = { name: string; name_zh?: string; description: string; unit: string; slice: [number, number | null] }
type TimeModeItem   = { name: string; desc_en: string; desc_zh?: string; indices: string }
type ObsConfig = {
  fixed_time_mode?: string
  fixed_channels?: number[] | null   // null=全选，整数列表=指定索引
  channel_name_template?: string
  channel_name_template_zh?: string
  time_modes?: TimeModeItem[]
}
type RegistryEntry = {
  domain_context?: string; output_description?: string
  param_info?: Record<string, [string, string]>
  output_info?: OutputInfoItem[]; observation_config?: ObsConfig
  [k: string]: unknown
}

// ── 通道采样编辑器 ────────────────────────────────────────────────

function ChannelEditor({ obs, onChange }: { obs: ObsConfig; onChange: (u: ObsConfig) => void }) {
  const isAll = obs.fixed_channels === null || obs.fixed_channels === undefined
  const channels = isAll ? [] : (obs.fixed_channels ?? [])
  const [input, setInput] = useState('')

  const addChannel = (val: string) => {
    const n = parseInt(val.trim())
    if (isNaN(n) || n < 0) return
    if (channels.includes(n)) return
    onChange({ ...obs, fixed_channels: [...channels, n].sort((a, b) => a - b) })
    setInput('')
  }

  const removeChannel = (n: number) => {
    const next = channels.filter(c => c !== n)
    onChange({ ...obs, fixed_channels: next.length === 0 ? null : next })
  }

  const toggleAll = () => {
    if (isAll) {
      // 从全选切换到空列表，让用户自己添加
      onChange({ ...obs, fixed_channels: [] })
    } else {
      onChange({ ...obs, fixed_channels: null })
    }
  }

  return (
    <div className="space-y-2">
      <div className="label text-xs flex items-center gap-1.5"><Layers size={12} />通道采样</div>

      {/* 全选开关 */}
      <button
        onClick={toggleAll}
        className={cn(
          'w-full rounded-xl border px-3 py-2 text-left flex items-center justify-between transition-all',
          isAll
            ? 'bg-sky-500/10 border-sky-500/40 text-sky-200'
            : 'bg-slate-800/40 border-slate-700/30 text-slate-400 hover:border-slate-600/50',
        )}
      >
        <div>
          <span className="text-xs font-semibold">全选所有通道</span>
          <span className="text-xs opacity-60 ml-2">fixed_channels = null</span>
        </div>
        {isAll && <Check size={13} className="text-sky-400 flex-shrink-0" />}
      </button>

      {/* 指定索引 */}
      {!isAll && (
        <div className="rounded-xl border border-slate-700/30 bg-slate-800/30 p-3 space-y-2">
          {/* 已选标签 */}
          <div className="flex flex-wrap gap-1.5 min-h-[28px]">
            {channels.length === 0 && (
              <span className="text-xs text-slate-600 italic">尚未添加通道索引</span>
            )}
            {channels.map(n => (
              <span key={n} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-sky-500/15 border border-sky-500/30 text-sky-300 text-xs font-mono">
                {n}
                <button onClick={() => removeChannel(n)} className="hover:text-red-400 transition-colors">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
          {/* 输入框 */}
          <div className="flex items-center gap-2">
            <input
              type="number" min={0}
              className="input w-24 text-xs py-1 px-2 font-mono"
              placeholder="索引"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { addChannel(input); e.preventDefault() } }}
            />
            <button
              className="btn-ghost py-1 px-2.5 text-xs text-sky-400 hover:text-sky-300"
              onClick={() => addChannel(input)}
            >
              + 添加
            </button>
            <span className="text-xs text-slate-600">0-based 整数索引，Enter 确认</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── 观测配置编辑器（点击选择时间模式）────────────────────────────

type TimeModeName = 'monthly' | 'weekly' | 'full' | 'every_n'

const TIME_MODE_OPTIONS: { value: TimeModeName; label: string; desc: string }[] = [
  { value: 'monthly', label: '月度',     desc: '每月取1个点，共12点' },
  { value: 'weekly',  label: '每周',     desc: '每7步取1个点，最多52点' },
  { value: 'full',    label: '全量',     desc: '保留所有时间步' },
  { value: 'every_n', label: '自定义步长', desc: '每 N 步取1个点' },
]

function parseTimeModeName(fixed: string): { type: TimeModeName; step: number } {
  if (fixed === 'monthly') return { type: 'monthly', step: 10 }
  if (fixed === 'weekly')  return { type: 'weekly',  step: 7 }
  if (fixed === 'full')    return { type: 'full',    step: 1 }
  if (fixed.startsWith('every_')) {
    const n = parseInt(fixed.slice(6))
    return { type: 'every_n', step: isNaN(n) || n <= 0 ? 10 : n }
  }
  return { type: 'monthly', step: 10 }
}

function buildTimeModeEntry(type: TimeModeName, step: number): { indices: string; desc_en: string; desc_zh: string } {
  if (type === 'monthly') return { indices: 'monthly', desc_en: 'monthly (day 15 of each month), 12 time points', desc_zh: '月度（每月第15天），共12个时间点' }
  if (type === 'weekly')  return { indices: 'weekly',  desc_en: 'weekly (every 7 steps), up to 52 time points',  desc_zh: '每周（每7步），最多52个时间点' }
  if (type === 'full')    return { indices: 'full',    desc_en: 'full time series, all time points',             desc_zh: '全量时间序列，所有时间步' }
  return { indices: `every_${step}`, desc_en: `every ${step} steps`, desc_zh: `每 ${step} 步取1个点` }
}

function ObsConfigEditor({ obs, onChange }: {
  obs: ObsConfig
  onChange: (updated: ObsConfig) => void
}) {
  const fixed = obs.fixed_time_mode ?? 'monthly'
  const { type: selectedType, step: everyStep } = parseTimeModeName(fixed)
  const [stepInput, setStepInput] = useState(String(everyStep))

  const selectMode = (type: TimeModeName, step?: number) => {
    const n = step ?? (type === 'every_n' ? (parseInt(stepInput) || 10) : 10)
    const modeStr = type === 'every_n' ? `every_${n}` : type
    const { indices, desc_en, desc_zh } = buildTimeModeEntry(type, n)
    onChange({ ...obs, fixed_time_mode: modeStr, time_modes: [{ name: modeStr, indices, desc_en, desc_zh }] })
  }

  const handleStepChange = (val: string) => {
    setStepInput(val)
    const n = parseInt(val)
    if (n > 0) selectMode('every_n', n)
  }

  return (
    <div className="space-y-4">
      {/* 时间模式选择 */}
      <div className="space-y-2">
        <div className="label text-xs flex items-center gap-1.5"><Clock size={12} />时间采样模式</div>
        <div className="grid grid-cols-2 gap-2">
          {TIME_MODE_OPTIONS.map(opt => {
            const active = selectedType === opt.value
            return (
              <button
                key={opt.value}
                onClick={() => { if (opt.value !== 'every_n') selectMode(opt.value); else selectMode('every_n') }}
                className={cn(
                  'rounded-xl border px-3 py-2.5 text-left transition-all',
                  active
                    ? 'bg-sky-500/10 border-sky-500/40 text-sky-200'
                    : 'bg-slate-800/40 border-slate-700/30 text-slate-400 hover:border-slate-600/50 hover:text-slate-300',
                )}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs font-semibold">{opt.label}</span>
                  {active && <Check size={12} className="text-sky-400 flex-shrink-0" />}
                </div>
                <span className="text-xs opacity-70">{opt.desc}</span>
                {opt.value === 'every_n' && active && (
                  <div className="mt-2 flex items-center gap-2" onClick={e => e.stopPropagation()}>
                    <span className="text-xs text-slate-400">N =</span>
                    <input type="number" min={1} className="input w-20 text-xs py-1 px-2"
                      value={stepInput} onChange={e => handleStepChange(e.target.value)} />
                    <span className="text-xs text-slate-500">步</span>
                  </div>
                )}
              </button>
            )
          })}
        </div>
        <div className="rounded-lg bg-slate-900/50 border border-slate-700/30 px-3 py-2">
          <span className="label text-xs">当前：</span>
          <span className="font-mono text-xs text-sky-300 ml-1">{fixed}</span>
        </div>
      </div>

      {/* 通道采样（可编辑）*/}
      <ChannelEditor obs={obs} onChange={onChange} />
    </div>
  )
}

// ── Domain tab ────────────────────────────────────────────────────

function DomainTab({ entry, saving, saveErr, onSave }: {
  entry: RegistryEntry; saving: boolean; saveErr: string | null
  onSave: (patch: Partial<RegistryEntry>) => Promise<void>
}) {
  const [dcVal, setDcVal] = useState(entry.domain_context ?? '')
  const [odVal, setOdVal] = useState(entry.output_description ?? '')
  const [dcDirty, setDcDirty] = useState(false)
  const [odDirty, setOdDirty] = useState(false)

  return (
    <div className="space-y-4">
      {/* domain_context */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="label flex items-center gap-1.5"><FileText size={12} />domain_context</div>
          <div className="flex-1" />
          {dcDirty && (
            <div className="flex gap-1">
              <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                onClick={async () => { await onSave({ domain_context: dcVal }); setDcDirty(false) }}>
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => { setDcVal(entry.domain_context ?? ''); setDcDirty(false) }}>
                <X size={12} /> 取消
              </button>
            </div>
          )}
        </div>
        <textarea
          className="input w-full text-sm resize-y leading-relaxed"
          rows={6}
          value={dcVal}
          onChange={e => { setDcVal(e.target.value); setDcDirty(true) }}
        />
      </div>
      {/* output_description */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="label flex items-center gap-1.5"><Eye size={12} />output_description</div>
          <div className="flex-1" />
          {odDirty && (
            <div className="flex gap-1">
              <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                onClick={async () => { await onSave({ output_description: odVal }); setOdDirty(false) }}>
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => { setOdVal(entry.output_description ?? ''); setOdDirty(false) }}>
                <X size={12} /> 取消
              </button>
            </div>
          )}
        </div>
        <input
          className="input w-full text-sm font-mono"
          placeholder="{ch} channels × {ts} timesteps of ..."
          value={odVal}
          onChange={e => { setOdVal(e.target.value); setOdDirty(true) }}
        />
        <p className="text-xs text-slate-600 mt-1">必须包含 {'{ch}'} 和 {'{ts}'} 占位符</p>
      </div>
      {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
    </div>
  )
}

// ── Output info tab ───────────────────────────────────────────────

function OutputInfoTab({ outputInfo, saving, saveErr, onSave }: {
  outputInfo: OutputInfoItem[]; saving: boolean; saveErr: string | null
  onSave: (updated: OutputInfoItem[]) => Promise<void>
}) {
  const [items, setItems] = useState<OutputInfoItem[]>(outputInfo)
  const [dirty, setDirty] = useState(false)

  const update = (i: number, patch: Partial<OutputInfoItem>) => {
    const next = items.map((x, j) => j === i ? { ...x, ...patch } : x)
    setItems(next); setDirty(true)
  }
  const addItem = () => {
    const last = items[items.length - 1]
    const start = last ? (last.slice?.[1] ?? 0) : 0
    setItems([...items, { name: '', description: '', unit: '-', slice: [start, null] }])
    setDirty(true)
  }
  const removeItem = (i: number) => { setItems(items.filter((_, j) => j !== i)); setDirty(true) }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="label flex items-center gap-1.5"><Layers size={12} />output_info — {items.length} 个通道组</div>
        <div className="flex items-center gap-2">
          {dirty && (
            <>
              <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                onClick={async () => { await onSave(items); setDirty(false) }}>
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => { setItems(outputInfo); setDirty(false) }}>
                <X size={12} /> 取消
              </button>
            </>
          )}
          <button className="btn-ghost py-0.5 px-2 text-xs" onClick={addItem}><Plus size={11} /> 添加</button>
        </div>
      </div>
      {items.map((o, i) => (
        <div key={i} className="rounded-xl border border-slate-700/30 p-3 space-y-2 bg-slate-900/30 group">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-slate-600 bg-slate-800 rounded px-1.5 py-0.5 flex-shrink-0">#{i}</span>
            <div className="flex-1 grid grid-cols-3 gap-2">
              <div>
                <label className="label text-xs block mb-1">name</label>
                <input className="input w-full text-xs py-1 px-2 font-mono" placeholder="snake_case"
                  value={o.name} onChange={e => update(i, { name: e.target.value })} />
              </div>
              <div>
                <label className="label text-xs block mb-1">name_zh</label>
                <input className="input w-full text-xs py-1 px-2" placeholder="中文名"
                  value={o.name_zh ?? ''} onChange={e => update(i, { name_zh: e.target.value })} />
              </div>
              <div>
                <label className="label text-xs block mb-1">unit</label>
                <input className="input w-full text-xs py-1 px-2 font-mono" placeholder="m/d"
                  value={o.unit} onChange={e => update(i, { unit: e.target.value })} />
              </div>
            </div>
            <button className="btn-ghost py-0.5 px-1 text-slate-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
              onClick={() => removeItem(i)}><Trash2 size={12} /></button>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="col-span-2">
              <label className="label text-xs block mb-1">description</label>
              <input className="input w-full text-xs py-1 px-2" placeholder="物理含义"
                value={o.description} onChange={e => update(i, { description: e.target.value })} />
            </div>
            <div>
              <label className="label text-xs block mb-1">slice [start, end]</label>
              <div className="flex items-center gap-1">
                <input type="number" className="input w-full text-xs py-1 px-2 font-mono" placeholder="0"
                  value={o.slice?.[0] ?? 0}
                  onChange={e => update(i, { slice: [parseInt(e.target.value) || 0, o.slice?.[1] ?? null] })} />
                <span className="text-slate-600 text-xs">:</span>
                <input type="number" className="input w-full text-xs py-1 px-2 font-mono" placeholder="null"
                  value={o.slice?.[1] ?? ''}
                  onChange={e => {
                    const v = e.target.value === '' ? null : parseInt(e.target.value)
                    update(i, { slice: [o.slice?.[0] ?? 0, v] })
                  }} />
              </div>
            </div>
          </div>
        </div>
      ))}
      {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
    </div>
  )
}

// ── Param info tab ────────────────────────────────────────────────

function ParamInfoTab({ params, saving, saveErr, onSave }: {
  params: Record<string, [string, string]>; saving: boolean; saveErr: string | null
  onSave: (updated: Record<string, [string, string]>) => Promise<void>
}) {
  type Row = { name: string; meaning: string; unit: string }
  const [rows, setRows] = useState<Row[]>(
    Object.entries(params).map(([name, info]) => ({
      name,
      meaning: Array.isArray(info) ? info[0] : String(info),
      unit: Array.isArray(info) ? info[1] : '-',
    }))
  )
  const [dirty, setDirty] = useState(false)
  const [editingRow, setEditingRow] = useState<number | null>(null)

  const updateRow = (i: number, patch: Partial<Row>) => {
    setRows(rows.map((r, j) => j === i ? { ...r, ...patch } : r))
    setDirty(true)
  }
  const addRow = () => { setRows([...rows, { name: '', meaning: '', unit: '-' }]); setDirty(true); setEditingRow(rows.length) }
  const removeRow = (i: number) => { setRows(rows.filter((_, j) => j !== i)); setDirty(true) }

  const toParamInfo = () => Object.fromEntries(rows.filter(r => r.name).map(r => [r.name, [r.meaning, r.unit] as [string, string]]))

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="label flex items-center gap-1.5"><Tag size={12} />param_info — {rows.length} 个参数</div>
        <div className="flex items-center gap-2">
          {dirty && (
            <>
              <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                onClick={async () => { await onSave(toParamInfo()); setDirty(false) }}>
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => {
                  setRows(Object.entries(params).map(([name, info]) => ({ name, meaning: Array.isArray(info) ? info[0] : String(info), unit: Array.isArray(info) ? info[1] : '-' })))
                  setDirty(false)
                }}>
                <X size={12} /> 取消
              </button>
            </>
          )}
          <button className="btn-ghost py-0.5 px-2 text-xs" onClick={addRow}><Plus size={11} /> 添加</button>
        </div>
      </div>
      <div className="rounded-xl border border-slate-700/30 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-900/50">
            <tr className="border-b border-slate-700/30">
              <th className="px-3 py-2 text-left label w-40">参数名</th>
              <th className="px-3 py-2 text-left label">物理含义</th>
              <th className="px-3 py-2 text-left label w-28">单位</th>
              <th className="px-3 py-2 w-8" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-slate-800/40 hover:bg-slate-700/10 group">
                {editingRow === i ? (
                  <>
                    <td className="px-2 py-1.5">
                      <input className="input w-full text-xs py-1 px-2 font-mono"
                        value={r.name} onChange={e => updateRow(i, { name: e.target.value })} autoFocus />
                    </td>
                    <td className="px-2 py-1.5">
                      <input className="input w-full text-xs py-1 px-2"
                        value={r.meaning} onChange={e => updateRow(i, { meaning: e.target.value })} />
                    </td>
                    <td className="px-2 py-1.5">
                      <input className="input w-full text-xs py-1 px-2 font-mono"
                        value={r.unit} onChange={e => updateRow(i, { unit: e.target.value })}
                        onKeyDown={e => e.key === 'Enter' && setEditingRow(null)} />
                    </td>
                    <td className="px-2 py-1.5">
                      <button className="btn-ghost py-0.5 px-1 text-emerald-400" onClick={() => setEditingRow(null)}>
                        <Check size={11} />
                      </button>
                    </td>
                  </>
                ) : (
                  <>
                    <td className="px-3 py-2 font-mono text-sky-300/80 cursor-pointer" onClick={() => setEditingRow(i)}>{r.name || <span className="text-slate-600 italic">（空）</span>}</td>
                    <td className="px-3 py-2 text-slate-300 cursor-pointer" onClick={() => setEditingRow(i)}>{r.meaning}</td>
                    <td className="px-3 py-2 font-mono text-slate-500 text-xs cursor-pointer" onClick={() => setEditingRow(i)}>{r.unit}</td>
                    <td className="px-2 py-2">
                      <button className="btn-ghost py-0.5 px-1 text-slate-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => removeRow(i)}><Trash2 size={11} /></button>
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
    </div>
  )
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

function RegistryEntryCard({ entryKey, entry, onSave, onDelete }: {
  entryKey: string; entry: RegistryEntry
  onSave: (key: string, data: RegistryEntry) => Promise<void>
  onDelete: (key: string) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'domain' | 'output' | 'obs' | 'params' | 'raw'>('domain')
  const [rawEdit, setRawEdit] = useState(false)
  const [rawVal, setRawVal] = useState('')
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
              <button key={t.id} disabled={t.disabled} onClick={() => { setTab(t.id) }}
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
              <DomainTab entry={entry} saving={saving} saveErr={saveErr}
                onSave={async (patch) => {
                  setSaving(true); setSaveErr(null)
                  try { await onSave(entryKey, { ...entry, ...patch }) }
                  catch (e) { setSaveErr(String(e)) }
                  finally { setSaving(false) }
                }} />
            )}
            {tab === 'output' && (
              <OutputInfoTab outputInfo={outputInfo} saving={saving} saveErr={saveErr}
                onSave={async (updated) => {
                  setSaving(true); setSaveErr(null)
                  try { await onSave(entryKey, { ...entry, output_info: updated }) }
                  catch (e) { setSaveErr(String(e)) }
                  finally { setSaving(false) }
                }} />
            )}
            {tab === 'obs' && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="label flex items-center gap-1.5"><Clock size={12} />observation_config</div>
                  <div className="flex-1" />
                  {saving && <Loader2 size={12} className="animate-spin text-slate-500" />}
                  {saveErr && <div className="text-xs text-red-400 flex items-center gap-1"><XCircle size={11} />{saveErr}</div>}
                </div>
                {hasObs && obs ? (
                  <ObsConfigEditor obs={obs}
                    onChange={async (updated) => {
                      setSaving(true); setSaveErr(null)
                      try { await onSave(entryKey, { ...entry, observation_config: updated }) }
                      catch (e) { setSaveErr(String(e)) }
                      finally { setSaving(false) }
                    }} />
                ) : <p className="text-sm text-slate-600 italic">（未填写）</p>}
              </div>
            )}
            {tab === 'params' && (
              <ParamInfoTab params={params} saving={saving} saveErr={saveErr}
                onSave={async (updated) => {
                  setSaving(true); setSaveErr(null)
                  try { await onSave(entryKey, { ...entry, param_info: updated }) }
                  catch (e) { setSaveErr(String(e)) }
                  finally { setSaving(false) }
                }} />
            )}
            {tab === 'raw' && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <div className="label flex items-center gap-1.5"><FileText size={12} />完整 JSON</div>
                  <div className="flex-1" />
                  {!rawEdit ? (
                    <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500 hover:text-slate-200"
                      onClick={() => { setRawVal(JSON.stringify(entry, null, 2)); setRawEdit(true); setSaveErr(null) }}>
                      <Pencil size={12} /> 编辑
                    </button>
                  ) : (
                    <div className="flex gap-1">
                      <button className="btn-ghost py-0.5 px-2 text-xs text-emerald-400" disabled={saving}
                        onClick={async () => {
                          setSaving(true); setSaveErr(null)
                          try { await onSave(entryKey, JSON.parse(rawVal)); setRawEdit(false) }
                          catch (e) { setSaveErr(e instanceof SyntaxError ? 'JSON 格式错误' : String(e)) }
                          finally { setSaving(false) }
                        }}>
                        {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
                      </button>
                      <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                        onClick={() => { setRawEdit(false); setSaveErr(null) }}>
                        <X size={12} /> 取消
                      </button>
                    </div>
                  )}
                </div>
                {rawEdit ? (
                  <div className="space-y-1.5">
                    <textarea className="input w-full font-mono text-sm resize-y leading-relaxed" rows={20}
                      value={rawVal} onChange={e => setRawVal(e.target.value)} autoFocus />
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

// ── 场景描述行 ────────────────────────────────────────────────────

function ScenarioRow({ simulator, scenario, description, onSave, onDelete }: {
  simulator: string; scenario: string; description: string
  onSave: (key: string, desc: string) => Promise<void>
  onDelete: (key: string) => Promise<void>
}) {
  const [editing, setEditing] = useState(false)
  const [val, setVal] = useState(description)
  const [saving, setSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const key = `${simulator}/${scenario}`

  const commit = async () => {
    setSaving(true)
    try { await onSave(key, val); setEditing(false) }
    finally { setSaving(false) }
  }

  return (
    <div className="flex items-start gap-3 px-4 py-2.5 hover:bg-slate-700/15 transition-colors group">
      {/* 连接线 */}
      <div className="flex flex-col items-center flex-shrink-0 mt-1" style={{ width: 20 }}>
        <div className="w-px h-2 bg-slate-700/60" />
        <div className="w-3 h-px bg-slate-700/60" />
      </div>
      <span className="font-mono text-sm text-sky-300/80 flex-shrink-0 w-40 truncate pt-0.5">{scenario}</span>
      {editing ? (
        <div className="flex-1 flex items-center gap-2">
          <input className="input flex-1 text-sm py-1" value={val} onChange={e => setVal(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && commit()} autoFocus />
          <button className="btn-ghost py-1 px-2 text-xs text-emerald-400" onClick={commit} disabled={saving}>
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
          </button>
          <button className="btn-ghost py-1 px-2 text-xs text-slate-500" onClick={() => { setEditing(false); setVal(description) }}>
            <X size={12} />
          </button>
        </div>
      ) : (
        <div className="flex-1 flex items-center gap-2 min-w-0">
          <span className="text-sm text-slate-400 truncate flex-1">
            {description || <span className="text-slate-600 italic">（无描述）</span>}
          </span>
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
            <button className="btn-ghost py-0.5 px-1.5 text-xs text-slate-500 hover:text-slate-200"
              onClick={() => { setEditing(true); setVal(description) }}>
              <Pencil size={11} />
            </button>
            {!confirmDelete ? (
              <button className="btn-ghost py-0.5 px-1.5 text-xs text-slate-600 hover:text-red-400"
                onClick={() => setConfirmDelete(true)}><Trash2 size={11} /></button>
            ) : (
              <div className="flex items-center gap-1">
                <span className="text-xs text-red-400">删除？</span>
                <button className="btn-ghost py-0.5 px-1.5 text-xs text-red-400" onClick={() => onDelete(key)}><Check size={11} /></button>
                <button className="btn-ghost py-0.5 px-1.5 text-xs text-slate-500" onClick={() => setConfirmDelete(false)}><X size={11} /></button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function RegistryPage() {
  const { data: registry, isLoading, mutate } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })
  const [search, setSearch] = useState('')

  const handleSave = async (key: string, data: RegistryEntry | string) => {
    await api.updateRegistryEntry(key, typeof data === 'string'
      ? { scenario_description: data }
      : data as Record<string, unknown>)
    mutate()
  }
  const handleDelete = async (key: string) => {
    await api.deleteRegistryEntry(key)
    mutate()
  }

  // 新结构：顶层 key 是 simulator，value 含 scenarios 子字段
  const reg = (registry ?? {}) as Record<string, RegistryEntry & { scenarios?: Record<string, string> }>
  const simulators = Object.entries(reg)
    .filter(([k]) => !search || k.toLowerCase().includes(search.toLowerCase()) ||
      Object.keys(reg[k]?.scenarios ?? {}).some(s => s.toLowerCase().includes(search.toLowerCase())))
    .sort(([a], [b]) => a.localeCompare(b))

  const totalSimulators = Object.keys(reg).length
  const totalScenarios = Object.values(reg).reduce((s, e) => s + Object.keys(e?.scenarios ?? {}).length, 0)

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 工具栏 */}
      <div className="flex items-center gap-3 px-5 py-3.5 border-b border-slate-700/40 bg-slate-900/40 flex-shrink-0">
        <div className="w-7 h-7 rounded-lg bg-sky-500/20 border border-sky-500/30 flex items-center justify-center flex-shrink-0">
          <FileText size={14} className="text-sky-400" />
        </div>
        <span className="font-semibold text-white text-sm">注册信息</span>
        <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30">
          {totalSimulators} 个 simulator · {totalScenarios} 个场景
        </span>
        <div className="flex-1" />
        <input type="text" className="input py-1.5 w-48 text-sm" placeholder="搜索…"
          value={search} onChange={e => setSearch(e.target.value)} />
        <button className="btn-ghost py-1.5" onClick={() => mutate()}>
          <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* 列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {isLoading && (
          <div className="flex items-center justify-center gap-2 py-16 text-slate-500">
            <RefreshCw size={15} className="animate-spin" /><span>加载注册信息…</span>
          </div>
        )}
        {!isLoading && simulators.length === 0 && (
          <EmptyState icon={BookOpen}
            title={totalSimulators === 0 ? '尚无注册信息' : `没有匹配 "${search}" 的记录`}
            description={totalSimulators === 0 ? '在「注册数据集」页面完成注册后，条目会显示在这里' : undefined}
          />
        )}
        {simulators.map(([simKey, simEntry]) => {
          const scenarios = simEntry?.scenarios ?? {}
          const scenarioCount = Object.keys(scenarios).length
          // simulator 级条目（去掉 scenarios 子字段传给卡片）
          const { scenarios: _s, ...simFields } = simEntry ?? {}
          return (
            <div key={simKey} className="border border-slate-700/40 rounded-2xl overflow-hidden">
              {/* Simulator 级卡片 */}
              <RegistryEntryCard
                entryKey={simKey}
                entry={simFields as RegistryEntry}
                onSave={async (key, data) => handleSave(key, data)}
                onDelete={handleDelete}
              />
              {/* 场景列表 */}
              {scenarioCount > 0 && (
                <div className="border-t border-slate-700/30 bg-slate-900/20">
                  <div className="px-4 py-2 flex items-center gap-2">
                    <span className="label text-xs">场景描述</span>
                    <span className="badge bg-slate-700/50 text-slate-500 border border-slate-600/30 text-xs">{scenarioCount} 个</span>
                  </div>
                  <div className="pb-2">
                    {Object.entries(scenarios)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([sc, desc]) => (
                        <ScenarioRow
                          key={sc}
                          simulator={simKey}
                          scenario={sc}
                          description={desc}
                          onSave={async (key, d) => handleSave(key, d)}
                          onDelete={handleDelete}
                        />
                      ))}
                  </div>
                </div>
              )}
              {scenarioCount === 0 && (
                <div className="border-t border-slate-700/30 px-4 py-2.5 text-xs text-slate-600 bg-slate-900/20">
                  暂无场景描述 — 在「注册数据集」页面用「第二步：注册场景描述」添加
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
