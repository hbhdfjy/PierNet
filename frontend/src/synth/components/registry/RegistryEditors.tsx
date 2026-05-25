import { useState } from 'react'
import { Check, Clock, Eye, FileText, Layers, Loader2, Plus, Save, Tag, Trash2, X, XCircle } from 'lucide-react'
import { cn } from '../../../lib/utils'
import { parseIntegerField, parseIntegerText, parseOptionalIntegerField } from '../../integerInput'
import type { ObsConfig, OutputInfoItem, RegistryEntry } from './registryTypes'

function normalizeChannelLevel(level?: string): 'row' | 'output_info' {
  return level === 'output' || level === 'output_info' ? 'output_info' : 'row'
}

function ChannelEditor({
  obs,
  outputInfo,
  onChange,
}: {
  obs: ObsConfig
  outputInfo: OutputInfoItem[]
  onChange: (u: ObsConfig) => void
}) {
  const isAll = obs.fixed_channels === null || obs.fixed_channels === undefined
  const channelLevel = normalizeChannelLevel(obs.channel_level)
  const isOutputLevel = channelLevel === 'output_info'
  const rawChannels = isAll ? [] : (obs.fixed_channels ?? [])
  const channels = rawChannels.map(v => parseIntegerText(String(v), { min: 0 })).filter((n): n is number => n !== null)
  const outputChannels = Array.from(
    new Set(
      rawChannels
        .map(v => {
          if (typeof v === 'number') return v
          const s = String(v).trim()
          const named = outputInfo.findIndex(info => info.name === s)
          if (named >= 0) return named
          return parseIntegerText(s, { min: 0, max: outputInfo.length - 1 })
        })
        .filter((n): n is number => n !== null),
    ),
  ).sort((a, b) => a - b)
  const [input, setInput] = useState('')
  const [localErr, setLocalErr] = useState<string | null>(null)

  const visibleErr =
    localErr ??
    (!isAll && isOutputLevel && outputChannels.length === 0
      ? '至少选择 1 个输出维度'
      : !isAll && !isOutputLevel && channels.length === 0
        ? '至少选择 1 个输出通道'
        : null)

  const setMode = (mode: 'all' | 'row' | 'output_info') => {
    setLocalErr(null)
    if (mode === 'all') {
      onChange({ ...obs, fixed_channels: null })
      return
    }
    if (mode === 'row') {
      onChange({ ...obs, channel_level: 'row', fixed_channels: channels.length > 0 ? channels : [0] })
      return
    }
    if (outputInfo.length === 0) {
      setLocalErr('当前条目没有 output_info，无法按输出维度选择')
      return
    }
    onChange({ ...obs, channel_level: 'output_info', fixed_channels: outputChannels.length > 0 ? outputChannels : [0] })
  }

  const addChannel = (val: string) => {
    const n = parseIntegerText(val, { min: 0 })
    if (n === null) {
      setLocalErr('请输入非负整数通道索引')
      return
    }
    if (channels.includes(n)) {
      setLocalErr(`通道 ${n} 已存在`)
      return
    }
    setLocalErr(null)
    onChange({ ...obs, channel_level: 'row', fixed_channels: [...channels, n].sort((a, b) => a - b) })
    setInput('')
  }

  const removeChannel = (n: number) => {
    if (channels.length <= 1) {
      setLocalErr('至少保留 1 个输出通道；如需全部通道，请选择“全选所有通道”')
      return
    }
    setLocalErr(null)
    onChange({ ...obs, channel_level: 'row', fixed_channels: channels.filter(c => c !== n) })
  }

  const toggleOutputChannel = (index: number) => {
    const selected = outputChannels
    if (selected.includes(index)) {
      if (selected.length <= 1) {
        setLocalErr('至少保留 1 个输出维度')
        return
      }
      setLocalErr(null)
      onChange({ ...obs, channel_level: 'output_info', fixed_channels: selected.filter(n => n !== index) })
      return
    }
    setLocalErr(null)
    onChange({ ...obs, channel_level: 'output_info', fixed_channels: [...selected, index].sort((a, b) => a - b) })
  }

  const modeButton = (
    mode: 'all' | 'row' | 'output_info',
    title: string,
    subtitle: string,
    active: boolean,
    disabled = false,
  ) => (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setMode(mode)}
      className={cn(
        'rounded-xl border px-3 py-2 text-left transition-all disabled:cursor-not-allowed disabled:opacity-45',
        active
          ? 'bg-sky-500/10 border-sky-500/40 text-sky-200'
          : 'bg-slate-800/40 border-slate-700/30 text-slate-400 hover:border-slate-600/50 hover:text-slate-300',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold">{title}</span>
        {active && <Check size={12} className="text-sky-400 flex-shrink-0" />}
      </div>
      <div className="text-[11px] opacity-60 mt-1 font-mono">{subtitle}</div>
    </button>
  )

  return (
    <div className="space-y-2">
      <div className="label text-xs flex items-center gap-1.5">
        <Layers size={12} />
        通道采样
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        {modeButton('all', '全选所有通道', 'fixed_channels = null', isAll)}
        {modeButton('row', '指定通道索引', 'channel_level = row', !isAll && !isOutputLevel)}
        {modeButton(
          'output_info',
          '指定输出维度',
          'channel_level = output_info',
          !isAll && isOutputLevel,
          outputInfo.length === 0,
        )}
      </div>

      {!isAll && !isOutputLevel && (
        <div className="rounded-xl border border-slate-700/30 bg-slate-800/30 p-3 space-y-2">
          <div className="flex flex-wrap gap-1.5 min-h-[28px]">
            {channels.length === 0 && <span className="text-xs text-slate-600 italic">尚未添加通道索引</span>}
            {channels.map(n => (
              <span
                key={n}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-sky-500/15 border border-sky-500/30 text-sky-300 text-xs font-mono"
              >
                {n}
                <button type="button" onClick={() => removeChannel(n)} className="hover:text-red-400 transition-colors">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="number"
              min={0}
              className="input w-24 text-xs py-1 px-2 font-mono"
              placeholder="索引"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  addChannel(input)
                  e.preventDefault()
                }
              }}
            />
            <button
              type="button"
              className="btn-ghost py-1 px-2.5 text-xs text-sky-400 hover:text-sky-300"
              onClick={() => addChannel(input)}
            >
              + 添加
            </button>
            <span className="text-xs text-slate-600">0-based 整数索引，Enter 确认</span>
          </div>
        </div>
      )}

      {!isAll && isOutputLevel && (
        <div className="rounded-xl border border-slate-700/30 bg-slate-800/30 p-3 space-y-2">
          <div className="text-xs text-slate-500">从 output_info 中选择需要采样的输出维度，必须至少选 1 个。</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {outputInfo.map((info, index) => {
              const selected = outputChannels.includes(index)
              return (
                <button
                  type="button"
                  key={`${info.name}-${index}`}
                  onClick={() => toggleOutputChannel(index)}
                  className={cn(
                    'rounded-xl border px-3 py-2 text-left transition-all',
                    selected
                      ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-200'
                      : 'bg-slate-900/40 border-slate-700/30 text-slate-400 hover:border-slate-600/60 hover:text-slate-300',
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold font-mono text-sky-200">
                      #{index} {info.name}
                    </span>
                    {selected && <Check size={12} className="text-emerald-400 flex-shrink-0" />}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500 truncate">
                    {info.name_zh || info.description || '-'}
                  </div>
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-600 font-mono">
                    <span>{info.unit || '-'}</span>
                    <span>{Array.isArray(info.slice) ? `[${info.slice[0]}, ${info.slice[1] ?? ''}]` : ''}</span>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {visibleErr && (
        <div className="text-xs text-amber-300 flex items-center gap-1">
          <XCircle size={11} />
          {visibleErr}
        </div>
      )}
    </div>
  )
}

// ── 观测配置编辑器（点击选择时间模式）────────────

type TimeModeName = 'monthly' | 'weekly' | 'full' | 'every_n'

const TIME_MODE_OPTIONS: { value: TimeModeName; label: string; desc: string }[] = [
  { value: 'monthly', label: '月度', desc: '每月取1个点，共12点' },
  { value: 'weekly', label: '每周', desc: '每7步取1个点，最多52点' },
  { value: 'full', label: '全量', desc: '保留所有时间步' },
  { value: 'every_n', label: '自定义步长', desc: '每 N 步取1个点' },
]

function parseTimeModeName(fixed: string): { type: TimeModeName; step: number } {
  if (fixed === 'monthly') return { type: 'monthly', step: 10 }
  if (fixed === 'weekly') return { type: 'weekly', step: 7 }
  if (fixed === 'full') return { type: 'full', step: 1 }
  if (fixed.startsWith('every_')) {
    const n = parseIntegerText(fixed.slice(6), { min: 1 })
    return { type: 'every_n', step: n ?? 10 }
  }
  return { type: 'monthly', step: 10 }
}

function buildTimeModeEntry(type: TimeModeName, step: number): { indices: string; desc_en: string; desc_zh: string } {
  if (type === 'monthly')
    return {
      indices: 'monthly',
      desc_en: 'monthly (day 15 of each month), 12 time points',
      desc_zh: '月度（每月第15天），共12个时间点',
    }
  if (type === 'weekly')
    return {
      indices: 'weekly',
      desc_en: 'weekly (every 7 steps), up to 52 time points',
      desc_zh: '每周（每7步），最多52个时间点',
    }
  if (type === 'full')
    return { indices: 'full', desc_en: 'full time series, all time points', desc_zh: '全量时间序列，所有时间步' }
  return { indices: `every_${step}`, desc_en: `every ${step} steps`, desc_zh: `每 ${step} 步取1个点` }
}

export function ObsConfigEditor({
  obs,
  outputInfo,
  onChange,
}: {
  obs: ObsConfig
  outputInfo: OutputInfoItem[]
  onChange: (updated: ObsConfig) => void
}) {
  const fixed = obs.fixed_time_mode ?? 'monthly'
  const { type: selectedType, step: everyStep } = parseTimeModeName(fixed)
  const [stepInput, setStepInput] = useState(String(everyStep))

  const selectMode = (type: TimeModeName, step?: number) => {
    const n = step ?? (type === 'every_n' ? parseIntegerField(stepInput, 10, { min: 1 }) : 10)
    const modeStr = type === 'every_n' ? `every_${n}` : type
    const { indices, desc_en, desc_zh } = buildTimeModeEntry(type, n)
    onChange({ ...obs, fixed_time_mode: modeStr, time_modes: [{ name: modeStr, indices, desc_en, desc_zh }] })
  }

  const handleStepChange = (val: string) => {
    setStepInput(val)
    const n = parseIntegerText(val, { min: 1 })
    if (n !== null) selectMode('every_n', n)
  }

  return (
    <div className="space-y-4">
      {/* 时间模式选择 */}
      <div className="space-y-2">
        <div className="label text-xs flex items-center gap-1.5">
          <Clock size={12} />
          时间采样模式
        </div>
        <div className="grid grid-cols-2 gap-2">
          {TIME_MODE_OPTIONS.map(opt => {
            const active = selectedType === opt.value
            return (
              <button
                key={opt.value}
                onClick={() => {
                  if (opt.value !== 'every_n') selectMode(opt.value)
                  else selectMode('every_n')
                }}
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
                    <input
                      type="number"
                      min={1}
                      className="input w-20 text-xs py-1 px-2"
                      value={stepInput}
                      onChange={e => handleStepChange(e.target.value)}
                    />
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
      <ChannelEditor obs={obs} outputInfo={outputInfo} onChange={onChange} />
    </div>
  )
}

// ── Domain tab ────────────────────────────────────────────────────

export function DomainTab({
  entry,
  saving,
  saveErr,
  onSave,
}: {
  entry: RegistryEntry
  saving: boolean
  saveErr: string | null
  onSave: (patch: Partial<RegistryEntry>) => Promise<boolean>
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
          <div className="label flex items-center gap-1.5">
            <FileText size={12} />
            domain_context
          </div>
          <div className="flex-1" />
          {dcDirty && (
            <div className="flex gap-1">
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-emerald-400"
                disabled={saving}
                onClick={async () => {
                  if (await onSave({ domain_context: dcVal })) setDcDirty(false)
                }}
              >
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => {
                  setDcVal(entry.domain_context ?? '')
                  setDcDirty(false)
                }}
              >
                <X size={12} /> 取消
              </button>
            </div>
          )}
        </div>
        <textarea
          className="input w-full text-sm resize-y leading-relaxed"
          rows={6}
          value={dcVal}
          onChange={e => {
            setDcVal(e.target.value)
            setDcDirty(true)
          }}
        />
      </div>
      {/* output_description */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="label flex items-center gap-1.5">
            <Eye size={12} />
            output_description
          </div>
          <div className="flex-1" />
          {odDirty && (
            <div className="flex gap-1">
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-emerald-400"
                disabled={saving}
                onClick={async () => {
                  if (await onSave({ output_description: odVal })) setOdDirty(false)
                }}
              >
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => {
                  setOdVal(entry.output_description ?? '')
                  setOdDirty(false)
                }}
              >
                <X size={12} /> 取消
              </button>
            </div>
          )}
        </div>
        <input
          className="input w-full text-sm font-mono"
          placeholder="{ch} channels × {ts} timesteps of ..."
          value={odVal}
          onChange={e => {
            setOdVal(e.target.value)
            setOdDirty(true)
          }}
        />
        <p className="text-xs text-slate-600 mt-1">
          必须包含 {'{ch}'} 和 {'{ts}'} 占位符
        </p>
      </div>
      {saveErr && (
        <div className="text-xs text-red-400 flex items-center gap-1">
          <XCircle size={11} />
          {saveErr}
        </div>
      )}
    </div>
  )
}

// ── Output info tab ───────────────────────────────────────────────

export function OutputInfoTab({
  outputInfo,
  saving,
  saveErr,
  onSave,
}: {
  outputInfo: OutputInfoItem[]
  saving: boolean
  saveErr: string | null
  onSave: (updated: OutputInfoItem[]) => Promise<boolean>
}) {
  const [items, setItems] = useState<OutputInfoItem[]>(outputInfo)
  const [dirty, setDirty] = useState(false)

  const update = (i: number, patch: Partial<OutputInfoItem>) => {
    const next = items.map((x, j) => (j === i ? { ...x, ...patch } : x))
    setItems(next)
    setDirty(true)
  }
  const addItem = () => {
    const last = items[items.length - 1]
    const start = last ? (last.slice?.[1] ?? 0) : 0
    setItems([...items, { name: '', description: '', unit: '-', slice: [start, null] }])
    setDirty(true)
  }
  const updateSliceStart = (i: number, item: OutputInfoItem, value: string) => {
    const start = parseIntegerField(value, item.slice?.[0] ?? 0, { min: 0 })
    const end = item.slice?.[1] ?? null
    update(i, { slice: [start, end !== null && end < start ? start : end] })
  }
  const updateSliceEnd = (i: number, item: OutputInfoItem, value: string) => {
    const start = item.slice?.[0] ?? 0
    update(i, { slice: [start, parseOptionalIntegerField(value, item.slice?.[1] ?? null, { min: start })] })
  }
  const removeItem = (i: number) => {
    setItems(items.filter((_, j) => j !== i))
    setDirty(true)
  }

  return (
    <div className="list-scroll-lg space-y-3">
      <div className="flex items-center justify-between">
        <div className="label flex items-center gap-1.5">
          <Layers size={12} />
          output_info — {items.length} 个通道组
        </div>
        <div className="flex items-center gap-2">
          {dirty && (
            <>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-emerald-400"
                disabled={saving}
                onClick={async () => {
                  if (await onSave(items)) setDirty(false)
                }}
              >
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => {
                  setItems(outputInfo)
                  setDirty(false)
                }}
              >
                <X size={12} /> 取消
              </button>
            </>
          )}
          <button className="btn-ghost py-0.5 px-2 text-xs" onClick={addItem}>
            <Plus size={11} /> 添加
          </button>
        </div>
      </div>
      {items.map((o, i) => (
        <div key={i} className="rounded-xl border border-slate-700/30 p-3 space-y-2 bg-slate-900/30 group">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-slate-600 bg-slate-800 rounded px-1.5 py-0.5 flex-shrink-0">
              #{i}
            </span>
            <div className="flex-1 grid grid-cols-3 gap-2">
              <div>
                <label className="label text-xs block mb-1">name</label>
                <input
                  className="input w-full text-xs py-1 px-2 font-mono"
                  placeholder="snake_case"
                  value={o.name}
                  onChange={e => update(i, { name: e.target.value })}
                />
              </div>
              <div>
                <label className="label text-xs block mb-1">name_zh</label>
                <input
                  className="input w-full text-xs py-1 px-2"
                  placeholder="中文名"
                  value={o.name_zh ?? ''}
                  onChange={e => update(i, { name_zh: e.target.value })}
                />
              </div>
              <div>
                <label className="label text-xs block mb-1">unit</label>
                <input
                  className="input w-full text-xs py-1 px-2 font-mono"
                  placeholder="m/d"
                  value={o.unit}
                  onChange={e => update(i, { unit: e.target.value })}
                />
              </div>
            </div>
            <button
              className="btn-ghost py-0.5 px-1 text-slate-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
              onClick={() => removeItem(i)}
            >
              <Trash2 size={12} />
            </button>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="col-span-2">
              <label className="label text-xs block mb-1">description</label>
              <input
                className="input w-full text-xs py-1 px-2"
                placeholder="物理含义"
                value={o.description}
                onChange={e => update(i, { description: e.target.value })}
              />
            </div>
            <div>
              <label className="label text-xs block mb-1">slice [start, end]</label>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  className="input w-full text-xs py-1 px-2 font-mono"
                  placeholder="0"
                  min={0}
                  value={o.slice?.[0] ?? 0}
                  onChange={e => updateSliceStart(i, o, e.target.value)}
                />
                <span className="text-slate-600 text-xs">:</span>
                <input
                  type="number"
                  className="input w-full text-xs py-1 px-2 font-mono"
                  min={o.slice?.[0] ?? 0}
                  placeholder="null"
                  value={o.slice?.[1] ?? ''}
                  onChange={e => updateSliceEnd(i, o, e.target.value)}
                />
              </div>
            </div>
          </div>
        </div>
      ))}
      {saveErr && (
        <div className="text-xs text-red-400 flex items-center gap-1">
          <XCircle size={11} />
          {saveErr}
        </div>
      )}
    </div>
  )
}

// ── Param info tab ────────────────────────────────────────────────

export function ParamInfoTab({
  params,
  saving,
  saveErr,
  onSave,
}: {
  params: Record<string, [string, string]>
  saving: boolean
  saveErr: string | null
  onSave: (updated: Record<string, [string, string]>) => Promise<boolean>
}) {
  type Row = { name: string; meaning: string; unit: string }
  const [rows, setRows] = useState<Row[]>(
    Object.entries(params).map(([name, info]) => ({
      name,
      meaning: Array.isArray(info) ? info[0] : String(info),
      unit: Array.isArray(info) ? info[1] : '-',
    })),
  )
  const [dirty, setDirty] = useState(false)
  const [editingRow, setEditingRow] = useState<number | null>(null)

  const updateRow = (i: number, patch: Partial<Row>) => {
    setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
    setDirty(true)
  }
  const addRow = () => {
    setRows([...rows, { name: '', meaning: '', unit: '-' }])
    setDirty(true)
    setEditingRow(rows.length)
  }
  const removeRow = (i: number) => {
    setRows(rows.filter((_, j) => j !== i))
    setDirty(true)
  }

  const toParamInfo = () =>
    Object.fromEntries(rows.filter(r => r.name).map(r => [r.name, [r.meaning, r.unit] as [string, string]]))

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="label flex items-center gap-1.5">
          <Tag size={12} />
          param_info — {rows.length} 个参数
        </div>
        <div className="flex items-center gap-2">
          {dirty && (
            <>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-emerald-400"
                disabled={saving}
                onClick={async () => {
                  if (await onSave(toParamInfo())) setDirty(false)
                }}
              >
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
              </button>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                onClick={() => {
                  setRows(
                    Object.entries(params).map(([name, info]) => ({
                      name,
                      meaning: Array.isArray(info) ? info[0] : String(info),
                      unit: Array.isArray(info) ? info[1] : '-',
                    })),
                  )
                  setDirty(false)
                }}
              >
                <X size={12} /> 取消
              </button>
            </>
          )}
          <button className="btn-ghost py-0.5 px-2 text-xs" onClick={addRow}>
            <Plus size={11} /> 添加
          </button>
        </div>
      </div>
      <div className="rounded-xl border border-slate-700/30 overflow-hidden">
        <div className="list-table-scroll">
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
                        <input
                          className="input w-full text-xs py-1 px-2 font-mono"
                          value={r.name}
                          onChange={e => updateRow(i, { name: e.target.value })}
                          autoFocus
                        />
                      </td>
                      <td className="px-2 py-1.5">
                        <input
                          className="input w-full text-xs py-1 px-2"
                          value={r.meaning}
                          onChange={e => updateRow(i, { meaning: e.target.value })}
                        />
                      </td>
                      <td className="px-2 py-1.5">
                        <input
                          className="input w-full text-xs py-1 px-2 font-mono"
                          value={r.unit}
                          onChange={e => updateRow(i, { unit: e.target.value })}
                          onKeyDown={e => e.key === 'Enter' && setEditingRow(null)}
                        />
                      </td>
                      <td className="px-2 py-1.5">
                        <button className="btn-ghost py-0.5 px-1 text-emerald-400" onClick={() => setEditingRow(null)}>
                          <Check size={11} />
                        </button>
                      </td>
                    </>
                  ) : (
                    <>
                      <td
                        className="px-3 py-2 font-mono text-sky-300/80 cursor-pointer"
                        onClick={() => setEditingRow(i)}
                      >
                        {r.name || <span className="text-slate-600 italic">（空）</span>}
                      </td>
                      <td className="px-3 py-2 text-slate-300 cursor-pointer" onClick={() => setEditingRow(i)}>
                        {r.meaning}
                      </td>
                      <td
                        className="px-3 py-2 font-mono text-slate-500 text-xs cursor-pointer"
                        onClick={() => setEditingRow(i)}
                      >
                        {r.unit}
                      </td>
                      <td className="px-2 py-2">
                        <button
                          className="btn-ghost py-0.5 px-1 text-slate-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => removeRow(i)}
                        >
                          <Trash2 size={11} />
                        </button>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {saveErr && (
        <div className="text-xs text-red-400 flex items-center gap-1">
          <XCircle size={11} />
          {saveErr}
        </div>
      )}
    </div>
  )
}
