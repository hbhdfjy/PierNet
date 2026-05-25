import { useState, useRef, useEffect, useCallback } from 'react'
import useSWR from 'swr'
import { api } from '../../../lib/api'
import type { AgentTurnResponse, InterviewMessage } from '../../../lib/types'
import {
  Play,
  Send,
  Check,
  X,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Bot,
  User,
  Github,
  Sparkles,
  SkipForward,
  Plus,
  Trash2,
  GripVertical,
  Info,
} from 'lucide-react'
import { cn } from '../../../lib/utils'
import { parseIntegerField, parseIntegerText, parseOptionalIntegerField } from '../../integerInput'

interface Props {
  onRegistryUpdate: () => void
}

// ── 步骤进度指示器 ────────────────────────────────────────────────

function StepIndicator({ current, prefilled }: { current: number; prefilled: number[] }) {
  // scenario 模式（step 7）
  if (current === 7) {
    return (
      <div className="flex items-center gap-3 px-5 py-3 border-b border-slate-700/40 bg-slate-900/50 flex-shrink-0">
        <div className="w-6 h-6 rounded-full bg-violet-500 text-white flex items-center justify-center text-xs font-bold shadow-sm shadow-violet-900/30">
          1
        </div>
        <div>
          <span className="text-sm font-semibold text-violet-300">场景描述</span>
          <span className="text-xs text-slate-500 ml-2">第二步注册 · 只需一句话</span>
        </div>
      </div>
    )
  }

  const steps = [1, 2, 3, 4, 5, 6]
  const stepNames: Record<number, string> = {
    1: '物理域',
    2: '参数',
    3: '输出',
    4: '通道',
    5: '时间',
    6: '确认',
  }

  return (
    <div className="flex-shrink-0 border-b border-slate-700/40 bg-slate-900/50 px-5 py-3">
      <div className="flex items-center gap-0 overflow-x-auto">
        {/* GitHub 阶段标记 */}
        {current === 0 && (
          <div className="flex items-center gap-2 mr-4 px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/20 flex-shrink-0">
            <Github size={13} className="text-sky-400" />
            <span className="text-xs font-medium text-sky-300 whitespace-nowrap">GitHub 预填充</span>
          </div>
        )}

        {steps.map((step, i) => {
          const done = current > 0 && step < current
          const active = step === current
          const isPrefilled = prefilled.includes(step)
          return (
            <div key={step} className="flex items-center flex-shrink-0">
              {/* 步骤圆点 + 标签 */}
              <div className="flex flex-col items-center gap-1 w-14">
                <div
                  className={cn(
                    'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 relative',
                    done
                      ? 'bg-emerald-500 text-white shadow-sm shadow-emerald-900/40'
                      : active
                        ? 'bg-sky-500 text-white shadow-md shadow-sky-900/40 ring-2 ring-sky-400/30'
                        : isPrefilled
                          ? 'bg-sky-500/25 text-sky-300 ring-1 ring-sky-500/40'
                          : 'bg-slate-700/80 text-slate-500',
                  )}
                >
                  {done ? <Check size={12} /> : step}
                  {isPrefilled && !done && !active && (
                    <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-sky-400 rounded-full border border-slate-900" />
                  )}
                </div>
                <span
                  className={cn(
                    'text-xs whitespace-nowrap transition-colors leading-none',
                    active ? 'text-sky-300 font-medium' : done ? 'text-emerald-400/60' : 'text-slate-600',
                  )}
                >
                  {stepNames[step]}
                </span>
              </div>

              {/* 连接线：垂直居中对齐圆点（圆点高 7，标签约 12，总高 ~32，圆点顶部偏移约 0）*/}
              {i < steps.length - 1 && (
                <div
                  className={cn(
                    'w-6 h-px flex-shrink-0 -mt-4 transition-colors duration-300',
                    done ? 'bg-emerald-500/50' : 'bg-slate-700/60',
                  )}
                />
              )}
            </div>
          )
        })}

        {prefilled.length > 0 && current > 0 && (
          <div className="ml-auto pl-3 flex items-center gap-1 text-xs text-sky-400/60 whitespace-nowrap flex-shrink-0">
            <Sparkles size={10} />
            {prefilled.length} 步已预填
          </div>
        )}
      </div>
    </div>
  )
}

// ── 消息气泡 ─────────────────────────────────────────────────────

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-0.5">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.8s' }}
        />
      ))}
    </div>
  )
}

function MessageBubble({ msg }: { msg: InterviewMessage & { loading?: boolean; isGithub?: boolean } }) {
  const isAssistant = msg.role === 'assistant'
  const isGithub = (msg as { isGithub?: boolean }).isGithub
  const isLoading = (msg as { loading?: boolean }).loading

  return (
    <div className={cn('flex gap-3 mb-5', isAssistant ? 'justify-start' : 'justify-end')}>
      {/* 助手头像 */}
      {isAssistant && (
        <div
          className={cn(
            'w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 shadow-sm',
            isGithub
              ? 'bg-slate-700 border border-slate-600/60'
              : 'bg-gradient-to-br from-sky-500 to-sky-600 shadow-sky-900/30',
          )}
        >
          {isGithub ? <Github size={14} className="text-slate-300" /> : <Bot size={15} className="text-white" />}
        </div>
      )}

      {/* 气泡 */}
      <div className={cn('max-w-[80%] text-sm leading-relaxed', isAssistant ? 'order-2' : 'order-1')}>
        <div
          className={cn(
            'rounded-2xl px-4 py-3 shadow-sm',
            isAssistant
              ? 'bg-slate-800 border border-slate-700/60 text-slate-200 rounded-tl-sm'
              : 'bg-gradient-to-br from-sky-600 to-sky-700 text-white rounded-tr-sm shadow-sky-900/20',
          )}
        >
          {isLoading ? <TypingDots /> : <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>}
        </div>
      </div>

      {/* 用户头像 */}
      {!isAssistant && (
        <div className="w-8 h-8 rounded-xl bg-slate-700 border border-slate-600/60 flex items-center justify-center flex-shrink-0 mt-0.5 order-2">
          <User size={14} className="text-slate-300" />
        </div>
      )}
    </div>
  )
}

// ── GitHub 预填充结果横幅 ─────────────────────────────────────────

function GithubPrefillBanner({ summary, steps }: { summary: string; steps: number[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mb-3 bg-sky-500/8 border border-sky-500/20 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left hover:bg-sky-500/5 transition-colors"
      >
        <Sparkles size={14} className="text-sky-400 flex-shrink-0" />
        <span className="text-sm text-sky-300 flex-1">{summary}</span>
        <span className="text-xs text-sky-500/60">步骤 {steps.join('、')} 已预填</span>
        {open ? (
          <ChevronUp size={12} className="text-sky-500/60" />
        ) : (
          <ChevronDown size={12} className="text-sky-500/60" />
        )}
      </button>
      {open && (
        <div className="px-4 pb-3 text-xs text-sky-400/70 border-t border-sky-500/15 pt-2">
          预填的步骤将直接展示结果供确认，你可以直接确认或修改 JSON 后再确认。
        </div>
      )}
    </div>
  )
}

// ── 结构化提取结果编辑器 ──────────────────────────────────────────

// Step 1：domain_context + output_description
function Step1Editor({
  data,
  onChange,
}: {
  data: Record<string, unknown>
  onChange: (d: Record<string, unknown>) => void
}) {
  return (
    <div className="space-y-3">
      <div>
        <label className="label block mb-1.5 text-xs">
          domain_context <span className="text-red-400 font-normal normal-case">≥50字符</span>
        </label>
        <textarea
          className="input w-full text-sm leading-relaxed resize-none"
          rows={5}
          placeholder="描述物理域背景、控制方程、数值方法…"
          value={String(data.domain_context ?? '')}
          onChange={e => onChange({ ...data, domain_context: e.target.value })}
        />
        <p className="text-xs text-slate-600 mt-1">{String(data.domain_context ?? '').length} 字符</p>
      </div>
      <div>
        <label className="label block mb-1.5 text-xs">
          output_description
          <span className="ml-2 text-slate-600 font-normal normal-case tracking-normal">
            必须含 {'{ch}'} 和 {'{ts}'}
          </span>
        </label>
        <input
          className="input w-full text-sm font-mono"
          placeholder="如：{ch} 观测井 × {ts} 天的水力水头（m）"
          value={String(data.output_description ?? '')}
          onChange={e => onChange({ ...data, output_description: e.target.value })}
        />
      </div>
      {(data.n_channels !== undefined || data.n_timesteps !== undefined) && (
        <div className="grid grid-cols-2 gap-3">
          {['n_channels', 'n_timesteps'].map(
            k =>
              data[k] !== undefined && (
                <div key={k}>
                  <label className="label block mb-1.5 text-xs">{k}</label>
                  <input
                    type="number"
                    className="input w-full text-sm"
                    min={0}
                    value={Number(data[k] ?? 0)}
                    onChange={e =>
                      onChange({
                        ...data,
                        [k]: parseIntegerField(e.target.value, Number(data[k] ?? 0) || 0, { min: 0 }),
                      })
                    }
                  />
                </div>
              ),
          )}
        </div>
      )}
    </div>
  )
}

// Step 2：param_info 表格
function Step2Editor({
  data,
  onChange,
}: {
  data: Record<string, unknown>
  onChange: (d: Record<string, unknown>) => void
}) {
  const params = (data.param_info ?? {}) as Record<string, [string, string]>
  const entries = Object.entries(params)

  const setParam = (name: string, meaning: string, unit: string) => {
    onChange({ ...data, param_info: { ...params, [name]: [meaning, unit] } })
  }
  const removeParam = (name: string) => {
    const next = { ...params }
    delete next[name]
    onChange({ ...data, param_info: next })
  }

  return (
    <div className="space-y-2">
      <div className="label text-xs mb-1">param_info — {entries.length} 个参数</div>
      <div className="rounded-xl border border-slate-700/30 overflow-hidden">
        <div className="list-table-scroll">
          <table className="w-full text-xs">
            <thead className="bg-slate-900/60">
              <tr className="border-b border-slate-700/30">
                <th className="px-3 py-2 text-left label">参数名</th>
                <th className="px-3 py-2 text-left label">物理含义</th>
                <th className="px-3 py-2 text-left label">单位</th>
                <th className="px-3 py-2 w-8" />
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, info]) => {
                const meaning = Array.isArray(info) ? info[0] : String(info)
                const unit = Array.isArray(info) ? info[1] : ''
                return (
                  <tr key={name} className="border-b border-slate-800/40 group">
                    <td className="px-3 py-1.5 font-mono text-sky-300/80 text-xs whitespace-nowrap">{name}</td>
                    <td className="px-2 py-1">
                      <input
                        className="input w-full text-xs py-0.5 px-2 bg-transparent border-transparent hover:border-slate-600 focus:border-sky-500/50"
                        value={meaning}
                        onChange={e => setParam(name, e.target.value, unit)}
                      />
                    </td>
                    <td className="px-2 py-1 w-28">
                      <input
                        className="input w-full text-xs py-0.5 px-2 font-mono bg-transparent border-transparent hover:border-slate-600 focus:border-sky-500/50"
                        value={unit}
                        onChange={e => setParam(name, meaning, e.target.value)}
                      />
                    </td>
                    <td className="px-2 py-1">
                      <button
                        className="btn-ghost py-0.5 px-1 text-slate-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => removeParam(name)}
                      >
                        <Trash2 size={11} />
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// Step 3：output_info 列表
function Step3Editor({
  data,
  onChange,
}: {
  data: Record<string, unknown>
  onChange: (d: Record<string, unknown>) => void
}) {
  type OI = { name: string; name_zh?: string; description: string; unit: string; slice: [number, number | null] }
  const items = (data.output_info ?? []) as OI[]

  const setItem = (i: number, patch: Partial<OI>) => {
    const next = items.map((x, j) => (j === i ? { ...x, ...patch } : x))
    onChange({ ...data, output_info: next })
  }
  const addItem = () =>
    onChange({
      ...data,
      output_info: [...items, { name: '', description: '', unit: '', slice: [0, null] as [number, null] }],
    })
  const updateSliceStart = (i: number, item: OI, value: string) => {
    const start = parseIntegerField(value, item.slice[0], { min: 0 })
    const end = item.slice[1]
    setItem(i, { slice: [start, end !== null && end < start ? start : end] })
  }
  const updateSliceEnd = (i: number, item: OI, value: string) =>
    setItem(i, { slice: [item.slice[0], parseOptionalIntegerField(value, item.slice[1], { min: item.slice[0] })] })
  const removeItem = (i: number) => onChange({ ...data, output_info: items.filter((_, j) => j !== i) })

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="label text-xs">output_info — {items.length} 个通道组</span>
        <button className="btn-ghost py-0.5 px-2 text-xs" onClick={addItem}>
          <Plus size={11} /> 添加
        </button>
      </div>
      {items.map((item, i) => (
        <div key={i} className="rounded-xl border border-slate-700/30 p-3 space-y-2 bg-slate-900/30 group">
          <div className="flex items-center gap-2">
            <GripVertical size={12} className="text-slate-600 flex-shrink-0" />
            <span className="text-xs text-slate-500 font-mono flex-shrink-0">#{i}</span>
            <div className="flex-1 grid grid-cols-2 gap-2">
              <div>
                <label className="label text-xs block mb-1">name</label>
                <input
                  className="input w-full text-xs py-1 px-2 font-mono"
                  placeholder="如 hydraulic_head"
                  value={item.name}
                  onChange={e => setItem(i, { name: e.target.value })}
                />
              </div>
              <div>
                <label className="label text-xs block mb-1">name_zh（可选）</label>
                <input
                  className="input w-full text-xs py-1 px-2"
                  placeholder="如 水力水头"
                  value={item.name_zh ?? ''}
                  onChange={e => setItem(i, { name_zh: e.target.value })}
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
          <div className="grid grid-cols-3 gap-2 pl-6">
            <div>
              <label className="label text-xs block mb-1">description</label>
              <input
                className="input w-full text-xs py-1 px-2"
                placeholder="物理含义描述"
                value={item.description}
                onChange={e => setItem(i, { description: e.target.value })}
              />
            </div>
            <div>
              <label className="label text-xs block mb-1">unit</label>
              <input
                className="input w-full text-xs py-1 px-2 font-mono"
                placeholder="如 m"
                value={item.unit}
                onChange={e => setItem(i, { unit: e.target.value })}
              />
            </div>
            <div>
              <label className="label text-xs block mb-1">slice [start, end]</label>
              <div className="flex gap-1 items-center">
                <input
                  type="number"
                  className="input w-full text-xs py-1 px-2"
                  placeholder="0"
                  min={0}
                  value={item.slice[0]}
                  onChange={e => updateSliceStart(i, item, e.target.value)}
                />
                <span className="text-slate-600 text-xs flex-shrink-0">:</span>
                <input
                  type="number"
                  className="input w-full text-xs py-1 px-2 font-mono"
                  min={item.slice[0]}
                  placeholder="null"
                  value={item.slice[1] === null ? '' : String(item.slice[1])}
                  onChange={e => updateSliceEnd(i, item, e.target.value)}
                />
              </div>
            </div>
          </div>
        </div>
      ))}
      {items.length === 0 && (
        <button
          onClick={addItem}
          className="w-full py-3 border border-dashed border-slate-700/50 rounded-xl text-slate-600 text-xs hover:text-slate-400 hover:border-slate-600 transition-colors"
        >
          <Plus size={12} className="inline mr-1" /> 添加通道组
        </button>
      )}
    </div>
  )
}

// Step 4：通道降采样策略
function Step4Editor({
  data,
  onChange,
}: {
  data: Record<string, unknown>
  onChange: (d: Record<string, unknown>) => void
}) {
  const cl = String(data.channel_level ?? 'row')
  const isRow = cl === 'row'
  const fixedCh = data.fixed_channels
  const fixedStr = fixedCh === null ? 'null' : JSON.stringify(fixedCh)

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label block mb-1.5 text-xs">channel_level</label>
          <div className="flex gap-2">
            {['row', 'output_info'].map(v => (
              <button
                key={v}
                onClick={() => onChange({ ...data, channel_level: v })}
                className={cn(
                  'flex-1 py-1.5 rounded-lg border text-xs font-mono transition-all',
                  cl === v
                    ? 'bg-sky-500/15 border-sky-500/40 text-sky-300'
                    : 'bg-slate-800/40 border-slate-700/40 text-slate-500 hover:text-slate-300',
                )}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="label block mb-1.5 text-xs">fixed_channels</label>
          <input
            className="input w-full text-xs py-1.5 font-mono"
            placeholder='null 或 [0,1,2] 或 ["bus_voltages"]'
            value={fixedStr}
            onChange={e => {
              try {
                onChange({ ...data, fixed_channels: JSON.parse(e.target.value) })
              } catch {
                /* typing */
              }
            }}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label block mb-1.5 text-xs">channel_min</label>
          <input
            type="number"
            className="input w-full text-xs py-1.5"
            min={1}
            value={Number(data.channel_min ?? 1)}
            onChange={e =>
              onChange({
                ...data,
                channel_min: parseIntegerField(e.target.value, Number(data.channel_min ?? 1) || 1, { min: 1 }),
              })
            }
          />
        </div>
        <div>
          <label className="label block mb-1.5 text-xs">
            channel_max <span className="text-slate-600 font-normal normal-case">（null=全选）</span>
          </label>
          <input
            className="input w-full text-xs py-1.5 font-mono"
            placeholder="null"
            value={data.channel_max === null || data.channel_max === undefined ? '' : String(data.channel_max)}
            onChange={e =>
              onChange({
                ...data,
                channel_max: parseOptionalIntegerField(
                  e.target.value,
                  data.channel_max === null || data.channel_max === undefined ? null : Number(data.channel_max) || null,
                  { min: 1 },
                ),
              })
            }
          />
        </div>
      </div>
      {isRow && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label block mb-1.5 text-xs">channel_name_template</label>
            <input
              className="input w-full text-xs py-1.5 font-mono"
              placeholder="well {i}"
              value={String(data.channel_name_template ?? '')}
              onChange={e => onChange({ ...data, channel_name_template: e.target.value })}
            />
            <p className="text-xs text-slate-600 mt-1">必须含 {'{' + 'i' + '}'}</p>
          </div>
          <div>
            <label className="label block mb-1.5 text-xs">
              channel_name_template_zh <span className="text-slate-600 font-normal">（可选）</span>
            </label>
            <input
              className="input w-full text-xs py-1.5 font-mono"
              placeholder="第{i}号观测井"
              value={String(data.channel_name_template_zh ?? '')}
              onChange={e => onChange({ ...data, channel_name_template_zh: e.target.value })}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// Step 5：时间采样模式
type TimeMode = 'monthly' | 'weekly' | 'full' | 'every_n'

const TIME_MODE_OPTIONS: { value: TimeMode; label: string; desc: string }[] = [
  { value: 'monthly', label: '月度', desc: '每月取1个点，共12点' },
  { value: 'weekly', label: '每周', desc: '每7步取1个点，最多52点' },
  { value: 'full', label: '全量', desc: '保留所有时间步' },
  { value: 'every_n', label: '自定义步长', desc: '每 N 步取1个点' },
]

// 从 fixed_time_mode 字符串反推 TimeMode 类型和 every_n 的步长
function parseTimeMode(fixed: string): { type: TimeMode; step: number } {
  if (fixed === 'monthly') return { type: 'monthly', step: 10 }
  if (fixed === 'weekly') return { type: 'weekly', step: 7 }
  if (fixed === 'full') return { type: 'full', step: 1 }
  if (fixed.startsWith('every_')) {
    const n = parseIntegerText(fixed.slice(6), { min: 1 })
    return { type: 'every_n', step: n ?? 10 }
  }
  return { type: 'monthly', step: 10 }
}

// 根据 TimeMode 类型和步长生成 fixed_time_mode 字符串及描述
function buildTimeMode(type: TimeMode, step: number): { indices: string; desc_en: string; desc_zh: string } {
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
    return {
      indices: 'full',
      desc_en: 'full time series, all time points',
      desc_zh: '全量时间序列，所有时间步',
    }
  // every_n
  return {
    indices: `every_${step}`,
    desc_en: `every ${step} steps`,
    desc_zh: `每 ${step} 步取1个点`,
  }
}

function Step5Editor({
  data,
  onChange,
}: {
  data: Record<string, unknown>
  onChange: (d: Record<string, unknown>) => void
}) {
  const fixed = String(data.fixed_time_mode ?? 'monthly')
  const { type: selectedType, step: everyStep } = parseTimeMode(fixed)
  const [stepInput, setStepInput] = useState(String(everyStep))

  const select = (type: TimeMode, step?: number) => {
    const n = step ?? (type === 'every_n' ? parseIntegerField(stepInput, 10, { min: 1 }) : 10)
    const modeStr = type === 'every_n' ? `every_${n}` : type
    const { indices, desc_en, desc_zh } = buildTimeMode(type, n)
    onChange({
      ...data,
      fixed_time_mode: modeStr,
      time_modes: [{ name: modeStr, indices, desc_en, desc_zh }],
    })
  }

  const handleStepChange = (val: string) => {
    setStepInput(val)
    const n = parseIntegerText(val, { min: 1 })
    if (n !== null) select('every_n', n)
  }

  return (
    <div className="space-y-3">
      <span className="label text-xs">时间采样模式（点击选择）</span>

      <div className="grid grid-cols-2 gap-2">
        {TIME_MODE_OPTIONS.map(opt => {
          const active = selectedType === opt.value
          return (
            <button
              key={opt.value}
              onClick={() => {
                if (opt.value !== 'every_n') select(opt.value)
                else select('every_n')
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

      {/* 当前配置预览 */}
      <div className="rounded-lg bg-slate-900/50 border border-slate-700/30 px-3 py-2">
        <span className="label text-xs">当前：</span>
        <span className="font-mono text-xs text-sky-300 ml-1">{fixed}</span>
      </div>
    </div>
  )
}

// ── 主提取结果预览（按 step 分发）────────────────────────────────

function ExtractionPreview({
  step,
  extracted,
  uncertain,
  onDataChange,
}: {
  step: number
  extracted: Record<string, unknown>
  uncertain: boolean
  onDataChange: (d: Record<string, unknown>) => void
}) {
  const [showRaw, setShowRaw] = useState(false)
  const [rawVal, setRawVal] = useState(() => JSON.stringify(extracted, null, 2))
  const [rawErr, setRawErr] = useState<string | null>(null)

  // 当 extracted 外部变化时同步 rawVal
  useEffect(() => {
    setRawVal(JSON.stringify(extracted, null, 2))
    setRawErr(null)
  }, [extracted])

  const handleRawChange = (v: string) => {
    setRawVal(v)
    try {
      onDataChange(JSON.parse(v))
      setRawErr(null)
    } catch {
      setRawErr('JSON 格式错误')
    }
  }

  const stepLabels: Record<number, string> = {
    1: '物理域描述',
    2: '参数含义',
    3: '输出通道结构',
    4: '通道采样策略',
    5: '时间采样模式',
    6: '完整预览',
  }

  return (
    <div
      className={cn(
        'mb-3 rounded-xl overflow-hidden border',
        uncertain ? 'border-amber-500/30' : 'border-slate-700/40',
      )}
    >
      {/* 标题栏：只在不确定时显示警告，否则仅显示步骤名和 JSON 切换 */}
      <div
        className={cn(
          'flex items-center gap-2 px-3 py-2 border-b',
          uncertain ? 'bg-amber-500/8 border-amber-500/20' : 'bg-slate-800/50 border-slate-700/30',
        )}
      >
        {uncertain && <AlertTriangle size={13} className="text-amber-400 flex-shrink-0" />}
        <span className={cn('text-xs flex-1 font-medium', uncertain ? 'text-amber-300' : 'text-slate-400')}>
          {uncertain ? '⚠️ 提取结果不确定，请仔细核对' : (stepLabels[step] ?? '编辑结果')}
        </span>
        <button
          onClick={() => setShowRaw(v => !v)}
          className={cn(
            'btn-ghost py-0.5 px-2 text-xs transition-colors',
            showRaw ? 'text-sky-400 bg-sky-500/10' : 'text-slate-500',
          )}
        >
          {showRaw ? '表单' : 'JSON'}
        </button>
      </div>

      {/* 内容区 */}
      <div className="p-3">
        {showRaw ? (
          <div className="space-y-1">
            <textarea
              className="w-full bg-slate-950/60 font-mono text-xs text-slate-300 p-2.5 resize-y rounded-lg border border-slate-700/40 focus:outline-none focus:ring-1 focus:ring-sky-500/40 leading-relaxed"
              rows={Math.min(20, rawVal.split('\n').length + 1)}
              value={rawVal}
              onChange={e => handleRawChange(e.target.value)}
            />
            {rawErr && <p className="text-xs text-red-400">{rawErr}</p>}
          </div>
        ) : (
          <>
            {step === 1 && <Step1Editor data={extracted} onChange={onDataChange} />}
            {step === 2 && <Step2Editor data={extracted} onChange={onDataChange} />}
            {step === 3 && <Step3Editor data={extracted} onChange={onDataChange} />}
            {step === 4 && <Step4Editor data={extracted} onChange={onDataChange} />}
            {step === 5 && <Step5Editor data={extracted} onChange={onDataChange} />}
            {(step === 6 || step === 7) && (
              <div className="text-xs text-slate-500 flex items-center gap-1.5">
                <Info size={12} />
                点击右上角「JSON」可查看完整内容并手动编辑
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────

export default function InterviewPanel({ onRegistryUpdate }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [step, setStep] = useState(0)
  const [agentStatus, setAgentStatus] = useState<'interviewing' | 'confirming' | 'done' | 'error'>('interviewing')
  const [prefilledSteps, setPrefilledSteps] = useState<number[]>([])
  const [githubPrefillInfo, setGithubPrefillInfo] = useState<{ summary: string; steps: number[] } | null>(null)

  const [messages, setMessages] = useState<(InterviewMessage & { loading?: boolean; isGithub?: boolean })[]>([])
  const [extracted, setExtracted] = useState<Record<string, unknown> | null>(null)
  const [extractionUncertain, setExtractionUncertain] = useState(false)
  const [editJson, setEditJson] = useState('')
  const [registryKey, setRegistryKey] = useState<string | null>(null)

  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 设置表单
  const [mode, setMode] = useState<'simulator' | 'scenario'>('simulator')
  const [simulator, setSimulator] = useState('')
  const [scenario, setScenario] = useState('')
  const [hdf5Path, setHdf5Path] = useState('')

  // 从 registry 提取已注册的 simulator 列表（用于 scenario 模式的下拉选择）
  const { data: registry } = useSWR<Record<string, unknown>>('registry-for-interview', () => api.getRegistry(), {
    revalidateOnFocus: false,
  })
  const registeredSimulators: string[] = (() => {
    if (!registry) return []
    const sims = new Set<string>()
    for (const key of Object.keys(registry)) {
      // key 格式：'simulator' 或 'simulator/scenario'
      sims.add(key.split('/')[0])
    }
    return Array.from(sims).sort()
  })()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const addMessage = useCallback(
    (role: 'assistant' | 'user', content: string, stepN: number, extra?: { isGithub?: boolean }) => {
      setMessages(prev => [...prev, { role, content, step: stepN, ts: Date.now() / 1000, ...extra }])
    },
    [],
  )

  const applyResponse = useCallback(
    (resp: AgentTurnResponse) => {
      setStep(resp.step)

      if (resp.question) {
        const isGithubMsg = resp.step === 0
        addMessage('assistant', resp.question, resp.step, { isGithub: isGithubMsg })
      }

      if (resp.extracted) {
        setExtracted(resp.extracted)
        setEditJson(JSON.stringify(resp.extracted, null, 2))
      } else {
        setExtracted(null)
      }

      setExtractionUncertain(resp.extraction_uncertain)

      if (resp.needs_confirmation) {
        setAgentStatus('confirming')
      } else {
        setAgentStatus('interviewing')
      }

      // GitHub 预填充结果
      if (resp.github_prefilled) {
        const gp = resp.github_prefilled as { steps: number[]; summary: string }
        setPrefilledSteps(gp.steps)
        if (gp.steps.length > 0) {
          setGithubPrefillInfo({ summary: gp.summary, steps: gp.steps })
        }
      }

      if (resp.done) {
        setAgentStatus('done')
        setRegistryKey(resp.registry_key)
        onRegistryUpdate()
      }
      if (resp.error) {
        setError(resp.error)
        setAgentStatus('error')
      }
    },
    [addMessage, onRegistryUpdate],
  )

  const handleStart = async () => {
    if (!simulator.trim()) {
      setError('请填写仿真器名称')
      return
    }
    if (mode === 'scenario' && !scenario.trim()) {
      setError('场景模式需要填写场景名称')
      return
    }
    setError(null)
    setLoading(true)
    setMessages([])
    setExtracted(null)
    setRegistryKey(null)
    setPrefilledSteps([])
    setGithubPrefillInfo(null)
    try {
      const resp = await api.startInterview({
        simulator: simulator.trim(),
        scenario: scenario.trim() || '_',
        hdf5_path: hdf5Path.trim() || undefined,
        mode,
      })
      setSessionId(resp.session_id)
      applyResponse(resp)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLoading(false)
    }
  }

  const handleSend = async (overrideMsg?: string) => {
    const msg = overrideMsg ?? inputText.trim()
    if (!msg || !sessionId || loading) return
    if (!overrideMsg) setInputText('')
    setError(null)
    addMessage('user', msg, step)

    // 加载指示（GitHub 阶段用特殊样式）
    const isGithubStep = step === 0
    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: '',
        step,
        ts: Date.now() / 1000,
        loading: true,
        isGithub: isGithubStep,
      },
    ])
    setLoading(true)

    try {
      const resp = await api.sendInterviewMessage(sessionId, msg)
      setMessages(prev => prev.filter(m => !(m as { loading?: boolean }).loading))
      applyResponse(resp)
    } catch (e: unknown) {
      setMessages(prev => prev.filter(m => !(m as { loading?: boolean }).loading))
      setError(e instanceof Error ? e.message : '发送失败')
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  const handleConfirm = async () => {
    if (!sessionId || loading) return
    setLoading(true)
    setError(null)
    // extracted 已通过 onDataChange 实时同步，直接使用
    // editJson 作为兜底（用户切到 JSON 模式手动编辑时）
    let editedData: Record<string, unknown> | undefined
    if (extracted) {
      try {
        editedData = editJson ? JSON.parse(editJson) : extracted
      } catch {
        setError('JSON 格式错误，请检查编辑内容')
        setLoading(false)
        return
      }
    }
    try {
      const resp = await api.confirmInterviewStep(sessionId, true, editedData)
      setMessages(prev => prev.filter(m => !(m as { loading?: boolean }).loading))
      setExtracted(null)
      applyResponse(resp)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '确认失败')
    } finally {
      setLoading(false)
    }
  }

  const handleReject = async () => {
    if (!sessionId || loading) return
    setLoading(true)
    setError(null)
    try {
      const resp = await api.confirmInterviewStep(sessionId, false)
      setExtracted(null)
      applyResponse(resp)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '操作失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCancel = async () => {
    if (sessionId) await api.cancelInterview(sessionId).catch(() => {})
    setSessionId(null)
    setMessages([])
    setExtracted(null)
    setAgentStatus('interviewing')
    setStep(0)
    setError(null)
    setRegistryKey(null)
    setPrefilledSteps([])
    setGithubPrefillInfo(null)
    // 不重置 mode/simulator/scenario，方便连续注册多个场景
  }

  // ── HDF5 格式要求说明卡片 ──────────────────────────────────────
  function HDF5FormatCard() {
    const [open, setOpen] = useState(false)
    return (
      <div className="rounded-2xl border border-slate-700/50 bg-slate-800/40 overflow-hidden">
        <button
          onClick={() => setOpen(o => !o)}
          className="w-full flex items-center justify-between px-5 py-3 hover:bg-slate-700/20 transition-colors text-left"
        >
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-lg bg-amber-500/15 border border-amber-500/25 flex items-center justify-center flex-shrink-0">
              <Info size={13} className="text-amber-400" />
            </div>
            <span className="text-sm font-semibold text-slate-200">HDF5 文件格式要求</span>
            <span className="badge bg-amber-500/12 text-amber-400/90 border border-amber-500/25 text-xs">必读</span>
          </div>
          <div className="flex items-center gap-2 text-slate-500">
            <span className="text-xs">{open ? '收起' : '展开查看'}</span>
            {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </div>
        </button>

        {open && (
          <div className="border-t border-slate-700/40 p-5 space-y-5">
            {/* 第一行：文件命名 + 数据集结构（上下）*/}
            <div className="space-y-4">
              {/* 文件命名 */}
              <div>
                <div className="flex items-center gap-2 mb-2.5">
                  <span className="w-5 h-5 rounded-md bg-sky-500/20 text-sky-400 flex items-center justify-center text-xs font-bold flex-shrink-0">
                    1
                  </span>
                  <span className="text-xs font-semibold text-slate-200">文件命名规则</span>
                </div>
                <div className="bg-slate-900/60 rounded-xl p-3.5 space-y-2 font-mono text-xs border border-slate-700/30">
                  <div>
                    <span className="text-emerald-400">✓ </span>
                    <span className="text-sky-300">{'{simulator}'}</span>
                    <span className="text-slate-400">_</span>
                    <span className="text-amber-300">{'{scenario}'}</span>
                    <span className="text-slate-400">.h5</span>
                  </div>
                  <p className="font-sans text-slate-500 text-xs pl-4">
                    放置于 <span className="font-mono text-slate-400">data/{'{simulator}'}/ </span>目录下
                  </p>
                  <div className="pl-4 space-y-1 pt-1 border-t border-slate-700/30">
                    <div className="flex items-start gap-1.5 text-slate-500 flex-wrap">
                      <span className="text-slate-600 flex-shrink-0">例：</span>
                      <span className="text-slate-400">
                        data/modflow/<span className="text-sky-400/70">modflow</span>_
                        <span className="text-amber-400/70">unified_aquifer</span>.h5
                      </span>
                    </div>
                    <div className="flex items-start gap-1.5 text-slate-500 flex-wrap">
                      <span className="text-slate-600 flex-shrink-0">例：</span>
                      <span className="text-slate-400">
                        data/power_flow/<span className="text-sky-400/70">power_flow</span>_
                        <span className="text-amber-400/70">ieee14_baseload</span>.h5
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* 数据集结构 */}
              <div>
                <div className="flex items-center gap-2 mb-2.5">
                  <span className="w-5 h-5 rounded-md bg-sky-500/20 text-sky-400 flex items-center justify-center text-xs font-bold flex-shrink-0">
                    2
                  </span>
                  <span className="text-xs font-semibold text-slate-200">HDF5 内部结构</span>
                </div>
                <div className="space-y-1.5">
                  <div className="text-xs text-slate-500 mb-1 pl-1">数据集（datasets）</div>
                  {[
                    { key: 'timeseries', shape: '[N, n_ch, n_ts]', dtype: 'float32', note: '时序数据' },
                    { key: 'params', shape: '[N, 18]', dtype: 'float32', note: '18维统一参数（经 pipeline 转换）' },
                    { key: 'param_names', shape: '[18]', dtype: 'bytes', note: '统一参数名列表（UTF-8 编码）' },
                  ].map(r => (
                    <div
                      key={r.key}
                      className="flex items-center gap-3 bg-slate-900/60 border border-slate-700/30 rounded-lg px-3 py-2"
                    >
                      <span className="font-mono text-sky-300 text-xs w-28 flex-shrink-0">{r.key}</span>
                      <span className="font-mono text-amber-300/80 text-xs w-32 flex-shrink-0">{r.shape}</span>
                      <span className="font-mono text-violet-400/70 text-xs w-14 flex-shrink-0">{r.dtype}</span>
                      <span className="text-slate-500 text-xs">{r.note}</span>
                    </div>
                  ))}
                  <div className="text-xs text-slate-500 mt-2 mb-1 pl-1">根属性（attrs）</div>
                  {[
                    { key: 'n_samples', val: 'int', note: '样本数 N' },
                    { key: 'n_channels', val: 'int', note: '通道数 n_ch' },
                    { key: 'n_timesteps', val: 'int', note: '时间步数 n_ts' },
                    { key: 'n_params', val: 'int', note: '参数维度（固定为 18）' },
                  ].map(r => (
                    <div
                      key={r.key}
                      className="flex items-center gap-3 bg-slate-900/40 border border-slate-700/20 rounded-lg px-3 py-1.5"
                    >
                      <span className="font-mono text-emerald-400/70 text-xs w-28 flex-shrink-0">{r.key}</span>
                      <span className="font-mono text-slate-500 text-xs w-32 flex-shrink-0">{r.val}</span>
                      <span className="font-mono text-slate-700 text-xs w-14 flex-shrink-0">attr</span>
                      <span className="text-slate-500 text-xs">{r.note}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 第二行：维度约定（全宽）*/}
            <div>
              <div className="flex items-center gap-2 mb-2.5">
                <span className="w-5 h-5 rounded-md bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs font-bold flex-shrink-0">
                  !
                </span>
                <span className="text-xs font-semibold text-slate-200">维度顺序（严格约定）</span>
              </div>
              <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
                <div className="flex items-center gap-6 flex-wrap">
                  <div className="font-mono text-sm">
                    <span className="text-slate-500">timeseries[i] → </span>
                    <span className="text-amber-300 font-semibold">(n_channels, n_timesteps)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-2 text-center">
                      <div className="text-amber-400 font-mono font-semibold">axis 0</div>
                      <div className="text-slate-400 text-xs mt-0.5">通道（观测井 / 发电机 / 物理量）</div>
                    </div>
                    <div className="bg-sky-500/10 border border-sky-500/20 rounded-lg px-4 py-2 text-center">
                      <div className="text-sky-400 font-mono font-semibold">axis 1</div>
                      <div className="text-slate-400 text-xs mt-0.5">时间步（时序长度）</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 text-amber-500/70 text-xs">
                    <span>⚠</span>
                    <span>
                      若原始为 <span className="font-mono">(ts, ch)</span>，需在 pipeline 中{' '}
                      <span className="font-mono text-amber-300">.T</span> 转置后存入
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* 第三行：Simulator 参考表（全宽）*/}
            <div>
              <div className="flex items-center gap-2 mb-2.5">
                <span className="w-5 h-5 rounded-md bg-sky-500/20 text-sky-400 flex items-center justify-center text-xs font-bold flex-shrink-0">
                  3
                </span>
                <span className="text-xs font-semibold text-slate-200">各仿真器输出形状参考</span>
              </div>
              <div className="bg-slate-900/60 rounded-xl border border-slate-700/30 overflow-hidden">
                <div className="list-table-scroll">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700/40 bg-slate-900/40">
                        <th className="px-4 py-2 text-left text-slate-500 font-normal">仿真器</th>
                        <th className="px-4 py-2 text-left text-slate-500 font-normal font-mono">timeseries[i] 形状</th>
                        <th className="px-4 py-2 text-left text-slate-500 font-normal">通道含义</th>
                        <th className="px-4 py-2 text-left text-slate-500 font-normal">时间含义</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ['modflow', '(5, 365)', '5 口观测井', '365 天（日步长）'],
                        ['simpeg', '(1, 100)', '1 个测量通道', '100 个测量点'],
                        ['power_flow', '(43, 365)', '14V+14θ+15P（IEEE14）', '365 天'],
                        ['transient', '(5, 1000)', '5 台发电机转子角（IEEE14）', '1000 步（100Hz×10s）'],
                        ['gcam', '(5, 16)', '5 个能源/气候变量', '16 个 5 年步长'],
                      ].map(([sim, shape, ch, ts], i, arr) => (
                        <tr key={sim} className={i < arr.length - 1 ? 'border-b border-slate-800/40' : ''}>
                          <td className="px-4 py-2 font-mono text-sky-300/80">{sim}</td>
                          <td className="px-4 py-2 font-mono text-amber-300/80">{shape}</td>
                          <td className="px-4 py-2 text-slate-400">{ch}</td>
                          <td className="px-4 py-2 text-slate-400">{ts}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // ── 未开始：设置表单 ──
  if (!sessionId) {
    const canStart = simulator.trim() && (mode === 'simulator' || scenario.trim())
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-y-auto">
        <div className="w-full max-w-2xl space-y-5">
          {/* 页头 */}
          <div className="text-center">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-sky-500/20 to-sky-600/10 border border-sky-500/25 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-sky-900/10">
              <Bot size={26} className="text-sky-400" />
            </div>
            <h2 className="text-xl font-bold text-white">智能注册向导</h2>
            <p className="text-sm text-slate-400 mt-1.5">通过对话引导，自动生成 registry 元数据</p>
          </div>

          {/* 注册模式切换 */}
          <div className="grid grid-cols-2 gap-3">
            {[
              {
                value: 'simulator' as const,
                label: '注册仿真器',
                sub: '第一步',
                desc: '物理域背景、参数含义、输出结构等完整元数据',
                steps: '6 步',
                color: 'sky' as const,
              },
              {
                value: 'scenario' as const,
                label: '注册场景描述',
                sub: '第二步',
                desc: '为具体场景补充一句话物理设置描述',
                steps: '1 步',
                color: 'violet' as const,
              },
            ].map(opt => {
              const active = mode === opt.value
              return (
                <button
                  key={opt.value}
                  onClick={() => setMode(opt.value)}
                  className={cn(
                    'relative flex flex-col items-start text-left px-4 py-4 rounded-2xl border-2 transition-all duration-200',
                    active
                      ? opt.color === 'sky'
                        ? 'bg-sky-500/10 border-sky-500/50 shadow-md shadow-sky-900/20'
                        : 'bg-violet-500/10 border-violet-500/50 shadow-md shadow-violet-900/20'
                      : 'bg-slate-800/40 border-slate-700/40 hover:border-slate-600/60 hover:bg-slate-800/60',
                  )}
                >
                  {/* 顶部：步骤标签 + 步骤数，右上角对勾不再与步骤数重叠 */}
                  <div className="flex items-center gap-2 w-full mb-2 pr-7">
                    <span
                      className={cn(
                        'text-xs font-semibold px-2 py-0.5 rounded-md flex-shrink-0',
                        active
                          ? opt.color === 'sky'
                            ? 'bg-sky-500/20 text-sky-300'
                            : 'bg-violet-500/20 text-violet-300'
                          : 'bg-slate-700/60 text-slate-500',
                      )}
                    >
                      {opt.sub}
                    </span>
                    <span
                      className={cn(
                        'text-xs',
                        active ? (opt.color === 'sky' ? 'text-sky-400/60' : 'text-violet-400/60') : 'text-slate-600',
                      )}
                    >
                      {opt.steps}
                    </span>
                  </div>
                  <span
                    className={cn(
                      'text-sm font-semibold mb-1',
                      active ? (opt.color === 'sky' ? 'text-sky-200' : 'text-violet-200') : 'text-slate-300',
                    )}
                  >
                    {opt.label}
                  </span>
                  <p
                    className={cn(
                      'text-xs leading-relaxed',
                      active ? (opt.color === 'sky' ? 'text-sky-400/70' : 'text-violet-400/70') : 'text-slate-500',
                    )}
                  >
                    {opt.desc}
                  </p>
                  {/* 对勾：绝对定位右上角，不与其他内容重叠 */}
                  <div
                    className={cn(
                      'absolute top-3 right-3 w-5 h-5 rounded-full flex items-center justify-center transition-all',
                      active
                        ? opt.color === 'sky'
                          ? 'bg-sky-500 opacity-100'
                          : 'bg-violet-500 opacity-100'
                        : 'opacity-0',
                    )}
                  >
                    <Check size={11} className="text-white" />
                  </div>
                </button>
              )
            })}
          </div>

          {/* 表单字段 */}
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                仿真器名称 <span className="text-red-400 normal-case tracking-normal">*</span>
              </label>
              {mode === 'scenario' && registeredSimulators.length > 0 ? (
                <select
                  className="select w-full"
                  value={simulator}
                  onChange={e => {
                    setSimulator(e.target.value)
                    setScenario('')
                  }}
                >
                  <option value="">— 选择已注册的仿真器 —</option>
                  {registeredSimulators.map(s => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="input w-full"
                  placeholder="如 modflow、simpeg、fenics"
                  value={simulator}
                  onChange={e => setSimulator(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && mode === 'simulator' && canStart && handleStart()}
                />
              )}
              {mode === 'scenario' && registeredSimulators.length === 0 && (
                <p className="text-xs text-amber-400/80 mt-1.5 flex items-center gap-1">
                  <span>⚠</span> 暂无已注册的仿真器，请先完成第一步注册
                </p>
              )}
            </div>

            {mode === 'scenario' && (
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  场景名称 <span className="text-red-400 normal-case tracking-normal">*</span>
                </label>
                <input
                  className="input w-full"
                  placeholder="如 unified_aquifer、coastal_seawater"
                  value={scenario}
                  onChange={e => setScenario(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && canStart && handleStart()}
                  autoFocus
                />
                <p className="text-xs text-slate-500 mt-1.5">必须与 HDF5 文件对应的场景名一致</p>
              </div>
            )}

            {mode === 'simulator' && (
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  HDF5 文件路径 <span className="text-slate-600 normal-case tracking-normal font-normal">（可选）</span>
                </label>
                <input
                  className="input w-full font-mono text-sm"
                  placeholder="data/modflow/modflow_unified_aquifer.h5"
                  value={hdf5Path}
                  onChange={e => setHdf5Path(e.target.value)}
                />
                <p className="text-xs text-slate-500 mt-1.5">提供后可自动预填参数含义，大幅加速注册</p>
              </div>
            )}
          </div>

          {/* 写入说明 */}
          <div
            className={cn(
              'rounded-xl px-4 py-3 text-xs leading-relaxed border',
              mode === 'simulator'
                ? 'bg-sky-500/5 border-sky-500/15 text-sky-400/70'
                : 'bg-violet-500/5 border-violet-500/15 text-violet-400/70',
            )}
          >
            {mode === 'simulator' ? (
              <span>
                将写入 <span className="font-mono text-sky-300/90">{simulator || 'simulator'}</span>
                ，包含完整元数据，所有场景共享
              </span>
            ) : (
              <span>
                将写入{' '}
                <span className="font-mono text-violet-300/90">
                  {simulator || 'simulator'}/{scenario || 'scenario'}
                </span>{' '}
                的场景描述
              </span>
            )}
          </div>

          {/* HDF5 格式要求（仅 simulator 注册时显示）*/}
          {mode === 'simulator' && <HDF5FormatCard />}

          {error && (
            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl px-3.5 py-2.5 text-sm text-red-300">
              <AlertTriangle size={14} className="flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            className={cn(
              'w-full flex items-center justify-center gap-2 py-3.5 rounded-xl font-semibold text-sm transition-all active:scale-[0.98]',
              canStart && !loading
                ? mode === 'simulator'
                  ? 'bg-gradient-to-r from-sky-600 to-sky-500 hover:from-sky-500 hover:to-sky-400 text-white shadow-lg shadow-sky-900/30'
                  : 'bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 text-white shadow-lg shadow-violet-900/30'
                : 'bg-slate-700/60 text-slate-500 cursor-not-allowed',
            )}
            onClick={handleStart}
            disabled={loading || !canStart}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> 启动中…
              </>
            ) : mode === 'simulator' ? (
              <>
                <Play size={16} /> 开始注册仿真器
              </>
            ) : (
              <>
                <Play size={16} /> 开始注册场景描述
              </>
            )}
          </button>

          {mode === 'simulator' && (
            <div className="flex items-center justify-center gap-1.5 text-xs text-slate-600">
              <Github size={11} />
              <span>支持从 GitHub 仓库自动预填充</span>
            </div>
          )}
        </div>
      </div>
    )
  }

  // ── 已开始：对话界面 ──
  const isGithubStep = step === 0
  const isScenarioMode = step === 7

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
      {/* 步骤指示器 */}
      <StepIndicator current={step} prefilled={prefilledSteps} />

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto pt-5 pb-2 min-h-0">
        <div className="max-w-2xl mx-auto px-5">
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 底部固定区域（居中容器统一在这里）*/}
      <div className="flex-shrink-0">
        <div className="max-w-2xl mx-auto px-5">
          {/* GitHub 预填充横幅 */}
          {githubPrefillInfo && step > 0 && (
            <GithubPrefillBanner summary={githubPrefillInfo.summary} steps={githubPrefillInfo.steps} />
          )}

          {/* 提取结果编辑器 */}
          {agentStatus === 'confirming' && extracted && (
            <ExtractionPreview
              step={step}
              extracted={extracted}
              uncertain={extractionUncertain}
              onDataChange={d => {
                setExtracted(d)
                setEditJson(JSON.stringify(d, null, 2))
              }}
            />
          )}

          {/* 完成横幅 */}
          {agentStatus === 'done' && registryKey && (
            <div className="mb-3 rounded-2xl overflow-hidden border border-emerald-500/25 bg-gradient-to-r from-emerald-500/8 to-emerald-600/5">
              <div className="flex items-center gap-3 px-4 py-3.5">
                <div className="w-9 h-9 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
                  <CheckCircle size={18} className="text-emerald-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-emerald-300">注册成功！</div>
                  <div className="text-xs text-emerald-500/80 font-mono mt-0.5 truncate">{registryKey}</div>
                </div>
                <button
                  className="flex-shrink-0 px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 text-sm font-medium transition-colors"
                  onClick={handleCancel}
                >
                  继续注册
                </button>
              </div>
            </div>
          )}

          {/* 错误提示 */}
          {error && (
            <div className="mb-3 flex items-start gap-2.5 bg-red-500/8 border border-red-500/20 rounded-xl px-3.5 py-3 text-red-300">
              <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* 底部操作区 */}
          <div className="pb-4 pt-1">
            {agentStatus === 'confirming' ? (
              /* 确认区 */
              <div className="flex gap-2.5">
                <button
                  className={cn(
                    'flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-medium text-sm transition-all',
                    loading
                      ? 'bg-sky-600/50 text-sky-300 cursor-not-allowed'
                      : 'bg-gradient-to-r from-sky-600 to-sky-500 hover:from-sky-500 hover:to-sky-400 text-white shadow-md shadow-sky-900/30 active:scale-[0.98]',
                  )}
                  onClick={handleConfirm}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={15} className="animate-spin" /> 处理中…
                    </>
                  ) : (
                    <>
                      <Check size={15} /> 确认，下一步
                    </>
                  )}
                </button>
                <button
                  className="px-4 py-3 rounded-xl border border-slate-700/60 bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 text-sm font-medium transition-all active:scale-[0.98] flex items-center gap-2"
                  onClick={handleReject}
                  disabled={loading}
                >
                  <X size={14} /> 重新回答
                </button>
              </div>
            ) : agentStatus === 'interviewing' ? (
              /* 输入区 */
              <div className="space-y-2">
                <div
                  className={cn(
                    'flex items-end gap-2 rounded-2xl border transition-all',
                    'bg-slate-800/60 border-slate-700/60 focus-within:border-sky-500/50 focus-within:bg-slate-800/80',
                  )}
                >
                  <textarea
                    ref={textareaRef}
                    className="flex-1 bg-transparent text-sm text-slate-200 placeholder:text-slate-500 resize-none px-4 py-3 focus:outline-none leading-relaxed min-h-[44px] max-h-[120px]"
                    rows={isGithubStep || isScenarioMode ? 1 : 2}
                    placeholder={
                      isGithubStep
                        ? 'https://github.com/owner/repo  （或输入「跳过」）'
                        : isScenarioMode
                          ? '用一句话描述该场景的物理设置…'
                          : '输入你的回答…'
                    }
                    value={inputText}
                    disabled={loading}
                    onChange={e => setInputText(e.target.value)}
                    onKeyDown={e => {
                      if (isScenarioMode && e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSend()
                      } else if (e.key === 'Enter' && e.ctrlKey) {
                        e.preventDefault()
                        handleSend()
                      }
                    }}
                  />
                  <div className="flex items-center gap-1.5 pr-2 pb-2 flex-shrink-0">
                    {isGithubStep && (
                      <button
                        className="px-3 py-1.5 rounded-lg text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-700/60 transition-colors flex items-center gap-1"
                        onClick={() => handleSend('跳过')}
                        disabled={loading}
                      >
                        <SkipForward size={12} /> 跳过
                      </button>
                    )}
                    {!isGithubStep && (
                      <button
                        className="p-1.5 rounded-lg text-slate-600 hover:text-slate-400 hover:bg-slate-700/40 transition-colors"
                        onClick={handleCancel}
                        title="取消注册"
                      >
                        <X size={14} />
                      </button>
                    )}
                    <button
                      className={cn(
                        'w-9 h-9 rounded-xl flex items-center justify-center transition-all active:scale-95',
                        inputText.trim() && !loading
                          ? isScenarioMode
                            ? 'bg-violet-600 hover:bg-violet-500 text-white shadow-sm'
                            : 'bg-sky-600 hover:bg-sky-500 text-white shadow-sm shadow-sky-900/30'
                          : 'bg-slate-700/60 text-slate-500 cursor-not-allowed',
                      )}
                      onClick={() => handleSend()}
                      disabled={loading || !inputText.trim()}
                    >
                      {loading ? (
                        <RefreshCw size={15} className="animate-spin" />
                      ) : isGithubStep ? (
                        <Github size={15} />
                      ) : (
                        <Send size={15} />
                      )}
                    </button>
                  </div>
                </div>
                <div className="flex items-center justify-between px-1">
                  {isGithubStep ? (
                    <p className="text-xs text-slate-600">粘贴仓库链接可自动预填大部分信息</p>
                  ) : isScenarioMode ? (
                    <p className="text-xs text-slate-600">Enter 发送 · Shift+Enter 换行</p>
                  ) : (
                    <p className="text-xs text-slate-600">Ctrl+Enter 发送 · Enter 换行</p>
                  )}
                </div>
              </div>
            ) : null}
          </div>
          {/* end 底部操作区 */}
        </div>
        {/* end max-w-2xl */}
      </div>
      {/* end 底部固定区域 */}
    </div>
  )
}
