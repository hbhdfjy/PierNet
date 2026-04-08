import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { DataDirEntry } from '../lib/types'
import {
  FolderOpen, Plus, Trash2, Save, RefreshCw, Check,
  AlertCircle, ChevronDown, ChevronUp, Zap, X,
  Database, ArrowRight, Hash, ChevronRight,
} from 'lucide-react'
import { cn } from '../lib/utils'

// simulator → 颜色主题
const SIM_THEME: Record<string, { dot: string; badge: string; ring: string }> = {
  modflow:         { dot: 'bg-blue-400',    badge: 'bg-blue-500/15 text-blue-300 border-blue-500/30',    ring: 'ring-blue-500/40'    },
  simpeg:          { dot: 'bg-purple-400',  badge: 'bg-purple-500/15 text-purple-300 border-purple-500/30', ring: 'ring-purple-500/40' },
  power_flow:      { dot: 'bg-amber-400',   badge: 'bg-amber-500/15 text-amber-300 border-amber-500/30',  ring: 'ring-amber-500/40'   },
  power_transient: { dot: 'bg-red-400',     badge: 'bg-red-500/15 text-red-300 border-red-500/30',        ring: 'ring-red-500/40'     },
  gcam:            { dot: 'bg-emerald-400', badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30', ring: 'ring-emerald-500/40' },
}
const defaultTheme = { dot: 'bg-slate-400', badge: 'bg-slate-700/60 text-slate-300 border-slate-600/40', ring: 'ring-slate-500/40' }
const getTheme = (sim: string) => SIM_THEME[sim] ?? defaultTheme

const SIMULATOR_OPTIONS = ['modflow', 'simpeg', 'power_flow', 'power_transient', 'gcam']

const EMPTY_ENTRY = (): DataDirEntry => ({
  key: '', path: '', simulator: 'modflow',
  file_suffix: '', transient_simulator: '', transient_keywords: [],
})

// ── 场景名预览 ────────────────────────────────────────────────────
function ScenamePreview({ path, suffix }: { path: string; suffix: string }) {
  if (!path) return null
  const example = path.split('/').pop() ?? path
  const scenarioExample = suffix ? `${example}_example${suffix}` : `${example}_example`
  const result = suffix && scenarioExample.endsWith(suffix)
    ? scenarioExample.slice(0, -suffix.length)
    : scenarioExample
  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-600 mt-1.5">
      <span className="font-mono text-slate-500">{scenarioExample}.h5</span>
      <ArrowRight size={10} className="text-slate-700" />
      <span className="font-mono text-sky-500/70">{result}</span>
    </div>
  )
}

// ── 单条目编辑卡片 ────────────────────────────────────────────────
function EntryCard({
  entry, index, total,
  onChange, onDelete, onMoveUp, onMoveDown,
}: {
  entry: DataDirEntry; index: number; total: number
  onChange: (e: DataDirEntry) => void
  onDelete: () => void; onMoveUp: () => void; onMoveDown: () => void
}) {
  const [advanced, setAdvanced] = useState(
    !!(entry.transient_simulator || entry.transient_keywords.length)
  )
  const [customSim, setCustomSim] = useState(!SIMULATOR_OPTIONS.includes(entry.simulator))

  const set = (patch: Partial<DataDirEntry>) => onChange({ ...entry, ...patch })
  const keywordsStr = entry.transient_keywords.join(', ')
  const setKeywords = (s: string) =>
    set({ transient_keywords: s.split(',').map(k => k.trim()).filter(Boolean) })

  const theme = getTheme(entry.simulator)
  const isValid = entry.key.trim() && entry.path.trim() && entry.simulator.trim()

  return (
    <div className={cn(
      'rounded-2xl border overflow-hidden transition-all duration-200',
      isValid
        ? 'bg-slate-800/70 border-slate-700/50 hover:border-slate-600/60'
        : 'bg-slate-800/40 border-slate-700/30 border-dashed',
    )}>
      {/* ── 标题栏 ── */}
      <div className="flex items-center gap-3 px-4 py-3 bg-slate-900/40 border-b border-slate-700/30">
        {/* 序号 + 颜色点 */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs text-slate-600 font-mono w-4 text-center">{index + 1}</span>
          <div className={cn('w-2.5 h-2.5 rounded-full flex-shrink-0', theme.dot)} />
        </div>

        {/* 键名 */}
        <span className={cn(
          'font-semibold text-sm flex-1 truncate',
          entry.key ? 'text-slate-100' : 'text-slate-600 italic',
        )}>
          {entry.key || '未命名'}
        </span>

        {/* simulator badge */}
        {entry.simulator && (
          <span className={cn('badge border text-xs flex-shrink-0', theme.badge)}>
            {entry.simulator}
          </span>
        )}

        {/* 路径 */}
        {entry.path && (
          <span className="text-xs text-slate-600 font-mono hidden xl:block truncate max-w-[180px]">
            {entry.path}
          </span>
        )}

        {/* 操作 */}
        <div className="flex items-center gap-0.5 flex-shrink-0">
          <button className="btn-ghost py-0.5 px-1.5 disabled:opacity-20" onClick={onMoveUp} disabled={index === 0}>
            <ChevronUp size={13} />
          </button>
          <button className="btn-ghost py-0.5 px-1.5 disabled:opacity-20" onClick={onMoveDown} disabled={index === total - 1}>
            <ChevronDown size={13} />
          </button>
          <button className="btn-ghost py-0.5 px-1.5 text-red-400/70 hover:text-red-300" onClick={onDelete}>
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {/* ── 字段区 ── */}
      <div className="p-4 space-y-4">

        {/* 行1：键名 + 路径 */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label block mb-1.5 text-xs">
              配置键名 <span className="text-red-400 normal-case tracking-normal font-normal">必填</span>
            </label>
            <input
              className={cn('input w-full text-sm py-1.5', !entry.key && 'border-red-500/30')}
              placeholder="如 modflow、power_system"
              value={entry.key}
              onChange={e => set({ key: e.target.value })}
            />
            <p className="text-xs text-slate-600 mt-1">唯一标识，前端分组标题</p>
          </div>
          <div>
            <label className="label block mb-1.5 text-xs">
              数据目录 <span className="text-red-400 normal-case tracking-normal font-normal">必填</span>
            </label>
            <div className="relative">
              <FolderOpen size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
              <input
                className={cn('input w-full text-sm py-1.5 pl-8 font-mono', !entry.path && 'border-red-500/30')}
                placeholder="data/modflow"
                value={entry.path}
                onChange={e => set({ path: e.target.value.trim() })}
              />
            </div>
            <p className="text-xs text-slate-600 mt-1">相对于项目根目录</p>
          </div>
        </div>

        {/* 行2：Simulator 类型（tag 按钮组）+ 后缀 */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label block mb-1.5 text-xs">Simulator 类型</label>
            <div className="flex flex-wrap gap-1.5">
              {SIMULATOR_OPTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => { set({ simulator: s }); setCustomSim(false) }}
                  className={cn(
                    'px-2.5 py-1 rounded-lg border text-xs font-mono transition-all',
                    entry.simulator === s && !customSim
                      ? cn('border', getTheme(s).badge, `ring-1 ${getTheme(s).ring}`)
                      : 'bg-slate-800/60 border-slate-700/40 text-slate-500 hover:text-slate-300 hover:border-slate-600',
                  )}
                >
                  {s}
                </button>
              ))}
              <button
                onClick={() => setCustomSim(true)}
                className={cn(
                  'px-2.5 py-1 rounded-lg border text-xs transition-all',
                  customSim
                    ? 'bg-slate-600/30 border-slate-500/40 text-slate-200'
                    : 'bg-slate-800/60 border-slate-700/40 text-slate-600 hover:text-slate-400',
                )}
              >
                自定义
              </button>
            </div>
            {customSim && (
              <input
                className="input w-full text-sm py-1.5 mt-2 font-mono"
                placeholder="输入自定义类型名"
                value={SIMULATOR_OPTIONS.includes(entry.simulator) ? '' : entry.simulator}
                onChange={e => set({ simulator: e.target.value.trim() })}
                autoFocus
              />
            )}
          </div>
          <div>
            <label className="label block mb-1.5 text-xs">文件名后缀去除</label>
            <div className="relative">
              <Hash size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
              <input
                className="input w-full text-sm py-1.5 pl-7 font-mono"
                placeholder="_groundwater_timeseries"
                value={entry.file_suffix}
                onChange={e => set({ file_suffix: e.target.value.trim() })}
              />
            </div>
            <ScenamePreview path={entry.path} suffix={entry.file_suffix} />
          </div>
        </div>

        {/* 暂态识别（可折叠）*/}
        <div className="rounded-xl border border-slate-700/30 overflow-hidden">
          <button
            onClick={() => setAdvanced(v => !v)}
            className="w-full flex items-center justify-between px-3.5 py-2.5 bg-slate-900/30 hover:bg-slate-700/20 transition-colors text-left"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-medium">暂态 Simulator 识别</span>
              <span className="text-xs text-slate-600">（可选）</span>
              {(entry.transient_simulator || entry.transient_keywords.length > 0) && (
                <span className="badge bg-amber-500/12 text-amber-400 border border-amber-500/25 text-xs py-0.5">
                  已配置
                </span>
              )}
            </div>
            {advanced
              ? <ChevronUp size={13} className="text-slate-600" />
              : <ChevronDown size={13} className="text-slate-600" />
            }
          </button>

          {advanced && (
            <div className="px-4 py-3 grid grid-cols-2 gap-3 border-t border-slate-700/30 bg-slate-900/20">
              <div>
                <label className="label block mb-1.5 text-xs">暂态 Simulator 类型</label>
                <input
                  className="input w-full text-sm py-1.5 font-mono"
                  placeholder="如 power_transient"
                  value={entry.transient_simulator}
                  onChange={e => set({ transient_simulator: e.target.value.trim() })}
                />
              </div>
              <div>
                <label className="label block mb-1.5 text-xs">触发关键词 <span className="text-slate-600 normal-case tracking-normal font-normal">（逗号分隔）</span></label>
                <input
                  className="input w-full text-sm py-1.5 font-mono"
                  placeholder="fault, trip, load_step"
                  value={keywordsStr}
                  onChange={e => setKeywords(e.target.value)}
                />
                {entry.transient_keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {entry.transient_keywords.map(kw => (
                      <span key={kw} className="badge bg-amber-500/10 text-amber-400/80 border border-amber-500/20 text-xs py-0.5">
                        {kw}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function DataDirsConfig() {
  const navigate = useNavigate()
  const { data: savedEntries, mutate: refresh, isLoading } =
    useSWR<DataDirEntry[]>('data-dirs', () => api.getDataDirs())

  const [entries, setEntries] = useState<DataDirEntry[] | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [showRegisterHint, setShowRegisterHint] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const working = entries ?? savedEntries ?? []
  const isDirty = entries !== null

  const update = (i: number, e: DataDirEntry) =>
    setEntries(prev => (prev ?? savedEntries ?? []).map((x, j) => j === i ? e : x))
  const remove = (i: number) =>
    setEntries(prev => (prev ?? savedEntries ?? []).filter((_, j) => j !== i))
  const add = () =>
    setEntries(prev => [...(prev ?? savedEntries ?? []), EMPTY_ENTRY()])
  const move = (i: number, dir: -1 | 1) => {
    const arr = [...working]
    const j = i + dir
    if (j < 0 || j >= arr.length) return
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
    setEntries(arr)
  }

  const handleSave = async () => {
    for (const e of working) {
      if (!e.key.trim()) { setError('存在未填写「配置键名」的条目'); return }
      if (!e.path.trim()) { setError(`条目「${e.key}」未填写路径`); return }
      if (!e.simulator.trim()) { setError(`条目「${e.key}」未填写 Simulator 类型`); return }
    }
    const keys = working.map(e => e.key)
    const dup = keys.find((k, i) => keys.indexOf(k) !== i)
    if (dup) { setError(`配置键名「${dup}」重复`); return }

    setError(null)
    setSaving(true)
    try {
      await api.saveDataDirs(working)
      setSaved(true)
      setShowRegisterHint(true)
      setEntries(null)
      await refresh()
      setTimeout(() => setSaved(false), 2500)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* ── 左栏 ── */}
      <div className="w-64 flex-shrink-0 border-r border-slate-700/40 flex flex-col bg-slate-900/50">

        {/* 页头 */}
        <div className="px-5 pt-5 pb-4 border-b border-slate-700/30">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-9 h-9 rounded-xl bg-sky-500/20 border border-sky-500/30 flex items-center justify-center flex-shrink-0">
              <Database size={16} className="text-sky-400" />
            </div>
            <div>
              <h1 className="text-base font-bold text-white">数据目录</h1>
              <p className="text-slate-500 text-xs">HDF5 数据源配置</p>
            </div>
          </div>

          {/* 统计 */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-slate-800/60 rounded-xl px-3 py-2 border border-slate-700/30">
              <div className="text-lg font-bold text-white tabular-nums">{working.length}</div>
              <div className="text-xs text-slate-500">数据目录</div>
            </div>
            <div className="bg-slate-800/60 rounded-xl px-3 py-2 border border-slate-700/30">
              <div className="text-lg font-bold text-white tabular-nums">
                {new Set(working.map(e => e.simulator).filter(Boolean)).size}
              </div>
              <div className="text-xs text-slate-500">Simulator 类型</div>
            </div>
          </div>
        </div>

        {/* 操作 */}
        <div className="px-4 py-4 space-y-2 border-b border-slate-700/30">
          <button
            className={cn(
              'btn w-full py-2.5 text-sm justify-center',
              saved ? 'bg-emerald-600 text-white' : 'btn-primary',
            )}
            style={!saved && !saving ? { background: 'linear-gradient(135deg, #0284c7, #0369a1)' } : {}}
            onClick={handleSave}
            disabled={saving || !isDirty}
          >
            {saving ? <><RefreshCw size={14} className="animate-spin" /> 保存中…</>
              : saved ? <><Check size={14} /> 已保存</>
              : <><Save size={14} /> 保存配置</>}
          </button>
          <div className="grid grid-cols-2 gap-2">
            <button
              className="btn-ghost py-2 text-xs justify-center"
              onClick={() => { setEntries(null); setError(null) }}
              disabled={!isDirty}
            >
              <X size={12} /> 放弃
            </button>
            <button className="btn-ghost py-2 text-xs justify-center" onClick={add}>
              <Plus size={12} /> 添加
            </button>
          </div>
        </div>

        {/* 状态提示 */}
        <div className="px-4 py-3 space-y-2">
          {isDirty && !error && (
            <div className="flex items-center gap-2 px-3 py-2 bg-amber-500/8 border border-amber-500/20 rounded-xl text-amber-300 text-xs">
              <Zap size={11} className="flex-shrink-0" /> 有未保存的修改
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={12} className="flex-shrink-0 mt-0.5" />
              <span className="text-xs leading-relaxed">{error}</span>
            </div>
          )}
          {showRegisterHint && !isDirty && (
            <div className="bg-sky-500/8 border border-sky-500/20 rounded-xl px-3 py-2.5 space-y-2">
              <p className="text-xs text-sky-300 font-medium">数据目录已保存</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                新增的数据目录需要先注册 simulator 元数据，才能进行语言模板生成。
              </p>
              <button
                className="w-full flex items-center justify-between px-3 py-2 bg-sky-500/15 hover:bg-sky-500/25 border border-sky-500/30 rounded-lg text-xs text-sky-300 transition-colors"
                onClick={() => navigate('/register')}
              >
                <span>前往注册数据集</span>
                <ChevronRight size={13} />
              </button>
              <button className="text-xs text-slate-600 hover:text-slate-400 w-full text-center transition-colors"
                onClick={() => setShowRegisterHint(false)}>
                稍后再说
              </button>
            </div>
          )}
        </div>

        {/* 目录索引 */}
        {working.length > 0 && (
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <div className="label mb-2 text-xs">目录列表</div>
            <div className="space-y-1">
              {working.map((e, i) => {
                const theme = getTheme(e.simulator)
                return (
                  <div key={i} className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg hover:bg-slate-700/30 transition-colors">
                    <div className={cn('w-2 h-2 rounded-full flex-shrink-0', theme.dot)} />
                    <span className={cn('text-xs truncate flex-1', e.key ? 'text-slate-300' : 'text-slate-600 italic')}>
                      {e.key || '未命名'}
                    </span>
                    {e.path && (
                      <span className="text-xs text-slate-700 font-mono truncate max-w-[80px]">{e.path}</span>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* 底部说明 */}
        <div className="mt-auto px-4 py-3 border-t border-slate-700/30">
          <p className="text-xs text-slate-600 text-center leading-relaxed">
            保存到<br />
            <span className="font-mono text-slate-500">configs/text2comp/default.yaml</span>
          </p>
        </div>
      </div>

      {/* ── 右栏：条目列表 ── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-3">
        {isLoading && (
          <div className="flex items-center justify-center gap-2 py-20 text-slate-500">
            <RefreshCw size={15} className="animate-spin" />
            <span className="text-sm">加载配置…</span>
          </div>
        )}

        {!isLoading && working.length === 0 && (
          <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
            <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center">
              <Database size={28} className="text-sky-400/50" />
            </div>
            <div>
              <p className="text-slate-300 text-sm font-medium">暂无数据目录配置</p>
              <p className="text-slate-600 text-xs mt-1.5">每个条目对应一个存放 HDF5 文件的目录</p>
            </div>
            <button
              className="btn-primary px-5 py-2.5 text-sm"
              style={{ background: 'linear-gradient(135deg, #0284c7, #0369a1)' }}
              onClick={add}
            >
              <Plus size={15} /> 添加第一个目录
            </button>
          </div>
        )}

        {working.map((entry, i) => (
          <EntryCard
            key={i}
            entry={entry}
            index={i}
            total={working.length}
            onChange={e => update(i, e)}
            onDelete={() => remove(i)}
            onMoveUp={() => move(i, -1)}
            onMoveDown={() => move(i, 1)}
          />
        ))}

        {working.length > 0 && (
          <button
            className="w-full py-3 text-sm text-slate-500 hover:text-slate-300 border border-dashed border-slate-700/50 hover:border-slate-600/60 rounded-2xl transition-all flex items-center justify-center gap-2 hover:bg-slate-800/30"
            onClick={add}
          >
            <Plus size={14} /> 添加数据目录
          </button>
        )}
      </div>
    </div>
  )
}
