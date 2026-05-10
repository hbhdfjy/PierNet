import { useState } from 'react'
import {
  Check,
  ChevronRight,
  Clock,
  FileText,
  Layers,
  Loader2,
  Pencil,
  Save,
  Tag,
  Trash2,
  X,
  XCircle,
} from 'lucide-react'
import { cn } from '../../../lib/utils'
import { DomainTab, ObsConfigEditor, OutputInfoTab, ParamInfoTab } from './RegistryEditors'
import type { OutputInfoItem, RegistryEntry } from './registryTypes'

function FieldBadge({ present, label }: { present: boolean; label: string }) {
  return (
    <span
      className={cn(
        'badge border text-xs',
        present
          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25'
          : 'bg-slate-700/30 text-slate-600 border-slate-700/30',
      )}
    >
      {present ? '✓' : '○'} {label}
    </span>
  )
}

export function RegistryEntryCard({
  entryKey,
  entry,
  onSave,
  onDelete,
}: {
  entryKey: string
  entry: RegistryEntry
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
  const hasObs = !!entry.observation_config
  const hasParams = !!entry.param_info
  const [simulator, scenario] = entryKey.includes('/') ? entryKey.split('/', 2) : [entryKey, '']
  const obs = entry.observation_config
  const params = entry.param_info ?? {}
  const outputInfo = (entry.output_info ?? []) as OutputInfoItem[]

  return (
    <div
      className={cn(
        'border rounded-2xl overflow-hidden transition-all duration-200',
        open ? 'border-sky-500/30 bg-slate-800/60' : 'border-slate-700/40 bg-slate-800/30 hover:border-slate-600/60',
      )}
    >
      <div className="flex items-center gap-3 px-4 py-3.5">
        <button onClick={() => setOpen(o => !o)} className="flex items-center gap-3 flex-1 text-left min-w-0">
          <span className={cn('transition-transform duration-200 text-slate-500 flex-shrink-0', open && 'rotate-90')}>
            <ChevronRight size={15} />
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-base text-slate-200 font-medium">{scenario || simulator}</span>
              {scenario && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">
                  {simulator}
                </span>
              )}
            </div>
          </div>
        </button>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <FieldBadge present={hasDomain} label="domain" />
          <FieldBadge present={hasOutput} label="output" />
          <FieldBadge present={hasObs} label="obs" />
          {!confirmDelete ? (
            <button
              className="btn-ghost py-1 px-2 text-xs text-slate-600 hover:text-red-400 ml-1"
              onClick={e => {
                e.stopPropagation()
                setConfirmDelete(true)
              }}
            >
              <Trash2 size={13} />
            </button>
          ) : (
            <div className="flex items-center gap-1 ml-1">
              <span className="text-xs text-red-400">确认？</span>
              <button
                className="btn-ghost py-0.5 px-2 text-xs text-red-400"
                onClick={async () => {
                  await onDelete(entryKey)
                  setConfirmDelete(false)
                }}
              >
                <Check size={12} />
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs text-slate-500" onClick={() => setConfirmDelete(false)}>
                <X size={12} />
              </button>
            </div>
          )}
        </div>
      </div>

      {open && (
        <div className="border-t border-slate-700/30">
          <div className="flex border-b border-slate-700/30 bg-slate-900/30 overflow-x-auto">
            {(
              [
                { id: 'domain', icon: <FileText size={13} />, label: '领域描述', disabled: false },
                { id: 'output', icon: <Layers size={13} />, label: '输出通道', disabled: !hasOutput },
                { id: 'obs', icon: <Clock size={13} />, label: '观测配置', disabled: !hasObs },
                {
                  id: 'params',
                  icon: <Tag size={13} />,
                  label: `参数 (${Object.keys(params).length})`,
                  disabled: !hasParams,
                },
                { id: 'raw', icon: <FileText size={13} />, label: 'Raw JSON', disabled: false },
              ] as const
            ).map(t => (
              <button
                key={t.id}
                disabled={t.disabled}
                onClick={() => {
                  setTab(t.id)
                }}
                className={cn(
                  'flex items-center gap-1.5 px-4 py-2.5 text-sm transition-colors border-b-2 whitespace-nowrap flex-shrink-0',
                  tab === t.id && !t.disabled
                    ? 'border-sky-500 text-sky-300 bg-sky-500/5'
                    : t.disabled
                      ? 'border-transparent text-slate-700 cursor-not-allowed'
                      : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-700/20',
                )}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </div>
          <div className="p-4">
            {tab === 'domain' && (
              <DomainTab
                entry={entry}
                saving={saving}
                saveErr={saveErr}
                onSave={async patch => {
                  setSaving(true)
                  setSaveErr(null)
                  try {
                    await onSave(entryKey, { ...entry, ...patch })
                  } catch (e) {
                    setSaveErr(String(e))
                  } finally {
                    setSaving(false)
                  }
                }}
              />
            )}
            {tab === 'output' && (
              <OutputInfoTab
                outputInfo={outputInfo}
                saving={saving}
                saveErr={saveErr}
                onSave={async updated => {
                  setSaving(true)
                  setSaveErr(null)
                  try {
                    await onSave(entryKey, { ...entry, output_info: updated })
                  } catch (e) {
                    setSaveErr(String(e))
                  } finally {
                    setSaving(false)
                  }
                }}
              />
            )}
            {tab === 'obs' && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="label flex items-center gap-1.5">
                    <Clock size={12} />
                    observation_config
                  </div>
                  <div className="flex-1" />
                  {saving && <Loader2 size={12} className="animate-spin text-slate-500" />}
                  {saveErr && (
                    <div className="text-xs text-red-400 flex items-center gap-1">
                      <XCircle size={11} />
                      {saveErr}
                    </div>
                  )}
                </div>
                {hasObs && obs ? (
                  <ObsConfigEditor
                    obs={obs}
                    outputInfo={outputInfo}
                    onChange={async updated => {
                      setSaving(true)
                      setSaveErr(null)
                      try {
                        await onSave(entryKey, { ...entry, observation_config: updated })
                      } catch (e) {
                        setSaveErr(String(e))
                      } finally {
                        setSaving(false)
                      }
                    }}
                  />
                ) : (
                  <p className="text-sm text-slate-600 italic">（未填写）</p>
                )}
              </div>
            )}
            {tab === 'params' && (
              <ParamInfoTab
                params={params}
                saving={saving}
                saveErr={saveErr}
                onSave={async updated => {
                  setSaving(true)
                  setSaveErr(null)
                  try {
                    await onSave(entryKey, { ...entry, param_info: updated })
                  } catch (e) {
                    setSaveErr(String(e))
                  } finally {
                    setSaving(false)
                  }
                }}
              />
            )}
            {tab === 'raw' && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <div className="label flex items-center gap-1.5">
                    <FileText size={12} />
                    完整 JSON
                  </div>
                  <div className="flex-1" />
                  {!rawEdit ? (
                    <button
                      className="btn-ghost py-0.5 px-2 text-xs text-slate-500 hover:text-slate-200"
                      onClick={() => {
                        setRawVal(JSON.stringify(entry, null, 2))
                        setRawEdit(true)
                        setSaveErr(null)
                      }}
                    >
                      <Pencil size={12} /> 编辑
                    </button>
                  ) : (
                    <div className="flex gap-1">
                      <button
                        className="btn-ghost py-0.5 px-2 text-xs text-emerald-400"
                        disabled={saving}
                        onClick={async () => {
                          setSaving(true)
                          setSaveErr(null)
                          try {
                            await onSave(entryKey, JSON.parse(rawVal))
                            setRawEdit(false)
                          } catch (e) {
                            setSaveErr(e instanceof SyntaxError ? 'JSON 格式错误' : String(e))
                          } finally {
                            setSaving(false)
                          }
                        }}
                      >
                        {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} 保存
                      </button>
                      <button
                        className="btn-ghost py-0.5 px-2 text-xs text-slate-500"
                        onClick={() => {
                          setRawEdit(false)
                          setSaveErr(null)
                        }}
                      >
                        <X size={12} /> 取消
                      </button>
                    </div>
                  )}
                </div>
                {rawEdit ? (
                  <div className="space-y-1.5">
                    <textarea
                      className="input w-full font-mono text-sm resize-y leading-relaxed"
                      rows={20}
                      value={rawVal}
                      onChange={e => setRawVal(e.target.value)}
                      autoFocus
                    />
                    {saveErr && (
                      <div className="text-xs text-red-400 flex items-center gap-1">
                        <XCircle size={11} />
                        {saveErr}
                      </div>
                    )}
                  </div>
                ) : (
                  <pre className="text-xs font-mono text-slate-400 bg-slate-950/50 rounded-xl p-4 border border-slate-700/30 overflow-x-auto max-h-96 overflow-y-auto leading-relaxed">
                    {JSON.stringify(entry, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
