import { useState } from 'react'
import { Check, Loader2, Pencil, Trash2, X } from 'lucide-react'

export function ScenarioRowGrid({
  simulator,
  scenario,
  description,
  onSave,
  onDelete,
}: {
  simulator: string
  scenario: string
  description: string
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
    try {
      await onSave(key, val)
      setEditing(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex items-start gap-3 px-4 py-3 hover:bg-slate-700/15 transition-colors group">
      <div className="flex flex-col items-center flex-shrink-0 mt-2" style={{ width: 20 }}>
        <div className="w-px h-2 bg-slate-700/60" />
        <div className="w-3 h-px bg-slate-700/60" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-start gap-3 min-w-0">
          <div className="min-w-0 flex-1">
            <div
              className="font-mono text-sm text-sky-300/90 leading-5 min-w-0"
              style={{
                display: 'block',
                maxWidth: '100%',
                minWidth: 0,
                overflowWrap: 'anywhere',
                wordBreak: 'break-word',
              }}
            >
              {scenario}
            </div>
          </div>

          {!editing && (
            <div className="flex items-center gap-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity flex-shrink-0">
              <button
                className="btn-ghost py-0.5 px-1.5 text-xs text-slate-500 hover:text-slate-200"
                onClick={() => {
                  setEditing(true)
                  setVal(description)
                }}
              >
                <Pencil size={11} />
              </button>
              {!confirmDelete ? (
                <button
                  className="btn-ghost py-0.5 px-1.5 text-xs text-slate-600 hover:text-red-400"
                  onClick={() => setConfirmDelete(true)}
                >
                  <Trash2 size={11} />
                </button>
              ) : (
                <div className="flex items-center gap-1 rounded-full border border-red-500/25 bg-red-500/10 px-2 py-1">
                  <span className="text-[11px] text-red-400">删除？</span>
                  <button className="btn-ghost py-0.5 px-1 text-xs text-red-400" onClick={() => onDelete(key)}>
                    <Check size={11} />
                  </button>
                  <button
                    className="btn-ghost py-0.5 px-1 text-xs text-slate-500"
                    onClick={() => setConfirmDelete(false)}
                  >
                    <X size={11} />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {editing ? (
          <div className="mt-3 space-y-2">
            <textarea
              className="input w-full text-sm py-2 leading-relaxed resize-y min-h-[88px]"
              rows={4}
              value={val}
              onChange={e => setVal(e.target.value)}
              onKeyDown={e => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                  e.preventDefault()
                  commit()
                }
              }}
              autoFocus
            />
            <div className="flex items-center justify-end gap-2">
              <button className="btn-ghost py-1 px-2 text-xs text-emerald-400" onClick={commit} disabled={saving}>
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                保存
              </button>
              <button
                className="btn-ghost py-1 px-2 text-xs text-slate-500"
                onClick={() => {
                  setEditing(false)
                  setVal(description)
                }}
              >
                <X size={12} />
                取消
              </button>
            </div>
          </div>
        ) : (
          <div className="mt-2 w-full min-w-0 rounded-xl border border-slate-700/35 bg-slate-950/35 px-3 py-2.5">
            {description ? (
              <p
                className="m-0 block w-full max-w-full text-sm text-slate-300/90 leading-6 whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
                style={{
                  whiteSpace: 'normal',
                  overflowWrap: 'anywhere',
                  wordBreak: 'break-word',
                }}
              >
                {description}
              </p>
            ) : (
              <span className="text-sm text-slate-600 italic">（无描述）</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
