import { useState } from 'react'
import type { SampleRecord } from '../../../lib/types'
import {
  LANGUAGE_LABELS, STYLE_LABELS, SIMULATOR_LABELS,
  getSimulatorBadgeClass, LANGUAGE_BADGE, STYLE_BADGE,
} from '../../../lib/utils'
import TimeseriesChart from './TimeseriesChart'
import { ChevronDown, Hash, AlignLeft, Target, Info, TrendingUp, AlertCircle } from 'lucide-react'
import { cn } from '../../../lib/utils'

interface Props {
  sample: SampleRecord
  index: number
}

function Section({ icon, title, children, defaultOpen = true }: {
  icon: React.ReactNode; title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-slate-700/30 last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="accordion-card-header w-full flex items-center gap-2 px-4 py-2.5 text-left transition-colors group"
      >
        <span className="text-slate-500 group-hover:text-slate-400 transition-colors">{icon}</span>
        <span className="text-base font-medium text-slate-300 flex-1">{title}</span>
        <span className={cn('text-slate-600 transition-transform duration-200', open && 'rotate-180')}>
          <ChevronDown size={13} />
        </span>
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

// Target 字段：支持展开/收起长文本
const PREVIEW_LEN = 280
function TargetSection({ target }: { target: string }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = target.length > PREVIEW_LEN
  return (
    <div className="border-b border-slate-700/30 last:border-0">
      <div className="accordion-card-header flex items-center gap-2 px-4 py-2.5">
        <span className="text-slate-500"><Target size={13} /></span>
        <span className="text-base font-medium text-slate-300 flex-1">Target</span>
        {isLong && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1"
          >
            {expanded ? '收起' : `展开（${target.length.toLocaleString()} 字符）`}
            <ChevronDown size={11} className={cn('transition-transform', expanded && 'rotate-180')} />
          </button>
        )}
      </div>
      <div className="px-4 pb-4">
        <div className={cn(
          'bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-400 leading-relaxed font-mono whitespace-pre-wrap break-all border border-slate-700/30',
          !expanded && isLong ? 'max-h-24 overflow-hidden' : 'max-h-96 overflow-y-auto',
        )}>
          {expanded || !isLong ? target : target.slice(0, PREVIEW_LEN) + '…'}
        </div>
        {!expanded && isLong && (
          <button
            onClick={() => setExpanded(true)}
            className="mt-1.5 text-xs text-sky-500/70 hover:text-sky-400 transition-colors"
          >
            显示全部 →
          </button>
        )}
      </div>
    </div>
  )
}

export default function SampleCard({ sample, index }: Props) {
  const meta = sample.metadata
  const obs = meta.observation

  const paramPairs = meta.param_names.map((name, i) => ({
    name,
    original: sample.number[i] ?? 0,
    transformed: sample.params_transformed[i] ?? 0,
    changed: Math.abs((sample.number[i] ?? 0) - (sample.params_transformed[i] ?? 0)) > 1e-9,
  }))
  const changedCount = paramPairs.filter(p => p.changed).length

  return (
    <div className="card overflow-hidden">
      {/* ── 标题行 ── */}
      <div className="card-header accordion-card-header justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-mono text-slate-600 flex-shrink-0">#{index + 1}</span>
          <span className={cn('badge border', getSimulatorBadgeClass(meta.simulator))}>
            {meta.simulator}
          </span>
          <span className="text-sm text-slate-400 truncate">{meta.scenario}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className={cn('badge border', LANGUAGE_BADGE[meta.language] ?? 'bg-slate-700 text-slate-300 border-slate-600')}>
            {LANGUAGE_LABELS[meta.language] ?? meta.language}
          </span>
          <span className={cn('badge border', STYLE_BADGE[meta.style] ?? 'bg-slate-700 text-slate-300 border-slate-600')}>
            {STYLE_LABELS[meta.style] ?? meta.style}
          </span>
          <span className="badge bg-slate-700/60 text-slate-400 border border-slate-600/40">
            {obs.time_mode} · {obs.n_time_points}pt
          </span>
        </div>
      </div>

      {/* ── Input ── */}
      <Section icon={<AlignLeft size={13} />} title="Input">
        <div className="bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-300 leading-relaxed
                        max-h-44 overflow-y-auto whitespace-pre-wrap border border-slate-700/30
                        ring-1 ring-inset ring-slate-700/20">
          {sample.input}
        </div>
      </Section>

      {/* ── Target ── */}
      <TargetSection target={sample.target} />

      {/* ── 时序图 ── */}
      <Section icon={<TrendingUp size={13} />} title="时序可视化">
        <TimeseriesChart sample={sample} />
      </Section>

      {/* ── 参数对比 ── */}
      <Section
        icon={<Hash size={13} />}
        title={`参数（${paramPairs.length} 维${changedCount > 0 ? `，${changedCount} 个已变换` : ''}）`}
        defaultOpen={changedCount > 0}
      >
        <div className="rounded-lg border border-slate-700/30 overflow-hidden">
          <div className="list-table-scroll">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-800/90">
                <tr className="border-b border-slate-700/40">
                  <th className="px-3 py-2 text-left label">参数名</th>
                  <th className="px-3 py-2 text-right label">原始值</th>
                  <th className="px-3 py-2 text-right label">变换后</th>
                </tr>
              </thead>
              <tbody>
                {paramPairs.map(({ name, original, transformed, changed }) => (
                  <tr key={name} className={cn(
                    'border-b border-slate-800/40 transition-colors',
                    changed ? 'bg-amber-500/5 hover:bg-amber-500/10' : 'hover:bg-slate-700/20',
                  )}>
                    <td className="px-3 py-1.5 font-mono text-slate-400">{name}</td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-500">
                      {original.toPrecision(5)}
                    </td>
                    <td className={cn(
                      'px-3 py-1.5 text-right font-mono tabular-nums',
                      changed ? 'text-amber-300 font-medium' : 'text-slate-500',
                    )}>
                      {transformed.toPrecision(5)}
                      {changed && <span className="ml-1 text-amber-500/60 text-xs">✎</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* ── Metadata ── */}
      <Section icon={<Info size={13} />} title="Metadata" defaultOpen={false}>
        <div className="space-y-3 text-sm">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {[
              ['Simulator', SIMULATOR_LABELS[meta.simulator] ?? meta.simulator],
              ['场景', meta.scenario],
              ['样本索引', meta.sample_idx],
              ['原始形状', `${meta.timeseries_shape[0]} × ${meta.timeseries_shape[1]}`],
              ['观测形状', `${meta.timeseries_shape_obs[0]} × ${meta.timeseries_shape_obs[1]}`],
              ['时间模式', obs.time_mode],
              ['时间点数', obs.n_time_points],
              ['通道索引', obs.channel_indices ? `[${obs.channel_indices.join(', ')}]` : '全选'],
              ['输出类型', obs.selected_output_names.join(', ')],
            ].map(([k, v]) => (
              <div key={String(k)} className="bg-slate-900/40 rounded-lg px-2.5 py-2 border border-slate-700/30">
                <div className="label mb-1">{k}</div>
                <div className="text-slate-300 font-mono text-xs break-all">{String(v)}</div>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {[
              { label: 'Input 模板', value: meta.input_template },
              { label: 'Target 模板', value: meta.target_template },
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-900/40 rounded-lg px-2.5 py-2 border border-slate-700/30">
                <div className="label mb-1.5">{label}</div>
                <div className="text-slate-400 font-mono text-xs leading-relaxed max-h-16 overflow-y-auto">
                  {value.slice(0, 300)}{value.length > 300 ? '…' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>
    </div>
  )
}
