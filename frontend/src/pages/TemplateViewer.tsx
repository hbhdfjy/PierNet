import { useState, useMemo } from 'react'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { TemplateFileInfo, TemplateRecord, TemplatesResponse, Text2CompScenariosConfig } from '../lib/types'
import {
  LANGUAGE_LABELS, STYLE_LABELS, getSimulatorBadgeClass,
  LANGUAGE_BADGE, STYLE_BADGE, SIMULATOR_BADGE, SIMULATOR_LABELS,
} from '../lib/utils'
import {
  BookTemplate, ChevronLeft, ChevronRight, Filter,
  RefreshCw, Hash, AlignLeft, Target, Info, ChevronDown,
  ArrowRightLeft, Clock, Layers, FileCode,
} from 'lucide-react'
import { cn } from '../lib/utils'

const PAGE_SIZE = 10

// ── 可折叠区块 ────────────────────────────────────────────────────

function Section({ icon, title, children, defaultOpen = true }: {
  icon: React.ReactNode; title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-slate-700/30 last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-slate-700/20 transition-colors group"
      >
        <span className="text-slate-500 group-hover:text-slate-400 transition-colors">{icon}</span>
        <span className="text-base font-medium text-slate-300 flex-1">{title}</span>
        <ChevronDown size={13} className={cn('text-slate-600 transition-transform duration-200', open && 'rotate-180')} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

// ── 模板卡片 ──────────────────────────────────────────────────────

function TemplateCard({ record, index }: { record: TemplateRecord; index: number }) {
  const transformedCount = record.transform_descs.filter(d => d.transform_type !== null).length

  return (
    <div className="card overflow-hidden shadow-lg shadow-black/20">
      {/* 标题行 */}
      <div className="card-header justify-between bg-slate-800/60">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-mono text-slate-600 flex-shrink-0">#{index + 1}</span>
          <span className={cn('badge border', getSimulatorBadgeClass(record.simulator))}>
            {record.simulator}
          </span>
          <span className="text-sm text-slate-400 truncate">{record.scenario}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className={cn('badge border', LANGUAGE_BADGE[record.language] ?? 'bg-slate-700 text-slate-300 border-slate-600')}>
            {LANGUAGE_LABELS[record.language] ?? record.language}
          </span>
          <span className={cn('badge border', STYLE_BADGE[record.style] ?? 'bg-slate-700 text-slate-300 border-slate-600')}>
            {STYLE_LABELS[record.style] ?? record.style}
          </span>
          <span className="badge bg-slate-700/60 text-slate-400 border border-slate-600/40">
            {record.time_mode} · {record.n_time_points}pt
          </span>
        </div>
      </div>

      {/* Input 模板 */}
      <Section icon={<AlignLeft size={13} />} title="Input 模板">
        <div className="bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-300 leading-relaxed
                        max-h-44 overflow-y-auto whitespace-pre-wrap border border-slate-700/30
                        ring-1 ring-inset ring-slate-700/20">
          {record.input_template}
        </div>
      </Section>

      {/* Target 模板 */}
      <Section icon={<Target size={13} />} title="Target 模板">
        <div className="bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-400 leading-relaxed
                        whitespace-pre-wrap border border-slate-700/30 font-mono">
          {record.target_template}
        </div>
      </Section>

      {/* 变换方案 */}
      <Section
        icon={<ArrowRightLeft size={13} />}
        title={`变换方案（${record.param_names.length} 参数${transformedCount > 0 ? `，${transformedCount} 个有变换` : '，无变换'}）`}
        defaultOpen={transformedCount > 0}
      >
        <div className="rounded-lg border border-slate-700/30 overflow-hidden">
          <div className="overflow-x-auto max-h-48">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-800/90">
                <tr className="border-b border-slate-700/40">
                  <th className="px-3 py-2 text-left label">参数名</th>
                  <th className="px-3 py-2 text-left label">变换类型</th>
                  <th className="px-3 py-2 text-left label">描述</th>
                  <th className="px-3 py-2 text-center label">用变换值</th>
                </tr>
              </thead>
              <tbody>
                {record.transform_descs.map((td, i) => {
                  const slot = record.placeholder_schema.find(s => s.param_index === td.param_index)
                  const hasTransform = td.transform_type !== null
                  return (
                    <tr key={i} className={cn(
                      'border-b border-slate-800/40 transition-colors',
                      hasTransform ? 'bg-amber-500/5 hover:bg-amber-500/8' : 'hover:bg-slate-700/20',
                    )}>
                      <td className="px-3 py-1.5 font-mono text-slate-400">
                        {td.param_name}
                        {slot && (
                          <span className="ml-1.5 text-violet-400/70 text-xs">{`{value_${slot.index}}`}</span>
                        )}
                      </td>
                      <td className="px-3 py-1.5">
                        {hasTransform ? (
                          <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/25">
                            {td.transform_type}
                          </span>
                        ) : (
                          <span className="text-slate-700">—</span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 text-slate-500">
                        {record.language === 'en' ? td.note_en : td.note_zh || td.note_en || '—'}
                      </td>
                      <td className="px-3 py-1.5 text-center">
                        {slot?.use_transformed ? (
                          <span className="text-emerald-400 text-xs">✓</span>
                        ) : (
                          <span className="text-slate-700 text-xs">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* 观测方案 */}
      <Section icon={<Clock size={13} />} title="观测方案" defaultOpen={false}>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
          {[
            ['时间模式', record.time_mode],
            ['时间点数', record.n_time_points],
            ['通道级别', record.channel_level],
            ['通道索引', record.channel_indices ? `[${record.channel_indices.join(', ')}]` : '全选（output_info 级别）'],
            ['输出类型', record.selected_output_names.join(', ') || '—'],
            ['原始形状', `${record.timeseries_shape_orig[0]} × ${record.timeseries_shape_orig[1]}`],
            ['观测形状', `${record.timeseries_shape_obs[0]} × ${record.timeseries_shape_obs[1]}`],
            ['时间索引数', record.time_indices.length],
          ].map(([k, v]) => (
            <div key={String(k)} className="bg-slate-900/40 rounded-lg px-2.5 py-2 border border-slate-700/30">
              <div className="label mb-1 text-xs">{k}</div>
              <div className="text-slate-300 font-mono text-xs break-all">{String(v)}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* 输出槽 */}
      <Section icon={<Layers size={13} />} title={`输出槽（${record.output_schema.length} 个占位符）`} defaultOpen={false}>
        <div className="space-y-1.5">
          {record.output_schema.map(slot => (
            <div key={slot.index} className="flex items-center gap-3 bg-slate-900/40 rounded-lg px-3 py-2 border border-slate-700/30 text-xs">
              <span className="text-violet-400 font-mono font-medium flex-shrink-0">{`{output_${slot.index}}`}</span>
              <span className="text-slate-400 flex-1 truncate">{slot.name}</span>
              <span className="text-slate-600 flex-shrink-0">
                slice [{slot.slice_start}, {slot.slice_end ?? '∞'}]
              </span>
              {slot.row_level && (
                <span className="badge bg-sky-500/10 text-sky-400 border border-sky-500/20">行级</span>
              )}
            </div>
          ))}
        </div>
      </Section>

      {/* 参数列表 */}
      <Section icon={<Hash size={13} />} title={`参数名（${record.param_names.length} 维）`} defaultOpen={false}>
        <div className="flex flex-wrap gap-1.5">
          {record.param_names.map((name, i) => (
            <span key={i} className="badge bg-slate-800/60 text-slate-400 border border-slate-700/40 font-mono text-xs">
              {i}: {name}
            </span>
          ))}
        </div>
      </Section>

      {/* Metadata */}
      <Section icon={<Info size={13} />} title="元信息" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            ['Simulator', record.simulator],
            ['Scenario', record.scenario],
            ['Language', record.language],
            ['Style', record.style],
          ].map(([k, v]) => (
            <div key={String(k)} className="bg-slate-900/40 rounded-lg px-2.5 py-2 border border-slate-700/30">
              <div className="label mb-1">{k}</div>
              <div className="text-slate-300 font-mono">{String(v)}</div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  )
}

// ── 主页面 ────────────────────────────────────────────────────────

export default function TemplateViewer() {
  const [scenario, setScenario] = useState('')
  const [page, setPage] = useState(0)
  const [language, setLanguage] = useState('')
  const [style, setStyle] = useState('')

  const { data: templateFiles, isLoading: filesLoading } =
    useSWR<TemplateFileInfo[]>('template-files-viewer', () => api.listTemplateFiles())

  const { data: scenariosCfg } =
    useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () => api.getText2CompScenarios())

  // 场景名 → simulator 映射
  const scenarioSimMap = useMemo(() => {
    const m: Record<string, string> = {}
    if (scenariosCfg) {
      for (const items of Object.values(scenariosCfg))
        for (const s of items) m[s.name] = s.simulator
    }
    return m
  }, [scenariosCfg])

  // 按 simulator 分组
  const grouped = useMemo(() => {
    const g: Record<string, TemplateFileInfo[]> = {}
    for (const f of templateFiles ?? []) {
      const sim = scenarioSimMap[f.scenario] ?? 'unknown'
      if (!g[sim]) g[sim] = []
      g[sim].push(f)
    }
    return g
  }, [templateFiles, scenarioSimMap])

  const selectedScenario = scenario || templateFiles?.[0]?.scenario || ''

  const { data: templatesData, isLoading: tLoading } = useSWR<TemplatesResponse>(
    selectedScenario ? ['template-items', selectedScenario, page, language, style] : null,
    () => api.getTemplateItems(selectedScenario, page, PAGE_SIZE, language || undefined, style || undefined),
    { revalidateOnFocus: false },
  )

  const totalPages = templatesData ? Math.ceil(templatesData.total / PAGE_SIZE) : 0

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 页头 */}
      <div className="page-header flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-violet-500/15 border border-violet-500/25 flex items-center justify-center flex-shrink-0">
            <FileCode size={13} className="text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">模板浏览</h1>
            <p className="text-sm text-slate-500 mt-0.5">Stage 2 生成的语言模板</p>
          </div>
        </div>
      </div>
      <div className="flex-1 flex overflow-hidden">

      {/* ── 左侧场景列表 ── */}
      <div className="w-52 flex-shrink-0 border-r border-slate-700/40 overflow-y-auto bg-slate-900/40 flex flex-col">
        <div className="px-4 py-3 border-b border-slate-700/40 flex-shrink-0">
          <div className="flex items-center gap-2">
            <BookTemplate size={13} className="text-violet-400" />
            <span className="label">模板库</span>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filesLoading && (
            <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
              <RefreshCw size={12} className="animate-spin" /> 加载中…
            </div>
          )}
          {Object.entries(grouped).map(([sim, files]) => {
            const badge = SIMULATOR_BADGE[sim]
            return (
              <div key={sim}>
                <div className={cn('flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider border-b border-slate-800/60 bg-slate-900/30', badge?.text ?? 'text-slate-500')}>
                  <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', badge?.dot ?? 'bg-slate-500')} />
                  {SIMULATOR_LABELS[sim] ?? sim}
                </div>
                {files.map(f => (
                  <button
                    key={f.scenario}
                    onClick={() => { setScenario(f.scenario); setPage(0) }}
                    className={cn(
                      'w-full text-left px-3 py-2 text-xs transition-all duration-150 border-b border-slate-800/30',
                      selectedScenario === f.scenario
                        ? 'bg-violet-500/8 text-violet-300 border-l-2 border-l-violet-500'
                        : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200 border-l-2 border-l-transparent',
                    )}
                  >
                    <div className="font-medium truncate">{f.scenario}</div>
                    <div className="text-slate-600 mt-0.5 tabular-nums">
                      {f.template_count.toLocaleString()} 条
                    </div>
                  </button>
                ))}
              </div>
            )
          })}
          {!filesLoading && (!templateFiles || templateFiles.length === 0) && (
            <div className="px-4 py-8 text-slate-600 text-sm flex flex-col items-center gap-2">
              <BookTemplate size={22} className="opacity-30" />
              <span>暂无模板文件</span>
            </div>
          )}
        </div>
      </div>

      {/* ── 右侧主区域 ── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* 工具栏 */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700/40 bg-slate-900/20 flex-shrink-0">
          <Filter size={12} className="text-slate-600 flex-shrink-0" />
          <select
            className="select text-xs py-1 px-2 w-24 h-7"
            value={language}
            onChange={e => { setLanguage(e.target.value); setPage(0) }}
          >
            <option value="">全部语言</option>
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
          <select
            className="select text-xs py-1 px-2 w-24 h-7"
            value={style}
            onChange={e => { setStyle(e.target.value); setPage(0) }}
          >
            <option value="">全部风格</option>
            <option value="technical">专业技术</option>
            <option value="popular">科普</option>
            <option value="concise">简洁</option>
          </select>

          <div className="flex-1" />

          {templatesData && (
            <span className="text-xs text-slate-500 tabular-nums">
              共 {templatesData.total.toLocaleString()} 条
            </span>
          )}

          {/* 翻页 */}
          {totalPages > 1 && (
            <div className="flex items-center gap-0.5">
              <button
                className="btn-ghost py-1 px-1.5"
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                <ChevronLeft size={13} />
              </button>
              <span className="text-xs text-slate-500 tabular-nums px-1">
                {page + 1} / {totalPages}
              </span>
              <button
                className="btn-ghost py-1 px-1.5"
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
              >
                <ChevronRight size={13} />
              </button>
            </div>
          )}
        </div>

        {/* 内容区 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {tLoading && (
            <div className="flex items-center justify-center gap-2 py-16 text-slate-500">
              <RefreshCw size={15} className="animate-spin" />
              <span className="text-sm">加载模板…</span>
            </div>
          )}

          {!tLoading && !selectedScenario && (
            <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
              <div className="w-14 h-14 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                <BookTemplate size={24} className="text-violet-400/60" />
              </div>
              <p className="text-slate-400 text-sm">左侧选择一个场景查看模板</p>
            </div>
          )}

          {!tLoading && selectedScenario && templatesData?.items.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
              <BookTemplate size={28} className="text-slate-600 opacity-50" />
              <p className="text-slate-500 text-sm">该场景暂无模板（或不符合筛选条件）</p>
            </div>
          )}

          {!tLoading && templatesData?.items.map((record, i) => (
            <TemplateCard
              key={`${record.scenario}-${page * PAGE_SIZE + i}`}
              record={record}
              index={page * PAGE_SIZE + i}
            />
          ))}

          {/* 底部翻页 */}
          {totalPages > 1 && !tLoading && (
            <div className="flex items-center justify-center gap-2 pt-2 pb-4">
              <button
                className="btn-ghost py-1.5 px-3 text-sm"
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                <ChevronLeft size={14} /> 上一页
              </button>
              <span className="text-xs text-slate-500 tabular-nums px-2">
                {page + 1} / {totalPages}
              </span>
              <button
                className="btn-ghost py-1.5 px-3 text-sm"
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
              >
                下一页 <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>
      </div>
      </div>
    </div>
  )
}
