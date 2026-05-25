import { useEffect, useMemo, useState } from 'react'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { TemplateFileInfo, TemplateRecord, TemplatesResponse, Text2CompScenariosConfig } from '../../lib/types'
import {
  LANGUAGE_BADGE,
  LANGUAGE_LABELS,
  SIMULATOR_BADGE,
  SIMULATOR_LABELS,
  STYLE_BADGE,
  STYLE_LABELS,
  getSimulatorBadgeClass,
} from '../../lib/utils'
import {
  AlignLeft,
  ArrowRightLeft,
  BookTemplate,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  FileCode,
  Filter,
  Hash,
  Info,
  Layers,
  RefreshCw,
  Target,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { templateScenarioSimulatorMap } from '../templateData'

const PAGE_SIZE = 10

function Section({
  icon,
  title,
  children,
  defaultOpen = true,
}: {
  icon: React.ReactNode
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
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
        <ChevronDown
          size={13}
          className={cn('text-slate-600 transition-transform duration-200', open && 'rotate-180')}
        />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

function TemplateCard({ record, index }: { record: TemplateRecord; index: number }) {
  const transformedCount = record.transform_descs.filter(d => d.transform_type !== null).length

  return (
    <div className="card overflow-hidden">
      <div className="card-header accordion-card-header record-card-header">
        <div className="record-card-main">
          <span className="record-card-index">#{index + 1}</span>
          <span className={cn('badge border', getSimulatorBadgeClass(record.simulator))}>{record.simulator}</span>
          <span className="record-card-title">{record.scenario}</span>
        </div>
        <div className="record-card-meta">
          <span
            className={cn(
              'badge border',
              LANGUAGE_BADGE[record.language] ?? 'bg-slate-700 text-slate-300 border-slate-600',
            )}
          >
            {LANGUAGE_LABELS[record.language] ?? record.language}
          </span>
          <span
            className={cn('badge border', STYLE_BADGE[record.style] ?? 'bg-slate-700 text-slate-300 border-slate-600')}
          >
            {STYLE_LABELS[record.style] ?? record.style}
          </span>
          <span className="badge bg-slate-700/60 text-slate-400 border border-slate-600/40">
            {record.time_mode} · {record.n_time_points}pt
          </span>
        </div>
      </div>

      <Section icon={<AlignLeft size={13} />} title="输入模板">
        <div className="bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-300 leading-relaxed max-h-44 overflow-y-auto whitespace-pre-wrap border border-slate-700/30 ring-1 ring-inset ring-slate-700/20">
          {record.input_template}
        </div>
      </Section>

      <Section icon={<Target size={13} />} title="目标模板">
        <div className="bg-slate-900/50 rounded-lg p-3.5 text-sm text-slate-400 leading-relaxed whitespace-pre-wrap border border-slate-700/30 font-mono">
          {record.target_template}
        </div>
      </Section>

      <Section
        icon={<ArrowRightLeft size={13} />}
        title={`参数变换（${record.param_names.length} 个参数${transformedCount > 0 ? `，${transformedCount} 个启用` : '，未启用'}）`}
        defaultOpen={transformedCount > 0}
      >
        <div className="rounded-lg border border-slate-700/30 overflow-hidden">
          <div className="list-table-scroll">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-800/90">
                <tr className="border-b border-slate-700/40">
                  <th className="px-3 py-2 text-left label">Param</th>
                  <th className="px-3 py-2 text-left label">Transform</th>
                  <th className="px-3 py-2 text-left label">Description</th>
                  <th className="px-3 py-2 text-center label">Uses transformed</th>
                </tr>
              </thead>
              <tbody>
                {record.transform_descs.map((td, i) => {
                  const slot = record.placeholder_schema.find(s => s.param_index === td.param_index)
                  const hasTransform = td.transform_type !== null
                  return (
                    <tr
                      key={i}
                      className={cn(
                        'border-b border-slate-800/40 transition-colors',
                        hasTransform ? 'bg-amber-500/5 hover:bg-amber-500/8' : 'hover:bg-slate-700/20',
                      )}
                    >
                      <td className="px-3 py-1.5 font-mono text-slate-400">
                        {td.param_name}
                        {slot && <span className="ml-1.5 text-violet-400/70 text-xs">{`{value_${slot.index}}`}</span>}
                      </td>
                      <td className="px-3 py-1.5">
                        {hasTransform ? (
                          <span className="badge bg-amber-500/15 text-amber-300 border border-amber-500/25">
                            {td.transform_type}
                          </span>
                        ) : (
                          <span className="text-slate-600">无</span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 text-slate-500">
                        {record.language === 'en' ? td.note_en : td.note_zh || td.note_en || '无说明'}
                      </td>
                      <td className="px-3 py-1.5 text-center">
                        {slot?.use_transformed ? (
                          <span className="text-emerald-400 text-xs">是</span>
                        ) : (
                          <span className="text-slate-500 text-xs">否</span>
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

      <Section icon={<Clock size={13} />} title="观测配置" defaultOpen={false}>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
          {[
            ['时间模式', record.time_mode],
            ['时间点', record.n_time_points],
            ['观测通道', record.channel_indices?.length ?? record.timeseries_shape_obs[0]],
            [
              '通道索引',
              record.channel_indices && record.channel_indices.length > 0
                ? `[${record.channel_indices.join(', ')}]`
                : '全部观测',
            ],
            ['输出变量', record.selected_output_names.join(', ') || '无'],
            ['原始形状', `${record.timeseries_shape_orig[0]} x ${record.timeseries_shape_orig[1]}`],
            ['观测形状', `${record.timeseries_shape_obs[0]} x ${record.timeseries_shape_obs[1]}`],
            ['时间索引数', record.time_indices.length],
          ].map(([k, v]) => (
            <div key={String(k)} className="bg-slate-900/40 rounded-lg px-2.5 py-2 border border-slate-700/30">
              <div className="label mb-1 text-xs">{k}</div>
              <div className="pretty-tooltip min-w-0" data-tooltip={String(v)}>
                <div className="truncate text-slate-300 font-mono text-xs">{String(v)}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={<Layers size={13} />} title={`输出槽位 (${record.output_schema.length})`} defaultOpen={false}>
        <div className="list-scroll-md space-y-1.5">
          {record.output_schema.map(slot => (
            <div
              key={slot.index}
              className="flex items-center gap-3 bg-slate-900/40 rounded-lg px-3 py-2 border border-slate-700/30 text-xs"
            >
              <span className="text-violet-400 font-mono font-medium flex-shrink-0">{`{output_${slot.index}}`}</span>
              <span className="pretty-tooltip min-w-0 flex-1" data-tooltip={slot.name || undefined}>
                <span className="block truncate text-slate-400">{slot.name || '未命名输出'}</span>
              </span>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={<Hash size={13} />} title={`参数 (${record.param_names.length})`} defaultOpen={false}>
        <div className="flex flex-wrap gap-1.5">
          {record.param_names.map((name, i) => (
            <span key={i} className="badge bg-slate-800/60 text-slate-400 border border-slate-700/40 font-mono text-xs">
              {i}: {name}
            </span>
          ))}
        </div>
      </Section>

      <Section icon={<Info size={13} />} title="元数据" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            ['仿真器', record.simulator],
            ['场景', record.scenario],
            ['语言', record.language],
            ['风格', record.style],
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

export default function TemplateViewer() {
  const [scenario, setScenario] = useState('')
  const [page, setPage] = useState(0)
  const [language, setLanguage] = useState('')
  const [style, setStyle] = useState('')

  const { data: templateFiles, isLoading: filesLoading } = useSWR<TemplateFileInfo[]>('template-files-viewer', () =>
    api.listTemplateFiles(),
  )

  const { data: scenariosCfg } = useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () =>
    api.getText2CompScenarios(),
  )

  const scenarioSimMap = useMemo(() => {
    return templateScenarioSimulatorMap(scenariosCfg)
  }, [scenariosCfg])

  const grouped = useMemo(() => {
    const g: Record<string, TemplateFileInfo[]> = {}
    for (const f of templateFiles ?? []) {
      const sim = f.simulator || scenarioSimMap[f.scenario] || 'unknown'
      if (!g[sim]) g[sim] = []
      g[sim].push(f)
    }
    return g
  }, [templateFiles, scenarioSimMap])

  const selectedScenario = scenario || templateFiles?.[0]?.scenario || ''

  useEffect(() => {
    if (!templateFiles || !scenario) return
    if (!templateFiles.some(file => file.scenario === scenario)) {
      setScenario('')
      setPage(0)
    }
  }, [templateFiles, scenario])

  const { data: templatesData, isLoading: tLoading } = useSWR<TemplatesResponse>(
    selectedScenario ? ['template-items', selectedScenario, page, language, style] : null,
    () => api.getTemplateItems(selectedScenario, page, PAGE_SIZE, language || undefined, style || undefined),
    { revalidateOnFocus: false },
  )

  const totalPages = templatesData ? Math.ceil(templatesData.total / PAGE_SIZE) : 0

  return (
    <div className="page-shell">
      <div className="page-header flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-violet-500/15 border border-violet-500/25 flex items-center justify-center flex-shrink-0">
            <FileCode size={13} className="text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">模板浏览</h1>
            <p className="text-sm text-slate-500 mt-0.5">阶段 2 语言模板</p>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="page-rail w-52">
          <div className="px-4 py-3 border-b border-slate-700/40 flex-shrink-0">
            <div className="flex items-center gap-2">
              <BookTemplate size={13} className="text-violet-400" />
              <span className="label">模板集</span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {filesLoading && (
              <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
                <RefreshCw size={12} className="animate-spin" /> 加载中...
              </div>
            )}
            {Object.entries(grouped).map(([sim, files]) => {
              const badge = SIMULATOR_BADGE[sim]
              return (
                <div key={sim}>
                  <div
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider border-b border-slate-800/60 bg-slate-900/30',
                      badge?.text ?? 'text-slate-500',
                    )}
                  >
                    <span className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', badge?.dot ?? 'bg-slate-500')} />
                    {SIMULATOR_LABELS[sim] ?? sim}
                  </div>
                  {files.map(f => (
                    <button
                      key={f.scenario}
                      onClick={() => {
                        setScenario(f.scenario)
                        setPage(0)
                      }}
                      className={cn(
                        'w-full text-left px-3 py-2 text-xs transition-all duration-150 border-b border-slate-800/30',
                        selectedScenario === f.scenario
                          ? 'bg-violet-500/8 text-violet-300 border-l-2 border-l-violet-500'
                          : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200 border-l-2 border-l-transparent',
                      )}
                    >
                      <div className="font-medium truncate">{f.scenario}</div>
                      {f.simulator && <div className="text-slate-700 mt-0.5 font-mono">{f.simulator}</div>}
                      <div className="text-slate-600 mt-0.5 tabular-nums">{f.template_count.toLocaleString()} 条</div>
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

        <div className="page-content">
          <div className="toolbar-strip flex items-center gap-2 px-3 py-2">
            <Filter size={12} className="text-slate-600 flex-shrink-0" />
            <select
              className="select text-xs py-1 px-2 w-28 h-7"
              value={language}
              onChange={e => {
                setLanguage(e.target.value)
                setPage(0)
              }}
            >
              <option value="">全部语言</option>
              <option value="zh">中文</option>
              <option value="en">English</option>
            </select>
            <select
              className="select text-xs py-1 px-2 w-28 h-7"
              value={style}
              onChange={e => {
                setStyle(e.target.value)
                setPage(0)
              }}
            >
              <option value="">全部风格</option>
              <option value="technical">专业技术</option>
              <option value="popular">科普</option>
              <option value="concise">简洁</option>
            </select>

            <div className="flex-1" />

            {templatesData && (
              <span className="text-xs text-slate-500 tabular-nums">{templatesData.total.toLocaleString()} 条</span>
            )}

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

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {tLoading && (
              <div className="flex items-center justify-center gap-2 py-16 text-slate-500">
                <RefreshCw size={15} className="animate-spin" />
                <span className="text-sm">模板加载中...</span>
              </div>
            )}

            {!tLoading && !selectedScenario && (
              <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
                <div className="w-14 h-14 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                  <BookTemplate size={24} className="text-violet-400/60" />
                </div>
                <p className="text-slate-400 text-sm">请在左侧选择场景</p>
              </div>
            )}

            {!tLoading && selectedScenario && templatesData?.items.length === 0 && (
              <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
                <BookTemplate size={28} className="text-slate-600 opacity-50" />
                <p className="text-slate-500 text-sm">当前筛选无匹配模板</p>
              </div>
            )}

            {!tLoading &&
              templatesData?.items.map((record, i) => (
                <TemplateCard
                  key={`${record.scenario}-${page * PAGE_SIZE + i}`}
                  record={record}
                  index={page * PAGE_SIZE + i}
                />
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
