import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { Text2CompScenariosConfig, Text2CompScenario, GenerationConfig, TemplateInfo, LLMConfig, TemplateFileInfo } from '../../lib/types'
import { Cpu, Settings, Layers, RefreshCw, AlertCircle, Sparkles, ChevronDown, ChevronUp, KeyRound, Trash2, FolderOpen, ChevronRight } from 'lucide-react'
import { cn, formatBytes } from '../../lib/utils'
import ScenarioButton from '../components/generation/ScenarioButton'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

export default function TemplateGenerator() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('templates')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 520,
    minWidth: 300,
    maxWidth: 640,
    storageKey: 'piern_template_sidebar_width',
  })

  const { data: scenariosCfg, isLoading: scLoading, mutate: refreshScenarios } =
    useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () => api.getText2CompScenarios(), { revalidateOnMount: true })
  const { data: genCfg } =
    useSWR<GenerationConfig>('config', () => api.getConfig())
  const { data: templatesStatus, mutate: refreshTemplates } =
    useSWR<TemplateInfo[]>('templates', () => api.getTemplatesStatus(), { refreshInterval: 5000 })
  const { data: llmCfg } =
    useSWR<LLMConfig>('llm-config', () => api.getLLMConfig())
  const { data: templateFiles, mutate: refreshTemplateFiles } =
    useSWR<TemplateFileInfo[]>('template-files', () => api.listTemplateFiles(), { refreshInterval: 10000 })

  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nTemplates, setNTemplates] = useState(100)
  const [nTemplatesInput, setNTemplatesInput] = useState('100')
  const [genMode, setGenMode] = useState<'overwrite' | 'skip' | 'append'>('append')
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filesOpen, setFilesOpen] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)
  const [clearingAll, setClearingAll] = useState(false)
  const [trimmingFile, setTrimmingFile] = useState<string | null>(null)
  const [trimTarget, setTrimTarget] = useState<Record<string, string>>({})

  const [languageMix, setLanguageMix] = useState<number | null>(null)
  const [transformProb, setTransformProb] = useState<number | null>(null)
  const [maxWorkers, setMaxWorkers] = useState<number | null>(null)

  useEffect(() => {
    if (genCfg?.generation?.n_samples_per_scenario) {
      setNTemplates(genCfg.generation.n_samples_per_scenario)
      setNTemplatesInput(String(genCfg.generation.n_samples_per_scenario))
    }
    if (genCfg?.generation?.language_mix != null) setLanguageMix(genCfg.generation.language_mix)
    if (genCfg?.generation?.transform_prob != null) setTransformProb(genCfg.generation.transform_prob)
    if (genCfg?.generation?.max_workers != null) setMaxWorkers(genCfg.generation.max_workers)
  }, [genCfg])

  useEffect(() => {
    if (monitor.status === 'done' || monitor.status === 'error') {
      refreshTemplates()
      refreshTemplateFiles()
    }
  }, [monitor.status, refreshTemplates, refreshTemplateFiles])

  const allScenarios: Text2CompScenario[] = scenariosCfg ? Object.values(scenariosCfg).flat() : []
  const scenariosWithData = allScenarios.filter(s => s.has_h5)
  const unregisteredWithData = scenariosWithData.filter(s => !s.registered)
  const templateMap: Record<string, TemplateInfo> = {}
  for (const t of templatesStatus ?? []) templateMap[t.scenario] = t

  const toggle = (name: string) => setSelected(prev => {
    const next = new Set(prev)
    if (next.has(name)) next.delete(name); else next.add(name)
    return next
  })

  const handleLaunch = async () => {
    if (selected.size === 0) { setError('请至少选择一个场景'); return }
    setError(null)
    setLaunching(true)
    try {
      const result = await api.startGenerateTemplates({
        scenarios: Array.from(selected),
        n_templates: nTemplates,
        skip_existing: genMode === 'skip',
        append_existing: genMode === 'append',
        config: 'configs/text2comp/default.yaml',
        ...(languageMix != null ? { language_mix: languageMix } : {}),
        ...(transformProb != null ? { transform_prob: transformProb } : {}),
        ...(maxWorkers != null ? { max_workers: maxWorkers } : {}),
      })
      monitor.start(result.job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  const handleDeleteTemplate = async (scenario: string) => {
    setDeletingFile(scenario)
    try {
      await api.deleteTemplateFile(scenario)
      refreshTemplateFiles()
      refreshTemplates()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '删除失败')
    } finally {
      setDeletingFile(null)
    }
  }

  const handleTrimTemplate = async (scenario: string) => {
    const n = parseInt(trimTarget[scenario] ?? '', 10)
    if (isNaN(n) || n < 1) { setError('请输入有效的数量'); return }
    setTrimmingFile(scenario)
    try {
      const res = await api.trimTemplateFile(scenario, n)
      if (res.after !== res.before) refreshTemplateFiles()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '截断失败')
    } finally {
      setTrimmingFile(null)
    }
  }

  const handleClearAllTemplates = async () => {
    if (!confirm('确认清空所有模板文件？此操作不可撤销。')) return
    setClearingAll(true)
    try {
      await api.clearAllTemplates()
      refreshTemplateFiles()
      refreshTemplates()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '清空失败')
    } finally {
      setClearingAll(false)
    }
  }

  const totalTemplates = Object.values(templateMap).reduce((s, t) => s + t.template_count, 0)
  const canLaunch = !monitor.status || monitor.status === 'idle' || monitor.status === 'done' || monitor.status === 'error' || monitor.status === 'terminated'
  const llmReady = llmCfg?.has_api_key === true

  return (
    <div className="workbench-shell">

      {/* ── 左栏：配置（可拖动宽度，内部分区滚动）── */}
      <div
        className="workbench-sidebar"
        style={{ width: sidebarWidth }}
      >
        {/* 顶部：页头（固定）*/}
        <div className="workbench-sidebar-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-violet-500/20 border border-violet-500/30 flex items-center justify-center flex-shrink-0">
                <Cpu size={14} className="text-violet-400" />
              </div>
              <h1 className="text-lg font-bold text-white">模板生成</h1>
              <span className="badge bg-violet-500/15 text-violet-300 border border-violet-500/20 text-xs">Stage 2</span>
            </div>
            {totalTemplates > 0 && (
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <Sparkles size={11} className="text-violet-500/70" />
                {totalTemplates.toLocaleString()}
              </div>
            )}
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">调用 LLM 生成语言模板，可被 Stage 3 反复复用</p>
        </div>

        {/* 中部：场景选择（flex-1，独立滚动）*/}
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {/* 场景工具栏（固定）*/}
          <div className="flex-shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700/20">
            <div className="flex items-center gap-2">
              <Layers size={12} className="text-slate-400" />
              <span className="font-medium text-slate-300 text-sm">选择场景</span>
              {selected.size > 0 && (
                <span className="badge bg-violet-500/15 text-violet-300 border border-violet-500/20 text-xs py-0.5">
                  {selected.size} 已选
                </span>
              )}
            </div>
            <div className="flex items-center gap-0.5">
              {(['全选', '无模板', '清空'] as const).map((label, i) => (
                <button key={label} className="btn-ghost py-0.5 px-2 text-sm"
                  onClick={[
                    () => setSelected(new Set(scenariosWithData.map(s => s.name))),
                    () => setSelected(new Set(scenariosWithData.filter(s => !templateMap[s.name]).map(s => s.name))),
                    () => setSelected(new Set()),
                  ][i]}>
                  {label}
                </button>
              ))}
              <button className="btn-ghost py-0.5 px-1.5" onClick={() => { refreshScenarios(); refreshTemplates() }}>
                <RefreshCw size={11} className={scLoading ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>

          {/* 场景列表（可滚动）*/}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {scLoading && (
              <div className="flex items-center gap-2 text-slate-500 text-xs py-2">
                <RefreshCw size={11} className="animate-spin text-violet-500" /> 扫描数据目录…
              </div>
            )}

            {/* 未注册警告 */}
            {!scLoading && unregisteredWithData.length > 0 && (
              <div className="flex items-start gap-2 bg-amber-500/8 border border-amber-500/20 rounded-xl px-3 py-2.5">
                <AlertCircle size={13} className="text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-amber-300 font-medium mb-1">
                    {unregisteredWithData.length} 个场景有数据但未注册元数据
                  </p>
                  <p className="text-xs text-slate-500 mb-2">未注册的场景生成时会因缺少 domain 信息而失败。</p>
                  <button className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 transition-colors"
                    onClick={() => navigate('/register')}>
                    前往注册数据集 <ChevronRight size={11} />
                  </button>
                </div>
              </div>
            )}

            {scenariosCfg && Object.entries(scenariosCfg).map(([dirKey, list]) => (
              <div key={dirKey}>
                <div className="workbench-group-label">{dirKey}</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {list.map(s => (
                    <ScenarioButton
                      key={s.name} s={s}
                      active={selected.has(s.name)}
                      onClick={() => toggle(s.name)}
                      templateCount={templateMap[s.name]?.template_count}
                      disabled={!s.has_h5}
                      tone="violet"
                    />
                  ))}
                </div>
              </div>
            ))}
            {!scLoading && allScenarios.length === 0 && (
              <div className="flex flex-col items-center gap-3 py-8 text-center">
                <Layers size={20} className="text-slate-700" />
                <div>
                  <p className="text-slate-500 text-xs font-medium">未找到任何场景</p>
                  <p className="text-slate-600 text-sm mt-1">请先配置数据目录或运行 Stage 1 物理仿真</p>
                </div>
                <button className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 transition-colors"
                  onClick={() => navigate('/data-dirs')}>
                  配置数据目录 <ChevronRight size={11} />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* 底部：参数 + 按钮（固定，不滚动）*/}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/20">

          {/* 参数区 */}
          <div className="px-4 py-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Settings size={12} className="text-slate-500" />
              <span className="text-sm font-medium text-slate-400">参数</span>
            </div>

            {/* 模板数 */}
            <div>
              <label className="label block mb-1 text-sm">每场景模板数</label>
              <input
                type="number"
                className="input w-full text-xs py-1.5 px-3"
                value={nTemplatesInput}
                min={1} max={10000}
                onChange={e => {
                  setNTemplatesInput(e.target.value)
                  const n = parseInt(e.target.value, 10)
                  if (!isNaN(n) && n >= 1) setNTemplates(n)
                }}
                onBlur={() => {
                  const n = parseInt(nTemplatesInput, 10)
                  const clamped = isNaN(n) ? 1 : Math.max(1, Math.min(10000, n))
                  setNTemplates(clamped)
                  setNTemplatesInput(String(clamped))
                }}
              />
            </div>

            {/* 生成模式三选一 */}
            <div>
              <label className="label block mb-1.5 text-sm">已有模板时的处理方式</label>
              <div className="grid grid-cols-3 gap-1.5">
                {([
                  { value: 'append',    label: '继续生成', desc: '追加到现有数量', color: 'emerald' },
                  { value: 'skip',      label: '跳过场景', desc: '已有则不处理',   color: 'slate'   },
                  { value: 'overwrite', label: '重新生成', desc: '清空后重新写入', color: 'red'     },
                ] as const).map(({ value, label, desc, color }) => (
                  <button
                    key={value}
                    onClick={() => setGenMode(value)}
                    className={cn(
                      'flex flex-col items-start px-2.5 py-2 rounded-xl border text-left transition-all',
                      genMode === value
                        ? color === 'emerald' ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                        : color === 'red'     ? 'bg-red-500/15 border-red-500/40 text-red-300'
                        :                      'bg-slate-600/30 border-slate-500/40 text-slate-200'
                        : 'bg-slate-800/40 border-slate-700/30 text-slate-500 hover:border-slate-600/50 hover:text-slate-400',
                    )}
                  >
                    <span className="text-xs font-medium leading-tight">{label}</span>
                    <span className="text-xs text-slate-500 leading-tight mt-0.5">{desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* 高级滑块 */}
            <div className="grid grid-cols-3 gap-2">
              {[
                {
                  label: '并发',
                  value: maxWorkers ?? genCfg?.generation?.max_workers ?? 1,
                  min: 1, max: 100,
                  display: String(maxWorkers ?? genCfg?.generation?.max_workers ?? 1),
                  onChange: (v: number) => setMaxWorkers(v),
                },
                {
                  label: '中文',
                  value: Math.round((languageMix ?? genCfg?.generation?.language_mix ?? 0.5) * 100),
                  min: 0, max: 100,
                  display: `${Math.round((languageMix ?? genCfg?.generation?.language_mix ?? 0.5) * 100)}%`,
                  onChange: (v: number) => setLanguageMix(v / 100),
                },
                {
                  label: '变换',
                  value: Math.round((transformProb ?? genCfg?.generation?.transform_prob ?? 0.1) * 100),
                  min: 0, max: 50,
                  display: `${Math.round((transformProb ?? genCfg?.generation?.transform_prob ?? 0.1) * 100)}%`,
                  onChange: (v: number) => setTransformProb(v / 100),
                },
              ].map(({ label, value, min, max, display, onChange }) => (
                <div key={label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="label text-xs">{label}</span>
                    <span className="text-xs text-slate-500 tabular-nums">{display}</span>
                  </div>
                  <input type="range" className="w-full accent-violet-500 h-1" min={min} max={max}
                    value={value} onChange={e => onChange(parseInt(e.target.value))} />
                </div>
              ))}
            </div>

            {/* LLM 配置入口 */}
            <button
              onClick={() => navigate('/llm-config')}
              className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-slate-800/40 border border-slate-700/30 hover:border-slate-600/50 hover:bg-slate-700/30 transition-all text-left"
            >
              <div className="flex items-center gap-2 min-w-0">
                <KeyRound size={12} className="text-slate-500 flex-shrink-0" />
                <span className="text-xs text-slate-400 truncate">
                  {llmCfg ? `${llmCfg.provider} · ${llmCfg.model || '未设置模型'}` : 'LLM 配置'}
                </span>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
                {llmCfg?.has_api_key
                  ? <span className="text-emerald-500 text-xs">✓</span>
                  : <span className="text-red-400 text-xs">⚠</span>
                }
                <span className="text-xs text-slate-600">→</span>
              </div>
            </button>
          </div>

          {/* 错误 */}
          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* 启动按钮 */}
          {canLaunch && (
            <div className="px-4 pb-4 space-y-1.5">
              <button
                className="btn-primary w-full py-2.5 text-sm justify-center shadow-lg"
                style={{ background: selected.size > 0 && !launching && llmReady ? 'linear-gradient(135deg, #7c3aed, #6d28d9)' : undefined }}
                onClick={handleLaunch}
                disabled={launching || selected.size === 0 || !llmReady}
              >
                {launching
                  ? <><RefreshCw size={14} className="animate-spin" /> 启动中…</>
                  : !llmReady
                  ? <><AlertCircle size={14} /> 请先配置 LLM API Key</>
                  : <><Cpu size={14} /> 开始生成{selected.size > 0 ? `（${selected.size} 个场景）` : ''}</>
                }
              </button>
              {(monitor.status === 'done' || monitor.status === 'error' || monitor.status === 'terminated') && (
                <button className="btn-ghost w-full py-1.5 justify-center text-xs" onClick={monitor.reset}>
                  重新配置
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <ResizeHandle onMouseDown={onResizeStart} color="violet" />

      {/* ── 右栏：监控 + 文件管理 ── */}
      <div className="workbench-main-scroll">

        <JobMonitorPanel
          status={monitor.status}
          logs={monitor.logs}
          progress={monitor.progress}
          stats={monitor.stats}
          autoScroll={monitor.autoScroll}
          onAutoScrollChange={monitor.setAutoScroll}
          onStop={monitor.stop}
          onDone={() => navigate('/fill')}
          doneLabel="去填充样本"
          jobId={monitor.jobId}
          jobIds={monitor.jobIds}
          stageLabel="模板生成"
          stageColor="text-violet-400"
          accentColor="violet"
        />

        {monitor.status === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
              <Cpu size={24} className="text-violet-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未启动任务</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景并配置参数，点击「开始生成」</p>
            </div>
          </div>
        )}

        {/* 文件管理 */}
        <div className="card overflow-hidden">
          <button
            onClick={() => setFilesOpen(o => !o)}
            className="w-full card-header accordion-card-header justify-between transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <FolderOpen size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-base">模板文件管理</span>
              {templateFiles && templateFiles.length > 0 && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">
                  {templateFiles.length} 个场景
                </span>
              )}
            </div>
            {filesOpen ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
          </button>

          {filesOpen && (
            <div className="p-3 space-y-2">
              {(!templateFiles || templateFiles.length === 0) ? (
                <p className="text-slate-500 text-xs text-center py-3">暂无模板文件</p>
              ) : (
                <>
                  <div className="list-table-scroll">
                    <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700/40">
                        <th className="px-3 py-1.5 text-left label">场景</th>
                        <th className="px-3 py-1.5 text-right label">模板数</th>
                        <th className="px-3 py-1.5 text-right label">大小</th>
                        <th className="px-3 py-1.5 text-right label">截断至</th>
                        <th className="px-3 py-1.5 text-right label">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {templateFiles.map(f => (
                        <tr key={f.scenario} className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors">
                          <td className="px-3 py-1.5 font-mono text-slate-300">{f.scenario}</td>
                          <td className="px-3 py-1.5 text-right tabular-nums text-violet-400">{f.template_count.toLocaleString()}</td>
                          <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">{formatBytes(f.file_size_bytes)}</td>
                          <td className="px-3 py-1.5 text-right">
                            <div className="flex items-center justify-end gap-1">
                              <input
                                type="number" min={1}
                                placeholder={String(f.template_count)}
                                value={trimTarget[f.scenario] ?? ''}
                                onChange={e => setTrimTarget(prev => ({ ...prev, [f.scenario]: e.target.value }))}
                                className="w-24 bg-slate-800 border border-slate-600/60 rounded-lg px-2 py-0.5 text-xs text-slate-200 text-right focus:outline-none focus:border-sky-500/50"
                              />
                              <button
                                className="btn-ghost py-0.5 px-1.5 text-amber-400 hover:text-amber-300 text-xs"
                                onClick={() => handleTrimTemplate(f.scenario)}
                                disabled={trimmingFile === f.scenario || !trimTarget[f.scenario]}
                                title="截断到指定数量">
                                {trimmingFile === f.scenario ? <RefreshCw size={10} className="animate-spin" /> : '截断'}
                              </button>
                            </div>
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            <button className="btn-ghost py-0.5 px-1.5 text-red-400 hover:text-red-300"
                              onClick={() => handleDeleteTemplate(f.scenario)}
                              disabled={deletingFile === f.scenario}>
                              {deletingFile === f.scenario
                                ? <RefreshCw size={10} className="animate-spin" />
                                : <Trash2 size={10} />}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                    </table>
                  </div>
                  <div className="flex justify-end pt-0.5">
                    <button className="btn-ghost py-1 px-2.5 text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                      onClick={handleClearAllTemplates} disabled={clearingAll}>
                      {clearingAll ? <RefreshCw size={11} className="animate-spin" /> : <Trash2 size={11} />}
                      清空全部
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
