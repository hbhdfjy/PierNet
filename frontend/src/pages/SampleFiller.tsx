import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { Text2CompScenariosConfig, Text2CompScenario, GenerationConfig, TemplateInfo, SampleFileInfo } from '../lib/types'
import { FlaskConical, Settings, Layers, RefreshCw, AlertCircle, Sparkles, FileText, Trash2, FolderOpen, ChevronDown, ChevronUp } from 'lucide-react'
import { cn, formatBytes } from '../lib/utils'
import ScenarioButton from '../components/generation/ScenarioButton'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import { useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'

export default function SampleFiller() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('fill')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 600,
    minWidth: 300,
    maxWidth: 600,
    storageKey: 'piern_fill_sidebar_width',
  })

  const { data: scenariosCfg, isLoading: scLoading, mutate: refreshScenarios } =
    useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () => api.getText2CompScenarios(), { revalidateOnMount: true })
  const { data: genCfg } =
    useSWR<GenerationConfig>('config', () => api.getConfig())
  const { data: templatesStatus, mutate: refreshTemplates } =
    useSWR<TemplateInfo[]>('templates', () => api.getTemplatesStatus(), { refreshInterval: 5000 })
  const { data: sampleFiles, isLoading: sfLoading, mutate: refreshSampleFiles } =
    useSWR<SampleFileInfo[]>('sample-files', () => api.listSampleFiles(), { refreshInterval: 10000 })

  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nSamples, setNSamples] = useState(100)
  const [skipExisting, setSkipExisting] = useState(false)
  const [seed, setSeed] = useState<number>(42)
  const [precision, setPrecision] = useState(4)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filesOpen, setFilesOpen] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)
  const [clearingAll, setClearingAll] = useState(false)

  useEffect(() => {
    if (genCfg?.generation?.n_samples_per_scenario) setNSamples(genCfg.generation.n_samples_per_scenario)
  }, [genCfg])

  useEffect(() => {
    if (monitor.status === 'done' || monitor.status === 'error') {
      refreshTemplates()
      // 刷新场景列表以更新 existing_jsonl_count
      refreshScenarios(undefined, { revalidate: true })
      refreshSampleFiles()
    }
  }, [monitor.status, refreshTemplates, refreshScenarios, refreshSampleFiles])

  const allScenarios: Text2CompScenario[] = scenariosCfg ? Object.values(scenariosCfg).flat() : []
  const templateMap: Record<string, TemplateInfo> = {}
  for (const t of templatesStatus ?? []) templateMap[t.scenario] = t

  const scenariosWithTemplates = allScenarios.filter(s => templateMap[s.name])

  const toggle = (name: string) => {
    if (!templateMap[name]) return
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name); else next.add(name)
      return next
    })
  }

  const handleLaunch = async () => {
    if (selected.size === 0) { setError('请至少选择一个场景'); return }
    const missing = Array.from(selected).filter(n => !templateMap[n])
    if (missing.length > 0) {
      setError(`以下场景缺少模板：${missing.slice(0, 3).join(', ')}${missing.length > 3 ? '…' : ''}`)
      return
    }
    setError(null)
    setLaunching(true)
    try {
      const result = await api.startFillSamples({
        scenarios: Array.from(selected),
        n_samples: nSamples,
        templates_dir: '',
        output_dir: '',
        skip_existing: skipExisting,
        config: 'configs/text2comp/default.yaml',
        seed,
        precision,
      })
      monitor.start(result.job_id)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  const handleDeleteSample = async (scenario: string) => {
    setDeletingFile(scenario)
    try {
      await api.deleteSampleFile(scenario)
      refreshSampleFiles()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '删除失败')
    } finally {
      setDeletingFile(null)
    }
  }

  const handleClearAllSamples = async () => {
    if (!confirm('确认清空所有样本文件？此操作不可撤销。')) return
    setClearingAll(true)
    try {
      await api.clearAllSamples()
      refreshSampleFiles()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '清空失败')
    } finally {
      setClearingAll(false)
    }
  }

  const isRunning = monitor.status === 'running'
  const canLaunch = !isRunning

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* ── 左栏：配置（可拖动宽度，内部分区滚动）── */}
      <div
        className="flex flex-col overflow-hidden border-r border-slate-700/40 flex-shrink-0"
        style={{ width: sidebarWidth }}
      >
        {/* 顶部：页头（固定）*/}
        <div className="flex-shrink-0 px-4 pt-4 pb-3 border-b border-slate-700/30">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
              <FlaskConical size={14} className="text-emerald-400" />
            </div>
            <h1 className="text-base font-bold text-white">样本填充</h1>
            <span className="badge bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 text-xs">阶段三</span>
          </div>
          <p className="text-slate-500 text-xs mt-1.5 ml-9">
            将模板库与 HDF5 数值结合，生成最终训练样本。
            <span className="text-emerald-500/80"> 不调用 LLM</span>
          </p>
        </div>

        {/* 模板库状态（固定，紧凑）*/}
        {Object.keys(templateMap).length > 0 ? (
          <div className="flex-shrink-0 border-b border-slate-700/20">
            <div className="flex items-center justify-between px-4 py-2">
              <div className="flex items-center gap-1.5">
                <FileText size={11} className="text-slate-500" />
                <span className="text-xs text-slate-400 font-medium">模板库</span>
                <span className="badge bg-slate-700/50 text-slate-500 border border-slate-600/30 text-xs py-0.5">
                  {Object.keys(templateMap).length} 个场景
                </span>
              </div>
              <span className="text-xs text-slate-600 tabular-nums">
                共 {Object.values(templateMap).reduce((s, t) => s + t.template_count, 0).toLocaleString()} 条
              </span>
            </div>
          </div>
        ) : (
          <div className="flex-shrink-0 border-b border-slate-700/20 px-4 py-2.5 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sparkles size={11} className="text-slate-600" />
              <span className="text-xs text-slate-600">暂无模板，请先完成阶段二</span>
            </div>
            <button className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
              onClick={() => navigate('/templates')}>
              去生成 →
            </button>
          </div>
        )}

        {/* 中部：场景选择（flex-1，独立滚动）*/}
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {/* 场景工具栏（固定）*/}
          <div className="flex-shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700/20">
            <div className="flex items-center gap-2">
              <Layers size={12} className="text-slate-400" />
              <span className="font-medium text-slate-300 text-xs">选择场景</span>
              {selected.size > 0 && (
                <span className="badge bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 text-xs py-0.5">
                  {selected.size} 已选
                </span>
              )}
            </div>
            <div className="flex items-center gap-0.5">
              <button className="btn-ghost py-0.5 px-2 text-xs"
                onClick={() => setSelected(new Set(scenariosWithTemplates.map(s => s.name)))}>
                全选
              </button>
              <button className="btn-ghost py-0.5 px-2 text-xs" onClick={() => setSelected(new Set())}>清空</button>
              <button className="btn-ghost py-0.5 px-1.5" onClick={() => { refreshScenarios(); refreshTemplates() }}>
                <RefreshCw size={11} className={scLoading ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>

          {/* 场景列表（可滚动）*/}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {scLoading && (
              <div className="flex items-center gap-2 text-slate-500 text-xs py-2">
                <RefreshCw size={11} className="animate-spin" /> 扫描中…
              </div>
            )}
            {scenariosCfg && Object.entries(scenariosCfg).map(([dirKey, list]) => (
              <div key={dirKey}>
                <div className="label mb-2 text-xs">{dirKey}</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {list.map(s => (
                    <ScenarioButton
                      key={s.name} s={s}
                      active={selected.has(s.name)}
                      onClick={() => toggle(s.name)}
                      templateCount={templateMap[s.name]?.template_count}
                      disabled={!templateMap[s.name]}
                    />
                  ))}
                </div>
              </div>
            ))}
            {scenariosWithTemplates.length === 0 && !scLoading && (
              <div className="flex flex-col items-center gap-3 py-8 text-center">
                <Layers size={20} className="text-slate-700" />
                <div>
                  <p className="text-slate-500 text-xs font-medium">暂无有模板的场景</p>
                  <p className="text-slate-600 text-xs mt-1">请先在「模板生成」页面生成语言模板</p>
                </div>
                <button className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors"
                  onClick={() => navigate('/templates')}>
                  前往模板生成 →
                </button>
              </div>
            )}
          </div>
        </div>

        {/* 底部：参数 + 按钮（固定）*/}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/30">
          <div className="px-4 py-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Settings size={12} className="text-slate-500" />
              <span className="text-xs font-medium text-slate-400">参数</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="label block mb-1 text-xs">每场景样本数</label>
                <input type="number" className="input w-full text-xs py-1.5 px-3" value={nSamples}
                  min={1} max={100000} onChange={e => setNSamples(parseInt(e.target.value) || 1)} />
              </div>
              <div>
                <label className="label block mb-1 text-xs">随机种子</label>
                <input type="number" className="input w-full text-xs py-1.5 px-3" value={seed}
                  min={0} onChange={e => setSeed(parseInt(e.target.value) || 0)} />
              </div>
              <div>
                <label className="label block mb-1 text-xs">数值精度（小数位）</label>
                <select className="select w-full text-xs py-1.5 px-3" value={precision}
                  onChange={e => setPrecision(parseInt(e.target.value))}>
                  {[2, 3, 4, 5, 6, 8].map(p => (
                    <option key={p} value={p}>{p} 位</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/40 rounded-xl border border-slate-700/30">
                <input id="skip-f" type="checkbox" className="w-3.5 h-3.5 accent-emerald-500 flex-shrink-0"
                  checked={skipExisting} onChange={e => setSkipExisting(e.target.checked)} />
                <label htmlFor="skip-f" className="text-xs text-slate-400 cursor-pointer leading-snug">
                  跳过已有
                </label>
              </div>
            </div>
          </div>

          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-xs">{error}</span>
            </div>
          )}

          {canLaunch && (
            <div className="px-4 pb-4">
              <button
                className={cn(
                  'w-full py-2.5 text-sm justify-center shadow-lg btn',
                  selected.size > 0 && !launching
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                    : 'bg-slate-700/60 text-slate-500 cursor-not-allowed',
                )}
                onClick={handleLaunch}
                disabled={launching || selected.size === 0}
              >
                {launching
                  ? <><RefreshCw size={14} className="animate-spin" /> 启动中…</>
                  : <><FlaskConical size={14} /> 开始填充{selected.size > 0 ? `（${selected.size} 个场景）` : ''}</>
                }
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── 拖拽手柄 ── */}
      <div
        onMouseDown={onResizeStart}
        className="w-1 flex-shrink-0 cursor-col-resize group relative hover:bg-emerald-500/40 active:bg-emerald-500/60 transition-colors"
        title="拖动调整宽度"
      >
        <div className="absolute inset-y-0 left-0 w-px bg-slate-700/60 group-hover:bg-emerald-500/50 transition-colors" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-8 rounded-full bg-slate-700/80 border border-slate-600/60 group-hover:bg-emerald-500/60 group-hover:border-emerald-400/50 transition-all flex flex-col items-center justify-center gap-0.5 opacity-0 group-hover:opacity-100">
          <div className="w-0.5 h-0.5 rounded-full bg-slate-300/80" />
          <div className="w-0.5 h-0.5 rounded-full bg-slate-300/80" />
          <div className="w-0.5 h-0.5 rounded-full bg-slate-300/80" />
        </div>
      </div>

      {/* ── 右栏：监控 + 文件管理 ── */}
      <div className="flex-1 flex flex-col overflow-y-auto p-5 space-y-4 min-w-0">

        {monitor.status === 'running' && (
          <JobMonitorPanel
            status={monitor.status}
            logs={monitor.logs}
            progress={monitor.progress}
            stats={monitor.stats}
            autoScroll={monitor.autoScroll}
            onAutoScrollChange={monitor.setAutoScroll}
            onStop={monitor.stop}
            onDone={() => navigate('/samples')}
            doneLabel="查看样本"
            jobId={monitor.jobId}
            jobIds={monitor.jobIds}
            stageLabel="样本填充"
            stageColor="text-emerald-400"
            accentColor="emerald"
          />
        )}

        {(monitor.status === 'idle' || monitor.status === 'done') && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            {monitor.status === 'done' ? (
              <>
                <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                  <FlaskConical size={24} className="text-emerald-400" />
                </div>
                <div>
                  <p className="text-slate-300 text-sm font-medium">填充完成</p>
                  <p className="text-slate-500 text-xs mt-1">样本已写入 data/text2comp/</p>
                </div>
                <div className="flex gap-2 mt-1">
                  <button className="btn-ghost py-1.5 px-3 text-xs" onClick={monitor.reset}>重新配置</button>
                  <button className="btn py-1.5 px-3 text-xs bg-emerald-600 hover:bg-emerald-500 text-white" onClick={() => navigate('/samples')}>查看样本 →</button>
                </div>
              </>
            ) : (
              <>
                <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                  <FlaskConical size={24} className="text-emerald-400/60" />
                </div>
                <div>
                  <p className="text-slate-400 text-sm font-medium">尚未启动任务</p>
                  <p className="text-slate-600 text-xs mt-1">在左侧选择场景并配置参数，点击「开始填充」</p>
                </div>
              </>
            )}
          </div>
        )}

        {(monitor.status === 'error' || monitor.status === 'terminated') && (
          <JobMonitorPanel
            status={monitor.status}
            logs={monitor.logs}
            progress={monitor.progress}
            stats={monitor.stats}
            autoScroll={monitor.autoScroll}
            onAutoScrollChange={monitor.setAutoScroll}
            onStop={monitor.stop}
            onDone={() => navigate('/samples')}
            doneLabel="查看样本"
            jobId={monitor.jobId}
            jobIds={monitor.jobIds}
            stageLabel="样本填充"
            stageColor="text-emerald-400"
            accentColor="emerald"
          />
        )}

        {/* 文件管理 */}
        <div className="card overflow-hidden">
          <button
            onClick={() => setFilesOpen(o => !o)}
            className="w-full card-header justify-between hover:bg-slate-700/20 transition-colors py-3"
          >
            <div className="flex items-center gap-2">
              <FolderOpen size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-sm">样本文件管理</span>
              {sampleFiles && sampleFiles.length > 0 && (
                <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30 text-xs">
                  {sampleFiles.length} 个场景
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span
                className="btn-ghost py-0.5 px-1.5 text-xs"
                onClick={e => { e.stopPropagation(); refreshSampleFiles() }}
                title="刷新"
              >
                <RefreshCw size={11} className={sfLoading ? 'animate-spin' : ''} />
              </span>
              {filesOpen ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
            </div>
          </button>

          {filesOpen && (
            <div className="p-3 space-y-2">
              {sfLoading ? (
                <div className="flex items-center justify-center gap-2 py-3 text-slate-500">
                  <RefreshCw size={12} className="animate-spin" />
                  <span className="text-xs">加载中…</span>
                </div>
              ) : (!sampleFiles || sampleFiles.length === 0) ? (
                <p className="text-slate-500 text-xs text-center py-3">暂无样本文件</p>
              ) : (
                <>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700/40">
                        <th className="px-3 py-1.5 text-left label">场景</th>
                        <th className="px-3 py-1.5 text-right label">样本数</th>
                        <th className="px-3 py-1.5 text-right label">大小</th>
                        <th className="px-3 py-1.5 text-right label">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sampleFiles.map(f => (
                        <tr key={f.scenario} className="border-b border-slate-800/40 hover:bg-slate-700/20 transition-colors">
                          <td className="px-3 py-1.5 font-mono text-slate-300">{f.scenario}</td>
                          <td className="px-3 py-1.5 text-right tabular-nums text-emerald-400">{f.sample_count.toLocaleString()}</td>
                          <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">{formatBytes(f.file_size_bytes)}</td>
                          <td className="px-3 py-1.5 text-right">
                            <button className="btn-ghost py-0.5 px-1.5 text-red-400 hover:text-red-300"
                              onClick={() => handleDeleteSample(f.scenario)}
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
                  <div className="flex justify-end pt-0.5">
                    <button className="btn-ghost py-1 px-2.5 text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                      onClick={handleClearAllSamples} disabled={clearingAll}>
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
