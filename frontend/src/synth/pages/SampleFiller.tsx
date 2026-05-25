import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeed } from '../../lib/seedContext'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type {
  Text2CompScenariosConfig,
  Text2CompScenario,
  GenerationConfig,
  TemplateInfo,
  JobStatus,
} from '../../lib/types'
import { FlaskConical, Settings, Layers, RefreshCw, AlertCircle, Sparkles, FileText, FolderOpen } from 'lucide-react'
import { cn } from '../../lib/utils'
import ScenarioButton from '../components/generation/ScenarioButton'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { isRestartableJobStatus, isTerminalJobStatus, useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'
import {
  duplicateText2CompScenarioNames,
  selectableText2CompScenarios,
  text2compScenarioKey,
} from '../text2compScenario'
import {
  SYNTH_SAMPLE_COUNT_MAX,
  SYNTH_SAMPLE_COUNT_MIN,
  SYNTH_WORKERS_MAX,
  SYNTH_WORKERS_MIN,
  normalizeSynthSampleCount,
  normalizeSynthWorkers,
} from '../generationLimits'

const ACTIVE_STATUS_LABEL: Partial<Record<JobStatus, string>> = {
  queued: '任务排队中',
  starting: '任务启动中',
  running: '任务运行中',
  evaluating: '任务评估中',
  stopping: '任务停止中',
}

export default function SampleFiller() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('fill')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 720,
    minWidth: 360,
    maxWidth: 920,
    storageKey: 'piern_fill_sidebar_width_v2',
  })

  const {
    data: scenariosCfg,
    isLoading: scLoading,
    mutate: refreshScenarios,
  } = useSWR<Text2CompScenariosConfig>('text2comp-scenarios-v2', () => api.getText2CompScenarios(), {
    revalidateOnMount: true,
  })
  const { data: genCfg } = useSWR<GenerationConfig>('config', () => api.getConfig())
  const { data: templatesStatus, mutate: refreshTemplates } = useSWR<TemplateInfo[]>(
    'templates',
    () => api.getTemplatesStatus(),
    { refreshInterval: 5000 },
  )
  const { seed } = useSeed()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nSamples, setNSamples] = useState(100)
  const [skipExisting, setSkipExisting] = useState(false)
  const [precision, setPrecision] = useState(4)
  const [maxWorkers, setMaxWorkers] = useState(1)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (genCfg?.generation?.n_samples_per_scenario) {
      setNSamples(normalizeSynthSampleCount(genCfg.generation.n_samples_per_scenario))
    }
    if (genCfg?.generation?.max_workers) setMaxWorkers(normalizeSynthWorkers(genCfg.generation.max_workers))
  }, [genCfg])

  useEffect(() => {
    if (isTerminalJobStatus(monitor.status)) {
      refreshTemplates()
      // 刷新场景列表以更新样本状态
      refreshScenarios(undefined, { revalidate: true })
    }
  }, [monitor.status, refreshTemplates, refreshScenarios])

  const allScenarios: Text2CompScenario[] = scenariosCfg ? Object.values(scenariosCfg).flat() : []
  const templateMap: Record<string, TemplateInfo> = {}
  for (const t of templatesStatus ?? []) templateMap[t.scenario] = t

  const scenariosWithTemplates = allScenarios.filter(s => templateMap[s.name])
  const duplicateNames = duplicateText2CompScenarioNames(allScenarios)
  const selectableScenariosWithTemplates = selectableText2CompScenarios(allScenarios, duplicateNames, s =>
    Boolean(templateMap[s.name]),
  )

  const toggle = (name: string) => {
    if (!templateMap[name]) return
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const handleLaunch = async () => {
    if (selected.size === 0) {
      setError('请至少选择一个场景')
      return
    }
    const missing = Array.from(selected).filter(n => !templateMap[n])
    const ambiguousSelected = Array.from(selected).filter(name => duplicateNames.has(name))
    if (missing.length > 0) {
      setError(`以下场景缺少模板：${missing.slice(0, 3).join(', ')}${missing.length > 3 ? '…' : ''}`)
      return
    }
    if (ambiguousSelected.length > 0) {
      setError(
        `以下场景名在多个 simulator 下重复，无法安全填充：${ambiguousSelected.slice(0, 3).join(', ')}${ambiguousSelected.length > 3 ? '…' : ''}`,
      )
      return
    }
    setError(null)
    setLaunching(true)
    try {
      const result = await api.startFillSamples({
        scenarios: Array.from(selected),
        n_samples: normalizeSynthSampleCount(nSamples),
        templates_dir: '',
        output_dir: '',
        output_format: 'parquet',
        compression: 'zstd',
        batch_size: 32768,
        max_workers: normalizeSynthWorkers(maxWorkers),
        skip_existing: skipExisting,
        config: 'configs/text2comp/default.yaml',
        seed,
        precision,
      })
      monitor.start(result.job_id, result.scenario_totals, result.status as JobStatus)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  const canLaunch = isRestartableJobStatus(monitor.status)

  return (
    <div className="workbench-shell">
      {/* ── 左栏：配置（可拖动宽度，内部分区滚动）── */}
      <div className="workbench-sidebar" style={{ width: sidebarWidth }}>
        {/* 顶部：页头（固定）*/}
        <div className="workbench-sidebar-header">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
              <FlaskConical size={14} className="text-emerald-400" />
            </div>
            <h1 className="text-lg font-bold text-white">样本填充</h1>
            <span className="badge bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 text-xs">
              阶段 3
            </span>
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">
            将模板库与 HDF5 数值结合，生成最终训练样本并直接保存为 Parquet。
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
                共{' '}
                {Object.values(templateMap)
                  .reduce((s, t) => s + t.template_count, 0)
                  .toLocaleString()}{' '}
                条
              </span>
            </div>
          </div>
        ) : (
          <div className="flex-shrink-0 border-b border-slate-700/20 px-4 py-2.5 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sparkles size={11} className="text-slate-600" />
              <span className="text-xs text-slate-600">暂无模板，请先完成阶段 2 模板生成</span>
            </div>
            <button
              className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
              onClick={() => navigate('/synth/templates')}
            >
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
              <span className="font-medium text-slate-300 text-sm">选择场景</span>
              {selected.size > 0 && (
                <span className="badge bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 text-xs py-0.5">
                  {selected.size} 已选
                </span>
              )}
            </div>
            <div className="flex items-center gap-0.5">
              <button
                className="btn-ghost py-0.5 px-2 text-sm"
                onClick={() => setSelected(new Set(selectableScenariosWithTemplates.map(s => s.name)))}
              >
                全选
              </button>
              <button className="btn-ghost py-0.5 px-2 text-sm" onClick={() => setSelected(new Set())}>
                清空
              </button>
              <button
                className="btn-ghost py-0.5 px-1.5"
                onClick={() => {
                  refreshScenarios()
                  refreshTemplates()
                }}
              >
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
            {scenariosCfg &&
              Object.entries(scenariosCfg).map(([dirKey, list]) => (
                <div key={dirKey}>
                  <div className="workbench-group-label">{dirKey}</div>
                  <div className="scenario-grid grid gap-1.5">
                    {list.map(s => {
                      const duplicateName = duplicateNames.has(s.name)
                      return (
                        <ScenarioButton
                          key={text2compScenarioKey(s)}
                          s={s}
                          active={selected.has(s.name)}
                          onClick={() => toggle(s.name)}
                          templateCount={templateMap[s.name]?.template_count}
                          disabled={!templateMap[s.name] || duplicateName}
                          disabledReason={
                            duplicateName ? '同名场景存在于多个 simulator，请先改名或清理配置' : undefined
                          }
                          tone="emerald"
                        />
                      )
                    })}
                  </div>
                </div>
              ))}
            {scenariosWithTemplates.length === 0 && !scLoading && (
              <div className="flex flex-col items-center gap-3 py-8 text-center">
                <Layers size={20} className="text-slate-700" />
                <div>
                  <p className="text-slate-500 text-xs font-medium">暂无有模板的场景</p>
                  <p className="text-slate-600 text-sm mt-1">请先在「模板生成」页面生成语言模板</p>
                </div>
                <button
                  className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors"
                  onClick={() => navigate('/synth/templates')}
                >
                  前往模板生成 →
                </button>
              </div>
            )}
          </div>
        </div>

        {/* 底部：参数 + 按钮（固定）*/}
        <div className="flex-shrink-0 border-t border-slate-700/30 bg-slate-900/20">
          <div className="px-4 py-3 space-y-3">
            <div className="flex items-center gap-1.5">
              <Settings size={12} className="text-slate-500" />
              <span className="text-sm font-medium text-slate-400">参数</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="label block mb-1 text-sm">每场景样本数</label>
                <input
                  type="number"
                  className="input w-full text-xs py-1.5 px-3"
                  value={nSamples}
                  min={SYNTH_SAMPLE_COUNT_MIN}
                  max={SYNTH_SAMPLE_COUNT_MAX}
                  onChange={e => setNSamples(normalizeSynthSampleCount(Number(e.target.value)))}
                />
              </div>
              <div>
                <label className="label block mb-1 text-sm">并行场景数</label>
                <input
                  type="number"
                  className="input w-full text-xs py-1.5 px-3"
                  value={maxWorkers}
                  min={SYNTH_WORKERS_MIN}
                  max={SYNTH_WORKERS_MAX}
                  onChange={e => setMaxWorkers(normalizeSynthWorkers(Number(e.target.value)))}
                />
              </div>
              <div>
                <label className="label block mb-1 text-sm">数值精度（小数位）</label>
                <select
                  className="select w-full text-xs py-1.5 px-3"
                  value={precision}
                  onChange={e => setPrecision(Number(e.target.value))}
                >
                  {[1, 2, 3, 4, 5, 6, 8, 10].map(p => (
                    <option key={p} value={p}>
                      {p} 位
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {/* 断点续跑开关 */}
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => setSkipExisting(v => !v)}>
              <div
                className={cn(
                  'relative w-8 h-4 rounded-full transition-all duration-200 flex-shrink-0',
                  skipExisting ? 'bg-emerald-500' : 'bg-slate-700',
                )}
              >
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all duration-200"
                  style={{ left: skipExisting ? '18px' : '2px' }}
                />
              </div>
              <span className="text-sm text-slate-300 font-medium flex-1">跳过已完成</span>
              <span className="text-xs text-slate-600">
                {skipExisting ? '已达目标则跳过，未满则重新生成' : '忽略已有样本重新生成'}
              </span>
            </div>
          </div>

          {error && (
            <div className="mx-4 mb-3 flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2 text-red-300">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          <div className="px-4 pb-4 space-y-1.5">
            {canLaunch ? (
              <>
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
                  {launching ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" /> 启动中…
                    </>
                  ) : (
                    <>
                      <FlaskConical size={14} /> 开始填充{selected.size > 0 ? `（${selected.size} 个场景）` : ''}
                    </>
                  )}
                </button>
                {isTerminalJobStatus(monitor.status) && (
                  <button className="btn-ghost w-full py-1.5 justify-center text-xs" onClick={monitor.reset}>
                    重新配置
                  </button>
                )}
              </>
            ) : (
              <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-2.5">
                <div className="flex items-center gap-2 text-sm font-medium text-emerald-300">
                  <RefreshCw size={13} className="animate-spin" />
                  {ACTIVE_STATUS_LABEL[monitor.status] ?? '任务进行中'}
                </div>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  进度已固定显示在右侧任务面板；如需重新配置，请先终止当前任务。
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <ResizeHandle onMouseDown={onResizeStart} color="emerald" />

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
          onDone={() => navigate('/synth/samples')}
          doneLabel="查看样本"
          jobId={monitor.jobId}
          jobIds={monitor.jobIds}
          stageLabel="样本填充"
          stageColor="text-emerald-400"
          accentColor="emerald"
        />

        {monitor.status === 'idle' && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <FlaskConical size={24} className="text-emerald-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未启动任务</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景并配置参数，点击「开始填充」</p>
            </div>
          </div>
        )}

        {/* File management lives in /synth/files */}
        <div className="card overflow-hidden">
          <div className="card-header justify-between py-3">
            <div className="flex items-center gap-2">
              <FolderOpen size={13} className="text-slate-400" />
              <span className="font-medium text-slate-200 text-base">Sample files</span>
            </div>
            <button className="btn-ghost py-1.5 text-xs" onClick={() => navigate('/synth/files')}>
              打开文件管理
            </button>
          </div>
          <div className="p-4">
            <div className="rounded-2xl border border-slate-700/35 bg-slate-900/30 p-4">
              <div className="font-semibold text-slate-100">Centralized file manager</div>
              <p className="mt-1 text-sm leading-6 text-slate-400">
                Sample delete, clear, and merged-file state now live in the unified file manager.
              </p>
              <button className="btn-ghost mt-3 text-xs text-emerald-300" onClick={() => navigate('/synth/files')}>
                打开统一文件管理
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
