import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type {
  Text2CompScenariosConfig,
  Text2CompScenario,
  GenerationConfig,
  TemplateInfo,
  LLMConfig,
  JobStatus,
} from '../../lib/types'
import {
  Cpu,
  Settings,
  Layers,
  RefreshCw,
  AlertCircle,
  Sparkles,
  KeyRound,
  ChevronRight,
  CheckCircle2,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import ScenarioButton from '../components/generation/ScenarioButton'
import JobMonitorPanel from '../components/generation/JobMonitorPanel'
import ResizeHandle from '../components/ui/ResizeHandle'
import { isRestartableJobStatus, isTerminalJobStatus, useJobMonitor } from '../hooks/useJobMonitor'
import { useResizable } from '../hooks/useResizable'
import { normalizeSynthWorkers, SYNTH_WORKERS_MAX, SYNTH_WORKERS_MIN } from '../generationLimits'
import {
  normalizeTemplateCount,
  normalizeTemplateProbability,
  TEMPLATE_COUNT_MAX,
  TEMPLATE_COUNT_MIN,
} from '../templateData'
import {
  duplicateText2CompScenarioNames,
  selectableText2CompScenarios,
  text2compScenarioKey,
} from '../text2compScenario'

export default function TemplateGenerator() {
  const navigate = useNavigate()
  const monitor = useJobMonitor('templates')
  const { width: sidebarWidth, onMouseDown: onResizeStart } = useResizable({
    defaultWidth: 720,
    minWidth: 360,
    maxWidth: 920,
    storageKey: 'PierNet_template_sidebar_width_v2',
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
  const { data: llmCfg } = useSWR<LLMConfig>('llm-config', () => api.getLLMConfig())
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [nTemplates, setNTemplates] = useState(100)
  const [nTemplatesInput, setNTemplatesInput] = useState('100')
  const [genMode, setGenMode] = useState<'overwrite' | 'skip' | 'append'>('append')
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [testingLLM, setTestingLLM] = useState(false)
  const [llmTest, setLLMTest] = useState<{ ok: boolean; message: string; response_preview?: string } | null>(null)
  const [lastLLMTestAt, setLastLLMTestAt] = useState<string | null>(null)

  const [languageMix, setLanguageMix] = useState<number | null>(null)
  const [transformProb, setTransformProb] = useState<number | null>(null)
  const [maxWorkers, setMaxWorkers] = useState<number | null>(null)

  useEffect(() => {
    if (genCfg?.generation?.n_samples_per_scenario) {
      const count = normalizeTemplateCount(genCfg.generation.n_samples_per_scenario)
      setNTemplates(count)
      setNTemplatesInput(String(count))
    }
    if (genCfg?.generation?.language_mix != null) {
      setLanguageMix(normalizeTemplateProbability(genCfg.generation.language_mix, 0.5))
    }
    if (genCfg?.generation?.transform_prob != null) {
      setTransformProb(normalizeTemplateProbability(genCfg.generation.transform_prob, 0.1))
    }
    if (genCfg?.generation?.max_workers != null) setMaxWorkers(normalizeSynthWorkers(genCfg.generation.max_workers))
  }, [genCfg])

  useEffect(() => {
    if (isTerminalJobStatus(monitor.status)) {
      refreshTemplates()
    }
  }, [monitor.status, refreshTemplates])

  const allScenarios: Text2CompScenario[] = scenariosCfg ? Object.values(scenariosCfg).flat() : []
  const duplicateNames = duplicateText2CompScenarioNames(allScenarios)
  const selectableScenariosWithData = selectableText2CompScenarios(allScenarios, duplicateNames, s => s.has_h5)
  const scenariosWithData = allScenarios.filter(s => s.has_h5)
  const unregisteredWithData = scenariosWithData.filter(s => !s.registered)
  const templateMap: Record<string, TemplateInfo> = {}
  for (const t of templatesStatus ?? []) templateMap[t.scenario] = t
  const ambiguousSelected = Array.from(selected).filter(name => duplicateNames.has(name))

  const toggle = (name: string) =>
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const handleTestLLM = async () => {
    if (!llmCfg) return
    setTestingLLM(true)
    setLLMTest(null)
    try {
      const result = await api.testLLMConfig({
        provider: llmCfg.provider,
        model: llmCfg.model,
        api_key: '',
        base_url: llmCfg.base_url,
        temperature: llmCfg.temperature,
        max_tokens: llmCfg.max_tokens,
        thinking: llmCfg.thinking,
      })
      setLLMTest(result)
      setLastLLMTestAt(new Date().toLocaleString('zh-CN'))
    } catch (e: unknown) {
      setLLMTest({ ok: false, message: e instanceof Error ? e.message : 'LLM 测试失败' })
      setLastLLMTestAt(new Date().toLocaleString('zh-CN'))
    } finally {
      setTestingLLM(false)
    }
  }

  const handleLaunch = async () => {
    if (selected.size === 0) {
      setError('请至少选择一个场景')
      return
    }
    if (ambiguousSelected.length > 0) {
      setError(
        `以下场景名在多个 simulator 下重复，无法安全生成：${ambiguousSelected.slice(0, 3).join(', ')}${ambiguousSelected.length > 3 ? '…' : ''}`,
      )
      return
    }
    setError(null)
    setLaunching(true)
    try {
      const result = await api.startGenerateTemplates({
        scenarios: Array.from(selected),
        n_templates: normalizeTemplateCount(nTemplates),
        skip_existing: genMode === 'skip',
        append_existing: genMode === 'append',
        config: 'configs/text2comp/default.yaml',
        ...(languageMix != null ? { language_mix: normalizeTemplateProbability(languageMix, 0.5) } : {}),
        ...(transformProb != null ? { transform_prob: normalizeTemplateProbability(transformProb, 0.1) } : {}),
        ...(maxWorkers != null ? { max_workers: normalizeSynthWorkers(maxWorkers) } : {}),
      })
      monitor.start(result.job_id, result.scenario_totals, result.status as JobStatus)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '启动失败')
    } finally {
      setLaunching(false)
    }
  }

  const totalTemplates = Object.values(templateMap).reduce((s, t) => s + t.template_count, 0)
  const canLaunch = isRestartableJobStatus(monitor.status)
  const llmReady = llmCfg?.has_api_key === true

  return (
    <div className="workbench-shell">
      {/* ── 左栏：配置（可拖动宽度，内部分区滚动）── */}
      <div className="workbench-sidebar" style={{ width: sidebarWidth }}>
        {/* 顶部：页头（固定）*/}
        <div className="workbench-sidebar-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-violet-500/20 border border-violet-500/30 flex items-center justify-center flex-shrink-0">
                <Cpu size={14} className="text-violet-400" />
              </div>
              <h1 className="text-lg font-bold text-white">模板生成</h1>
              <span className="badge bg-violet-500/15 text-violet-300 border border-violet-500/20 text-xs">阶段 2</span>
            </div>
            {totalTemplates > 0 && (
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <Sparkles size={11} className="text-violet-500/70" />
                {totalTemplates.toLocaleString()}
              </div>
            )}
          </div>
          <p className="text-slate-500 text-sm mt-1 ml-9">调用 LLM 生成语言模板，可被阶段 3 反复复用</p>
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
                <button
                  key={label}
                  className="btn-ghost py-0.5 px-2 text-sm"
                  onClick={
                    [
                      () => setSelected(new Set(selectableScenariosWithData.map(s => s.name))),
                      () =>
                        setSelected(
                          new Set(selectableScenariosWithData.filter(s => !templateMap[s.name]).map(s => s.name)),
                        ),
                      () => setSelected(new Set()),
                    ][i]
                  }
                >
                  {label}
                </button>
              ))}
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
                  <button
                    className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 transition-colors"
                    onClick={() => navigate('/synth/register')}
                  >
                    前往注册数据集 <ChevronRight size={11} />
                  </button>
                </div>
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
                          disabled={!s.has_h5 || duplicateName}
                          disabledReason={
                            duplicateName ? '同名场景存在于多个 simulator，请先改名或清理配置' : undefined
                          }
                          tone="violet"
                        />
                      )
                    })}
                  </div>
                </div>
              ))}
            {!scLoading && allScenarios.length === 0 && (
              <div className="flex flex-col items-center gap-3 py-8 text-center">
                <Layers size={20} className="text-slate-700" />
                <div>
                  <p className="text-slate-500 text-xs font-medium">未找到任何场景</p>
                  <p className="text-slate-600 text-sm mt-1">请先配置数据目录或运行阶段 1 物理仿真</p>
                </div>
                <button
                  className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 transition-colors"
                  onClick={() => navigate('/synth/simulate')}
                >
                  前往物理仿真 <ChevronRight size={11} />
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
                min={TEMPLATE_COUNT_MIN}
                max={TEMPLATE_COUNT_MAX}
                onChange={e => {
                  setNTemplatesInput(e.target.value)
                  const n = Number(e.target.value)
                  if (!isNaN(n)) setNTemplates(normalizeTemplateCount(n))
                }}
                onBlur={() => {
                  const n = Number(nTemplatesInput)
                  const clamped = normalizeTemplateCount(n)
                  setNTemplates(clamped)
                  setNTemplatesInput(String(clamped))
                }}
              />
            </div>

            {/* 生成模式三选一 */}
            <div>
              <label className="label block mb-1.5 text-sm">已有模板时的处理方式</label>
              <div className="grid grid-cols-3 gap-1.5">
                {(
                  [
                    { value: 'append', label: '继续生成', desc: '追加到现有数量', color: 'emerald' },
                    { value: 'skip', label: '跳过场景', desc: '已有则不处理', color: 'slate' },
                    { value: 'overwrite', label: '重新生成', desc: '清空后重新写入', color: 'red' },
                  ] as const
                ).map(({ value, label, desc, color }) => (
                  <button
                    key={value}
                    onClick={() => setGenMode(value)}
                    className={cn(
                      'flex flex-col items-start px-2.5 py-2 rounded-xl border text-left transition-all',
                      genMode === value
                        ? color === 'emerald'
                          ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                          : color === 'red'
                            ? 'bg-red-500/15 border-red-500/40 text-red-300'
                            : 'bg-slate-600/30 border-slate-500/40 text-slate-200'
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
                  value: normalizeSynthWorkers(maxWorkers ?? genCfg?.generation?.max_workers ?? SYNTH_WORKERS_MIN),
                  min: SYNTH_WORKERS_MIN,
                  max: SYNTH_WORKERS_MAX,
                  display: String(
                    normalizeSynthWorkers(maxWorkers ?? genCfg?.generation?.max_workers ?? SYNTH_WORKERS_MIN),
                  ),
                  onChange: (v: number) => setMaxWorkers(normalizeSynthWorkers(v)),
                },
                {
                  label: '中文',
                  value: Math.round(
                    normalizeTemplateProbability(languageMix ?? genCfg?.generation?.language_mix ?? 0.5, 0.5) * 100,
                  ),
                  min: 0,
                  max: 100,
                  display: `${Math.round(
                    normalizeTemplateProbability(languageMix ?? genCfg?.generation?.language_mix ?? 0.5, 0.5) * 100,
                  )}%`,
                  onChange: (v: number) => setLanguageMix(normalizeTemplateProbability(v / 100, 0.5)),
                },
                {
                  label: '变换',
                  value: Math.round(
                    normalizeTemplateProbability(transformProb ?? genCfg?.generation?.transform_prob ?? 0.1, 0.1) * 100,
                  ),
                  min: 0,
                  max: 100,
                  display: `${Math.round(
                    normalizeTemplateProbability(transformProb ?? genCfg?.generation?.transform_prob ?? 0.1, 0.1) * 100,
                  )}%`,
                  onChange: (v: number) => setTransformProb(normalizeTemplateProbability(v / 100, 0.1)),
                },
              ].map(({ label, value, min, max, display, onChange }) => (
                <div key={label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="label text-xs">{label}</span>
                    <span className="text-xs text-slate-500 tabular-nums">{display}</span>
                  </div>
                  <input
                    type="range"
                    className="w-full accent-violet-500 h-1"
                    min={min}
                    max={max}
                    value={value}
                    onChange={e => onChange(Number(e.target.value))}
                  />
                </div>
              ))}
            </div>

            {/* LLM 状态与内嵌测试 */}
            <div
              className={cn(
                'rounded-xl border p-3',
                llmReady ? 'border-emerald-500/20 bg-emerald-500/8' : 'border-amber-500/20 bg-amber-500/8',
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    {llmReady ? (
                      <CheckCircle2 size={13} className="text-emerald-300" />
                    ) : (
                      <AlertCircle size={13} className="text-amber-300" />
                    )}
                    <span className="text-xs font-semibold text-slate-100">
                      {llmReady ? 'LLM 连接信息已保存' : 'LLM Key 未配置'}
                    </span>
                  </div>
                  <div className="mt-1 truncate text-xs text-slate-500">
                    {llmCfg ? `${llmCfg.provider} · ${llmCfg.model || '未设置模型'}` : '正在读取配置'}
                  </div>
                  {lastLLMTestAt && <div className="mt-1 text-[11px] text-slate-600">最近测试：{lastLLMTestAt}</div>}
                </div>
                <button className="btn-ghost flex-shrink-0 py-1 text-xs" onClick={() => navigate('/synth/llm-config')}>
                  配置
                </button>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  className="btn-ghost flex-1 justify-center py-1.5 text-xs"
                  onClick={handleTestLLM}
                  disabled={!llmReady || testingLLM}
                >
                  {testingLLM ? <RefreshCw size={12} className="animate-spin" /> : <KeyRound size={12} />}
                  测试连接
                </button>
              </div>
              {llmTest && (
                <div
                  className={cn(
                    'mt-2 rounded-lg border px-2.5 py-2 text-xs leading-5',
                    llmTest.ok
                      ? 'border-emerald-500/20 bg-emerald-500/8 text-emerald-200'
                      : 'border-red-500/20 bg-red-500/8 text-red-200',
                  )}
                >
                  <div>{llmTest.message}</div>
                  {llmTest.response_preview && (
                    <div className="mt-1 truncate font-mono opacity-75">{llmTest.response_preview}</div>
                  )}
                </div>
              )}
            </div>
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
                style={{
                  background:
                    selected.size > 0 && !launching && llmReady
                      ? 'linear-gradient(135deg, #7c3aed, #6d28d9)'
                      : undefined,
                }}
                onClick={handleLaunch}
                disabled={launching || selected.size === 0 || !llmReady}
              >
                {launching ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" /> 启动中…
                  </>
                ) : !llmReady ? (
                  <>
                    <AlertCircle size={14} /> 请先配置 LLM API Key
                  </>
                ) : (
                  <>
                    <Cpu size={14} /> 开始生成{selected.size > 0 ? `（${selected.size} 个场景）` : ''}
                  </>
                )}
              </button>
              {isTerminalJobStatus(monitor.status) && (
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
          onDone={() => navigate('/synth/fill')}
          doneLabel="去填充样本"
          jobId={monitor.jobId}
          jobIds={monitor.jobIds}
          stageLabel="模板生成"
          stageColor="text-violet-400"
          accentColor="violet"
        />

        {monitor.status === 'idle' && (
          <div className="workbench-idle-panel text-center">
            <div className="w-14 h-14 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
              <Cpu size={24} className="text-violet-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-base font-medium">尚未启动任务</p>
              <p className="text-slate-600 text-sm mt-1">在左侧选择场景并配置参数，点击「开始生成」</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
