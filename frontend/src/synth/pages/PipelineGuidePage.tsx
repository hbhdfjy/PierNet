import { useMemo } from 'react'
import useSWR from 'swr'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  ArrowRight,
  BadgeCheck,
  Boxes,
  BrainCircuit,
  Check,
  Circle,
  Database,
  FileText,
  GitBranch,
  Loader2,
  MessageSquareText,
  RefreshCw,
  UploadCloud,
} from 'lucide-react'
import { api } from '../../lib/api'
import type {
  DashboardSummary,
  TemplateInfo,
  Text2CompJobSummary,
  Text2CompScenariosConfig,
  TrainingJobSummary,
} from '../../lib/types'
import { cn } from '../../lib/utils'
import { buildPipelineGuide, type PipelineGuideStep, type PipelineMode } from '../pipelineGuide'

const STEP_ICONS = {
  source: UploadCloud,
  register: BadgeCheck,
  template: FileText,
  fill: Database,
  router: GitBranch,
  training: BrainCircuit,
  assembly: Boxes,
  inference: MessageSquareText,
} as const

function StepStateIcon({ step }: { step: PipelineGuideStep }) {
  if (step.status === 'complete') return <Check size={15} />
  if (step.status === 'current') return <Circle size={10} fill="currentColor" />
  return <Circle size={10} />
}

function PipelineRail({ steps }: { steps: PipelineGuideStep[] }) {
  return (
    <div className="overflow-x-auto pb-1">
      <div className="grid min-w-[58rem] grid-cols-8">
        {steps.map((step, index) => (
          <div key={step.id} className="relative flex flex-col items-center gap-2 px-1 text-center">
            {index < steps.length - 1 && (
              <div
                className={cn(
                  'absolute left-1/2 top-4 h-px w-full',
                  steps[index + 1].status === 'complete' ? 'bg-emerald-500/45' : 'bg-slate-700/60',
                )}
              />
            )}
            <div
              className={cn(
                'relative z-10 flex h-8 w-8 items-center justify-center rounded-full border text-xs font-semibold',
                step.status === 'complete' && 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300',
                step.status === 'current' && 'border-sky-400/60 bg-sky-500/15 text-sky-300 ring-4 ring-sky-500/8',
                step.status === 'blocked' && 'border-slate-700 bg-slate-900 text-slate-600',
              )}
            >
              {step.status === 'complete' ? <Check size={14} /> : index + 1}
            </div>
            <span
              className={cn(
                'text-[11px] font-medium',
                step.status === 'complete'
                  ? 'text-emerald-200/85'
                  : step.status === 'current'
                    ? 'text-sky-200'
                    : 'text-slate-600',
              )}
            >
              {step.title}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PipelineStepRow({
  step,
  index,
  onNavigate,
}: {
  step: PipelineGuideStep
  index: number
  onNavigate: (path: string) => void
}) {
  const Icon = STEP_ICONS[step.id]
  return (
    <div
      className={cn(
        'grid gap-4 px-5 py-5 transition-colors lg:grid-cols-[3rem_minmax(12rem,0.75fr)_minmax(16rem,1.4fr)_auto] lg:items-center',
        step.status === 'current' && 'bg-sky-500/[0.045]',
      )}
    >
      <div
        className={cn(
          'flex h-10 w-10 items-center justify-center rounded-lg border',
          step.status === 'complete' && 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300',
          step.status === 'current' && 'border-sky-500/35 bg-sky-500/12 text-sky-300',
          step.status === 'blocked' && 'border-slate-700/60 bg-slate-900/45 text-slate-600',
        )}
      >
        <Icon size={17} />
      </div>

      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-slate-600">{String(index + 1).padStart(2, '0')}</span>
          <h2 className="text-[15px] font-semibold text-slate-100">{step.title}</h2>
        </div>
        <div
          className={cn(
            'mt-1 inline-flex items-center gap-1.5 text-xs font-medium',
            step.status === 'complete'
              ? 'text-emerald-300'
              : step.status === 'current'
                ? 'text-sky-300'
                : 'text-slate-600',
          )}
        >
          <StepStateIcon step={step} />
          {step.status === 'complete' ? '已完成' : step.status === 'current' ? '当前步骤' : '等待前置步骤'}
        </div>
      </div>

      <div className="min-w-0">
        <div className="text-sm leading-6 text-slate-300">{step.completed}</div>
        {step.missing.length > 0 ? (
          <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
            {step.missing.map(item => (
              <span key={item} className="text-xs leading-5 text-amber-300/85">
                缺：{item}
              </span>
            ))}
          </div>
        ) : (
          <div className="mt-1 text-xs text-slate-600">无缺口</div>
        )}
      </div>

      <button
        className={cn(
          'btn-ghost w-full justify-center px-3 py-2 text-xs lg:w-auto',
          step.status === 'current' && 'border-sky-500/30 bg-sky-500/10 text-sky-200 hover:bg-sky-500/15',
        )}
        onClick={() => onNavigate(step.actionPath)}
      >
        {step.actionLabel}
        <ArrowRight size={12} />
      </button>
    </div>
  )
}

export default function PipelineGuidePage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const mode: PipelineMode = searchParams.get('mode') === 'complex' ? 'complex' : 'simple'
  const swrOptions = { revalidateOnFocus: false }

  const {
    data: scenarios,
    isLoading: loadingScenarios,
    mutate: refreshScenarios,
  } = useSWR<Text2CompScenariosConfig>('pipeline-scenarios', api.getText2CompScenarios, swrOptions)
  const {
    data: templates,
    isLoading: loadingTemplates,
    mutate: refreshTemplates,
  } = useSWR<TemplateInfo[]>('pipeline-templates', api.getTemplatesStatus, swrOptions)
  const {
    data: summary,
    isLoading: loadingSummary,
    mutate: refreshSummary,
  } = useSWR<DashboardSummary>('pipeline-dashboard', api.getDashboardSummary, swrOptions)
  const {
    data: trainingJobs,
    isLoading: loadingTraining,
    mutate: refreshTraining,
  } = useSWR<TrainingJobSummary[]>('pipeline-training-jobs', api.getTrainingJobs, swrOptions)
  const {
    data: text2compJobs,
    isLoading: loadingText2Comp,
    mutate: refreshText2Comp,
  } = useSWR<Text2CompJobSummary[]>('pipeline-text2comp-jobs', api.getText2CompJobs, swrOptions)
  const {
    data: assembly,
    isLoading: loadingAssembly,
    mutate: refreshAssembly,
  } = useSWR('pipeline-assembly', api.getAssemblyStatus, swrOptions)

  const loading =
    loadingScenarios || loadingTemplates || loadingSummary || loadingTraining || loadingText2Comp || loadingAssembly

  const steps = useMemo(() => {
    const scenarioList = scenarios ? Object.values(scenarios).flat() : []
    const h5Count = scenarioList.filter(item => item.has_h5).length
    const registeredCount = scenarioList.filter(item => item.has_h5 && item.registered).length
    const templateItems = templates ?? []
    const templateCount = templateItems.reduce((sum, item) => sum + item.template_count, 0)
    const templateScenarioCount = templateItems.filter(item => item.template_count > 0).length
    const assemblyProfiles = (assembly?.assembly_profiles ?? []).filter(item => item.trained && item.chat_enabled)

    return buildPipelineGuide(mode, {
      h5Count,
      registeredCount,
      templateCount,
      templateScenarioCount,
      totalSamples: summary?.stats.total_samples ?? 0,
      routerTotal: summary?.router.total ?? 0,
      trainingJobs: trainingJobs ?? [],
      text2compJobs: text2compJobs ?? [],
      assemblyProfileCount: assemblyProfiles.length,
      assemblyLoaded: Boolean(
        assembly?.loaded_models.assembly_profile?.loaded ||
        (assembly?.loaded_models.llm.loaded &&
          assembly?.loaded_models.router.loaded &&
          assembly?.loaded_models.text2comp.loaded),
      ),
      lastInferenceTestAt: assembly?.last_test_at,
    })
  }, [assembly, mode, scenarios, summary, templates, text2compJobs, trainingJobs])

  const completedCount = steps.filter(step => step.complete).length
  const currentStep = steps.find(step => step.status === 'current')

  const refresh = () => {
    refreshScenarios()
    refreshTemplates()
    refreshSummary()
    refreshTraining()
    refreshText2Comp()
    refreshAssembly()
  }

  return (
    <div className="page-shell">
      <div className="page-content space-y-4 p-4">
        <section className="training-hero training-hero--compact">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="training-eyebrow">
                <span>Pipeline</span>
                <span className="text-slate-600">/</span>
                <span>{mode === 'simple' ? '简洁流程' : '复杂流程'}</span>
              </div>
              <h1 className="mt-2 text-2xl font-bold text-white">全流程向导</h1>
              <p className="training-copy mt-1">
                {currentStep ? `当前推进到「${currentStep.title}」` : '数据、训练、拼装和推理链路均已完成。'}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex rounded-lg border border-slate-700/60 bg-slate-950/35 p-1">
                {(
                  [
                    ['simple', '简洁流程'],
                    ['complex', '复杂流程'],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={value}
                    className={cn(
                      'rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                      mode === value
                        ? 'bg-sky-500/16 text-sky-200 shadow-sm'
                        : 'text-slate-500 hover:bg-slate-800/70 hover:text-slate-300',
                    )}
                    onClick={() => setSearchParams({ mode: value })}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button className="btn-ghost" onClick={refresh}>
                <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
                刷新
              </button>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
            <div>
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="text-slate-500">完成进度</span>
                <span className="font-mono font-semibold text-slate-200">
                  {completedCount}/{steps.length}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-800/80">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-500"
                  style={{ width: `${(completedCount / steps.length) * 100}%` }}
                />
              </div>
            </div>
            <div className="text-right font-mono text-xl font-semibold text-sky-300">
              {Math.round((completedCount / steps.length) * 100)}%
            </div>
          </div>
        </section>

        <section className="training-card overflow-hidden">
          <div className="border-b border-slate-700/40 px-5 py-4">
            {loading && steps.every(step => !step.complete) ? (
              <div className="flex h-16 items-center justify-center gap-2 text-sm text-slate-500">
                <Loader2 size={15} className="animate-spin" />
                正在读取流程状态
              </div>
            ) : (
              <PipelineRail steps={steps} />
            )}
          </div>

          <div className="divide-y divide-slate-800/70">
            {steps.map((step, index) => (
              <PipelineStepRow key={step.id} step={step} index={index} onNavigate={navigate} />
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
