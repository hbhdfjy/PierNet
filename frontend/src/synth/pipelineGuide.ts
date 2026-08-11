import type { Text2CompJobSummary, TrainingJobSummary } from '../lib/types'

export type PipelineMode = 'simple' | 'complex'

export type PipelineStepStatus = 'complete' | 'current' | 'blocked'

export type PipelineGuideStep = {
  id: 'source' | 'register' | 'template' | 'fill' | 'router' | 'training' | 'assembly' | 'inference'
  title: string
  complete: boolean
  status: PipelineStepStatus
  completed: string
  missing: string[]
  actionLabel: string
  actionPath: string
}

export type PipelineGuideInput = {
  h5Count: number
  registeredCount: number
  templateCount: number
  templateScenarioCount: number
  totalSamples: number
  routerTotal: number
  trainingJobs: TrainingJobSummary[]
  text2compJobs: Text2CompJobSummary[]
  assemblyProfileCount: number
  assemblyLoaded: boolean
  lastInferenceTestAt?: number | null
}

type StepDraft = Omit<PipelineGuideStep, 'status'>

const DONE_TRAINING_STATUSES = new Set(['done'])

function finishedSimpleJob(job: TrainingJobSummary): boolean {
  return (
    Boolean(job.config.simple_pipeline_enabled) &&
    DONE_TRAINING_STATUSES.has(job.status) &&
    (!job.router_status || job.router_status === 'done') &&
    (!job.text2comp_status || job.text2comp_status === 'done')
  )
}

function finishedRouterJob(job: TrainingJobSummary): boolean {
  return !job.config.simple_pipeline_enabled && DONE_TRAINING_STATUSES.has(job.status)
}

function finishedText2CompJob(job: Text2CompJobSummary): boolean {
  return DONE_TRAINING_STATUSES.has(job.status)
}

function applyStepStatuses(steps: StepDraft[]): PipelineGuideStep[] {
  const firstIncomplete = steps.findIndex(step => !step.complete)
  return steps.map((step, index) => ({
    ...step,
    status: step.complete ? 'complete' : firstIncomplete === index ? 'current' : 'blocked',
  }))
}

export function buildPipelineGuide(mode: PipelineMode, input: PipelineGuideInput): PipelineGuideStep[] {
  const allH5Registered = input.h5Count > 0 && input.registeredCount >= input.h5Count
  const allRegisteredHaveTemplates =
    input.registeredCount > 0 && input.templateScenarioCount >= input.registeredCount && input.templateCount > 0
  const simpleTrainingDone = input.trainingJobs.some(finishedSimpleJob)
  const routerTrainingDone = input.trainingJobs.some(finishedRouterJob)
  const text2compTrainingDone = input.text2compJobs.some(finishedText2CompJob)
  const trainingDone = mode === 'simple' ? simpleTrainingDone : routerTrainingDone && text2compTrainingDone
  const assemblyDone = input.assemblyProfileCount > 0 || input.assemblyLoaded
  const inferenceDone = Boolean(input.lastInferenceTestAt)
  const routerDataReady = input.routerTotal > 0

  const steps: StepDraft[] = [
    {
      id: 'source',
      title: '上传 / 仿真',
      complete: input.h5Count > 0,
      completed: input.h5Count > 0 ? `已发现 ${input.h5Count} 个 HDF5 场景` : '尚未形成可用物理数据',
      missing: input.h5Count > 0 ? [] : ['需要上传 HDF5，或运行物理仿真生成数据'],
      actionLabel: input.h5Count > 0 ? '查看数据来源' : '准备物理数据',
      actionPath: '/synth/simulate',
    },
    {
      id: 'register',
      title: '注册',
      complete: allH5Registered,
      completed:
        input.registeredCount > 0 ? `已注册 ${input.registeredCount}/${input.h5Count} 个场景` : '尚未注册场景接口',
      missing: allH5Registered
        ? []
        : [
            input.h5Count === 0 ? '需要先准备 HDF5 数据' : '',
            input.h5Count > input.registeredCount ? `还有 ${input.h5Count - input.registeredCount} 个场景待注册` : '',
          ].filter(Boolean),
      actionLabel: allH5Registered ? '查看注册信息' : '注册场景',
      actionPath: '/synth/register',
    },
    {
      id: 'template',
      title: '生成模板',
      complete: allRegisteredHaveTemplates,
      completed:
        input.templateCount > 0
          ? `${input.templateScenarioCount} 个场景，共 ${input.templateCount.toLocaleString()} 条模板`
          : '尚未生成语言模板',
      missing: allRegisteredHaveTemplates
        ? []
        : [
            input.registeredCount === 0 ? '需要先完成场景注册' : '',
            input.registeredCount > input.templateScenarioCount
              ? `还有 ${input.registeredCount - input.templateScenarioCount} 个已注册场景缺少模板`
              : '',
          ].filter(Boolean),
      actionLabel: allRegisteredHaveTemplates ? '查看模板' : '生成模板',
      actionPath: '/synth/templates',
    },
    {
      id: 'fill',
      title: '填充样本',
      complete: input.totalSamples > 0,
      completed:
        input.totalSamples > 0 ? `已生成 ${input.totalSamples.toLocaleString()} 条 Text2Comp 样本` : '尚未填充训练样本',
      missing: input.totalSamples > 0 ? [] : ['需要将语言模板与物理数值填充为训练样本'],
      actionLabel: input.totalSamples > 0 ? '查看样本' : '填充样本',
      actionPath: input.totalSamples > 0 ? '/synth/samples' : '/synth/fill',
    },
    {
      id: 'router',
      title: '构建 Router 数据',
      complete: routerDataReady,
      completed:
        input.routerTotal > 0
          ? `已生成 ${input.routerTotal.toLocaleString()} 条二分类 Router 数据`
          : '尚未构建 Router 训练数据',
      missing: routerDataReady ? [] : ['需要从已填充样本构建二分类 Router 数据'],
      actionLabel: routerDataReady ? '查看 Router 数据' : '构建 Router 数据',
      actionPath: routerDataReady ? '/synth/router-viewer' : '/synth/router',
    },
    {
      id: 'training',
      title: '训练',
      complete: trainingDone,
      completed:
        mode === 'simple'
          ? simpleTrainingDone
            ? '简洁训练已完成 Router 与 Text2Comp'
            : '尚未完成简洁训练'
          : routerTrainingDone && text2compTrainingDone
            ? 'Router 与 Text2Comp 均已有完成任务'
            : [
                routerTrainingDone ? 'Router 已完成' : 'Router 未完成',
                text2compTrainingDone ? 'Text2Comp 已完成' : 'Text2Comp 未完成',
              ].join('，'),
      missing:
        mode === 'simple'
          ? simpleTrainingDone
            ? []
            : ['需要启动一次简洁训练任务']
          : [
              routerTrainingDone ? '' : '缺少已完成的 Router 训练任务',
              text2compTrainingDone ? '' : '缺少已完成的 Text2Comp 训练任务',
            ].filter(Boolean),
      actionLabel: trainingDone ? '查看训练任务' : mode === 'simple' ? '开始简洁训练' : '配置复杂训练',
      actionPath: trainingDone
        ? mode === 'simple'
          ? '/training/simple/tasks'
          : '/training/jobs'
        : mode === 'simple'
          ? '/training/simple'
          : '/training/new',
    },
    {
      id: 'assembly',
      title: '拼装',
      complete: assemblyDone,
      completed: assemblyDone
        ? `${input.assemblyProfileCount} 个可用拼装模型${input.assemblyLoaded ? '，当前已有模型加载' : ''}`
        : '尚未形成可推理的拼装模型',
      missing: assemblyDone ? [] : ['需要选择 LLM、Router、Text2Comp 与专家模型完成拼装'],
      actionLabel: assemblyDone ? '查看拼装模型' : '进入模型拼装',
      actionPath: mode === 'simple' ? '/training/simple/assembly' : '/training/assembly',
    },
    {
      id: 'inference',
      title: '推理测试',
      complete: inferenceDone,
      completed: inferenceDone
        ? `最近一次推理测试：${new Date(Number(input.lastInferenceTestAt) * 1000).toLocaleString('zh-CN')}`
        : '尚未记录成功的推理测试',
      missing: inferenceDone ? [] : ['需要加载拼装模型并完成一次端到端推理'],
      actionLabel: inferenceDone ? '再次测试' : '开始推理测试',
      actionPath: mode === 'simple' ? '/training/simple/assembly' : '/training/assembly',
    },
  ]

  return applyStepStatuses(steps)
}
