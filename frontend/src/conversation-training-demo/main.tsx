import { StrictMode, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Clock3,
  Database,
  ExternalLink,
  FileCheck2,
  FileUp,
  FlaskConical,
  Layers3,
  LoaderCircle,
  MessageSquare,
  MessageSquarePlus,
  Paperclip,
  Play,
  RefreshCw,
  Send,
  Upload,
  User,
  X,
  Zap,
} from 'lucide-react'
import {
  type AssemblyProfile,
  type SimulationPreset,
  type TrainingDataset,
  type TrainingJob,
  type WorkflowSnapshot,
  wait,
  waitForWorkflow,
  workflowApi,
} from './workflowApi'
import { goalSimulatorMismatch, recommendedSimulationKey } from './goalRouting'
import './styles.css'

type Phase = 'goal' | 'data' | 'preparing' | 'ready' | 'training' | 'complete' | 'error'
type DataMode = 'existing' | 'simulation' | 'upload'
type AssemblyState = 'idle' | 'loading' | 'ready' | 'error'
type ChatItem = { id: number; role: 'assistant' | 'user'; content: string }
const DEMO_MODE = false

const initialMessages: ChatItem[] = [
  {
    id: 1,
    role: 'assistant',
    content: '您好，我会带您完成从数据准备到模型训练和产物登记的完整流程。请先告诉我，您希望模型解决什么问题？',
  },
]

const exampleGoals = [
  '训练一个地下水水位预测模型',
  '用我的 HDF5 数据训练一个完整预测模型',
  '训练一个能够识别并求解科学计算任务的模型',
]

const terminalStatuses = new Set(['done', 'error', 'terminated', 'external_terminated'])

function percent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1)}%`
}

function metric(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}

function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return '估算中'
  if (seconds < 60) return `约 ${Math.ceil(seconds)} 秒`
  if (seconds < 3600) return `约 ${Math.ceil(seconds / 60)} 分钟`
  return `约 ${(seconds / 3600).toFixed(1)} 小时`
}

function jobProgress(job: TrainingJob | null): number {
  if (!job) return 0
  if (job.status === 'done') return 100
  if (job.status === 'error' || job.status === 'terminated' || job.status === 'external_terminated') return 100
  if (job.status === 'queued') return 5
  if (job.status === 'starting') return job.pipeline_stage === 'text2comp' ? 64 : 14
  if (job.status === 'evaluating') return job.pipeline_stage === 'text2comp' ? 92 : 57
  if (job.pipeline_stage === 'text2comp') {
    const epochs = Number(job.config.simple_text2comp_epochs || 0)
    const current = Number(job.latest_epoch || 0)
    return Math.min(91, 65 + (epochs > 0 ? (current / epochs) * 25 : 12))
  }
  const epochs = Number(job.config.epochs || 0)
  const current = Number(job.latest_epoch || 0)
  return Math.min(60, 18 + (epochs > 0 ? (current / epochs) * 40 : 20))
}

export function App() {
  const [phase, setPhase] = useState<Phase>('goal')
  const [goal, setGoal] = useState('')
  const [messages, setMessages] = useState<ChatItem[]>(initialMessages)
  const [input, setInput] = useState('')
  const [dataOpen, setDataOpen] = useState(false)
  const [dataMode, setDataMode] = useState<DataMode>('simulation')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [datasets, setDatasets] = useState<TrainingDataset[]>([])
  const [simulations, setSimulations] = useState<SimulationPreset[]>([])
  const [simulationKey, setSimulationKey] = useState('modflow/unified_aquifer')
  const [selectedDataset, setSelectedDataset] = useState<TrainingDataset | null>(null)
  const [workflow, setWorkflow] = useState<WorkflowSnapshot | null>(null)
  const [preparingMessage, setPreparingMessage] = useState('正在建立数据工作流')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [completionAnnounced, setCompletionAnnounced] = useState(false)
  const [assemblyState, setAssemblyState] = useState<AssemblyState>('idle')
  const [assemblyProfile, setAssemblyProfile] = useState<AssemblyProfile | null>(null)
  const [chatBusy, setChatBusy] = useState(false)
  const messagesRef = useRef<HTMLDivElement>(null)
  const nextId = useRef(2)
  const assemblyStarted = useRef(false)

  const addMessage = (role: ChatItem['role'], content: string) => {
    setMessages(current => [...current, { id: nextId.current++, role, content }])
  }

  useEffect(() => {
    const container = messagesRef.current
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages, phase, workflow, job])

  useEffect(() => {
    void Promise.all([workflowApi.simpleDatasets(), workflowApi.presets()])
      .then(([readyDatasets, presets]) => {
        setDatasets(readyDatasets)
        setSimulations(presets.simulations.filter(item => item.has_data))
      })
      .catch(() => undefined)
    const savedJobId = window.localStorage.getItem('piern-conversation-training-job')
    if (DEMO_MODE) {
      window.localStorage.removeItem('piern-conversation-training-job')
      return
    }
    if (!savedJobId) return
    void workflowApi
      .trainingJob(savedJobId)
      .then(savedJob => {
        setJob(savedJob)
        setPhase(savedJob.status === 'done' ? 'complete' : terminalStatuses.has(savedJob.status) ? 'error' : 'training')
        setGoal(savedJob.name)
      })
      .catch(() => window.localStorage.removeItem('piern-conversation-training-job'))
  }, [])

  useEffect(() => {
    const recommendedKey = recommendedSimulationKey(goal, simulations)
    if (recommendedKey) setSimulationKey(recommendedKey)
  }, [goal, simulations])

  useEffect(() => {
    if (DEMO_MODE) return
    if (phase !== 'training' || !job?.job_id) return
    let cancelled = false
    const update = async () => {
      try {
        const current = await workflowApi.trainingJob(job.job_id)
        if (cancelled) return
        setJob(current)
        if (current.status === 'done') {
          setPhase('complete')
        } else if (terminalStatuses.has(current.status)) {
          setError(current.text2comp_error_message || current.error_message || '训练任务未正常完成')
          setPhase('error')
        }
      } catch (requestError) {
        if (!cancelled) setError(requestError instanceof Error ? requestError.message : '无法获取训练状态')
      }
    }
    void update()
    const timer = window.setInterval(() => void update(), 2500)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [phase, job?.job_id])

  useEffect(() => {
    if (phase !== 'complete' || !job || completionAnnounced) return
    addMessage(
      'assistant',
      DEMO_MODE
        ? '流程演示已经完成。Router、Text2Comp 和模型登记步骤均已展示，本次演示没有提交真实训练任务。'
        : `真实训练已经完成。Router 和 Text2Comp 产物已写入任务目录，任务编号为 ${job.job_id}。接下来将自动完成拼装注册和加载。`,
    )
    setCompletionAnnounced(true)
  }, [completionAnnounced, job, phase])

  useEffect(() => {
    if (DEMO_MODE || phase !== 'complete' || job?.status !== 'done' || assemblyStarted.current) return
    assemblyStarted.current = true
    setAssemblyState('loading')
    addMessage('assistant', '训练产物已生成，正在注册完整拼装模型并自动加载到可用 GPU。')
    void workflowApi
      .registerAndLoad(job.job_id)
      .then(result => {
        setAssemblyProfile(result.profile)
        setAssemblyState('ready')
        addMessage(
          'assistant',
          `模型“${result.profile.name}”已加入“已注册拼装模型”并加载完成。现在可以直接在这里继续对话。`,
        )
      })
      .catch(reason => {
        const message = reason instanceof Error ? reason.message : '自动注册并加载模型失败'
        setAssemblyState('error')
        setError(message)
        addMessage('assistant', `训练已完成，但自动拼装加载失败：${message}`)
      })
  }, [job?.job_id, job?.status, phase])

  const steps = useMemo(() => {
    const rank: Record<Phase, number> = { goal: 0, data: 1, preparing: 1, ready: 2, training: 3, complete: 4, error: 3 }
    const currentRank = rank[phase]
    const rows = [
      { label: '训练目标', detail: goal || '正在了解您的需求' },
      {
        label: '准备数据',
        detail:
          phase === 'preparing'
            ? preparingMessage
            : selectedDataset
              ? `${selectedDataset.total_count.toLocaleString()} 条 Router 样本`
              : '等待选择数据',
      },
      {
        label: '确认方案',
        detail: phase === 'ready' ? '等待您的确认' : currentRank > 2 ? '已确认真实训练' : '等待开始',
      },
      {
        label: '训练与评估',
        detail:
          phase === 'training'
            ? `${Math.round(jobProgress(job))}% · ${job?.pipeline_stage === 'text2comp' ? 'Text2Comp' : 'Router'}`
            : phase === 'complete'
              ? '真实任务已完成'
              : phase === 'error'
                ? '需要处理'
                : '等待开始',
      },
      {
        label: '交付模型',
        detail:
          assemblyState === 'ready'
            ? '已注册、加载，可对话'
            : assemblyState === 'loading'
              ? '正在注册并加载'
              : phase === 'complete'
                ? '训练产物已生成'
                : '等待开始',
      },
    ]
    return rows.map((row, index) => ({ ...row, current: index === currentRank, done: index < currentRank }))
  }, [assemblyState, goal, job, phase, preparingMessage, selectedDataset])

  function submitGoal(value = input) {
    const normalized = value.trim()
    if (!normalized) return
    setGoal(normalized)
    const recommendedKey = recommendedSimulationKey(normalized, simulations)
    if (recommendedKey) setSimulationKey(recommendedKey)
    addMessage('user', normalized)
    setInput('')
    window.setTimeout(() => {
      addMessage(
        'assistant',
        '需求已记录。请选择已有训练数据、复用平台内置仿真，或上传符合 Piern 规范的 HDF5 文件。数据准备完成后，我会给出真实训练方案供您确认。',
      )
      setPhase('data')
      setDataOpen(true)
    }, 250)
  }

  async function refreshReadyDataset(datasetId: string): Promise<TrainingDataset> {
    for (let attempt = 0; attempt < 12; attempt += 1) {
      const current = await workflowApi.simpleDatasets()
      setDatasets(current)
      const match = current.find(item => item.dataset_id === datasetId)
      if (match) return match
      await wait(500)
    }
    throw new Error('训练数据已生成，但尚未进入简洁训练列表，请稍后刷新重试')
  }

  async function prepareWorkflow(source: (workflowId: string) => Promise<WorkflowSnapshot>, sourceLabel: string) {
    setBusy(true)
    setError(null)
    setDataOpen(false)
    setPhase('preparing')
    addMessage('user', sourceLabel)
    try {
      setPreparingMessage('正在建立数据工作流')
      await workflowApi.createSession()
      const created = await workflowApi.createWorkflow(goal.slice(0, 100) || '对话训练任务')
      setWorkflow(created)
      setPreparingMessage('正在校验并规范化源数据')
      let snapshot = await source(created.workflow_id)
      setWorkflow(snapshot)
      if (snapshot.status === 'running') snapshot = await waitForWorkflow(created.workflow_id, setWorkflow)
      if (!snapshot.source?.ready || !snapshot.definition) throw new Error('源数据校验后没有生成可用的数据定义')
      const mismatch = goalSimulatorMismatch(goal, snapshot.definition.simulator || snapshot.source.simulator || '')
      if (mismatch) throw new Error(mismatch)
      const sourceCount = Number(snapshot.source.sample_count || 0)
      if (sourceCount < 13) throw new Error('至少需要 13 条源样本，才能生成满足完整训练要求的数据集')
      setPreparingMessage('正在确认输入输出定义')
      snapshot = await workflowApi.saveDefinition(created.workflow_id, {
        ...snapshot.definition,
        task_description: goal,
      })
      setWorkflow(snapshot)
      const maxSamples = Math.max(4, Math.min(100, sourceCount))
      const variants = Math.min(8, Math.max(1, Math.ceil(100 / maxSamples)))
      setPreparingMessage('正在生成 Router 与 Text2Comp 训练数据')
      await workflowApi.generate(created.workflow_id, maxSamples, variants)
      snapshot = await waitForWorkflow(created.workflow_id, update => {
        setWorkflow(update)
        setPreparingMessage(update.artifacts?.message || '正在生成训练数据')
      })
      const routerId = snapshot.artifacts?.router?.dataset_id
      if (!routerId) throw new Error('数据生成完成，但没有找到 Router 数据集编号')
      const dataset = await refreshReadyDataset(routerId)
      setSelectedDataset(dataset)
      setPhase('ready')
      addMessage(
        'assistant',
        `数据准备完成：Router ${snapshot.artifacts?.router?.sample_count || dataset.total_count} 条，Text2Comp ${snapshot.artifacts?.text2comp?.sample_count || '—'} 条。两类数据已经配对并登记，可以创建真实完整训练任务。`,
      )
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : '数据准备失败'
      setError(message)
      setPhase('error')
      addMessage('assistant', `数据准备没有完成：${message}`)
    } finally {
      setBusy(false)
    }
  }

  function chooseExisting(dataset: TrainingDataset) {
    const mismatch = goalSimulatorMismatch(goal, dataset.simulator)
    if (mismatch) {
      setError(mismatch)
      addMessage('assistant', mismatch)
      return
    }
    setSelectedDataset(dataset)
    setDataOpen(false)
    setPhase('ready')
    addMessage('user', `使用已有数据：${dataset.display_name || dataset.simulator}`)
    addMessage(
      'assistant',
      `已选择 ${dataset.total_count.toLocaleString()} 条 Router 样本，并确认存在配对的 Text2Comp 数据。可以创建真实训练任务。`,
    )
  }

  function useSelectedSimulation() {
    const preset = simulations.find(item => `${item.simulator}/${item.scenario}` === simulationKey)
    if (!preset) {
      setError('请选择一个可用的内置仿真场景')
      return
    }
    const mismatch = goalSimulatorMismatch(goal, preset.simulator)
    if (mismatch) {
      setError(mismatch)
      addMessage('assistant', mismatch)
      return
    }
    void prepareWorkflow(
      workflowId => workflowApi.useSimulation(workflowId, preset),
      `复用内置数据：${preset.simulator} / ${preset.scenario}`,
    )
  }

  function uploadSelectedFile() {
    if (!selectedFile) return
    void prepareWorkflow(
      workflowId => workflowApi.uploadSource(workflowId, selectedFile),
      `上传数据：${selectedFile.name}`,
    )
  }

  async function beginTraining() {
    if (!selectedDataset) return
    const mismatch = goalSimulatorMismatch(goal, selectedDataset.simulator)
    if (mismatch) {
      setError(mismatch)
      addMessage('assistant', mismatch)
      return
    }
    if (DEMO_MODE) {
      const demoJob: TrainingJob = {
        job_id: 'demo-training-workflow',
        name: goal || '完整模型训练演示',
        status: 'running',
        simulator: selectedDataset.simulator,
        scenarios: selectedDataset.scenarios.map(item => item.scenario),
        gpu_id: 1,
        latest_epoch: 1,
        eta_seconds: 45,
        pipeline_stage: 'router',
        router_status: 'running',
        text2comp_status: null,
        config: { epochs: 5, simple_text2comp_epochs: 10 },
      }
      setJob(demoJob)
      setPhase('training')
      addMessage('user', '确认并开始训练流程')
      addMessage('assistant', '训练流程演示已启动，将依次展示 Router、Text2Comp 和模型登记步骤。')
      window.setTimeout(() => {
        setJob(current =>
          current
            ? {
                ...current,
                latest_epoch: 4,
                eta_seconds: 20,
                pipeline_stage: 'text2comp',
                router_status: 'done',
                text2comp_status: 'running',
                router_metrics: { accuracy: 0.98, precision: 0.98, recall: 0.98, f1: 0.98 },
              }
            : current,
        )
      }, 1200)
      window.setTimeout(() => {
        setJob(current =>
          current
            ? {
                ...current,
                status: 'done',
                latest_epoch: 10,
                eta_seconds: 0,
                pipeline_stage: 'done',
                router_status: 'done',
                text2comp_status: 'done',
                router_metrics: { accuracy: 0.98, precision: 0.98, recall: 0.98, f1: 0.98 },
                text2comp_metrics: { normalized_rmse: 0.12, r2: 0.91 },
              }
            : current,
        )
        setPhase('complete')
      }, 2800)
      return
    }
    setBusy(true)
    setError(null)
    try {
      const created = await workflowApi.createQuickJob(selectedDataset)
      setJob(created)
      window.localStorage.setItem('piern-conversation-training-job', created.job_id)
      setPhase('training')
      addMessage('user', '确认，提交真实训练任务')
      addMessage(
        'assistant',
        `任务 ${created.job_id} 已进入平台训练队列，将依次训练 Router 和 Text2Comp。系统自动分配 GPU ${created.gpu_id}，页面会持续读取真实进度。`,
      )
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : '启动训练失败'
      setError(message)
      setPhase('error')
      addMessage('assistant', `训练任务创建失败：${message}`)
    } finally {
      setBusy(false)
    }
  }

  function resetAfterError() {
    setError(null)
    if (selectedDataset) setPhase('ready')
    else {
      setPhase('data')
      setDataOpen(true)
    }
  }

  async function sendConversationMessage() {
    const content = input.trim()
    if (!content || chatBusy) return
    if (phase !== 'complete' || assemblyState !== 'ready' || !assemblyProfile) {
      addMessage('assistant', '模型尚未完成注册和加载，请等待当前工作流完成。')
      return
    }
    addMessage('user', content)
    setInput('')
    setChatBusy(true)
    setError(null)
    try {
      const result = await workflowApi.testAssembly(assemblyProfile.model_id, content)
      const answer = result.final_answer || result.first_cot_result || '模型没有返回可显示的内容。'
      const route = result.router_class_name || result.router_prediction || 'unknown'
      addMessage('assistant', `${answer}\n\nRouter：${route} · ${result.latency_ms.toFixed(0)} ms`)
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : '模型对话失败'
      setError(message)
      setInput(content)
      addMessage('assistant', `对话失败：${message}`)
    } finally {
      setChatBusy(false)
    }
  }

  function submitComposer() {
    if (phase === 'goal') {
      submitGoal()
      return
    }
    if (phase === 'complete') {
      void sendConversationMessage()
      return
    }
    if (!input.trim()) return
    addMessage('user', input.trim())
    setInput('')
    window.setTimeout(() => addMessage('assistant', '补充要求已记录。请继续使用当前操作卡推进真实工作流。'), 200)
  }

  function startNewConversation() {
    window.localStorage.removeItem('piern-conversation-training-job')
    nextId.current = 2
    setPhase('goal')
    setGoal('')
    setMessages(initialMessages)
    setInput('')
    setDataOpen(false)
    setDataMode('simulation')
    setSelectedFile(null)
    setSimulationKey('modflow/unified_aquifer')
    setSelectedDataset(null)
    setWorkflow(null)
    setPreparingMessage('正在建立数据工作流')
    setBusy(false)
    setError(null)
    setAdvancedOpen(false)
    setJob(null)
    setCompletionAnnounced(false)
    setAssemblyState('idle')
    setAssemblyProfile(null)
    setChatBusy(false)
    assemblyStarted.current = false
  }

  const progress = jobProgress(job)
  const routerMetrics = job?.router_metrics || job?.latest_metrics
  const text2compNeedsTuning = Number(job?.text2comp_metrics?.normalized_rmse ?? 0) > 0.25

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-group">
          <div className="brand-mark">P</div>
          <div>
            <div className="brand-title">Piern 对话训练</div>
            <div className="brand-subtitle">完整模型工作流</div>
          </div>
        </div>
        <div className="topbar-actions">
          <button
            className="new-conversation-button"
            type="button"
            onClick={startNewConversation}
            disabled={busy}
            title="新建训练对话"
          >
            <MessageSquarePlus size={16} />
            <span>新建对话</span>
          </button>
          <div className="demo-badge real-badge">
            <Zap size={14} />
            正式环境 · 真实训练
          </div>
        </div>
      </header>

      <main className="workspace">
        <aside className="workflow-panel" aria-label="训练流程">
          <div className="panel-heading">本次训练</div>
          <div className="workflow-list">
            {steps.map((step, index) => (
              <div
                className={`workflow-step ${step.current ? 'current' : ''} ${step.done ? 'done' : ''}`}
                key={step.label}
              >
                <div className="step-rail">
                  <div className="step-icon">
                    {step.done ? (
                      <Check size={14} />
                    ) : step.current ? (
                      <LoaderCircle size={14} className={phase === 'training' || phase === 'preparing' ? 'spin' : ''} />
                    ) : (
                      <Circle size={10} />
                    )}
                  </div>
                  {index < steps.length - 1 && <div className="step-line" />}
                </div>
                <div className="step-copy">
                  <div className="step-label">{step.label}</div>
                  <div className="step-detail">{step.detail}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="session-note">
            <MessageSquare size={16} />
            <span>真实任务编号会保存在当前浏览器</span>
          </div>
        </aside>

        <section className="conversation-panel">
          <div className="conversation-head">
            <div>
              <h1>创建完整模型</h1>
              <p>{job ? `任务 ${job.job_id}` : workflow ? `数据工作流 ${workflow.workflow_id}` : '尚未创建任务'}</p>
            </div>
            <span className={`status-pill status-${phase}`}>
              {phase === 'complete'
                ? assemblyState === 'loading'
                  ? '正在加载'
                  : assemblyState === 'ready'
                    ? '可对话'
                    : '已完成'
                : phase === 'training'
                  ? '训练中'
                  : phase === 'preparing'
                    ? '准备数据'
                    : phase === 'error'
                      ? '需要处理'
                      : '进行中'}
            </span>
          </div>

          <div className="messages" aria-live="polite" ref={messagesRef}>
            {messages.map(message => (
              <div className={`message-row ${message.role}`} key={message.id}>
                <div className="avatar">{message.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}</div>
                <div className="message-body">
                  <div className="message-author">{message.role === 'assistant' ? '训练助手' : '您'}</div>
                  <div className="bubble">{message.content}</div>
                </div>
              </div>
            ))}

            {phase === 'goal' && (
              <div className="action-block intro-actions">
                <div className="action-label">您也可以从示例开始</div>
                <div className="example-list">
                  {exampleGoals.map(item => (
                    <button className="example-button" key={item} onClick={() => submitGoal(item)}>
                      <span>{item}</span>
                      <ArrowRight size={16} />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {phase === 'data' && (
              <div className="action-block requirement-card">
                <div className="action-icon data">
                  <Database size={20} />
                </div>
                <div className="action-main">
                  <div className="action-title">选择真实训练数据</div>
                  <div className="action-description">
                    可以复用平台数据，也可以上传 HDF5 后自动生成 Router 和 Text2Comp 数据。
                  </div>
                  <div className="button-row">
                    <button className="primary-button" onClick={() => setDataOpen(true)}>
                      <Database size={16} />
                      选择数据来源
                    </button>
                  </div>
                </div>
              </div>
            )}

            {phase === 'preparing' && (
              <div className="action-block prep-card">
                <div className="training-topline">
                  <div>
                    <span className="live-dot" />
                    {preparingMessage}
                  </div>
                  <strong>
                    {Math.round(((workflow?.artifacts?.progress ?? workflow?.source?.progress ?? 0.1) as number) * 100)}
                    %
                  </strong>
                </div>
                <div className="progress-track">
                  <div
                    style={{
                      width: `${Math.max(8, (workflow?.artifacts?.progress ?? workflow?.source?.progress ?? 0.1) * 100)}%`,
                    }}
                  />
                </div>
                <p>正在生成并登记真实训练数据，请勿关闭当前页面。</p>
              </div>
            )}

            {phase === 'ready' && selectedDataset && (
              <div className="action-block plan-card">
                <div className="plan-header">
                  <div>
                    <div className="eyebrow">
                      <CheckCircle2 size={15} />
                      真实数据已就绪
                    </div>
                    <div className="action-title">
                      {selectedDataset.display_name || selectedDataset.simulator} · 完整训练
                    </div>
                  </div>
                  <span className="estimate">GPU 自动分配</span>
                </div>
                <div className="summary-grid">
                  <div>
                    <span>Router 样本</span>
                    <strong>{selectedDataset.total_count.toLocaleString()} 条</strong>
                  </div>
                  <div>
                    <span>训练场景</span>
                    <strong>{selectedDataset.scenarios.length} 个</strong>
                  </div>
                  <div>
                    <span>训练组件</span>
                    <strong>Router + Text2Comp</strong>
                  </div>
                </div>
                <button
                  className="advanced-toggle"
                  onClick={() => setAdvancedOpen(open => !open)}
                  aria-expanded={advancedOpen}
                >
                  查看真实执行内容
                  <ChevronDown size={16} className={advancedOpen ? 'rotate' : ''} />
                </button>
                {advancedOpen && (
                  <div className="stage-preview">
                    {[
                      '检查并缓存 Router 输入',
                      '训练和评估 Router',
                      '训练和评估 Text2Comp',
                      '保存并登记两个模型产物',
                    ].map((item, index) => (
                      <div key={item}>
                        <span>{index + 1}</span>
                        <div>
                          <strong>{item}</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="plan-footer">
                  <span>点击后会创建真实任务并自动分配可用 GPU</span>
                  <button className="primary-button" disabled={busy} onClick={() => void beginTraining()}>
                    {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
                    {busy ? '提交中' : '确认并开始训练'}
                  </button>
                </div>
              </div>
            )}

            {phase === 'training' && job && (
              <div className="action-block training-card">
                <div className="training-topline">
                  <div>
                    <span className="live-dot" />
                    {job.pipeline_stage === 'text2comp'
                      ? '正在训练 Text2Comp'
                      : job.status === 'queued'
                        ? '正在等待训练资源'
                        : '正在训练 Router'}
                  </div>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div className="progress-track">
                  <div style={{ width: `${progress}%` }} />
                </div>
                <div className="runtime-grid">
                  <div>
                    <span>GPU</span>
                    <strong>{job.gpu_id}</strong>
                  </div>
                  <div>
                    <span>当前轮次</span>
                    <strong>{Number(job.latest_epoch ?? 0) + 1}</strong>
                  </div>
                  <div>
                    <span>预计剩余</span>
                    <strong>{formatEta(job.eta_seconds)}</strong>
                  </div>
                </div>
                <div className="training-stages">
                  {['Router 训练与评估', 'Text2Comp 训练与评估', '登记模型产物'].map((label, index) => {
                    const activeIndex = job.pipeline_stage === 'text2comp' ? 1 : 0
                    const done = index < activeIndex
                    const active = index === activeIndex
                    return (
                      <div className={`training-stage ${done ? 'done' : ''} ${active ? 'active' : ''}`} key={label}>
                        <span>
                          {done ? (
                            <Check size={14} />
                          ) : active ? (
                            <LoaderCircle className="spin" size={14} />
                          ) : (
                            <Circle size={9} />
                          )}
                        </span>
                        <div>
                          <strong>{label}</strong>
                        </div>
                      </div>
                    )
                  })}
                </div>
                <a
                  className="secondary-button inline-link"
                  href={`/training/simple/jobs/${encodeURIComponent(job.job_id)}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  查看任务详情
                  <ExternalLink size={14} />
                </a>
              </div>
            )}

            {phase === 'complete' && job && (
              <div className="action-block result-card">
                <div className="result-banner">
                  <div className="result-check">
                    <Check size={24} />
                  </div>
                  <div>
                    <div className="eyebrow">真实训练完成</div>
                    <div className="action-title">{job.name}</div>
                  </div>
                  <span className="ready-badge">
                    {assemblyState === 'loading'
                      ? '正在自动加载'
                      : assemblyState === 'ready'
                        ? '已注册并加载'
                        : '训练产物已生成'}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <span>Router F1</span>
                    <strong>{percent(routerMetrics?.f1)}</strong>
                  </div>
                  <div>
                    <span>Text2Comp R²</span>
                    <strong>{metric(job.text2comp_metrics?.r2)}</strong>
                  </div>
                  <div>
                    <span>Text2Comp NRMSE</span>
                    <strong>{metric(job.text2comp_metrics?.normalized_rmse)}</strong>
                  </div>
                </div>
                <div className="artifact-note">
                  <FileCheck2 size={17} />
                  <span>
                    {assemblyState === 'ready'
                      ? `“${assemblyProfile?.name || '完整拼装模型'}”已加入已注册拼装模型，可以直接在下方输入框对话。`
                      : assemblyState === 'loading'
                        ? '正在将训练产物与对应 Expert 注册为完整拼装模型，并自动加载到可用 GPU。'
                        : text2compNeedsTuning
                          ? '真实产物已进入模型扫描目录；当前 Text2Comp 指标低于严格质量要求，建议增加数据或调参后重训。'
                          : 'Router checkpoint 与 Text2Comp 模型已进入模型扫描目录，可以在模型拼装页自由选择。'}
                  </span>
                </div>
                <div className="button-row result-actions">
                  <a className="primary-button" href="/training/simple/assembly">
                    <Layers3 size={16} />
                    前往模型拼装
                  </a>
                  <a className="secondary-button" href={`/training/simple/jobs/${encodeURIComponent(job.job_id)}`}>
                    查看训练任务
                  </a>
                </div>
              </div>
            )}

            {phase === 'error' && (
              <div className="action-block error-card">
                <AlertTriangle size={22} />
                <div>
                  <div className="action-title">流程未完成</div>
                  <p>{error || '训练任务未正常完成，请查看任务详情。'}</p>
                  <div className="button-row">
                    <button className="secondary-button" onClick={resetAfterError}>
                      <RefreshCw size={16} />
                      返回上一步
                    </button>
                    {job && (
                      <a className="secondary-button" href={`/training/simple/jobs/${encodeURIComponent(job.job_id)}`}>
                        查看任务详情
                      </a>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="composer-wrap">
            <div className="composer">
              <button
                className="icon-button"
                title="选择数据"
                disabled={phase === 'complete'}
                onClick={() => setDataOpen(true)}
              >
                <Paperclip size={19} />
              </button>
              <textarea
                aria-label="对话输入"
                value={input}
                onChange={event => setInput(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    submitComposer()
                  }
                }}
                placeholder={
                  phase === 'goal'
                    ? '描述您希望训练的模型…'
                    : phase === 'complete'
                      ? assemblyState === 'ready'
                        ? '向已加载的模型提问…'
                        : '正在注册并加载模型…'
                      : '继续询问或补充要求…'
                }
                disabled={chatBusy || assemblyState === 'loading'}
                rows={1}
              />
              <button
                className="send-button"
                title="发送"
                disabled={chatBusy || assemblyState === 'loading'}
                onClick={submitComposer}
              >
                {chatBusy ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
            <div className="composer-note">
              {assemblyState === 'ready'
                ? `当前模型：${assemblyProfile?.name || '已注册拼装模型'}`
                : '数据生成和 GPU 训练只会在您明确确认后执行'}
            </div>
          </div>
        </section>
      </main>

      {dataOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={event => event.target === event.currentTarget && !busy && setDataOpen(false)}
        >
          <div className="modal data-modal" role="dialog" aria-modal="true" aria-labelledby="data-title">
            <div className="modal-header">
              <div>
                <div className="modal-icon">
                  <Database size={20} />
                </div>
                <div>
                  <h2 id="data-title">选择数据来源</h2>
                  <p>数据将写入平台受管目录</p>
                </div>
              </div>
              <button className="icon-button" title="关闭" disabled={busy} onClick={() => setDataOpen(false)}>
                <X size={19} />
              </button>
            </div>
            <div className="data-tabs" role="tablist">
              {(
                [
                  ['simulation', '内置数据', FlaskConical],
                  ['upload', '上传 HDF5', FileUp],
                  ['existing', '已准备数据', Database],
                ] as const
              ).map(([mode, label, Icon]) => (
                <button key={mode} className={dataMode === mode ? 'active' : ''} onClick={() => setDataMode(mode)}>
                  <Icon size={16} />
                  {label}
                </button>
              ))}
            </div>
            <div className="data-modal-body">
              {dataMode === 'simulation' && (
                <div className="source-form">
                  <label>
                    <span>选择可复用场景</span>
                    <select value={simulationKey} onChange={event => setSimulationKey(event.target.value)}>
                      {simulations.map(item => (
                        <option key={`${item.simulator}/${item.scenario}`} value={`${item.simulator}/${item.scenario}`}>
                          {item.simulator} / {item.scenario} · {item.sample_count.toLocaleString()} 条
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="source-hint">
                    <CheckCircle2 size={17} />
                    <span>复用已有 HDF5，不会重新运行物理仿真；随后会生成配对的 Router 和 Text2Comp 数据。</span>
                  </div>
                  <button
                    className="primary-button full-button"
                    disabled={busy || simulations.length === 0}
                    onClick={useSelectedSimulation}
                  >
                    {busy ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />}接入并准备训练数据
                  </button>
                </div>
              )}
              {dataMode === 'upload' && (
                <div className="source-form">
                  <label className={`dropzone ${selectedFile ? 'has-file' : ''}`}>
                    <input
                      type="file"
                      accept=".h5,.hdf5"
                      onChange={event => setSelectedFile(event.target.files?.[0] || null)}
                    />
                    {selectedFile ? (
                      <>
                        <FileCheck2 size={30} />
                        <strong>{selectedFile.name}</strong>
                        <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB · 等待上传</span>
                      </>
                    ) : (
                      <>
                        <Upload size={30} />
                        <strong>选择 HDF5 数据文件</strong>
                        <span>必须包含 params、param_names 和 timeseries，至少 13 条样本</span>
                      </>
                    )}
                  </label>
                  <button
                    className="primary-button full-button"
                    disabled={!selectedFile || busy}
                    onClick={uploadSelectedFile}
                  >
                    <Upload size={16} />
                    上传、校验并生成训练数据
                  </button>
                </div>
              )}
              {dataMode === 'existing' && (
                <div className="source-list">
                  {datasets.length ? (
                    datasets.map(dataset => (
                      <button
                        className="source-option"
                        key={dataset.dataset_id || `${dataset.simulator}-${dataset.total_count}`}
                        onClick={() => chooseExisting(dataset)}
                      >
                        <div>
                          <strong>{dataset.display_name || dataset.simulator}</strong>
                          <span>
                            {dataset.total_count.toLocaleString()} 条 · {dataset.scenarios.length} 个场景
                          </span>
                        </div>
                        <ArrowRight size={17} />
                      </button>
                    ))
                  ) : (
                    <div className="empty-source">
                      <Clock3 size={22} />
                      <strong>暂无可直接完整训练的数据</strong>
                      <span>请先使用内置数据或上传 HDF5。</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const rootElement = document.getElementById('conversation-training-root') as HTMLElement & {
  __piernConversationTrainingRoot?: ReturnType<typeof createRoot>
}
const root = rootElement.__piernConversationTrainingRoot ?? createRoot(rootElement)
rootElement.__piernConversationTrainingRoot = root

root.render(
  <StrictMode>
    <App />
  </StrictMode>,
)
