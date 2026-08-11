import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import {
  ArrowRight,
  Bot,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Loader2,
  MessageSquare,
  Route,
  Send,
  Sparkles,
  UserRound,
  Workflow,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import { formatAssemblyAnswer } from './assemblyResult'

type AssemblyStatus = Awaited<ReturnType<typeof api.getAssemblyStatus>>
type AssemblyProfile = NonNullable<AssemblyStatus['assembly_profiles']>[number]

const MANUAL_CHAIN_ID = '__manual_chain__'

type UserMessage = {
  id: string
  role: 'user'
  content: string
}

type AssistantMessage = {
  id: string
  role: 'assistant'
  content: string
  router: string
  latencyMs: number
  expertUsed: boolean
  fullResponse: string
}

type ChatMessage = UserMessage | AssistantMessage

function pathName(path?: string | null): string {
  if (!path) return '未加载'
  return path.split('/').filter(Boolean).pop() || path
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || '未知错误')
}

function loadedComponentNames(
  status?: AssemblyStatus,
  selectedProfile?: AssemblyProfile | null,
  selectedReady = false,
) {
  const loaded = status?.loaded_models
  const profileId = loaded?.assembly_profile?.model_id
  const loadedProfile = (status?.assembly_profiles ?? []).find(item => item.model_id === profileId)
  const profile = selectedProfile ?? loadedProfile
  const profileSelected = Boolean(selectedProfile)
  const expertPath = profileSelected
    ? profile?.expert_path
    : loaded?.uploaded_expert?.loaded
      ? loaded.uploaded_expert.model_id
      : loaded?.fno?.paths?.[0] || profile?.expert_path

  return [
    {
      key: 'llm',
      label: 'LLM',
      value: pathName(profileSelected ? profile?.llm_path : loaded?.llm?.path || profile?.llm_path),
      ready: profileSelected ? selectedReady : Boolean(loaded?.llm?.loaded || loaded?.assembly_profile?.loaded),
    },
    {
      key: 'router',
      label: 'Router',
      value: pathName(profileSelected ? profile?.router_path : loaded?.router?.path || profile?.router_path),
      ready: profileSelected ? selectedReady : Boolean(loaded?.router?.loaded || loaded?.assembly_profile?.loaded),
    },
    {
      key: 'text2comp',
      label: 'Text2Comp',
      value: pathName(
        profileSelected ? profile?.text2comp_path : loaded?.text2comp?.paths?.[0] || profile?.text2comp_path,
      ),
      ready: profileSelected ? selectedReady : Boolean(loaded?.text2comp?.loaded || loaded?.assembly_profile?.loaded),
    },
    {
      key: 'expert',
      label: 'Expert',
      value: pathName(expertPath),
      ready: profileSelected
        ? selectedReady
        : Boolean(loaded?.fno?.loaded || loaded?.uploaded_expert?.loaded || loaded?.assembly_profile?.loaded),
    },
  ]
}

export default function ModelChatPage({ assemblyPath = '/training/assembly' }: { assemblyPath?: string }) {
  const {
    data: status,
    error,
    isLoading,
    mutate,
  } = useSWR<AssemblyStatus>('assembly-status', api.getAssemblyStatus, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const [selectedModelId, setSelectedModelId] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const [switching, setSwitching] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [injectedProfileId, setInjectedProfileId] = useState('')
  const sequence = useRef(0)
  const threadEnd = useRef<HTMLDivElement>(null)

  const loadedModels = status?.loaded_models
  const loadedProfileId = loadedModels?.assembly_profile?.loaded ? loadedModels.assembly_profile.model_id || '' : ''
  const loadedExpertId = loadedModels?.uploaded_expert?.loaded ? loadedModels.uploaded_expert.model_id || '' : ''
  const availableProfiles = useMemo(
    () => (status?.assembly_profiles ?? []).filter(item => item.trained && item.chat_enabled),
    [status?.assembly_profiles],
  )
  const loadedProfile = useMemo(
    () => (status?.assembly_profiles ?? []).find(item => item.model_id === loadedProfileId) ?? null,
    [loadedProfileId, status?.assembly_profiles],
  )
  const loadedExpert = useMemo(
    () => (status?.custom_experts ?? []).find(item => item.model_id === loadedExpertId) ?? null,
    [loadedExpertId, status?.custom_experts],
  )
  const manualChainName =
    loadedExpert?.simulator?.trim().toLowerCase() === 'gcam'
      ? 'GCAM 能源-气候模型'
      : loadedExpert?.name || '当前手动拼装链路'
  const standardChainReady = Boolean(
    loadedModels?.llm?.loaded &&
    loadedModels?.router?.loaded &&
    loadedModels?.text2comp?.loaded &&
    (loadedModels?.fno?.loaded || loadedModels?.uploaded_expert?.loaded),
  )
  const manualChainAvailable = Boolean(!loadedProfileId && standardChainReady)
  const activeModelId = loadedProfileId || (manualChainAvailable ? MANUAL_CHAIN_ID : '')
  const promptSourceId = loadedProfileId
    ? `profile:${loadedProfileId}`
    : manualChainAvailable
      ? `manual:${loadedExpertId || 'fno'}`
      : ''
  const selectedProfile = useMemo(
    () => availableProfiles.find(item => item.model_id === selectedModelId) ?? null,
    [availableProfiles, selectedModelId],
  )
  const ready = Boolean(selectedModelId && selectedModelId === activeModelId)
  const componentNames = useMemo(
    () => loadedComponentNames(status, selectedProfile, ready),
    [ready, selectedProfile, status],
  )
  const modelName =
    selectedProfile?.name ||
    (selectedModelId === MANUAL_CHAIN_ID ? manualChainName : loadedProfile?.name) ||
    loadedModels?.assembly_profile?.name ||
    '模型对话'

  useEffect(() => {
    if (switching) return
    if (activeModelId) {
      setSelectedModelId(activeModelId)
      return
    }
    setSelectedModelId(current =>
      current && availableProfiles.some(profile => profile.model_id === current) ? current : '',
    )
  }, [activeModelId, availableProfiles, switching])

  useEffect(() => {
    if (promptSourceId === injectedProfileId) return
    setInjectedProfileId(promptSourceId)
    setInput(loadedProfile?.demo_prompt?.trim() || loadedExpert?.demo_prompt?.trim() || '')
    setMessages([])
    setActionError(null)
  }, [injectedProfileId, loadedExpert?.demo_prompt, loadedProfile?.demo_prompt, promptSourceId])

  useEffect(() => {
    threadEnd.current?.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' })
  }, [busy, messages])

  const nextId = (role: ChatMessage['role']) => {
    sequence.current += 1
    return `${role}-${sequence.current}`
  }

  const switchModel = async (modelId: string) => {
    setSelectedModelId(modelId)
    setActionError(null)
    if (!modelId || modelId === MANUAL_CHAIN_ID || modelId === loadedProfileId) return

    const profile = availableProfiles.find(item => item.model_id === modelId)
    if (!profile) {
      setActionError('所选模型已不可用，请刷新后重试。')
      setSelectedModelId(activeModelId)
      return
    }

    const gpuId =
      loadedModels?.llm?.gpu_id ?? status?.gpus.find(item => item.available)?.index ?? status?.gpus[0]?.index ?? 0
    setSwitching(true)
    try {
      await api.loadAssemblyModels({
        assembly_profile_id: profile.model_id,
        llm_gpu_id: gpuId,
        router_gpu_id: gpuId,
        force_split: Boolean(profile.force_split),
      })
      await mutate()
    } catch (reason) {
      setActionError(`切换模型失败：${errorMessage(reason)}`)
      setSelectedModelId(activeModelId)
    } finally {
      setSwitching(false)
    }
  }

  const sendMessage = async () => {
    const content = input.trim()
    if (!content || !ready || busy || switching) return

    setMessages(current => [...current, { id: nextId('user'), role: 'user', content }])
    setInput('')
    setBusy(true)
    setActionError(null)
    try {
      const result = await api.testAssembly({
        config: {
          main_llm_path: selectedProfile?.llm_path || loadedModels?.llm?.path || undefined,
          assembly_profile_id: selectedProfile?.model_id || undefined,
          gpu_config: {
            llm_gpu_ids: loadedModels?.llm?.gpu_id == null ? [] : [loadedModels.llm.gpu_id],
          },
        },
        test_input: content,
      })
      const rawAnswer = result.final_answer || result.first_cot_result?.split('\n').pop() || ''
      setMessages(current => [
        ...current,
        {
          id: nextId('assistant'),
          role: 'assistant',
          content: formatAssemblyAnswer(rawAnswer),
          router: result.router_class_name || result.router_prediction || '--',
          latencyMs: result.latency_ms,
          expertUsed: Boolean(result.expert_used),
          fullResponse: result.llm_response || result.first_cot_result || '',
        },
      ])
    } catch (reason) {
      setInput(content)
      setActionError(errorMessage(reason))
    } finally {
      setBusy(false)
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    void sendMessage()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
    event.preventDefault()
    void sendMessage()
  }

  if (isLoading) {
    return (
      <div className="training-page">
        <div className="training-page__body flex items-center justify-center">
          <div className="flex items-center gap-3 text-sm text-slate-400">
            <Loader2 size={18} className="animate-spin" />
            正在读取拼装模型
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="model-chat-page">
          <section className="training-hero training-hero--compact model-chat-hero">
            <div>
              <div className="training-eyebrow">模型对话</div>
              <h1 className="mt-2 text-2xl font-semibold text-white">{ready ? modelName : '模型对话'}</h1>
              <p className="mt-1 text-sm text-slate-400">
                {switching
                  ? '正在切换模型，请稍候'
                  : ready
                    ? '当前拼装链路已就绪'
                    : availableProfiles.length > 0
                      ? '选择一个已部署模型后开始对话'
                      : '请先在模型拼装中装载一条完整链路'}
              </p>
            </div>
            <div className="model-chat-hero__actions">
              {(availableProfiles.length > 0 || manualChainAvailable) && (
                <label className="model-chat-model-picker">
                  <span>已部署模型 · {availableProfiles.length}</span>
                  <div className="model-chat-model-picker__control">
                    <select
                      aria-label="已部署模型"
                      value={selectedModelId}
                      onChange={event => void switchModel(event.target.value)}
                      disabled={switching}
                    >
                      {!activeModelId && <option value="">请选择模型</option>}
                      {manualChainAvailable && <option value={MANUAL_CHAIN_ID}>{manualChainName}</option>}
                      {availableProfiles.map(profile => (
                        <option key={profile.model_id} value={profile.model_id}>
                          {profile.name}
                        </option>
                      ))}
                    </select>
                    {switching ? <Loader2 size={15} className="animate-spin" /> : <ChevronDown size={15} />}
                  </div>
                </label>
              )}
              {switching ? (
                <span className="model-chat-switching">
                  <Loader2 size={15} className="animate-spin" />
                  切换中
                </span>
              ) : ready ? (
                <span className="model-chat-ready">
                  <CheckCircle2 size={15} />
                  可对话
                </span>
              ) : availableProfiles.length === 0 ? (
                <Link className="btn-primary" to={assemblyPath}>
                  前往模型拼装
                  <ArrowRight size={15} />
                </Link>
              ) : null}
            </div>
          </section>

          {(error || actionError) && (
            <div className="model-chat-error">{actionError ?? `无法读取模型状态：${error?.message}`}</div>
          )}

          <section className="model-chat-status" aria-label="当前模型链路">
            {componentNames.map((item, index) => {
              const icons = [Cpu, Route, Workflow, Sparkles]
              const Icon = icons[index]
              return (
                <div className="model-chat-status__item" key={item.key}>
                  <span>
                    <Icon size={15} />
                    {item.label}
                  </span>
                  <strong title={item.value}>{item.value}</strong>
                  <small className={item.ready ? 'is-ready' : undefined}>{item.ready ? '已加载' : '未加载'}</small>
                </div>
              )
            })}
          </section>

          <section className="model-chat-shell" aria-label="模型对话区">
            <div className="model-chat-shell__header">
              <MessageSquare size={17} />
              <span>对话</span>
              {messages.length > 0 && <small>{messages.filter(message => message.role === 'user').length} 轮</small>}
            </div>

            <div className="model-chat-thread" aria-live="polite">
              {messages.length === 0 ? (
                <div className="model-chat-empty">
                  <Bot size={28} />
                  <strong>{ready ? modelName : '尚未装载模型'}</strong>
                  <span>{ready ? '输入问题后开始计算' : '完成模型拼装后即可在此对话'}</span>
                </div>
              ) : (
                messages.map(message => (
                  <article className={`model-chat-message model-chat-message--${message.role}`} key={message.id}>
                    <div className="model-chat-message__avatar">
                      {message.role === 'user' ? <UserRound size={16} /> : <Bot size={16} />}
                    </div>
                    <div className="model-chat-message__content">
                      <div className="model-chat-message__bubble">{message.content}</div>
                      {message.role === 'assistant' && (
                        <>
                          <div className="model-chat-message__meta">
                            <span>Router：{message.router}</span>
                            <span>{message.latencyMs.toFixed(2)} ms</span>
                            {message.expertUsed && <span>Expert 已调用</span>}
                          </div>
                          {message.fullResponse && message.fullResponse.trim() !== message.content.trim() && (
                            <details className="model-chat-message__details">
                              <summary>查看完整响应</summary>
                              <pre>{message.fullResponse}</pre>
                            </details>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                ))
              )}
              {busy && (
                <div className="model-chat-thinking">
                  <Loader2 size={16} className="animate-spin" />
                  模型正在计算
                </div>
              )}
              <div ref={threadEnd} />
            </div>

            <form className="model-chat-composer" onSubmit={submit}>
              <textarea
                aria-label="对话输入"
                value={input}
                onChange={event => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={ready ? '输入计算问题' : '请先装载拼装模型'}
                disabled={!ready || busy || switching}
                rows={3}
              />
              <button
                type="submit"
                className="btn-primary model-chat-composer__send"
                disabled={!ready || !input.trim() || busy || switching}
                aria-label="发送"
                title="发送"
              >
                {busy ? <Loader2 size={17} className="animate-spin" /> : <Send size={17} />}
              </button>
            </form>
          </section>
        </div>
      </div>
    </div>
  )
}
