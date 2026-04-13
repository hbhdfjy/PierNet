import { useState, useEffect } from 'react'
import useSWR from 'swr'
import { api } from '../lib/api'
import type { LLMConfig, LLMConfigRequest } from '../lib/types'
import {
  KeyRound, Eye, EyeOff, Save, Check, RefreshCw,
  CheckCircle, XCircle, Zap, AlertCircle, Info,
} from 'lucide-react'
import { cn } from '../lib/utils'

// 各 provider 的默认模型建议
const MODEL_SUGGESTIONS: Record<string, string[]> = {
  siliconflow: [
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-V2.5',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
  ],
  openai: [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-3.5-turbo',
  ],
  anthropic: [
    'claude-opus-4-5',
    'claude-sonnet-4-5',
    'claude-haiku-4-5',
    'claude-3-5-sonnet-20241022',
  ],
  local: [
    'llama3',
    'mistral',
    'qwen2.5',
  ],
}

const PROVIDER_LABELS: Record<string, { label: string; defaultUrl: string; keyPlaceholder: string }> = {
  siliconflow: {
    label: 'SiliconFlow',
    defaultUrl: 'https://api.siliconflow.cn/v1',
    keyPlaceholder: 'sk-xxxxxxxxxxxxxxxxxxxxxxxx',
  },
  openai: {
    label: 'OpenAI',
    defaultUrl: 'https://api.openai.com/v1',
    keyPlaceholder: 'sk-xxxxxxxxxxxxxxxxxxxxxxxx',
  },
  anthropic: {
    label: 'Anthropic',
    defaultUrl: 'https://api.anthropic.com/v1',
    keyPlaceholder: 'sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx',
  },
  local: {
    label: 'Local (OpenAI-compatible)',
    defaultUrl: 'http://localhost:11434/v1',
    keyPlaceholder: '（本地部署可留空）',
  },
}

type TestState = 'idle' | 'testing' | 'ok' | 'fail'

export default function LLMConfig() {
  const { data: llmCfg, mutate: refreshLLMCfg } =
    useSWR<LLMConfig>('llm-config', () => api.getLLMConfig())

  const [provider, setProvider] = useState('siliconflow')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [temperature, setTemperature] = useState(1.0)
  const [maxTokens, setMaxTokens] = useState(1024)
  const [showKey, setShowKey] = useState(false)

  const [saving, setSaving] = useState(false)
  const [saveErr, setSaveErr] = useState<string | null>(null)

  const [testState, setTestState] = useState<TestState>('idle')
  const [testMessage, setTestMessage] = useState('')
  const [testPreview, setTestPreview] = useState('')

  // 从服务端加载配置
  useEffect(() => {
    if (llmCfg) {
      setProvider(llmCfg.provider || 'siliconflow')
      setModel(llmCfg.model || '')
      setBaseUrl(llmCfg.base_url || '')
      setTemperature(llmCfg.temperature ?? 1.0)
      setMaxTokens(llmCfg.max_tokens ?? 1024)
      // api_key 不回填（脱敏后无意义）
    }
  }, [llmCfg])

  const buildReq = (): LLMConfigRequest => ({
    provider,
    model,
    api_key: apiKey,
    base_url: baseUrl,
    temperature,
    max_tokens: maxTokens,
  })

  const handleSaveAndTest = async () => {
    setSaving(true)
    setSaveErr(null)
    setTestState('idle')
    setTestMessage('')
    setTestPreview('')

    try {
      // 1. 先保存
      await api.saveLLMConfig(buildReq())
      setApiKey('')   // 保存后清空输入框
      await refreshLLMCfg()

      // 2. 再测试（api_key 留空，后端从已保存配置读取）
      setTestState('testing')
      const result = await api.testLLMConfig({ ...buildReq(), api_key: '' })
      setTestState(result.ok ? 'ok' : 'fail')
      setTestMessage(result.message)
      setTestPreview(result.response_preview)
    } catch (e: unknown) {
      setSaveErr(e instanceof Error ? e.message : '操作失败')
      setTestState('idle')
    } finally {
      setSaving(false)
    }
  }

  const handleTestOnly = async () => {
    setTestState('testing')
    setTestMessage('')
    setTestPreview('')
    try {
      const result = await api.testLLMConfig(buildReq())
      setTestState(result.ok ? 'ok' : 'fail')
      setTestMessage(result.message)
      setTestPreview(result.response_preview)
    } catch (e: unknown) {
      setTestState('fail')
      setTestMessage(e instanceof Error ? e.message : '测试失败')
    }
  }

  const providerInfo = PROVIDER_LABELS[provider] ?? PROVIDER_LABELS.siliconflow
  const suggestions = MODEL_SUGGESTIONS[provider] ?? []

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* ── 左侧：表单 ── */}
      <div className="flex flex-col overflow-y-auto border-r border-slate-700/40"
        style={{ width: '480px', minWidth: '400px', flexShrink: 0 }}>
        <div className="p-6 space-y-5">

          {/* 页头 */}
          <div className="flex items-center gap-3 mb-2">
            <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
              <KeyRound size={17} className="text-amber-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">LLM 配置</h1>
              <p className="text-slate-500 text-xs mt-0.5">配置模板生成所用的大语言模型 API</p>
            </div>
            {llmCfg?.has_api_key && (
              <span className="ml-auto badge bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 text-xs">
                ✓ Key 已配置
              </span>
            )}
          </div>

          {/* Provider */}
          <div className="card p-4 space-y-4">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-slate-200 text-base">API 提供商</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(PROVIDER_LABELS).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => { setProvider(key); setModel(''); setBaseUrl('') }}
                  className={cn(
                    'px-3 py-2.5 rounded-xl border text-sm font-medium transition-all text-left',
                    provider === key
                      ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                      : 'bg-slate-800/40 border-slate-700/40 text-slate-400 hover:border-slate-600 hover:text-slate-200',
                  )}
                >
                  {info.label}
                </button>
              ))}
            </div>
          </div>

          {/* 模型 */}
          <div className="card p-4 space-y-3">
            <label className="font-medium text-slate-200 text-base block">模型名称</label>
            <input
              type="text"
              className="input w-full"
              placeholder={suggestions[0] ?? '输入模型名称'}
              value={model}
              onChange={e => setModel(e.target.value)}
            />
            {suggestions.length > 0 && (
              <div>
                <div className="label mb-2 text-xs">常用模型</div>
                <div className="flex flex-wrap gap-1.5">
                  {suggestions.map(s => (
                    <button
                      key={s}
                      onClick={() => setModel(s)}
                      className={cn(
                        'px-2.5 py-1 rounded-lg border text-xs font-mono transition-all',
                        model === s
                          ? 'bg-amber-500/15 border-amber-500/35 text-amber-300'
                          : 'bg-slate-800/40 border-slate-700/40 text-slate-500 hover:text-slate-300 hover:border-slate-600',
                      )}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* API Key */}
          <div className="card p-4 space-y-3">
            <div>
              <label className="font-medium text-slate-200 text-base block mb-1">API Key</label>
              {llmCfg?.has_api_key && (
                <p className="text-xs text-slate-500 mb-2">
                  当前：<span className="font-mono text-slate-400">{llmCfg.api_key_masked}</span>
                  <span className="ml-2 text-slate-600">（留空保持不变）</span>
                </p>
              )}
              <div className="relative">
                <input
                  type={showKey ? 'text' : 'password'}
                  className="input w-full pr-11"
                  placeholder={llmCfg?.has_api_key ? '留空保持现有 Key 不变' : providerInfo.keyPlaceholder}
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  autoComplete="off"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                  onClick={() => setShowKey(v => !v)}
                  tabIndex={-1}
                >
                  {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {/* Base URL */}
            <div>
              <label className="font-medium text-slate-200 text-base block mb-1">
                Base URL
                <span className="ml-2 text-slate-600 font-normal text-xs">（可选，用于代理或自部署）</span>
              </label>
              <input
                type="text"
                className="input w-full"
                placeholder={providerInfo.defaultUrl}
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
              />
              <p className="text-xs text-slate-600 mt-1.5">
                留空使用默认：<span className="font-mono">{providerInfo.defaultUrl}</span>
              </p>
            </div>
          </div>

          {/* 高级参数 */}
          <div className="card p-4 space-y-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-slate-200 text-base">生成参数</span>
              <span className="badge bg-slate-700/50 text-slate-500 border border-slate-600/30 text-xs">高级</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label block mb-2 text-xs">
                  Temperature
                  <span className="ml-1.5 text-slate-500 font-normal">({temperature.toFixed(1)})</span>
                </label>
                <input type="range" className="w-full accent-amber-500" min={0} max={2} step={0.1}
                  value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} />
                <div className="flex justify-between text-xs text-slate-600 mt-0.5"><span>0</span><span>2</span></div>
              </div>
              <div>
                <label className="label block mb-2 text-xs">Max Tokens</label>
                <input type="number" className="input w-full text-sm py-1.5"
                  min={64} max={8192} step={64}
                  value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value) || 1024)} />
              </div>
            </div>
          </div>

          {/* 错误 */}
          {saveErr && (
            <div className="flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-xl px-3 py-2.5 text-red-300">
              <AlertCircle size={15} className="flex-shrink-0 mt-0.5" />
              <span className="text-xs">{saveErr}</span>
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-2">
            <button
              className={cn(
                'btn flex-1 py-3 text-sm justify-center',
                saving ? 'btn-ghost' : 'btn-primary',
              )}
              style={saving ? {} : { background: 'linear-gradient(135deg, #d97706, #b45309)' }}
              onClick={handleSaveAndTest}
              disabled={saving || testState === 'testing'}
            >
              {saving ? (
                <><RefreshCw size={15} className="animate-spin" /> 保存中…</>
              ) : testState === 'testing' ? (
                <><RefreshCw size={15} className="animate-spin" /> 测试中…</>
              ) : (
                <><Save size={15} /> 保存并测试</>
              )}
            </button>
            <button
              className="btn btn-ghost py-3 px-4 text-sm"
              onClick={handleTestOnly}
              disabled={saving || testState === 'testing'}
              title="仅测试，不保存"
            >
              <Zap size={15} />
            </button>
          </div>

          <p className="text-xs text-slate-600 text-center -mt-2">
            配置保存到 <span className="font-mono">configs/text2comp/generation.yaml</span>
          </p>

        </div>
      </div>

      {/* ── 右侧：测试结果 + 说明 ── */}
      <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-5">

        {/* 测试结果卡片 */}
        {testState !== 'idle' && (
          <div className={cn(
            'card overflow-hidden',
            testState === 'ok' ? 'border-emerald-500/30' :
            testState === 'fail' ? 'border-red-500/30' :
            'border-slate-700/40',
          )}>
            <div className={cn(
              'flex items-center gap-3 px-4 py-3 border-b border-slate-700/30',
              testState === 'ok' ? 'bg-emerald-500/5' :
              testState === 'fail' ? 'bg-red-500/5' :
              'bg-slate-800/30',
            )}>
              {testState === 'testing' && <RefreshCw size={16} className="animate-spin text-amber-400" />}
              {testState === 'ok'      && <CheckCircle size={16} className="text-emerald-400" />}
              {testState === 'fail'    && <XCircle size={16} className="text-red-400" />}
              <span className={cn(
                'font-medium text-sm',
                testState === 'ok' ? 'text-emerald-300' :
                testState === 'fail' ? 'text-red-300' :
                'text-amber-300',
              )}>
                {testState === 'testing' ? '正在测试连通性…' :
                 testState === 'ok' ? '连接成功' : '连接失败'}
              </span>
            </div>

            {testState !== 'testing' && (
              <div className="p-4 space-y-3">
                <div className="text-sm text-slate-400">
                  {testMessage}
                </div>
                {testPreview && (
                  <div>
                    <div className="label mb-1.5 text-xs">模型响应预览</div>
                    <div className="bg-slate-900/60 rounded-lg px-3 py-2.5 font-mono text-sm text-slate-300 border border-slate-700/30">
                      {testPreview}
                    </div>
                  </div>
                )}
              </div>
            )}

            {testState === 'testing' && (
              <div className="p-4">
                <div className="flex gap-1.5 justify-center py-2">
                  {[0,1,2,3,4].map(i => (
                    <div key={i}
                      className="w-2 h-2 rounded-full bg-amber-500 animate-bounce"
                      style={{ animationDelay: `${i * 0.12}s` }}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 空状态 */}
        {testState === 'idle' && (
          <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
            <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Zap size={24} className="text-amber-400/60" />
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">配置后点击「保存并测试」</p>
              <p className="text-slate-600 text-xs mt-1">将发送一条极短的测试消息验证 API 连通性</p>
            </div>
          </div>
        )}

        {/* 使用说明 */}
        <div className="card p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Info size={13} className="text-slate-500" />
            <span className="font-medium text-slate-300 text-sm">使用说明</span>
          </div>
          <div className="space-y-2.5 text-xs text-slate-500 leading-relaxed">
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span>配置保存到 <span className="font-mono text-slate-400">configs/text2comp/generation.yaml</span>，模板生成时自动读取</span>
            </div>
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span>API Key 留空时保持当前已保存的 Key 不变；测试时始终使用已保存的 Key</span>
            </div>
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span><span className="text-slate-400">SiliconFlow</span>：国内高速访问，支持 DeepSeek、Qwen 等主流开源模型，推荐用于大规模生成</span>
            </div>
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span><span className="text-slate-400">Base URL</span>：可配置代理地址（如内网 mcli 服务），格式为 <span className="font-mono text-slate-400">https://host/v1</span></span>
            </div>
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span><span className="text-slate-400">Temperature</span> 建议 0.8–1.2，过低模板多样性差，过高容易出现格式错误</span>
            </div>
            <div className="flex gap-2">
              <span className="text-amber-500/70 flex-shrink-0 mt-0.5">•</span>
              <span><span className="text-slate-400">Max Tokens</span> 建议 512–1024，模板生成通常不超过 600 token</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
