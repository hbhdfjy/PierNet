import useSWR from 'swr'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { AlertTriangle, CheckCircle2, KeyRound, RefreshCw } from 'lucide-react'
import { api } from '../../lib/api'
import type { LLMConfig, Text2CompScenario, Text2CompScenariosConfig } from '../../lib/types'
import { cn } from '../../lib/utils'
import InterviewPanel from '../components/interview/InterviewPanel'

function pendingKey(item: Text2CompScenario) {
  return `${item.simulator}/${item.name}`
}

export default function RegisterSimulator() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { data: registry, mutate: mutateRegistry } = useSWR('registry', () => api.getRegistry(), {
    revalidateOnFocus: false,
  })
  const {
    data: scenariosCfg,
    isLoading,
    mutate: refreshScenarios,
  } = useSWR<Text2CompScenariosConfig>('register-text2comp-scenarios', () => api.getText2CompScenarios(), {
    refreshInterval: 10000,
    revalidateOnFocus: false,
  })
  const { data: llmCfg } = useSWR<LLMConfig>('llm-config', () => api.getLLMConfig(), { revalidateOnFocus: false })

  const pending = scenariosCfg
    ? Object.values(scenariosCfg)
        .flat()
        .filter(item => item.has_h5 && !item.registered)
        .sort((a, b) => pendingKey(a).localeCompare(pendingKey(b)))
    : []

  const querySimulator = searchParams.get('simulator') ?? ''
  const queryScenario = searchParams.get('scenario') ?? ''
  const queryHdf5Path = searchParams.get('hdf5_path') ?? ''
  const registeredSimulators = registry
    ? new Set(Object.keys(registry).map(key => key.split('/')[0]))
    : new Set<string>()
  const initialMode: 'simulator' | 'scenario' =
    queryScenario && registeredSimulators.has(querySimulator) ? 'scenario' : 'simulator'
  const panelKey = `${initialMode}:${querySimulator}:${queryScenario}:${queryHdf5Path}`

  const openPending = (item: Text2CompScenario) => {
    const params = new URLSearchParams({
      simulator: item.simulator,
      scenario: item.name,
    })
    if (item.h5_file) params.set('hdf5_path', item.h5_file)
    navigate(`/synth/register?${params.toString()}`)
  }

  return (
    <div className="page-shell">
      <div className="page-content space-y-4 p-4">
        <section className="training-card overflow-hidden">
          <div className="card-header">
            <CheckCircle2 size={17} className="text-sky-400" />
            <div>
              <div className="training-panel-title">注册前检查</div>
              <div className="training-panel-copy">直接处理有 HDF5 但还没有 registry 的数据。</div>
            </div>
            <div className="flex-1" />
            <button className="btn-ghost py-1.5 text-xs" onClick={() => refreshScenarios()}>
              <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />
              刷新
            </button>
          </div>

          <div className="grid grid-cols-1 gap-3 p-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
            <div className="rounded-xl border border-slate-700/35 bg-slate-900/25 p-3">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-slate-100">待注册数据列表</div>
                  <div className="mt-0.5 text-xs text-slate-500">{pending.length} 个 HDF5 场景未注册</div>
                </div>
              </div>
              {pending.length > 0 ? (
                <div className="grid max-h-64 grid-cols-1 gap-2 overflow-y-auto pr-1 md:grid-cols-2 2xl:grid-cols-3">
                  {pending.map(item => (
                    <button
                      key={pendingKey(item)}
                      className={cn(
                        'rounded-xl border px-3 py-2 text-left transition-all',
                        querySimulator === item.simulator && queryScenario === item.name
                          ? 'border-sky-400/45 bg-sky-500/12'
                          : 'border-slate-700/35 bg-slate-950/25 hover:border-slate-600/65 hover:bg-slate-900/45',
                      )}
                      onClick={() => openPending(item)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-sm font-semibold text-slate-100">{item.name}</span>
                        <span className="flex-shrink-0 font-mono text-xs text-sky-300">
                          {item.sample_count.toLocaleString()}
                        </span>
                      </div>
                      <div className="mt-1 truncate font-mono text-xs text-slate-500">{item.simulator}</div>
                      <div className="mt-1 truncate font-mono text-[11px] text-slate-600">
                        {item.h5_file ?? 'HDF5 path unknown'}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-5 text-center text-sm text-emerald-200">
                  当前没有待注册的 HDF5 数据。
                </div>
              )}
            </div>

            <div
              className={cn(
                'rounded-xl border p-3',
                llmCfg?.has_api_key ? 'border-emerald-500/25 bg-emerald-500/8' : 'border-amber-500/25 bg-amber-500/8',
              )}
            >
              <div className="flex items-start gap-2">
                {llmCfg?.has_api_key ? (
                  <CheckCircle2 size={16} className="mt-0.5 text-emerald-300" />
                ) : (
                  <AlertTriangle size={16} className="mt-0.5 text-amber-300" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold text-slate-100">
                    {llmCfg?.has_api_key ? 'LLM Key 已配置' : 'LLM Key 未配置'}
                  </div>
                  <div className="mt-1 truncate text-xs text-slate-400">
                    {llmCfg ? `${llmCfg.provider} · ${llmCfg.model || '未设置模型'}` : '加载配置中'}
                  </div>
                </div>
              </div>
              {!llmCfg?.has_api_key && (
                <button
                  className="btn-primary mt-3 w-full justify-center py-2 text-xs"
                  onClick={() => navigate('/synth/llm-config')}
                >
                  <KeyRound size={12} />
                  先配置 LLM
                </button>
              )}
            </div>
          </div>
        </section>

        <InterviewPanel
          key={panelKey}
          initialMode={initialMode}
          initialSimulator={querySimulator}
          initialScenario={queryScenario}
          initialHdf5Path={queryHdf5Path}
          onRegistryUpdate={() => {
            mutateRegistry()
            refreshScenarios()
          }}
        />
      </div>
    </div>
  )
}
