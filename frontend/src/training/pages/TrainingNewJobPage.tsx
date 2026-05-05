import { Check, Cpu, Database, PlayCircle, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { api } from '../../lib/api'
import type {
  TrainingCreateJobRequest,
  TrainingDatasetInfo,
  TrainingGPUInfo,
  TrainingJobSummary,
} from '../../lib/types'
import {
  formatBytes,
  formatCount,
  formatDateTime,
  formatMetric,
  gpuUsageLabel,
  statusBadgeClass,
  statusLabel,
} from '../shared'

function toNumber(value: string, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function SectionTitle({ title, copy }: { title: string; copy: string }) {
  return (
    <div>
      <div className="training-panel-title">{title}</div>
      <div className="training-panel-copy">{copy}</div>
    </div>
  )
}

function Field({ label, children, note }: { label: string; children: React.ReactNode; note?: string }) {
  return (
    <div>
      <label className="training-label">{label}</label>
      <div className="mt-1.5">{children}</div>
      {note && <div className="mt-1 training-note">{note}</div>}
    </div>
  )
}

function UsageBar({ value }: { value: number }) {
  return (
    <div className="training-progress mt-1.5">
      <div className="training-progress__fill" style={{ width: `${Math.max(4, Math.min(100, value))}%` }} />
    </div>
  )
}

export default function TrainingNewJobPage() {
  const { mutate } = useSWRConfig()
  const navigate = useNavigate()

  const { data: datasets } = useSWR<TrainingDatasetInfo[]>('training-datasets', api.getTrainingDatasets, {
    refreshInterval: 15000,
    revalidateOnFocus: false,
  })
  const { data: gpus } = useSWR<TrainingGPUInfo[]>('training-gpus', api.getTrainingGPUs, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const { data: jobs } = useSWR<TrainingJobSummary[]>('training-jobs', api.getTrainingJobs, {
    refreshInterval: 10000,
    revalidateOnFocus: false,
  })

  const [simulator, setSimulator] = useState('modflow')
  const [jobName, setJobName] = useState('')
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([])
  const [gpuId, setGpuId] = useState<number | null>(null)
  const [epochs, setEpochs] = useState('0')
  const [infiniteEpochs, setInfiniteEpochs] = useState(true)
  const [evalInterval, setEvalInterval] = useState('1')
  const [keepLastEpochs, setKeepLastEpochs] = useState('5')
  const [batchSize, setBatchSize] = useState('256')
  const [testBatchSize, setTestBatchSize] = useState('256')
  const [learningRate, setLearningRate] = useState('0.0002')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [numWorkers, setNumWorkers] = useState('8')
  const [testRatio, setTestRatio] = useState('0.1')
  const [resumeFrom, setResumeFrom] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const dataset = useMemo(
    () => datasets?.find(item => item.simulator === simulator) ?? datasets?.[0] ?? null,
    [datasets, simulator],
  )
  const availableGpus = useMemo(() => (gpus ?? []).filter(gpu => gpu.available), [gpus])
  const checkpointCandidates = useMemo(
    () =>
      (jobs ?? [])
        .filter(job => job.status === 'done')
        .filter(job => job.simulator === (dataset?.simulator ?? simulator))
        .filter(job => {
          const selected = [...selectedScenarios].sort()
          const candidate = [...job.scenarios].sort()
          return selected.length > 0 && selected.length === candidate.length && selected.every((item, idx) => item === candidate[idx])
        })
        .map(job => ({
          label: `${job.name} · 最新权重`,
          value: `${job.run_dir}/router_latest.pt`,
        })),
    [dataset?.simulator, jobs, selectedScenarios, simulator],
  )
  const selectedScenarioDetails = useMemo(
    () => (dataset?.scenarios ?? []).filter(item => selectedScenarios.includes(item.scenario)),
    [dataset?.scenarios, selectedScenarios],
  )
  const selectedRouterCount = selectedScenarioDetails.reduce((sum, item) => sum + item.router_count, 0)
  const selectedFileSize = selectedScenarioDetails.reduce((sum, item) => sum + item.file_size_bytes, 0)
  const selectedGpu = useMemo(() => (gpus ?? []).find(gpu => gpu.index === gpuId) ?? null, [gpus, gpuId])

  useEffect(() => {
    if (!dataset) return
    setSimulator(dataset.simulator)
    setSelectedScenarios(dataset.scenarios.map(item => item.scenario))
  }, [dataset?.simulator])

  useEffect(() => {
    if (!dataset) return
    setSelectedScenarios(prev => {
      const available = new Set(dataset.scenarios.map(item => item.scenario))
      const kept = prev.filter(item => available.has(item))
      return kept.length > 0 ? kept : dataset.scenarios.map(item => item.scenario)
    })
  }, [dataset])

  useEffect(() => {
    if (gpuId != null && (gpus ?? []).some(gpu => gpu.index === gpuId && gpu.available)) return
    setGpuId(availableGpus[0]?.index ?? null)
  }, [availableGpus, gpus, gpuId])

  const toggleScenario = (scenario: string) => {
    setSelectedScenarios(current =>
      current.includes(scenario) ? current.filter(item => item !== scenario) : [...current, scenario],
    )
  }

  const submit = async () => {
    if (!dataset) {
      setError('当前没有可用训练数据。')
      return
    }
    if (selectedScenarios.length === 0) {
      setError('至少选择一个训练子场景。')
      return
    }
    if (gpuId == null) {
      setError('当前没有可用 GPU。')
      return
    }

    const payload: TrainingCreateJobRequest = {
      name: jobName.trim() || null,
      simulator: dataset.simulator,
      scenarios: selectedScenarios,
      gpu_id: gpuId,
      epochs: infiniteEpochs ? 0 : Math.max(1, Math.floor(toNumber(epochs, 1))),
      eval_interval: Math.max(1, Math.floor(toNumber(evalInterval, 1))),
      keep_last_epochs: Math.max(0, Math.floor(toNumber(keepLastEpochs, 5))),
      batch_size: Math.max(1, Math.floor(toNumber(batchSize, 256))),
      test_batch_size: Math.max(1, Math.floor(toNumber(testBatchSize, 256))),
      learning_rate: Math.max(1e-8, toNumber(learningRate, 2e-4)),
      weight_decay: Math.max(0, toNumber(weightDecay, 0.01)),
      num_workers: Math.max(0, Math.floor(toNumber(numWorkers, 8))),
      test_ratio: Math.min(0.5, Math.max(0.01, toNumber(testRatio, 0.1))),
      resume_from: resumeFrom.trim() || null,
    }

    try {
      setSubmitting(true)
      setError(null)
      const job = await api.createTrainingJob(payload)
      await Promise.all([
        mutate('training-overview'),
        mutate('training-datasets'),
        mutate('training-gpus'),
        mutate('training-jobs'),
      ])
      navigate(`/training/jobs/${job.job_id}`)
      return
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="training-eyebrow">新建训练</div>
                <h1 className="mt-2 text-[1.65rem] font-semibold tracking-tight text-white xl:text-[1.9rem]">Token Router 训练</h1>
                <div className="mt-2 flex flex-wrap gap-2 text-[12px] text-slate-400">
                  <span className="training-chip">场景 {selectedScenarios.length}/{dataset?.scenarios.length ?? 0}</span>
                  <span className="training-chip">样本 {formatCount(selectedRouterCount)}</span>
                  <span className="training-chip">GPU {gpuId ?? '—'}</span>
                </div>
              </div>
              <button type="button" className="btn-primary" onClick={submit} disabled={submitting}>
                <PlayCircle size={14} />
                {submitting ? '启动中...' : '启动训练'}
              </button>
            </div>
          </section>

          {error && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {error}
            </div>
          )}

          <div className="grid gap-4 xl:grid-cols-[1.22fr_0.78fr]">
            <section className="training-stack">
              <div className="training-card training-card--compact flex-1 min-h-0">
                <div className="card-header">
                  <Database size={16} className="text-sky-300" />
                  <SectionTitle title="训练数据" copy="选择大场景和子场景范围" />
                </div>
                <div className="training-card__body training-scroll list-scroll-lg">
                  <div className="grid gap-3 md:grid-cols-2">
                    <Field label="任务名称" note="为空时自动使用任务 ID">
                      <input className="input" value={jobName} onChange={e => setJobName(e.target.value)} placeholder="例如：modflow-router-v1" />
                    </Field>
                    <Field label="大场景">
                      <select className="select" value={dataset?.simulator ?? simulator} onChange={e => setSimulator(e.target.value)}>
                        {(datasets ?? []).map(item => (
                          <option key={item.simulator} value={item.simulator}>
                            {item.simulator.toUpperCase()} · {formatCount(item.total_count)} 条
                          </option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  {dataset ? (
                    <div className="space-y-2.5">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="text-[14px] font-medium text-slate-100">
                          已选择 {selectedScenarios.length} / {dataset.scenarios.length} 个子场景
                          <span className="ml-2 text-slate-500">{formatCount(selectedRouterCount)} 条 · {formatBytes(selectedFileSize)}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button type="button" className="btn-ghost" onClick={() => setSelectedScenarios(dataset.scenarios.map(item => item.scenario))}>
                            全选
                          </button>
                          <button type="button" className="btn-ghost" onClick={() => setSelectedScenarios([])}>
                            清空
                          </button>
                        </div>
                      </div>

                      <div className="grid gap-2 sm:grid-cols-2 2xl:grid-cols-4">
                        {dataset.scenarios.map(scenario => {
                          const checked = selectedScenarios.includes(scenario.scenario)
                          return (
                            <button
                              key={scenario.scenario}
                              type="button"
                              onClick={() => toggleScenario(scenario.scenario)}
                              className={`training-select-chip ${
                                checked
                                  ? 'training-select-chip--active'
                                  : 'training-select-chip--idle'
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="min-w-0">
                                  <div className="truncate text-[14px] font-semibold">{scenario.scenario}</div>
                                  <div className="mt-0.5 truncate text-[12px] text-slate-400">
                                    {formatCount(scenario.router_count)} · {formatBytes(scenario.file_size_bytes)}
                                  </div>
                                </div>
                                {checked && (
                                  <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-lg border border-sky-500/30 bg-sky-500/20 text-sky-300">
                                    <Check size={13} />
                                  </span>
                                )}
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="training-surface text-sm text-slate-400">当前没有可用训练数据。</div>
                  )}
                </div>
              </div>

              <div className="training-card training-card--compact min-h-0">
                <div className="card-header">
                  <Cpu size={16} className="text-emerald-300" />
                  <SectionTitle title="GPU 分配" copy="选择一张空闲 GPU" />
                </div>
                <div className="training-card__body training-scroll list-scroll-lg">
                  <div className="grid gap-2 md:grid-cols-2">
                    {(gpus ?? []).map(gpu => {
                      const memoryRatio = gpu.memory_total_mib > 0 ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0
                      return (
                        <label
                          key={gpu.index}
                          className={`block rounded-2xl border p-3 transition-all ${gpuId === gpu.index ? 'border-emerald-500/40 bg-emerald-500/8' : 'border-slate-700/40 bg-slate-900/30'} ${gpu.available ? 'cursor-pointer' : 'cursor-not-allowed opacity-70'}`}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex min-w-0 items-center gap-2.5">
                              <input type="radio" className="mt-1" checked={gpuId === gpu.index} disabled={!gpu.available} onChange={() => setGpuId(gpu.index)} />
                              <div className="min-w-0">
                                <div className="text-[15px] font-semibold text-slate-100">GPU {gpu.index}</div>
                                <div className="mt-0.5 truncate text-[12px] text-slate-400">{gpu.name}</div>
                              </div>
                            </div>
                            <span className={gpu.available ? 'badge border border-emerald-500/20 bg-emerald-500/8 text-emerald-300' : 'badge border border-amber-500/20 bg-amber-500/12 text-amber-300'}>
                              {gpu.available ? '可用' : gpu.reason ?? '占用中'}
                            </span>
                          </div>
                          <div className="mt-2 training-stat-grid">
                            <div>
                              <div className="training-label">显存</div>
                              <div className="mt-0.5 text-[13px] text-slate-200">{gpuUsageLabel(gpu.memory_used_mib, gpu.memory_total_mib)}</div>
                              <UsageBar value={memoryRatio} />
                            </div>
                            <div>
                              <div className="training-label">利用率</div>
                              <div className="mt-0.5 text-[13px] text-slate-200">{gpu.utilization_gpu}%</div>
                              <UsageBar value={gpu.utilization_gpu} />
                            </div>
                          </div>
                        </label>
                      )
                    })}
                  </div>
                </div>
              </div>
            </section>

            <section className="training-stack">
              <div className="training-card training-card--compact">
                <div className="card-header">
                  <Sparkles size={16} className="text-violet-300" />
                  <SectionTitle title="训练参数" copy="全序列训练配置" />
                </div>
                <div className="grid gap-3 p-3 md:grid-cols-2 2xl:grid-cols-4">
                  <Field label="训练轮数">
                    <div className="flex items-center justify-between gap-3 rounded-2xl border border-slate-700/40 bg-slate-900/30 px-3 py-2.5">
                      <label className="flex items-center gap-2 text-[14px] text-slate-300">
                        <input type="checkbox" checked={infiniteEpochs} onChange={e => setInfiniteEpochs(e.target.checked)} />
                        无限训练
                      </label>
                      <input
                        className="input mono w-24 px-2.5 py-1.5 text-center"
                        value={infiniteEpochs ? '∞' : epochs}
                        onChange={e => setEpochs(e.target.value)}
                        disabled={infiniteEpochs}
                        placeholder="∞"
                      />
                    </div>
                  </Field>
                  <Field label="测试间隔">
                    <input className="input mono" value={evalInterval} onChange={e => setEvalInterval(e.target.value)} />
                  </Field>
                  <Field label="保留最近权重" note="每个 epoch 保存一份权重，仅保留最近 N 份；0 表示只保留 latest/final 权重。">
                    <input className="input mono" value={keepLastEpochs} onChange={e => setKeepLastEpochs(e.target.value)} />
                  </Field>
                  <Field label="训练批大小">
                    <input className="input mono" value={batchSize} onChange={e => setBatchSize(e.target.value)} />
                  </Field>
                  <Field label="测试批大小">
                    <input className="input mono" value={testBatchSize} onChange={e => setTestBatchSize(e.target.value)} />
                  </Field>
                  <Field label="学习率">
                    <input className="input mono" value={learningRate} onChange={e => setLearningRate(e.target.value)} />
                  </Field>
                  <Field label="权重衰减">
                    <input className="input mono" value={weightDecay} onChange={e => setWeightDecay(e.target.value)} />
                  </Field>
                  <Field label="数据加载线程">
                    <input className="input mono" value={numWorkers} onChange={e => setNumWorkers(e.target.value)} />
                  </Field>
                  <Field label="测试集比例">
                    <input className="input mono" value={testRatio} onChange={e => setTestRatio(e.target.value)} />
                  </Field>
                  <div className="md:col-span-2 2xl:col-span-4">
                    <Field label="恢复权重" note="只列出与当前大场景和子场景集合匹配的已完成任务。">
                      <select className="select" value={resumeFrom} onChange={e => setResumeFrom(e.target.value)}>
                        <option value="">不恢复，重新训练</option>
                        {checkpointCandidates.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </div>
                </div>
              </div>

              <div className="training-card training-card--compact">
                <div className="card-header">
                  <Database size={16} className="text-amber-300" />
                  <SectionTitle title="当前配置" copy="提交前核对" />
                </div>
                <div className="space-y-2 p-3">
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div className="training-surface--compact">
                      <div className="training-label">任务</div>
                      <div className="mt-1 truncate text-[15px] font-semibold text-slate-100">{jobName.trim() || '默认使用任务 ID'}</div>
                      <div className="mt-1 text-[12px] text-slate-500">{dataset?.simulator.toUpperCase() ?? '—'} · {selectedScenarios.length} 个场景</div>
                    </div>
                    <div className="training-surface--compact">
                      <div className="training-label">GPU</div>
                      <div className="mt-1 text-[15px] font-semibold text-slate-100">{gpuId != null ? `GPU ${gpuId}` : '未选择'}</div>
                      <div className="mt-1 truncate text-[12px] text-slate-500">{selectedGpu?.name ?? selectedGpu?.reason ?? '等待选择'}</div>
                    </div>
                  </div>
                  <div className="training-surface--compact">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="training-label">训练子场景</div>
                        <div className="mt-1 text-[13px] text-slate-400">{formatCount(selectedRouterCount)} 条 · {formatBytes(selectedFileSize)}</div>
                      </div>
                      <div className="text-[13px] font-semibold text-slate-200">{selectedScenarios.length}/{dataset?.scenarios.length ?? 0}</div>
                    </div>
                    <div className="mt-2 training-chip-grid">
                      {(selectedScenarios.length > 0 ? selectedScenarios : ['未选择']).map(item => (
                        <span key={item} className="training-chip max-w-[180px] truncate">{item}</span>
                      ))}
                    </div>
                  </div>
                  <div className="training-surface--compact">
                    <div className="training-label">训练配置</div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-[13px] text-slate-400">
                      <div>训练轮数： <span className="mono text-slate-200">{infiniteEpochs ? '∞' : epochs}</span></div>
                      <div>测试间隔： <span className="mono text-slate-200">{evalInterval}</span></div>
                      <div>保留权重： <span className="mono text-slate-200">{keepLastEpochs}</span></div>
                      <div>训练批量： <span className="mono text-slate-200">{batchSize}</span></div>
                      <div>测试批量： <span className="mono text-slate-200">{testBatchSize}</span></div>
                      <div>学习率： <span className="mono text-slate-200">{learningRate}</span></div>
                      <div>权重衰减： <span className="mono text-slate-200">{weightDecay}</span></div>
                      <div>线程： <span className="mono text-slate-200">{numWorkers}</span></div>
                      <div>测试比例： <span className="mono text-slate-200">{testRatio}</span></div>
                    </div>
                  </div>
                </div>
              </div>

            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
