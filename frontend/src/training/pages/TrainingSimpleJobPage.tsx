import { useEffect, useMemo, useState } from 'react'
import { Check, Database, Layers3, MousePointerClick, PlayCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import useSWR, { mutate } from 'swr'
import { api } from '../../lib/api'
import type { TrainingDatasetInfo } from '../../lib/types'
import { TrainingSectionTitle as SectionTitle } from '../components/common'
import { formatBytes, formatCount, trainingSimpleJobProgressPath } from '../shared'

function selectedStats(dataset: TrainingDatasetInfo | undefined, selectedScenarios: string[]) {
  if (!dataset) return { count: 0, size: 0 }
  const selected = new Set(selectedScenarios)
  return dataset.scenarios.reduce(
    (acc, scenario) => {
      if (selected.has(scenario.scenario)) {
        acc.count += scenario.router_count
        acc.size += scenario.file_size_bytes
      }
      return acc
    },
    { count: 0, size: 0 },
  )
}

export default function TrainingSimpleJobPage() {
  const navigate = useNavigate()
  const {
    data: datasets,
    error: datasetError,
    isLoading,
  } = useSWR('training-datasets', api.getTrainingDatasets, {
    refreshInterval: 10000,
    revalidateOnFocus: false,
  })
  const [simulator, setSimulator] = useState('')
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!simulator && datasets?.length) {
      setSimulator(datasets[0].simulator)
      setSelectedScenarios(datasets[0].scenarios.map(item => item.scenario))
    }
  }, [datasets, simulator])

  const dataset = useMemo(
    () => datasets?.find(item => item.simulator === simulator) ?? datasets?.[0],
    [datasets, simulator],
  )

  const stats = useMemo(() => selectedStats(dataset, selectedScenarios), [dataset, selectedScenarios])
  const canSubmit = Boolean(dataset && selectedScenarios.length > 0 && !submitting)

  const selectDataset = (nextSimulator: string) => {
    setSimulator(nextSimulator)
    const nextDataset = datasets?.find(item => item.simulator === nextSimulator)
    setSelectedScenarios(nextDataset?.scenarios.map(item => item.scenario) ?? [])
  }

  const toggleScenario = (scenario: string) => {
    setSelectedScenarios(current =>
      current.includes(scenario) ? current.filter(item => item !== scenario) : [...current, scenario],
    )
  }

  const submit = async () => {
    if (!dataset) return
    if (selectedScenarios.length === 0) {
      setError('请至少选择一个训练场景。')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const job = await api.createQuickTrainingJob({
        simulator: dataset.simulator,
        scenarios: selectedScenarios,
      })
      await Promise.all([mutate('training-overview'), mutate('training-jobs'), mutate('training-gpus')])
      navigate(trainingSimpleJobProgressPath(job.job_id))
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动一键训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-hero">
            <div className="training-simple-hero__copy">
              <div className="training-eyebrow">一键训练</div>
              <h1 className="training-simple-hero__title">选择场景后开始训练</h1>
              <p className="training-copy">只选择要覆盖的训练场景，任务启动后进入简化进度页。</p>
            </div>
            <button
              type="button"
              className="btn-primary training-simple-hero__action"
              onClick={submit}
              disabled={!canSubmit}
            >
              <PlayCircle size={15} />
              {submitting ? '启动中...' : '开始训练'}
            </button>
          </section>

          {(error || datasetError) && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              {error ?? `无法加载训练数据：${datasetError?.message}`}
            </div>
          )}

          <div className="training-simple-grid">
            <section className="training-card training-card--compact training-simple-panel">
              <div className="card-header">
                <Database size={16} className="text-sky-300" />
                <SectionTitle title="大场景" copy="选择本次训练的数据范围" />
              </div>
              <div className="training-card__body">
                {isLoading && !datasets ? (
                  <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                    {[0, 1, 2].map(item => (
                      <div key={item} className="skeleton h-24 rounded-xl" />
                    ))}
                  </div>
                ) : datasets?.length ? (
                  <div className="training-simple-dataset-grid">
                    {datasets.map(item => {
                      const active = item.simulator === dataset?.simulator
                      return (
                        <button
                          key={item.simulator}
                          type="button"
                          className={`training-simple-dataset ${active ? 'training-simple-dataset--active' : ''}`}
                          onClick={() => selectDataset(item.simulator)}
                        >
                          <div className="training-simple-dataset__top">
                            <div className="training-simple-dataset__name">{item.simulator}</div>
                            {active && <Check size={15} />}
                          </div>
                          <div className="training-simple-dataset__meta">
                            {item.scenarios.length} 个子场景 · {formatCount(item.total_count)} 条
                          </div>
                        </button>
                      )
                    })}
                  </div>
                ) : (
                  <div className="training-surface text-sm text-slate-400">当前没有可用训练数据。</div>
                )}
              </div>
            </section>

            <section className="training-card training-card--compact training-simple-panel training-simple-panel--scenarios">
              <div className="card-header">
                <Layers3 size={16} className="text-emerald-300" />
                <SectionTitle title="子场景" copy="勾选需要训练的场景" />
              </div>
              <div className="training-card__body space-y-3">
                <div className="training-simple-summary">
                  <div>
                    <div className="training-label">已选择</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">
                      {selectedScenarios.length}/{dataset?.scenarios.length ?? 0} 个场景
                    </div>
                  </div>
                  <div>
                    <div className="training-label">样本</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">{formatCount(stats.count)}</div>
                  </div>
                  <div>
                    <div className="training-label">数据量</div>
                    <div className="mt-1 text-[15px] font-semibold text-slate-100">{formatBytes(stats.size)}</div>
                  </div>
                </div>

                {dataset && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-[13px] text-slate-400">{dataset.simulator.toUpperCase()}</div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="btn-ghost"
                        onClick={() => setSelectedScenarios(dataset.scenarios.map(item => item.scenario))}
                      >
                        全选
                      </button>
                      <button type="button" className="btn-ghost" onClick={() => setSelectedScenarios([])}>
                        清空
                      </button>
                    </div>
                  </div>
                )}

                <div className="training-simple-scenario-grid training-scroll">
                  {dataset?.scenarios.map(scenario => {
                    const checked = selectedScenarios.includes(scenario.scenario)
                    return (
                      <button
                        key={scenario.scenario}
                        type="button"
                        onClick={() => toggleScenario(scenario.scenario)}
                        className={`training-simple-scenario ${checked ? 'training-simple-scenario--active' : ''}`}
                        title={`${scenario.scenario} · ${formatCount(scenario.router_count)} 条 · ${formatBytes(scenario.file_size_bytes)}`}
                      >
                        <span className="training-simple-scenario__check">{checked && <Check size={13} />}</span>
                        <span className="min-w-0 flex-1">
                          <span className="training-simple-scenario__name">{scenario.scenario}</span>
                          <span className="training-simple-scenario__meta">
                            {formatCount(scenario.router_count)} · {formatBytes(scenario.file_size_bytes)}
                          </span>
                        </span>
                      </button>
                    )
                  }) ?? <div className="training-surface text-sm text-slate-400">请选择大场景。</div>}
                </div>
              </div>
            </section>
          </div>

          <div className="training-simple-footer">
            <div className="flex min-w-0 items-center gap-2 text-[13px] text-slate-400">
              <MousePointerClick size={15} className="text-sky-300" />
              <span className="truncate">训练任务创建后会进入简化进度页。</span>
            </div>
            <button type="button" className="btn-primary" onClick={submit} disabled={!canSubmit}>
              <PlayCircle size={15} />
              {submitting ? '启动中...' : '开始训练'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
