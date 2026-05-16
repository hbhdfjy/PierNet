import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  FileText,
  PauseCircle,
  RadioTower,
  RefreshCcw,
  Save,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useSWR, { useSWRConfig } from 'swr'
import { ConfirmDialog, MetricTile, StatusBadge, TruncatedText } from '../../shared/ui'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer } from 'recharts'
import {
  ChartCard,
  ChartEmpty,
  ChartHoverPanel,
  ChartTooltip,
  ChartXAxis,
  ChartYAxis,
  CheckpointList,
  MetaField,
  SectionTitle,
} from '../components/jobDetail/detailParts'
import {
  CHART_MARGIN,
  LEGEND_STYLE,
  buildActiveDot,
  buildChartHoverSnapshot,
  buildEpochSeries,
  buildNumericDomain,
  buildUnitMetricDomain,
  compactPathName,
  formatUnitDomainTick,
  inputRepresentationLabel,
  normalizeToDomain,
} from '../components/jobDetail/chartUtils'
import type { ChartHoverSnapshot, ChartMouseState, TrainingAxisMode } from '../components/jobDetail/chartUtils'
import { api } from '../../lib/api'
import type { TrainingCurvesResponse, TrainingJobDetail, TrainingLogResponse } from '../../lib/types'
import {
  formatDateTime,
  formatDuration,
  formatMetric,
  statusBadgeClass,
  statusLabel,
  trainingJobNotice,
} from '../shared'

export default function TrainingJobDetailPage() {
  const { jobId = '' } = useParams()
  const navigate = useNavigate()
  const { mutate } = useSWRConfig()
  const [trainingAxisMode, setTrainingAxisMode] = useState<TrainingAxisMode>('step')
  const [isStopping, setIsStopping] = useState(false)
  const [lossHover, setLossHover] = useState<ChartHoverSnapshot | null>(null)
  const [metricHover, setMetricHover] = useState<ChartHoverSnapshot | null>(null)
  const [scenarioHover, setScenarioHover] = useState<ChartHoverSnapshot | null>(null)
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const { data: job, error: jobError } = useSWR<TrainingJobDetail>(
    jobId ? `training-job-${jobId}` : null,
    () => api.getTrainingJob(jobId),
    {
      refreshInterval: current => {
        if (!current) return 2000
        if (current.status === 'starting' || current.status === 'stopping') return 2000
        return ['running', 'evaluating'].includes(current.status) ? 5000 : 0
      },
      revalidateOnFocus: false,
    },
  )

  const refreshInterval = useMemo(() => {
    if (!job) return 2000
    if (job.status === 'starting' || job.status === 'stopping') return 2000
    return ['running', 'evaluating'].includes(job.status) ? 5000 : 0
  }, [job])

  const notice = job ? trainingJobNotice(job) : null

  const { data: curves } = useSWR<TrainingCurvesResponse>(
    jobId ? `training-curves-${jobId}` : null,
    () => api.getTrainingCurves(jobId, 2000),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const { data: logs } = useSWR<TrainingLogResponse>(
    jobId ? `training-logs-${jobId}` : null,
    () => api.getTrainingLogs(jobId, 400),
    {
      refreshInterval,
      revalidateOnFocus: false,
    },
  )

  const trainingChart = useMemo(() => {
    const raw = curves?.training_points ?? []
    const epochSeries = curves?.training_epoch_points ?? buildEpochSeries(raw)
    return {
      xKey: trainingAxisMode === 'step' ? 'global_step' : 'epoch',
      data: trainingAxisMode === 'step' ? raw : epochSeries,
      subtitleSuffix: trainingAxisMode === 'step' ? '步骤' : '轮次',
    }
  }, [curves?.training_epoch_points, curves?.training_points, trainingAxisMode])

  const lossDomain = useMemo(
    () =>
      buildNumericDomain(
        trainingChart.data.map(point => point.avg_loss),
        {
          minClamp: 0,
          minPad: 0.00005,
          padRatio: 0.06,
          lowerQuantile: 0.08,
          upperQuantile: 0.92,
        },
      ),
    [trainingChart.data],
  )

  const testMetricDomain = useMemo(
    () =>
      buildUnitMetricDomain(
        (curves?.test_points ?? []).flatMap(point => [point.precision, point.recall, point.f1, point.pr_auc]),
      ),
    [curves?.test_points],
  )
  const testMetricPlotData = useMemo(() => {
    const domain = testMetricDomain ?? [0, 1]
    return (curves?.test_points ?? []).map(point => ({
      ...point,
      精确率: point.precision,
      召回率: point.recall,
      F1: point.f1,
      'PR-AUC': point.pr_auc,
      f1_plot: normalizeToDomain(point.f1, domain),
      pr_auc_plot: normalizeToDomain(point.pr_auc, domain),
      precision_plot: normalizeToDomain(point.precision, domain),
      recall_plot: normalizeToDomain(point.recall, domain),
    }))
  }, [curves?.test_points, testMetricDomain])

  const scenarioMetricData = useMemo(() => {
    const scenarioNames = new Set<string>()
    const rows = new Map<number, Record<string, number>>()
    const values: number[] = []

    for (const point of curves?.test_points ?? []) {
      const row = rows.get(point.epoch) ?? { epoch: point.epoch }
      for (const [scenario, metrics] of Object.entries(point.per_scenario)) {
        scenarioNames.add(scenario)
        const rawValue = metrics.f1
        if (rawValue === null || rawValue === undefined) {
          continue
        }
        const value = Number(rawValue)
        if (!Number.isFinite(value)) {
          continue
        }
        row[scenario] = value
        values.push(value)
      }
      rows.set(point.epoch, row)
    }

    const domain = buildUnitMetricDomain(values) ?? [0.9, 1.001]
    const names = Array.from(scenarioNames)
    const data = Array.from(rows.values())
      .sort((a, b) => Number(a.epoch) - Number(b.epoch))
      .map(row => {
        const nextRow: Record<string, number> = { ...row }
        for (const scenario of names) {
          const normalized = normalizeToDomain(row[scenario], domain)
          if (normalized !== undefined) {
            nextRow[`${scenario}__plot`] = normalized
          }
        }
        return nextRow
      })

    return {
      scenarioNames: names,
      data,
      domain,
    }
  }, [curves?.test_points])

  const stopJob = async () => {
    if (isStopping || job?.status === 'stopping') return
    setIsStopping(true)
    try {
      await api.stopTrainingJob(jobId)
      await Promise.all([
        mutate(`training-job-${jobId}`),
        mutate(`training-curves-${jobId}`),
        mutate(`training-logs-${jobId}`),
        mutate('training-jobs'),
        mutate('training-overview'),
        mutate('training-gpus'),
      ])
    } finally {
      setIsStopping(false)
    }
  }

  const deleteJob = async () => {
    if (!job || isDeleting) return
    setIsDeleting(true)
    try {
      await api.deleteTrainingJob(job.job_id)
      await Promise.all([
        mutate('training-jobs'),
        mutate('training-overview'),
        mutate('training-gpus'),
        mutate(`training-job-${job.job_id}`),
        mutate(`training-curves-${job.job_id}`),
        mutate(`training-logs-${job.job_id}`),
      ])
      navigate('/training/jobs')
    } finally {
      setIsDeleting(false)
      setConfirmDeleteOpen(false)
    }
  }

  if (!jobId) {
    return (
      <div className="training-page">
        <div className="training-page__body">
          <div className="training-surface text-[15px] text-slate-400">缺少训练任务 ID。</div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-page">
      <div className="training-page__body">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="training-eyebrow">任务详情</div>
                <h1
                  className="pretty-tooltip mt-2 max-w-5xl truncate text-[1.55rem] font-semibold tracking-tight text-white xl:text-[1.75rem]"
                  data-tooltip={job?.name ?? jobId}
                >
                  {job?.name ?? jobId}
                </h1>
                <div className="mono mt-1 text-[12px] text-slate-500">{jobId}</div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button type="button" className="btn-ghost" onClick={() => navigate('/training/jobs')}>
                  <ArrowLeft size={14} />
                  返回任务列表
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => {
                    mutate(`training-job-${jobId}`)
                    mutate(`training-curves-${jobId}`)
                    mutate(`training-logs-${jobId}`)
                  }}
                >
                  <RefreshCcw size={14} />
                  刷新
                </button>
                {job && ['starting', 'running', 'evaluating', 'stopping'].includes(job.status) ? (
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={stopJob}
                    disabled={isStopping || job.status === 'stopping'}
                  >
                    <PauseCircle size={14} />
                    {isStopping || job.status === 'stopping' ? '\u7ec8\u6b62\u4e2d...' : '\u7ec8\u6b62\u8bad\u7ec3'}
                  </button>
                ) : job ? (
                  <button type="button" className="btn-ghost" onClick={() => setConfirmDeleteOpen(true)}>
                    <Trash2 size={14} />
                    删除任务
                  </button>
                ) : null}
              </div>
            </div>

            {job && (
              <div className="mt-4 training-kpi-grid">
                <MetricTile
                  label="状态"
                  value={statusLabel(job.status)}
                  note={`GPU ${job.gpu_id} / PID ${job.pid ?? '—'}`}
                  icon={<RadioTower size={16} />}
                />
                <MetricTile
                  label="轮次 / 步数"
                  value={`${job.latest_epoch ?? '—'} / ${job.latest_step ?? '—'}`}
                  note={`全局步数 ${job.global_step ?? '—'}`}
                  icon={<BarChart3 size={16} />}
                />
                <MetricTile
                  label="损失"
                  value={formatMetric(job.avg_loss, 6)}
                  note={`${formatMetric(job.steps_per_sec, 2)} 步/秒`}
                  icon={<ActivityIcon />}
                />
                <MetricTile
                  label="最近 F1"
                  value={formatMetric(job.latest_metrics?.f1, 4)}
                  note={`PR-AUC ${formatMetric(job.latest_metrics?.pr_auc, 4)}`}
                  icon={<Save size={16} />}
                />
                <MetricTile
                  label="预计剩余"
                  value={formatDuration(job.eta_seconds)}
                  note={`创建于 ${formatDateTime(job.created_at)}`}
                  icon={<RefreshCcw size={16} />}
                />
              </div>
            )}
          </section>

          {jobError && (
            <div className="card border border-rose-500/20 bg-rose-500/8 p-4 text-sm text-rose-300">
              加载训练任务失败：{jobError.message}
            </div>
          )}

          {job && (
            <>
              <div className="grid gap-4 xl:grid-cols-[1.08fr_0.92fr]">
                <section className="training-card">
                  <div className="card-header">
                    <RadioTower size={16} className="text-violet-300" />
                    <SectionTitle title="任务摘要" copy="配置与路径" />
                  </div>
                  <div className="training-card__body">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="training-surface">
                        <div className="training-meta-grid">
                          <MetaField label="任务名称" value={job.name} />
                          <MetaField
                            label="状态"
                            value={
                              <StatusBadge className={statusBadgeClass(job.status)}>
                                {statusLabel(job.status)}
                              </StatusBadge>
                            }
                          />
                          <MetaField
                            label="训练数据"
                            value={`${job.simulator.toUpperCase()} / ${job.scenarios.join(', ')}`}
                          />
                          <MetaField label="测试比例" value={formatMetric(job.config.test_ratio, 2)} />
                          <MetaField label="评测间隔" value={`${job.config.eval_interval} 轮`} />
                          <MetaField label="总轮数" value={job.config.epochs === 0 ? '∞' : job.config.epochs} />
                          <MetaField label="保留权重" value={`${job.config.keep_last_epochs ?? 5} 个轮次`} />
                          <MetaField label="随机种子" value={job.config.seed ?? 42} mono />
                        </div>
                      </div>
                      <div className="training-surface">
                        <div className="training-meta-grid">
                          <MetaField label="训练批大小" value={job.config.batch_size} mono />
                          <MetaField label="测试批大小" value={job.config.test_batch_size} mono />
                          <MetaField label="加载线程" value={job.config.num_workers} mono />
                          <MetaField
                            label="预处理线程"
                            value={job.config.prepare_workers ?? job.config.num_workers}
                            mono
                          />
                          <MetaField label="学习率" value={job.config.learning_rate} mono />
                          <MetaField label="权重衰减" value={job.config.weight_decay} mono />
                          <MetaField label="恢复训练" value={job.config.resume_from ? '是' : '否'} />
                          <MetaField
                            label="输入表示"
                            value={inputRepresentationLabel(job.config.input_representation)}
                            title={job.config.input_representation ?? 'pretrained_embeddings'}
                          />
                          <MetaField
                            label="嵌入模型"
                            value={compactPathName(job.config.embedding_model || job.config.embedding_tokenizer)}
                            mono
                            title={job.config.embedding_model || job.config.embedding_tokenizer || undefined}
                          />
                        </div>
                      </div>
                      <div className="training-surface md:col-span-2">
                        <div className="grid gap-3 md:grid-cols-2">
                          <MetaField
                            label="运行目录"
                            value={<TruncatedText value={job.run_dir} className="mono text-[13px] text-slate-200" />}
                            title={job.run_dir}
                          />
                          <MetaField
                            label="日志文件"
                            value={<TruncatedText value={job.log_path} className="mono text-[13px] text-slate-200" />}
                            title={job.log_path}
                          />
                        </div>
                        {notice && (
                          <div
                            className={`mt-3 flex items-start gap-2 rounded-xl border px-3 py-2 text-sm ${notice.tone === 'amber' ? 'border-amber-400/25 bg-amber-400/8 text-amber-200' : 'border-rose-500/20 bg-rose-500/8 text-rose-300'}`}
                          >
                            <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
                            <span>{notice.message}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <section className="training-card min-h-0">
                  <div className="card-header">
                    <Save size={16} className="text-emerald-300" />
                    <SectionTitle title="权重文件" copy="已保存的模型权重" />
                  </div>
                  <div className="training-card__body training-scroll list-scroll-lg">
                    <CheckpointList checkpoints={curves?.checkpoints ?? job.checkpoints} />
                  </div>
                </section>
              </div>

              <div className="grid gap-4">
                <ChartCard
                  title="训练损失"
                  subtitle={`平均损失 / ${trainingChart.subtitleSuffix}`}
                  overlay={<ChartHoverPanel snapshot={lossHover} />}
                  actions={
                    <div className="training-segmented">
                      <button
                        type="button"
                        className={`training-segmented__button ${trainingAxisMode === 'step' ? 'training-segmented__button--active' : ''}`}
                        onClick={() => setTrainingAxisMode('step')}
                      >
                        步骤
                      </button>
                      <button
                        type="button"
                        className={`training-segmented__button ${trainingAxisMode === 'epoch' ? 'training-segmented__button--active' : ''}`}
                        onClick={() => setTrainingAxisMode('epoch')}
                      >
                        轮次
                      </button>
                    </div>
                  }
                >
                  {trainingChart.data.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={trainingChart.data}
                        margin={CHART_MARGIN}
                        onMouseMove={(state: ChartMouseState) =>
                          setLossHover(buildChartHoverSnapshot(trainingAxisMode === 'step' ? '步骤' : '轮次', state))
                        }
                        onMouseLeave={() => setLossHover(null)}
                      >
                        <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                        <ChartXAxis
                          dataKey={trainingChart.xKey}
                          type={trainingAxisMode === 'step' ? 'number' : 'category'}
                          allowDecimals={trainingAxisMode === 'step'}
                        />
                        <ChartYAxis
                          domain={lossDomain}
                          tickCount={5}
                          width={52}
                          tickFormatter={(value: number) => value.toFixed(value >= 1 ? 3 : 4)}
                          allowDataOverflow
                        />
                        <ChartTooltip axisLabel={trainingAxisMode === 'step' ? '步骤' : '轮次'} />
                        <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                        <Line
                          type="monotone"
                          dataKey="avg_loss"
                          name="平均损失"
                          stroke="#38bdf8"
                          dot={false}
                          activeDot={buildActiveDot('#38bdf8')}
                          strokeWidth={2.25}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <ChartEmpty message="当前还没有训练曲线点。" />
                  )}
                </ChartCard>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <ChartCard title="测试指标" subtitle="精确率 / 召回率 / F1 / PR-AUC">
                  {curves?.test_points?.length ? (
                    <>
                      <ChartHoverPanel snapshot={metricHover} />
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={testMetricPlotData}
                          margin={CHART_MARGIN}
                          onMouseMove={(state: ChartMouseState) =>
                            setMetricHover(buildChartHoverSnapshot('轮次', state))
                          }
                          onMouseLeave={() => setMetricHover(null)}
                        >
                          <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                          <ChartXAxis dataKey="epoch" type="number" allowDecimals={false} />
                          <ChartYAxis
                            domain={[0, 1]}
                            tickCount={5}
                            width={56}
                            tickFormatter={(value: number) => formatUnitDomainTick(value, testMetricDomain ?? [0, 1])}
                            allowDataOverflow
                          />
                          <ChartTooltip axisLabel="轮次" />
                          <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                          <Line
                            type="monotone"
                            dataKey="precision_plot"
                            name="精确率"
                            stroke="#38bdf8"
                            dot={false}
                            activeDot={buildActiveDot('#38bdf8')}
                            strokeWidth={2.1}
                            isAnimationActive={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="recall_plot"
                            name="召回率"
                            stroke="#f59e0b"
                            dot={false}
                            activeDot={buildActiveDot('#f59e0b')}
                            strokeWidth={2.1}
                            isAnimationActive={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="f1_plot"
                            name="F1"
                            stroke="#34d399"
                            dot={false}
                            activeDot={buildActiveDot('#34d399')}
                            strokeWidth={2.1}
                            isAnimationActive={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="pr_auc_plot"
                            name="PR-AUC"
                            stroke="#a78bfa"
                            dot={false}
                            activeDot={buildActiveDot('#a78bfa')}
                            strokeWidth={2.1}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </>
                  ) : (
                    <ChartEmpty message="当前还没有测试点，需要等到测试间隔触发。" />
                  )}
                </ChartCard>

                <ChartCard title="分场景 F1" subtitle="每个子场景单独观察">
                  {scenarioMetricData.scenarioNames.length ? (
                    <>
                      <ChartHoverPanel snapshot={scenarioHover} variant="emphasis" />
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={scenarioMetricData.data}
                          margin={CHART_MARGIN}
                          onMouseMove={(state: ChartMouseState) =>
                            setScenarioHover(buildChartHoverSnapshot('轮次', state))
                          }
                          onMouseLeave={() => setScenarioHover(null)}
                        >
                          <CartesianGrid stroke="rgba(51,65,85,0.26)" strokeDasharray="3 4" vertical={false} />
                          <ChartXAxis dataKey="epoch" type="number" allowDecimals={false} />
                          <ChartYAxis
                            domain={[0, 1]}
                            tickCount={5}
                            width={56}
                            tickFormatter={(value: number) => formatUnitDomainTick(value, scenarioMetricData.domain)}
                            allowDataOverflow
                          />
                          <ChartTooltip axisLabel="轮次" />
                          <Legend wrapperStyle={LEGEND_STYLE} iconType="circle" />
                          {scenarioMetricData.scenarioNames.map((scenario, index) => {
                            const colors = ['#38bdf8', '#34d399', '#f59e0b', '#f472b6', '#a78bfa', '#fb7185']
                            return (
                              <Line
                                key={scenario}
                                dataKey={`${scenario}__plot`}
                                name={scenario}
                                stroke={colors[index % colors.length]}
                                dot={false}
                                activeDot={{ ...buildActiveDot(colors[index % colors.length]), r: 6 }}
                                strokeWidth={2.6}
                                isAnimationActive={false}
                              />
                            )
                          })}
                        </LineChart>
                      </ResponsiveContainer>
                    </>
                  ) : (
                    <ChartEmpty message="当前还没有分场景测试曲线。" />
                  )}
                </ChartCard>
              </div>

              <section className="training-card min-h-0">
                <div className="card-header">
                  <FileText size={16} className="text-amber-300" />
                  <SectionTitle title="训练日志" copy="最近 400 行输出" />
                </div>
                <div className="training-card__body min-h-0">
                  <div className="grid min-w-0 gap-3 xl:grid-cols-[minmax(260px,320px)_minmax(0,1fr)]">
                    <div className="training-surface--dense min-w-0">
                      <div className="training-panel-title whitespace-nowrap">日志摘要</div>
                      <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                        <MetaField label="总行数" value={logs?.lines.length ?? 0} mono />
                        <MetaField label="状态" value={statusLabel(job.status)} />
                        <MetaField label="最近 epoch" value={job.latest_epoch ?? '—'} mono />
                        <MetaField label="最近 step" value={job.latest_step ?? '—'} mono />
                      </div>
                    </div>
                    <div className="min-w-0 rounded-xl border border-slate-700/40 bg-slate-950/72 p-3.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
                      <div className="mb-3 flex min-w-0 items-center justify-between gap-3">
                        <div className="min-w-0 flex-shrink-0">
                          <div className="training-panel-title whitespace-nowrap">终端输出</div>
                          <div className="training-panel-copy">自动刷新最新内容</div>
                        </div>
                        <div
                          className="mono min-w-0 truncate text-right text-[11px] text-slate-500"
                          title={job.log_path}
                        >
                          {job.log_path}
                        </div>
                      </div>
                      <pre className="list-scroll-xl min-w-0 whitespace-pre-wrap break-words [overflow-wrap:anywhere] rounded-xl border border-slate-800/70 bg-slate-950/65 px-3 py-2.5 text-[12px] leading-5 text-slate-300">
                        {(logs?.lines ?? []).join('\n') || '暂无日志输出。'}
                      </pre>
                    </div>
                  </div>
                </div>
              </section>
            </>
          )}
          {job && (
            <ConfirmDialog
              open={confirmDeleteOpen}
              title="删除训练任务"
              description={
                <>
                  将彻底删除 <span className="font-semibold text-slate-100">{job.name}</span>{' '}
                  的任务记录、运行目录、权重、曲线和日志。共享预处理缓存会保留。
                </>
              }
              confirmLabel="删除"
              danger
              busy={isDeleting}
              onCancel={() => setConfirmDeleteOpen(false)}
              onConfirm={deleteJob}
            />
          )}
        </div>
      </div>
    </div>
  )
}

function ActivityIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 12h3l2-5 4 10 2-5h5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
