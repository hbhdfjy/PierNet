import { useState } from 'react'
import useSWR, { useSWRConfig } from 'swr'
import {
  Brain,
  Cpu,
  FolderOpen,
  PlayCircle,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Plus,
  Eye,
  Timer,
  TrendingDown,
} from 'lucide-react'

interface Dataset {
  path: string
  simulator: string
  scenario: string
  n_samples: number
}

interface Job {
  job_id: string
  name: string
  status: string
  simulator: string
  scenario: string
  gpu_id: number
  latest_epoch: number | null
  avg_loss: number | null
  global_step: number | null
  steps_per_epoch: number | null
  eta_seconds: number | null
  steps_per_sec: number | null
  run_dir: string | null
  log_path: string | null
  error_message: string | null
  started_at: number | null
  ended_at: number | null
}

interface Text2CompStatus {
  expert_models: Array<{ name: string; domain: string; output_dim: number; description: string }>
  datasets: string[]
  gpus: Array<{
    index: number
    name: string
    memory_used_mib: number
    memory_total_mib: number
    utilization_gpu: number
    available: boolean
  }>
  jobs: Job[]
  running_job_count: number
  completed_job_count: number
}

const api = {
  getText2CompStatus: () => fetch('/api/text2comp/status').then(r => r.json()),
  getDatasets: () => fetch('/api/text2comp/datasets').then(r => r.json()) as Promise<Dataset[]>,
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return '--'
  if (seconds < 60) return `${seconds.toFixed(0)}秒`
  if (seconds < 3600) return `${(seconds / 60).toFixed(0)}分${(seconds % 60).toFixed(0)}秒`
  return `${(seconds / 3600).toFixed(1)}小时`
}

function formatTime(timestamp: number | null): string {
  if (!timestamp) return '--'
  return new Date(timestamp * 1000).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export default function Text2CompPage() {
  const { mutate } = useSWRConfig()
  const {
    data: status,
    error,
    isLoading,
  } = useSWR<Text2CompStatus>('t2c-status', api.getText2CompStatus, { refreshInterval: 3000 })
  const { data: datasets } = useSWR<Dataset[]>('t2c-datasets', api.getDatasets)

  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({ simulator: '', dataset: '', gpu_id: 0, epochs: 100 })
  const [submitting, setSubmitting] = useState(false)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [expandedJob, setExpandedJob] = useState<string | null>(null)

  if (error) return <div className="p-6 text-red-400">加载失败: {error.message}</div>
  if (isLoading || !status) return <div className="p-6 text-slate-400">加载中...</div>

  const availableGpus = status.gpus.filter(g => g.available)
  const selectedModel = status.expert_models.find(m => m.name === formData.simulator)
  const matchingDatasets = datasets?.filter(d => d.simulator === formData.simulator) || []
  const selectedDataset = matchingDatasets.find(d => d.path === formData.dataset)

  const handleStartTrain = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.simulator || !formData.dataset) {
      setErrMsg('请选择模型和数据集')
      return
    }
    setSubmitting(true)
    setErrMsg(null)
    try {
      const res = await fetch('/api/text2comp/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          simulator: formData.simulator,
          scenario: selectedDataset?.scenario || formData.simulator,
          train_path: formData.dataset,
          output_dim: selectedModel?.output_dim || 128,
          epochs: formData.epochs,
          batch_size: 8,
          learning_rate: 0.00001,
          weight_decay: 0.01,
          gpu_id: formData.gpu_id,
        }),
      })
      const data = await res.json()
      if (!data.ok) setErrMsg(data.error || '启动失败')
      else {
        setShowForm(false)
        mutate('t2c-status')
      }
    } catch (e) {
      setErrMsg(String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const runningJobs = status.jobs.filter(j => j.status === 'running')
  const completedJobs = status.jobs.filter(j => j.status === 'done')
  const errorJobs = status.jobs.filter(j => j.status === 'error')

  return (
    <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(100vh-120px)]">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
          <Brain size={28} className="text-emerald-400" />
          文生计算模块训练
        </h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowForm(true)}
            className="flex items-center gap-2 px-3 py-2 rounded bg-emerald-800 text-emerald-100"
          >
            <Plus size={16} /> 新建训练
          </button>
          <button
            onClick={() => {
              mutate('t2c-status')
              mutate('t2c-datasets')
            }}
            className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800 text-slate-300"
          >
            <RefreshCw size={16} /> 刷新
          </button>
        </div>
      </div>

      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <form
            onSubmit={handleStartTrain}
            className="bg-slate-900 rounded-lg p-6 w-full max-w-md border border-slate-700 space-y-4"
          >
            <h2 className="text-xl font-semibold text-slate-200">新建训练</h2>
            <div>
              <label className="block text-sm text-slate-400 mb-1">专家模型</label>
              <select
                value={formData.simulator}
                onChange={e => setFormData({ ...formData, simulator: e.target.value, dataset: '' })}
                className="w-full rounded bg-slate-800 border border-slate-600 text-slate-200 p-2"
              >
                <option value="">选择模型...</option>
                {status.expert_models.map(m => (
                  <option key={m.name} value={m.name}>
                    {m.name} ({m.domain})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">数据集 ({matchingDatasets.length} 个可用)</label>
              <select
                value={formData.dataset}
                onChange={e => setFormData({ ...formData, dataset: e.target.value })}
                className="w-full rounded bg-slate-800 border border-slate-600 text-slate-200 p-2"
              >
                <option value="">选择数据集...</option>
                {matchingDatasets.map(d => (
                  <option key={d.path} value={d.path}>
                    {d.scenario} ({d.n_samples} 样本)
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">GPU</label>
              <select
                value={formData.gpu_id}
                onChange={e => setFormData({ ...formData, gpu_id: parseInt(e.target.value) })}
                className="w-full rounded bg-slate-800 border border-slate-600 text-slate-200 p-2"
              >
                {availableGpus.map(g => (
                  <option key={g.index} value={g.index}>
                    GPU {g.index} ({g.name})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">训练轮数</label>
              <input
                type="number"
                value={formData.epochs}
                onChange={e => setFormData({ ...formData, epochs: parseInt(e.target.value) || 100 })}
                className="w-full rounded bg-slate-800 border border-slate-600 text-slate-200 p-2"
              />
            </div>
            {errMsg && <div className="text-red-400 text-sm">{errMsg}</div>}
            <div className="flex gap-2 justify-end">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-3 py-2 rounded bg-slate-700 text-slate-300"
              >
                取消
              </button>
              <button type="submit" disabled={submitting} className="px-3 py-2 rounded bg-emerald-700 text-emerald-100">
                {submitting ? '启动中...' : '启动'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Running Jobs - 重点显示正在运行的训练 */}
      {runningJobs.length > 0 && (
        <div className="rounded-lg bg-emerald-900/30 border border-emerald-700 p-4">
          <h2 className="text-lg font-semibold text-emerald-200 mb-4 flex items-center gap-2">
            <PlayCircle size={20} className="animate-pulse" /> 正在训练 ({runningJobs.length})
          </h2>
          <div className="space-y-3">
            {runningJobs.map(j => (
              <div key={j.job_id} className="bg-emerald-900/40 rounded-lg p-4 border border-emerald-600">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-emerald-100 font-medium">{j.name || j.job_id}</span>
                    <span className="px-2 py-0.5 rounded text-xs bg-emerald-800 text-emerald-200 animate-pulse">
                      RUNNING
                    </span>
                  </div>
                  <button
                    onClick={() => setExpandedJob(expandedJob === j.job_id ? null : j.job_id)}
                    className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                  >
                    <Eye size={14} /> {expandedJob === j.job_id ? '收起' : '详情'}
                  </button>
                </div>

                {/* 训练进度条 */}
                <div className="mb-3">
                  <div className="flex items-center gap-2 text-sm text-emerald-300">
                    <TrendingDown size={14} />
                    <span>Epoch {j.latest_epoch || 0}</span>
                    <span className="text-emerald-500">·</span>
                    <span>Step {j.global_step || 0}</span>
                    <span className="text-emerald-500">·</span>
                    <span>Loss: {j.avg_loss?.toFixed(4) || '--'}</span>
                    <span className="text-emerald-500">·</span>
                    <span>{j.steps_per_sec?.toFixed(1) || '--'} steps/s</span>
                  </div>
                  {j.eta_seconds && (
                    <div className="flex items-center gap-1 text-xs text-emerald-400 mt-1">
                      <Timer size={12} />
                      <span>预计剩余: {formatDuration(j.eta_seconds)}</span>
                    </div>
                  )}
                </div>

                {/* GPU使用 */}
                <div className="text-xs text-emerald-500">
                  GPU {j.gpu_id} · 开始时间: {formatTime(j.started_at)}
                </div>

                {/* 展开详情 */}
                {expandedJob === j.job_id && (
                  <div className="mt-3 pt-3 border-t border-emerald-700 text-xs text-emerald-400 space-y-1">
                    <div>
                      <span className="text-emerald-500">运行目录:</span> {j.run_dir}
                    </div>
                    <div>
                      <span className="text-emerald-500">日志文件:</span> {j.log_path}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 专家模型列表 */}
      <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <FolderOpen size={20} className="text-amber-400" /> 专家模型
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {status.expert_models.map(m => (
            <button
              key={m.name}
              onClick={() => {
                setFormData({ ...formData, simulator: m.name, dataset: '' })
                setShowForm(true)
              }}
              className="rounded bg-slate-900/50 p-3 border border-slate-600 hover:border-emerald-600 text-left transition-colors"
            >
              <div className="font-medium text-slate-200">{m.name}</div>
              <div className="text-sm text-slate-400">{m.domain}</div>
              <div className="text-xs text-slate-500 mt-1">输出维度: {m.output_dim}</div>
              <div className="text-xs text-emerald-400 mt-1">点击训练 →</div>
            </button>
          ))}
        </div>
      </div>

      {/* GPU状态 */}
      <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <Cpu size={20} className="text-sky-400" /> GPU ({availableGpus.length}/{status.gpus.length} 可用)
        </h2>
        <div className="grid grid-cols-4 gap-2">
          {status.gpus.map(g => (
            <div
              key={g.index}
              className={`rounded p-2 border ${g.available ? 'bg-emerald-900/20 border-emerald-700' : 'bg-red-900/20 border-red-700'}`}
            >
              <div className="flex items-center gap-1">
                {g.available ? (
                  <CheckCircle2 size={14} className="text-emerald-400" />
                ) : (
                  <XCircle size={14} className="text-red-400" />
                )}
                <span className="text-xs text-slate-200">GPU {g.index}</span>
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {g.memory_used_mib}/{g.memory_total_mib} MiB
              </div>
              <div className="text-xs text-slate-500">利用率: {g.utilization_gpu}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* 完成和错误任务 */}
      <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <PlayCircle size={20} className="text-violet-400" /> 训练历史 (完成:{completedJobs.length}, 失败:
          {errorJobs.length})
        </h2>
        {status.jobs.filter(j => j.status !== 'running').length === 0 ? (
          <div className="text-slate-500 text-center py-4">暂无历史任务</div>
        ) : (
          <div className="space-y-2">
            {[...completedJobs, ...errorJobs].map(j => (
              <div
                key={j.job_id}
                onClick={() => setExpandedJob(expandedJob === j.job_id ? null : j.job_id)}
                className={`rounded p-3 border cursor-pointer transition-colors ${
                  j.status === 'done'
                    ? 'bg-sky-900/20 border-sky-700 hover:border-sky-500'
                    : j.status === 'error'
                      ? 'bg-red-900/20 border-red-700 hover:border-red-500'
                      : 'bg-slate-900/50 border-slate-600'
                }`}
              >
                <div className="flex justify-between">
                  <span className="text-slate-200 font-medium">{j.name || j.job_id}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-xs ${
                      j.status === 'done'
                        ? 'bg-sky-800 text-sky-200'
                        : j.status === 'error'
                          ? 'bg-red-800 text-red-200'
                          : 'bg-slate-700 text-slate-300'
                    }`}
                  >
                    {j.status}
                  </span>
                </div>

                {/* 完成任务显示结果 */}
                {j.status === 'done' && (
                  <div className="text-xs text-sky-400 mt-1 flex items-center gap-3">
                    <span>Epoch {j.latest_epoch}</span>
                    <span>·</span>
                    <span>Loss {j.avg_loss?.toFixed(4)}</span>
                    <span>·</span>
                    <span>耗时 {formatDuration(j.ended_at && j.started_at ? j.ended_at - j.started_at : null)}</span>
                  </div>
                )}

                {/* 错误任务显示原因 */}
                {j.status === 'error' && j.error_message && (
                  <div className="text-xs text-red-400 mt-1 truncate">{j.error_message}</div>
                )}

                {/* 展开详情 */}
                {expandedJob === j.job_id && (
                  <div className="mt-2 pt-2 border-t border-slate-600 text-xs space-y-1">
                    <div className="text-slate-400">
                      <span className="text-slate-500">Simulator:</span> {j.simulator}
                    </div>
                    <div className="text-slate-400">
                      <span className="text-slate-500">GPU:</span> {j.gpu_id}
                    </div>
                    <div className="text-slate-400">
                      <span className="text-slate-500">开始:</span> {formatTime(j.started_at)}
                    </div>
                    <div className="text-slate-400">
                      <span className="text-slate-500">结束:</span> {formatTime(j.ended_at)}
                    </div>
                    {j.run_dir && (
                      <div className="text-slate-400">
                        <span className="text-slate-500">输出目录:</span>
                        <span className="text-emerald-400 ml-1">{j.run_dir}</span>
                      </div>
                    )}
                    {j.log_path && (
                      <div className="text-slate-400">
                        <span className="text-slate-500">日志路径:</span>
                        <span className="text-amber-400 ml-1">{j.log_path}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 文件存储位置说明 */}
      <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <FolderOpen size={20} className="text-amber-400" /> 文件存储位置
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="bg-slate-900/50 rounded p-3 border border-slate-600">
            <div className="text-emerald-300 font-medium">训练数据</div>
            <div className="text-slate-400 mt-1 font-mono text-xs">data/text2comp/{'{simulator}'}.jsonl</div>
            <div className="text-xs text-slate-500">JSONL格式，包含prompt和label字段</div>
          </div>
          <div className="bg-slate-900/50 rounded p-3 border border-slate-600">
            <div className="text-sky-300 font-medium">模型输出</div>
            <div className="text-slate-400 mt-1 font-mono text-xs">
              artifacts/text2comp/{'{simulator}'}/runs/{'{job_id}'}/
            </div>
            <div className="text-xs text-slate-500">包含final_model.pt和checkpoints</div>
          </div>
          <div className="bg-slate-900/50 rounded p-3 border border-slate-600">
            <div className="text-amber-300 font-medium">训练日志</div>
            <div className="text-slate-400 mt-1 font-mono text-xs">.runlogs/text2comp/{'{job_id}'}.log</div>
            <div className="text-xs text-slate-500">详细的训练过程日志</div>
          </div>
          <div className="bg-slate-900/50 rounded p-3 border border-slate-600">
            <div className="text-violet-300 font-medium">训练配置</div>
            <div className="text-slate-400 mt-1 font-mono text-xs">
              artifacts/text2comp/{'{simulator}'}/runs/{'{job_id}'}/config.json
            </div>
            <div className="text-xs text-slate-500">训练参数配置记录</div>
          </div>
        </div>
      </div>
    </div>
  )
}
