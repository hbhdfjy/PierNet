import useSWR from 'swr'
import { Brain, FolderOpen, HardDrive, RefreshCw, CheckCircle } from 'lucide-react'

interface TrainedModel {
  name: string
  simulator: string
  path: string
  size_mb: number
  mtime: number
  has_config: boolean
}

interface ModelsResponse {
  models: TrainedModel[]
  total: number
}

const api = {
  getTrainedModels: (): Promise<ModelsResponse> => fetch('/api/text2comp/models').then(r => r.json()),
}

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function TrainedModelsPage() {
  const { data, error, isLoading, mutate } = useSWR<ModelsResponse>('trained-models', api.getTrainedModels)

  if (error) return <div className="p-6 text-red-400">加载失败: {error.message}</div>
  if (isLoading || !data) return <div className="p-6 text-slate-400">加载中...</div>

  return (
    <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(100vh-120px)]">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
          <HardDrive size={28} className="text-emerald-400" />
          训练模型文件 ({data.total})
        </h1>
        <button
          onClick={() => mutate()}
          className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800 text-slate-300"
        >
          <RefreshCw size={16} /> 刷新
        </button>
      </div>

      {data.models.length === 0 ? (
        <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-8 text-center text-slate-400">
          <FolderOpen size={48} className="mx-auto mb-3 opacity-50" />
          <p>暂无已训练的模型文件</p>
          <p className="text-sm mt-2">完成训练后，模型将保存在 artifacts/text2comp_models/ 目录</p>
        </div>
      ) : (
        <div className="rounded-lg bg-slate-800/50 border border-slate-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-900/50 border-b border-slate-600">
              <tr>
                <th className="px-4 py-3 text-left text-slate-300">任务名称</th>
                <th className="px-4 py-3 text-left text-slate-300">模拟器</th>
                <th className="px-4 py-3 text-left text-slate-300">大小</th>
                <th className="px-4 py-3 text-left text-slate-300">时间</th>
                <th className="px-4 py-3 text-left text-slate-300">路径</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {data.models.map(model => (
                <tr key={model.name + model.simulator} className="hover:bg-slate-900/30">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Brain size={16} className="text-emerald-400" />
                      <span className="text-slate-200 font-medium">{model.name}</span>
                      {model.has_config && <CheckCircle size={14} className="text-emerald-400" />}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-1 rounded bg-emerald-900/30 text-emerald-300 text-xs">
                      {model.simulator}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-400">{model.size_mb} MB</td>
                  <td className="px-4 py-3 text-slate-400">{formatTime(model.mtime)}</td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-slate-500 font-mono truncate max-w-xs block">{model.path}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="rounded-lg bg-slate-800/50 border border-slate-700 p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-3 flex items-center gap-2">
          <FolderOpen size={20} className="text-amber-400" /> 存储位置说明
        </h2>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">•</span>
            <span className="text-slate-400">模型存储:</span>
            <span className="font-mono text-slate-300">artifacts/text2comp_models/</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">•</span>
            <span className="text-slate-400">每个任务包含:</span>
            <span className="text-slate-300">final_model.pt, checkpoint_epoch_*.pt, config.json, train_log.jsonl</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">•</span>
            <span className="text-slate-400">日志位置:</span>
            <span className="font-mono text-slate-300">.runlogs/text2comp/{'{job_id}'}.log</span>
          </div>
        </div>
      </div>
    </div>
  )
}
