import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileCode2,
  FolderOpen,
  Info,
  RefreshCw,
  ShieldCheck,
  UploadCloud,
  XCircle,
} from 'lucide-react'
import { api } from '../../lib/api'
import type { Hdf5DataFileInfo, Hdf5ValidationResult } from '../../lib/types'
import { cn, formatBytes, getSimulatorBadgeClass, SIMULATOR_LABELS } from '../../lib/utils'

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/

const SPEC_ITEMS = [
  'timeseries: 数值型 3 维 [N, C, T]，所有值必须有限',
  'params: 数值型 2 维 [N, P]，N 必须与 timeseries 一致',
  'param_names: 字符串 1 维 [P]，长度必须与 params 的 P 一致',
  '根属性 n_samples / n_channels / n_timesteps / n_params 必须存在且与 shape 完全一致',
]

function deriveScenarioName(fileName: string, simulator: string): string {
  const stem = fileName.replace(/\.(h5|hdf5)$/i, '')
  const prefix = `${simulator}_`
  return stem.startsWith(prefix) ? stem.slice(prefix.length) : stem
}

function ValidationPanel({ validation }: { validation: Hdf5ValidationResult }) {
  const hasMessages = validation.errors.length > 0 || validation.warnings.length > 0
  return (
    <div className={cn(
      'rounded-2xl border p-4',
      validation.valid
        ? 'border-emerald-500/25 bg-emerald-500/8'
        : 'border-red-500/25 bg-red-500/8',
    )}>
      <div className="flex items-center gap-2">
        {validation.valid ? <CheckCircle2 size={16} className="text-emerald-400" /> : <XCircle size={16} className="text-red-400" />}
        <div className="font-semibold text-slate-100">{validation.valid ? '预检通过' : '预检未通过，注册前需要修复'}</div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
        <Metric label="样本" value={validation.sample_count.toLocaleString()} />
        <Metric label="输出 shape" value={validation.output_shape ? validation.output_shape.join(' × ') : '—'} />
        <Metric label="参数 shape" value={validation.params_shape ? validation.params_shape.join(' × ') : '—'} />
        <Metric label="大小" value={formatBytes(validation.file_size_bytes)} />
      </div>
      {validation.param_names_preview.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {validation.param_names_preview.map(name => (
            <span key={name} className="rounded-lg border border-slate-700/40 bg-slate-900/35 px-2 py-1 font-mono text-xs text-slate-300">{name}</span>
          ))}
        </div>
      )}
      {hasMessages && (
        <div className="mt-3 space-y-1.5 text-sm">
          {validation.errors.map(msg => <div key={`err-${msg}`} className="text-red-300">错误：{msg}</div>)}
          {validation.warnings.map(msg => <div key={`warn-${msg}`} className="text-amber-300">警告：{msg}</div>)}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700/35 bg-slate-900/30 p-3">
      <div className="label mb-1 text-[11px]">{label}</div>
      <div className="font-mono text-sm font-semibold text-slate-100">{value}</div>
    </div>
  )
}

function FileStatus({ file }: { file: Hdf5DataFileInfo }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1 rounded-lg border px-2 py-0.5 text-xs font-semibold',
      file.valid
        ? 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300'
        : 'border-red-500/25 bg-red-500/10 text-red-300',
    )}>
      {file.valid ? <CheckCircle2 size={11} /> : <AlertTriangle size={11} />}
      {file.valid ? '合规' : '需修复'}
    </span>
  )
}

export default function DataUploadPage() {
  const navigate = useNavigate()
  const { data: files, isLoading, mutate } = useSWR<Hdf5DataFileInfo[]>('hdf5-data-files', () => api.listHdf5DataFiles())
  const { mutate: refreshSimulationScenarios } = useSWR('simulation-scenarios')

  const [simulator, setSimulator] = useState('modflow')
  const [scenario, setScenario] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [overwrite, setOverwrite] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Hdf5ValidationResult | null>(null)

  const cleanSimulator = simulator.trim()
  const cleanScenario = scenario.trim()
  const simulatorValid = NAME_RE.test(cleanSimulator)
  const scenarioValid = NAME_RE.test(cleanScenario)
  const targetPath = cleanSimulator && cleanScenario ? `data/${cleanSimulator}/${cleanSimulator}_${cleanScenario}.h5` : 'data/{big_scene}/{big_scene}_{scenario}.h5'

  const summary = useMemo(() => {
    const list = files ?? []
    return {
      total: list.length,
      valid: list.filter(item => item.valid).length,
      samples: list.reduce((sum, item) => sum + item.sample_count, 0),
      size: list.reduce((sum, item) => sum + item.file_size_bytes, 0),
    }
  }, [files])

  const handleFileChange = (nextFile: File | null) => {
    setFile(nextFile)
    setResult(null)
    setError(null)
    if (nextFile && !cleanScenario) {
      setScenario(deriveScenarioName(nextFile.name, cleanSimulator))
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('请选择 .h5 或 .hdf5 文件')
      return
    }
    if (!simulatorValid) {
      setError('仿真器/大场景名只能使用字母、数字、下划线或短横线，且必须以字母或数字开头')
      return
    }
    if (!scenarioValid) {
      setError('场景名只能使用字母、数字、下划线或短横线，且必须以字母或数字开头')
      return
    }
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const response = await api.uploadHdf5Data({ simulator: cleanSimulator, scenario: cleanScenario, file, overwrite })
      setResult(response.validation)
      await mutate()
      await refreshSimulationScenarios()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="page-shell">
      <div className="page-content p-6 space-y-5">
        <section className="rounded-[28px] border border-slate-700/35 bg-gradient-to-br from-slate-900/85 via-slate-900/60 to-sky-950/35 p-6 shadow-2xl shadow-slate-950/30">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-amber-300">
                Stage 1 Upload
              </div>
              <h1 className="mt-4 text-3xl font-black tracking-tight text-slate-50">上传物理数据</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                将外部仿真器/大场景的数据纳入 PiERN 合成链路。上传后即时预检；注册场景时会执行强校验，不合规数据不能进入模板生成。
              </p>
            </div>
            <div className="grid min-w-[420px] grid-cols-4 gap-2 max-lg:min-w-0 max-lg:w-full">
              <Metric label="文件" value={summary.total.toLocaleString()} />
              <Metric label="合规" value={summary.valid.toLocaleString()} />
              <Metric label="样本" value={summary.samples.toLocaleString()} />
              <Metric label="容量" value={formatBytes(summary.size)} />
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[0.95fr_1.05fr]">
          <section className="training-card overflow-hidden">
            <div className="card-header">
              <UploadCloud size={17} className="text-amber-400" />
              <div>
                <div className="training-panel-title">上传与落盘</div>
                <div className="training-panel-copy">目标路径固定为 data/&lt;big_scene&gt;/&lt;big_scene&gt;_&lt;scenario&gt;.h5</div>
              </div>
            </div>

            <div className="space-y-4 p-5">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-[260px_1fr]">
                <label className="space-y-1.5">
                  <span className="label">仿真器 / 大场景</span>
                  <input
                    className={cn('input w-full font-mono', simulator && !simulatorValid && 'border-red-500/50')}
                    value={simulator}
                    onChange={e => setSimulator(e.target.value)}
                    placeholder="new_big_scene"
                  />
                  <div className="text-xs text-slate-500">直接输入大场景命名空间，例如 modflow、simpeg 或 new_big_scene。</div>
                </label>
                <label className="space-y-1.5">
                  <span className="label">场景名</span>
                  <input
                    className={cn('input w-full font-mono', scenario && !scenarioValid && 'border-red-500/50')}
                    value={scenario}
                    onChange={e => setScenario(e.target.value)}
                    placeholder="new_scenario"
                  />
                </label>
              </div>

              <div className="rounded-2xl border border-slate-700/35 bg-slate-900/30 p-4">
                <div className="flex items-start gap-3">
                  <FileCode2 size={18} className="mt-1 text-sky-300" />
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-slate-100">HDF5 文件</div>
                    <input
                      type="file"
                      accept=".h5,.hdf5"
                      className="mt-3 block w-full text-sm text-slate-400 file:mr-3 file:rounded-xl file:border file:border-slate-700/40 file:bg-slate-800/70 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-slate-200 hover:file:bg-slate-700/70"
                      onChange={e => handleFileChange(e.target.files?.[0] ?? null)}
                    />
                    {file && (
                      <div className="mt-3 text-sm text-slate-400">
                        已选择 <span className="font-mono text-slate-200">{file.name}</span>，大小 {formatBytes(file.size)}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-sky-500/20 bg-sky-500/8 p-4">
                <div className="flex items-start gap-2">
                  <Info size={16} className="mt-0.5 text-sky-300" />
                  <div>
                    <div className="text-sm font-semibold text-slate-100">写入位置</div>
                    <div className="mt-1 break-all font-mono text-sm text-sky-300">{targetPath}</div>
                  </div>
                </div>
              </div>

              <label className="flex items-center gap-2 text-sm text-slate-400">
                <input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
                覆盖已存在的同名 HDF5 文件
              </label>

              {error && (
                <div className="rounded-2xl border border-red-500/25 bg-red-500/10 p-3 text-sm text-red-200">
                  {error}
                </div>
              )}

              {result && <ValidationPanel validation={result} />}

              <button
                className="btn-primary w-full justify-center py-3 text-sm"
                disabled={uploading || !file || !simulatorValid || !scenarioValid}
                onClick={handleUpload}
              >
                {uploading ? <RefreshCw size={15} className="animate-spin" /> : <UploadCloud size={15} />}
                {uploading ? '上传并预检中…' : '上传并预检'}
              </button>
            </div>
          </section>

          <section className="training-card overflow-hidden">
            <div className="card-header">
              <ShieldCheck size={17} className="text-emerald-400" />
              <div>
                <div className="training-panel-title">校验规范</div>
                <div className="training-panel-copy">上传页只提示预检结果，注册场景时按此规范强制校验。</div>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-3 p-5 md:grid-cols-2">
              {SPEC_ITEMS.map((item, index) => (
                <div key={item} className="rounded-2xl border border-slate-700/35 bg-slate-900/30 p-4">
                  <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
                    <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500/12 font-mono text-xs text-emerald-300">{index + 1}</span>
                    必需项
                  </div>
                  <div className="text-sm leading-6 text-slate-400">{item}</div>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="training-card overflow-hidden">
          <div className="card-header">
            <Database size={17} className="text-sky-400" />
            <div>
              <div className="training-panel-title">已接入 HDF5 文件</div>
              <div className="training-panel-copy">所有仿真器/大场景的数据都会从 data/ 下扫描。</div>
            </div>
            <div className="flex-1" />
            <button className="btn-ghost py-1.5 text-xs" onClick={() => mutate()}>
              <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />刷新
            </button>
          </div>

          <div className="list-table-scroll max-h-[420px]">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/40 bg-slate-800/40">
                  <th className="px-5 py-3 text-left label">状态</th>
                  <th className="px-5 py-3 text-left label">仿真器 / 大场景</th>
                  <th className="px-5 py-3 text-left label">场景</th>
                  <th className="px-5 py-3 text-right label">样本</th>
                  <th className="px-5 py-3 text-left label">输出</th>
                  <th className="px-5 py-3 text-right label">大小</th>
                  <th className="px-5 py-3 text-right label">修改时间</th>
                </tr>
              </thead>
              <tbody>
                {(files ?? []).map((item, index) => (
                  <tr key={item.path} className={cn('border-b border-slate-800/45 hover:bg-slate-700/18', index % 2 === 0 ? '' : 'bg-slate-800/10')}>
                    <td className="px-5 py-3"><FileStatus file={item} /></td>
                    <td className="px-5 py-3">
                      <span className={cn('badge border', getSimulatorBadgeClass(item.simulator))}>{SIMULATOR_LABELS[item.simulator] ?? item.simulator}</span>
                    </td>
                    <td className="px-5 py-3">
                      <div className="font-mono font-semibold text-slate-100">{item.scenario}</div>
                      <div className="mt-1 max-w-[520px] truncate font-mono text-xs text-slate-600">{item.path}</div>
                    </td>
                    <td className="px-5 py-3 text-right font-mono tabular-nums text-sky-300">{item.sample_count.toLocaleString()}</td>
                    <td className="px-5 py-3 font-mono text-slate-400">{item.output_shape ? item.output_shape.join(' × ') : '—'}</td>
                    <td className="px-5 py-3 text-right font-mono text-slate-400">{formatBytes(item.file_size_bytes)}</td>
                    <td className="px-5 py-3 text-right text-slate-500">{new Date(item.mtime * 1000).toLocaleString('zh-CN')}</td>
                  </tr>
                ))}
                {!isLoading && (!files || files.length === 0) && (
                  <tr>
                    <td colSpan={7} className="px-5 py-12 text-center text-slate-500">暂无 HDF5 文件</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  )
}
