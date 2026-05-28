import { useState } from 'react'
import useSWR from 'swr'
import { BookOpen, FileText, RefreshCw } from 'lucide-react'
import { api } from '../../lib/api'
import { EmptyState } from '../../shared/ui'
import { RegistryEntryCard } from '../components/registry/RegistryEntryCard'
import type { RegistryEntry } from '../components/registry/registryTypes'
import { ScenarioRowGrid } from '../components/registry/ScenarioRowGrid'

export default function RegistryPage() {
  const {
    data: registry,
    isLoading,
    mutate,
  } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })
  const [search, setSearch] = useState('')

  const handleSave = async (key: string, data: RegistryEntry | string) => {
    await api.updateRegistryEntry(
      key,
      typeof data === 'string' ? { scenario_description: data } : (data as Record<string, unknown>),
    )
    mutate()
  }
  const handleDelete = async (key: string) => {
    await api.deleteRegistryEntry(key)
    mutate()
  }

  // 新结构：顶层 key 是 simulator，value 含 scenarios 子字段
  const reg = (registry ?? {}) as Record<string, RegistryEntry & { scenarios?: Record<string, string> }>
  const simulators = Object.entries(reg)
    .filter(
      ([k]) =>
        !search ||
        k.toLowerCase().includes(search.toLowerCase()) ||
        Object.keys(reg[k]?.scenarios ?? {}).some(s => s.toLowerCase().includes(search.toLowerCase())),
    )
    .sort(([a], [b]) => a.localeCompare(b))

  const totalSimulators = Object.keys(reg).length
  const totalScenarios = Object.values(reg).reduce((s, e) => s + Object.keys(e?.scenarios ?? {}).length, 0)

  return (
    <div className="h-full min-h-0 overflow-y-auto overflow-x-hidden" style={{ WebkitOverflowScrolling: 'touch' }}>
      {/* 页头 */}
      <div className="page-header">
        <div className="w-7 h-7 rounded-lg bg-sky-500/20 border border-sky-500/30 flex items-center justify-center flex-shrink-0">
          <FileText size={14} className="text-sky-400" />
        </div>
        <span className="text-lg font-bold text-white">注册信息</span>
        <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/30">
          {totalSimulators} 个仿真器 · {totalScenarios} 个场景
        </span>
        <div className="flex-1" />
        <input
          type="text"
          className="input py-1.5 w-48 text-sm"
          placeholder="搜索…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <button className="btn-ghost py-1.5" onClick={() => mutate()}>
          <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* 列表 */}
      <div className="px-4 pb-4 pt-3 space-y-3">
        {isLoading && (
          <div className="flex items-center justify-center gap-2 py-16 text-slate-500">
            <RefreshCw size={15} className="animate-spin" />
            <span>加载注册信息…</span>
          </div>
        )}
        {!isLoading && simulators.length === 0 && (
          <EmptyState
            icon={BookOpen}
            title={totalSimulators === 0 ? '尚无注册信息' : `没有匹配 "${search}" 的记录`}
            description={totalSimulators === 0 ? '在「注册数据集」页面完成注册后，条目会显示在这里' : undefined}
          />
        )}
        {simulators.map(([simKey, simEntry]) => {
          const scenarios = simEntry?.scenarios ?? {}
          const scenarioCount = Object.keys(scenarios).length
          const sortedScenarios = Object.entries(scenarios).sort(([a], [b]) => a.localeCompare(b))
          // simulator 级条目（去掉 scenarios 子字段传给卡片）
          const { scenarios: _s, ...simFields } = simEntry ?? {}
          return (
            <div key={simKey} className="border border-slate-700/40 rounded-2xl overflow-hidden">
              {/* Simulator 级卡片 */}
              <RegistryEntryCard
                entryKey={simKey}
                entry={simFields as RegistryEntry}
                onSave={async (key, data) => handleSave(key, data)}
                onDelete={handleDelete}
              />
              {/* 场景列表 */}
              {scenarioCount > 0 && (
                <div className="border-t border-slate-700/30 bg-slate-900/20 min-h-0">
                  <div className="px-4 py-2 flex items-center gap-2 border-b border-slate-700/30">
                    <span className="label text-xs">场景描述</span>
                    <span className="badge bg-slate-700/50 text-slate-500 border border-slate-600/30 text-xs">
                      {scenarioCount} 个
                    </span>
                    <span className="text-xs text-slate-600">全部展开显示</span>
                  </div>
                  <div className="pb-2">
                    {sortedScenarios.map(([sc, desc]) => (
                      <ScenarioRowGrid
                        key={sc}
                        simulator={simKey}
                        scenario={sc}
                        description={desc}
                        onSave={async (key, d) => handleSave(key, d)}
                        onDelete={handleDelete}
                      />
                    ))}
                  </div>
                </div>
              )}
              {scenarioCount === 0 && (
                <div className="border-t border-slate-700/30 px-4 py-2.5 text-xs text-slate-600 bg-slate-900/20">
                  暂无场景描述 — 在「注册数据集」页面用「第二步：注册场景描述」添加
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
