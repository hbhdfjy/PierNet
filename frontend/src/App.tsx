import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import {
  Database, BarChart2, BookOpen, Cpu, FlaskConical, BookTemplate,
  KeyRound, FolderOpen, FileText, Zap, Sun, Moon, GitBranch, Network,
  Shuffle, ChevronRight,
} from 'lucide-react'
import { cn } from './lib/utils'
import { SeedContext } from './lib/seedContext'
import SampleViewer from './pages/SampleViewer'
import DatasetStats from './pages/DatasetStats'
import RegisterSimulator from './pages/RegisterSimulator'
import TemplateGenerator from './pages/TemplateGenerator'
import SampleFiller from './pages/SampleFiller'
import TemplateViewer from './pages/TemplateViewer'
import LLMConfigPage from './pages/LLMConfig'
import DataDirsConfig from './pages/DataDirsConfig'
import RegistryPage from './pages/RegistryPage'
import SimulationRunner from './pages/SimulationRunner'
import RouterDataBuilder from './pages/RouterDataBuilder'
import RouterViewer from './pages/RouterViewer'

// ── 主题 ──────────────────────────────────────────────────────────

type Theme = 'dark' | 'light'

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('piern-theme') as Theme) ?? 'dark'
  })
  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('light', theme === 'light')
    root.classList.toggle('dark',  theme === 'dark')
    localStorage.setItem('piern-theme', theme)
  }, [theme])
  return [theme, () => setTheme(t => t === 'dark' ? 'light' : 'dark')]
}

function useSeedState(): [number, (v: number) => void] {
  const [seed, setSeedRaw] = useState<number>(() => {
    const n = parseInt(localStorage.getItem('piern-seed') ?? '42', 10)
    return isNaN(n) ? 42 : Math.max(0, n)
  })
  const setSeed = (v: number) => { setSeedRaw(v); localStorage.setItem('piern-seed', String(v)) }
  return [seed, setSeed]
}

// ── 侧边栏常量 ────────────────────────────────────────────────────

// Stage 颜色配置
const STAGE_COLORS = {
  amber:   { active: 'text-amber-400',   bg: 'bg-amber-500/10',   dot: 'bg-amber-400',   border: 'border-amber-500/30'   },
  sky:     { active: 'text-sky-400',     bg: 'bg-sky-500/10',     dot: 'bg-sky-400',     border: 'border-sky-500/30'     },
  violet:  { active: 'text-violet-400',  bg: 'bg-violet-500/10',  dot: 'bg-violet-400',  border: 'border-violet-500/30'  },
  emerald: { active: 'text-emerald-400', bg: 'bg-emerald-500/10', dot: 'bg-emerald-400', border: 'border-emerald-500/30' },
  rose:    { active: 'text-rose-400',    bg: 'bg-rose-500/10',    dot: 'bg-rose-400',    border: 'border-rose-500/30'    },
} as const

type StageColor = keyof typeof STAGE_COLORS

// ── 侧边栏区块标题 ────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 px-2 mb-1">
      <span className="label text-[10px] whitespace-nowrap">{children}</span>
      <div className="flex-1 h-px" style={{ background: 'hsl(var(--border) / 0.4)' }} />
    </div>
  )
}

// ── 单个导航链接 ──────────────────────────────────────────────────

function NavItem({
  to,
  icon: Icon,
  label,
  color,
  step,
  rightIcon,
}: {
  to: string
  icon: React.ElementType
  label: string
  color?: StageColor
  step?: string
  rightIcon?: React.ElementType
}) {
  const c = color ? STAGE_COLORS[color] : null

  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        'group relative flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm',
        'transition-all duration-150 select-none',
        isActive
          ? cn('font-medium', c ? c.active : 'text-slate-200', c ? c.bg : 'bg-slate-700/40')
          : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/40',
      )}
    >
      {({ isActive }) => (
        <>
          {/* 左侧 active 指示条 */}
          {isActive && (
            <span
              className={cn(
                'absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-full',
                c ? c.dot : 'bg-slate-400',
              )}
            />
          )}

          {/* 图标区域 */}
          <div className={cn(
            'w-5 h-5 flex items-center justify-center flex-shrink-0 rounded-md transition-all',
            step
              ? cn(
                  'text-xs font-bold',
                  isActive
                    ? cn(c?.active ?? 'text-slate-200')
                    : 'text-slate-600',
                )
              : isActive
              ? cn(c?.active ?? 'text-slate-300')
              : 'text-slate-600 group-hover:text-slate-400',
          )}>
            {step ? step : <Icon size={14} />}
          </div>

          {/* 标签 */}
          <span className="flex-1 truncate text-[13px]">{label}</span>

          {/* 右侧图标（仅 inactive 时显示） */}
          {!isActive && rightIcon && (
            <span className="opacity-0 group-hover:opacity-40 transition-opacity flex-shrink-0">
              {React.createElement(rightIcon, { size: 11 })}
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}

// ── 主应用 ────────────────────────────────────────────────────────

// 需要手动 import React 因为 createElement
import React from 'react'

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [seed, setSeed] = useSeedState()
  const [seedInput, setSeedInput] = useState(String(seed))

  return (
    <SeedContext.Provider value={{ seed, setSeed }}>
      <div className="flex h-screen overflow-hidden" style={{ backgroundColor: 'hsl(var(--bg))' }}>

        {/* ════════════════════════════════════════════
            侧边栏
            ════════════════════════════════════════════ */}
        <aside
          className="w-52 flex-shrink-0 flex flex-col select-none"
          style={{
            backgroundColor: 'hsl(var(--bg-sub))',
            borderRight: '1px solid hsl(var(--border) / 0.5)',
          }}
        >
          {/* Logo */}
          <div
            className="flex items-center gap-3 px-4 py-4 flex-shrink-0"
            style={{ borderBottom: '1px solid hsl(var(--border) / 0.4)' }}
          >
            {/* Logo 图标 */}
            <div className="relative flex-shrink-0">
              <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, hsl(199 89% 42%), hsl(220 90% 48%))',
                  boxShadow: '0 2px 8px rgba(14,165,233,0.35)',
                }}>
                <span className="text-white font-black text-sm tracking-tight">P</span>
              </div>
              {/* 状态点 */}
              <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-emerald-400 border-2"
                style={{ borderColor: 'hsl(var(--bg-sub))' }} />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight leading-none"
                style={{ color: 'hsl(var(--text))' }}>
                PiERN
              </div>
              <div className="text-[11px] mt-0.5 leading-none"
                style={{ color: 'hsl(var(--text-faint))' }}>
                多模拟器数据集
              </div>
            </div>
          </div>

          {/* 导航 */}
          <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-4">

            {/* Stage 1 */}
            <div>
              <SectionLabel>Stage 1 · 物理仿真</SectionLabel>
              <NavItem to="/simulate" icon={Zap} label="仿真运行" color="amber" />
            </div>

            {/* Stage 2 */}
            <div>
              <SectionLabel>Stage 2 · 语言模板</SectionLabel>
              <div className="space-y-0.5">
                <NavItem to="/register"  icon={BookOpen} label="注册数据集" color="sky"    step="01" />
                <NavItem to="/templates" icon={Cpu}      label="模板生成"   color="violet" step="02" />
              </div>
            </div>

            {/* Stage 3 */}
            <div>
              <SectionLabel>Stage 3 · 样本填充</SectionLabel>
              <NavItem to="/fill" icon={FlaskConical} label="样本填充" color="emerald" />
            </div>

            {/* Stage 4 */}
            <div>
              <SectionLabel>Stage 4 · 路由数据</SectionLabel>
              <NavItem to="/router" icon={GitBranch} label="路由数据" color="rose" />
            </div>

            {/* 数据查看 */}
            <div>
              <SectionLabel>数据查看</SectionLabel>
              <div className="space-y-0.5">
                <NavItem to="/template-viewer" icon={BookTemplate} label="模板浏览"  rightIcon={ChevronRight} />
                <NavItem to="/samples"         icon={Database}     label="样本浏览"  rightIcon={ChevronRight} />
                <NavItem to="/router-viewer"   icon={Network}      label="路由浏览"  rightIcon={ChevronRight} />
                <NavItem to="/stats"           icon={BarChart2}    label="数据统计"  rightIcon={ChevronRight} />
              </div>
            </div>

            {/* 设置 */}
            <div>
              <SectionLabel>设置</SectionLabel>
              <div className="space-y-0.5">
                <NavItem to="/registry"   icon={FileText}   label="注册信息" />
                <NavItem to="/data-dirs"  icon={FolderOpen} label="数据目录" />
                <NavItem to="/llm-config" icon={KeyRound}   label="LLM 配置" color="amber" />
              </div>
            </div>
          </nav>

          {/* 底部：种子 + 主题 */}
          <div
            className="flex-shrink-0"
            style={{ borderTop: '1px solid hsl(var(--border) / 0.4)' }}
          >
            {/* 随机种子 */}
            <div className="px-3 pt-2.5 pb-2">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-1.5" style={{ color: 'hsl(var(--text-faint))' }}>
                  <Shuffle size={10} />
                  <span className="text-[11px]">全局随机种子</span>
                </div>
              </div>
              <input
                type="number"
                min={0}
                value={seedInput}
                onChange={e => {
                  setSeedInput(e.target.value)
                  const n = parseInt(e.target.value, 10)
                  if (!isNaN(n) && n >= 0) setSeed(n)
                }}
                onBlur={() => {
                  const n = parseInt(seedInput, 10)
                  const v = isNaN(n) ? 42 : Math.max(0, n)
                  setSeed(v); setSeedInput(String(v))
                }}
                className="w-full text-xs font-mono px-2.5 py-1.5 rounded-lg outline-none transition-all duration-150"
                style={{
                  backgroundColor: 'hsl(var(--surface2))',
                  border: '1px solid hsl(var(--border) / 0.6)',
                  color: 'hsl(var(--text))',
                }}
                onFocus={e => {
                  e.currentTarget.style.borderColor = 'hsl(199 89% 48% / 0.5)'
                  e.currentTarget.style.boxShadow = '0 0 0 2px hsl(199 89% 48% / 0.12)'
                }}
              />
            </div>

            {/* 主题 + 版本 */}
            <div className="px-3 pb-3 flex items-center justify-between">
              <span className="text-[11px] font-mono" style={{ color: 'hsl(var(--text-ghost))' }}>
                v2.0
              </span>
              <button
                onClick={toggleTheme}
                className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-all duration-150"
                style={{ color: 'hsl(var(--text-muted))' }}
                onMouseEnter={e => {
                  e.currentTarget.style.backgroundColor = 'hsl(var(--surface2))'
                  e.currentTarget.style.color = 'hsl(var(--text))'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.backgroundColor = 'transparent'
                  e.currentTarget.style.color = 'hsl(var(--text-muted))'
                }}
              >
                {theme === 'dark'
                  ? <><Sun size={12} /><span>日间</span></>
                  : <><Moon size={12} /><span>夜间</span></>
                }
              </button>
            </div>
          </div>
        </aside>

        {/* ════════════════════════════════════════════
            主内容区
            ════════════════════════════════════════════ */}
        <main
          className="flex-1 overflow-hidden flex flex-col"
          style={{ backgroundColor: 'hsl(var(--bg))' }}
        >
          <Routes>
            <Route path="/"               element={<Navigate to="/simulate" replace />} />
            <Route path="/simulate"       element={<SimulationRunner />} />
            <Route path="/register"       element={<RegisterSimulator />} />
            <Route path="/templates"      element={<TemplateGenerator />} />
            <Route path="/fill"           element={<SampleFiller />} />
            <Route path="/router"         element={<RouterDataBuilder />} />
            <Route path="/template-viewer" element={<TemplateViewer />} />
            <Route path="/samples"        element={<SampleViewer />} />
            <Route path="/router-viewer"  element={<RouterViewer />} />
            <Route path="/stats"          element={<DatasetStats />} />
            <Route path="/registry"       element={<RegistryPage />} />
            <Route path="/data-dirs"      element={<DataDirsConfig />} />
            <Route path="/llm-config"     element={<LLMConfigPage />} />
            <Route path="/monitor"        element={<Navigate to="/templates" replace />} />
            <Route path="/launch"         element={<Navigate to="/templates" replace />} />
            <Route path="*"               element={<Navigate to="/simulate" replace />} />
          </Routes>
        </main>
      </div>
    </SeedContext.Provider>
  )
}
