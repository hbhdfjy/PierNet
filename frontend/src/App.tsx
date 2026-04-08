import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { Database, BarChart2, BookOpen, Cpu, FlaskConical, BookTemplate, KeyRound, FolderOpen, FileText, Zap, Sun, Moon } from 'lucide-react'
import { cn } from './lib/utils'
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

// ── 主题 ────────────────────────────────────────────────────────

type Theme = 'dark' | 'light'

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('piern-theme') as Theme | null
    return saved ?? 'dark'
  })

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'light') {
      root.classList.add('light')
      root.classList.remove('dark')
    } else {
      root.classList.remove('light')
      root.classList.add('dark')
    }
    localStorage.setItem('piern-theme', theme)
  }, [theme])

  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')
  return [theme, toggle]
}

// ── 侧边栏导航项定义 ──────────────────────────────────────────────

const STAGE2_STEPS = [
  { to: '/register',  icon: BookOpen, label: '注册数据集', step: '01', color: 'sky'    },
  { to: '/templates', icon: Cpu,      label: '模板生成',   step: '02', color: 'violet' },
] as const

const DATA_ITEMS = [
  { to: '/template-viewer', icon: BookTemplate, label: '模板浏览' },
  { to: '/samples',         icon: Database,     label: '样本浏览' },
  { to: '/stats',           icon: BarChart2,    label: '数据统计' },
] as const

const SETTINGS_ITEMS = [
  { to: '/registry',   icon: FileText,   label: '注册信息', color: 'sky'   },
  { to: '/data-dirs',  icon: FolderOpen, label: '数据目录', color: 'sky'   },
  { to: '/llm-config', icon: KeyRound,   label: 'LLM 配置', color: 'amber' },
] as const

type StepColor = 'sky' | 'violet' | 'emerald'
const STEP_COLORS: Record<StepColor, { active: string; dot: string }> = {
  sky:     { active: 'bg-sky-500/15 text-sky-600 dark:text-sky-300',           dot: 'bg-sky-500'     },
  violet:  { active: 'bg-violet-500/15 text-violet-600 dark:text-violet-300',  dot: 'bg-violet-500'  },
  emerald: { active: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300', dot: 'bg-emerald-500' },
}

// ── 侧边栏区块标题 ────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 mb-1 flex items-center gap-1.5">
      <span className="text-[10px] font-bold uppercase tracking-widest whitespace-nowrap text-slate-400 dark:text-slate-600">{children}</span>
      <div className="flex-1 h-px bg-slate-200 dark:bg-slate-700/30" />
    </div>
  )
}

// ── 通用导航链接 ──────────────────────────────────────────────────

function NavItem({
  to, icon: Icon, label,
  activeClass = 'bg-slate-200/80 dark:bg-slate-700/50 text-slate-900 dark:text-slate-200',
}: {
  to: string; icon: React.ElementType; label: string; activeClass?: string
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        'flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all duration-150',
        isActive
          ? activeClass + ' font-medium'
          : 'text-slate-500 dark:text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-700 dark:hover:text-slate-300',
      )}
    >
      <Icon size={13} className="flex-shrink-0 opacity-70" />
      <span className="truncate">{label}</span>
    </NavLink>
  )
}

// ── 主应用 ────────────────────────────────────────────────────────

export default function App() {
  const [theme, toggleTheme] = useTheme()

  return (
    <div className="flex h-screen overflow-hidden bg-[hsl(var(--bg))]">

      {/* ── 侧边栏 ── */}
      <aside className="w-48 flex-shrink-0 bg-[hsl(var(--bg-sub))] border-r border-[hsl(var(--border)/0.6)] flex flex-col select-none">

        {/* Logo */}
        <div className="px-3 py-3.5 border-b border-[hsl(var(--border)/0.5)]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-sky-900/30">
              <span className="text-white font-black text-sm">P</span>
            </div>
            <div>
              <div className="text-sm font-bold tracking-wide leading-none text-[hsl(var(--text))]">PiERN</div>
              <div className="text-[10px] mt-0.5 leading-none text-[hsl(var(--text-faint))]">多模拟器数据集</div>
            </div>
          </div>
        </div>

        {/* 导航主体 */}
        <nav className="flex-1 overflow-y-auto px-2 py-2.5 space-y-3">

          {/* Stage 1 */}
          <div>
            <SectionLabel>Stage 1 · 物理仿真</SectionLabel>
            <NavLink
              to="/simulate"
              className={({ isActive }) => cn(
                'flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all duration-150',
                isActive
                  ? 'bg-amber-500/15 text-amber-600 dark:text-amber-300 font-medium'
                  : 'text-slate-500 dark:text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-700 dark:hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-5 h-5 rounded flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'bg-amber-500/20 text-amber-500 dark:text-amber-400' : 'text-slate-400 dark:text-slate-600',
                  )}>
                    <Zap size={12} />
                  </div>
                  <span className="flex-1 truncate">仿真运行</span>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />}
                </>
              )}
            </NavLink>
          </div>

          {/* Stage 2 */}
          <div>
            <SectionLabel>Stage 2 · 语言模板</SectionLabel>
            <div className="space-y-0.5">
              {STAGE2_STEPS.map(({ to, icon: Icon, label, step, color }) => {
                const colors = STEP_COLORS[color]
                return (
                  <NavLink
                    key={to}
                    to={to}
                    className={({ isActive }) => cn(
                      'flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all duration-150',
                      isActive
                        ? `${colors.active} font-medium`
                        : 'text-slate-500 dark:text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-700 dark:hover:text-slate-300',
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        <div className={cn(
                          'w-5 h-5 rounded flex items-center justify-center flex-shrink-0 text-[10px] font-bold transition-all',
                          isActive ? 'opacity-90' : 'text-slate-400 dark:text-slate-600',
                        )}>
                          {step}
                        </div>
                        <span className="flex-1 truncate">{label}</span>
                        {isActive
                          ? <div className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', colors.dot)} />
                          : <Icon size={11} className="flex-shrink-0 opacity-30" />
                        }
                      </>
                    )}
                  </NavLink>
                )
              })}
            </div>
          </div>

          {/* Stage 3 */}
          <div>
            <SectionLabel>Stage 3 · 样本填充</SectionLabel>
            <NavLink
              to="/fill"
              className={({ isActive }) => cn(
                'flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all duration-150',
                isActive
                  ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 font-medium'
                  : 'text-slate-500 dark:text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-700 dark:hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-5 h-5 rounded flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'text-emerald-500 dark:text-emerald-400' : 'text-slate-400 dark:text-slate-600',
                  )}>
                    <FlaskConical size={12} />
                  </div>
                  <span className="flex-1 truncate">样本填充</span>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />}
                </>
              )}
            </NavLink>
          </div>

          {/* 数据查看 */}
          <div>
            <SectionLabel>数据查看</SectionLabel>
            <div className="space-y-0.5">
              {DATA_ITEMS.map(({ to, icon, label }) => (
                <NavItem key={to} to={to} icon={icon} label={label} />
              ))}
            </div>
          </div>

          {/* 设置 */}
          <div>
            <SectionLabel>设置</SectionLabel>
            <div className="space-y-0.5">
              {SETTINGS_ITEMS.map(({ to, icon, label, color }) => (
                <NavItem
                  key={to} to={to} icon={icon} label={label}
                  activeClass={color === 'amber'
                    ? 'bg-amber-500/10 text-amber-600 dark:text-amber-300'
                    : 'bg-slate-200/80 dark:bg-slate-700/50 text-slate-900 dark:text-slate-200'}
                />
              ))}
            </div>
          </div>
        </nav>

        {/* 底部：主题切换 + 版本 */}
        <div className="px-3 py-2 border-t border-[hsl(var(--border)/0.5)] flex items-center justify-between">
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] text-[hsl(var(--text-faint))] font-mono">v2.0</span>
          </div>
          {/* 主题切换按钮 */}
          <button
            onClick={toggleTheme}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all duration-150 text-[hsl(var(--text-muted))] hover:bg-[hsl(var(--surface2))] hover:text-[hsl(var(--text))]"
            title={theme === 'dark' ? '切换到日间模式' : '切换到夜间模式'}
          >
            {theme === 'dark'
              ? <><Sun size={12} /><span>日间</span></>
              : <><Moon size={12} /><span>夜间</span></>
            }
          </button>
        </div>
      </aside>

      {/* ── 主内容区 ── */}
      <main className="flex-1 overflow-hidden flex flex-col bg-[hsl(var(--bg))]">
        <Routes>
          <Route path="/"          element={<Navigate to="/simulate" replace />} />
          <Route path="/simulate"  element={<SimulationRunner />} />
          <Route path="/register"  element={<RegisterSimulator />} />
          <Route path="/templates" element={<TemplateGenerator />} />
          <Route path="/fill"      element={<SampleFiller />} />
          <Route path="/template-viewer" element={<TemplateViewer />} />
          <Route path="/samples"         element={<SampleViewer />} />
          <Route path="/stats"      element={<DatasetStats />} />
          <Route path="/registry"   element={<RegistryPage />} />
          <Route path="/data-dirs"  element={<DataDirsConfig />} />
          <Route path="/llm-config" element={<LLMConfigPage />} />
          <Route path="/monitor"    element={<Navigate to="/templates" replace />} />
          <Route path="/launch"     element={<Navigate to="/templates" replace />} />
          <Route path="*"           element={<Navigate to="/simulate" replace />} />
        </Routes>
      </main>
    </div>
  )
}
