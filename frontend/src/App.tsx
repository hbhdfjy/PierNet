import { useState, useEffect, createContext, useContext } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { Database, BarChart2, BookOpen, Cpu, FlaskConical, BookTemplate, KeyRound, FolderOpen, FileText, Zap, Sun, Moon, GitBranch, Network, Shuffle } from 'lucide-react'
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
  { to: '/router-viewer',   icon: Network,      label: '路由浏览' },
  { to: '/stats',           icon: BarChart2,    label: '数据统计' },
] as const

const SETTINGS_ITEMS = [
  { to: '/registry',   icon: FileText,   label: '注册信息', color: 'sky'   },
  { to: '/data-dirs',  icon: FolderOpen, label: '数据目录', color: 'sky'   },
  { to: '/llm-config', icon: KeyRound,   label: 'LLM 配置', color: 'amber' },
] as const

type StepColor = 'sky' | 'violet' | 'emerald'
const STEP_COLORS: Record<StepColor, { active: string; dot: string }> = {
  sky:     { active: 'bg-sky-500/15 text-sky-400',     dot: 'bg-sky-500'     },
  violet:  { active: 'bg-violet-500/15 text-violet-400', dot: 'bg-violet-500'  },
  emerald: { active: 'bg-emerald-500/15 text-emerald-400', dot: 'bg-emerald-500' },
}

// ── 侧边栏区块标题 ────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 mb-1.5 flex items-center gap-2">
      <span className="text-xs font-bold uppercase tracking-wider whitespace-nowrap text-slate-500">{children}</span>
      <div className="flex-1 h-px bg-slate-700/40" />
    </div>
  )
}

// ── 通用导航链接 ──────────────────────────────────────────────────

function NavItem({
  to, icon: Icon, label,
  activeClass = 'bg-slate-700/50 text-slate-200',
}: {
  to: string; icon: React.ElementType; label: string; activeClass?: string
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150',
        isActive
          ? activeClass + ' font-medium'
          : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300',
      )}
    >
      <Icon size={16} className="flex-shrink-0 opacity-70" />
      <span className="truncate">{label}</span>
    </NavLink>
  )
}

// ── 主应用 ────────────────────────────────────────────────────────

function useSeedState(): [number, (v: number) => void] {
  const [seed, setSeedRaw] = useState<number>(() => {
    const saved = localStorage.getItem('piern-seed')
    const n = saved !== null ? parseInt(saved, 10) : 42
    return isNaN(n) ? 42 : Math.max(0, n)
  })
  const setSeed = (v: number) => {
    setSeedRaw(v)
    localStorage.setItem('piern-seed', String(v))
  }
  return [seed, setSeed]
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [seed, setSeed] = useSeedState()
  const [seedInput, setSeedInput] = useState(String(seed))

  return (
    <SeedContext.Provider value={{ seed, setSeed }}>
    <div className="flex h-screen overflow-hidden bg-[hsl(var(--bg))]">

      {/* ── 侧边栏 ── */}
      <aside className="w-56 flex-shrink-0 bg-[hsl(var(--bg-sub))] border-r border-[hsl(var(--border)/0.6)] flex flex-col select-none">

        {/* Logo */}
        <div className="px-4 py-4 border-b border-[hsl(var(--border)/0.5)]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-sky-900/30">
              <span className="text-white font-black text-base">P</span>
            </div>
            <div>
              <div className="text-base font-bold tracking-wide leading-none text-[hsl(var(--text))]">PiERN</div>
              <div className="text-xs mt-1 leading-none text-[hsl(var(--text-faint))]">多模拟器数据集</div>
            </div>
          </div>
        </div>

        {/* 导航主体 */}
        <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-4">

          {/* Stage 1 */}
          <div>
            <SectionLabel>Stage 1 · 物理仿真</SectionLabel>
            <NavLink
              to="/simulate"
              className={({ isActive }) => cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150',
                isActive
                  ? 'bg-amber-500/15 text-amber-400 font-medium'
                  : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'bg-amber-500/20 text-amber-400' : 'text-slate-500',
                  )}>
                    <Zap size={15} />
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
                      'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150',
                      isActive
                        ? `${colors.active} font-medium`
                        : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300',
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        <div className={cn(
                          'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 text-xs font-bold transition-all',
                          isActive ? 'opacity-90' : 'text-slate-500',
                        )}>
                          {step}
                        </div>
                        <span className="flex-1 truncate">{label}</span>
                        {isActive
                          ? <div className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', colors.dot)} />
                          : <Icon size={14} className="flex-shrink-0 opacity-30" />
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
                'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150',
                isActive
                  ? 'bg-emerald-500/15 text-emerald-400 font-medium'
                  : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'text-emerald-400' : 'text-slate-500',
                  )}>
                    <FlaskConical size={15} />
                  </div>
                  <span className="flex-1 truncate">样本填充</span>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />}
                </>
              )}
            </NavLink>
          </div>

          {/* Stage 4 */}
          <div>
            <SectionLabel>Stage 4 · 路由数据</SectionLabel>
            <NavLink
              to="/router"
              className={({ isActive }) => cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150',
                isActive
                  ? 'bg-rose-500/15 text-rose-400 font-medium'
                  : 'text-slate-500 hover:bg-slate-800/50 hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'text-rose-400' : 'text-slate-500',
                  )}>
                    <GitBranch size={15} />
                  </div>
                  <span className="flex-1 truncate">路由数据</span>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-rose-500 flex-shrink-0" />}
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
                    ? 'bg-amber-500/10 text-amber-400'
                    : 'bg-slate-700/50 text-slate-200'}
                />
              ))}
            </div>
          </div>
        </nav>

        {/* 底部：随机种子 + 主题切换 */}
        <div className="border-t border-[hsl(var(--border)/0.5)] flex-shrink-0">
          {/* 随机种子 */}
          <div className="px-3 pt-2.5 pb-2">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-1.5 text-[hsl(var(--text-faint))]">
                <Shuffle size={11} />
                <span className="text-xs">随机种子</span>
              </div>
              <span className="text-xs font-mono text-[hsl(var(--text-faint))]">全局</span>
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
                setSeed(v)
                setSeedInput(String(v))
              }}
              className="w-full bg-[hsl(var(--surface2))] border border-[hsl(var(--border)/0.6)] rounded-lg px-2.5 py-1.5 text-xs font-mono text-[hsl(var(--text))] focus:outline-none focus:ring-1 focus:ring-sky-500/50 focus:border-sky-500/40 transition-all"
            />
          </div>
          {/* 主题切换 + 版本 */}
          <div className="px-3 pb-2.5 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs text-[hsl(var(--text-faint))] font-mono">v2.0</span>
            </div>
            <button
              onClick={toggleTheme}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 text-[hsl(var(--text-muted))] hover:bg-[hsl(var(--surface2))] hover:text-[hsl(var(--text))]"
              title={theme === 'dark' ? '切换到日间模式' : '切换到夜间模式'}
            >
              {theme === 'dark'
                ? <><Sun size={14} /><span>日间</span></>
                : <><Moon size={14} /><span>夜间</span></>
              }
            </button>
          </div>
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
          <Route path="/router"    element={<RouterDataBuilder />} />
          <Route path="/template-viewer" element={<TemplateViewer />} />
          <Route path="/samples"         element={<SampleViewer />} />
          <Route path="/router-viewer"   element={<RouterViewer />} />
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
    </SeedContext.Provider>
  )
}
