import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { Database, BarChart2, BookOpen, Cpu, FlaskConical, BookTemplate, KeyRound, FolderOpen, FileText, Zap } from 'lucide-react'
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
  sky:     { active: 'bg-sky-500/15 text-sky-300 border-sky-500/25',            dot: 'bg-sky-500'     },
  violet:  { active: 'bg-violet-500/15 text-violet-300 border-violet-500/25',   dot: 'bg-violet-500'  },
  emerald: { active: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25', dot: 'bg-emerald-500' },
}

// ── 侧边栏区块标题 ────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 mb-1.5 flex items-center gap-2">
      <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">{children}</span>
      <div className="flex-1 h-px bg-slate-700/40" />
    </div>
  )
}

// ── 通用导航链接 ──────────────────────────────────────────────────

function NavItem({
  to, icon: Icon, label,
  activeClass = 'bg-slate-700/60 text-slate-200',
}: {
  to: string; icon: React.ElementType; label: string; activeClass?: string
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        'flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition-all duration-150',
        isActive ? activeClass + ' font-medium' : 'text-slate-500 hover:bg-slate-800/60 hover:text-slate-300',
      )}
    >
      <Icon size={14} className="flex-shrink-0" />
      <span className="truncate">{label}</span>
    </NavLink>
  )
}

// ── 主应用 ────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">

      {/* ── 侧边栏 ── */}
      <aside className="w-52 flex-shrink-0 bg-slate-900/98 border-r border-slate-700/40 flex flex-col select-none">

        {/* Logo */}
        <div className="px-4 py-4 border-b border-slate-700/40">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-500/30 to-blue-600/20 border border-sky-500/40 flex items-center justify-center flex-shrink-0">
              <span className="text-sky-400 font-black text-sm">P</span>
            </div>
            <div>
              <div className="text-white font-bold text-sm tracking-wide leading-none">PiERN</div>
              <div className="text-slate-600 text-xs mt-0.5">多模拟器数据集</div>
            </div>
          </div>
        </div>

        {/* 导航主体 */}
        <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-4">

          {/* Stage 1 */}
          <div>
            <SectionLabel>Stage 1 · 物理仿真</SectionLabel>
            <NavLink
              to="/simulate"
              className={({ isActive }) => cn(
                'flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition-all duration-150 border',
                isActive
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/25 font-medium'
                  : 'text-slate-500 border-transparent hover:bg-slate-800/60 hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 transition-all',
                    isActive ? 'bg-amber-500 text-white' : 'bg-slate-700/60 text-slate-500',
                  )}>
                    <Zap size={11} />
                  </div>
                  <span className="flex-1 truncate">仿真运行</span>
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
                      'flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition-all duration-150 border',
                      isActive
                        ? `${colors.active} font-medium`
                        : 'text-slate-500 border-transparent hover:bg-slate-800/60 hover:text-slate-300',
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        <div className={cn(
                          'w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 text-xs font-bold transition-all',
                          isActive ? colors.dot + ' text-white' : 'bg-slate-700/60 text-slate-500',
                        )}>
                          {step}
                        </div>
                        <span className="flex-1 truncate">{label}</span>
                        <Icon size={12} className="flex-shrink-0 opacity-50" />
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
                'flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition-all duration-150 border',
                isActive
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25 font-medium'
                  : 'text-slate-500 border-transparent hover:bg-slate-800/60 hover:text-slate-300',
              )}
            >
              {({ isActive }) => (
                <>
                  <div className={cn(
                    'w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 text-xs font-bold transition-all',
                    isActive ? 'bg-emerald-500 text-white' : 'bg-slate-700/60 text-slate-500',
                  )}>
                    01
                  </div>
                  <span className="flex-1 truncate">样本填充</span>
                  <FlaskConical size={12} className="flex-shrink-0 opacity-50" />
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
                  activeClass={color === 'amber' ? 'bg-amber-500/15 text-amber-300' : 'bg-sky-500/15 text-sky-300'}
                />
              ))}
            </div>
          </div>
        </nav>

        {/* 底部版本信息 */}
        <div className="px-4 py-2.5 border-t border-slate-700/40 flex items-center justify-between">
          <span className="text-xs text-slate-700 font-mono">ICML 2026</span>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500/60 animate-pulse" />
            <span className="text-xs text-slate-700">v2</span>
          </div>
        </div>
      </aside>

      {/* ── 主内容区 ── */}
      <main className="flex-1 overflow-hidden flex flex-col bg-slate-900">
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
