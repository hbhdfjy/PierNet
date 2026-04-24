import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import {
  Database, BarChart2, BookOpen, Cpu, FlaskConical, BookTemplate,
  KeyRound, FileText, Zap, Sun, Moon, GitBranch, Network,
  Shuffle, ChevronRight,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { SeedContext } from '../lib/seedContext'
import DatasetStats from './pages/DatasetStats'
import LLMConfigPage from './pages/LLMConfig'
import RegisterSimulator from './pages/RegisterSimulator'
import RegistryPage from './pages/RegistryPage'
import RouterDataBuilder from './pages/RouterDataBuilder'
import RouterViewer from './pages/RouterViewer'
import SampleFiller from './pages/SampleFiller'
import SampleViewer from './pages/SampleViewer'
import SimulationRunner from './pages/SimulationRunner'
import TemplateGenerator from './pages/TemplateGenerator'
import TemplateViewer from './pages/TemplateViewer'
import type { Theme } from '../shared/theme'

const STAGE_COLORS = {
  amber:   { accent: 'hsl(36 96% 61%)', soft: 'rgba(245, 158, 11, 0.14)', border: 'rgba(245, 158, 11, 0.24)' },
  sky:     { accent: 'hsl(198 93% 60%)', soft: 'rgba(14, 165, 233, 0.14)', border: 'rgba(14, 165, 233, 0.24)' },
  violet:  { accent: 'hsl(262 88% 68%)', soft: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.24)' },
  emerald: { accent: 'hsl(158 64% 52%)', soft: 'rgba(16, 185, 129, 0.14)', border: 'rgba(16, 185, 129, 0.22)' },
  rose:    { accent: 'hsl(347 86% 62%)', soft: 'rgba(244, 63, 94, 0.14)', border: 'rgba(244, 63, 94, 0.22)' },
} as const

type StageColor = keyof typeof STAGE_COLORS

function useSeedState(): [number, (v: number) => void] {
  const [seed, setSeedRaw] = useState<number>(() => {
    const n = parseInt(localStorage.getItem('piern-seed') ?? '42', 10)
    return isNaN(n) ? 42 : Math.max(0, n)
  })

  const setSeed = (v: number) => {
    setSeedRaw(v)
    localStorage.setItem('piern-seed', String(v))
  }

  return [seed, setSeed]
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="app-section-label">
      <span className="label text-[11px] whitespace-nowrap">{children}</span>
      <div className="app-section-label__line" />
    </div>
  )
}

function NavItem({
  to,
  icon: Icon,
  label,
  color,
  step,
  rightIcon: RightIcon,
}: {
  to: string
  icon: React.ElementType
  label: string
  color?: StageColor
  step?: string
  rightIcon?: React.ElementType
}) {
  const tone = color ? STAGE_COLORS[color] : null
  const toneStyle = tone
    ? ({
        ['--tone' as string]: tone.accent,
        ['--tone-soft' as string]: tone.soft,
        ['--tone-border' as string]: tone.border,
      } as CSSProperties)
    : undefined

  return (
    <NavLink
      to={to}
      style={toneStyle}
      className={({ isActive }) => cn('nav-item', isActive && 'nav-item--active')}
    >
      {({ isActive }) => (
        <>
          {isActive && <span className="nav-item__rail" />}
          <div className={cn('nav-item__icon', step && 'nav-item__icon--step')}>
            {step ? step : <Icon size={14} />}
          </div>
          <span className="nav-item__label">{label}</span>
          {!isActive && RightIcon && (
            <span className="nav-item__hint">
              <RightIcon size={11} />
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}

export default function SynthApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  const [seed, setSeed] = useSeedState()
  const [seedInput, setSeedInput] = useState(String(seed))

  useEffect(() => {
    setSeedInput(String(seed))
  }, [seed])

  return (
    <SeedContext.Provider value={{ seed, setSeed }}>
      <div className="app-shell">
        <aside className="app-sidebar w-56 flex-shrink-0">
          <div className="app-brand">
            <div className="app-brand__mark-wrap">
              <div className="app-brand__mark">P</div>
              <span className="app-brand__status" />
            </div>
            <div className="min-w-0">
              <div className="app-brand__title">PiERN Synth</div>
              <div className="app-brand__subtitle">数据合成工作台</div>
            </div>
          </div>

          <nav className="app-nav">
            <div>
              <SectionLabel>{'\u6570\u636e\u603b\u89c8'}</SectionLabel>
              <NavItem to="/synth" icon={BarChart2} label={'\u6570\u636e\u603b\u89c8'} />
            </div>

            <div>
              <SectionLabel>Stage 1 - Simulation</SectionLabel>
              <NavItem to="/synth/simulate" icon={Zap} label="物理仿真" color="amber" />
            </div>

            <div>
              <SectionLabel>Stage 2 - Templates</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/register" icon={BookOpen} label="注册场景" color="sky" step="01" />
                <NavItem to="/synth/templates" icon={Cpu} label="生成模板" color="violet" step="02" />
              </div>
            </div>

            <div>
              <SectionLabel>Stage 3 - Samples</SectionLabel>
              <NavItem to="/synth/fill" icon={FlaskConical} label="填充样本" color="emerald" />
            </div>

            <div>
              <SectionLabel>Stage 4 - Router</SectionLabel>
              <NavItem to="/synth/router" icon={GitBranch} label="构建路由" color="rose" />
            </div>

            <div>
              <SectionLabel>数据视图</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/template-viewer" icon={BookTemplate} label="模板浏览" rightIcon={ChevronRight} />
                <NavItem to="/synth/samples" icon={Database} label="样本浏览" rightIcon={ChevronRight} />
                <NavItem to="/synth/router-viewer" icon={Network} label="路由浏览" rightIcon={ChevronRight} />
              </div>
            </div>

            <div>
              <SectionLabel>平台切换</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/training" icon={GitBranch} label="训练平台" rightIcon={ChevronRight} />
              </div>
            </div>

            <div>
              <SectionLabel>系统设置</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/registry" icon={FileText} label="注册信息" />
                <NavItem to="/synth/llm-config" icon={KeyRound} label="LLM 配置" color="amber" />
              </div>
            </div>
          </nav>

          <div className="app-sidebar__footer">
            <div className="app-seed-card">
              <div className="app-seed-card__label-row">
                <div className="app-seed-card__label">
                  <Shuffle size={11} />
                  <span>随机种子</span>
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
                  const value = isNaN(n) ? 42 : Math.max(0, n)
                  setSeed(value)
                  setSeedInput(String(value))
                }}
                className="input app-seed-input mono"
              />
            </div>

            <div className="app-sidebar__footer-row">
              <button type="button" onClick={toggleTheme} className="theme-toggle">
                {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
                <span>{theme === 'dark' ? '日间' : '夜间'}</span>
              </button>
            </div>
          </div>
        </aside>

        <main className="app-main">
          <Routes>
            <Route index element={<DatasetStats />} />
            <Route path="simulate" element={<SimulationRunner />} />
            <Route path="register" element={<RegisterSimulator />} />
            <Route path="templates" element={<TemplateGenerator />} />
            <Route path="fill" element={<SampleFiller />} />
            <Route path="router" element={<RouterDataBuilder />} />
            <Route path="template-viewer" element={<TemplateViewer />} />
            <Route path="samples" element={<SampleViewer />} />
            <Route path="router-viewer" element={<RouterViewer />} />
            <Route path="stats" element={<Navigate to="/synth" replace />} />
            <Route path="registry" element={<RegistryPage />} />
            <Route path="llm-config" element={<LLMConfigPage />} />
            <Route path="*" element={<Navigate to="/synth/simulate" replace />} />
          </Routes>
        </main>
      </div>
    </SeedContext.Provider>
  )
}
