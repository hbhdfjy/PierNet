import { lazy, Suspense, useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import {
  Database,
  BarChart2,
  BookOpen,
  Cpu,
  FlaskConical,
  BookTemplate,
  KeyRound,
  FileText,
  Zap,
  Sun,
  Moon,
  GitBranch,
  Network,
  Shuffle,
  ChevronRight,
  UploadCloud,
  FolderOpen,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { SeedContext, readStoredSeed, writeStoredSeed } from '../lib/seedContext'
import type { Theme } from '../shared/theme'
import { PlatformSwitcher } from '../platform/PlatformSwitcher'

const DatasetStats = lazy(() => import('./pages/DatasetStats'))
const LLMConfigPage = lazy(() => import('./pages/LLMConfig'))
const RegisterSimulator = lazy(() => import('./pages/RegisterSimulator'))
const RegistryPage = lazy(() => import('./pages/RegistryPage'))
const RouterDataBuilder = lazy(() => import('./pages/RouterDataBuilder'))
const RouterViewer = lazy(() => import('./pages/RouterViewer'))
const SampleFiller = lazy(() => import('./pages/SampleFiller'))
const SampleViewer = lazy(() => import('./pages/SampleViewer'))
const SimulationRunner = lazy(() => import('./pages/SimulationRunner'))
const TemplateGenerator = lazy(() => import('./pages/TemplateGenerator'))
const TemplateViewer = lazy(() => import('./pages/TemplateViewer'))
const DataUploadPage = lazy(() => import('./pages/DataUploadPage'))
const FileManagerContent = lazy(() =>
  import('../files/FileManagerPage').then(module => ({ default: module.FileManagerContent })),
)

const STAGE_COLORS = {
  amber: { accent: 'hsl(36 96% 61%)', soft: 'rgba(245, 158, 11, 0.14)', border: 'rgba(245, 158, 11, 0.24)' },
  sky: { accent: 'hsl(198 93% 60%)', soft: 'rgba(14, 165, 233, 0.14)', border: 'rgba(14, 165, 233, 0.24)' },
  violet: { accent: 'hsl(262 88% 68%)', soft: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.24)' },
  emerald: { accent: 'hsl(158 64% 52%)', soft: 'rgba(16, 185, 129, 0.14)', border: 'rgba(16, 185, 129, 0.22)' },
  rose: { accent: 'hsl(347 86% 62%)', soft: 'rgba(244, 63, 94, 0.14)', border: 'rgba(244, 63, 94, 0.22)' },
} as const

type StageColor = keyof typeof STAGE_COLORS

function useSeedState(): [number, (v: number) => void] {
  const [seed, setSeedRaw] = useState<number>(() => readStoredSeed())

  const setSeed = (value: number) => {
    setSeedRaw(writeStoredSeed(value))
  }

  return [seed, setSeed]
}

function PageFallback() {
  return <div className="px-6 py-5 text-sm text-slate-400">加载中...</div>
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
  end = false,
}: {
  to: string
  icon: React.ElementType
  label: string
  color?: StageColor
  step?: string
  rightIcon?: React.ElementType
  end?: boolean
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
      end={end}
      style={toneStyle}
      className={({ isActive }) => cn('nav-item', isActive && 'nav-item--active')}
    >
      {({ isActive }) => (
        <>
          {isActive && <span className="nav-item__rail" />}
          <div className={cn('nav-item__icon', step && 'nav-item__icon--step')}>{step ? step : <Icon size={14} />}</div>
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
              <div className="app-brand__title">PiERN 数据</div>
              <div className="app-brand__subtitle">数据合成工作台</div>
            </div>
          </div>

          <PlatformSwitcher active="synth" />

          <nav className="app-nav">
            <div>
              <SectionLabel>{'\u6570\u636e\u603b\u89c8'}</SectionLabel>
              <NavItem to="/synth" end icon={BarChart2} label={'\u6570\u636e\u603b\u89c8'} />
            </div>

            <div>
              <SectionLabel>阶段 1 · 物理仿真</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/simulate" icon={Zap} label="物理仿真" color="amber" />
                <NavItem to="/synth/upload" icon={UploadCloud} label="上传数据" color="amber" />
              </div>
            </div>

            <div>
              <SectionLabel>阶段 2 · 语言模板</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/register" icon={BookOpen} label="注册场景" color="sky" step="01" />
                <NavItem to="/synth/templates" icon={Cpu} label="生成模板" color="violet" step="02" />
              </div>
            </div>

            <div>
              <SectionLabel>阶段 3 · 样本填充</SectionLabel>
              <NavItem to="/synth/fill" icon={FlaskConical} label="填充样本" color="emerald" />
            </div>

            <div>
              <SectionLabel>阶段 4 · 路由数据</SectionLabel>
              <NavItem to="/synth/router" icon={GitBranch} label="构建路由" color="rose" />
            </div>

            <div>
              <SectionLabel>数据视图</SectionLabel>
              <div className="space-y-1">
                <NavItem to="/synth/template-viewer" icon={BookTemplate} label="模板浏览" rightIcon={ChevronRight} />
                <NavItem to="/synth/samples" icon={Database} label="样本浏览" rightIcon={ChevronRight} />
                <NavItem to="/synth/router-viewer" icon={Network} label="路由浏览" rightIcon={ChevronRight} />
                <NavItem
                  to="/synth/files"
                  icon={FolderOpen}
                  label={'\u6587\u4ef6\u7ba1\u7406'}
                  rightIcon={ChevronRight}
                />
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
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route index element={<DatasetStats />} />
              <Route path="simulate" element={<SimulationRunner />} />
              <Route path="upload" element={<DataUploadPage />} />
              <Route path="register" element={<RegisterSimulator />} />
              <Route path="templates" element={<TemplateGenerator />} />
              <Route path="fill" element={<SampleFiller />} />
              <Route path="router" element={<RouterDataBuilder />} />
              <Route path="template-viewer" element={<TemplateViewer />} />
              <Route path="samples" element={<SampleViewer />} />
              <Route path="router-viewer" element={<RouterViewer />} />
              <Route path="files" element={<FileManagerContent />} />
              <Route path="stats" element={<Navigate to="/synth" replace />} />
              <Route path="registry" element={<RegistryPage />} />
              <Route path="llm-config" element={<LLMConfigPage />} />
              <Route path="*" element={<Navigate to="/synth/simulate" replace />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </SeedContext.Provider>
  )
}
