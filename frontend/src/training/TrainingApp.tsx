import { Moon, Sun, ArrowLeft, BarChart3, Cpu, PlayCircle, Workflow } from 'lucide-react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import TrainingOverviewPage from './pages/TrainingOverviewPage'
import TrainingNewJobPage from './pages/TrainingNewJobPage'
import TrainingJobsPage from './pages/TrainingJobsPage'
import TrainingJobDetailPage from './pages/TrainingJobDetailPage'

type Theme = 'dark' | 'light'

function SectionLabel({ children }: { children: React.ReactNode }) {
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
  end = false,
}: {
  to: string
  icon: React.ElementType
  label: string
  end?: boolean
}) {
  return (
    <NavLink to={to} end={end} className={({ isActive }) => `nav-item ${isActive ? 'nav-item--active' : ''}`}>
      {({ isActive }) => (
        <>
          {isActive && <span className="nav-item__rail" />}
          <div className="nav-item__icon"><Icon size={14} /></div>
          <span className="nav-item__label">{label}</span>
        </>
      )}
    </NavLink>
  )
}

export default function TrainingApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar w-56 flex-shrink-0">
        <div className="app-brand">
          <div className="app-brand__mark-wrap">
            <div className="app-brand__mark">T</div>
            <span className="app-brand__status" />
          </div>
          <div className="min-w-0">
            <div className="app-brand__title">PiERN Training</div>
            <div className="app-brand__subtitle">Single-GPU token router workbench</div>
          </div>
        </div>

        <nav className="app-nav">
          <div>
            <SectionLabel>训练平台</SectionLabel>
            <div className="space-y-1">
              <NavItem to="/training" end icon={BarChart3} label="总览" />
              <NavItem to="/training/new" icon={PlayCircle} label="新建训练" />
              <NavItem to="/training/jobs" icon={Workflow} label="任务管理" />
            </div>
          </div>

          <div>
            <SectionLabel>平台切换</SectionLabel>
            <div className="space-y-1">
              <NavItem to="/simulate" icon={ArrowLeft} label="返回数据平台" />
            </div>
          </div>

          <div>
            <SectionLabel>当前范围</SectionLabel>
            <div className="rounded-2xl border border-slate-700/40 bg-slate-900/30 p-3 text-xs text-slate-400">
              <div className="flex items-center gap-2 text-slate-200">
                <Cpu size={13} />
                单 GPU
              </div>
              <div className="mt-2 leading-6">
                当前只实现 Token Router 训练。多卡分配和 DDP 暂不开放，前后端能力保持一致。
              </div>
            </div>
          </div>
        </nav>

        <div className="app-sidebar__footer">
          <div className="app-sidebar__footer-row">
            <span className="app-footnote mono">training v1</span>
            <button type="button" onClick={toggleTheme} className="theme-toggle">
              {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
              <span>{theme === 'dark' ? '日间' : '夜间'}</span>
            </button>
          </div>
        </div>
      </aside>

      <main className="app-main">
        <Routes>
          <Route path="/training" element={<TrainingOverviewPage />} />
          <Route path="/training/new" element={<TrainingNewJobPage />} />
          <Route path="/training/jobs" element={<TrainingJobsPage />} />
          <Route path="/training/jobs/:jobId" element={<TrainingJobDetailPage />} />
          <Route path="*" element={<Navigate to="/training" replace />} />
        </Routes>
      </main>
    </div>
  )
}
