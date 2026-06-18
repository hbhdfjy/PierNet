import { lazy } from 'react'
import { Activity, Cpu, MousePointerClick } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingSimpleJobPage = lazy(() => import('./pages/TrainingSimpleJobPage'))
const TrainingSimpleProgressPage = lazy(() => import('./pages/TrainingSimpleProgressPage'))

const navGroups: ShellNavGroup[] = [
  {
    label: '简洁训练',
    items: [{ to: '/training/simple', end: true, icon: MousePointerClick, label: '开始训练', tone: 'emerald' }],
  },
  {
    label: '训练闭环',
    note: (
      <div className="grid gap-1 text-[12px] font-medium text-slate-200">
        <div className="flex items-center gap-2">
          <Cpu size={13} />
          <span>自动配置 Router 训练</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Activity size={13} />
          <span>进度、结果、继续训练</span>
        </div>
      </div>
    ),
  },
]

export default function TrainingSimpleApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <AppShell
      platform="simple-training"
      mark="S"
      title="PierNet 简洁训练"
      subtitle="低配置训练入口"
      navGroups={navGroups}
      theme={theme}
      toggleTheme={toggleTheme}
    >
      <Routes>
        <Route index element={<TrainingSimpleJobPage />} />
        <Route path="jobs/:jobId" element={<TrainingSimpleProgressPage />} />
        <Route path="*" element={<Navigate to="/training/simple" replace />} />
      </Routes>
    </AppShell>
  )
}
