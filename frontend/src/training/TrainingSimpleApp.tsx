import { lazy } from 'react'
import { Activity, Cpu, Layers3, PlayCircle, Workflow } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingSimpleJobPage = lazy(() => import('./pages/TrainingSimpleJobPage'))
const TrainingSimpleTasksPage = lazy(() => import('./pages/TrainingSimpleTasksPage'))
const TrainingSimpleAssemblyPage = lazy(() => import('./pages/TrainingSimpleAssemblyPage'))
const TrainingSimpleProgressPage = lazy(() => import('./pages/TrainingSimpleProgressPage'))

const navGroups: ShellNavGroup[] = [
  {
    label: '简洁训练',
    items: [
      { to: '/training/simple', end: true, icon: PlayCircle, label: '模型训练', tone: 'emerald' },
      { to: '/training/simple/tasks', icon: Workflow, label: '训练任务', tone: 'sky' },
      { to: '/training/simple/assembly', icon: Layers3, label: '模型拼装', tone: 'violet' },
    ],
  },
  {
    label: '训练闭环',
    note: (
      <div className="grid gap-1 text-[12px] font-medium text-slate-200">
        <div className="flex items-center gap-2">
          <Cpu size={13} />
          <span>场景选择 · 自动训练</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Activity size={13} />
          <span>进度、任务、拼装</span>
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
        <Route path="tasks" element={<TrainingSimpleTasksPage />} />
        <Route path="assembly" element={<TrainingSimpleAssemblyPage />} />
        <Route path="jobs/:jobId" element={<TrainingSimpleProgressPage />} />
        <Route path="router" element={<Navigate to="/training/simple" replace />} />
        <Route path="text2comp" element={<Navigate to="/training/simple" replace />} />
        <Route path="*" element={<Navigate to="/training/simple" replace />} />
      </Routes>
    </AppShell>
  )
}
