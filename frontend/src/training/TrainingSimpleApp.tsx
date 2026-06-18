import { lazy } from 'react'
import { Activity, BarChart3, Brain, Cpu, Layers3, MousePointerClick } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingSimpleHubPage = lazy(() => import('./pages/TrainingSimpleHubPage'))
const TrainingSimpleJobPage = lazy(() => import('./pages/TrainingSimpleJobPage'))
const TrainingSimpleText2CompPage = lazy(() => import('./pages/TrainingSimpleText2CompPage'))
const TrainingSimpleAssemblyPage = lazy(() => import('./pages/TrainingSimpleAssemblyPage'))
const TrainingSimpleProgressPage = lazy(() => import('./pages/TrainingSimpleProgressPage'))

const navGroups: ShellNavGroup[] = [
  {
    label: '简洁训练',
    items: [
      { to: '/training/simple', end: true, icon: BarChart3, label: '工作台', tone: 'sky' },
      { to: '/training/simple/router', icon: MousePointerClick, label: 'Router 训练', tone: 'emerald' },
      { to: '/training/simple/text2comp', icon: Brain, label: '文生计算', tone: 'emerald' },
      { to: '/training/simple/assembly', icon: Layers3, label: '模型拼装', tone: 'violet' },
    ],
  },
  {
    label: '粗粒度闭环',
    note: (
      <div className="grid gap-1 text-[12px] font-medium text-slate-200">
        <div className="flex items-center gap-2">
          <Cpu size={13} />
          <span>Router · Text2Comp · Expert</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Activity size={13} />
          <span>训练、拼装、测试、结果</span>
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
      subtitle="粗粒度训练入口"
      navGroups={navGroups}
      theme={theme}
      toggleTheme={toggleTheme}
    >
      <Routes>
        <Route index element={<TrainingSimpleHubPage />} />
        <Route path="router" element={<TrainingSimpleJobPage />} />
        <Route path="text2comp" element={<TrainingSimpleText2CompPage />} />
        <Route path="assembly" element={<TrainingSimpleAssemblyPage />} />
        <Route path="jobs/:jobId" element={<TrainingSimpleProgressPage />} />
        <Route path="*" element={<Navigate to="/training/simple" replace />} />
      </Routes>
    </AppShell>
  )
}
