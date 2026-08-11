import { lazy } from 'react'
import { Layers3, MessageSquare, PlayCircle, Workflow } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingSimpleJobPage = lazy(() => import('./pages/TrainingSimpleJobPage'))
const TrainingSimpleTasksPage = lazy(() => import('./pages/TrainingSimpleTasksPage'))
const TrainingSimpleAssemblyPage = lazy(() => import('./pages/TrainingSimpleAssemblyPage'))
const ModelChatPage = lazy(() => import('./pages/ModelChatPage'))
const TrainingSimpleProgressPage = lazy(() => import('./pages/TrainingSimpleProgressPage'))

const navGroups: ShellNavGroup[] = [
  {
    label: '简洁训练',
    items: [
      { to: '/training/simple', end: true, icon: PlayCircle, label: '模型训练', tone: 'emerald' },
      { to: '/training/simple/tasks', icon: Workflow, label: '训练任务', tone: 'sky' },
      { to: '/training/simple/assembly', icon: Layers3, label: '模型拼装', tone: 'violet' },
      { to: '/training/simple/chat', icon: MessageSquare, label: '模型对话', tone: 'emerald' },
    ],
  },
]

export default function TrainingSimpleApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <AppShell
      platform="simple-training"
      mark="S"
      title="Piern 简洁训练"
      subtitle="训练、拼装与对话"
      navGroups={navGroups}
      theme={theme}
      toggleTheme={toggleTheme}
    >
      <Routes>
        <Route index element={<TrainingSimpleJobPage />} />
        <Route path="tasks" element={<TrainingSimpleTasksPage />} />
        <Route path="assembly" element={<TrainingSimpleAssemblyPage />} />
        <Route path="chat" element={<ModelChatPage assemblyPath="/training/simple/assembly" />} />
        <Route path="jobs/:jobId" element={<TrainingSimpleProgressPage />} />
        <Route path="router" element={<Navigate to="/training/simple" replace />} />
        <Route path="text2comp" element={<Navigate to="/training/simple" replace />} />
        <Route path="*" element={<Navigate to="/training/simple" replace />} />
      </Routes>
    </AppShell>
  )
}
