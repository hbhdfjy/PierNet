import { lazy } from 'react'
import { BarChart3, Brain, Cpu, FolderOpen, HardDrive, Layers, MessageSquare, PlayCircle, Workflow } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingOverviewPage = lazy(() => import('./pages/TrainingOverviewPage'))
const TrainingNewJobPage = lazy(() => import('./pages/TrainingNewJobPage'))
const TrainingJobsPage = lazy(() => import('./pages/TrainingJobsPage'))
const TrainingJobDetailPage = lazy(() => import('./pages/TrainingJobDetailPage'))
const Text2CompPage = lazy(() => import('./pages/Text2CompPage'))
const AssemblyPage = lazy(() => import('./pages/AssemblyPage'))
const ModelChatPage = lazy(() => import('./pages/ModelChatPage'))
const TrainedModelsPage = lazy(() => import('./pages/TrainedModelsPage'))
const FileManagerContent = lazy(() =>
  import('../files/FileManagerPage').then(module => ({ default: module.FileManagerContent })),
)

const navGroups: ShellNavGroup[] = [
  {
    label: '训练平台',
    items: [
      { to: '/training', end: true, icon: BarChart3, label: '总览', tone: 'sky' },
      { to: '/training/new', icon: PlayCircle, label: '新建训练', tone: 'emerald' },
      { to: '/training/jobs', icon: Workflow, label: '任务管理', tone: 'violet' },
      { to: '/training/text2comp', icon: Brain, label: '文生计算', tone: 'emerald' },
      { to: '/training/assembly', icon: Layers, label: '模型拼装', tone: 'violet' },
      { to: '/training/chat', icon: MessageSquare, label: '模型对话', tone: 'sky' },
      { to: '/training/models', icon: HardDrive, label: '模型文件', tone: 'amber' },
      { to: '/training/files', icon: FolderOpen, label: '文件管理', tone: 'neutral' },
    ],
  },
  {
    label: '当前范围',
    note: (
      <div className="flex items-center gap-2 text-[12px] font-medium text-slate-200">
        <Cpu size={13} />
        <span>Token Router · Text2Comp · FNO</span>
      </div>
    ),
  },
]

export default function TrainingApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <AppShell
      platform="training"
      mark="T"
      title="Piern 训练"
      subtitle="模型训练工作台"
      navGroups={navGroups}
      theme={theme}
      toggleTheme={toggleTheme}
    >
      <Routes>
        <Route index element={<TrainingOverviewPage />} />
        <Route path="new" element={<TrainingNewJobPage />} />
        <Route path="jobs" element={<TrainingJobsPage />} />
        <Route path="jobs/:jobId" element={<TrainingJobDetailPage />} />
        <Route path="text2comp" element={<Text2CompPage />} />
        <Route path="assembly" element={<AssemblyPage />} />
        <Route path="chat" element={<ModelChatPage />} />
        <Route path="models" element={<TrainedModelsPage />} />
        <Route
          path="files"
          element={
            <FileManagerContent
              initialPlatform="training"
              lockPlatform
              title="训练文件管理"
              copy="集中查看训练任务、权重文件、曲线、日志和可删除的历史产物。"
            />
          }
        />
        <Route path="*" element={<Navigate to="/training" replace />} />
      </Routes>
    </AppShell>
  )
}
