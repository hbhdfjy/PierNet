import { lazy } from 'react'
import { BarChart3, Cpu, FolderOpen, PlayCircle, Workflow } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

const TrainingOverviewPage = lazy(() => import('./pages/TrainingOverviewPage'))
const TrainingNewJobPage = lazy(() => import('./pages/TrainingNewJobPage'))
const TrainingJobsPage = lazy(() => import('./pages/TrainingJobsPage'))
const TrainingJobDetailPage = lazy(() => import('./pages/TrainingJobDetailPage'))
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
      { to: '/training/files', icon: FolderOpen, label: '文件管理', tone: 'amber' },
    ],
  },
  {
    label: '当前范围',
    note: (
      <div className="flex items-center gap-2 text-[12px] font-medium text-slate-200">
        <Cpu size={13} />
        <span>单卡 · Token Router</span>
      </div>
    ),
  },
]

export default function TrainingApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <AppShell
      platform="training"
      mark="T"
      title="PierNet 训练"
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
