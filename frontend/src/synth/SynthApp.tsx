import { lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import {
  BarChart2,
  BookOpen,
  BookTemplate,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  Bot,
  FolderOpen,
  GitBranch,
  KeyRound,
  Network,
  UploadCloud,
  Zap,
} from 'lucide-react'
import type { Theme } from '../shared/theme'
import { AppShell, type ShellNavGroup } from '../platform/AppShell'

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
const ExpertModelManager = lazy(() => import('./pages/ExpertModelManager'))
const FileManagerContent = lazy(() =>
  import('../files/FileManagerPage').then(module => ({ default: module.FileManagerContent })),
)

const navGroups: ShellNavGroup[] = [
  {
    label: '数据总览',
    items: [{ to: '/synth', end: true, icon: BarChart2, label: '数据总览' }],
  },
  {
    label: '阶段 1 · 物理仿真',
    items: [
      { to: '/synth/simulate', icon: Zap, label: '物理仿真', tone: 'amber' },
      { to: '/synth/upload', icon: UploadCloud, label: '上传数据', tone: 'amber' },
    ],
  },
  {
    label: '阶段 2 · 语言模板',
    items: [
      { to: '/synth/register', icon: BookOpen, label: '注册场景', tone: 'sky', step: '01' },
      { to: '/synth/templates', icon: Cpu, label: '生成模板', tone: 'violet', step: '02' },
    ],
  },
  {
    label: '阶段 3 · 样本填充',
    items: [{ to: '/synth/fill', icon: FlaskConical, label: '填充样本', tone: 'emerald' }],
  },
  {
    label: '阶段 4 · 路由数据',
    items: [{ to: '/synth/router', icon: GitBranch, label: '构建路由', tone: 'rose' }],
  },
  {
    label: '数据视图',
    items: [
      { to: '/synth/template-viewer', icon: BookTemplate, label: '模板浏览', rightIcon: ChevronRight },
      { to: '/synth/samples', icon: Database, label: '样本浏览', rightIcon: ChevronRight },
      { to: '/synth/router-viewer', icon: Network, label: '路由浏览', rightIcon: ChevronRight },
      { to: '/synth/files', icon: FolderOpen, label: '文件管理', rightIcon: ChevronRight },
    ],
  },
  {
    label: '系统设置',
    items: [
      { to: '/synth/registry', icon: FileText, label: '注册信息' },
      { to: '/synth/expert-models', icon: Bot, label: '专家模型', tone: 'violet' },
      { to: '/synth/llm-config', icon: KeyRound, label: 'LLM 配置', tone: 'amber' },
    ],
  },
]

export default function SynthApp({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <AppShell
      platform="synth"
      mark="P"
      title="PierNet 数据"
      subtitle="数据合成工作台"
      navGroups={navGroups}
      theme={theme}
      toggleTheme={toggleTheme}
    >
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
        <Route path="expert-models" element={<ExpertModelManager />} />
        <Route path="llm-config" element={<LLMConfigPage />} />
        <Route path="*" element={<Navigate to="/synth/simulate" replace />} />
      </Routes>
    </AppShell>
  )
}
