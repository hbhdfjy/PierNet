import { ArrowRight, Database, GitBranch } from 'lucide-react'
import { Link } from 'react-router-dom'

function EntryCard({
  to,
  title,
  copy,
  tone,
  icon,
  points,
}: {
  to: string
  title: string
  copy: string
  tone: string
  icon: React.ReactNode
  points: string[]
}) {
  return (
    <Link to={to} className="platform-entry-card group">
      <div className={`platform-entry-card__tone ${tone}`} />
      <div className="relative">
        <div className="platform-entry-card__header">
          <div className="platform-entry-card__icon">{icon}</div>
          <ArrowRight size={18} className="mt-1 text-slate-500 transition group-hover:translate-x-1 group-hover:text-slate-200" />
        </div>

        <h2 className="platform-entry-card__title">{title}</h2>
        <p className="platform-entry-card__copy">{copy}</p>

        <div className="platform-entry-card__points">
          {points.map(point => (
            <div key={point} className="platform-entry-card__point">
              {point}
            </div>
          ))}
        </div>
      </div>
    </Link>
  )
}

export default function LandingPage() {
  return (
    <div className="platform-home-shell text-slate-100">
      <div className="platform-home-wrap">
        <header className="training-hero">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-2xl">
              <div className="training-eyebrow">
                <span>PiERN Gateway</span>
              </div>
              <h1 className="mt-4 text-[2.2rem] font-semibold tracking-tight text-white xl:text-[2.6rem]">选择工作平台</h1>
              <p className="mt-3 max-w-xl training-copy">
                数据合成与模型训练已拆为独立入口，请选择要进入的工作面。
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Link to="/synth" className="btn-primary">
                <Database size={15} />
                数据合成
              </Link>
              <Link to="/training" className="btn-ghost">
                <GitBranch size={15} />
                模型训练
              </Link>
            </div>
          </div>
        </header>

        <main className="platform-home-grid">
          <EntryCard
            to="/synth"
            title="数据合成"
            copy="物理仿真、场景注册、模板生成、样本填充与 Router 数据构建。"
            tone="bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.12),transparent_34%)]"
            icon={<Database size={22} className="text-sky-300" />}
            points={[
              'Stage 1–4 完整数据流水线',
              '样本统计、模板与 Router 数据浏览',
              '注册表与 LLM 配置管理',
            ]}
          />

          <EntryCard
            to="/training"
            title="模型训练"
            copy="Token Router 单 GPU 训练、任务调度、指标曲线与 checkpoint 管理。"
            tone="bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.12),transparent_34%)]"
            icon={<GitBranch size={22} className="text-emerald-300" />}
            points={[
              '单 GPU 训练任务创建与调度',
              '实时损失曲线、测试指标与分场景 F1',
              '任务列表、checkpoint 与运行日志',
            ]}
          />
        </main>
      </div>
    </div>
  )
}
