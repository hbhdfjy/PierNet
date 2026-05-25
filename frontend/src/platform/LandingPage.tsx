import {
  ArrowRight,
  BarChart3,
  Boxes,
  CheckCircle2,
  Cpu,
  Database,
  GitBranch,
  Moon,
  Network,
  Server,
  Sun,
  Workflow,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Theme } from '../shared/theme'

type ModuleTone = 'synth' | 'training'

type ModuleCardProps = {
  to: string
  tone: ModuleTone
  icon: React.ReactNode
  title: string
  route: string
  copy: string
  metrics: Array<{ label: string; value: string }>
  bullets: string[]
}

function ModuleCard({ to, tone, icon, title, route, copy, metrics, bullets }: ModuleCardProps) {
  return (
    <Link to={to} className={`platform-module-card platform-module-card--${tone}`}>
      <div className="platform-module-card__header">
        <span className="platform-module-card__icon">{icon}</span>
        <span className="platform-module-card__route mono">{route}</span>
      </div>
      <div className="platform-module-card__main">
        <div>
          <h2>{title}</h2>
          <p>{copy}</p>
        </div>
        <ArrowRight size={18} />
      </div>
      <div className="platform-module-card__metrics">
        {metrics.map(item => (
          <div key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      <div className="platform-module-card__bullets">
        {bullets.map(item => (
          <span key={item}>
            <CheckCircle2 size={13} />
            {item}
          </span>
        ))}
      </div>
    </Link>
  )
}

function PipelineStep({ index, title, copy }: { index: string; title: string; copy: string }) {
  return (
    <div className="platform-pipeline-step">
      <span>{index}</span>
      <div>
        <strong>{title}</strong>
        <p>{copy}</p>
      </div>
    </div>
  )
}

function InfoPill({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="platform-info-pill">
      <span className="platform-info-pill__icon">{icon}</span>
      <div>
        <div className="platform-info-pill__label">{label}</div>
        <div className="platform-info-pill__value">{value}</div>
      </div>
    </div>
  )
}

export default function LandingPage({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <div className="platform-home-shell">
      <div className="platform-home-wrap">
        <header className="platform-topbar">
          <div className="platform-brand-lockup">
            <div className="platform-brand-lockup__mark">P</div>
            <div>
              <div className="platform-brand-lockup__title">PierNet 控制台</div>
              <div className="platform-brand-lockup__subtitle">物理工程数据合成与 Token Router 训练</div>
            </div>
          </div>
          <div className="platform-topbar__actions">
            <Link to="/synth" className="btn-ghost">
              数据合成
            </Link>
            <Link to="/training" className="btn-primary">
              自动训练
            </Link>
            <button type="button" onClick={toggleTheme} className="theme-toggle platform-theme-toggle">
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              <span>{theme === 'dark' ? '日间' : '夜间'}</span>
            </button>
          </div>
        </header>

        <main className="platform-home-main">
          <section className="platform-command-panel">
            <div className="platform-command-panel__copy">
              <div className="training-eyebrow">统一工作台</div>
              <h1>面向工程仿真数据生产与路由模型训练的一体化控制台</h1>
              <p>
                数据合成平台负责原始物理数据、语言模板、样本填充和 Router 数据构建；自动训练平台负责 GPU
                分配、任务监控、曲线分析、日志与权重管理。两个平台共享主题、入口和数据契约，但运行边界清晰。
              </p>
              <div className="platform-command-panel__actions">
                <Link to="/synth" className="btn-primary">
                  <Database size={15} />
                  打开数据平台
                </Link>
                <Link to="/training" className="btn-ghost">
                  <GitBranch size={15} />
                  打开训练平台
                </Link>
              </div>
            </div>

            <div className="platform-system-panel" aria-label="平台运行链路">
              <div className="platform-system-panel__top">
                <span>运行链路</span>
                <strong>API :8000 · UI :3000</strong>
              </div>
              <div className="platform-system-panel__metrics">
                <div>
                  <span>仿真器</span>
                  <strong>5</strong>
                </div>
                <div>
                  <span>数据阶段</span>
                  <strong>4</strong>
                </div>
                <div>
                  <span>训练模式</span>
                  <strong>1 GPU</strong>
                </div>
              </div>
              <div className="platform-pipeline">
                <PipelineStep index="01" title="物理与语言数据" copy="仿真产物、语言模板、填充样本" />
                <PipelineStep index="02" title="Router 数据" copy="阶段 4 构建训练输入与场景索引" />
                <PipelineStep index="03" title="训练闭环" copy="任务、曲线、日志、权重与文件管理" />
              </div>
            </div>
          </section>

          <section className="platform-launch-grid" aria-label="平台入口">
            <ModuleCard
              to="/synth"
              tone="synth"
              icon={<Database size={22} />}
              title="数据合成平台"
              route="/synth"
              copy="从科学计算原始数据出发，完成场景注册、模板生成、样本填充、Router 数据构建与文件管理。"
              metrics={[
                { label: '流程', value: '阶段 1-4' },
                { label: '对象', value: '数据生产' },
              ]}
              bullets={['HDF5 / JSONL / Parquet 产物管理', '模板、样本、路由数据可视化', '注册信息与 LLM 配置集中维护']}
            />
            <ModuleCard
              to="/training"
              tone="training"
              icon={<GitBranch size={22} />}
              title="自动训练平台"
              route="/training"
              copy="选择 Router 数据和可用 GPU，启动训练任务，持续查看指标、日志、权重和历史产物。"
              metrics={[
                { label: '模型', value: 'Token Router' },
                { label: '对象', value: '训练闭环' },
              ]}
              bullets={[
                'GPU 状态、任务状态与指标同步',
                '支持权重保留策略与文件管理',
                '曲线、日志、产物在同一详情页汇总',
              ]}
            />
          </section>

          <section className="platform-info-grid" aria-label="系统边界">
            <InfoPill label="部署入口" value="单 FastAPI 应用托管 API 与前端静态资源" icon={<Server size={16} />} />
            <InfoPill label="数据读取" value="清单、索引和分区产物用于迁移与快速浏览" icon={<BarChart3 size={16} />} />
            <InfoPill
              label="代码边界"
              value="synth / training 独立命名空间，shared 只放基础设施"
              icon={<Boxes size={16} />}
            />
            <InfoPill label="训练核心" value="Qwen embedding + full-sequence conv router" icon={<Cpu size={16} />} />
            <InfoPill label="平台联通" value="通过阶段 4 Router 数据契约衔接两个平台" icon={<Network size={16} />} />
            <InfoPill
              label="运行约束"
              value="当前是单机单卡训练控制台，不是通用集群平台"
              icon={<Workflow size={16} />}
            />
          </section>
        </main>
      </div>
    </div>
  )
}
