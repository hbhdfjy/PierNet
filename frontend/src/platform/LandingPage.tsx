import {
  ArrowRight,
  BarChart3,
  Boxes,
  Cpu,
  Database,
  GitBranch,
  Moon,
  Network,
  Server,
  ShieldCheck,
  Sparkles,
  Sun,
  Workflow,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Theme } from '../shared/theme'

type EntryTone = 'synth' | 'training'

type EntryCardProps = {
  to: string
  title: string
  route: string
  copy: string
  tone: EntryTone
  icon: React.ReactNode
  points: string[]
  stats: Array<{ label: string; value: string }>
}

type InfoItem = {
  label: string
  value: string
  icon: React.ReactNode
}

function EntryCard({ to, title, route, copy, tone, icon, points, stats }: EntryCardProps) {
  return (
    <Link to={to} className={`platform-entry-card platform-entry-card--${tone} group`}>
      <div className="platform-entry-card__topline">
        <div className="platform-entry-card__icon">{icon}</div>
        <span className="platform-entry-card__route mono">{route}</span>
      </div>

      <div className="platform-entry-card__main">
        <div>
          <h2 className="platform-entry-card__title">{title}</h2>
          <p className="platform-entry-card__copy">{copy}</p>
        </div>
        <ArrowRight size={20} className="platform-entry-card__arrow" />
      </div>

      <div className="platform-entry-card__stats">
        {stats.map(item => (
          <div key={item.label} className="platform-entry-card__stat">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

      <div className="platform-entry-card__points">
        {points.map(point => (
          <div key={point} className="platform-entry-card__point">
            <ShieldCheck size={14} />
            <span>{point}</span>
          </div>
        ))}
      </div>
    </Link>
  )
}

function InfoPill({ label, value, icon }: InfoItem) {
  return (
    <div className="platform-info-pill">
      <div className="platform-info-pill__icon">{icon}</div>
      <div>
        <div className="platform-info-pill__label">{label}</div>
        <div className="platform-info-pill__value">{value}</div>
      </div>
    </div>
  )
}

function FlowStep({ step, title, copy }: { step: string; title: string; copy: string }) {
  return (
    <div className="platform-flow-step">
      <span className="platform-flow-step__index">{step}</span>
      <div>
        <div className="platform-flow-step__title">{title}</div>
        <div className="platform-flow-step__copy">{copy}</div>
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
              <div className="platform-brand-lockup__title">PiERN 工作台</div>
              <div className="platform-brand-lockup__subtitle">物理工程数据与 Token Router 训练入口</div>
            </div>
          </div>
          <div className="platform-topbar__actions">
            <Link to="/synth" className="btn-ghost">
              进入数据合成
            </Link>
            <Link to="/training" className="btn-primary">
              进入训练平台
            </Link>
            <button type="button" onClick={toggleTheme} className="theme-toggle platform-theme-toggle">
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              <span>{theme === 'dark' ? '日间' : '夜间'}</span>
            </button>
          </div>
        </header>

        <main className="platform-home-main">
          <section className="platform-hero-panel">
            <div className="platform-hero-panel__content">
              <div className="training-eyebrow">
                <Sparkles size={13} />
                <span>统一工作台</span>
              </div>
              <h1 className="platform-hero-title">从物理仿真到路由训练的统一工作台</h1>
              <p className="platform-hero-copy">
                `/synth` 负责阶段 1-4 数据合成，`/training` 负责 Token Router
                单卡训练。两个平台逻辑隔离，统一使用同一个服务入口、主题系统和数据契约。
              </p>
              <div className="platform-hero-actions">
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

            <div className="platform-hero-board" aria-label="PiERN system summary">
              <div className="platform-hero-board__header">
                <span>运行边界</span>
                <strong>后端 :8000</strong>
              </div>
              <div className="platform-hero-metrics">
                <div>
                  <span>模拟器</span>
                  <strong>5</strong>
                </div>
                <div>
                  <span>场景</span>
                  <strong>22</strong>
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
              <div className="platform-flow-line">
                <FlowStep step="01" title="数据平台" copy="仿真、模板、样本、Router 数据" />
                <FlowStep step="02" title="训练平台" copy="训练任务、曲线、日志、权重" />
              </div>
            </div>
          </section>

          <section className="platform-home-grid" aria-label="Platform entry cards">
            <EntryCard
              to="/synth"
              title="数据合成平台"
              route="/synth"
              copy="面向物理与工程时序数据，覆盖仿真、注册、模板、样本填充与 Router 数据构建。"
              tone="synth"
              icon={<Database size={23} />}
              stats={[
                { label: '流程', value: '阶段 1-4' },
                { label: '仿真器', value: '5 类' },
              ]}
              points={[
                'HDF5 / JSONL 源产物统一管理',
                '模板、样本、路由数据可视化浏览',
                'LLM 配置、注册信息与观测配置集中维护',
              ]}
            />

            <EntryCard
              to="/training"
              title="模型训练平台"
              route="/training"
              copy="聚焦 Token Router：选择路由数据、分配空闲 GPU、查看训练曲线与任务产物。"
              tone="training"
              icon={<GitBranch size={23} />}
              stats={[
                { label: '模型', value: 'Token Router' },
                { label: '设备', value: '单卡 GPU' },
              ]}
              points={[
                '按大场景管理训练数据与任务',
                '实时查看 loss、accuracy、F1 与日志',
                '支持终止、删除历史任务与 权重 管理',
              ]}
            />
          </section>

          <section className="platform-info-grid" aria-label="Required project information">
            <InfoPill label="部署入口" value="单 FastAPI 应用托管 API 与前端静态资源" icon={<Server size={16} />} />
            <InfoPill label="数据读取" value="阶段 2-4 使用清单和索引加速浏览" icon={<BarChart3 size={16} />} />
            <InfoPill
              label="代码边界"
              value="synth / training 独立命名空间，shared 只放基础设施"
              icon={<Boxes size={16} />}
            />
            <InfoPill label="训练核心" value="Qwen embedding + full-sequence conv router" icon={<Cpu size={16} />} />
            <InfoPill
              label="平台联通"
              value="两个平台仅通过入口链接与阶段 4 数据契约衔接"
              icon={<Network size={16} />}
            />
            <InfoPill label="运行约束" value="训练平台当前不是通用多卡训练系统" icon={<Workflow size={16} />} />
          </section>
        </main>
      </div>
    </div>
  )
}
