import { ArrowRight, Boxes, Database, MessageSquare, Moon, Sparkles, Sun, Workflow } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Theme } from '../shared/theme'

const workflowSteps = [
  {
    icon: Database,
    title: '接入数据',
    copy: '连接自己的时序数据与专家模型',
  },
  {
    icon: Sparkles,
    title: '准备任务',
    copy: '定义希望模型完成的科学任务',
  },
  {
    icon: Workflow,
    title: '自动训练',
    copy: '由平台组织数据与模型训练流程',
  },
  {
    icon: Boxes,
    title: '拼装应用',
    copy: '通过自然语言调用和验证模型',
  },
]

export default function LandingPage({ theme, toggleTheme }: { theme: Theme; toggleTheme: () => void }) {
  return (
    <div className="platform-home-shell">
      <div className="platform-home-wrap">
        <header className="platform-topbar">
          <div className="platform-brand-lockup">
            <div className="platform-brand-lockup__mark">P</div>
            <div>
              <div className="platform-brand-lockup__title">Piern</div>
              <div className="platform-brand-lockup__subtitle">时序语言多模态训练</div>
            </div>
          </div>
          <div className="platform-topbar__actions">
            <Link to="/training/simple" className="btn-primary">
              进入平台
              <ArrowRight size={15} />
            </Link>
            <button
              type="button"
              onClick={toggleTheme}
              className="theme-toggle platform-theme-toggle"
              aria-label={theme === 'dark' ? '切换到日间模式' : '切换到夜间模式'}
              title={theme === 'dark' ? '切换到日间模式' : '切换到夜间模式'}
            >
              {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </header>

        <main className="platform-home-main">
          <section className="platform-intro">
            <img className="platform-intro__visual" src="/assets/piern-home-hero.jpg" alt="" aria-hidden="true" />
            <div className="platform-intro__content">
              <div className="platform-intro__eyebrow">Piern</div>
              <h1>工业与科学时序语言多模态大模型自动化训练平台</h1>
              <p className="platform-intro__copy">
                让工业与科学计算模型理解自然语言，用自己的时序数据和专家模型，完成从训练到应用的全过程。
              </p>
              <div className="platform-intro__actions">
                <a href="/new-synth/" className="btn-primary">
                  开始创建
                  <ArrowRight size={16} />
                </a>
                <Link to="/training/simple" className="platform-intro__secondary-link">
                  查看已有任务
                  <MessageSquare size={15} />
                </Link>
              </div>
              <div className="platform-intro__scope" aria-label="适用场景">
                <span>适用场景</span>
                <strong>能源与气候</strong>
                <strong>流体与地下水</strong>
                <strong>电力系统</strong>
                <strong>工业过程</strong>
              </div>
            </div>
          </section>

          <section className="platform-workflow-section" aria-labelledby="platform-workflow-title">
            <div className="platform-section-heading">
              <div>
                <span>一条完整路径</span>
                <h2 id="platform-workflow-title">从自己的数据，到可使用的模型</h2>
              </div>
              <p>平台将复杂过程组织成连续步骤，首次使用也能沿流程完成。</p>
            </div>
            <ol className="platform-workflow">
              {workflowSteps.map((step, index) => {
                const Icon = step.icon
                return (
                  <li key={step.title}>
                    <div className="platform-workflow__index">{String(index + 1).padStart(2, '0')}</div>
                    <div className="platform-workflow__icon">
                      <Icon size={18} />
                    </div>
                    <div className="platform-workflow__copy">
                      <strong>{step.title}</strong>
                      <p>{step.copy}</p>
                    </div>
                  </li>
                )
              })}
            </ol>
            <div className="platform-workflow-section__action">
              <a href="/new-synth/">
                打开流程向导
                <ArrowRight size={14} />
              </a>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}
