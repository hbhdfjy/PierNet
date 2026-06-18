import { Activity, ArrowRight, Brain, CheckCircle2, Cpu, Layers3, MousePointerClick, Workflow } from 'lucide-react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { api } from '../../lib/api'
import type { Text2CompOverview, TrainingOverview } from '../../lib/types'
import { formatCount, gpuUsageLabel } from '../shared'

type ModuleTone = 'emerald' | 'sky' | 'violet'

type SimpleModuleCardProps = {
  to: string
  tone: ModuleTone
  icon: React.ReactNode
  title: string
  copy: string
  primary: string
  meta: Array<{ label: string; value: string }>
  bullets: string[]
}

function SimpleModuleCard({ to, tone, icon, title, copy, primary, meta, bullets }: SimpleModuleCardProps) {
  return (
    <Link to={to} className={`training-simple-module-card training-simple-module-card--${tone}`}>
      <div className="training-simple-module-card__top">
        <span className="training-simple-module-card__icon">{icon}</span>
        <span className="training-simple-module-card__action">
          {primary}
          <ArrowRight size={14} />
        </span>
      </div>
      <div>
        <h2>{title}</h2>
        <p>{copy}</p>
      </div>
      <div className="training-simple-module-card__metrics">
        {meta.map(item => (
          <div key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      <div className="training-simple-module-card__bullets">
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

function bestTrainingGpuLabel(data: TrainingOverview | undefined): string {
  const gpus = data?.gpus ?? []
  if (!gpus.length) return '等待 GPU'
  const available = gpus.filter(gpu => gpu.available)
  return `${available.length}/${gpus.length} 可用`
}

function bestText2CompGpuLabel(data: Text2CompOverview | undefined): string {
  const gpus = data?.gpus ?? []
  if (!gpus.length) return '等待 GPU'
  const best = [...gpus].sort((a, b) => Number(b.available) - Number(a.available))[0]
  return `GPU ${best.index} · ${best.available ? '可用' : '排队'}`
}

export default function TrainingSimpleHubPage() {
  const { data: training } = useSWR<TrainingOverview>('training-overview', api.getTrainingOverview, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const { data: text2comp } = useSWR<Text2CompOverview>('text2comp-overview', api.getText2CompOverview, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })
  const { data: assembly } = useSWR('assembly-status', api.getAssemblyStatus, {
    refreshInterval: 5000,
    revalidateOnFocus: false,
  })

  const routerSamples = training?.datasets.reduce((sum, item) => sum + item.total_count, 0) ?? 0
  const text2CompDatasets = text2comp?.datasets?.length ?? 0
  const loadedCount = assembly?.loaded_models
    ? [
        assembly.loaded_models.llm,
        assembly.loaded_models.router,
        assembly.loaded_models.text2comp,
        assembly.loaded_models.fno,
        assembly.loaded_models.uploaded_expert,
      ].filter(item => item?.loaded).length
    : 0
  const assemblyGpu = assembly?.gpus?.[0]

  return (
    <div className="training-page">
      <div className="training-page__body training-simple-page">
        <div className="space-y-4 p-4">
          <section className="training-hero training-hero--compact training-simple-workbench-hero">
            <div>
              <div className="training-eyebrow">简洁训练</div>
              <h1 className="training-simple-hero__title">PierNet 训练工作台</h1>
              <p className="training-copy">按模块启动和管理 PierNet 训练链路，只保留粗粒度决策。</p>
            </div>
            <div className="training-simple-workbench-hero__status">
              <div>
                <span>Router 样本</span>
                <strong>{formatCount(routerSamples)}</strong>
              </div>
              <div>
                <span>Text2Comp 数据</span>
                <strong>{formatCount(text2CompDatasets)}</strong>
              </div>
              <div>
                <span>拼装状态</span>
                <strong>{loadedCount > 0 ? `${loadedCount} 已加载` : '未加载'}</strong>
              </div>
            </div>
          </section>

          <section className="training-simple-module-grid" aria-label="简洁训练模块">
            <SimpleModuleCard
              to="/training/simple/router"
              tone="emerald"
              icon={<MousePointerClick size={22} />}
              title="Router 训练"
              copy="选择大场景、子场景和资源策略，启动 Token Router 训练。"
              primary="进入 Router"
              meta={[
                { label: '样本', value: formatCount(routerSamples) },
                { label: 'GPU', value: bestTrainingGpuLabel(training) },
              ]}
              bullets={['重新训练或继续训练', '简洁进度与结果', '终止、删除和历史复用']}
            />
            <SimpleModuleCard
              to="/training/simple/text2comp"
              tone="sky"
              icon={<Brain size={22} />}
              title="文生计算训练"
              copy="选择专家域和训练数据，启动 Text2Comp 模块训练。"
              primary="进入文生计算"
              meta={[
                { label: '专家域', value: formatCount(text2comp?.expert_models?.length ?? 0) },
                { label: '资源', value: bestText2CompGpuLabel(text2comp) },
              ]}
              bullets={['自动使用默认训练策略', '按专家域筛选数据集', '展示任务状态和最近损失']}
            />
            <SimpleModuleCard
              to="/training/simple/assembly"
              tone="violet"
              icon={<Layers3 size={22} />}
              title="模型拼装"
              copy="自动选择 LLM、Router、Text2Comp 和 Expert，完成 PierNet 推理链路装载。"
              primary="进入拼装"
              meta={[
                { label: '组件', value: `${loadedCount}/5 已加载` },
                {
                  label: 'GPU',
                  value: assemblyGpu
                    ? `GPU ${assemblyGpu.index} · ${gpuUsageLabel(assemblyGpu.memory_used_mb, assemblyGpu.memory_total_mb)}`
                    : '等待 GPU',
                },
              ]}
              bullets={['一键加载或卸载', 'FNO / Uploaded Expert', '可执行粗粒度推理测试']}
            />
          </section>

          <section className="training-simple-workflow-strip">
            <div>
              <Workflow size={16} />
              <span>推荐顺序</span>
            </div>
            <strong>Router 训练</strong>
            <ArrowRight size={14} />
            <strong>文生计算训练</strong>
            <ArrowRight size={14} />
            <strong>模型拼装与测试</strong>
            <span className="training-simple-workflow-strip__tail">每一步都可以单独进入，不需要配置复杂训练参数。</span>
          </section>

          <section className="training-simple-health-grid">
            <div className="training-simple-health-card">
              <Cpu size={16} />
              <div>
                <span>Router GPU</span>
                <strong>{bestTrainingGpuLabel(training)}</strong>
              </div>
            </div>
            <div className="training-simple-health-card">
              <Activity size={16} />
              <div>
                <span>活跃 Router 任务</span>
                <strong>{formatCount(training?.running_job_count ?? 0)}</strong>
              </div>
            </div>
            <div className="training-simple-health-card">
              <Brain size={16} />
              <div>
                <span>活跃 Text2Comp 任务</span>
                <strong>{formatCount(text2comp?.running_job_count ?? 0)}</strong>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
