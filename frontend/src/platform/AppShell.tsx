import { Suspense, useEffect, useState, type CSSProperties, type ElementType, type ReactNode } from 'react'
import useSWR from 'swr'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Loader2,
  Moon,
  Shuffle,
  Square,
  Sun,
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { cn } from '../lib/utils'
import { SEED_MAX, SeedContext, parseSeedInput, readStoredSeed, writeStoredSeed } from '../lib/seedContext'
import type { Theme } from '../shared/theme'
import { PlatformSwitcher } from './PlatformSwitcher'
import { api } from '../lib/api'
import type { JobStatusSnapshot, TrainingJobSummary } from '../lib/types'

type ShellPlatform = 'synth' | 'simple-training' | 'training'
type NavTone = 'amber' | 'sky' | 'violet' | 'emerald' | 'rose' | 'neutral'

export type ShellNavItem = {
  to: string
  label: string
  icon: ElementType
  end?: boolean
  tone?: NavTone
  step?: string
  rightIcon?: ElementType
}

export type ShellNavGroup = {
  label: string
  items?: ShellNavItem[]
  note?: ReactNode
}

const TONES: Record<NavTone, { accent: string; soft: string; border: string }> = {
  amber: { accent: 'hsl(38 92% 58%)', soft: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.24)' },
  sky: { accent: 'hsl(199 89% 58%)', soft: 'rgba(14, 165, 233, 0.12)', border: 'rgba(14, 165, 233, 0.24)' },
  violet: { accent: 'hsl(262 85% 68%)', soft: 'rgba(139, 92, 246, 0.12)', border: 'rgba(139, 92, 246, 0.24)' },
  emerald: { accent: 'hsl(158 66% 48%)', soft: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.22)' },
  rose: { accent: 'hsl(347 83% 62%)', soft: 'rgba(244, 63, 94, 0.12)', border: 'rgba(244, 63, 94, 0.22)' },
  neutral: { accent: 'hsl(var(--brand))', soft: 'hsl(var(--brand) / 0.1)', border: 'hsl(var(--brand) / 0.22)' },
}

function useSeedState(): [number, (v: number) => void] {
  const [seed, setSeedRaw] = useState<number>(() => readStoredSeed())

  const setSeed = (value: number) => {
    setSeedRaw(writeStoredSeed(value))
  }

  return [seed, setSeed]
}

export function PageFallback() {
  return <div className="page-fallback">加载中...</div>
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="app-section-label">
      <span className="label whitespace-nowrap">{children}</span>
      <div className="app-section-label__line" />
    </div>
  )
}

function ShellNavLink({ item }: { item: ShellNavItem }) {
  const Icon = item.icon
  const RightIcon = item.rightIcon
  const tone = TONES[item.tone ?? 'neutral']
  const style = {
    ['--tone' as string]: tone.accent,
    ['--tone-soft' as string]: tone.soft,
    ['--tone-border' as string]: tone.border,
  } as CSSProperties

  return (
    <NavLink
      to={item.to}
      end={item.end}
      style={style}
      className={({ isActive }) => cn('nav-item', isActive && 'nav-item--active')}
    >
      {({ isActive }) => (
        <>
          <span className="nav-item__rail" />
          <span className={cn('nav-item__icon', item.step && 'nav-item__icon--step')}>
            {item.step ? item.step : <Icon size={14} />}
          </span>
          <span className="nav-item__label">{item.label}</span>
          {!isActive && RightIcon && (
            <span className="nav-item__hint">
              <RightIcon size={11} />
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}

const ACTIVE_SYNTH_STATUSES = new Set(['queued', 'starting', 'running', 'evaluating', 'stopping'])
const ACTIVE_TRAINING_STATUSES = new Set(['queued', 'starting', 'running', 'evaluating', 'stopping'])

function shortStatus(status: string): string {
  switch (status) {
    case 'queued':
      return '排队'
    case 'starting':
      return '启动'
    case 'running':
      return '运行'
    case 'evaluating':
      return '评估'
    case 'stopping':
      return '停止中'
    default:
      return status
  }
}

function TaskCenter() {
  const [open, setOpen] = useState(false)
  const [stoppingId, setStoppingId] = useState<string | null>(null)
  const { data: synthJobs, mutate: refreshSynth } = useSWR<JobStatusSnapshot[]>(
    'app-task-center-synth-jobs',
    () => api.listGenerationJobs(),
    { refreshInterval: 5000, revalidateOnFocus: false },
  )
  const { data: trainingJobs, mutate: refreshTraining } = useSWR<TrainingJobSummary[]>(
    'app-task-center-training-jobs',
    api.getTrainingJobs,
    { refreshInterval: 5000, revalidateOnFocus: false },
  )

  const activeSynth = (synthJobs ?? []).filter(job => ACTIVE_SYNTH_STATUSES.has(job.status))
  const activeTraining = (trainingJobs ?? []).filter(job => ACTIVE_TRAINING_STATUSES.has(job.status))
  const totalActive = activeSynth.length + activeTraining.length
  const queuedCount = [...activeSynth, ...activeTraining].filter(job => job.status === 'queued').length
  const runningCount = totalActive - queuedCount
  const lockKeys = Array.from(new Set(activeSynth.flatMap(job => job.lock_keys ?? []))).slice(0, 4)
  const isBusy = totalActive > 0

  const stopSynth = async (jobId: string) => {
    setStoppingId(jobId)
    try {
      await api.stopGeneration(jobId)
      await refreshSynth()
    } finally {
      setStoppingId(null)
    }
  }

  const stopTraining = async (jobId: string) => {
    setStoppingId(jobId)
    try {
      await api.stopTrainingJob(jobId)
      await refreshTraining()
    } finally {
      setStoppingId(null)
    }
  }

  return (
    <div className={cn('task-center', open && 'task-center--open', isBusy && 'task-center--busy')}>
      <button className="task-center__summary" onClick={() => setOpen(v => !v)}>
        <span className="task-center__icon">
          <Activity size={13} />
        </span>
        <span className="task-center__main">
          <span className="task-center__title-row">
            <span className="task-center__title">任务中心</span>
            <span
              className={cn('task-center__state', isBusy ? 'task-center__state--busy' : 'task-center__state--idle')}
            >
              {isBusy ? `${totalActive} 活动` : '空闲'}
            </span>
          </span>
          <span className="task-center__metrics">
            <span>
              <strong>{runningCount}</strong>运行
            </span>
            <span>
              <strong>{queuedCount}</strong>排队
            </span>
            <span>
              <strong>{lockKeys.length}</strong>锁
            </span>
          </span>
        </span>
        {open ? (
          <ChevronUp size={13} className="task-center__chevron" />
        ) : (
          <ChevronDown size={13} className="task-center__chevron" />
        )}
      </button>

      {open && (
        <div className="task-center__body">
          {totalActive === 0 ? (
            <div className="task-center__empty">
              <CheckCircle2 size={14} />
              <div>
                <div className="task-center__empty-title">当前无活动任务</div>
                <div className="task-center__empty-copy">可以直接启动填充、路由构建或训练。</div>
              </div>
            </div>
          ) : (
            <div className="task-center__list">
              {activeSynth.map(job => (
                <div key={job.job_id} className="task-center__item">
                  <div className="task-center__item-head">
                    <div className="task-center__item-title">
                      <span className="task-center__platform">合成</span>
                      <span>{job.job_type ?? 'job'}</span>
                    </div>
                    <span className="task-center__status">{shortStatus(job.status)}</span>
                  </div>
                  <div className="task-center__item-meta mono">{job.job_id}</div>
                  <div className="task-center__locks">
                    {(job.lock_keys && job.lock_keys.length > 0 ? job.lock_keys : ['fill/router 互斥资源']).map(
                      lock => (
                        <span key={lock} className="task-center__lock mono">
                          {lock}
                        </span>
                      ),
                    )}
                  </div>
                  {job.status !== 'stopping' && (
                    <button
                      className="task-center__stop"
                      onClick={() => stopSynth(job.job_id)}
                      disabled={stoppingId === job.job_id}
                    >
                      {stoppingId === job.job_id ? (
                        <Loader2 size={11} className="animate-spin" />
                      ) : (
                        <Square size={11} />
                      )}
                      终止
                    </button>
                  )}
                </div>
              ))}

              {activeTraining.map(job => (
                <div key={job.job_id} className="task-center__item">
                  <div className="task-center__item-head">
                    <div className="task-center__item-title">
                      <span className="task-center__platform task-center__platform--training">训练</span>
                      <span>GPU {job.gpu_id}</span>
                    </div>
                    <span className="task-center__status">{shortStatus(job.status)}</span>
                  </div>
                  <div className="task-center__item-meta mono">{job.job_id}</div>
                  <div className="task-center__item-note">{job.name}</div>
                  {job.status !== 'stopping' && (
                    <button
                      className="task-center__stop"
                      onClick={() => stopTraining(job.job_id)}
                      disabled={stoppingId === job.job_id}
                    >
                      {stoppingId === job.job_id ? (
                        <Loader2 size={11} className="animate-spin" />
                      ) : (
                        <Square size={11} />
                      )}
                      终止
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}

          {lockKeys.length > 0 && (
            <div className="task-center__hint">
              <AlertTriangle size={12} />
              <span className="truncate">409 时优先检查这些锁：{lockKeys.join('、')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function AppShell({
  platform,
  mark,
  title,
  subtitle,
  navGroups,
  theme,
  toggleTheme,
  children,
}: {
  platform: ShellPlatform
  mark: string
  title: string
  subtitle: string
  navGroups: ShellNavGroup[]
  theme: Theme
  toggleTheme: () => void
  children: ReactNode
}) {
  const [seed, setSeed] = useSeedState()
  const [seedInput, setSeedInput] = useState(String(seed))
  const location = useLocation()

  useEffect(() => {
    setSeedInput(String(seed))
  }, [seed])

  useEffect(() => {
    const active = document.querySelector<HTMLElement>('.app-nav .nav-item--active')
    const nav = active?.closest<HTMLElement>('.app-nav')
    if (!active || !nav) return
    if (nav.scrollWidth > nav.clientWidth + 2) {
      const targetLeft = active.offsetLeft - nav.offsetLeft - 8
      nav.scrollTo({ left: Math.max(0, targetLeft), behavior: 'auto' })
    }
    if (nav.scrollHeight > nav.clientHeight + 2) {
      const maxTop = nav.scrollHeight - nav.clientHeight
      const targetTop =
        active.offsetTop - nav.offsetTop - Math.max(0, Math.round((nav.clientHeight - active.offsetHeight) / 2))
      nav.scrollTo({
        top: Math.max(0, Math.min(maxTop, targetTop)),
        behavior: 'auto',
      })
    }
  }, [location.pathname])

  return (
    <SeedContext.Provider value={{ seed, setSeed }}>
      <div className={`app-shell app-shell--${platform}`}>
        <aside className="app-sidebar">
          <div className="app-brand">
            <div className="app-brand__mark-wrap">
              <div className="app-brand__mark">{mark}</div>
              <span className="app-brand__status" />
            </div>
            <div className="min-w-0">
              <div className="app-brand__title">{title}</div>
              <div className="app-brand__subtitle">{subtitle}</div>
            </div>
          </div>

          <PlatformSwitcher active={platform} />

          <nav className="app-nav">
            {navGroups.map(group => (
              <div key={group.label} className="app-nav__group">
                <SectionLabel>{group.label}</SectionLabel>
                {group.note && <div className="app-sidebar-note">{group.note}</div>}
                {group.items && group.items.length > 0 && (
                  <div className="app-nav__items">
                    {group.items.map(item => (
                      <ShellNavLink key={`${item.to}:${item.label}`} item={item} />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>

          <div className="app-sidebar__footer">
            <TaskCenter />

            <div className="app-seed-card">
              <div className="app-seed-card__label-row">
                <div className="app-seed-card__label">
                  <Shuffle size={11} />
                  <span>随机种子</span>
                </div>
              </div>
              <input
                type="number"
                min={0}
                max={SEED_MAX}
                value={seedInput}
                onChange={event => {
                  setSeedInput(event.target.value)
                  const parsed = parseSeedInput(event.target.value)
                  if (parsed !== null) setSeed(parsed)
                }}
                onBlur={() => {
                  const value = parseSeedInput(seedInput) ?? 42
                  setSeed(value)
                  setSeedInput(String(value))
                }}
                className="input app-seed-input mono"
              />
            </div>

            <div className="app-sidebar__footer-row">
              <button type="button" onClick={toggleTheme} className="theme-toggle">
                {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
                <span>{theme === 'dark' ? '日间' : '夜间'}</span>
              </button>
            </div>
          </div>
        </aside>

        <main className="app-main">
          <Suspense fallback={<PageFallback />}>{children}</Suspense>
        </main>
      </div>
    </SeedContext.Provider>
  )
}
