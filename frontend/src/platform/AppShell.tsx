import { Suspense, useEffect, useState, type CSSProperties, type ElementType, type ReactNode } from 'react'
import { Moon, Shuffle, Sun } from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { cn } from '../lib/utils'
import { SeedContext, readStoredSeed, writeStoredSeed } from '../lib/seedContext'
import type { Theme } from '../shared/theme'
import { PlatformSwitcher } from './PlatformSwitcher'

type ShellPlatform = 'synth' | 'training'
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
    const targetLeft = active.offsetLeft - nav.offsetLeft - 8
    nav.scrollTo({ left: Math.max(0, targetLeft), behavior: 'auto' })
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
                value={seedInput}
                onChange={event => {
                  setSeedInput(event.target.value)
                  const parsed = parseInt(event.target.value, 10)
                  if (!isNaN(parsed) && parsed >= 0) setSeed(parsed)
                }}
                onBlur={() => {
                  const parsed = parseInt(seedInput, 10)
                  const value = isNaN(parsed) ? 42 : Math.max(0, parsed)
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
