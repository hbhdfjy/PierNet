import { Database, MousePointerClick, Sparkles, Workflow, type LucideIcon } from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'

type PlatformId = 'synth' | 'new-synth' | 'simple-training' | 'training'

type PlatformSwitcherProps = {
  active: PlatformId
}

const PLATFORMS: Array<{ id: PlatformId; to: string; label: string; icon: LucideIcon; external?: boolean }> = [
  {
    id: 'synth',
    to: '/synth',
    label: '数据合成',
    icon: Database,
  },
  {
    id: 'new-synth',
    to: '/new-synth/',
    label: '新数据合成',
    icon: Sparkles,
    external: true,
  },
  {
    id: 'simple-training',
    to: '/training/simple',
    label: '简洁训练',
    icon: MousePointerClick,
  },
  {
    id: 'training',
    to: '/training',
    label: '复杂训练',
    icon: Workflow,
  },
]

function platformRouteActive(id: PlatformId, pathname: string): boolean {
  if (id === 'synth') return pathname === '/synth' || pathname.startsWith('/synth/')
  if (id === 'new-synth') return pathname === '/new-synth' || pathname.startsWith('/new-synth/')
  if (id === 'simple-training') return pathname === '/training/simple' || pathname.startsWith('/training/simple/')
  const isSimpleTrainingRoute = pathname === '/training/simple' || pathname.startsWith('/training/simple/')
  return pathname === '/training' || (pathname.startsWith('/training/') && !isSimpleTrainingRoute)
}

export function PlatformSwitcher({ active }: PlatformSwitcherProps) {
  const location = useLocation()

  return (
    <div className="platform-switcher" aria-label="平台切换">
      {PLATFORMS.map(platform => {
        const Icon = platform.icon
        const isActive = active === platform.id || platformRouteActive(platform.id, location.pathname)
        if (platform.external) {
          return (
            <a
              key={platform.id}
              href={platform.to}
              className={`platform-switcher__item ${isActive ? 'platform-switcher__item--active' : ''}`}
            >
              <Icon size={14} />
              <span>{platform.label}</span>
            </a>
          )
        }
        return (
          <NavLink
            key={platform.id}
            to={platform.to}
            className={() => `platform-switcher__item ${isActive ? 'platform-switcher__item--active' : ''}`}
          >
            <Icon size={14} />
            <span>{platform.label}</span>
          </NavLink>
        )
      })}
    </div>
  )
}
