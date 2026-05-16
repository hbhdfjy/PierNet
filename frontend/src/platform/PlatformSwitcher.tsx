import { Database, Workflow } from 'lucide-react'
import { NavLink } from 'react-router-dom'

type PlatformSwitcherProps = {
  active: 'synth' | 'training'
}

const PLATFORMS = [
  {
    id: 'synth',
    to: '/synth',
    label: '数据合成',
    icon: Database,
  },
  {
    id: 'training',
    to: '/training',
    label: '自动训练',
    icon: Workflow,
  },
] as const

export function PlatformSwitcher({ active }: PlatformSwitcherProps) {
  return (
    <div className="platform-switcher" aria-label="平台切换">
      {PLATFORMS.map(platform => {
        const Icon = platform.icon
        const isActive = active === platform.id
        return (
          <NavLink
            key={platform.id}
            to={platform.to}
            className={({ isActive: routeActive }) =>
              `platform-switcher__item ${isActive || routeActive ? 'platform-switcher__item--active' : ''}`
            }
          >
            <Icon size={14} />
            <span>{platform.label}</span>
          </NavLink>
        )
      })}
    </div>
  )
}
