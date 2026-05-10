import { describe, expect, it } from 'vitest'
import { cn, formatBytes, getLineColor, getSimulatorBadgeClass } from './utils'

describe('frontend utility helpers', () => {
  it('formats byte counts with stable units', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(1023)).toBe('1023 B')
    expect(formatBytes(1024)).toBe('1.0 KB')
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB')
  })

  it('merges Tailwind classes predictably', () => {
    const hidden = false
    expect(cn('px-2', hidden ? 'hidden' : null, 'px-4')).toBe('px-4')
  })

  it('falls back for unknown simulators and cycles chart colors', () => {
    expect(getSimulatorBadgeClass('unknown')).toContain('bg-slate-700/50')
    expect(getLineColor(0)).toBe(getLineColor(10))
  })
})
