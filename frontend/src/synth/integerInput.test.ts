import { describe, expect, it } from 'vitest'
import { parseIntegerField, parseIntegerText, parseOptionalIntegerField } from './integerInput'

describe('integer input helpers', () => {
  it('accepts only complete integer text within bounds', () => {
    expect(parseIntegerText('12', { min: 0 })).toBe(12)
    expect(parseIntegerText('12abc', { min: 0 })).toBeNull()
    expect(parseIntegerText('1.5', { min: 0 })).toBeNull()
    expect(parseIntegerText('1e3', { min: 0 })).toBeNull()
    expect(parseIntegerText('-1', { min: 0 })).toBeNull()
    expect(parseIntegerText('65', { max: 64 })).toBeNull()
  })

  it('keeps required integer fields on invalid edits', () => {
    expect(parseIntegerField('', 5, { min: 0 })).toBe(5)
    expect(parseIntegerField('bad', 5, { min: 0 })).toBe(5)
    expect(parseIntegerField('0', 5, { min: 0 })).toBe(0)
  })

  it('allows optional integer fields to be cleared', () => {
    expect(parseOptionalIntegerField('', 9, { min: 0 })).toBeNull()
    expect(parseOptionalIntegerField('4', null, { min: 0 })).toBe(4)
    expect(parseOptionalIntegerField('3', 9, { min: 4 })).toBe(9)
  })
})
