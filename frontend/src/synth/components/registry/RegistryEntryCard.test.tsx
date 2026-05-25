import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { RegistryEntryCard } from './RegistryEntryCard'
import type { RegistryEntry } from './registryTypes'

const entry: RegistryEntry = {
  domain_context: 'old context',
  output_description: 'old output',
}

describe('RegistryEntryCard', () => {
  it('keeps domain edits dirty when save fails', async () => {
    const onSave = vi.fn().mockRejectedValue(new Error('registry save denied'))
    const onDelete = vi.fn()

    render(<RegistryEntryCard entryKey="modflow/coastal" entry={entry} onSave={onSave} onDelete={onDelete} />)

    fireEvent.click(screen.getByRole('button', { name: /展开 modflow\/coastal/ }))
    const domainBox = screen.getByDisplayValue('old context') as HTMLTextAreaElement
    fireEvent.change(domainBox, { target: { value: 'new context' } })
    fireEvent.click(screen.getByRole('button', { name: /保存/ }))

    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith('modflow/coastal', { ...entry, domain_context: 'new context' }),
    )
    expect(await screen.findByText('registry save denied')).toBeTruthy()
    expect(domainBox.value).toBe('new context')
    expect(screen.getByRole('button', { name: /保存/ })).toBeTruthy()
  })

  it('shows delete failures and keeps the entry visible', async () => {
    const onSave = vi.fn()
    const onDelete = vi.fn().mockRejectedValue(new Error('registry delete denied'))

    render(<RegistryEntryCard entryKey="modflow/coastal" entry={entry} onSave={onSave} onDelete={onDelete} />)

    fireEvent.click(screen.getByRole('button', { name: /删除 modflow\/coastal/ }))
    fireEvent.click(screen.getByRole('button', { name: /确认删除 modflow\/coastal/ }))

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith('modflow/coastal'))
    expect(await screen.findByText('registry delete denied')).toBeTruthy()
    expect(screen.getByText('coastal')).toBeTruthy()
  })
})
