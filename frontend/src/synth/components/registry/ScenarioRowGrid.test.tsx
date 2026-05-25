import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ScenarioRowGrid } from './ScenarioRowGrid'

describe('ScenarioRowGrid', () => {
  it('shows save failures without leaving an unhandled action', async () => {
    const onSave = vi.fn().mockRejectedValue(new Error('registry save denied'))
    const onDelete = vi.fn()

    render(
      <ScenarioRowGrid
        simulator="modflow"
        scenario="coastal"
        description="old description"
        onSave={onSave}
        onDelete={onDelete}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /编辑 coastal/ }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'new description' } })
    fireEvent.click(screen.getByRole('button', { name: /保存/ }))

    await waitFor(() => expect(onSave).toHaveBeenCalledWith('modflow/coastal', 'new description'))
    expect(await screen.findByText(/保存失败：registry save denied/)).toBeTruthy()
  })

  it('shows delete failures and keeps the row visible', async () => {
    const onSave = vi.fn()
    const onDelete = vi.fn().mockRejectedValue(new Error('registry delete denied'))

    render(
      <ScenarioRowGrid
        simulator="modflow"
        scenario="coastal"
        description="old description"
        onSave={onSave}
        onDelete={onDelete}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /删除 coastal/ }))
    fireEvent.click(screen.getByRole('button', { name: /确认删除 coastal/ }))

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith('modflow/coastal'))
    expect(await screen.findByText(/删除失败：registry delete denied/)).toBeTruthy()
    expect(screen.getByText('coastal')).toBeTruthy()
  })
})
