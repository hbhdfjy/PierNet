import { describe, expect, it } from 'vitest'

import {
  conversationTitle,
  loadConversationHistory,
  removeConversation,
  saveConversationHistory,
  type ConversationRecord,
  upsertConversation,
} from './conversationHistory'

function record(id: string, updatedAt: number): ConversationRecord {
  return {
    id,
    title: id,
    createdAt: updatedAt,
    updatedAt,
    phase: 'goal',
    goal: '',
    messages: [],
    jobId: null,
    job: null,
    workflow: null,
    selectedDataset: null,
    completionBoundaryId: null,
    assemblyProfile: null,
  }
}

function memoryStorage(): Storage {
  const values = new Map<string, string>()
  return {
    get length() {
      return values.size
    },
    clear: () => values.clear(),
    getItem: key => values.get(key) ?? null,
    key: index => [...values.keys()][index] ?? null,
    removeItem: key => values.delete(key),
    setItem: (key, value) => values.set(key, value),
  }
}

describe('conversation training history', () => {
  it('keeps the most recently updated conversation first', () => {
    const result = upsertConversation([record('older', 1), record('newer', 3)], record('older', 4))
    expect(result.map(item => item.id)).toEqual(['older', 'newer'])
  })

  it('round-trips valid records through storage', () => {
    const storage = memoryStorage()
    const records = [record('one', 1), record('two', 2)]
    saveConversationHistory(storage, records)
    expect(loadConversationHistory(storage).map(item => item.id)).toEqual(['two', 'one'])
  })

  it('ignores malformed storage and removes a selected record', () => {
    const storage = memoryStorage()
    storage.setItem('piern-conversation-training-history-v1', '{bad json')
    expect(loadConversationHistory(storage)).toEqual([])
    expect(removeConversation([record('one', 1), record('two', 2)], 'one').map(item => item.id)).toEqual(['two'])
  })

  it('creates compact history titles from natural-language goals', () => {
    expect(conversationTitle('  训练一个地下水模型  ')).toBe('训练一个地下水模型')
    expect(conversationTitle('这是一个非常长的训练目标，需要在历史列表中以紧凑方式展示并避免撑开布局')).toMatch(/…$/)
  })
})
