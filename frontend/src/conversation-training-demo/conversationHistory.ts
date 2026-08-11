import type { AssemblyProfile, TrainingDataset, TrainingJob, WorkflowSnapshot } from './workflowApi'

export type ConversationPhase = 'goal' | 'data' | 'preparing' | 'ready' | 'training' | 'complete' | 'error'

export type ConversationMessage = {
  id: number
  role: 'assistant' | 'user'
  content: string
}

export type ConversationRecord = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  phase: ConversationPhase
  goal: string
  messages: ConversationMessage[]
  jobId: string | null
  job: TrainingJob | null
  workflow: WorkflowSnapshot | null
  selectedDataset: TrainingDataset | null
  completionBoundaryId: number | null
  assemblyProfile: AssemblyProfile | null
}

export const CONVERSATION_HISTORY_KEY = 'piern-conversation-training-history-v1'
export const CURRENT_CONVERSATION_KEY = 'piern-conversation-training-current-v1'

const MAX_HISTORY_ITEMS = 30

function isConversationRecord(value: unknown): value is ConversationRecord {
  if (!value || typeof value !== 'object') return false
  const record = value as Partial<ConversationRecord>
  return (
    typeof record.id === 'string' &&
    typeof record.title === 'string' &&
    typeof record.createdAt === 'number' &&
    typeof record.updatedAt === 'number' &&
    typeof record.phase === 'string' &&
    Array.isArray(record.messages)
  )
}

export function loadConversationHistory(storage: Storage): ConversationRecord[] {
  try {
    const raw = storage.getItem(CONVERSATION_HISTORY_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isConversationRecord).sort((left, right) => right.updatedAt - left.updatedAt)
  } catch {
    return []
  }
}

export function saveConversationHistory(storage: Storage, records: ConversationRecord[]): void {
  storage.setItem(CONVERSATION_HISTORY_KEY, JSON.stringify(records.slice(0, MAX_HISTORY_ITEMS)))
}

export function upsertConversation(records: ConversationRecord[], record: ConversationRecord): ConversationRecord[] {
  return [record, ...records.filter(item => item.id !== record.id)]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, MAX_HISTORY_ITEMS)
}

export function removeConversation(records: ConversationRecord[], conversationId: string): ConversationRecord[] {
  return records.filter(item => item.id !== conversationId)
}

export function createConversationId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  return `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function conversationTitle(goal: string): string {
  const normalized = goal.trim().replace(/\s+/g, ' ')
  if (!normalized) return '新对话'
  return normalized.length > 28 ? `${normalized.slice(0, 28)}…` : normalized
}
