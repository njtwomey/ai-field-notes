import { createContext, useContext } from 'react'
import type { NoteMeta } from '@/lib/content'

export const NoteContext = createContext<NoteMeta | null>(null)

export function useCurrentNote(): NoteMeta {
  const note = useContext(NoteContext)
  if (!note) throw new Error('content components must render inside a note page')
  return note
}
