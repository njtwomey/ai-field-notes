/** Shared state between citation markers and the margin notes: which markers exist, and which reference is hovered. */
import { createContext, useContext, type RefObject } from 'react'

export type Anchor = { key: string; el: HTMLElement }

type SidenoteState = {
  register: (id: string, anchor: Anchor | null) => void
  anchors: Map<string, Anchor>
  version: number
  active: string | null
  setActive: (key: string | null) => void
  /** Element whose size changes should trigger a re-layout: the article column. */
  contentRef: RefObject<HTMLElement | null>
}

export const SidenoteContext = createContext<SidenoteState | null>(null)

export function useSidenotes(): SidenoteState | null {
  return useContext(SidenoteContext)
}
