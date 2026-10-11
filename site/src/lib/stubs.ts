import { useSyncExternalStore } from 'react'
import { notes, type NoteMeta } from '@/lib/content'

/**
 * A per-viewer preference: hide placeholder notes (`status: stub`) from the home page, the browse page and the topic
 * counts, to see what is actually written. Kept in localStorage; when storage is unavailable it lasts for the session.
 */
const STORAGE_KEY = 'mlc-hide-stubs'
const listeners = new Set<() => void>()

function read(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

let hidden = read()

function subscribe(listener: () => void) {
  listeners.add(listener)
  // Another tab changed the preference.
  const onStorage = (e: StorageEvent) => {
    if (e.key !== STORAGE_KEY) return
    hidden = read()
    listener()
  }
  window.addEventListener('storage', onStorage)
  return () => {
    listeners.delete(listener)
    window.removeEventListener('storage', onStorage)
  }
}

export function setHideStubs(value: boolean) {
  hidden = value
  try {
    if (value) localStorage.setItem(STORAGE_KEY, '1')
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Storage unavailable (private mode). The preference lasts for this session only.
  }
  listeners.forEach((l) => l())
}

export function useHideStubs(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => hidden,
    () => false,
  )
}

export const isStub = (note: NoteMeta) => note.status === 'stub'

/** The number of placeholder notes in the whole tree. */
export const stubCount = notes.filter(isStub).length

/** `list` without its placeholder notes when `hide` is set. */
export const withoutStubs = (list: readonly NoteMeta[], hide: boolean): NoteMeta[] =>
  hide ? list.filter((n) => !isStub(n)) : [...list]
