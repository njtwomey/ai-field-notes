import { useEffect, type RefObject } from 'react'

/** The nearest ancestor that scrolls vertically on its own, if any (not the page). */
export function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY
    if ((overflow === 'auto' || overflow === 'scroll') && p.scrollHeight > p.clientHeight) return p
  }
  return null
}

/**
 * Scroll `el` into view inside its own scrolling panel, never the page: scrollIntoView would also pan a
 * pinch-zoomed page across to the panel, away from what is being read. `center` puts it in the middle of the panel;
 * otherwise the panel moves only as far as needed.
 */
export function scrollIntoPanel(el: HTMLElement, center = false) {
  const box = scrollParent(el)
  if (!box) return
  const top = el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop
  if (center) box.scrollTop = top - (box.clientHeight - el.offsetHeight) / 2
  else if (top < box.scrollTop) box.scrollTop = top
  else if (top + el.offsetHeight > box.scrollTop + box.clientHeight)
    box.scrollTop = top + el.offsetHeight - box.clientHeight
}

/**
 * Centre the element marked `aria-current` inside `ref` within its scrolling panel whenever `key` changes, so a
 * long index opens at the reader's place.
 */
export function useScrollToCurrent(ref: RefObject<HTMLElement | null>, key: unknown) {
  useEffect(() => {
    const el = ref.current?.querySelector<HTMLElement>('[aria-current]')
    if (el) scrollIntoPanel(el, true)
  }, [ref, key])
}
