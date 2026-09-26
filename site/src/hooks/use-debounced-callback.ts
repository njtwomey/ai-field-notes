import { useEffect, useMemo, useRef, type RefObject } from 'react'

export type Debounced<A extends unknown[]> = {
  (...args: A): void
  /** Run the pending call now, if any. */
  flush: () => void
  cancel: () => void
}

/** Plain (non-React) trailing debounce with an upper bound on how long a call can be deferred. */
function createDebounced<A extends unknown[]>(
  fn: RefObject<(...args: A) => void>,
  wait: number,
  maxWait: number,
): Debounced<A> {
  let timer: ReturnType<typeof setTimeout> | undefined
  let firstCall: number | undefined
  let pending: A | undefined

  const cancel = () => {
    if (timer) clearTimeout(timer)
    timer = undefined
    firstCall = undefined
    pending = undefined
  }
  const flush = () => {
    const args = pending
    cancel()
    if (args) fn.current(...args)
  }
  const call = ((...args: A) => {
    pending = args
    const now = Date.now()
    firstCall ??= now
    if (timer) clearTimeout(timer)
    if (now - firstCall >= maxWait) flush()
    else timer = setTimeout(flush, wait)
  }) as Debounced<A>
  call.flush = flush
  call.cancel = cancel
  return call
}

/**
 * Trailing debounce: `fn` runs `wait` ms after the last call, and at least every `maxWait` ms while calls keep
 * arriving, so a slider drag still updates the figure periodically. `fn` may change between renders.
 */
export function useDebouncedCallback<A extends unknown[]>(
  fn: (...args: A) => void,
  wait: number,
  maxWait = wait * 3,
): Debounced<A> {
  const fnRef = useRef(fn)
  useEffect(() => {
    fnRef.current = fn
  })
  // The ref object is only stored here; `.current` is read later, in timers and event handlers.
  // oxlint-disable-next-line react/refs
  const debounced = useMemo(() => createDebounced(fnRef, wait, maxWait), [wait, maxWait])
  useEffect(() => debounced.cancel, [debounced])
  return debounced
}
