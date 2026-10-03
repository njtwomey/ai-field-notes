import { useEffect, useLayoutEffect, useRef } from 'react'

/**
 * Calls `advance(n)` once per animation frame while `playing`, with n the whole number of steps due at `rate` steps per
 * second. `advance` returns false to stop. The latest `advance` is always used, so it may close over current state.
 */
export function usePlayLoop(playing: boolean, rate: number, advance: (n: number) => boolean) {
  const latest = useRef(advance)
  useLayoutEffect(() => {
    latest.current = advance
  })
  useEffect(() => {
    if (!playing) return
    let handle = 0
    let last = performance.now()
    let budget = 0
    const loop = (now: number) => {
      // A background tab pauses frames; cap the catch-up so returning to it does not jump far ahead.
      budget += (Math.min(100, now - last) / 1000) * rate
      last = now
      const n = Math.floor(budget)
      budget -= n
      if (n > 0 && !latest.current(n)) return
      handle = requestAnimationFrame(loop)
    }
    handle = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(handle)
  }, [playing, rate])
}
