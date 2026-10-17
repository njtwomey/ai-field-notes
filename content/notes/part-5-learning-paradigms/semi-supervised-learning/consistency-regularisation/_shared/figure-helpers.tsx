import { useContext, useEffect, useState, type ReactNode } from 'react'
import { FrameContext } from 'aifn-render'

// Copied from operator-learning/_shared/figure-helpers.tsx (useSlicedRun, usePlayerPosition, FixedHeight), so that
// this chapter does not import from another chapter's folder.

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Milliseconds of work per slice, so the page stays responsive while a run fills in. */
const SLICE = 25

/**
 * Runs a generator of checkpoints in short slices between frames and returns the checkpoints so far, keyed to the
 * setup that started it: a new setup restarts the run.
 */
export function useSlicedRun<S, C>(setup: S, run: (setup: S) => Generator<C>) {
  const [state, setState] = useState<{ setup: S; checkpoints: C[]; done: boolean }>({
    setup,
    checkpoints: [],
    done: false,
  })
  useEffect(() => {
    const it = run(setup)
    const checkpoints: C[] = []
    let timer = 0
    const slice = () => {
      const start = performance.now()
      let done = false
      while (performance.now() - start < SLICE) {
        const next = it.next()
        if (next.done) {
          done = true
          break
        }
        checkpoints.push(next.value)
      }
      setState({ setup, checkpoints: [...checkpoints], done })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
  }, [setup, run])
  return state.setup === setup ? state : { setup, checkpoints: [] as C[], done: false }
}

/** A player's position, back at 0 whenever the setup changes. */
export function usePlayerPosition<S>(setup: S): [number, (at: number) => void] {
  const [pos, setPos] = useState({ setup, at: 0 })
  return [pos.setup === setup ? pos.at : 0, (at: number) => setPos({ setup, at })]
}
