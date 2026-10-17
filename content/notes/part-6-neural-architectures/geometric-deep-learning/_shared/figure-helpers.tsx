/**
 * Figure plumbing shared by the geometric-deep-learning notes: a fixed plot height inside a Figure and a training run
 * sliced between frames. Copied from graph-neural-networks/graph-learning-tasks/_shared/figure-helpers.tsx (a note may
 * not import across sub-chapters); the two copies should merge into site/src/components/widgets/.
 */
import { useContext, useEffect, useState, type ReactNode } from 'react'
import { FrameContext } from 'aifn-render'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Milliseconds of work per slice, so the page stays responsive while a run fills in. */
const SLICE = 25
/** Checkpoints per slice at most, so a slice ends even where the clock does not advance during a task. */
const MAX_PER_SLICE = 50

/**
 * Runs a generator of checkpoints in short slices between frames and returns the checkpoints so far, keyed to the
 * setup that started it: a new setup restarts the run. A generator that throws ends the run with its error.
 */
export function useSlicedRun<S, C>(setup: S, run: (setup: S) => Generator<C>) {
  const [state, setState] = useState<{ setup: S; checkpoints: C[]; done: boolean; error: string | null }>({
    setup,
    checkpoints: [],
    done: false,
    error: null,
  })
  useEffect(() => {
    const it = run(setup)
    const checkpoints: C[] = []
    let timer = 0
    const slice = () => {
      const start = performance.now()
      let done = false
      let error: string | null = null
      try {
        for (let k = 0; k < MAX_PER_SLICE && performance.now() - start < SLICE; k++) {
          const next = it.next()
          if (next.done) {
            done = true
            break
          }
          checkpoints.push(next.value)
        }
      } catch (e) {
        console.error(e)
        error = e instanceof Error ? e.message : String(e)
        done = true
      }
      setState({ setup, checkpoints: [...checkpoints], done, error })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
  }, [setup, run])
  return state.setup === setup ? state : { setup, checkpoints: [] as C[], done: false, error: null }
}
