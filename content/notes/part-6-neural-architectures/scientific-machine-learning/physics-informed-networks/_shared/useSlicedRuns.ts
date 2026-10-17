import { useContext, useEffect, useState, createElement, type ReactNode } from 'react'
import { FrameContext } from 'aifn-render'

/** Milliseconds of training per slice, so the page stays responsive while the runs fill in. */
const SLICE = 25

type Runs<S> = { key: unknown; snapshots: S[][]; done: boolean }

/**
 * Steps several training generators in short slices between frames, round-robin, and returns every snapshot each has
 * yielded so far (its return value included). The runs restart whenever `key` changes identity, so pass a memoised
 * setup object.
 *
 * @param key The setup the runs are made from; a new object restarts them.
 * @param make Builds the generators for that setup.
 * @returns The snapshots of each run so far, and whether all have finished.
 */
export function useSlicedRuns<S>(key: unknown, make: () => Generator<S, S>[]): { snapshots: S[][]; done: boolean } {
  const [runs, setRuns] = useState<Runs<S>>({ key, snapshots: [], done: false })
  useEffect(() => {
    const gens = make()
    const snapshots: S[][] = gens.map(() => [])
    const finished = gens.map(() => false)
    let timer = 0
    const slice = () => {
      const start = performance.now()
      while (performance.now() - start < SLICE && finished.some((f) => !f)) {
        gens.forEach((g, i) => {
          if (finished[i]) return
          const next = g.next()
          snapshots[i].push(next.value)
          if (next.done) finished[i] = true
        })
      }
      const done = finished.every(Boolean)
      setRuns({ key, snapshots: snapshots.map((s) => [...s]), done })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
    // `make` is rebuilt on every render; the runs depend on `key` only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return runs.key === key ? runs : { snapshots: [], done: false }
}

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return createElement(FrameContext.Provider, { value: { ...frame, height } }, children)
}

/** The snapshot at checkpoint `at` of a run, clamped to the ones that exist. */
export const pick = <S>(run: readonly S[] | undefined, at: number): S | undefined =>
  run && run.length ? run[Math.min(at, run.length - 1)] : undefined

/** Rows of a row-major $[n_t \times n_x]$ field, for a `Raster`. */
export const rowsOf = (values: readonly number[], nx: number): number[][] => {
  const rows: number[][] = []
  for (let i = 0; i < values.length; i += nx) rows.push(values.slice(i, i + nx))
  return rows
}
