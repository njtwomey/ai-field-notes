import { useEffect, useState } from 'react'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

// Figure plumbing for the generative semi-supervised notes. useSlicedRun and usePlayerPosition follow the
// consistency-regularisation chapter's helpers, with a fixed cap on the steps per slice so that a frozen clock (as in
// the headless renderer) cannot hang the page.

/** Milliseconds of work per slice, so the page stays responsive while a run fills in. */
const SLICE_MS = 25
/** Generator steps per slice at most, whatever the clock says. */
const SLICE_STEPS = 50

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
      for (let k = 0; k < SLICE_STEPS && performance.now() - start < SLICE_MS; k++) {
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

/** An evaluation grid over a box: the axis values and the points row by row (y outer, x inner). */
export type Grid = { xs: number[]; ys: number[]; points: Tensor }

export function grid(x: readonly [number, number], y: readonly [number, number], nx: number, ny: number): Grid {
  const xs = Array.from({ length: nx }, (_, j) => x[0] + ((x[1] - x[0]) * j) / (nx - 1))
  const ys = Array.from({ length: ny }, (_, i) => y[0] + ((y[1] - y[0]) * i) / (ny - 1))
  const data = new Float64Array(nx * ny * 2)
  ys.forEach((b, i) =>
    xs.forEach((a, j) => {
      data[2 * (i * nx + j)] = a
      data[2 * (i * nx + j) + 1] = b
    }),
  )
  return { xs, ys, points: fromData(data, [nx * ny, 2]) }
}

/** Raster rows of P(class 1) − ½ from row-major two-class probabilities on a grid. */
export function centredField(probs: ArrayLike<number>, g: Grid): number[][] {
  const W = g.xs.length
  return g.ys.map((_, i) => g.xs.map((_, j) => probs[2 * (i * W + j) + 1] - 0.5))
}

/** Raster rows of the arg max class from row-major K-class probabilities on a grid. */
export function argmaxField(probs: ArrayLike<number>, K: number, g: Grid): number[][] {
  const W = g.xs.length
  return g.ys.map((_, i) =>
    g.xs.map((_, j) => {
      const r = i * W + j
      let best = 0
      for (let k = 1; k < K; k++) if (probs[r * K + k] > probs[r * K + best]) best = k
      return best
    }),
  )
}

/** Points of the rows whose class is `c`, as x and y arrays for one Points layer. */
export function pointsOf(rows: ArrayLike<number>, cls: ArrayLike<number>, c: number) {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < cls.length; i++)
    if (cls[i] === c) {
      x.push(rows[2 * i])
      y.push(rows[2 * i + 1])
    }
  return { x, y }
}

/** A percentage with no decimals, or an ellipsis while a value is pending. */
export const percent = (v: number | undefined) =>
  v === undefined || !Number.isFinite(v) ? '…' : `${Math.round(100 * v)}%`
