import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
  choice,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { nestedCvInterval } from 'aifn-compute/learning/selection'
import { type Fittable } from 'aifn-compute/learning/validate'
import { linearRegression, type LinearRegressionModel } from 'aifn-methods/learning/linear'
import { kNearestNeighboursRegression, type NeighboursRegressor } from 'aifn-methods/learning/neighbours'
import { draw, LEARNERS, ROOT, TEST } from './problem'

const PANEL = 300
const LEVEL = 0.9
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Interval = readonly [number, number]
type Row = { naive: Interval; nested: Interval; err: number; inflation: number; bias: number }

/**
 * Naive and nested cross-validation intervals (Bates, Hastie and Tibshirani, 2024) on the same replicate data sets,
 * against the conditional error of the model fitted to each and the expected error.
 */
export function NestedIntervals() {
  const state = useFigureState({
    learner: choice(LEARNERS, 'least squares', { label: 'learner' }),
    n: int(100, { ge: 30, le: 400, suggestions: [50, 100, 200], label: 'training rows n' }),
    p: int(20, { ge: 5, le: 60, suggestions: [5, 20, 40], label: 'inputs p' }),
    folds: int(5, { ge: 3, le: 10, suggestions: [3, 5, 10], label: 'folds K' }),
    repetitions: int(5, { ge: 1, le: 50, suggestions: [3, 5, 10, 20], label: 'repetitions R' }),
    replicates: int(50, { ge: 10, le: 400, suggestions: [50, 100, 200], label: 'replicate data sets' }),
  })
  const { learner, n, p, folds, repetitions, replicates } = state
  const computed = useComputed(
    () => {
      const root = stream(ROOT)
      const test = draw(child(root, 'test'), TEST, p)
      const est = (
        learner === 'least squares' ? linearRegression({ intercept: false }) : kNearestNeighboursRegression({ k: 10 })
      ) as Fittable<Tensor, LinearRegressionModel | NeighboursRegressor>
      return Array.from({ length: replicates }, (_, r): Row => {
        const s = child(root, r)
        const data = draw(s, n, p)
        const nested = nestedCvInterval(est, data, {
          folds,
          repetitions,
          stream: child(s, 'nested'),
          level: LEVEL,
        })
        return {
          naive: nested.naive.interval,
          nested: nested.interval,
          err: meanSquaredError(test.y, est.fit(data).decide(test.x)),
          inflation: nested.inflation,
          bias: nested.bias,
        }
      })
    },
    [learner, n, p, folds, repetitions, replicates],
    { mode: 'release' },
  )
  const rows = computed.value
  const view = useMemo(() => {
    const sorted = [...rows].sort((a, b) => a.err - b.err)
    const mean = (f: (r: Row) => number) => rows.reduce((a, r) => a + f(r), 0) / rows.length
    const average = mean((r) => r.err)
    const covers = (i: Interval, target: number) => i[0] <= target && target <= i[1]
    const panel = (key: 'naive' | 'nested') => {
      const bars = sorted.map((r, i) => ({ from: [i, r[key][0]] as const, to: [i, r[key][1]] as const }))
      return {
        hit: bars.filter((_, i) => covers(sorted[i][key], sorted[i].err)),
        miss: bars.filter((_, i) => !covers(sorted[i][key], sorted[i].err)),
        conditional: mean((r) => +covers(r[key], r.err)),
        expected: mean((r) => +covers(r[key], average)),
        width: mean((r) => r[key][1] - r[key][0]),
      }
    }
    const ends = rows.flatMap((r) => [r.naive[0], r.naive[1], r.nested[0], r.nested[1], r.err])
    const lo = Math.min(...ends)
    const hi = Math.max(...ends)
    return {
      average,
      index: sorted.map((_, i) => i),
      err: sorted.map((r) => r.err),
      naive: panel('naive'),
      nested: panel('nested'),
      inflation: mean((r) => r.inflation),
      bias: mean((r) => r.bias),
      range: [lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)] as [number, number],
    }
  }, [rows])
  const xLabel = 'replicate data set, sorted by Err_D'
  const nx = useAxis({ label: xLabel, integer: true })
  const ny = useAxis({ label: 'mean squared error', range: view.range })
  const cx = useAxis({ label: xLabel, integer: true })
  const cy = useAxis({ label: 'mean squared error', range: view.range })
  const panel = (which: 'naive' | 'nested', x: typeof nx, y: typeof ny, title: string) => (
    <FixedHeight height={PANEL}>
      <Plot x={x} y={y} title={title}>
        {view[which].hit.length > 0 && (
          <Segments name="interval covers Err_D" segments={view[which].hit} slot={0} width={2} />
        )}
        {view[which].miss.length > 0 && (
          <Segments name="interval misses Err_D" segments={view[which].miss} slot={1} width={2} />
        )}
        <Points name="Err_D" x={view.index} y={view.err} emphasis size={4} stale={computed.stale} />
        <Annotation y={view.average} text="Err" dashed emphasis />
      </Plot>
    </FixedHeight>
  )
  return (
    <Figure
      title="Nested cross-validation intervals cover their target"
      state={state}
      hoverReadout={false}
      caption={`The data sets of the figure above (the same n, p and seeds), with ${LEVEL * 100}% intervals from R repetitions of K-fold cross-validation. Left: the naive interval, the cross-validation estimate ± 1.645 σ/√n, with σ the standard deviation of the n held-out losses. Right: Bates, Hastie and Tibshirani's nested cross-validation interval, centred on the bias-corrected estimate, with half-width 1.645 times the square root of the estimated mean squared error of cross-validation, which is clamped between 1 and √K times the naive standard error. Each column is one data set, sorted by the conditional error Err_D of the model fitted to all n rows, measured on ${TEST} fresh rows (dots). Intervals that miss their dot take the second colour; the dashed line is the average of Err_D, an estimate of Err. Each repetition fits K(K + 1)/2 models, so large K, R or replicate counts take several seconds.`}
      readouts={{
        'covers Err_D': (
          <>
            <Readout label="naive" value={fmt(view.naive.conditional)} />
            <Readout label="nested" value={fmt(view.nested.conditional)} />
            <Readout label="nominal" value={String(LEVEL)} />
          </>
        ),
        'covers Err': (
          <>
            <Readout label="naive" value={fmt(view.naive.expected)} />
            <Readout label="nested" value={fmt(view.nested.expected)} />
          </>
        ),
        'mean interval width': (
          <>
            <Readout label="naive" value={fmt(view.naive.width)} />
            <Readout label="nested" value={fmt(view.nested.width)} />
            <Readout label="standard error inflation" value={fmt(view.inflation)} />
            <Readout label="bias correction" value={fmt(view.bias)} />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {panel('naive', nx, ny, 'naive intervals')}
        {panel('nested', cx, cy, 'nested cross-validation intervals')}
      </div>
    </Figure>
  )
}
