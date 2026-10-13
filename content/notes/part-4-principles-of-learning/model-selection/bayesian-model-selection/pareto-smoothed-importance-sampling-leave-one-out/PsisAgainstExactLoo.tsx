import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linspace, reshape, tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { psisLoo, waic } from 'aifn-compute/learning/selection'
import {
  drawData,
  flat,
  fmt,
  logLikDraws,
  polynomialEvidence,
  posteriorMean,
  predictiveLogDensity,
} from '../_shared/blr'

const GRID = reshape(linspace(-1, 1, 161), [161, 1]) as Tensor
const GRID_X = flat(GRID)
const column = (v: number[]) => tensor(v.map((u) => [u]))

/**
 * PSIS-LOO on a Bayesian polynomial regression with one draggable point: each point's Pareto k̂, and each point's
 * PSIS estimate of its leave-one-out log predictive density against the exact value from refitting without it.
 */
export function PsisAgainstExactLoo() {
  const state = useFigureState({
    degree: int(5, { ge: 1, le: 12, suggestions: [1, 3, 5, 9], label: 'polynomial degree' }),
    alpha: float(1, {
      min: 0.001,
      max: 100,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1, 10],
      label: 'prior precision α of each weight',
    }),
    n: int(30, { ge: 10, le: 200, suggestions: [15, 30, 100], label: 'data points n' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ (known to the model)' }),
    draws: int(2000, { ge: 100, le: 8000, suggestions: [500, 2000, 8000], label: 'posterior draws S' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    ox: float(0.95, { min: -1, max: 1, step: 0.01, label: 'moved point x' }),
    oy: float(-1.5, { min: -3, max: 3, step: 0.01, label: 'moved point y' }),
  })
  const { degree, alpha, n, noise, draws, seed, ox, oy } = state
  const base = useMemo(() => {
    const d = drawData(`psis/${seed}`, n, noise)
    return { x: flat(d.x), y: flat(d.y) }
  }, [n, noise, seed])
  const run = useComputed(
    () => {
      const xs = [...base.x, ox]
      const ys = [...base.y, oy]
      const beta = 1 / (noise * noise)
      const { X, map, ev } = polynomialEvidence(column(xs), tensor(ys), degree, alpha, beta)
      const L = logLikDraws(ev, X, tensor(ys), draws, `psis/draws/${seed}`)
      const loo = psisLoo(L)
      const exact = xs.map((_, i) => {
        const keep = xs.map((_, j) => j).filter((j) => j !== i)
        const fit = polynomialEvidence(
          column(keep.map((j) => xs[j])),
          tensor(keep.map((j) => ys[j])),
          degree,
          alpha,
          beta,
        )
        return flat(predictiveLogDensity(fit.ev, map(column([xs[i]])), tensor([ys[i]])))[0]
      })
      return {
        xs,
        ys,
        khat: flat(loo.khat),
        psis: flat(loo.pointwise.elpd),
        exact,
        elpd: loo.elpd,
        exactTotal: exact.reduce((a, b) => a + b, 0),
        waic: waic(L).elpd,
        threshold: loo.khatThreshold,
        bad: loo.bad,
        curve: flat(posteriorMean(ev, map(GRID))),
      }
    },
    [base, ox, oy, degree, alpha, draws, noise, seed],
    { mode: 'release' },
  )
  const v = run.value
  const last = v.xs.length - 1
  const lo = Math.min(...v.exact, ...v.psis)
  const hi = Math.max(...v.exact, ...v.psis)
  const key = `${degree}/${alpha}/${n}/${noise}/${seed}`
  const xx = useAxis({ label: 'x', range: [-1.05, 1.05], nice: false })
  const yy = useAxis({ label: 'y', range: [-3, 3] })
  const ky = useAxis({ label: 'Pareto k̂', hold: 'union', key })
  const ex = useAxis({ label: 'exact leave-one-out log density', hold: 'union', key: `e/${key}` })
  const ey = useAxis({ label: 'PSIS estimate', hold: 'union', key: `p/${key}` })
  return (
    <Figure
      title="PSIS leave-one-out against exact refits"
      state={state}
      hoverReadout={false}
      caption="A Bayesian polynomial regression with a N(0, 1/α) prior on each coefficient, fitted to n points of y = sin(πx) + σε and one more point that can be dragged. Left: the data, the moved point (the handle) and the posterior mean. Middle: each point's Pareto k̂ from S posterior draws, with the reliability threshold dashed. Right: each point's PSIS estimate of log p(yᵢ | the other points) against the exact value from refitting without it; points on the diagonal are estimated well. Drag the point to an edge of the range and away from the curve: its k̂ rises past the threshold and its estimate leaves the diagonal. A high degree with a weak prior (small α) raises k̂ at the ends of the range even for ordinary points, because each end point there determines the fit near it."
      readouts={
        <>
          <Readout
            label="elpd: PSIS-LOO / exact LOO / WAIC"
            value={`${fmt(v.elpd, 4)} / ${fmt(v.exactTotal, 4)} / ${fmt(v.waic, 4)}`}
          />
          <Readout label="k̂ of the moved point" value={fmt(v.khat[last])} />
          <Readout label="moved point: PSIS / exact" value={`${fmt(v.psis[last])} / ${fmt(v.exact[last])}`} />
          <Readout label={`points with k̂ > ${fmt(v.threshold, 2)}`} value={`${v.bad} of ${v.xs.length}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={xx} y={yy} title="data and posterior mean">
          <Points name="data" x={v.xs.slice(0, last)} y={v.ys.slice(0, last)} slot={0} />
          <Curve name="posterior mean" x={GRID_X} y={v.curve} slot={1} stale={run.stale} />
          <Handle
            kind="point"
            at={[ox, oy]}
            label="moved point"
            onDrag={([px, py]) => {
              state.set('ox', Math.max(-1, Math.min(1, Math.round(px * 100) / 100)))
              state.set('oy', Math.max(-3, Math.min(3, Math.round(py * 100) / 100)))
            }}
          />
        </Plot>
        <Plot x={xx} y={ky} title="Pareto k̂ of each point">
          <Points name="k̂" x={v.xs} y={v.khat} slot={0} stale={run.stale} />
          <Points name="moved point" x={[v.xs[last]]} y={[v.khat[last]]} emphasis />
          <Curve name="threshold" x={[-1.05, 1.05]} y={[v.threshold, v.threshold]} emphasis dashed />
        </Plot>
        <Plot x={ex} y={ey} title="PSIS against exact">
          <Curve name="equal" x={[lo, hi]} y={[lo, hi]} emphasis dashed />
          <Points name="points" x={v.exact} y={v.psis} slot={0} stale={run.stale} />
          <Points name="moved point" x={[v.exact[last]]} y={[v.psis[last]]} emphasis />
        </Plot>
      </div>
    </Figure>
  )
}
