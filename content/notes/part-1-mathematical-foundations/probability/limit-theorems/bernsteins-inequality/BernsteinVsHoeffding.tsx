import { concentrationStudy } from 'aifn-methods/theory/concentration'
import { stream } from 'aifn-compute/foundation/random'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const STEP = 0.005
/** The deviations t at which the tails are read: 0.005, 0.010, …, 0.300. */
const THRESHOLDS = Array.from({ length: 60 }, (_, i) => Number(((i + 1) * STEP).toFixed(3)))
const FLOOR = 1e-6
const TRIALS = 20_000
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const floored = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(FLOOR, v))

/**
 * The simulated two-sided tail of a Bernoulli mean against Hoeffding's and Bernstein's bounds (and the relative-entropy
 * Chernoff bound), from the engine's concentration study.
 */
export function BernsteinVsHoeffding() {
  const state = useFigureState({
    p: slider(0.01, 0.5, 0.02, { step: 0.01, label: 'Bernoulli p' }),
    n: int(200, { ge: 10, le: 5000, suggestions: [50, 200, 1000, 5000], label: 'n (variables per mean)' }),
    t: slider(STEP, 0.3, 0.05, { step: STEP, label: 'deviation t' }),
  })
  const { p, n, t } = state
  const study = useComputed(
    () =>
      concentrationStudy(stream('bernstein-vs-hoeffding'), {
        law: 'bernoulli',
        p,
        n,
        trials: TRIALS,
        thresholds: THRESHOLDS,
      }),
    [p, n],
    { mode: 'release' },
  )
  const c = study.value
  const at = Math.min(THRESHOLDS.length - 1, Math.max(0, Math.round(t / STEP) - 1))
  const tAxis = useAxis({ label: 'deviation t', range: [0, 0.3] })
  const pAxis = useAxis({ label: 'P(|X̄ₙ − p| ≥ t)', log: true, range: [FLOOR, 1] })
  return (
    <Figure
      title="Bernstein against Hoeffding"
      state={state}
      readouts={
        <>
          <Readout label="variance p(1 − p)" value={fmt(c.variance)} />
          <Readout
            label={`simulated at t = ${fmt(c.t[at])}`}
            value={c.empirical[at] > 0 ? fmt(c.empirical[at]) : `< ${fmt(1 / TRIALS)}`}
          />
          <Readout label="Bernstein" value={fmt(c.bernstein[at])} />
          <Readout label="Hoeffding" value={fmt(c.hoeffding[at])} />
          <Readout label="Hoeffding ÷ Bernstein" value={fmt(c.hoeffding[at] / c.bernstein[at])} />
        </>
      }
      caption="The probability that the mean of n Bernoulli(p) draws misses p by at least t, on a log scale. The simulated tail (20,000 means) stops where no trial deviated that far. Hoeffding's bound 2e^(−2nt²) ignores the variance; Bernstein's 2e^(−nt²/(2p(1−p) + 2t/3)) uses it, and the Chernoff bound uses the whole law. Lower p to shrink the variance and watch Bernstein pull away from Hoeffding; raise p to 0.5 and Hoeffding is slightly tighter. Drag t on the chart. Bounds are floored at 10⁻⁶."
    >
      <Plot x={tAxis} y={pAxis} height={320}>
        <Curve
          name="simulated"
          x={c.t}
          y={Array.from(c.empirical, (v) => (v > 0 ? v : NaN))}
          emphasis
          showPoints
          stale={study.stale}
        />
        <Curve name="Bernstein" slot={0} x={c.t} y={floored(c.bernstein)} />
        <Curve name="Hoeffding" slot={1} x={c.t} y={floored(c.hoeffding)} />
        <Curve name="Chernoff (KL)" slot={2} x={c.t} y={floored(c.chernoff)} dashed />
        <Handle kind="x" at={c.t[at]} label="t" onDrag={(v) => state.set('t', Math.max(STEP, Math.min(0.3, v)))} />
      </Plot>
    </Figure>
  )
}
