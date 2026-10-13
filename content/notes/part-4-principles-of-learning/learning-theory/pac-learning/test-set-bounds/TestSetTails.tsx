import { useMemo } from 'react'
import { concentrationStudy, tailBounds } from 'aifn-methods/theory/concentration'
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

const FLOOR = 1e-4
const TRIALS = 5000
const POINTS = 60

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * The test error of a classifier with true risk R on m held-out examples is the mean of m Bernoulli(R) losses. The
 * figure draws many test sets and plots how often the test error misses R by at least t, beside Hoeffding's bound and
 * the relative-entropy (Chernoff) bound that the exact binomial inversion tightens further.
 */
export function TestSetTails() {
  const state = useFigureState({
    risk: slider(0.01, 0.5, 0.05, { step: 0.01, label: 'true risk R(h)' }),
    m: int(1000, { ge: 20, le: 5000, suggestions: [100, 300, 1000, 3000], label: 'test examples m' }),
    t: slider(0.001, 0.5, 0.02, { step: 0.001, label: 'deviation t' }),
    delta: slider(0.001, 0.5, 0.05, { step: 0.001, label: 'confidence δ' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { risk, m, t, delta, seed } = state
  // The deviations worth plotting shrink like 1/√m.
  const tMax = Math.min(0.5, 2.5 / Math.sqrt(m))
  const thresholds = useMemo(() => Array.from({ length: POINTS }, (_, i) => ((i + 1) * tMax) / POINTS), [tMax])
  const study = useComputed(
    () =>
      concentrationStudy(stream(`test-set-bounds/${seed}`), {
        law: 'bernoulli',
        p: risk,
        n: m,
        trials: TRIALS,
        thresholds,
      }),
    [risk, m, seed, thresholds],
    { mode: 'release' },
  )
  const c = study.value
  const at = tailBounds([t], m, c.mean, c.variance)
  // The simulated tail at the grid deviation nearest to t.
  const nearest = Math.min(POINTS - 1, Math.max(0, Math.round((t / tMax) * POINTS) - 1))
  const floor = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(FLOOR, v))
  const tAxis = useAxis({ label: 'deviation t = |test error − R(h)|', range: [0, tMax], key: `${m}` })
  const pAxis = useAxis({ label: 'probability of a deviation ≥ t', log: true, range: [FLOOR, 1] })
  return (
    <Figure
      title="How far a test error strays from the true risk"
      purpose="A test error is the mean of m independent Bernoulli(R) losses. Hoeffding's bound on its deviation ignores R; the relative-entropy bound uses it and sits far closer to the simulated tail when R is small."
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label={`simulated at t ≈ ${fmt(c.t[nearest])}`} value={fmt(c.empirical[nearest])} />
          <Readout label="relative-entropy bound" value={fmt(at.chernoff[0])} />
          <Readout label="Hoeffding bound" value={fmt(at.hoeffding[0])} />
          <Readout label="δ" value={fmt(delta)} />
        </>
      }
      caption="Each of 5,000 simulated test sets has m examples, each misclassified with probability R(h). The ink curve is the fraction of test sets whose error misses R(h) by at least t (two-sided; it ends where no test set missed by that much). The coloured curves are Hoeffding's bound 2 exp(−2mt²) and the relative-entropy bound exp(−m kl(R + t ‖ R)) + exp(−m kl(R − t ‖ R)), floored at 10⁻⁴. A bound at confidence δ is the deviation where its curve crosses the horizontal δ line. Drag the vertical line to read the three tails at one deviation, and the horizontal line to change δ. At the opening values (R(h) = 0.05, m = 1,000, δ = 0.05) the relative-entropy curve crosses δ at t = 0.019 and Hoeffding's at t = 0.043."
    >
      <Plot x={tAxis} y={pAxis} height={340}>
        <Curve
          name="simulated"
          x={c.t}
          y={Array.from(c.empirical, (v) => (v > 0 ? v : NaN))}
          emphasis
          showPoints
          stale={study.stale}
        />
        <Curve name="relative-entropy (Chernoff)" slot={0} x={c.t} y={floor(c.chernoff)} />
        <Curve name="Hoeffding" slot={2} x={c.t} y={floor(c.hoeffding)} />
        <Handle kind="x" at={t} onDrag={(v) => state.set('t', Math.max(0.001, Math.min(tMax, v)))} label="t" />
        <Handle kind="y" at={delta} onDrag={(v) => state.set('delta', Math.max(0.001, Math.min(0.5, v)))} label="δ" />
      </Plot>
    </Figure>
  )
}
