import { useMemo } from 'react'
import {
  Annotation,
  Area,
  Density,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { accuracy } from 'aifn-compute/learning/metrics'
import { kFold, repeated } from 'aifn-compute/learning/validate'
import { bayesianCorrelatedTTest, correctedResampledTTest } from 'aifn-compute/probability/tests'
import { differences, drawPair, learner } from '../_shared/equalLearners'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const FOLDS = 10
const REPEATS = 10
/** The correlation between fold differences that the correlated t-test assumes: the test fraction 1/K. */
const RHO = 1 / FOLDS
const GRID = 80
/** Training sets averaged for the true mean difference, and the fresh rows each is scored on. */
const TRUTH_SETS = 200
const FRESH = 2000

/**
 * The posterior of the mean difference in accuracy from the Bayesian correlated t-test on 10 repeats of 10-fold
 * cross-validation, with a draggable region of practical equivalence and the posterior that ignores the correlation.
 */
export function PosteriorWithRope() {
  const state = useFigureState({
    n: int(200, { ge: 60, le: 600, suggestions: [100, 200, 400], label: 'rows n' }),
    handicap: float(0.15, { ge: 0, le: 1, suggestions: [0, 0.15, 0.4], label: "noise on B's inputs" }),
    rope: float(0.01, { ge: 0, le: 0.1, suggestions: [0, 0.01, 0.02], label: 'rope half-width' }),
    seed: int(2, { ge: 1, le: 999, label: 'data set' }),
  })
  const { n, handicap, rope, seed } = state
  const computed = useComputed(
    () => {
      const s = stream(`comparing/bayesian/${seed}`)
      const pair = drawPair(child(s, 'data'), n, handicap)
      return differences(pair, repeated(kFold({ k: FOLDS, shuffle: true }), REPEATS), child(s, 'cv'))
    },
    [n, handicap, seed],
    { mode: 'release' },
  )
  const diffs = computed.value
  // The estimand: A's minus B's expected accuracy when trained on 9n/10 rows, averaged over fresh training sets.
  const truth = useComputed(
    () => {
      const s = stream('comparing/bayesian/truth')
      const fresh = drawPair(child(s, 'fresh'), FRESH, handicap)
      let total = 0
      for (let r = 0; r < TRUTH_SETS; r++) {
        const train = drawPair(child(s, 'train', r), Math.round((n * (FOLDS - 1)) / FOLDS), handicap)
        total +=
          accuracy(fresh.a.y, learner.fit(train.a).decide(fresh.a.x)) -
          accuracy(fresh.b.y, learner.fit(train.b).decide(fresh.b.x))
      }
      return total / TRUTH_SETS
    },
    [n, handicap],
    { mode: 'release' },
  )
  const view = useMemo(() => {
    const correlated = bayesianCorrelatedTTest(diffs, { rho: RHO, rope })
    const naive = bayesianCorrelatedTTest(diffs, { rho: 0, rope })
    const frequentist = correctedResampledTTest(diffs, {
      trainSize: (n * (FOLDS - 1)) / FOLDS,
      testSize: n / FOLDS,
    })
    const post = correlated.posterior
    const xs = Array.from({ length: GRID + 1 }, (_, i) => -rope + (2 * rope * i) / GRID)
    const ys = post ? xs.map((x) => post.prob(x) as number) : xs.map(() => 0)
    return { correlated, naive, frequentist, xs, ys }
  }, [diffs, rope, n])
  const { correlated, naive, frequentist } = view
  const x = useAxis({ label: 'mean difference in accuracy, A minus B' })
  const y = useAxis({ label: 'posterior density', range: [0, undefined] })
  return (
    <Figure
      title="Posterior probabilities around a region of practical equivalence"
      state={state}
      caption="One simulated data set of n rows from the circle problem: learner A is a depth-4 tree on the inputs, B the same tree on rotated inputs with added noise, so B is worse when the noise is positive and equal to A when it is 0. The solid curve is the posterior of the mean difference in accuracy from the Bayesian correlated t-test on the 100 fold differences of 10 repeats of 10-fold cross-validation, with correlation ρ = 1/10. The shaded area is its mass inside the region of practical equivalence (rope); drag the line at its right edge to resize it. The solid vertical line is the true mean difference, A's minus B's expected accuracy when trained on 9n/10 rows, estimated from 200 fresh training sets scored on 2000 fresh rows (Monte Carlo error about 0.004). The dashed curve sets ρ = 0, treating the 100 differences as independent: it is far narrower than the evidence supports."
      readouts={{
        'correlated t-test (ρ = 0.1)': (
          <>
            <Readout label="P(B practically better)" value={fmt(correlated.pLeft)} />
            <Readout label="P(practically equivalent)" value={fmt(correlated.pRope)} />
            <Readout label="P(A practically better)" value={fmt(correlated.pRight)} />
          </>
        ),
        'ignoring the correlation (ρ = 0)': <Readout label="P(A practically better)" value={fmt(naive.pRight)} />,
        'corrected resampled t-test': <Readout label="p" value={fmt(frequentist.pValue)} />,
        truth: <Readout label="true mean difference" value={fmt(truth.value)} />,
      }}
    >
      <Plot x={x} y={y}>
        {correlated.posterior && <Density name="posterior, ρ = 0.1" dist={correlated.posterior} slot={0} />}
        {naive.posterior && <Density name="posterior, ρ = 0" dist={naive.posterior} slot={1} dashed />}
        {rope > 0 && <Area name="mass in the rope" x={view.xs} y={view.ys} slot={0} opacity={0.3} line={false} />}
        <Annotation x={-rope} dashed />

        <Annotation x={truth.value} text="truth" emphasis />
        <Handle kind="x" at={rope} label="rope" onDrag={(v) => state.set('rope', Math.min(0.1, Math.max(0, v)))} />
      </Plot>
    </Figure>
  )
}
