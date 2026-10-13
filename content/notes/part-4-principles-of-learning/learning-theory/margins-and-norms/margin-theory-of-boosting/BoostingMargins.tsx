import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { argmax, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { errorRate } from 'aifn-compute/learning/metrics'
import { ecdf } from 'aifn-compute/probability/stats'
import { circles, moons } from 'aifn-methods/data/synthetic'
import { adaBoost } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const TRAIN = 300
const TEST = 2000
/** Rounds at which the margin distribution is drawn, before the last round. */
const SNAPSHOTS = [5, 50]

/** Rounds at which the errors are evaluated: about 40, spread evenly on a log scale. */
const checkpoints = (rounds: number) => {
  const out = new Set<number>()
  for (let i = 0; i <= 40; i++) out.add(Math.max(1, Math.round(Math.exp((i / 40) * Math.log(rounds)))))
  return [...out].sort((a, b) => a - b)
}
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The steps of an empirical CDF as a polyline: a riser at each distinct value. */
function steps(values: number[], probabilities: number[]) {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < values.length; i++) {
    x.push(values[i], values[i])
    y.push(i === 0 ? 0 : probabilities[i - 1], probabilities[i])
  }
  return { x, y }
}

/**
 * AdaBoost with small trees. The normalised margin of a training example is the vote for its label minus the vote
 * against, as a fraction of all votes. Its distribution keeps moving right after the training error has reached zero,
 * while the test error stays put, as the margin theory of boosting predicts.
 */
export function BoostingMargins() {
  const state = useFigureState({
    data: choice(
      [
        { value: 'moons', label: 'two moons' },
        { value: 'circles', label: 'two circles' },
      ],
      'moons',
      { label: 'data' },
    ),
    noise: slider(0, 0.4, 0.15, { step: 0.01, label: 'input noise' }),
    depth: int(2, { ge: 1, le: 4, suggestions: [1, 2, 3], label: 'tree depth of each learner' }),
    rounds: int(1000, { ge: 60, le: 2000, suggestions: [100, 300, 1000], label: 'rounds T' }),
    seed: int(2, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { data, noise, depth, rounds, seed } = state
  const run = useComputed(
    () => {
      const make = data === 'moons' ? moons : circles
      const train = make(stream(`boosting-margins/${seed}/train`), { n: TRAIN, noise })
      const test = make(stream(`boosting-margins/${seed}/test`), { n: TEST, noise })
      const model = adaBoost({ rounds, base: { maxDepth: depth } }).fit(dataset(train.x, train.y!))
      const learned = model.learners.length
      const labels = toFlat(train.y!)
      // For two classes the normalised votes sum to 1, so the margin is (vote for y) − (vote against) = 2 v_y − 1.
      const margins = (r: number) => {
        const votes = toFlat(model.votesUpTo(train.x, r))
        return labels.map((c, i) => 2 * votes[2 * i + c] - 1)
      }
      const at = [...SNAPSHOTS.filter((r) => r < learned), learned]
      const distributions = at.map((r) => {
        const m = margins(r)
        const cdf = ecdf(m)
        return {
          round: r,
          curve: steps(toFlat(cdf.values), toFlat(cdf.probabilities)),
          smallest: Math.min(...m),
          belowTenth: m.filter((v) => v <= 0.1).length / m.length,
        }
      })
      const errorRounds = checkpoints(learned)
      const testError = errorRounds.map((r) => errorRate(test.y!, argmax(model.votesUpTo(test.x, r), 1)))
      const trainIndex = Array.from(model.training.index)
      const trainSeries = Array.from(toFlat(model.training.series.trainingError))
      const keep = trainIndex.map((t) => t >= 1)
      return {
        learned,
        distributions,
        errorRounds,
        testError,
        trainIndex: trainIndex.filter((_, i) => keep[i]),
        trainError: trainSeries.filter((_, i) => keep[i]),
      }
    },
    [data, noise, depth, rounds, seed],
    { mode: 'release' },
  )
  const r = run.value
  const firstZero = useMemo(() => {
    const i = r.trainError.findIndex((e) => e === 0)
    return i < 0 ? NaN : r.trainIndex[i]
  }, [r.trainError, r.trainIndex])
  const last = r.distributions[r.distributions.length - 1]
  const tAxis = useAxis({ label: 'rounds T', range: [1, Math.max(2, rounds)], log: true })
  const eAxis = useAxis({ label: 'error of the vote', range: [0, 0.3] })
  const mAxis = useAxis({ label: 'normalised margin y f(x)', range: [-1, 1] })
  const fAxis = useAxis({ label: 'fraction of training points with margin ≤ value', range: [0, 1] })
  return (
    <Figure
      title="Margins keep growing after the training error is zero"
      purpose="The training error of AdaBoost reaches zero after a few rounds, but later rounds keep moving the training margins to the right. The margin bound depends on those margins and on the base class, not on the number of rounds, and the test error does not rise."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="trees kept" value={String(r.learned)} />
          <Readout
            label="training error zero from round"
            value={Number.isFinite(firstZero) ? String(firstZero) : 'never'}
          />
          <Readout label={`smallest margin after ${last.round}`} value={fmt(last.smallest)} />
          <Readout label={`fraction with margin ≤ 0.1 after ${last.round}`} value={fmt(last.belowTenth)} />
          <Readout label="final test error" value={fmt(r.testError[r.testError.length - 1])} />
        </>
      }
      caption="AdaBoost (aifn adaBoost) with decision trees of the chosen depth on 300 training points, tested on 2,000 fresh points. Left: the training error (ink) and the test error of the weighted vote after T rounds, on a log scale of rounds. Right: the cumulative distribution of the normalised training margins y f(x) = (vote for the true label − vote against)/(total vote), after 5 rounds, 50 rounds and the last round; a curve further right means larger margins. At the opening values (two moons, noise 0.15, depth 2, 1,000 rounds, seed 2) the training error is zero from round 9. The smallest training margin then grows from 0.08 after 20 rounds to 0.19 after 50 and 0.25 after 1,000, while the test error moves only from 0.012 to 0.016 although the vote holds 50 times as many trees. Raise the noise to 0.3 so that the classes overlap: the training error reaches zero only at round 143, by fitting points that sit on the wrong side. The smallest margin creeps up to 0.05, but the share of training points with margin at most 0.1 rises from 0.15 after 50 rounds to 0.40 after 1,000, and the test error rises from 0.100 to 0.116."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={eAxis} title="errors against rounds">
          <Curve name="training error" x={r.trainIndex} y={r.trainError} emphasis stale={run.stale} />
          <Curve name="test error" x={r.errorRounds} y={r.testError} slot={1} showPoints />
        </Plot>
        <Plot x={mAxis} y={fAxis} title="training margin distribution">
          <Annotation x={0} text="0" dashed />
          {r.distributions.map((d, i) => (
            <Curve
              key={d.round}
              name={`after ${d.round} rounds`}
              x={d.curve.x}
              y={d.curve.y}
              slot={i === r.distributions.length - 1 ? undefined : i === 0 ? 0 : 2}
              emphasis={i === r.distributions.length - 1}
              stale={run.stale}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
