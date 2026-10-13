import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { argmax, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { errorRate } from 'aifn-compute/learning/metrics'
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
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const TRAIN = 200
const TEST = 2000
/** Rounds at which the test error is evaluated: about 40, spread evenly on a log scale. */
const checkpoints = (rounds: number) => {
  const out = new Set<number>()
  for (let i = 0; i <= 40; i++) out.add(Math.max(1, Math.round(Math.exp((i / 40) * Math.log(rounds)))))
  return [...out].sort((a, b) => a - b)
}
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * AdaBoost with decision stumps as the weak learners. Each stump only beats a coin on its reweighted sample; the vote
 * of many drives the training error to zero, as the boosting theorem predicts, while the test error levels off.
 */
export function WeakToStrong() {
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
    rounds: int(300, { ge: 10, le: 1000, suggestions: [50, 100, 300, 1000], label: 'rounds T' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { data, noise, rounds, seed } = state
  const run = useComputed(
    () => {
      const make = data === 'moons' ? moons : circles
      const train = make(stream(`weak-strong/${seed}/train`), { n: TRAIN, noise })
      const test = make(stream(`weak-strong/${seed}/test`), { n: TEST, noise })
      const model = adaBoost({ rounds }).fit(dataset(train.x, train.y!))
      const learned = model.learners.length
      const at = checkpoints(learned)
      const testError = at.map((r) => errorRate(test.y!, argmax(model.votesUpTo(test.x, r), 1)))
      return {
        weighted: Array.from(toFlat(model.errors)),
        // Round 0 has no learner yet; the log axis starts at round 1.
        trainIndex: Array.from(model.training.index).filter((t) => t >= 1),
        trainError: Array.from(toFlat(model.training.series.trainingError)).filter(
          (_, i) => model.training.index[i] >= 1,
        ),
        at,
        testError,
        learned,
        stopped: model.training.final.stopped,
      }
    },
    [data, noise, rounds, seed],
    { mode: 'release' },
  )
  const r = run.value
  const roundNumbers = useMemo(() => r.weighted.map((_, i) => i + 1), [r.weighted])
  const lastTrain = r.trainError[r.trainError.length - 1]
  const lastTest = r.testError[r.testError.length - 1]
  const firstEdge = 0.5 - r.weighted[0]
  const lastEdge = 0.5 - r.weighted[r.weighted.length - 1]
  const tAxis = useAxis({ label: 'round t', range: [1, Math.max(2, rounds)], log: true })
  const eAxis = useAxis({ label: 'weighted error εₜ of stump t', range: [0, 0.55] })
  const t2Axis = useAxis({ label: 'rounds T', range: [1, Math.max(2, rounds)], log: true })
  const errAxis = useAxis({ label: 'error of the vote', range: [0, 0.5] })
  return (
    <Figure
      title="Weak stumps, strong vote"
      purpose="Every stump AdaBoost adds beats a coin on its own reweighted sample, by an edge γₜ = ½ − εₜ that shrinks as the rounds go on. The weighted vote of T stumps drives the training error to zero, as the boosting theorem promises when the edges stay away from zero."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="stumps kept" value={`${r.learned}${r.stopped ? ` (stopped: ${r.stopped})` : ''}`} />
          <Readout label="first edge γ₁" value={fmt(firstEdge)} />
          <Readout label="last edge" value={fmt(lastEdge)} />
          <Readout label="training error" value={fmt(lastTrain)} />
          <Readout label="test error" value={fmt(lastTest)} />
        </>
      }
      caption="AdaBoost (aifn adaBoost, decision stumps) on 200 training points, tested on 2,000 fresh points from the same distribution. Left: the weighted training error εₜ of each round's stump, on the weights of that round; every value lies below ½ (dashed), so every stump is a weak learner, but the edges shrink as the weights concentrate on hard points. Right: the training error (ink) and test error of the weighted vote after T rounds, on a log scale of rounds. At the opening values the training error reaches zero within a dozen rounds and the test error settles near 0.03. Raise the input noise to 0.3 so that the moons overlap: no stump sequence is consistent any more, the edges shrink faster, the training error stays above zero after 300 rounds, and the test error creeps up as the vote fits noise (0.108 after 50 rounds, 0.117 after 300)."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={eAxis} title="each weak learner">
          <Annotation y={0.5} text="½: chance" dashed />
          <Points name="εₜ" x={roundNumbers} y={r.weighted} slot={0} thin />
        </Plot>
        <Plot x={t2Axis} y={errAxis} title="the vote of T learners">
          <Curve name="training error" x={r.trainIndex} y={r.trainError} emphasis stale={run.stale} />
          <Curve name="test error" x={r.at} y={r.testError} slot={1} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
