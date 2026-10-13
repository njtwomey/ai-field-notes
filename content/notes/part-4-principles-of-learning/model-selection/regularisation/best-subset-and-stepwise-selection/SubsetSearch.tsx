import { useMemo } from 'react'
import {
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
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { bestSubset, stepwise } from 'aifn-methods/learning/linear'
import { child, stream } from 'aifn-compute/foundation/random'
import { matmul, toRows, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'

/** Ten predictors, three of them active. */
const WEIGHTS = [3, 1.5, 0, 0, 2, 0, 0, 0, 0, 0]
const P = WEIGHTS.length
const NOISE = 3
const TEST = 4000
const SIZES = Array.from({ length: P + 1 }, (_, k) => k)

const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/** Mean squared error on (x, y) of each column of a coefficient matrix, with its intercept. */
function errors(x: Tensor, y: Tensor, coef: Tensor, intercept: readonly number[]) {
  const preds = toRows(transpose(matmul(x, coef))) as number[][]
  return preds.map((row, k) =>
    meanSquaredError(
      y,
      row.map((v) => v + intercept[k]),
    ),
  )
}

/**
 * Best subset against forward and backward stepwise selection on ten correlated predictors: the training residual sum
 * of squares of every subset against its size, and the true error of each method's model at each size.
 */
export function SubsetSearch() {
  const state = useFigureState({
    n: int(40, { ge: 15, le: 400, suggestions: [20, 40, 100, 400], label: 'training rows n' }),
    correlation: slider(0, 0.9, 0.6, { step: 0.1, label: 'neighbour correlation ρ' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3], label: 'data set (seed)' }),
  })
  const { n, correlation, seed } = state

  const computed = useComputed(
    () => {
      const options = { weights: WEIGHTS, noise: NOISE, correlation }
      const train = linearRegressionData(child(stream(`best-subset/${seed}`), 'train'), { n, ...options })
      const test = linearRegressionData(child(stream('best-subset'), 'test'), { n: TEST, ...options })
      const best = bestSubset(train.x, train.y!)
      const forward = stepwise(train.x, train.y!, { direction: 'forward' })
      const backward = stepwise(train.x, train.y!, { direction: 'backward' })
      // Backward visits the sizes from P down to 0; reverse it so index k is size k.
      const backwardBySize = [...backward.steps].reverse()
      const backwardTest = errors(test.x, test.y!, backward.coef, backward.intercept).reverse()
      return {
        all: { sizes: best.all.sizes, rss: best.all.rss.map((r) => r / n) },
        best: { rss: best.rss.map((r) => r / n), test: errors(test.x, test.y!, best.coef, best.intercept) },
        forward: {
          rss: forward.steps.map((s) => s.rss / n),
          test: errors(test.x, test.y!, forward.coef, forward.intercept),
          sets: forward.steps.map((s) => s.active),
        },
        backward: {
          rss: backwardBySize.map((s) => s.rss / n),
          test: backwardTest,
          sets: backwardBySize.map((s) => s.active),
        },
        bestSets: best.subsets,
      }
    },
    [n, correlation, seed],
    { mode: 'release' },
  )
  const r = computed.value

  const view = useMemo(() => {
    const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i])
    const misses = (sets: number[][]) => SIZES.filter((k) => !same(sets[k], r.bestSets[k])).join(', ') || 'none'
    const argmin = (v: number[]) => v.reduce((b, x, i) => (x < v[b] ? i : b), 0)
    return {
      forwardMisses: misses(r.forward.sets),
      backwardMisses: misses(r.backward.sets),
      bestK: argmin(r.best.test),
      forwardK: argmin(r.forward.test),
      backwardK: argmin(r.backward.test),
    }
  }, [r])

  const kx = useAxis({ label: 'subset size k', range: [-0.3, P + 0.3] })
  const rssY = useAxis({ label: 'training RSS / n' })
  const errY = useAxis({ label: 'true error' })
  const chosen = (label: string, k: number, v: number[]) => (
    <Readout label={label} value={`k = ${k}, true error ${fmt(v[k])}`} />
  )
  return (
    <Figure
      title="Every subset, the best ones and the greedy ones"
      state={state}
      caption={`Least squares on subsets of 10 standard-normal predictors, neighbours correlated ρ, with true coefficients (3, 1.5, 0, 0, 2, 0, …, 0) and noise of standard deviation ${NOISE}. Top: the training residual sum of squares over n of all 1,024 subsets (grey), plotted against their size; best subset is the lower envelope, and forward and backward stepwise follow their own sequences. Bottom: the true error of each method's model of each size, measured on ${TEST} fresh rows. The training error falls with every predictor added; the true error is smallest near the three true predictors. Raise ρ to make the greedy searches miss the best subset more often.`}
      readouts={
        <>
          <Readout label="sizes where forward stepwise misses the best subset" value={view.forwardMisses} />
          <Readout label="sizes where backward stepwise misses it" value={view.backwardMisses} />
          {chosen('best size for best subset', view.bestK, r.best.test)}
          {chosen('best size for forward stepwise', view.forwardK, r.forward.test)}
          {chosen('best size for backward stepwise', view.backwardK, r.backward.test)}
        </>
      }
    >
      <Plots rows={2} cols={1} heights={[3, 2]}>
        <Plot x={kx} y={rssY}>
          <Points name="every subset" x={r.all.sizes} y={r.all.rss} muted thin stale={computed.stale} />
          <Curve name="best subset" x={SIZES} y={r.best.rss} emphasis showPoints stale={computed.stale} />
          <Curve name="forward stepwise" x={SIZES} y={r.forward.rss} slot={0} showPoints stale={computed.stale} />
          <Curve
            name="backward stepwise"
            x={SIZES}
            y={r.backward.rss}
            slot={1}
            dashed
            showPoints
            stale={computed.stale}
          />
        </Plot>
        <Plot x={kx} y={errY}>
          <Curve name="best subset" x={SIZES} y={r.best.test} emphasis showPoints stale={computed.stale} />
          <Curve name="forward stepwise" x={SIZES} y={r.forward.test} slot={0} showPoints stale={computed.stale} />
          <Curve
            name="backward stepwise"
            x={SIZES}
            y={r.backward.test}
            slot={1}
            dashed
            showPoints
            stale={computed.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
