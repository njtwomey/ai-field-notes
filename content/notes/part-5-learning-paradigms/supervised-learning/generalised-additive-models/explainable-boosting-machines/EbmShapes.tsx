import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
  type AxisModel,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, type Stream, uniform } from 'aifn-compute/foundation/random'

const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

const N = 500
const N_TEST = 1000
const BINS = 32
const ROUNDS = 3000
const NOISE = 0.4
const TRUE: [(x: number) => number, (x: number) => number] = [
  (x) => Math.sin(2 * Math.PI * x),
  (x) => 4 * (x - 0.5) ** 2 - 1 / 3,
]
const GRID = toFlat(linspace(0, 1, 101))
const bin = (x: number) => Math.min(BINS - 1, Math.floor(x * BINS))

type Sample = { bins: [number[], number[]]; y: number[] }

function sample(r: Stream, n: number): Sample {
  const x = Array.from({ length: n }, () => [uniform(r), uniform(r)])
  return {
    bins: [x.map(([a]) => bin(a)), x.map(([, b]) => bin(b))],
    y: x.map(([a, b]) => TRUE[0](a) + TRUE[1](b) + NOISE * normal(r)),
  }
}

/**
 * Cyclic boosting: in every round, for each feature in turn, fit a two-leaf tree on that feature's bins to the current
 * residuals and add it with learning rate ν. Records both shape functions after every round.
 */
function boost(train: Sample, test: Sample, nu: number) {
  const intercept = mean(train.y)
  const shape = [Array(BINS).fill(0), Array(BINS).fill(0)]
  const pred = train.y.map(() => intercept)
  const history: { shapes: number[][]; trainRmse: number; testRmse: number }[] = []
  for (let m = 0; m < ROUNDS; m++) {
    for (const j of [0, 1]) {
      const sum = Array(BINS).fill(0)
      const count = Array(BINS).fill(0)
      train.bins[j].forEach((b, i) => {
        sum[b] += train.y[i] - pred[i]
        count[b]++
      })
      const total = sum.reduce((a, b) => a + b, 0)
      // The best split maximises S_L²/n_L + S_R²/n_R, the drop in squared error for a two-leaf fit.
      let [best, cut, sL, nL] = [-Infinity, 0, 0, 0]
      let [runS, runN] = [0, 0]
      for (let b = 0; b < BINS - 1; b++) {
        runS += sum[b]
        runN += count[b]
        if (runN === 0 || runN === N) continue
        const gain = runS ** 2 / runN + (total - runS) ** 2 / (N - runN)
        if (gain > best) [best, cut, sL, nL] = [gain, b, runS, runN]
      }
      const left = (nu * sL) / nL
      const right = (nu * (total - sL)) / (N - nL)
      for (let b = 0; b < BINS; b++) shape[j][b] += b <= cut ? left : right
      train.bins[j].forEach((b, i) => (pred[i] += b <= cut ? left : right))
    }
    const rmse = (s: Sample, p: number[]) => Math.sqrt(mean(s.y.map((y, i) => (y - p[i]) ** 2)))
    const testPred = test.y.map((_, i) => intercept + shape[0][test.bins[0][i]] + shape[1][test.bins[1][i]])
    history.push({ shapes: shape.map((s) => [...s]), trainRmse: rmse(train, pred), testRmse: rmse(test, testPred) })
  }
  return { history, trainBins: train.bins }
}

export function EbmShapes() {
  const state = useFigureState({
    rounds: int(100, { ge: 1, le: ROUNDS, scale: 'log10', suggestions: [1, 10, 100, 1000, ROUNDS], label: 'rounds' }),
    nu: float(0.01, { gt: 0, le: 1, scale: 'log10', suggestions: [0.01, 0.1, 1], label: 'learning rate ν' }),
    seed: int(2, { ge: 0, label: 'seed' }),
  })
  const { nu, seed } = state

  const samples = useMemo(() => {
    const r = stream(seed)
    return { train: sample(r, N), test: sample(r, N_TEST) }
  }, [seed])
  const run = useMemo(() => boost(samples.train, samples.test, nu), [samples, nu])
  const rounds = Math.min(ROUNDS, Math.max(1, Math.round(state.rounds)))
  const current = run.history[rounds - 1]
  const bestTest = run.history.reduce((b, h, i) => (h.testRmse < run.history[b].testRmse ? i : b), 0)

  const axes: [AxisModel, AxisModel][] = [
    [useAxis({ label: 'x₁', range: [0, 1] }), useAxis({ label: 'f₁(x₁)', range: [-1.4, 1.4] })],
    [useAxis({ label: 'x₂', range: [0, 1] }), useAxis({ label: 'f₂(x₂)', range: [-1.4, 1.4] })],
  ]
  const panel = (j: 0 | 1) => {
    // Centre each shape over the training points, as EBM reports it; the intercept carries the mean.
    const c = mean(run.trainBins[j].map((b) => current.shapes[j][b]))
    const trueMean = mean(GRID.map(TRUE[j]))
    const xs = Array.from({ length: BINS }, (_, b) => [b / BINS, (b + 1) / BINS]).flat()
    const ys = current.shapes[j].flatMap((v) => [v - c, v - c])
    return (
      <Plot x={axes[j][0]} y={axes[j][1]} height={240}>
        <Curve name="true effect" x={GRID} y={GRID.map((g) => TRUE[j](g) - trueMean)} slot={2} dashed />
        <Curve name={`f${j === 0 ? '₁' : '₂'} (step function)`} x={xs} y={ys} slot={1} />
      </Plot>
    )
  }

  return (
    <Figure
      title="Cyclic boosting builds an additive model of step functions"
      state={state}
      caption={
        <>
          Five hundred points from y = sin(2πx₁) + 4(x₂ − ½)² + noise with standard deviation 0.4. Each feature is cut
          into 32 bins. Every round fits a two-leaf tree to the residuals on x₁ alone, adds it with learning rate ν,
          then does the same for x₂. Each shape function is the running sum of these steps. With ν = 0.01 the shapes
          grow slowly and the round order hardly matters; with ν = 1 each tree takes its full step and the shapes become
          ragged sooner. The test error is measured on a thousand fresh points; its floor is the noise, 0.4.
        </>
      }
      readouts={
        <>
          <Readout label="rounds" value={String(rounds)} />
          <Readout label="train RMSE" value={formatNumber(current.trainRmse)} />
          <Readout label="test RMSE" value={formatNumber(current.testRmse)} />
          <Readout
            label="best test RMSE (round)"
            value={`${formatNumber(run.history[bestTest].testRmse)} (${bestTest + 1})`}
          />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        {panel(0)}
        {panel(1)}
      </div>
    </Figure>
  )
}
