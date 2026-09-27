import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from '@/components/viz'
import { linspace, mean, rng } from '@/lib/math'

const N = 500
const N_TEST = 1000
const BINS = 32
const ROUNDS = 3000
const NOISE = 0.4
const TRUE: [(x: number) => number, (x: number) => number] = [
  (x) => Math.sin(2 * Math.PI * x),
  (x) => 4 * (x - 0.5) ** 2 - 1 / 3,
]
const GRID = linspace(0, 1, 101)
const bin = (x: number) => Math.min(BINS - 1, Math.floor(x * BINS))

type Sample = { bins: [number[], number[]]; y: number[] }

function sample(r: ReturnType<typeof rng>, n: number): Sample {
  const x = Array.from({ length: n }, () => [r.uniform(), r.uniform()])
  return {
    bins: [x.map(([a]) => bin(a)), x.map(([, b]) => bin(b))],
    y: x.map(([a, b]) => TRUE[0](a) + TRUE[1](b) + NOISE * r.normal()),
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
  const [nu, setNu] = useState<'0.01' | '0.1' | '1'>('0.01')
  const [logRounds, setLogRounds] = useState(2)
  const [seed, setSeed] = useState(2)

  const samples = useMemo(() => {
    const r = rng(seed)
    return { train: sample(r, N), test: sample(r, N_TEST) }
  }, [seed])
  const run = useMemo(() => boost(samples.train, samples.test, Number(nu)), [samples, nu])
  const rounds = Math.min(ROUNDS, Math.max(1, Math.round(10 ** logRounds)))
  const state = run.history[rounds - 1]
  const bestTest = run.history.reduce((b, h, i) => (h.testRmse < run.history[b].testRmse ? i : b), 0)

  const panel = (j: 0 | 1) => {
    // Centre each shape over the training points, as EBM reports it; the intercept carries the mean.
    const c = mean(run.trainBins[j].map((b) => state.shapes[j][b]))
    const trueMean = mean(GRID.map(TRUE[j]))
    const xs = Array.from({ length: BINS }, (_, b) => [b / BINS, (b + 1) / BINS]).flat()
    const ys = state.shapes[j].flatMap((v) => [v - c, v - c])
    const series: XYSeries[] = [
      { name: 'true effect', type: 'line', x: GRID, y: GRID.map((g) => TRUE[j](g) - trueMean), slot: 2, dashed: true },
      { name: `f${j === 0 ? '₁' : '₂'} (step function)`, type: 'line', x: xs, y: ys, slot: 1 },
    ]
    return (
      <XYChart
        series={series}
        xRange={[0, 1]}
        yRange={[-1.4, 1.4]}
        xLabel={j === 0 ? 'x₁' : 'x₂'}
        yLabel={j === 0 ? 'f₁(x₁)' : 'f₂(x₂)'}
        height={240}
      />
    )
  }

  return (
    <Interactive
      title="Cyclic boosting builds an additive model of step functions"
      caption={
        <>
          Five hundred points from y = sin(2πx₁) + 4(x₂ − ½)² + noise with standard deviation 0.4. Each feature is cut
          into 32 bins. Every round fits a two-leaf tree to the residuals on x₁ alone, adds it with learning rate ν,
          then does the same for x₂. Each shape function is the running sum of these steps. With ν = 0.01 the shapes
          grow slowly and the round order hardly matters; with ν = 1 each tree takes its full step and the shapes become
          ragged sooner. The test error is measured on a thousand fresh points; its floor is the noise, 0.4.
        </>
      }
      controls={
        <>
          <ParamSlider
            label="rounds"
            value={logRounds}
            onChange={setLogRounds}
            min={0}
            max={Math.log10(ROUNDS)}
            step={0.02}
            format={(v) => String(Math.min(ROUNDS, Math.round(10 ** v)))}
          />
          <ParamChoice
            label="learning rate ν"
            value={nu}
            onChange={setNu}
            options={[
              { value: '0.01', label: '0.01' },
              { value: '0.1', label: '0.1' },
              { value: '1', label: '1' },
            ]}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="rounds" value={String(rounds)} />
          <Readout label="train RMSE" value={formatNumber(state.trainRmse)} />
          <Readout label="test RMSE" value={formatNumber(state.testRmse)} />
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
    </Interactive>
  )
}
