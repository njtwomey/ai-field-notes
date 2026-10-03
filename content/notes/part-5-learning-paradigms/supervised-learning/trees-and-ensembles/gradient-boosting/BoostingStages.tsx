import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { fitTree, predictTree, type TreeNode } from '../_shared/trees'

const N_TRAIN = 80
const N_TEST = 400
const NOISE = 0.3
const M_MAX = 300
const GRID = linspace(0, 10, 201)
const X_RANGE: [number, number] = [0, 10]
const Y_RANGE: [number | undefined, number | undefined] = [-2.2, 2.2]
const truth = (x: number) => Math.sin(x) + 0.4 * Math.sin(3 * x)

function sample(n: number, seed: number) {
  const g = rng(seed)
  const x = Array.from({ length: n }, () => 10 * g.uniform())
  return { x, y: x.map((v) => truth(v) + NOISE * g.normal()) }
}

const DEPTHS = [
  { value: '1', label: 'stumps' },
  { value: '2', label: 'depth 2' },
  { value: '3', label: 'depth 3' },
] as const

/** Gradient boosting with squared loss: each stage fits a small tree to the residuals and adds a shrunken copy. */
export function BoostingStages() {
  const stage = useParam(20, { min: 0, max: M_MAX, step: 1 })
  const rate = useParam(0.1, { min: 0.01, max: 1, step: 0.01 })
  const [depth, setDepth] = useState<'1' | '2' | '3'>('1')

  const train = useMemo(() => sample(N_TRAIN, 11), [])
  const test = useMemo(() => sample(N_TEST, 12), [])

  const fit = useMemo(() => {
    const X = train.x.map((v) => [v])
    const f0 = train.y.reduce((a, v) => a + v, 0) / N_TRAIN
    const F = train.x.map(() => f0)
    const Ftest = test.x.map(() => f0)
    const trees: TreeNode[] = []
    const mse = (pred: number[], y: number[]) => pred.reduce((a, p, i) => a + (y[i] - p) ** 2, 0) / y.length
    const trainLoss = [mse(F, train.y)]
    const testLoss = [mse(Ftest, test.y)]
    for (let m = 0; m < M_MAX; m++) {
      // Pseudo-residuals of the squared loss ½(y − F)² are the ordinary residuals.
      const r = train.y.map((y, i) => y - F[i])
      const tree = fitTree(X, r, { maxDepth: Number(depth), criterion: 'squared' })
      trees.push(tree)
      train.x.forEach((v, i) => (F[i] += rate.value * predictTree(tree, [v])))
      test.x.forEach((v, i) => (Ftest[i] += rate.value * predictTree(tree, [v])))
      trainLoss.push(mse(F, train.y))
      testLoss.push(mse(Ftest, test.y))
    }
    return { f0, trees, trainLoss, testLoss }
  }, [train, test, rate.value, depth])

  const curve = useMemo(
    () =>
      GRID.map(
        (v) => fit.f0 + fit.trees.slice(0, stage.value).reduce((a, t) => a + rate.value * predictTree(t, [v]), 0),
      ),
    [fit, stage.value, rate.value],
  )

  const best = fit.testLoss.indexOf(Math.min(...fit.testLoss))
  const stages = fit.trainLoss.map((_, m) => m)
  const fitSeries: XYSeries[] = [
    { name: 'training data', type: 'scatter', x: train.x, y: train.y, muted: true },
    { name: 'true function', type: 'line', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
    { name: `model after ${stage.value} stages`, type: 'line', x: GRID, y: curve, slot: 0 },
  ]
  const lossSeries: XYSeries[] = [
    { name: 'training MSE', type: 'line', x: stages, y: fit.trainLoss, slot: 0 },
    { name: 'test MSE', type: 'line', x: stages, y: fit.testLoss, slot: 1 },
    { name: 'noise variance', type: 'line', x: [0, M_MAX], y: [NOISE ** 2, NOISE ** 2], muted: true, dashed: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: stage.value, label: 'stage', onDrag: (x) => stage.set(Math.round(x)) }]

  return (
    <Interactive
      title="Gradient boosting, stage by stage"
      caption="Eighty noisy points from the dashed curve. The model starts at the mean of y. Each stage fits a small regression tree to the current residuals and adds it, multiplied by the learning rate. Right: training and test mean squared error against the number of stages, with the noise variance 0.09 as the floor for test error. Training error keeps falling. Test error falls to a minimum well above the noise floor, because 80 points cannot pin the curve down, then rises as later trees fit noise. A smaller learning rate needs proportionally more stages and flattens the test curve around its minimum, so the exact number of stages matters less. Deeper trees reach the minimum in fewer stages. Drag the stage line or use the slider."
      controls={
        <>
          <ParamSlider label="stages M" param={stage} format={(v) => String(v)} withArrows />
          <ParamSlider label="learning rate ν" param={rate} />
          <ParamChoice label="trees" value={depth} onChange={setDepth} options={DEPTHS} />
        </>
      }
      readout={
        <>
          <Readout label="training MSE" value={formatNumber(fit.trainLoss[stage.value])} />
          <Readout label="test MSE" value={formatNumber(fit.testLoss[stage.value])} />
          <Readout label="best stage (test)" value={`${best} (${formatNumber(fit.testLoss[best])})`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={fitSeries} xLabel="x" yLabel="y" xRange={X_RANGE} yRange={Y_RANGE} height={320} />
        <XYChart
          series={lossSeries}
          xLabel="stage m"
          yLabel="mean squared error"
          yRange={[0, 0.6]}
          handles={handles}
          height={320}
        />
      </div>
    </Interactive>
  )
}
