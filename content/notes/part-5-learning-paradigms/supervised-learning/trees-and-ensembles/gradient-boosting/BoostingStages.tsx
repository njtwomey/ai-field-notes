import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { fitTree, predictTree, type TreeNode } from '../_shared/trees'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N_TRAIN = 80
const N_TEST = 400
const NOISE = 0.3
const M_MAX = 300
const GRID = toFlat(linspace(0, 10, 201))
const X_RANGE: [number, number] = [0, 10]
const Y_RANGE: [number | undefined, number | undefined] = [-2.2, 2.2]
const truth = (x: number) => Math.sin(x) + 0.4 * Math.sin(3 * x)

function sample(n: number, seed: number) {
  const g = stream(seed)
  const x = Array.from({ length: n }, () => 10 * uniform(g))
  return { x, y: x.map((v) => truth(v) + NOISE * normal(g)) }
}

const DEPTHS = [
  { value: '1', label: 'stumps' },
  { value: '2', label: 'depth 2' },
  { value: '3', label: 'depth 3' },
] as const

/** Gradient boosting with squared loss: each stage fits a small tree to the residuals and adds a shrunken copy. */
export function BoostingStages() {
  const state = useFigureState({
    stage: int(20, { min: 0, max: M_MAX, step: 1, label: 'stages M', format: (v) => String(v) }),
    rate: float(0.1, { min: 0.01, max: 1, step: 0.01, label: 'learning rate ν' }),
    depth: choice<'1' | '2' | '3'>(DEPTHS, '1', { label: 'trees' }),
  })

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
      const tree = fitTree(X, r, { maxDepth: Number(state.depth), criterion: 'squared' })
      trees.push(tree)
      train.x.forEach((v, i) => (F[i] += state.rate * predictTree(tree, [v])))
      test.x.forEach((v, i) => (Ftest[i] += state.rate * predictTree(tree, [v])))
      trainLoss.push(mse(F, train.y))
      testLoss.push(mse(Ftest, test.y))
    }
    return { f0, trees, trainLoss, testLoss }
  }, [train, test, state.rate, state.depth])

  const curve = useMemo(
    () =>
      GRID.map(
        (v) => fit.f0 + fit.trees.slice(0, state.stage).reduce((a, t) => a + state.rate * predictTree(t, [v]), 0),
      ),
    [fit, state.stage, state.rate],
  )

  const best = fit.testLoss.indexOf(Math.min(...fit.testLoss))
  const stages = fit.trainLoss.map((_, m) => m)
  const fitSeries = [
    { name: 'training data', x: train.x, y: train.y, muted: true },
    { name: 'true function', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
    { name: `model after ${state.stage} stages`, x: GRID, y: curve, slot: 0 },
  ] as const
  const lossSeries = [
    { name: 'training MSE', x: stages, y: fit.trainLoss, slot: 0 },
    { name: 'test MSE', x: stages, y: fit.testLoss, slot: 1 },
    { name: 'noise variance', x: [0, M_MAX], y: [NOISE ** 2, NOISE ** 2], muted: true, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'y', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'stage m', hold: 'union' })
  const yAxis2 = useAxis({ label: 'mean squared error', range: [0, 0.6] })
  return (
    <Figure
      title="Gradient boosting, stage by stage"
      state={state}
      caption="Eighty noisy points from the dashed curve. The model starts at the mean of y. Each stage fits a small regression tree to the current residuals and adds it, multiplied by the learning rate. Right: training and test mean squared error against the number of stages, with the noise variance 0.09 as the floor for test error. Training error keeps falling. Test error falls to a minimum well above the noise floor, because 80 points cannot pin the curve down, then rises as later trees fit noise. A smaller learning rate needs proportionally more stages and flattens the test curve around its minimum, so the exact number of stages matters less. Deeper trees reach the minimum in fewer stages. Drag the stage line or use the slider."

      readouts={
        <>
          <Readout label="training MSE" value={formatNumber(fit.trainLoss[state.stage])} />
          <Readout label="test MSE" value={formatNumber(fit.testLoss[state.stage])} />
          <Readout label="best stage (test)" value={`${best} (${formatNumber(fit.testLoss[best])})`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Points {...fitSeries[0]} />
          <Curve {...fitSeries[1]} />
          <Curve {...fitSeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...lossSeries[0]} />
          <Curve {...lossSeries[1]} />
          <Curve {...lossSeries[2]} />
          <Handle kind="x" at={state.stage} label="stage" onDrag={(x) => state.set('stage', Math.round(x))} />
        </Plot>
      </div>
    </Figure>
  )
}
