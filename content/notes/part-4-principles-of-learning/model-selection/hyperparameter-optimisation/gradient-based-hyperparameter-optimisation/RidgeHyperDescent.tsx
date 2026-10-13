import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { ridgeHyperDescentSteps, ridgeHypergradient } from 'aifn-methods/learning/linear'

const FEATURES = 20
const WEIGHTS = Array.from({ length: FEATURES }, (_, j) => 2 * 0.75 ** j)
const NOISE = 2
const TEST_ROWS = 4000
const LOG_LAMBDA = toFlat(linspace(-2, 3, 101))
const STEPS = 40

/**
 * Gradient descent on log λ for ridge regression, using the implicit-function hypergradient of the validation loss,
 * beside the test error of the same fits on a large fresh sample.
 */
export function RidgeHyperDescent() {
  const state = useFigureState({
    start: slider(-2, 3, -1, { step: 0.05, label: 'start log₁₀ λ₀' }),
    eta: float(1, { gt: 0, le: 10, suggestions: [0.1, 0.3, 1, 3], label: 'step size η on ln λ' }),
    nTrain: int(40, { ge: 25, le: 400, suggestions: [30, 40, 100, 200], label: 'training rows n' }),
    nVal: int(40, { ge: 5, le: 400, suggestions: [10, 20, 40, 100], label: 'validation rows m' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { start, eta, nTrain, nVal, seed } = state

  const data = useMemo(() => {
    const s = stream(`ridge-hyper-descent/${seed}`)
    const make = (name: string, n: number) => {
      const d = linearRegressionData(child(s, name), { n, weights: WEIGHTS, noise: NOISE })
      if (!d.y) throw new Error('linearRegressionData returned no targets')
      return { x: d.x, y: d.y }
    }
    const train = make('train', nTrain)
    const val = make('val', nVal)
    const test = make('test', TEST_ROWS)
    const lambdas = LOG_LAMBDA.map((v) => 10 ** v)
    const valLoss = lambdas.map((l) => ridgeHypergradient(train.x, train.y, val.x, val.y, l).loss)
    const testLoss = lambdas.map((l) => ridgeHypergradient(train.x, train.y, test.x, test.y, l).loss)
    return { train, val, test, lambdas, valLoss, testLoss }
  }, [nTrain, nVal, seed])

  const steps = useMemo(() => {
    const alg = ridgeHyperDescentSteps({
      xTrain: data.train.x,
      yTrain: data.train.y,
      xVal: data.val.x,
      yVal: data.val.y,
      eta,
    })
    return trace(alg, { lambda: 10 ** start }, STEPS, { keep: 'all' }).steps
  }, [data, eta, start])

  // The visible step, back to 0 whenever the run changes.
  const key = `${start}|${eta}|${nTrain}|${nVal}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const at = Math.min(cursor.key === key ? cursor.step : 0, steps.length - 1)
  const now = steps[at]
  const testNow = ridgeHypergradient(data.train.x, data.train.y, data.test.x, data.test.y, now.lambda).loss

  // The tangent of the validation loss in ln λ at the current iterate, over a factor of 3 either side.
  const tangent = [now.lambda / 3, now.lambda * 3]
  const tangentY = tangent.map((l) => now.loss + now.gradient * Math.log(l / now.lambda))
  const path = steps.slice(0, at + 1)
  const argminOf = (ys: number[]) => data.lambdas[ys.indexOf(Math.min(...ys))]

  const xAxis = useAxis({ label: 'penalty λ', log: true, range: [0.01, 1000] })
  const yAxis = useAxis({ label: 'mean squared error', hold: 'union' })
  return (
    <Figure
      title="Descending on a ridge penalty by its hypergradient"
      state={state}
      caption="Ridge regression on n training rows of 20 standard-normal inputs with true coefficients 2·0.75^(j−1) and noise of standard deviation 2. Orange: the validation loss on m rows, the objective being minimised. Text colour: the test error of the same fits on 4,000 fresh rows, the truth the validation loss estimates. Each step of the player moves ln λ against the hypergradient dL/d ln λ, computed by the implicit function theorem from one extra solve; the short line is the tangent it follows. Drag the start λ₀. A large step size overshoots; a small validation set puts the validation minimum far from the test minimum."
      controls={<Player value={at} onChange={(v) => setCursor({ key, step: v })} count={steps.length} label="step" />}
      readouts={
        <>
          <Readout label="step" value={String(now.t)} />
          <Readout label="λ" value={formatNumber(now.lambda)} />
          <Readout label="validation loss" value={formatNumber(now.loss)} />
          <Readout label="dL/d ln λ" value={formatNumber(now.gradient)} />
          <Readout label="test error at λ" value={formatNumber(testNow)} />
          <Readout label="λ minimising validation (grid)" value={formatNumber(argminOf(data.valLoss))} />
          <Readout label="λ minimising test (grid)" value={formatNumber(argminOf(data.testLoss))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve name="validation loss" x={data.lambdas} y={data.valLoss} slot={1} />
        <Curve name="test error (truth)" x={data.lambdas} y={data.testLoss} emphasis />
        <Curve name="tangent" x={tangent} y={tangentY} muted dashed />
        <Points name="iterates" x={path.map((s) => s.lambda)} y={path.map((s) => s.loss)} slot={1} />
        <Handle kind="x" at={10 ** start} onDrag={(x: number) => state.set('start', Math.log10(x))} label="λ₀" />
      </Plot>
    </Figure>
  )
}
