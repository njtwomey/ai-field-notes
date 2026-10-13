import { useMemo } from 'react'
import { drawTrainingSet, targetFunction, unitGrid } from 'aifn-methods/theory'
import { fitAndPredict } from 'aifn-methods/theory/bias-variance'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const MAX_DEGREE = 15
const DEGREES = Array.from({ length: MAX_DEGREE + 1 }, (_, d) => d)
const TEST_SIZE = 2000
const GRID = unitGrid(201)
const TRUTH = Array.from(GRID, targetFunction('sine'))
const LOW = 1e-3
const HIGH = 1e3
const Y_LIMIT = 2.5

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const clip = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/**
 * One training set of n points from y = sin(πx) + σε, fitted by least-squares polynomials of every degree from 0 to 15.
 * Left: the data, the truth and the fit of the chosen degree. Right: each fit's training error and its test error on
 * 2000 fresh points from the same distribution, against the degree.
 */
export function FitAndErrors() {
  const state = useFigureState({
    degree: int(3, { min: 0, max: MAX_DEGREE, label: 'polynomial degree' }),
    n: int(20, { min: 5, max: 200, suggestions: [10, 20, 50, 100], label: 'training points n' }),
    noise: slider(0.05, 1, 0.3, { step: 0.05, label: 'noise σ' }),
    seed: int(0, { ge: 0, label: 'seed' }),
  })

  const train = useMemo(
    () => drawTrainingSet(stream(`overfitting/train/${state.seed}`), 'sine', state.n, state.noise),
    [state.seed, state.n, state.noise],
  )
  const test = useMemo(
    () => drawTrainingSet(stream(`overfitting/test/${state.seed}`), 'sine', TEST_SIZE, state.noise),
    [state.seed, state.noise],
  )
  const errors = useMemo(
    () =>
      DEGREES.map((degree) => {
        const model = { kind: 'polynomial', degree } as const
        return {
          train: meanSquaredError(train.y, fitAndPredict(model, train.x, train.y, train.x)),
          test: meanSquaredError(test.y, fitAndPredict(model, train.x, train.y, test.x)),
        }
      }),
    [train, test],
  )
  const fit = useMemo(
    () =>
      Array.from(fitAndPredict({ kind: 'polynomial', degree: state.degree }, train.x, train.y, GRID), (v) =>
        clip(v, -Y_LIMIT - 1, Y_LIMIT + 1),
      ),
    [train, state.degree],
  )

  const chosen = errors[state.degree]
  const noiseVariance = state.noise ** 2

  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-Y_LIMIT, Y_LIMIT] })
  const degreeAxis = useAxis({ label: 'polynomial degree', range: [0, MAX_DEGREE] })
  const errorAxis = useAxis({ label: 'mean squared error', range: [LOW, HIGH], log: true })
  return (
    <Figure
      title="Underfitting, a good fit and overfitting"
      state={state}
      caption="One training set of n points with x uniform on [−1, 1] and y = sin(πx) + σε. Left: the data, the true curve (dashed) and the least-squares polynomial of the chosen degree. Right: for every degree, the fit's mean squared error on its own training points and on 2000 fresh test points from the same distribution; the dashed line is the noise variance σ², the smallest test error any predictor can reach. Training error falls with every added degree. Test error falls while the fit gains the shape of the curve, then rises once the fit follows the noise and swings between and beyond the data. Drag the degree on the right chart; change the seed to draw another training set. Errors are clipped to [0.001, 1000]."
      readouts={
        <>
          <Readout label="training error" value={fmt(chosen.train)} />
          <Readout label="test error" value={fmt(chosen.test)} />
          <Readout label="test − training" value={fmt(chosen.test - chosen.train)} />
          <Readout label="noise σ²" value={fmt(noiseVariance)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={xAxis} y={yAxis} title={`degree ${state.degree}`}>
          <Points name="training points" x={train.x} y={train.y} muted />
          <Curve name="true curve" x={GRID} y={TRUTH} emphasis dashed />
          <Curve name={`degree-${state.degree} fit`} x={GRID} y={fit} slot={0} />
        </Plot>
        <Plot x={degreeAxis} y={errorAxis} title={`n = ${state.n}`}>
          <Curve name="noise σ²" x={[0, MAX_DEGREE]} y={[noiseVariance, noiseVariance]} muted dashed />
          <Curve
            name="training error"
            slot={1}
            x={DEGREES}
            y={errors.map((e) => clip(e.train, LOW, HIGH))}
            showPoints
          />
          <Curve name="test error" slot={0} x={DEGREES} y={errors.map((e) => clip(e.test, LOW, HIGH))} showPoints />
          <Handle {...state.handle('degree', { label: `degree ${state.degree}` })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
