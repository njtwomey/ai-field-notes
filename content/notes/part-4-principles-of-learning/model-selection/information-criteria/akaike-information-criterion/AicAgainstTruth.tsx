import { useMemo } from 'react'
import { drawTrainingSet, targetFunction, unitGrid } from 'aifn-methods/theory'
import { glm } from 'aifn-methods/learning/generalised/glm'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { gaussianFamily } from 'aifn-compute/probability/likelihoods'
import { gaussianLogScore } from 'aifn-compute/learning/metrics'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  Curve,
  Figure,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  toggle,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const DEGREES = Array.from({ length: 11 }, (_, d) => d)
const FRESH = 2000
const REPLICATES = 40
const GRID = unitGrid(201)
const TRUTH = Array.from(GRID, targetFunction('sine'))
const LOW = -2.5
const HIGH = 3.5

const column = (x: Float64Array) => fromData(x, [x.length, 1])
const clip = (v: number) => (Number.isFinite(v) ? Math.max(LOW, Math.min(HIGH, v)) : HIGH)
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—')
const argmin = (v: readonly number[]) => v.reduce((best, x, i) => (x < v[best] ? i : best), 0)

/** A polynomial of degree d with Gaussian noise, fitted by maximum likelihood; the constant is the first feature. */
function fitPolynomial(degree: number, x: Float64Array, y: Float64Array) {
  const features = polynomialFeatures({ degree, includeBias: true }).fit({ x: column(x) })
  const model = glm({ family: gaussianFamily(), intercept: false }).fit({
    x: features.transform(column(x)),
    y: fromData(y),
  })
  const predict = (at: Float64Array): Tensor => model.decide(features.transform(column(at)))
  return { model, predict, sd: Math.sqrt(model.deviance / x.length) }
}

/**
 * For one sample: per degree, the training deviance −2 log L̂ / n, AIC / n, and the true value of what AIC estimates,
 * −2 E log f(Y | θ̂) per observation, measured on fresh data from the same generator.
 */
function scoreSample(seed: string, n: number, noise: number) {
  const s = stream(seed)
  const train = drawTrainingSet(child(s, 'train'), 'sine', n, noise)
  const fresh = drawTrainingSet(child(s, 'fresh'), 'sine', FRESH, noise)
  const rows = DEGREES.map((degree) => {
    const { model, predict, sd } = fitPolynomial(degree, train.x, train.y)
    return {
      training: (-2 * model.logLikelihood) / n,
      aic: model.aic / n,
      truth: 2 * gaussianLogScore(fresh.y, { mean: predict(fresh.x), sd }),
    }
  })
  return { train, rows }
}

/**
 * AIC of polynomial regressions of degree 0 to 10 beside the quantity it estimates. Left: per observation, the training
 * deviance, AIC and the true expected deviance on new data, for one sample or averaged over 40. Right: the sample and the
 * fit at the degree under the handle.
 */
export function AicAgainstTruth() {
  const state = useFigureState({
    degree: int(3, { min: 0, max: 10, label: 'degree' }),
    n: int(40, { min: 20, max: 500, scale: 'log10', suggestions: [20, 40, 100, 400], label: 'observations n' }),
    noise: slider(0.1, 1, 0.3, { step: 0.05, label: 'noise σ' }),
    seed: int(0, { ge: 0, label: 'seed' }),
    average: toggle(false, { label: `average over ${REPLICATES} samples` }),
  })

  const sample = useMemo(
    () => scoreSample(`aic-against-truth/${state.seed}`, state.n, state.noise),
    [state.seed, state.n, state.noise],
  )

  const averaged = useComputed(
    () => {
      if (!state.average) return null
      const runs = Array.from({ length: REPLICATES }, (_, r) =>
        scoreSample(`aic-against-truth/replicate/${state.seed}/${r}`, state.n, state.noise),
      )
      const mean = (key: 'training' | 'aic' | 'truth', d: number) =>
        runs.reduce((a, run) => a + run.rows[d][key], 0) / REPLICATES
      return DEGREES.map((d) => ({ training: mean('training', d), aic: mean('aic', d), truth: mean('truth', d) }))
    },
    [state.average, state.seed, state.n, state.noise],
    { mode: 'release' },
  )

  const rows = state.average && averaged.value ? averaged.value : sample.rows
  const stale = state.average && averaged.stale
  const aicBest = argmin(rows.map((r) => r.aic))
  const trueBest = argmin(rows.map((r) => r.truth))
  const here = rows[state.degree]

  const fit = useMemo(() => {
    const { predict } = fitPolynomial(state.degree, sample.train.x, sample.train.y)
    return Array.from(toFlat(predict(GRID)), (v) => Math.max(-3, Math.min(3, v)))
  }, [sample, state.degree])

  const degreeAxis = useAxis({ label: 'polynomial degree', range: [0, 10] })
  const devianceAxis = useAxis({ label: 'deviance per observation', range: [LOW, HIGH] })
  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2.5, 2.5] })
  return (
    <Figure
      title="AIC against the quantity it estimates"
      state={state}
      caption={`Polynomials of degree 0 to 10 fitted by maximum likelihood, with Gaussian noise of unknown variance, to n points with x uniform on [−1, 1] and y = sin(πx) + σε. Left, per observation: the training deviance −2 log L̂ / n, AIC / n, and the true expected deviance on new data, −2 E log f(Y | θ̂) per observation, measured on ${FRESH} fresh points (dashed). The training deviance keeps falling with the degree. AIC adds 2p/n, with p = degree + 2 parameters, and tracks the true curve. On one sample AIC can sit above or below the truth; switch on the average over ${REPLICATES} samples to see that it is close to unbiased at low degrees. At high degrees AIC undercharges: p is then a sizeable fraction of n, the fit swings between and beyond the training inputs, and the large-sample derivation no longer holds. Right: the sample, the true curve (dashed) and the fit at the degree under the handle. Drag the handle on the left chart. Values are clipped to [${LOW}, ${HIGH}].`}
      readouts={
        <>
          <Readout label="training deviance at degree" value={fmt(here.training)} />
          <Readout label="AIC / n at degree" value={fmt(here.aic)} />
          <Readout label="true deviance at degree" value={fmt(here.truth)} />
          <Readout label="degree chosen by AIC" value={String(aicBest)} />
          <Readout label="degree best on truth" value={String(trueBest)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={degreeAxis} y={devianceAxis} title={state.average ? `mean of ${REPLICATES} samples` : 'one sample'}>
          <Curve
            name="training deviance"
            slot={1}
            x={DEGREES}
            y={rows.map((r) => clip(r.training))}
            showPoints
            stale={stale}
          />
          <Curve name="AIC / n" slot={0} x={DEGREES} y={rows.map((r) => clip(r.aic))} showPoints stale={stale} />
          <Curve
            name="true deviance on new data"
            emphasis
            dashed
            x={DEGREES}
            y={rows.map((r) => clip(r.truth))}
            stale={stale}
          />
          <Handle {...state.handle('degree', { label: 'degree' })} />
        </Plot>
        <Plot x={xAxis} y={yAxis} title={`degree ${state.degree}`}>
          <Points name="training points" x={sample.train.x} y={sample.train.y} muted />
          <Curve name="true curve" x={GRID} y={TRUTH} emphasis dashed />
          <Curve name="fit" x={GRID} y={fit} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}
