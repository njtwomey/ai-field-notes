import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { regression1d } from 'aifn-methods/data/synthetic'
import { linearRegression } from 'aifn-methods/learning/linear'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat } from 'aifn-compute/foundation/tensor'

const DEGREE = 12
const TEST = 2000
const NOISE = 0.2
const SIGMAS = [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5]
const GRID = Array.from(toFlat(linspace(-1, 1, 201)))
const TRUE = (x: number) => Math.sin(Math.PI * x)
const TRUTH = GRID.map(TRUE)
const CLIP = 3

const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))
const column = (x: ArrayLike<number>) => fromData(Float64Array.from(x), [x.length, 1])
const model = () => pipeline(polynomialFeatures({ degree: DEGREE }), linearRegression())

/**
 * A degree-12 polynomial fitted by least squares to K noisy copies of each of n points, the inputs jittered by noise of
 * standard deviation σ, against the fit to the clean points and the true error at each σ.
 */
export function NoisyInputFits() {
  const state = useFigureState({
    sigma: float(0.1, { min: 0, max: 0.5, step: 0.01, label: 'input noise σ' }),
    copies: int(50, { ge: 1, le: 500, suggestions: [1, 10, 50, 200], label: 'noisy copies K per point' }),
    n: int(15, { ge: 14, le: 200, suggestions: [15, 30, 100], label: 'training points n' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3], label: 'data set (seed)' }),
  })
  const { copies, n, seed } = state

  const base = useMemo(() => {
    const train = regression1d(child(stream(`noise-injection/${seed}`), 'train'), {
      n,
      fn: TRUE,
      noise: NOISE,
      range: [-1, 1],
    })
    const test = regression1d(child(stream('noise-injection'), 'test'), {
      n: TEST,
      fn: TRUE,
      noise: NOISE,
      range: [-1, 1],
    })
    const x = Array.from(toFlat(train.x))
    const y = Array.from(toFlat(train.y!))
    return { x, y, test }
  }, [n, seed])

  /** Fit on K copies of each point with inputs x + σz; the same draws z serve every σ. */
  const fitAt = useMemo(() => {
    const centres = fromData(Float64Array.from({ length: copies * n }, (_, i) => base.x[i % n]))
    const targets = fromData(Float64Array.from({ length: copies * n }, (_, i) => base.y[i % n]))
    const s = child(stream(`noise-injection/${seed}`), 'jitter')
    return (sigma: number) => {
      const xs = sigma === 0 ? centres : normals(s, [copies * n], centres, sigma)
      return model().fit(dataset(column(toFlat(xs)), targets))
    }
  }, [base, copies, n, seed])

  const sweep = useComputed(
    () => SIGMAS.map((sigma) => meanSquaredError(base.test.y!, toFlat(fitAt(sigma).decide(base.test.x)))),
    [fitAt, base],
    { mode: 'release' },
  )
  const curves = useMemo(() => {
    const clip = (v: ArrayLike<number>) => Array.from(v, (u) => Math.max(-CLIP, Math.min(CLIP, u)))
    const clean = fitAt(0)
    const noisy = fitAt(state.sigma)
    return {
      clean: clip(toFlat(clean.decide(column(GRID)))),
      noisy: clip(toFlat(noisy.decide(column(GRID)))),
      error: meanSquaredError(base.test.y!, toFlat(noisy.decide(base.test.x))),
    }
  }, [fitAt, state.sigma, base])

  const errors = sweep.value
  const bestIndex = errors.reduce((b, v, i) => (v < errors[b] ? i : b), 0)
  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-CLIP, CLIP] })
  const sAxis = useAxis({ label: 'input noise σ', range: [0, 0.5] })
  const eAxis = useAxis({ label: 'true error', log: true })
  return (
    <Figure
      title="Jittered inputs smooth a polynomial fit"
      state={state}
      caption={`Left: n points from y = sin(πx) + ε with noise of standard deviation ${NOISE}, the true curve (dashed), a degree-${DEGREE} polynomial fitted by least squares to the clean points, and the same polynomial fitted to K copies of each point with its input jittered by Gaussian noise of standard deviation σ. Right: the true error (on ${TEST} fresh points) of the jittered fit against σ; drag the handle to change σ. Without jitter the polynomial oscillates between the points; a little jitter penalises its slope and smooths it, and too much flattens the signal. Values are clipped to [−${CLIP}, ${CLIP}].`}
      readouts={
        <>
          <Readout label="true error, clean fit" value={fmt(errors[0])} />
          <Readout label="true error at σ" value={fmt(curves.error)} />
          <Readout label="best σ on the grid" value={`${SIGMAS[bestIndex]} (error ${fmt(errors[bestIndex])})`} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={xAxis} y={yAxis}>
          <Points name="training points" x={base.x} y={base.y} muted />
          <Curve name="true curve" x={GRID} y={TRUTH} emphasis dashed />
          <Curve name="clean fit" x={GRID} y={curves.clean} slot={1} />
          <Curve name={`jittered fit, σ = ${state.sigma}`} x={GRID} y={curves.noisy} slot={0} />
        </Plot>
        <Plot x={sAxis} y={eAxis}>
          <Curve name="true error" x={SIGMAS} y={errors} emphasis showPoints stale={sweep.stale} />
          <Handle {...state.handle('sigma', { label: 'σ' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
