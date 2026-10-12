import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Rug,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import {
  drawSamples,
  kernelRatio,
  logRatio,
  pickCentres,
  ratioErrors,
  ulsif,
  ulsifScores,
  type Shift,
} from '../_shared/ratio'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 241))
const LOG_SIGMA = toFlat(linspace(-1, 0.5, 13))
const LOG_LAMBDA = toFlat(linspace(-3, 1, 13))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/**
 * uLSIF's closed-form fit for a chosen kernel width σ and penalty λ, beside its cross-validated squared-loss score over
 * a grid of (σ, λ). The score needs only the two samples, so it picks both without knowing the true ratio.
 */
export function LsifSelection() {
  const state = useFigureState({
    logSigma: float(-0.3, { min: -1, max: 0.5, step: 0.01, label: 'log₁₀ σ', onChart: true }),
    logLambda: float(-1, { min: -3, max: 1, step: 0.01, label: 'log₁₀ λ', onChart: true }),
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p' }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    n: int(200, { min: 20, max: 1000, label: 'points in each sample', suggestions: [50, 200, 1000] }),
    b: int(50, { min: 5, max: 100, label: 'kernel centres b', suggestions: [10, 50, 100] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const data = useMemo(() => {
    const { xp, xq } = drawSamples(state.seed, state.n, state.n, shift)
    return { xp, xq, centres: pickCentres(xp, state.b) }
  }, [state.seed, state.n, shift, state.b])

  // Rows follow λ (the y axis), columns σ (the x axis).
  const cv = useComputed(
    () => {
      const bySigma = ulsifScores(
        data.xp,
        data.xq,
        data.centres,
        LOG_SIGMA.map((v) => 10 ** v),
        LOG_LAMBDA.map((v) => 10 ** v),
      )
      const z = LOG_LAMBDA.map((_, j) => LOG_SIGMA.map((_, i) => bySigma[i][j]))
      let best = { i: 0, j: 0 }
      z.forEach((row, j) => row.forEach((v, i) => v < z[best.j][best.i] && (best = { i, j })))
      const low = z[best.j][best.i]
      // The scores span orders of magnitude, so the map shows log₁₀ of the gap to the lowest one.
      const gap = z.map((row) => row.map((v) => Math.log10(v - low + 1e-3)))
      return { gap, best: { logSigma: LOG_SIGMA[best.i], logLambda: LOG_LAMBDA[best.j], score: low } }
    },
    [data],
    { mode: 'release' },
  )

  const sigma = 10 ** state.logSigma
  const lambda = 10 ** state.logLambda
  const fit = useMemo(() => {
    const alpha = ulsif(data.xp, data.xq, data.centres, sigma, lambda)
    return { alpha, r: kernelRatio(alpha, data.centres, sigma, X) }
  }, [data, sigma, lambda])
  const truth = useMemo(() => X.map((x) => Math.exp(logRatio(x, shift))), [shift])
  const errors = ratioErrors(
    X,
    fit.r.map((v) => Math.log(v)),
    shift,
  )
  const zeros = fit.alpha.filter((a) => a === 0).length

  const drag = ([x, y]: [number, number]) => {
    state.set('logSigma', clamp(x, -1, 0.5))
    state.set('logLambda', clamp(y, -3, 1))
  }

  const xa = useAxis({ label: 'log₁₀ σ', range: [-1, 0.5] })
  const ya = useAxis({ label: 'log₁₀ λ', range: [-3, 1] })
  const xb = useAxis({ label: 'x', range: [LO, HI] })
  const yb = useAxis({ label: 'r(x)', range: [0, undefined] })
  return (
    <Figure
      title="Choosing σ and λ by cross-validation"
      state={state}
      caption="Left: the five-fold cross-validated score ½ mean over held-out q points of r̂² − mean over held-out p points of r̂, for every kernel width σ and penalty λ. Lower is better, and it differs from the squared error ½ E_q[(r̂ − r)²] by a constant, so its minimum (the square) estimates the best pair without the true ratio. The colour is log₁₀ of each score's excess over the lowest, since the scores span orders of magnitude. Drag the marker to choose a pair. Right: uLSIF's fit for that pair against the true ratio (dashed). A narrow kernel with a small penalty follows individual points; a wide kernel or a large penalty flattens the fit towards a constant."
      readouts={
        <>
          <Readout label="σ" value={formatNumber(sigma)} />
          <Readout label="λ" value={formatNumber(lambda)} />
          <Readout label="squared error ½ E_q[(r̂ − r)²]" value={formatNumber(errors.squared)} />
          <Readout label="coefficients clipped to 0" value={`${zeros} of ${fit.alpha.length}`} />
          <Readout
            label="cross-validation's choice"
            value={`σ = ${formatNumber(10 ** cv.value.best.logSigma)}, λ = ${formatNumber(10 ** cv.value.best.logLambda)}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={300}>
          <Raster x={LOG_SIGMA} y={LOG_LAMBDA} z={cv.value.gap} valueLabel="log₁₀ excess" stale={cv.stale} />
          <Points
            name="lowest score"
            x={[cv.value.best.logSigma]}
            y={[cv.value.best.logLambda]}
            emphasis
            size={12}
            shape={1}
          />
          <Handle kind="point" at={[state.logSigma, state.logLambda]} label="(σ, λ)" onDrag={drag} />
        </Plot>
        <Plot x={xb} y={yb} height={300}>
          <Curve name="r(x)" x={X} y={truth} emphasis dashed />
          <Curve name="uLSIF r̂(x)" x={X} y={fit.r} slot={3} width={3} />
          <Rug name="q sample" values={data.xq} slot={0} />
          <Rug name="p sample" values={data.xp} slot={1} />
        </Plot>
      </div>
    </Figure>
  )
}
