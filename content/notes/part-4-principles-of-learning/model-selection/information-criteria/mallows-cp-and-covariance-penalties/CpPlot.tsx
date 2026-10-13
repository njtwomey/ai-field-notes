import { useMemo } from 'react'
import { unitGrid } from 'aifn-methods/theory'
import { linearRegression } from 'aifn-methods/learning/linear'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { mallowsCp } from 'aifn-compute/learning/selection'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const P_MAX = 8
const PS = Array.from({ length: P_MAX }, (_, i) => i + 1)
const LOW = -2
const HIGH = 20
const clip = (v: number) => (Number.isFinite(v) ? Math.max(LOW, Math.min(HIGH, v)) : HIGH)
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—')
const argmin = (v: readonly number[]) => v.reduce((best, x, i) => (x < v[best] ? i : best), 0)

/** Least-squares polynomial with p coefficients (degree p − 1), fitted at the design and evaluated there. */
function fittedValues(p: number, x: Float64Array, y: Float64Array) {
  const model = pipeline(
    polynomialFeatures({ degree: p - 1, includeBias: true }),
    linearRegression({ intercept: false }),
  )
  const design = fromData(x, [x.length, 1])
  return toFlat(model.fit(dataset(design, fromData(y))).decide(design))
}

/**
 * The Cp plot for polynomial regressions with p = 1, …, 8 coefficients on a fixed design: Mallows' Cp of one sample
 * against p, beside the truth it estimates, the scaled total squared error J_p = p + B_p / σ², and the line Cp = p.
 */
export function CpPlot() {
  const state = useFigureState({
    cubic: slider(0, 2, 0.6, { step: 0.05, label: 'cubic coefficient' }),
    noise: slider(0.2, 2, 1, { step: 0.05, label: 'noise σ' }),
    n: int(40, { min: 15, max: 400, scale: 'log10', suggestions: [20, 40, 100, 400], label: 'observations n' }),
    seed: int(0, { ge: 0, label: 'seed' }),
  })

  const result = useMemo(() => {
    const x = unitGrid(state.n)
    const f = Float64Array.from(x, (v) => 1 + 2 * v - 1.5 * v * v + state.cubic * v ** 3)
    const eps = toFlat(normals(stream(`cp-plot/${state.seed}`), state.n, 0, state.noise))
    const y = Float64Array.from(f, (v, i) => v + eps[i])
    const rss = PS.map((p) => state.n * meanSquaredError(y, fittedValues(p, x, y)))
    const bias = PS.map((p) => state.n * meanSquaredError(f, fittedValues(p, x, f)))
    const sigma2 = rss[P_MAX - 1] / (state.n - P_MAX)
    const cp = PS.map((p, i) => mallowsCp({ rss: rss[i], df: p, sigma2, n: state.n }).classical)
    const truth = PS.map((p, i) => p + bias[i] / state.noise ** 2)
    return { cp, truth, sigma2 }
  }, [state.cubic, state.noise, state.n, state.seed])

  const pAxis = useAxis({ label: 'coefficients p', range: [0.5, P_MAX + 0.5] })
  const cAxis = useAxis({ label: 'Cp (Mallows’ scale)', range: [LOW, HIGH] })
  return (
    <Figure
      title="The Cp plot"
      state={state}
      caption={`n inputs equally spaced on [−1, 1], f(x) = 1 + 2x − 1.5x² + c·x³ with c the cubic coefficient, and Gaussian noise σ. Candidates are least-squares polynomials with p = 1 to ${P_MAX} coefficients. Points: Mallows' Cp of one sample, with σ² estimated from the largest model. Dashed: the quantity Cp estimates, the scaled total squared error J(p) = p + B(p)/σ², with B(p) the squared bias of the p-coefficient fit, computed from the noise-free f. Muted: the line Cp = p, where unbiased candidates sit on average. Cp scatters about J(p) by several units at n = 40, and its points move together because they share one noise draw: change the seed to see the spread. With a small cubic coefficient, p = 3 has the smaller J(p) although the true model has p = 4: dropping a weak term costs less bias than it saves variance. Raise the cubic coefficient or n to see the minimum of J(p) move to p = 4. Values are clipped to [${LOW}, ${HIGH}].`}
      readouts={
        <>
          <Readout label="σ̂² from the largest model" value={fmt(result.sigma2)} />
          <Readout label="p chosen by Cp" value={String(PS[argmin(result.cp)])} />
          <Readout label="p best on the truth J(p)" value={String(PS[argmin(result.truth)])} />
        </>
      }
    >
      <Plot x={pAxis} y={cAxis}>
        <Curve name="Cp = p" x={[0.5, P_MAX + 0.5]} y={[0.5, P_MAX + 0.5]} muted dashed />
        <Curve name="truth J(p)" emphasis dashed x={PS} y={result.truth.map(clip)} />
        <Points name="Mallows' Cp" slot={0} x={PS} y={result.cp.map(clip)} />
      </Plot>
    </Figure>
  )
}
