import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'
import { normalPdf } from 'aifn/numerics/special'

// Shimodaira's (2000) example: y = −x + x³ + noise, training inputs from N(0.5, 0.5²), a linear model.
const N = 100
const SRC_MEAN = 0.5
const SRC_SD = 0.5
const TGT_SD = 0.3
const NOISE_SD = 0.3
const truth = (x: number) => -x + x ** 3
const GRID = toFlat(linspace(-1.5, 2, 141))

const DATA = (() => {
  const r = stream(5)
  const x = Array.from({ length: N }, () => SRC_MEAN + SRC_SD * normal(r))
  return { x, y: x.map((v) => truth(v) + NOISE_SD * normal(r)) }
})()

/** Weighted least squares for y ≈ a + b x. */
function wls(x: number[], y: number[], w: number[]) {
  let sw = 0
  let sx = 0
  let sy = 0
  let sxx = 0
  let sxy = 0
  x.forEach((xi, i) => {
    sw += w[i]
    sx += w[i] * xi
    sy += w[i] * y[i]
    sxx += w[i] * xi * xi
    sxy += w[i] * xi * y[i]
  })
  const b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
  return { a: (sy - b * sx) / sw, b }
}

/** Expected squared error of a line under the target input distribution, including the noise. */
function targetMse(fit: { a: number; b: number }, mean: number) {
  const xs = toFlat(linspace(mean - 5 * TGT_SD, mean + 5 * TGT_SD, 400))
  const dx = xs[1] - xs[0]
  let total = 0
  for (const x of xs) total += (truth(x) - fit.a - fit.b * x) ** 2 * (normalPdf((x - mean) / TGT_SD) / TGT_SD) * dx
  return total + NOISE_SD ** 2
}

export function CovariateShift() {
  const state = useFigureState({
    mean: float(0, { min: -0.5, max: 1, step: 0.05, label: 'target input mean' }),
    lambda: float(1, { min: 0, max: 1, step: 0.05, label: 'weight exponent λ' }),
  })

  const result = useMemo(() => {
    // Importance weights p_t(x)/p_s(x), flattened by the exponent λ (λ = 0: ordinary least squares).
    const w = DATA.x.map((x) => {
      const ratio = normalPdf((x - state.mean) / TGT_SD) / TGT_SD / (normalPdf((x - SRC_MEAN) / SRC_SD) / SRC_SD)
      return ratio ** state.lambda
    })
    const plain = wls(DATA.x, DATA.y, new Array(N).fill(1))
    const weighted = wls(DATA.x, DATA.y, w)
    const sw = w.reduce((a, b) => a + b, 0)
    const ess = (sw * sw) / w.reduce((a, b) => a + b * b, 0)
    const series = [
      { name: 'training data (source)', x: DATA.x, y: DATA.y, muted: true },
      { name: 'true function', x: GRID, y: GRID.map(truth), emphasis: true },
      { name: 'least squares', x: GRID, y: GRID.map((x) => plain.a + plain.b * x), slot: 0 },
      { name: 'importance-weighted', x: GRID, y: GRID.map((x) => weighted.a + weighted.b * x), slot: 1 },
      {
        name: 'target input density (scaled)',
        x: GRID,
        y: GRID.map((x) => -1.5 + (normalPdf((x - state.mean) / TGT_SD) / TGT_SD) * 0.8),
        slot: 2,
      },
    ] as const
    return { series, plain: targetMse(plain, state.mean), weighted: targetMse(weighted, state.mean), ess }
  }, [state.mean, state.lambda])

  const xAxis = useAxis({ label: 'x', range: [-1.5, 2] })
  const yAxis = useAxis({ label: 'y', range: [-1.5, 4] })
  return (
    <Figure
      title="A linear model under covariate shift"
      purpose="Move the target input distribution and change the weight exponent to compare ordinary and importance-weighted least squares."
      state={state}
      caption="The true function is cubic and the model is a line, so the best line depends on where the inputs fall. Ordinary least squares fits the training inputs; weighting each training point by p_t(x)/p_s(x) fits the target inputs, shown as the shaded density. The exponent λ trades the two: λ = 1 removes the bias but leaves few effective samples when the distributions differ a lot."

      readouts={
        <>
          <Readout label="target MSE, least squares" value={formatNumber(result.plain)} />
          <Readout label="target MSE, weighted" value={formatNumber(result.weighted)} />
          <Readout label="effective sample size" value={`${formatNumber(result.ess)} of ${N}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Points {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Curve {...result.series[2]} />
        <Curve {...result.series[3]} />
        <Area {...result.series[4]} />
      </Plot>
    </Figure>
  )
}
