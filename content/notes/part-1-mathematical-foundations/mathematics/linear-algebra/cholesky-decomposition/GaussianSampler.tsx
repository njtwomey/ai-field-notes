import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'
import { cholesky2, eigh2 } from 'aifn-compute/numerics/linalg'

const N = 400
const R = 4.5

/** Samples x = Lz from N(0, Σ), where Σ = LLᵀ is built from two standard deviations and a correlation. */
export function GaussianSampler() {
  const state = useFigureState({
    s1: float(1.5, { min: 0.2, max: 2, step: 0.05, label: 'σ₁' }),
    s2: float(0.8, { min: 0.2, max: 2, step: 0.05, label: 'σ₂' }),
    rho: slider(-0.95, 0.95, 0.6, { step: 0.05, label: 'correlation ρ' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })

  // Standard normal draws, fixed per seed so that moving a slider transforms the same points.
  const z = useMemo(() => {
    const g = stream(state.seed)
    return Array.from({ length: N }, () => [normal(g), normal(g)] as [number, number])
  }, [state.seed])

  const r = useMemo(() => {
    const p = state.s1 ** 2
    const q = state.rho * state.s1 * state.s2
    const rr = state.s2 ** 2
    const L = cholesky2([
      [p, q],
      [q, rr],
    ])!
    const x = z.map(([a, b]) => [L[0][0] * a, L[1][0] * a + L[1][1] * b] as [number, number])
    // Sample covariance, to compare with Σ.
    const m = [x.reduce((s, v) => s + v[0], 0) / N, x.reduce((s, v) => s + v[1], 0) / N]
    const c = (i: number, j: number) => x.reduce((s, v) => s + (v[i] - m[i]) * (v[j] - m[j]), 0) / (N - 1)
    // The 2σ ellipse: the image of the circle of radius 2 under L, drawn along Σ's eigenvectors.
    const { values, vectors } = eigh2([
      [p, q],
      [q, rr],
    ])
    const t = toFlat(linspace(0, 2 * Math.PI, 97))
    const ellipse = {
      x: t.map(
        (s) =>
          2 * (Math.sqrt(values[0]) * Math.cos(s) * vectors[0][0] + Math.sqrt(values[1]) * Math.sin(s) * vectors[1][0]),
      ),
      y: t.map(
        (s) =>
          2 * (Math.sqrt(values[0]) * Math.cos(s) * vectors[0][1] + Math.sqrt(values[1]) * Math.sin(s) * vectors[1][1]),
      ),
    }
    const series: SeriesSpec[] = [
      { name: 'samples Lz', type: 'scatter', x: x.map((v) => v[0]), y: x.map((v) => v[1]), slot: 0 },
      { name: '2σ contour of Σ', type: 'line', ...ellipse, slot: 1 },
    ]
    return { L, series, sample: [c(0, 0), c(0, 1), c(1, 1)], sigma: [p, q, rr] }
  }, [z, state.s1, state.s2, state.rho])

  const { L } = r
  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Correlated Gaussian samples from a Cholesky factor"
      state={state}
      caption="Each point starts as a pair of independent standard normal draws z. Multiplying by the Cholesky factor L of Σ gives x = Lz, whose covariance is LLᵀ = Σ. The first coordinate uses only z₁; the second mixes in z₁ through the off-diagonal entry of L, which creates the correlation. The ellipse is the contour two standard deviations out."

      readouts={
        <>
          <Readout
            label="L"
            value={`[[${formatNumber(L[0][0])}, 0], [${formatNumber(L[1][0])}, ${formatNumber(L[1][1])}]]`}
          />
          <Readout
            label="Σ"
            value={`[[${formatNumber(r.sigma[0])}, ${formatNumber(r.sigma[1])}], [·, ${formatNumber(r.sigma[2])}]]`}
          />
          <Readout
            label="sample covariance"
            value={`[[${formatNumber(r.sample[0])}, ${formatNumber(r.sample[1])}], [·, ${formatNumber(r.sample[2])}]]`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(r.series)}
        </Plot>
      </div>
    </Figure>
  )
}
