import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { cholesky2, eigSym } from '@/lib/math/mat2'

const N = 400
const R = 4.5

/** Samples x = Lz from N(0, Σ), where Σ = LLᵀ is built from two standard deviations and a correlation. */
export function GaussianSampler() {
  const s1 = useParam(1.5, { min: 0.2, max: 2, step: 0.05 })
  const s2 = useParam(0.8, { min: 0.2, max: 2, step: 0.05 })
  const rho = useParam(0.6, { min: -0.95, max: 0.95, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  // Standard normal draws, fixed per seed so that moving a slider transforms the same points.
  const z = useMemo(() => {
    const g = rng(seed.value)
    return Array.from({ length: N }, () => [g.normal(), g.normal()] as [number, number])
  }, [seed.value])

  const r = useMemo(() => {
    const p = s1.value ** 2
    const q = rho.value * s1.value * s2.value
    const rr = s2.value ** 2
    const L = cholesky2(p, q, rr)!
    const x = z.map(([a, b]) => [L[0][0] * a, L[1][0] * a + L[1][1] * b] as [number, number])
    // Sample covariance, to compare with Σ.
    const m = [x.reduce((s, v) => s + v[0], 0) / N, x.reduce((s, v) => s + v[1], 0) / N]
    const c = (i: number, j: number) => x.reduce((s, v) => s + (v[i] - m[i]) * (v[j] - m[j]), 0) / (N - 1)
    // The 2σ ellipse: the image of the circle of radius 2 under L, drawn along Σ's eigenvectors.
    const { values, vectors } = eigSym(p, q, rr)
    const t = linspace(0, 2 * Math.PI, 97)
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
    const series: XYSeries[] = [
      { name: 'samples Lz', type: 'scatter', x: x.map((v) => v[0]), y: x.map((v) => v[1]), slot: 0 },
      { name: '2σ contour of Σ', type: 'line', ...ellipse, slot: 1 },
    ]
    return { L, series, sample: [c(0, 0), c(0, 1), c(1, 1)], sigma: [p, q, rr] }
  }, [z, s1.value, s2.value, rho.value])

  const { L } = r
  return (
    <Interactive
      title="Correlated Gaussian samples from a Cholesky factor"
      caption="Each point starts as a pair of independent standard normal draws z. Multiplying by the Cholesky factor L of Σ gives x = Lz, whose covariance is LLᵀ = Σ. The first coordinate uses only z₁; the second mixes in z₁ through the off-diagonal entry of L, which creates the correlation. The ellipse is the contour two standard deviations out."
      controls={
        <>
          <ParamSlider label="σ₁" param={s1} />
          <ParamSlider label="σ₂" param={s2} />
          <ParamSlider label="correlation ρ" param={rho} />
          <ParamSlider label="seed" param={seed} />
        </>
      }
      readout={
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
        <XYChart equalAspect xRange={[-R, R]} yRange={[-R, R]} xLabel="x₁" yLabel="x₂" series={r.series} />
      </div>
    </Interactive>
  )
}
