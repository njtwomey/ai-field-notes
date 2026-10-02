import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const GRID = linspace(-5, 5, 201)
const X_RANGE: [number, number] = [-5, 5]
const S_RANGE: [number | undefined, number | undefined] = [-4, 4]
const K_RANGE: [number | undefined, number | undefined] = [0, undefined]
/** Basis centres cover a wider interval than the plot, so the edges of the plot are not starved of bumps. */
const LO = -8
const HI = 8
const MAX_M = 80
const NORMALS = (() => {
  const g = rng(21)
  return Array.from({ length: 3 }, () => Array.from({ length: MAX_M }, () => g.normal()))
})()

/**
 * Bayesian linear regression on M Gaussian bumps of width λ, with weight variance proportional to the bump spacing Δ.
 * As M grows its covariance tends to the squared exponential kernel √π λ exp(−(x − x′)²/4λ²).
 */
export function BasisToKernel() {
  const m = useParam(8, { min: 2, max: MAX_M, step: 1 })
  const lambda = useParam(0.7, { min: 0.3, max: 1.5, step: 0.05 })

  const r = useMemo(() => {
    const centres = linspace(LO, HI, m.value)
    const delta = (HI - LO) / (m.value - 1)
    const bump = (x: number, c: number) => Math.exp(-((x - c) ** 2) / (2 * lambda.value ** 2))
    const feats = GRID.map((x) => centres.map((c) => bump(x, c)))
    const draws = NORMALS.map((z) => feats.map((f) => f.reduce((s, v, i) => s + v * Math.sqrt(delta) * z[i], 0)))
    const kFinite = GRID.map((x) => centres.reduce((s, c) => s + delta * bump(x, c) * bump(0, c), 0))
    const varFinite = GRID.map((x) => centres.reduce((s, c) => s + delta * bump(x, c) ** 2, 0))
    const amp = Math.sqrt(Math.PI) * lambda.value
    const kLimit = GRID.map((x) => amp * Math.exp(-(x * x) / (4 * lambda.value ** 2)))
    const spread = Math.max(...varFinite) - Math.min(...varFinite)
    return { draws, kFinite, varFinite, kLimit, amp, spread, delta }
  }, [m.value, lambda.value])

  const samples: XYSeries[] = r.draws.map((d, i) => ({ name: `sample ${i + 1}`, type: 'line', x: GRID, y: d, slot: i }))
  const kernels: XYSeries[] = [
    { name: 'limit: SE kernel k(0, x)', type: 'line', x: GRID, y: r.kLimit, slot: 2, dashed: true },
    { name: 'M bumps: k(0, x)', type: 'line', x: GRID, y: r.kFinite, slot: 0 },
    { name: 'M bumps: variance k(x, x)', type: 'line', x: GRID, y: r.varFinite, slot: 1 },
  ]

  return (
    <Interactive
      title="From basis functions to a kernel"
      caption="Left: three prior draws of a linear model f(x) = Σ wₘ φₘ(x) on M Gaussian bumps of width λ, centred evenly on [−8, 8], with weights wₘ ~ N(0, Δ) for spacing Δ. Right: the model's covariance k(0, x) = Σ Δ φₘ(0) φₘ(x) and its variance k(x, x), against the squared exponential limit. With few, widely spaced bumps the prior is lumpy: the variance rises and falls between centres, so the model is not stationary. When the spacing falls below about λ, the covariance matches the squared exponential kernel with length-scale √2 λ, and the linear model is a Gaussian process with that kernel."
      controls={
        <>
          <ParamSlider label="basis functions M" param={m} format={(v) => String(v)} />
          <ParamSlider label="bump width λ" param={lambda} />
        </>
      }
      readout={
        <>
          <Readout label="spacing Δ / width λ" value={formatNumber(r.delta / lambda.value)} />
          <Readout label="variance ripple (max − min)" value={formatNumber(r.spread)} />
          <Readout label="limit variance √π λ" value={formatNumber(r.amp)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={samples} xLabel="x" yLabel="f(x)" xRange={X_RANGE} yRange={S_RANGE} height={320} />
        <XYChart series={kernels} xLabel="x" yLabel="covariance" xRange={X_RANGE} yRange={K_RANGE} height={320} />
      </div>
    </Interactive>
  )
}
