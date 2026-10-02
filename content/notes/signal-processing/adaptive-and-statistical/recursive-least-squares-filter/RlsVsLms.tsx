import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

const M = 8
const STEPS = 800
const RUNS = 20
const NOISE = 0.01
const UNKNOWN = [0.8, -0.5, 0.35, 0.2, -0.15, 0.1, 0.05, -0.02]
/** The unknown system flips sign halfway, to show tracking. */
const CHANGE = 400

/**
 * RLS and NLMS identifying the same system from strongly correlated input, averaged over runs. The system changes at
 * n = 400, so the forgetting factor's effect on tracking is visible as well as the convergence speed.
 */
export function RlsVsLms() {
  const lambda = useParam(0.99, { min: 0.9, max: 1, step: 0.005 })
  const corr = useParam(0.9, { min: 0, max: 0.95, step: 0.05 })

  const r = useMemo(() => {
    const a = corr.value
    const rls = new Array(STEPS).fill(0)
    const nlms = new Array(STEPS).fill(0)
    for (let run = 0; run < RUNS; run++) {
      const g = rng(77 + run)
      const wR = new Array(M).fill(0)
      const wL = new Array(M).fill(0)
      // P = δ⁻¹ I initialises the inverse correlation matrix; large, since nothing is known yet.
      let P: number[][] = Array.from({ length: M }, (_, i) => Array.from({ length: M }, (_, j) => (i === j ? 100 : 0)))
      const buf = new Array(M).fill(0)
      let u = g.normal()
      for (let n = 0; n < STEPS; n++) {
        u = a * u + Math.sqrt(1 - a * a) * g.normal()
        buf.unshift(u)
        buf.pop()
        const h = n < CHANGE ? UNKNOWN : UNKNOWN.map((v) => -v)
        const d = h.reduce((s, hk, k) => s + hk * buf[k], 0) + Math.sqrt(NOISE) * g.normal()
        // RLS: k = P x / (λ + xᵀ P x), e = d − wᵀx, w += k e, P = (P − k xᵀ P) / λ.
        const Px = P.map((row) => row.reduce((s, p, j) => s + p * buf[j], 0))
        const denom = lambda.value + buf.reduce((s, x, i) => s + x * Px[i], 0)
        const k = Px.map((v) => v / denom)
        const eR = d - wR.reduce((s, w, i) => s + w * buf[i], 0)
        for (let i = 0; i < M; i++) wR[i] += k[i] * eR
        P = P.map((row, i) => row.map((p, j) => (p - k[i] * Px[j]) / lambda.value))
        // NLMS with μ̃ = 0.5 for comparison.
        const eL = d - wL.reduce((s, w, i) => s + w * buf[i], 0)
        const power = 1e-6 + buf.reduce((s, x) => s + x * x, 0)
        for (let i = 0; i < M; i++) wL[i] += (0.5 / power) * eL * buf[i]
        rls[n] += (eR * eR) / RUNS
        nlms[n] += (eL * eL) / RUNS
      }
    }
    return { rls, nlms }
  }, [lambda.value, corr.value])

  const t = Array.from({ length: STEPS }, (_, n) => n)
  const toDb = (v: number) => Math.max(-40, 10 * Math.log10(Math.max(v, 1e-12)))
  const series: XYSeries[] = [
    { name: 'NLMS (μ̃ = 0.5)', type: 'line', x: t, y: r.nlms.map(toDb), slot: 0 },
    { name: `RLS (λ = ${lambda.value})`, type: 'line', x: t, y: r.rls.map(toDb), slot: 1 },
    { name: 'noise floor', type: 'line', x: [0, STEPS], y: [toDb(NOISE), toDb(NOISE)], dashed: true, slot: 2 },
  ]
  const memory = lambda.value < 1 ? 1 / (1 - lambda.value) : Infinity

  return (
    <Interactive
      title="RLS against NLMS on correlated input"
      caption="Both filters identify an 8-tap system from AR(1) input with correlation a; at n = 400 the system flips sign. RLS whitens the input through its inverse-correlation estimate P, so its convergence does not depend on the eigenvalue spread: it reaches the floor in a few tens of samples where NLMS takes hundreds. The forgetting factor λ sets RLS's memory, about 1/(1 − λ) samples: λ = 1 never forgets and cannot track the change; smaller λ tracks faster but leaves more excess error."
      controls={
        <>
          <ParamSlider label="forgetting factor λ" param={lambda} />
          <ParamSlider label="input correlation a" param={corr} />
        </>
      }
      readout={
        <>
          <Readout
            label="effective memory 1/(1 − λ)"
            value={Number.isFinite(memory) ? `${formatNumber(memory)} samples` : '∞'}
          />
        </>
      }
    >
      <XYChart series={series} xLabel="iteration n" yLabel="mean-squared error (dB)" yRange={[-40, 10]} height={300} />
    </Interactive>
  )
}
