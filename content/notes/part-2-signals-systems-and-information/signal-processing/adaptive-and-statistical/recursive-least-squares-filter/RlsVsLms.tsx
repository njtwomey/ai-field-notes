import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

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
  const state = useFigureState({
    lambda: float(0.99, { min: 0.9, max: 1, step: 0.005, label: 'forgetting factor λ' }),
    corr: slider(0, 0.95, 0.9, { step: 0.05, label: 'input correlation a' }),
  })

  const r = useMemo(() => {
    const a = state.corr
    const rls = new Array(STEPS).fill(0)
    const nlms = new Array(STEPS).fill(0)
    for (let run = 0; run < RUNS; run++) {
      const g = stream(77 + run)
      const wR = new Array(M).fill(0)
      const wL = new Array(M).fill(0)
      // P = δ⁻¹ I initialises the inverse correlation matrix; large, since nothing is known yet.
      let P: number[][] = Array.from({ length: M }, (_, i) => Array.from({ length: M }, (_, j) => (i === j ? 100 : 0)))
      const buf = new Array(M).fill(0)
      let u = normal(g)
      for (let n = 0; n < STEPS; n++) {
        u = a * u + Math.sqrt(1 - a * a) * normal(g)
        buf.unshift(u)
        buf.pop()
        const h = n < CHANGE ? UNKNOWN : UNKNOWN.map((v) => -v)
        const d = h.reduce((s, hk, k) => s + hk * buf[k], 0) + Math.sqrt(NOISE) * normal(g)
        // RLS: k = P x / (λ + xᵀ P x), e = d − wᵀx, w += k e, P = (P − k xᵀ P) / λ.
        const Px = P.map((row) => row.reduce((s, p, j) => s + p * buf[j], 0))
        const denom = state.lambda + buf.reduce((s, x, i) => s + x * Px[i], 0)
        const k = Px.map((v) => v / denom)
        const eR = d - wR.reduce((s, w, i) => s + w * buf[i], 0)
        for (let i = 0; i < M; i++) wR[i] += k[i] * eR
        P = P.map((row, i) => row.map((p, j) => (p - k[i] * Px[j]) / state.lambda))
        // NLMS with μ̃ = 0.5 for comparison.
        const eL = d - wL.reduce((s, w, i) => s + w * buf[i], 0)
        const power = 1e-6 + buf.reduce((s, x) => s + x * x, 0)
        for (let i = 0; i < M; i++) wL[i] += (0.5 / power) * eL * buf[i]
        rls[n] += (eR * eR) / RUNS
        nlms[n] += (eL * eL) / RUNS
      }
    }
    return { rls, nlms }
  }, [state.lambda, state.corr])

  const t = Array.from({ length: STEPS }, (_, n) => n)
  const toDb = (v: number) => Math.max(-40, 10 * Math.log10(Math.max(v, 1e-12)))
  const series = [
    { name: 'NLMS (μ̃ = 0.5)', x: t, y: r.nlms.map(toDb), slot: 0 },
    { name: `RLS (λ = ${state.lambda})`, x: t, y: r.rls.map(toDb), slot: 1 },
    { name: 'noise floor', x: [0, STEPS], y: [toDb(NOISE), toDb(NOISE)], dashed: true, slot: 2 },
  ] as const
  const memory = state.lambda < 1 ? 1 / (1 - state.lambda) : Infinity

  const xAxis = useAxis({ label: 'iteration n', hold: 'union' })
  const yAxis = useAxis({ label: 'mean-squared error (dB)', range: [-40, 10] })
  return (
    <Figure
      title="RLS against NLMS on correlated input"
      state={state}
      caption="Both filters identify an 8-tap system from AR(1) input with correlation a; at n = 400 the system flips sign. RLS whitens the input through its inverse-correlation estimate P, so its convergence does not depend on the eigenvalue spread: it reaches the floor in a few tens of samples where NLMS takes hundreds. The forgetting factor λ sets RLS's memory, about 1/(1 − λ) samples: λ = 1 never forgets and cannot track the change; smaller λ tracks faster but leaves more excess error."

      readouts={
        <>
          <Readout
            label="effective memory 1/(1 − λ)"
            value={Number.isFinite(memory) ? `${formatNumber(memory)} samples` : '∞'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
