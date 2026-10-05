import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 32
const T = 60

const matVec = (m: number[][], v: number[]) => m.map((row) => row.reduce((s, a, j) => s + a * v[j], 0))
const vecMat = (v: number[], m: number[][]) => m[0].map((_, j) => v.reduce((s, vi, i) => s + vi * m[i][j], 0))

/** A random recurrent matrix rescaled to spectral radius 1, a fixed input sequence and a unit adjoint. */
const BASE = (() => {
  const r = stream(7)
  const g = Array.from({ length: N }, () => Array.from({ length: N }, () => normal(r)))
  // Gelfand's formula: the mean log growth of a generic vector under repeated multiplication is log ρ(G).
  let u = Array.from({ length: N }, () => normal(r))
  let logGrowth = 0
  const steps = 400
  for (let s = 0; s < steps; s++) {
    u = matVec(g, u)
    const norm = Math.hypot(...u)
    logGrowth += Math.log(norm)
    u = u.map((ui) => ui / norm)
  }
  const radius = Math.exp(logGrowth / steps)
  const w = g.map((row) => row.map((a) => a / radius))
  const x = Array.from({ length: T }, () => Array.from({ length: N }, () => 0.5 * normal(r)))
  const v = Array.from({ length: N }, () => normal(r))
  const norm = Math.hypot(...v)
  return { w, x, v: v.map((vi) => vi / norm) }
})()

/**
 * Backpropagate a unit adjoint from h_T through k steps: v ← v · diag(φ'(a_t)) · W. Returns ‖v‖ after each step,
 * for tanh and for a linear recurrence with the same weights.
 */
function adjointNorms(rho: number) {
  const w = BASE.w.map((row) => row.map((a) => rho * a))
  // Forward pass with tanh, storing the derivative 1 − h² at each step.
  let h = new Array<number>(N).fill(0)
  const deriv: number[][] = []
  for (let t = 0; t < T; t++) {
    const a = matVec(w, h).map((ai, i) => ai + BASE.x[t][i])
    h = a.map(Math.tanh)
    deriv.push(h.map((hi) => 1 - hi * hi))
  }
  let vTanh = BASE.v
  let vLin = BASE.v
  const tanhNorms = [1]
  const linNorms = [1]
  for (let k = 1; k < T; k++) {
    const d = deriv[T - k]
    vTanh = vecMat(
      vTanh.map((vi, i) => vi * d[i]),
      w,
    )
    vLin = vecMat(vLin, w)
    tanhNorms.push(Math.hypot(...vTanh))
    linNorms.push(Math.hypot(...vLin))
  }
  return { tanhNorms, linNorms }
}

const FLOOR = 1e-12
const CEIL = 1e20
const clip = (v: number) => Math.min(Math.max(v, FLOOR), CEIL)
const LAGS = Array.from({ length: T }, (_, k) => k)

/** Norm of the backpropagated gradient against the number of time steps it travels. */
export function GradientDecay() {
  const state = useFigureState({
    rho: float(1, { min: 0.5, max: 2, step: 0.05, label: 'spectral radius ρ' }),
  })
  const { tanhNorms, linNorms } = useMemo(() => adjointNorms(state.rho), [state.rho])

  const series = [
    { name: 'tanh RNN', x: LAGS, y: tanhNorms.map(clip), slot: 0 },
    { name: 'linear RNN', x: LAGS, y: linNorms.map(clip), slot: 1 },
    {
      name: 'ρᵏ',
      x: LAGS,
      y: LAGS.map((k) => clip(state.rho ** k)),
      muted: true,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'steps back in time k', hold: 'union' })
  const yAxis = useAxis({ label: 'gradient norm', hold: 'union', log: true })
  return (
    <Figure
      title="Gradients through time"
      state={state}
      caption="A 32-unit RNN with random recurrent weights scaled to spectral radius ρ runs for 60 steps. A unit gradient at the last step is propagated backwards with the Jacobians diag(φ′(aₜ)) W. Its norm after k steps is the size of the signal that reaches a state k steps in the past. The linear network follows ρᵏ up to a constant: it vanishes for ρ < 1 and explodes for ρ > 1. The tanh network multiplies by φ′ ≤ 1 at every step, so its gradient is usually smaller than the linear one. For large ρ its units saturate and φ′ shrinks, so the gradient grows far more slowly than ρᵏ, but it can still grow."

      readouts={
        <>
          <Readout label="‖gradient‖ after 30 steps, tanh" value={formatNumber(tanhNorms[30])} />
          <Readout label="linear" value={formatNumber(linNorms[30])} />
          <Readout label="ρ³⁰" value={formatNumber(state.rho ** 30)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
