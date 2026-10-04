import { useMemo } from 'react'
import { Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const REPS = 300
const BETA: [number, number] = [1, 1]
const RANGE: [number, number] = [-1.5, 3.5]

/**
 * Repeated samples from y = x₁ + x₂ + ε with corr(x₁, x₂) = ρ. Each point is one sample's least-squares estimate
 * (β̂₁, β̂₂). Correlated predictors stretch the cloud along β₁ + β₂ = const: the sum is well determined, the split is not.
 */
export function CollinearEstimates() {
  const state = useFigureState({
    rho: slider(-0.99, 0.99, 0.9, { step: 0.01, label: 'correlation ρ of x₁ and x₂' }),
    n: int(50, { min: 10, max: 200, step: 10, label: 'sample size n' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const b1: number[] = []
    const b2: number[] = []
    const c = Math.sqrt(1 - state.rho * state.rho)
    for (let rep = 0; rep < REPS; rep++) {
      let s11 = 0
      let s22 = 0
      let s12 = 0
      let s1y = 0
      let s2y = 0
      let m1 = 0
      let m2 = 0
      let my = 0
      const x1: number[] = []
      const x2: number[] = []
      const y: number[] = []
      for (let i = 0; i < state.n; i++) {
        const a = normal(g)
        const b = state.rho * a + c * normal(g)
        x1.push(a)
        x2.push(b)
        y.push(BETA[0] * a + BETA[1] * b + normal(g))
        m1 += a / state.n
        m2 += b / state.n
        my += y[i] / state.n
      }
      for (let i = 0; i < state.n; i++) {
        const u = x1[i] - m1
        const v = x2[i] - m2
        const w = y[i] - my
        s11 += u * u
        s22 += v * v
        s12 += u * v
        s1y += u * w
        s2y += v * w
      }
      // Solve the 2×2 normal equations for the centred predictors (the intercept is absorbed by centring).
      const det = s11 * s22 - s12 * s12
      b1.push((s22 * s1y - s12 * s2y) / det)
      b2.push((s11 * s2y - s12 * s1y) / det)
    }
    const sd = (xs: number[]) => {
      const m = xs.reduce((a, b) => a + b, 0) / xs.length
      return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1))
    }
    const sums = b1.map((v, i) => v + b2[i])
    const series = [
      { name: 'estimates (β̂₁, β̂₂)', x: b1, y: b2, slot: 0 },
      { name: 'true (β₁, β₂)', x: [BETA[0]], y: [BETA[1]], emphasis: true },
    ] as const
    return { series, sd1: sd(b1), sdSum: sd(sums), vif: 1 / (1 - state.rho * state.rho) }
  }, [state.rho, state.n, state.seed])

  const xAxis = useAxis({ label: 'β̂₁', range: RANGE })
  const yAxis = useAxis({ label: 'β̂₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="Correlated predictors make coefficients unstable"
      state={state}
      caption="Each dot is the least-squares estimate from one of 300 independent samples of size n, with true coefficients (1, 1). As the correlation ρ between the two predictors grows, the cloud stretches along the line β̂₁ + β̂₂ = 2: the data determine the sum of the coefficients well and their split poorly. The standard deviation of each coefficient grows like √VIF = 1/√(1 − ρ²)."

      readouts={
        <>
          <Readout label="VIF = 1/(1 − ρ²)" value={formatNumber(r.vif)} />
          <Readout label="sd of β̂₁" value={formatNumber(r.sd1)} />
          <Readout label="sd of β̂₁ + β̂₂" value={formatNumber(r.sdSum)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...r.series[0]} />
        <Points {...r.series[1]} />
      </Plot>
    </Figure>
  )
}
