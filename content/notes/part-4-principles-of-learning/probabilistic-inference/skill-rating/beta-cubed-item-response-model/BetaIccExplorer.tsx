import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { logGamma, regularisedBeta } from 'aifn/numerics/special'

const THETA = toFlat(linspace(0.005, 0.995, 100))
const P_GRID = toFlat(linspace(0.0025, 0.9975, 200))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** α and β of the response distribution Beta(α, β) at ability θ. */
const shape = (t: number, d: number, a: number) => ({
  al: Math.pow(t / d, a),
  be: Math.pow((1 - t) / (1 - d), a),
})

/** The q-quantile of Beta(α, β) by bisection on the regularised incomplete beta function. */
function betaQuantile(q: number, al: number, be: number): number {
  let lo = 0
  let hi = 1
  for (let k = 0; k < 40; k++) {
    const mid = (lo + hi) / 2
    if (regularisedBeta(al, be, mid) < q) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function shapeName(a: number): string {
  if (Math.abs(a) < 0.025) return 'flat: responses ignore ability'
  const dir = a > 0 ? 'increasing' : 'decreasing'
  const m = Math.abs(a)
  if (Math.abs(m - 1) < 0.025) return `${dir}, no inflection`
  return m > 1 ? `${dir} sigmoid` : `${dir} anti-sigmoid`
}

export function BetaIccExplorer() {
  const state = useFigureState({
    delta: float(0.5, { min: 0.05, max: 0.95, step: 0.01, label: 'difficulty δ' }),
    a: float(2, { min: -3, max: 4, step: 0.05, label: 'discrimination a' }),
    theta0: float(0.7, { min: 0.02, max: 0.98, step: 0.01, label: 'ability θ' }),
  })

  const curves = useMemo(() => {
    const mean = THETA.map((t) => {
      const { al, be } = shape(t, state.delta, state.a)
      return al / (al + be)
    })
    const q10 = THETA.map((t) => {
      const { al, be } = shape(t, state.delta, state.a)
      return betaQuantile(0.1, al, be)
    })
    const q90 = THETA.map((t) => {
      const { al, be } = shape(t, state.delta, state.a)
      return betaQuantile(0.9, al, be)
    })
    return [
      { name: 'mean response (ICC)', x: THETA, y: mean, emphasis: true },
      { name: '10% quantile', x: THETA, y: q10, dashed: true, slot: 0 },
      { name: '90% quantile', x: THETA, y: q90, dashed: true, slot: 0 },
    ] as const
  }, [state.delta, state.a])

  const { al, be } = shape(state.theta0, state.delta, state.a)
  const density = useMemo(() => {
    const logB = logGamma(al) + logGamma(be) - logGamma(al + be)
    const y = P_GRID.map((p) => Math.exp((al - 1) * Math.log(p) + (be - 1) * Math.log(1 - p) - logB))
    return [{ name: `response density at θ = ${formatNumber(state.theta0)}`, x: P_GRID, y, slot: 1 }] as const
  }, [al, be, state.theta0])
  const yTop = Math.min(8, Math.max(...density[0].y) * 1.05)

  const xAxis = useAxis({ label: 'ability θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'response p', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'response p', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'density', range: [0, yTop] })
  return (
    <Figure
      title="β³ item characteristic curves and response distributions"
      state={state}
      caption="The top chart shows the mean response of one item against ability (solid) and the 10% and 90% quantiles of the Beta response distribution (dashed). Drag the line marked δ to move the difficulty and the line marked θ to pick an ability; the bottom chart shows the response density at that ability. Discrimination above 1 gives a sigmoid, between 0 and 1 an anti-sigmoid that is steep near θ = 0 and θ = 1, and a negative value reverses the curve. At θ = δ the response is uniform on (0, 1) whatever the discrimination."

      readouts={
        <>
          <Readout label="shape:" value={shapeName(state.a)} />
          <Readout label="α" value={formatNumber(al)} />
          <Readout label="β" value={formatNumber(be)} />
          <Readout label="mean response" value={formatNumber(al / (al + be))} />
          <Readout label="slope at θ = δ" value={formatNumber(state.a / (4 * state.delta * (1 - state.delta)))} />
        </>
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
          <Curve {...curves[2]} />
          <Handle
            kind="x"
            at={state.delta}
            label="δ"
            onDrag={(x) => state.set('delta', Math.round(clamp(x, 0.05, 0.95) * 100) / 100)}
          />
          <Handle
            kind="x"
            at={state.theta0}
            label="θ"
            onDrag={(x) => state.set('theta0', Math.round(clamp(x, 0.02, 0.98) * 100) / 100)}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          <Area {...density[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
