import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * Two equally likely classes, p(x | A) = N(−1, 1) and p(x | B) = N(1, 1). Classifier-free guidance with weight w
 * targets p̃(x | A) ∝ p(x | A)^{1+w} p(x)^{−w} = p(x | A) p(A | x)^w (up to a constant), computed here on a grid.
 */
const X = toFlat(linspace(-5, 5, 401))
const DX = X[1] - X[0]
const normal = (x: number, m: number) => Math.exp(-((x - m) ** 2) / 2) / Math.sqrt(2 * Math.PI)
const P_A = X.map((x) => normal(x, -1))
const P_B = X.map((x) => normal(x, 1))
const P = X.map((_, i) => 0.5 * P_A[i] + 0.5 * P_B[i])
const POST_A = X.map((_, i) => (0.5 * P_A[i]) / P[i])

export function GuidedDensity() {
  const state = useFigureState({
    w: float(2, { min: 0, max: 8, step: 0.1, label: 'guidance weight w' }),
  })

  const view = useMemo(() => {
    const unnorm = X.map((_, i) => Math.exp((1 + state.w) * Math.log(P_A[i]) - state.w * Math.log(P[i])))
    const z = unnorm.reduce((s, v) => s + v, 0) * DX
    const guided = unnorm.map((v) => v / z)
    const mean = guided.reduce((s, p, i) => s + p * X[i], 0) * DX
    const sd = Math.sqrt(guided.reduce((s, p, i) => s + p * (X[i] - mean) ** 2, 0) * DX)
    const confidence = guided.reduce((s, p, i) => s + p * POST_A[i], 0) * DX
    const series = [
      { name: 'p(x), both classes', x: X, y: P, muted: true },
      { name: 'p(x | A), unguided', x: X, y: P_A, slot: 0, dashed: true },
      { name: 'p(x | B)', x: X, y: P_B, slot: 1, dashed: true },
      { name: 'guided target for A', x: X, y: guided, slot: 0 },
    ] as const
    return { series, mean, sd, confidence }
  }, [state.w])

  const xAxis = useAxis({ label: 'x', range: [-5, 5] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="What guidance does to a class-conditional distribution"
      state={state}
      caption="Class A is N(−1, 1) and class B is N(1, 1). Guidance with weight w targets p(x | A) p(A | x)^w, renormalised. At w = 0 this is the plain conditional. Larger w moves mass away from the region class B also occupies and narrows the distribution: samples become more typical of A and less diverse."

      readouts={
        <>
          <Readout label="mean" value={formatNumber(view.mean)} />
          <Readout label="standard deviation" value={formatNumber(view.sd)} />
          <Readout label="mean classifier confidence p(A | x)" value={formatNumber(view.confidence)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...view.series[0]} />
        <Curve {...view.series[1]} />
        <Curve {...view.series[2]} />
        <Area {...view.series[3]} />
      </Plot>
    </Figure>
  )
}
