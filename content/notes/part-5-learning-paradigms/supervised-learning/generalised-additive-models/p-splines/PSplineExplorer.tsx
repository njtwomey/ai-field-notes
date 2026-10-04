import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { solve } from '../../regression/nonlinear-regression/_shared/splines'
import {
  crossProduct,
  designMatrix,
  differencePenalty,
  penalisedFit,
  times,
  uniformKnots,
} from '../_shared/core-smoothing'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

type Order = '1' | '2' | '3'
const ORDER_OPTIONS = [
  { value: '1' as const, label: 'd = 1' },
  { value: '2' as const, label: 'd = 2' },
  { value: '3' as const, label: 'd = 3' },
]
const N = 100
const GRID = toFlat(linspace(0, 1, 201))
const truth = (x: number) => Math.sin(2 * Math.PI * x) + 2 * x * x

/** Least-squares polynomial of degree deg, by the normal equations (deg ≤ 2 on [0, 1] is well conditioned). */
function polyFit(x: number[], y: number[], deg: number): number[] {
  const V = x.map((xi) => Array.from({ length: deg + 1 }, (_, j) => xi ** j))
  const VtV = V[0].map((_, j) => V[0].map((__, k) => V.reduce((acc, row) => acc + row[j] * row[k], 0)))
  const Vty = V[0].map((_, j) => V.reduce((acc, row, i) => acc + row[j] * y[i], 0))
  return solve(VtV, Vty)
}

export function PSplineExplorer() {
  const state = useFigureState({
    p: int(20, { min: 8, max: 40, step: 1, label: 'basis functions p' }),
    order: choice<Order>(ORDER_OPTIONS, '2', { label: 'difference order' }),
    logLambda: float(0, { min: -4, max: 8, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    seed: int(4, { ge: 0, label: 'seed' }),
  })
  const d = Number(state.order)

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    return { x, y: x.map((xi) => truth(xi) + 0.25 * normal(r)) }
  }, [state.seed])
  // K = p − 3 equal spans on [0, 1], knots continued beyond both ends so that every B-spline has the same shape.
  const t = useMemo(() => uniformKnots(state.p - 3, 0, 1), [state.p])
  const B = useMemo(() => designMatrix(data.x, t), [data, t])
  const BtB = useMemo(() => crossProduct(B), [B])
  const gridB = useMemo(() => designMatrix(GRID, t), [t])
  const P = useMemo(() => differencePenalty(state.p, d), [state.p, d])
  const fit = useMemo(() => penalisedFit(B, BtB, data.y, P, 10 ** state.logLambda), [B, BtB, data, P, state.logLambda])
  const poly = useMemo(() => polyFit(data.x, data.y, d - 1), [data, d])

  const curve = times(gridB, fit.coef)
  const polyCurve = GRID.map((x) => poly.reduce((acc, c, j) => acc + c * x ** j, 0))
  const gap = Math.max(...curve.map((v, i) => Math.abs(v - polyCurve[i])))
  const fitted = times(B, fit.coef)
  const moment = (k: number) => data.x.reduce((acc, x, i) => acc + x ** k * (fitted[i] - data.y[i]), 0)
  // Each coefficient sits at its B-spline's peak, the middle interior knot.
  const centres = fit.coef.map((_, k) => t[k + 2])

  const series = [
    { name: 'data', x: data.x, y: data.y, muted: true },
    { name: `least-squares polynomial of degree ${d - 1}`, x: GRID, y: polyCurve, slot: 1, dashed: true },
    { name: 'P-spline fit', x: GRID, y: curve, slot: 0 },
    { name: 'coefficients β_k', x: centres, y: fit.coef, emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [-0.25, 1.25] })
  const yAxis = useAxis({ label: 'y', range: [-1.5, 3.5] })
  return (
    <Figure
      title="A P-spline and its polynomial limit"
      state={state}
      caption={
        <>
          One hundred points from sin(2πx) + 2x² plus noise, fitted with p cubic B-splines on equally spaced knots and a
          d-th order difference penalty on their coefficients (dark markers, placed at each B-spline&apos;s peak). As λ
          grows the coefficients are forced onto a polynomial of degree d − 1 in k, and the fit tends to the dashed
          least-squares polynomial of degree d − 1. The sums Σ xᵏ(ŷ − y) are zero for every k below d, whatever λ and p.
        </>
      }

      readouts={
        <>
          <Readout label="edf" value={formatNumber(fit.edf)} />
          <Readout label="max |fit − polynomial|" value={formatNumber(gap)} />
          <Readout
            label="Σ xᵏ(ŷ − y), k = 0, 1, 2"
            value={[0, 1, 2].map((k) => formatNumber(Math.abs(moment(k)) < 1e-6 ? 0 : moment(k))).join(', ')}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Points {...series[3]} />
      </Plot>
    </Figure>
  )
}
