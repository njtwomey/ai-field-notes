import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { solve } from '../../regression/nonlinear-regression/_shared/splines'
import {
  crossProduct,
  designMatrix,
  differencePenalty,
  penalisedFit,
  times,
  uniformKnots,
} from '../_shared/core-smoothing'

type Order = '1' | '2' | '3'
const ORDER_OPTIONS = [
  { value: '1' as const, label: 'd = 1' },
  { value: '2' as const, label: 'd = 2' },
  { value: '3' as const, label: 'd = 3' },
]
const N = 100
const GRID = linspace(0, 1, 201)
const truth = (x: number) => Math.sin(2 * Math.PI * x) + 2 * x * x

/** Least-squares polynomial of degree deg, by the normal equations (deg ≤ 2 on [0, 1] is well conditioned). */
function polyFit(x: number[], y: number[], deg: number): number[] {
  const V = x.map((xi) => Array.from({ length: deg + 1 }, (_, j) => xi ** j))
  const VtV = V[0].map((_, j) => V[0].map((__, k) => V.reduce((acc, row) => acc + row[j] * row[k], 0)))
  const Vty = V[0].map((_, j) => V.reduce((acc, row, i) => acc + row[j] * y[i], 0))
  return solve(VtV, Vty)
}

export function PSplineExplorer() {
  const [p, setP] = useState(20)
  const [order, setOrder] = useState<Order>('2')
  const d = Number(order)
  const [logLambda, setLogLambda] = useState(0)
  const [seed, setSeed] = useState(4)

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    return { x, y: x.map((xi) => truth(xi) + 0.25 * r.normal()) }
  }, [seed])
  // K = p − 3 equal spans on [0, 1], knots continued beyond both ends so that every B-spline has the same shape.
  const t = useMemo(() => uniformKnots(p - 3, 0, 1), [p])
  const B = useMemo(() => designMatrix(data.x, t), [data, t])
  const BtB = useMemo(() => crossProduct(B), [B])
  const gridB = useMemo(() => designMatrix(GRID, t), [t])
  const P = useMemo(() => differencePenalty(p, d), [p, d])
  const fit = useMemo(() => penalisedFit(B, BtB, data.y, P, 10 ** logLambda), [B, BtB, data, P, logLambda])
  const poly = useMemo(() => polyFit(data.x, data.y, d - 1), [data, d])

  const curve = times(gridB, fit.coef)
  const polyCurve = GRID.map((x) => poly.reduce((acc, c, j) => acc + c * x ** j, 0))
  const gap = Math.max(...curve.map((v, i) => Math.abs(v - polyCurve[i])))
  const fitted = times(B, fit.coef)
  const moment = (k: number) => data.x.reduce((acc, x, i) => acc + x ** k * (fitted[i] - data.y[i]), 0)
  // Each coefficient sits at its B-spline's peak, the middle interior knot.
  const centres = fit.coef.map((_, k) => t[k + 2])

  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: `least-squares polynomial of degree ${d - 1}`, type: 'line', x: GRID, y: polyCurve, slot: 1, dashed: true },
    { name: 'P-spline fit', type: 'line', x: GRID, y: curve, slot: 0 },
    { name: 'coefficients β_k', type: 'scatter', x: centres, y: fit.coef, emphasis: true },
  ]

  return (
    <Interactive
      title="A P-spline and its polynomial limit"
      caption={
        <>
          One hundred points from sin(2πx) + 2x² plus noise, fitted with p cubic B-splines on equally spaced knots and a
          d-th order difference penalty on their coefficients (dark markers, placed at each B-spline&apos;s peak). As λ
          grows the coefficients are forced onto a polynomial of degree d − 1 in k, and the fit tends to the dashed
          least-squares polynomial of degree d − 1. The sums Σ xᵏ(ŷ − y) are zero for every k below d, whatever λ and p.
        </>
      }
      controls={
        <>
          <ParamSlider label="basis functions p" value={p} onChange={setP} min={8} max={40} step={1} withArrows />
          <ParamChoice label="difference order" value={order} onChange={setOrder} options={ORDER_OPTIONS} />
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-4}
            max={8}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamButton onClick={() => setSeed((v) => v + 1)}>New sample</ParamButton>
        </>
      }
      readout={
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
      <XYChart series={series} xRange={[-0.25, 1.25]} yRange={[-1.5, 3.5]} xLabel="x" yLabel="y" height={340} />
    </Interactive>
  )
}
