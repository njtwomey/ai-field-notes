import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import {
  basisSize,
  crossProduct,
  demmlerReinsch,
  derivativePenalty,
  designMatrix,
  times,
  transposeTimes,
  uniformKnots,
} from '../_shared/core-smoothing'

type Order = '1' | '2' | '3'
const ORDER_OPTIONS = [
  { value: '1' as const, label: "f'" },
  { value: '2' as const, label: "f''" },
  { value: '3' as const, label: "f'''" },
]
const PRIME = { '1': '′', '2': '″', '3': '‴' }
const GRID = linspace(0, 1, 201)

/** Greville abscissa of B-spline k: the mean of its interior knots, where it peaks for equally spaced knots. */
const greville = (t: number[], k: number, q = 3) => t.slice(k + 1, k + q + 1).reduce((a, b) => a + b, 0) / q

/** Eight cubic B-splines, their m-th derivatives and the penalty matrix S_kl = ∫ B_k^(m) B_l^(m) dx. */
export function PenaltyMatrix() {
  const [order, setOrder] = useState<Order>('2')
  const m = Number(order)
  const t = useMemo(() => uniformKnots(5, 0, 1), [])
  const p = basisSize(t)
  const k = useParam(4, { min: 1, max: p, step: 1 })
  const values = useMemo(() => designMatrix(GRID, t), [t])
  const derivs = useMemo(() => designMatrix(GRID, t, 3, m), [t, m])
  const S = useMemo(() => derivativePenalty(t, 3, m, 0, 1), [t, m])
  const scale = Math.max(...S.flat().map(Math.abs))
  const idx = Array.from({ length: p }, (_, i) => i + 1)

  const lines = (M: number[][]): XYSeries[] =>
    idx.map((j) => ({
      name: j === k.value ? `B${j}` : 'other B-splines',
      type: 'line',
      x: GRID,
      y: M.map((row) => row[j - 1]),
      ...(j === k.value ? { slot: 0 } : { muted: true }),
    }))
  const handles: Handle[] = [
    {
      kind: 'x',
      at: greville(t, k.value - 1),
      label: `B${k.value}`,
      onDrag: (x) => {
        // Pick the basis function whose peak is nearest the pointer.
        const dist = idx.map((j) => Math.abs(greville(t, j - 1) - x))
        k.set(dist.indexOf(Math.min(...dist)) + 1)
      },
    },
  ]
  const row = S[k.value - 1]
  const rank = p - m
  const rowSum = row.reduce((a, b) => a + b, 0)

  return (
    <Interactive
      title="From basis functions to the penalty matrix"
      caption={
        <>
          Eight cubic B-splines on equally spaced knots over [0, 1] (left), their m-th derivatives (centre) and the
          penalty matrix S with entries ∫ B_k^(m) B_l^(m) dx, scaled by its largest entry (right). Drag the guide on the
          left chart, or use the slider, to pick a basis function. Each row of S sums to zero because a constant has no
          derivative, and S has rank p − m.
        </>
      }
      controls={
        <>
          <ParamChoice label="penalised derivative" value={order} onChange={setOrder} options={ORDER_OPTIONS} />
          <ParamSlider label="basis function k" param={k} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label={`S_kk`} value={formatNumber(row[k.value - 1])} />
          <Readout label="row sum of S" value={formatNumber(Math.abs(rowSum) < 1e-9 * scale ? 0 : rowSum)} />
          <Readout label="rank of S" value={`${rank} of ${p}`} />
          <Readout label="null space" value={m === 1 ? 'constants' : m === 2 ? 'lines' : 'quadratics'} />
        </>
      }
    >
      <div className="grid gap-2 lg:grid-cols-3">
        <XYChart
          series={lines(values)}
          xRange={[0, 1]}
          yRange={[0, 0.7]}
          xLabel="x"
          yLabel="B_k(x)"
          handles={handles}
          height={240}
        />
        <XYChart series={lines(derivs)} xRange={[0, 1]} xLabel="x" yLabel={`B_k${PRIME[order]}(x)`} height={240} />
        <Heatmap
          x={idx}
          y={idx}
          z={S.map((r) => r.map((v) => v / scale))}
          scale="diverging"
          range={[-1, 1]}
          xLabel="l"
          yLabel="k"
          valueLabel="S_kl / max|S|"
          marker={[k.value, k.value]}
          height={240}
        />
      </div>
    </Interactive>
  )
}

const N = 80
const K = 20
const truth = (x: number) => Math.sin(2 * Math.PI * x) + 0.6 * x

/** A penalised cubic spline in the Demmler–Reinsch basis: the fit, its m-th derivative and the shrinkage factors. */
export function PenaltyShrinkage() {
  const [order, setOrder] = useState<Order>('2')
  const m = Number(order)
  const [logLambda, setLogLambda] = useState(-4)
  const [seed, setSeed] = useState(3)
  const lambda = 10 ** logLambda

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    return { x, y: x.map((xi) => truth(xi) + 0.3 * r.normal()) }
  }, [seed])
  const t = useMemo(() => uniformKnots(K, 0, 1), [])
  const B = useMemo(() => designMatrix(data.x, t), [data, t])
  const BtB = useMemo(() => crossProduct(B), [B])
  const S = useMemo(() => derivativePenalty(t, 3, m, 0, 1), [t, m])
  const dr = useMemo(() => demmlerReinsch(BtB, S), [BtB, S])
  // Coordinates of the data in the orthonormal Demmler–Reinsch basis: s = (B L⁻ᵀ U)ᵀ y.
  const s = useMemo(() => {
    const Bty = transposeTimes(B, data.y)
    return dr.toCoef[0].map((_, k) => dr.toCoef.reduce((acc, row, j) => acc + row[k] * Bty[j], 0))
  }, [B, data, dr])
  const gridB = useMemo(() => designMatrix(GRID, t), [t])
  const gridD = useMemo(() => designMatrix(GRID, t, 3, m), [t, m])

  // Null-space eigenvalues are zero up to round-off relative to the largest eigenvalue.
  const nullDim = dr.d.filter((d) => d < 1e-9 * dr.d[dr.d.length - 1]).length
  const shrink = dr.d.map((d, k) => (k < nullDim ? 1 : 1 / (1 + lambda * d)))
  const coefOf = (factors: number[]) =>
    times(
      dr.toCoef,
      s.map((v, k) => v * factors[k]),
    )
  const beta = coefOf(shrink)
  const limit = coefOf(shrink.map((_, k) => (k < nullDim ? 1 : 0)))
  const fitted = times(B, beta)
  const rss = data.y.reduce((acc, y, i) => acc + (y - fitted[i]) ** 2, 0)
  const penalty = dr.d.reduce((acc, d, k) => acc + (k < nullDim ? 0 : d) * (s[k] * shrink[k]) ** 2, 0)
  const edf = shrink.reduce((a, b) => a + b, 0)

  const fitSeries: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: 'λ → ∞ limit', type: 'line', x: GRID, y: times(gridB, limit), slot: 1, dashed: true },
    { name: 'fit', type: 'line', x: GRID, y: times(gridB, beta), slot: 0 },
  ]
  const derivSeries: XYSeries[] = [{ name: `f${PRIME[order]}`, type: 'line', x: GRID, y: times(gridD, beta), slot: 0 }]
  const ks = shrink.map((_, k) => k + 1)
  const shrinkSeries: XYSeries[] = [{ name: '1 / (1 + λ d_k)', type: 'bar', x: ks, y: shrink, slot: 0 }]

  return (
    <Interactive
      title="The penalty as shrinkage"
      caption={
        <>
          Eighty noisy points and a penalised cubic spline with 23 basis functions. The centre chart shows the m-th
          derivative of the fit, whose squared integral is the penalty. The right chart shows the factor 1/(1 + λd_k) by
          which the fit shrinks each Demmler–Reinsch component, ordered from smoothest to wiggliest. The first m factors
          are always 1: those components span the polynomials of degree below m, which the penalty cannot see. As λ
          grows the fit tends to the dashed least-squares polynomial.
        </>
      }
      controls={
        <>
          <ParamChoice label="penalised derivative" value={order} onChange={setOrder} options={ORDER_OPTIONS} />
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-10}
            max={4}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamButton onClick={() => setSeed((v) => v + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="edf = Σ 1/(1 + λd_k)" value={formatNumber(edf)} />
          <Readout label={`∫ f${PRIME[order]}(x)² dx`} value={formatNumber(penalty)} />
          <Readout label="RSS" value={formatNumber(rss)} />
          <Readout label="zero eigenvalues d_k" value={String(nullDim)} />
        </>
      }
    >
      <div className="grid gap-2 lg:grid-cols-3">
        <XYChart series={fitSeries} xRange={[0, 1]} yRange={[-2, 2.5]} xLabel="x" yLabel="y" height={250} />
        <XYChart series={derivSeries} xRange={[0, 1]} xLabel="x" yLabel={`f${PRIME[order]}(x)`} height={250} />
        <XYChart
          series={shrinkSeries}
          xRange={[0.5, ks.length + 0.5]}
          yRange={[0, 1]}
          xLabel="k"
          yLabel="shrinkage"
          height={250}
        />
      </div>
    </Interactive>
  )
}
