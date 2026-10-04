import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  slider,
  useFigureState,
} from 'aifn-render'
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
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

type Order = '1' | '2' | '3'
const ORDER_OPTIONS = [
  { value: '1' as const, label: "f'" },
  { value: '2' as const, label: "f''" },
  { value: '3' as const, label: "f'''" },
]
const PRIME = { '1': '′', '2': '″', '3': '‴' }
const GRID = toFlat(linspace(0, 1, 201))
/** Five equal intervals on [0, 1]: eight cubic B-splines. */
const KNOTS = uniformKnots(5, 0, 1)
const P = basisSize(KNOTS)

/** Greville abscissa of B-spline k: the mean of its interior knots, where it peaks for equally spaced knots. */
const greville = (t: number[], k: number, q = 3) => t.slice(k + 1, k + q + 1).reduce((a, b) => a + b, 0) / q

/** Eight cubic B-splines, their m-th derivatives and the penalty matrix S_kl = ∫ B_k^(m) B_l^(m) dx. */
export function PenaltyMatrix() {
  const state = useFigureState({
    order: choice<Order>(ORDER_OPTIONS, '2', { label: 'penalised derivative' }),
    k: slider(1, P, 4, { step: 1, label: 'basis function k', format: (v) => String(v) }),
  })
  const { order, k } = state
  const m = Number(order)
  const t = KNOTS
  const p = P
  const values = useMemo(() => designMatrix(GRID, t), [t])
  const derivs = useMemo(() => designMatrix(GRID, t, 3, m), [t, m])
  const S = useMemo(() => derivativePenalty(t, 3, m, 0, 1), [t, m])
  const scale = Math.max(...S.flat().map(Math.abs))
  const idx = Array.from({ length: p }, (_, i) => i + 1)

  const lines = (M: number[][]): SeriesSpec[] =>
    idx.map((j) => ({
      name: j === k ? `B${j}` : 'other B-splines',
      type: 'line',
      x: GRID,
      y: M.map((row) => row[j - 1]),
      ...(j === k ? { slot: 0 } : { muted: true }),
    }))
  const row = S[k - 1]
  const rank = p - m
  const rowSum = row.reduce((a, b) => a + b, 0)

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'B_k(x)', range: [0, 0.7] })
  const xAxis2 = useAxis({ label: 'x', range: [0, 1] })
  const yAxis2 = useAxis({ label: `B_k${PRIME[order]}(x)`, hold: 'union' })
  const xAxis3 = useAxis({ label: 'l' })
  const yAxis3 = useAxis({ label: 'k' })
  return (
    <Figure
      title="From basis functions to the penalty matrix"
      caption={
        <>
          Eight cubic B-splines on equally spaced knots over [0, 1] (left), their m-th derivatives (centre) and the
          penalty matrix S with entries ∫ B_k^(m) B_l^(m) dx, scaled by its largest entry (right). Drag the guide on the
          left chart, or use the slider, to pick a basis function. Each row of S sums to zero because a constant has no
          derivative, and S has rank p − m.
        </>
      }
      state={state}
      readouts={
        <>
          <Readout label={`S_kk`} value={formatNumber(row[k - 1])} />
          <Readout label="row sum of S" value={formatNumber(Math.abs(rowSum) < 1e-9 * scale ? 0 : rowSum)} />
          <Readout label="rank of S" value={`${rank} of ${p}`} />
          <Readout label="null space" value={m === 1 ? 'constants' : m === 2 ? 'lines' : 'quadratics'} />
        </>
      }
    >
      <div className="grid gap-2 lg:grid-cols-3">
        <Plot x={xAxis} y={yAxis} height={240}>
          {seriesLayers(lines(values))}
          <Handle
            kind="x"
            at={greville(t, k - 1)}
            label={`B${k}`}
            onDrag={(x) => {
              // Pick the basis function whose peak is nearest the pointer.
              const dist = idx.map((j) => Math.abs(greville(t, j - 1) - x))
              state.set('k', dist.indexOf(Math.min(...dist)) + 1)
            }}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={240}>
          {seriesLayers(lines(derivs))}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={240}>
          <Raster
            x={idx}
            y={idx}
            z={S.map((r) => r.map((v) => v / scale))}
            scale={'diverging'}
            range={[-1, 1]}
            valueLabel={'S_kl / max|S|'}
          />
          <Points x={[k]} y={[k]} emphasis live />
        </Plot>
      </div>
    </Figure>
  )
}

const N = 80
const K = 20
const truth = (x: number) => Math.sin(2 * Math.PI * x) + 0.6 * x

/** A penalised cubic spline in the Demmler–Reinsch basis: the fit, its m-th derivative and the shrinkage factors. */
export function PenaltyShrinkage() {
  const state = useFigureState({
    order: choice<Order>(ORDER_OPTIONS, '2', { label: 'penalised derivative' }),
    logLambda: float(-4, { min: -10, max: 4, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    seed: int(3, { ge: 0, label: 'seed' }),
  })
  const m = Number(state.order)
  const lambda = 10 ** state.logLambda

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    return { x, y: x.map((xi) => truth(xi) + 0.3 * normal(r)) }
  }, [state.seed])
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

  const fitSeries = [
    { name: 'data', x: data.x, y: data.y, muted: true },
    { name: 'λ → ∞ limit', x: GRID, y: times(gridB, limit), slot: 1, dashed: true },
    { name: 'fit', x: GRID, y: times(gridB, beta), slot: 0 },
  ] as const
  const derivSeries = [{ name: `f${PRIME[state.order]}`, x: GRID, y: times(gridD, beta), slot: 0 }] as const
  const ks = shrink.map((_, k) => k + 1)
  const shrinkSeries = [{ name: '1 / (1 + λ d_k)', x: ks, y: shrink, slot: 0 }] as const

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2, 2.5] })
  const xAxis2 = useAxis({ label: 'x', range: [0, 1] })
  const yAxis2 = useAxis({ label: `f${PRIME[state.order]}(x)`, hold: 'union' })
  const xAxis3 = useAxis({ label: 'k', range: [0.5, ks.length + 0.5] })
  const yAxis3 = useAxis({ label: 'shrinkage', range: [0, 1] })
  return (
    <Figure
      title="The penalty as shrinkage"
      state={state}
      caption={
        <>
          Eighty noisy points and a penalised cubic spline with 23 basis functions. The centre chart shows the m-th
          derivative of the fit, whose squared integral is the penalty. The right chart shows the factor 1/(1 + λd_k) by
          which the fit shrinks each Demmler–Reinsch component, ordered from smoothest to wiggliest. The first m factors
          are always 1: those components span the polynomials of degree below m, which the penalty cannot see. As λ
          grows the fit tends to the dashed least-squares polynomial.
        </>
      }

      readouts={
        <>
          <Readout label="edf = Σ 1/(1 + λd_k)" value={formatNumber(edf)} />
          <Readout label={`∫ f${PRIME[state.order]}(x)² dx`} value={formatNumber(penalty)} />
          <Readout label="RSS" value={formatNumber(rss)} />
          <Readout label="zero eigenvalues d_k" value={String(nullDim)} />
        </>
      }
    >
      <div className="grid gap-2 lg:grid-cols-3">
        <Plot x={xAxis} y={yAxis} height={250}>
          <Points {...fitSeries[0]} />
          <Curve {...fitSeries[1]} />
          <Curve {...fitSeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={250}>
          <Curve {...derivSeries[0]} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={250}>
          <Bars {...shrinkSeries[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
