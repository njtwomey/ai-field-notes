import { useMemo, useState } from 'react'
import {
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
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { addDiagonal, cholesky, forward, gram, logDet, logMarginal, makeKernel, posterior } from '../_shared/gp'

type Method = 'vfe' | 'fitc' | 'sor'
const METHODS = [
  { value: 'vfe' as const, label: 'VFE (Titsias)' },
  { value: 'fitc' as const, label: 'FITC' },
  { value: 'sor' as const, label: 'SoR' },
]

const N = 100
const NOISE = 0.25
const MAX_M = 20
const GRID = linspace(-6, 6, 151)
const X_RANGE: [number, number] = [-6, 6]
const Y_RANGE: [number | undefined, number | undefined] = [-3, 3]
const truth = (x: number) => Math.sin(1.4 * x) + 0.5 * Math.sin(3.1 * x) * Math.exp(-0.1 * x * x)
const spread = (m: number) => linspace(-4.5, 4.5, m)

const DATA = (() => {
  const g = rng(8)
  const x = Array.from({ length: N }, () => -5 + 10 * g.uniform()).sort((a, b) => a - b)
  return { x, y: x.map((v) => truth(v) + NOISE * g.normal()) }
})()

/**
 * Sparse GP regression through M inducing inputs z. Every method shares the Nyström matrix Q = K_fu K_uu⁻¹ K_uf and
 * differs in the diagonal Λ added to it (σ²I, or σ²I + diag(K − Q) for FITC) and in the test conditional (SoR drops
 * the K** − Q** term).
 */
function sparse(method: Method, z: number[], ell: number) {
  const k = makeKernel('se', { ell, sf: 1 })
  const { x, y } = DATA
  const s2 = NOISE * NOISE
  const Luu = cholesky(addDiagonal(gram(k, z, z), 1e-6))
  // Column i of V is L_uu⁻¹ k_u(x_i), so Q_ij = v_iᵀ v_j.
  const V = x.map((xi) =>
    forward(
      Luu,
      z.map((zj) => k(zj, xi)),
    ),
  )
  const qDiag = V.map((v) => v.reduce((s, u) => s + u * u, 0))
  const gap = qDiag.map((q, i) => Math.max(k(x[i], x[i]) - q, 0))
  const lam = method === 'fitc' ? gap.map((g) => s2 + g) : x.map(() => s2)
  // In the whitened basis Σ⁻¹ becomes I + V Λ⁻¹ Vᵀ.
  const m = z.length
  const A = Array.from({ length: m }, (_, a) =>
    Array.from({ length: m }, (_, b) => V.reduce((s, v, i) => s + (v[a] * v[b]) / lam[i], a === b ? 1 : 0)),
  )
  const La = cholesky(A)
  const bvec = Array.from({ length: m }, (_, a) => V.reduce((s, v, i) => s + (v[a] * y[i]) / lam[i], 0))
  const c = forward(La, bvec)
  // log N(y | 0, Q + Λ) via the matrix determinant lemma and Woodbury identity.
  const quad = y.reduce((s, v, i) => s + (v * v) / lam[i], 0) - c.reduce((s, v) => s + v * v, 0)
  const logdet = logDet(La) + lam.reduce((s, v) => s + Math.log(v), 0)
  const approx = -0.5 * quad - 0.5 * logdet - 0.5 * N * Math.log(2 * Math.PI)
  const trace = gap.reduce((s, v) => s + v, 0)
  const bound = approx - trace / (2 * s2)
  const mean: number[] = []
  const sd: number[] = []
  for (const s of GRID) {
    const w = forward(
      Luu,
      z.map((zj) => k(zj, s)),
    )
    const wa = forward(La, w)
    mean.push(wa.reduce((acc, v, a) => acc + v * c[a], 0))
    const explained = wa.reduce((acc, v) => acc + v * v, 0)
    const dropped = method === 'sor' ? 0 : k(s, s) - w.reduce((acc, v) => acc + v * v, 0)
    sd.push(Math.sqrt(Math.max(dropped + explained, 0)))
  }
  const zMean = z.map((zj) => {
    const w = forward(
      Luu,
      z.map((zi) => k(zi, zj)),
    )
    return forward(La, w).reduce((acc, v, a) => acc + v * c[a], 0)
  })
  return { mean, sd, approx, bound, trace, zMean }
}

/** Inducing points as draggable handles: a sparse GP against the exact GP on 100 points. */
export function SparseGp() {
  const [method, setMethod] = useState<Method>('vfe')
  const [z, setZ] = useState<number[]>(spread(10))
  const logEll = useParam(-0.22, { min: -0.8, max: 0.4, step: 0.01 })
  const ell = 10 ** logEll.value

  const exact = useMemo(() => {
    const k = makeKernel('se', { ell, sf: 1 })
    const post = posterior(k, DATA.x, DATA.y, NOISE * NOISE, GRID)
    return { post, lml: logMarginal(k, DATA.x, DATA.y, NOISE * NOISE).value }
  }, [ell])
  const r = useMemo(() => sparse(method, z, ell), [method, z, ell])

  const exactSd = exact.post.variance.map(Math.sqrt)
  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: DATA.x, y: DATA.y, muted: true },
    { name: 'exact GP mean', type: 'line', x: GRID, y: exact.post.mean, slot: 2, dashed: true },
    {
      name: 'exact GP ± 2 sd',
      type: 'line',
      x: GRID,
      y: exact.post.mean.map((m, i) => m + 2 * exactSd[i]),
      muted: true,
    },
    {
      name: 'exact GP ± 2 sd',
      type: 'line',
      x: GRID,
      y: exact.post.mean.map((m, i) => m - 2 * exactSd[i]),
      muted: true,
    },
    { name: 'sparse ± 2 sd', type: 'line', x: GRID, y: r.mean.map((m, i) => m + 2 * r.sd[i]), slot: 0, dashed: true },
    { name: 'sparse ± 2 sd', type: 'line', x: GRID, y: r.mean.map((m, i) => m - 2 * r.sd[i]), slot: 0, dashed: true },
    { name: 'sparse mean', type: 'line', x: GRID, y: r.mean, slot: 0 },
  ]
  const handles: Handle[] = z.map((zj, j) => ({
    kind: 'point',
    at: [zj, r.zMean[j]],
    label: `inducing input ${j + 1}`,
    onDrag: ([nx]) =>
      setZ((prev) => prev.map((v, i) => (i === j ? Math.min(Math.max(nx, X_RANGE[0]), X_RANGE[1]) : v))),
  }))
  const setCount = (m: number) => setZ(spread(Math.round(m)))

  return (
    <Interactive
      title="Inducing points"
      caption="One hundred noisy points, the exact GP posterior (green dashed mean, grey ±2 sd) and a sparse approximation through M inducing inputs (blue). The black markers are the inducing variables u = f(z); drag them sideways to move the inducing inputs. VFE and FITC keep the full prior variance away from the inducing inputs, so their bands widen there; SoR uses only the M basis functions, so its band collapses between and beyond them. The VFE bound is always below the exact log marginal likelihood, and the gap is the trace term: crowd the inducing inputs into one region and watch it grow. The SoR marginal likelihood, which DTC shares, is not a bound: at the default setting it exceeds the exact value."
      controls={
        <>
          <ParamChoice label="approximation" value={method} onChange={setMethod} options={METHODS} />
          <ParamSlider label="inducing points M" value={z.length} onChange={setCount} min={2} max={MAX_M} step={1} />
          <ParamSlider label="length-scale ℓ" param={logEll} format={(v) => formatNumber(10 ** v)} />
          <ParamButton onClick={() => setZ(spread(z.length))}>Spread evenly</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="exact ln p(y)" value={formatNumber(exact.lml)} />
          <Readout
            label={method === 'vfe' ? 'VFE bound' : `${method === 'fitc' ? 'FITC' : 'SoR'} ln p(y)`}
            value={formatNumber(method === 'vfe' ? r.bound : r.approx)}
          />
          <Readout label="trace term tr(K − Q)/2σ²" value={formatNumber(r.trace / (2 * NOISE * NOISE))} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xLabel="x"
        yLabel="f(x)"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        height={400}
      />
    </Interactive>
  )
}
