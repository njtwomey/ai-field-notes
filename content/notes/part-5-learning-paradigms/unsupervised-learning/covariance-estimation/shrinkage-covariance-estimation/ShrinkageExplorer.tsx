import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { eigSymmetric, type Matrix } from '../../_shared/linalg'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const P = 40
const RHO = 0.5
/** True covariance Σ_ij = 0.5^|i−j|, an AR(1) process: eigenvalues between about 0.33 and 2.97. */
const SIGMA: Matrix = Array.from({ length: P }, (_, i) => Array.from({ length: P }, (_, j) => RHO ** Math.abs(i - j)))
/** Cholesky factor of an AR(1) covariance in closed form: x_i = ρ x_{i−1} + √(1 − ρ²) ε_i. */
function draw(n: number, seed: number): Matrix {
  const r = stream(seed)
  return Array.from({ length: n }, () => {
    const x = new Array<number>(P)
    x[0] = normal(r)
    for (let i = 1; i < P; i++) x[i] = RHO * x[i - 1] + Math.sqrt(1 - RHO * RHO) * normal(r)
    return x
  })
}

/** Squared Frobenius norm divided by p, the loss Ledoit and Wolf use. */
const loss = (a: Matrix, b: Matrix) =>
  a.reduce((s, row, i) => s + row.reduce((t, v, j) => t + (v - b[i][j]) ** 2, 0), 0) / P

/** Sample covariance (mean known to be zero, divisor n) and the Ledoit–Wolf shrinkage intensity toward mI. */
function fit(x: Matrix) {
  const n = x.length
  const s: Matrix = Array.from({ length: P }, (_, a) =>
    Array.from({ length: P }, (_, b) => x.reduce((t, row) => t + row[a] * row[b], 0) / n),
  )
  const m = s.reduce((t, row, i) => t + row[i], 0) / P
  const target = s.map((row, i) => row.map((_, j) => (i === j ? m : 0)))
  const d2 = loss(s, target)
  // b̄² = (1/n²) Σ_k ‖x_k x_kᵀ − S‖², the estimated sampling variance of S.
  const bBar2 =
    x.reduce(
      (t, row) => t + s.reduce((u, srow, a) => u + srow.reduce((v, sab, b) => v + (row[a] * row[b] - sab) ** 2, 0), 0),
      0,
    ) /
    (P * n * n)
  const b2 = Math.min(bBar2, d2)
  return { s, m, delta: d2 > 0 ? b2 / d2 : 1 }
}

const shrink = (s: Matrix, m: number, delta: number): Matrix =>
  s.map((row, i) => row.map((v, j) => (1 - delta) * v + (i === j ? delta * m : 0)))

const TRUE_EIG = eigSymmetric(SIGMA).values
const INDEX = Array.from({ length: P }, (_, i) => i + 1)
const DELTAS = toFlat(linspace(0, 1, 51))

export function ShrinkageExplorer() {
  const state = useFigureState({
    n: int(50, { min: 10, max: 400, step: 5, label: 'observations n' }),
    delta: float(0.3, { min: 0, max: 1, step: 0.01, label: 'shrinkage δ' }),
  })
  const sample = useMemo(() => fit(draw(state.n, 9)), [state.n])
  const sampleEig = useMemo(() => eigSymmetric(sample.s).values, [sample])
  // Shrinkage keeps the eigenvectors and maps each eigenvalue λ to (1 − δ)λ + δm.
  const shrunkEig = sampleEig.map((l) => (1 - state.delta) * l + state.delta * sample.m)
  const curve = useMemo(() => DELTAS.map((d) => loss(shrink(sample.s, sample.m, d), SIGMA)), [sample])
  const current = loss(shrink(sample.s, sample.m, state.delta), SIGMA)
  const oracle = DELTAS[curve.indexOf(Math.min(...curve))]
  const cond = (e: number[]) => (e[P - 1] > 1e-9 ? formatNumber(e[0] / e[P - 1]) : '∞')

  const xAxis = useAxis({ label: 'eigenvalue rank', hold: 'union' })
  const yAxis = useAxis({ label: 'eigenvalue', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'shrinkage δ', range: [0, 1] })
  const yAxis2 = useAxis({ label: '‖Σ̂ − Σ‖² / p', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Shrinking the sample covariance toward a multiple of the identity"
      state={state}
      caption="Forty variables with covariance 0.5^|i−j|. Left: the sorted eigenvalues of the true covariance (dashed), the sample covariance and the shrunk estimate. With few observations the sample eigenvalues spread far beyond the true range, and below n = 40 the smallest are zero. Right: the estimation error for every shrinkage intensity δ. Drag the line labelled δ, use the slider, or apply the Ledoit–Wolf estimate, which lands near the minimum without knowing the truth."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => state.set('delta', sample.delta)}>
            Ledoit–Wolf δ
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="Ledoit–Wolf δ" value={formatNumber(sample.delta)} />
          <Readout label="best δ for this sample" value={formatNumber(oracle)} />
          <Readout label="error of S" value={formatNumber(curve[0])} />
          <Readout label="error at δ" value={formatNumber(current)} />
          <Readout label="condition number: S, shrunk" value={`${cond(sampleEig)}, ${cond(shrunkEig)}`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve name="true Σ" x={INDEX} y={TRUE_EIG} dashed slot={2} />
          <Curve name="sample S" x={INDEX} y={sampleEig} slot={0} />
          <Curve name="shrunk" x={INDEX} y={shrunkEig} slot={1} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve name="error" x={DELTAS} y={curve} slot={1} />
          <Points name="current δ" x={[state.delta]} y={[current]} emphasis />
          <Handle {...state.handle('delta', { label: 'δ' })} />
        </Plot>
      </div>
    </Figure>
  )
}
