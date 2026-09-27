import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { eigSymmetric, type Matrix } from '../../_shared/linalg'

const P = 40
const RHO = 0.5
/** True covariance Σ_ij = 0.5^|i−j|, an AR(1) process: eigenvalues between about 0.33 and 2.97. */
const SIGMA: Matrix = Array.from({ length: P }, (_, i) => Array.from({ length: P }, (_, j) => RHO ** Math.abs(i - j)))
/** Cholesky factor of an AR(1) covariance in closed form: x_i = ρ x_{i−1} + √(1 − ρ²) ε_i. */
function draw(n: number, seed: number): Matrix {
  const r = rng(seed)
  return Array.from({ length: n }, () => {
    const x = new Array<number>(P)
    x[0] = r.normal()
    for (let i = 1; i < P; i++) x[i] = RHO * x[i - 1] + Math.sqrt(1 - RHO * RHO) * r.normal()
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
const DELTAS = linspace(0, 1, 51)

export function ShrinkageExplorer() {
  const [n, setN] = useState(50)
  const delta = useParam(0.3, { min: 0, max: 1, step: 0.01 })
  const sample = useMemo(() => fit(draw(n, 9)), [n])
  const sampleEig = useMemo(() => eigSymmetric(sample.s).values, [sample])
  // Shrinkage keeps the eigenvectors and maps each eigenvalue λ to (1 − δ)λ + δm.
  const shrunkEig = sampleEig.map((l) => (1 - delta.value) * l + delta.value * sample.m)
  const curve = useMemo(() => DELTAS.map((d) => loss(shrink(sample.s, sample.m, d), SIGMA)), [sample])
  const current = loss(shrink(sample.s, sample.m, delta.value), SIGMA)
  const oracle = DELTAS[curve.indexOf(Math.min(...curve))]
  const handles: Handle[] = [{ kind: 'x', at: delta.value, label: 'δ', onDrag: (x) => delta.set(x) }]
  const cond = (e: number[]) => (e[P - 1] > 1e-9 ? formatNumber(e[0] / e[P - 1]) : '∞')

  return (
    <Interactive
      title="Shrinking the sample covariance toward a multiple of the identity"
      caption="Forty variables with covariance 0.5^|i−j|. Left: the sorted eigenvalues of the true covariance (dashed), the sample covariance and the shrunk estimate. With few observations the sample eigenvalues spread far beyond the true range, and below n = 40 the smallest are zero. Right: the estimation error for every shrinkage intensity δ. Drag the line labelled δ, use the slider, or apply the Ledoit–Wolf estimate, which lands near the minimum without knowing the truth."
      controls={
        <>
          <ParamSlider label="observations n" value={n} onChange={setN} min={10} max={400} step={5} />
          <ParamSlider label="shrinkage δ" param={delta} />
          <ParamButton onClick={() => delta.set(sample.delta)}>Ledoit–Wolf δ</ParamButton>
        </>
      }
      readout={
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
        <XYChart
          height={320}
          xLabel="eigenvalue rank"
          yLabel="eigenvalue"
          yRange={[0, undefined]}
          series={[
            { name: 'true Σ', type: 'line', x: INDEX, y: TRUE_EIG, dashed: true, slot: 2 },
            { name: 'sample S', type: 'line', x: INDEX, y: sampleEig, slot: 0 },
            { name: 'shrunk', type: 'line', x: INDEX, y: shrunkEig, slot: 1 },
          ]}
        />
        <XYChart
          height={320}
          xLabel="shrinkage δ"
          yLabel="‖Σ̂ − Σ‖² / p"
          xRange={[0, 1]}
          yRange={[0, undefined]}
          handles={handles}
          series={[
            { name: 'error', type: 'line', x: DELTAS, y: curve, slot: 1 },
            { name: 'current δ', type: 'scatter', x: [delta.value], y: [current], emphasis: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
