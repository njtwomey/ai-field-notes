import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

type KernelName = 'rbf' | 'polynomial' | 'linear'
const KERNELS = [
  { value: 'rbf' as const, label: 'Gaussian (RBF)' },
  { value: 'polynomial' as const, label: 'polynomial, degree 3' },
  { value: 'linear' as const, label: 'linear' },
]

const N = 30
const GRID = linspace(-3.2, 3.2, 161)

/** Noisy samples of a smooth curve, sorted by x so that the Gram matrix shows locality along its diagonal. */
function makeData(seed: number) {
  const g = rng(seed)
  const x = Array.from({ length: N }, () => -3 + 6 * g.uniform()).sort((a, b) => a - b)
  const y = x.map((v) => Math.sin(1.5 * v) + 0.3 * v + 0.25 * g.normal())
  return { x, y }
}

function kernel(name: KernelName, a: number, b: number, width: number): number {
  if (name === 'rbf') return Math.exp(-((a - b) ** 2) / (2 * width * width))
  if (name === 'polynomial') return (a * b + 1) ** 3
  return a * b
}

/** Solve Mx = y for symmetric positive definite M by Cholesky; M is small (N × N). */
function solveSpd(M: number[][], y: number[]): number[] {
  const n = y.length
  const L = M.map(() => new Array<number>(n).fill(0))
  for (let j = 0; j < n; j++) {
    let d = M[j][j]
    for (let k = 0; k < j; k++) d -= L[j][k] ** 2
    L[j][j] = Math.sqrt(Math.max(d, 1e-12))
    for (let i = j + 1; i < n; i++) {
      let s = M[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      L[i][j] = s / L[j][j]
    }
  }
  const z = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let s = y[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k]
    z[i] = s / L[i][i]
  }
  const x = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]
    x[i] = s / L[i][i]
  }
  return x
}

/** Kernel ridge regression: a linear model in the kernel's feature space, fitted through the N × N Gram matrix. */
export function KernelRegression() {
  const [name, setName] = useState<KernelName>('rbf')
  const width = useParam(0.6, { min: 0.1, max: 3, step: 0.05 })
  const logLambda = useParam(-2, { min: -5, max: 1, step: 0.1 })
  const seed = useParam(3, { min: 1, max: 20, step: 1 })
  const data = useMemo(() => makeData(seed.value), [seed.value])

  const r = useMemo(() => {
    const { x, y } = data
    const lambda = 10 ** logLambda.value
    const K = x.map((a) => x.map((b) => kernel(name, a, b, width.value)))
    const alpha = solveSpd(
      K.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0))),
      y,
    )
    const fit = GRID.map((g) => x.reduce((s, xi, i) => s + alpha[i] * kernel(name, g, xi, width.value), 0))
    const fitted = x.map((_, i) => K[i].reduce((s, k, j) => s + alpha[j] * k, 0))
    const rmse = Math.sqrt(fitted.reduce((s, f, i) => s + (f - y[i]) ** 2, 0) / N)
    const series: XYSeries[] = [
      { name: 'training data', type: 'scatter', x, y, slot: 0 },
      { name: 'f(x) = Σ αᵢ k(x, xᵢ)', type: 'line', x: GRID, y: fit, slot: 1 },
    ]
    const peak = Math.max(...K.flat().map(Math.abs))
    return { K, series, rmse, peak, alphaMax: Math.max(...alpha.map(Math.abs)) }
  }, [data, name, width.value, logLambda.value])

  const index = Array.from({ length: N }, (_, i) => i + 1)
  return (
    <Interactive
      title="A linear model in feature space, fitted through the Gram matrix"
      caption="Kernel ridge regression solves (K + λI)α = y for the N × N Gram matrix K and predicts f(x) = Σ αᵢ k(x, xᵢ). The model is linear in the kernel's features but not in x. The right panel shows K with the points sorted by x: the Gaussian kernel makes each point similar only to its neighbours, and the width sets how far that reaches. A narrow width with small λ chases the noise; a wide one smooths it away. The polynomial kernel can only produce cubics, and the linear kernel only straight lines."
      controls={
        <>
          <ParamChoice label="kernel" value={name} onChange={setName} options={KERNELS} />
          <ParamSlider label="RBF width ℓ" param={width} />
          <ParamSlider label="log₁₀ λ (ridge penalty)" param={logLambda} />
          <ParamSlider label="data seed" param={seed} />
        </>
      }
      readout={
        <>
          <Readout label="training RMSE" value={formatNumber(r.rmse)} />
          <Readout label="largest |αᵢ|" value={formatNumber(r.alphaMax)} />
          <Readout
            label="features used"
            value={name === 'rbf' ? 'infinitely many' : name === 'polynomial' ? '4' : '1'}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart height={340} xLabel="x" yLabel="y" series={r.series} xRange={[-3.2, 3.2]} yRange={[-3, 3]} />
        <Heatmap
          x={index}
          y={index}
          z={r.K}
          scale={name === 'rbf' ? 'sequential' : 'diverging'}
          range={name === 'rbf' ? [0, 1] : [-r.peak, r.peak]}
          xLabel="point j (sorted by x)"
          yLabel="point i"
          valueLabel="k(xᵢ, xⱼ)"
          height={340}
        />
      </div>
    </Interactive>
  )
}
