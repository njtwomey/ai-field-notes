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
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

type KernelName = 'rbf' | 'polynomial' | 'linear'
const KERNELS = [
  { value: 'rbf' as const, label: 'Gaussian (RBF)' },
  { value: 'polynomial' as const, label: 'polynomial, degree 3' },
  { value: 'linear' as const, label: 'linear' },
]

const N = 30
const GRID = toFlat(linspace(-3.2, 3.2, 161))

/** Noisy samples of a smooth curve, sorted by x so that the Gram matrix shows locality along its diagonal. */
function makeData(seed: number) {
  const g = stream(seed)
  const x = Array.from({ length: N }, () => -3 + 6 * uniform(g)).sort((a, b) => a - b)
  const y = x.map((v) => Math.sin(1.5 * v) + 0.3 * v + 0.25 * normal(g))
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
  const state = useFigureState({
    name: choice<KernelName>(KERNELS, 'rbf', { label: 'kernel' }),
    width: float(0.6, { min: 0.1, max: 3, step: 0.05, label: 'RBF width ℓ' }),
    logLambda: float(-2, { min: -5, max: 1, step: 0.1, label: 'log₁₀ λ (ridge penalty)' }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'data seed' }),
  })
  const data = useMemo(() => makeData(state.seed), [state.seed])

  const r = useMemo(() => {
    const { x, y } = data
    const lambda = 10 ** state.logLambda
    const K = x.map((a) => x.map((b) => kernel(state.name, a, b, state.width)))
    const alpha = solveSpd(
      K.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0))),
      y,
    )
    const fit = GRID.map((g) => x.reduce((s, xi, i) => s + alpha[i] * kernel(state.name, g, xi, state.width), 0))
    const fitted = x.map((_, i) => K[i].reduce((s, k, j) => s + alpha[j] * k, 0))
    const rmse = Math.sqrt(fitted.reduce((s, f, i) => s + (f - y[i]) ** 2, 0) / N)
    const series = [
      { name: 'training data', x, y, slot: 0 },
      { name: 'f(x) = Σ αᵢ k(x, xᵢ)', x: GRID, y: fit, slot: 1 },
    ] as const
    const peak = Math.max(...K.flat().map(Math.abs))
    return { K, series, rmse, peak, alphaMax: Math.max(...alpha.map(Math.abs)) }
  }, [data, state.name, state.width, state.logLambda])

  const index = Array.from({ length: N }, (_, i) => i + 1)
  const xAxis = useAxis({ label: 'x', range: [-3.2, 3.2] })
  const yAxis = useAxis({ label: 'y', range: [-3, 3] })
  const xAxis2 = useAxis({ label: 'point j (sorted by x)' })
  const yAxis2 = useAxis({ label: 'point i' })
  return (
    <Figure
      title="A linear model in feature space, fitted through the Gram matrix"
      state={state}
      caption="Kernel ridge regression solves (K + λI)α = y for the N × N Gram matrix K and predicts f(x) = Σ αᵢ k(x, xᵢ). The model is linear in the kernel's features but not in x. The right panel shows K with the points sorted by x: the Gaussian kernel makes each point similar only to its neighbours, and the width sets how far that reaches. A narrow width with small λ chases the noise; a wide one smooths it away. The polynomial kernel can only produce cubics, and the linear kernel only straight lines."

      readouts={
        <>
          <Readout label="training RMSE" value={formatNumber(r.rmse)} />
          <Readout label="largest |αᵢ|" value={formatNumber(r.alphaMax)} />
          <Readout
            label="features used"
            value={state.name === 'rbf' ? 'infinitely many' : state.name === 'polynomial' ? '4' : '1'}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Points {...r.series[0]} />
          <Curve {...r.series[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Raster
            x={index}
            y={index}
            z={r.K}
            scale={state.name === 'rbf' ? 'sequential' : 'diverging'}
            range={state.name === 'rbf' ? [0, 1] : [-r.peak, r.peak]}
            valueLabel={'k(xᵢ, xⱼ)'}
          />
        </Plot>
      </div>
    </Figure>
  )
}
