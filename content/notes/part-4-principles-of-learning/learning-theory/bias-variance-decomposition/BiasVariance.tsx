import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const MAX_DEGREE = 14
const MAX_N = 60
const MAX_FITS = 50
const GRID = toFlat(linspace(-1, 1, 161))
const truth = (x: number) => Math.exp(-x) * Math.sin(5 * x)

/** Legendre polynomials P₀ … P_d at x, by the three-term recurrence. Well conditioned on [−1, 1]. */
function legendre(x: number, d: number): number[] {
  const p = [1, x]
  for (let k = 1; k < d; k++) p.push(((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1))
  return p.slice(0, d + 1)
}

/** Cholesky factor of a symmetric positive-definite matrix, and a solver using it. */
function choleskySolver(a: number[][]) {
  const m = a.length
  const l = a.map(() => new Array<number>(m).fill(0))
  for (let i = 0; i < m; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / l[j][j]
    }
  }
  return (b: number[]) => {
    const z = new Array<number>(m)
    for (let i = 0; i < m; i++) {
      let s = b[i]
      for (let k = 0; k < i; k++) s -= l[i][k] * z[k]
      z[i] = s / l[i][i]
    }
    const x = new Array<number>(m)
    for (let i = m - 1; i >= 0; i--) {
      let s = z[i]
      for (let k = i + 1; k < m; k++) s -= l[k][i] * x[k]
      x[i] = s / l[i][i]
    }
    return x
  }
}

const dot = (a: number[], b: number[]) => a.reduce((s, ai, i) => s + ai * b[i], 0)

/**
 * Least-squares polynomial fits on a fixed design of n equally spaced inputs. For each degree the bias² and variance
 * are exact: the fit at x is φ(x)ᵀG⁻¹Xᵀy with G = XᵀX, so its mean is φ(x)ᵀG⁻¹Xᵀf and its variance σ²·φ(x)ᵀG⁻¹φ(x).
 */
function analyse(n: number) {
  const xs = toFlat(linspace(-1, 1, n))
  const f = xs.map(truth)
  const degrees = Array.from({ length: MAX_DEGREE + 1 }, (_, d) => d)
  const phiGrid = GRID.map((x) => legendre(x, MAX_DEGREE))
  const phiTrain = xs.map((x) => legendre(x, MAX_DEGREE))
  return degrees.map((d) => {
    const rows = phiTrain.map((p) => p.slice(0, d + 1))
    const gram = Array.from({ length: d + 1 }, (_, i) =>
      Array.from({ length: d + 1 }, (__, j) => rows.reduce((s, r) => s + r[i] * r[j], 0)),
    )
    const solve = choleskySolver(gram)
    const project = (y: number[]) =>
      solve(Array.from({ length: d + 1 }, (_, i) => rows.reduce((s, r, k) => s + r[i] * y[k], 0)))
    const meanCoef = project(f)
    let bias2 = 0
    let spread = 0
    const meanFit = phiGrid.map((p, g) => {
      const q = p.slice(0, d + 1)
      const m = dot(q, meanCoef)
      bias2 += (m - truth(GRID[g])) ** 2
      spread += dot(q, solve(q))
      return m
    })
    return { d, project, meanFit, bias2: bias2 / GRID.length, spread: spread / GRID.length }
  })
}

/** Polynomial fits of increasing degree to many resampled datasets, with the exact bias², variance and their sum. */
export function BiasVariance() {
  const state = useFigureState({
    degree: int(3, { min: 0, max: MAX_DEGREE, step: 1, label: 'polynomial degree', format: (v) => String(v) }),
    n: int(25, { min: 16, max: MAX_N, step: 1, label: 'training points n', format: (v) => String(v) }),
    sigma: float(0.4, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    count: int(20, { min: 1, max: MAX_FITS, step: 1, label: 'training sets', format: (v) => String(v) }),
  })

  // One fixed table of standard normals, so moving σ rescales the same noise rather than drawing new noise. Each
  // training set has its own stream, so raising the count adds fits without changing the earlier ones.
  const noise = useMemo(
    () =>
      Array.from({ length: MAX_FITS }, (_, k) => {
        const g = stream(11 * 1000 + k)
        return Array.from({ length: MAX_N }, () => normal(g))
      }),
    [],
  )
  const fits = useMemo(() => analyse(state.n), [state.n])
  const s2 = state.sigma ** 2
  const curves = useMemo(() => {
    const ds = fits.map((r) => r.d)
    return [
      { name: 'bias²', x: ds, y: fits.map((r) => Math.max(r.bias2, 1e-5)), slot: 0 },
      { name: 'variance', x: ds, y: fits.map((r) => s2 * r.spread), slot: 1 },
      {
        name: 'bias² + variance',
        x: ds,
        y: fits.map((r) => Math.max(r.bias2, 1e-5) + s2 * r.spread),
        slot: 2,
      },
    ] as const
  }, [fits, s2])

  const chosen = fits[state.degree]
  const panel = useMemo(() => {
    const xs = toFlat(linspace(-1, 1, state.n))
    const phi = GRID.map((x) => legendre(x, chosen.d))
    const fx: number[] = []
    const fy: number[] = []
    // All fits go in one series, separated by gaps, so the legend has one entry and the chart one line series.
    for (let r = 0; r < state.count; r++) {
      const y = xs.map((x, i) => truth(x) + state.sigma * noise[r][i])
      const coef = chosen.project(y)
      GRID.forEach((x, g) => {
        fx.push(x)
        fy.push(dot(phi[g], coef))
      })
      fx.push(NaN)
      fy.push(NaN)
    }
    return [
      {
        name: state.count > 1 ? `${state.count} fits to resampled data` : 'one fit to resampled data',
        x: fx,
        y: fy,
        slot: 1,
        thin: state.count > 1,
      },
      { name: 'average fit', x: GRID, y: chosen.meanFit, slot: 0 },
      { name: 'true function', x: GRID, y: GRID.map(truth), emphasis: true, dashed: true },
    ] as const
  }, [chosen, state.n, state.sigma, noise, state.count])

  const bias2 = chosen.bias2
  const variance = s2 * chosen.spread

  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-3, 3] })
  const xAxis2 = useAxis({ label: 'polynomial degree', range: [0, MAX_DEGREE] })
  const yAxis2 = useAxis({ label: 'error', range: [1e-3, 3], log: true })
  return (
    <Figure
      title="Bias and variance of polynomial fits"
      state={state}
      caption="The true function is e^(−x) sin 5x on n equally spaced inputs, with Gaussian noise of standard deviation σ. Left: least-squares fits of the chosen degree to independent noisy datasets (light lines; the training sets slider sets how many), the exact average fit over all datasets (solid) and the truth (dashed). Right: bias² and variance averaged over x, computed exactly, against degree (log scale). Low degrees miss the shape in the same way every time; high degrees follow the noise and differ from dataset to dataset. Drag the degree line on the right or use the slider."

      readouts={
        <>
          <Readout label="bias²" value={formatNumber(bias2)} />
          <Readout label="variance" value={formatNumber(variance)} />
          <Readout label="expected test error (+ σ²)" value={formatNumber(bias2 + variance + s2)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...panel[0]} />
          <Curve {...panel[1]} />
          <Curve {...panel[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
          <Curve {...curves[2]} />
          <Handle {...state.handle('degree', { label: 'degree' })} />
        </Plot>
      </div>
    </Figure>
  )
}
