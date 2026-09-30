import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const MAX_DEGREE = 14
const MAX_N = 60
const MAX_FITS = 50
const GRID = linspace(-1, 1, 161)
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
  const xs = linspace(-1, 1, n)
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
  const degree = useParam(3, { min: 0, max: MAX_DEGREE, step: 1 })
  const n = useParam(25, { min: 16, max: MAX_N, step: 1 })
  const sigma = useParam(0.4, { min: 0.05, max: 1, step: 0.05 })
  const count = useParam(20, { min: 1, max: MAX_FITS, step: 1 })

  // One fixed table of standard normals, so moving σ rescales the same noise rather than drawing new noise. Each
  // training set has its own stream, so raising the count adds fits without changing the earlier ones.
  const noise = useMemo(
    () =>
      Array.from({ length: MAX_FITS }, (_, k) => {
        const g = rng(11 * 1000 + k)
        return Array.from({ length: MAX_N }, () => g.normal())
      }),
    [],
  )
  const fits = useMemo(() => analyse(n.value), [n.value])
  const s2 = sigma.value ** 2
  const curves = useMemo((): XYSeries[] => {
    const ds = fits.map((r) => r.d)
    return [
      { name: 'bias²', type: 'line', x: ds, y: fits.map((r) => Math.max(r.bias2, 1e-5)), slot: 0 },
      { name: 'variance', type: 'line', x: ds, y: fits.map((r) => s2 * r.spread), slot: 1 },
      {
        name: 'bias² + variance',
        type: 'line',
        x: ds,
        y: fits.map((r) => Math.max(r.bias2, 1e-5) + s2 * r.spread),
        slot: 2,
      },
    ]
  }, [fits, s2])

  const chosen = fits[degree.value]
  const panel = useMemo((): XYSeries[] => {
    const xs = linspace(-1, 1, n.value)
    const phi = GRID.map((x) => legendre(x, chosen.d))
    const fx: number[] = []
    const fy: number[] = []
    // All fits go in one series, separated by gaps, so the legend has one entry and the chart one line series.
    for (let r = 0; r < count.value; r++) {
      const y = xs.map((x, i) => truth(x) + sigma.value * noise[r][i])
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
        name: count.value > 1 ? `${count.value} fits to resampled data` : 'one fit to resampled data',
        type: 'line',
        x: fx,
        y: fy,
        slot: 1,
        thin: count.value > 1,
      },
      { name: 'average fit', type: 'line', x: GRID, y: chosen.meanFit, slot: 0 },
      { name: 'true function', type: 'line', x: GRID, y: GRID.map(truth), emphasis: true, dashed: true },
    ]
  }, [chosen, n.value, sigma.value, noise, count.value])

  const handles: Handle[] = [{ kind: 'x', at: degree.value, label: 'degree', onDrag: (x) => degree.set(x) }]
  const bias2 = chosen.bias2
  const variance = s2 * chosen.spread

  return (
    <Interactive
      title="Bias and variance of polynomial fits"
      caption="The true function is e^(−x) sin 5x on n equally spaced inputs, with Gaussian noise of standard deviation σ. Left: least-squares fits of the chosen degree to independent noisy datasets (light lines; the training sets slider sets how many), the exact average fit over all datasets (solid) and the truth (dashed). Right: bias² and variance averaged over x, computed exactly, against degree (log scale). Low degrees miss the shape in the same way every time; high degrees follow the noise and differ from dataset to dataset. Drag the degree line on the right or use the slider."
      controls={
        <>
          <ParamSlider label="polynomial degree" param={degree} format={(v) => String(v)} withArrows />
          <ParamSlider label="training points n" param={n} format={(v) => String(v)} />
          <ParamSlider label="noise σ" param={sigma} />
          <ParamSlider label="training sets" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="bias²" value={formatNumber(bias2)} />
          <Readout label="variance" value={formatNumber(variance)} />
          <Readout label="expected test error (+ σ²)" value={formatNumber(bias2 + variance + s2)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={panel} xLabel="x" yLabel="y" xRange={[-1, 1]} yRange={[-3, 3]} height={320} />
        <XYChart
          series={curves}
          xLabel="polynomial degree"
          yLabel="error"
          xRange={[0, MAX_DEGREE]}
          yRange={[1e-3, 3]}
          yLog
          handles={handles}
          height={320}
        />
      </div>
    </Interactive>
  )
}
