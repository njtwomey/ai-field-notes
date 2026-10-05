import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type KernelName = 'brownian' | 'laplace' | 'gauss'
const KERNELS = [
  { value: 'brownian' as const, label: 'min(x, x′)' },
  { value: 'laplace' as const, label: 'exp(−|x − x′| / ℓ)' },
  { value: 'gauss' as const, label: 'exp(−(x − x′)² / 2ℓ²)' },
]

const GRID = toFlat(linspace(0, 1, 201))
const X_RANGE: [number, number] = [0, 1]
const Y_RANGE: [number | undefined, number | undefined] = [-2, 2]
const INITIAL: [number, number][] = [
  [0.2, 0.8],
  [0.45, -0.4],
  [0.6, 0.3],
  [0.85, 1.1],
]

function kernelFor(name: KernelName, ell: number) {
  if (name === 'brownian') return (a: number, b: number) => Math.min(a, b)
  if (name === 'laplace') return (a: number, b: number) => Math.exp(-Math.abs(a - b) / ell)
  return (a: number, b: number) => Math.exp(-((a - b) ** 2) / (2 * ell * ell))
}

/** Solves K α = y by Gaussian elimination with partial pivoting; K is at most 4 × 4. */
function solve(K: number[][], y: number[]): number[] {
  const n = y.length
  const a = K.map((row, i) => [...row, y[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r
    ;[a[c], a[p]] = [a[p], a[c]]
    for (let r = c + 1; r < n; r++) {
      const f = a[r][c] / a[c][c]
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k]
    }
  }
  const x = new Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = a[r][n]
    for (let k = r + 1; k < n; k++) s -= a[r][k] * x[k]
    x[r] = s / a[r][r]
  }
  return x
}

/** The smallest-norm function in an RKHS through the draggable points, f = Σ αᵢ k(xᵢ, ·) with α = K⁻¹y. */
export function MinNormInterpolant() {
  const [points, setPoints] = useState<[number, number][]>(INITIAL)
  const state = useFigureState({
    name: choice<KernelName>(KERNELS, 'brownian', { label: 'kernel' }),
    ell: float(0.2, { min: 0.05, max: 0.6, step: 0.01, label: 'length-scale ℓ' }),
  })

  const r = useMemo(() => {
    const k = kernelFor(state.name, state.ell)
    const x = points.map((p) => p[0])
    const y = points.map((p) => p[1])
    // A tiny ridge keeps K invertible when two points are dragged onto the same x.
    const K = x.map((a, i) => x.map((b, j) => k(a, b) + (i === j ? 1e-9 : 0)))
    const alpha = solve(K, y)
    const f = GRID.map((g) => x.reduce((s, xi, i) => s + alpha[i] * k(xi, g), 0))
    const norm = Math.sqrt(
      Math.max(
        y.reduce((s, v, i) => s + v * alpha[i], 0),
        0,
      ),
    )
    return { f, norm, alpha }
  }, [points, state.name, state.ell])

  const series = [
    { name: 'minimum-norm interpolant', x: GRID, y: r.f, slot: 0 },
    { name: 'points', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 1 },
  ] as const
  const handles: Handle[] = points.map((p, i) => ({
    kind: 'point',
    at: p,
    label: `point ${i + 1}`,
    onDrag: ([x, y]) =>
      setPoints((prev) =>
        prev.map((q, j): [number, number] =>
          j === i ? [Math.min(Math.max(x, 0.01), 1), Math.min(Math.max(y, -1.9), 1.9)] : q,
        ),
      ),
  }))

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'f(x)', range: Y_RANGE })
  return (
    <Figure
      title="The smallest function through the points"
      state={state}
      caption="Drag the points. The curve is the function of smallest RKHS norm that passes through all of them, f = Σ αᵢ k(xᵢ, ·) with α = K⁻¹y, and the readout is its norm √(yᵀK⁻¹y). With k = min(x, x′) the norm is the square root of ∫f′², so the interpolant is piecewise linear, starts at f(0) = 0 and stays flat after the last point. The exponential kernel's norm also charges for f itself, so the curve decays between and beyond the points. The Gaussian kernel gives smooth curves that swing wildly when two nearby points disagree, and the norm grows accordingly."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setPoints(INITIAL)}>
            Reset points
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="‖f‖ in the RKHS" value={formatNumber(r.norm)} />
          <Readout label="coefficients α" value={r.alpha.map((a) => formatNumber(a)).join(', ')} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
