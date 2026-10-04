import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bsplineBasis, solve } from '../_shared/splines'

const K = 3
const GRID = Array.from({ length: 201 }, (_, i) => -1 + i / 100)

/** Uniform knots on [-1, 1] with G intervals, extended by K knots on each side, as in a KAN layer. */
function knots(G: number): number[] {
  const h = 2 / G
  return Array.from({ length: G + 2 * K + 1 }, (_, i) => -1 + (i - K) * h)
}

/** Greville abscissae: the knot averages at which each coefficient acts as a control point of the graph. */
function greville(t: number[]): number[] {
  return Array.from({ length: t.length - K - 1 }, (_, i) => t.slice(i + 1, i + K + 1).reduce((a, b) => a + b, 0) / K)
}

const silu = (x: number) => x / (1 + Math.exp(-x))

/** Least-squares coefficients on a new grid for the spline with coefficients c on the old grid: grid extension. */
function refit(c: number[], from: number[], to: number[]): number[] {
  const f = GRID.map((x) => bsplineBasis(x, from, K).reduce((s, b, i) => s + b * c[i], 0))
  const B = GRID.map((x) => bsplineBasis(x, to, K))
  const m = B[0].length
  const A = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => B.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? 1e-10 : 0)),
  )
  const v = Array.from({ length: m }, (_, i) => B.reduce((s, row, r) => s + row[i] * f[r], 0))
  return solve(A, v)
}

/**
 * One learnable KAN activation, phi(x) = w_b silu(x) + sum_i c_i B_i(x), with cubic B-splines on a uniform grid.
 * The coefficients are draggable; changing the grid refits the spline by least squares.
 */
export function SplineActivation() {
  const state = useFigureState({
    grid: int(5, { min: 2, max: 16, step: 1, label: 'Grid intervals G', suggestions: [3, 5, 8, 16] }),
    wb: float(0, { min: 0, max: 1, step: 0.05, label: 'Base weight w_b on SiLU' }),
  })
  const grid = state.grid
  // The dragged coefficients with the grid they belong to; a new grid refits them by least squares.
  const [fit, setFit] = useState(() => ({ grid: 5, coefs: greville(knots(5)).map((x) => 0.8 * Math.sin(3 * x)) }))
  const coefs = useMemo(
    () => (fit.grid === grid ? fit.coefs : refit(fit.coefs, knots(fit.grid), knots(grid))),
    [fit, grid],
  )
  const t = useMemo(() => knots(grid), [grid])
  const xi = useMemo(() => greville(t), [t])

  const r = useMemo(() => {
    const rows = GRID.map((x) => bsplineBasis(x, t, K))
    const spline = rows.map((row) => row.reduce((s, b, i) => s + b * coefs[i], 0))
    const phi = spline.map((v, i) => v + state.wb * silu(GRID[i]))
    const top: SeriesSpec[] = [
      { name: 'coefficients (ξᵢ, cᵢ)', type: 'line', x: xi, y: coefs, muted: true, dashed: true },
      { name: 'φ(x)', type: 'line', x: GRID, y: phi, slot: 0 },
    ]
    if (state.wb > 0) top.push({ name: 'spline part', type: 'line', x: GRID, y: spline, slot: 1, dashed: true })
    const basis: SeriesSpec[] = coefs.map((_, j) => ({
      name: 'B-splines',
      type: 'line',
      x: GRID,
      y: rows.map((row) => row[j]),
      muted: true,
    }))
    return { top, basis }
  }, [t, xi, coefs, state.wb])

  // The outer coefficients sit beyond [-1, 1] on the extended grid, so the axis shows them too.
  const xRange: [number, number] = [xi[0] - 0.1, xi[xi.length - 1] + 0.1]
  const handles: Handle[] = coefs.map((c, i) => ({
    kind: 'point',
    at: [xi[i], c],
    onDrag: ([, y]) => setFit({ grid, coefs: coefs.map((v, j) => (j === i ? Math.min(Math.max(y, -2), 2) : v)) }),
  }))

  const xAxis = useAxis({ label: 'x', range: xRange })
  const yAxis = useAxis({ label: 'φ(x)', range: [-2, 2] })
  const xAxis2 = useAxis({ label: 'x', range: xRange })
  const yAxis2 = useAxis({ label: 'Bᵢ(x)', range: [0, 1] })
  return (
    <Figure
      title="A learnable spline activation"
      state={state}
      caption="Each dark point is one coefficient cᵢ, drawn at its Greville abscissa ξᵢ (the outermost ones lie beyond [−1, 1], where the extended grid lives); drag it up or down to reshape φ. Only the four nearest grid intervals move, because cubic B-splines have local support. Change the grid size G: the spline is refitted to the new grid by least squares, which is how KANs refine a trained activation. Raise w_b to add the SiLU term that KANs keep beside the spline."
      readouts={
        <>
          <Readout label="spline coefficients G + k" value={grid + K} />
          <Readout label="parameters per edge" value={grid + K + 2} />
          <Readout label="φ(0)" value={formatNumber(bsplineBasis(0, t, K).reduce((s, b, i) => s + b * coefs[i], 0))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        {seriesLayers(r.top)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={160}>
        {seriesLayers(r.basis)}
      </Plot>
    </Figure>
  )
}
