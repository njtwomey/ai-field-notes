import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal as drawNormal, stream } from 'aifn/foundation/random'
import { histogramDensity } from '../_shared/ode'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalPdf, normalQuantile } from 'aifn/numerics/special'

type Field = 'shift' | 'stretch' | 'split' | 'heat'
const S0 = 0.5
const D = 0.5
/** Velocity v(x, t) and its x-derivative. `heat` is the probability flow of the heat equation for a Gaussian start. */
const FIELDS: Record<
  Field,
  {
    label: string
    formula: string
    v: (x: number, t: number, c: number) => number
    dv: (x: number, t: number) => number
  }
> = {
  shift: { label: 'shift', formula: 'v = 1', v: () => 1, dv: () => 0 },
  stretch: { label: 'stretch', formula: 'v = x / 2', v: (x) => 0.5 * x, dv: () => 0.5 },
  split: { label: 'split', formula: 'v = x − x³', v: (x) => x - x ** 3, dv: (x) => 1 - 3 * x * x },
  heat: {
    label: 'heat flow',
    formula: 'v = D (x − c)/(σ₀² + 2Dt)',
    v: (x, t, c) => (D * (x - c)) / (S0 * S0 + 2 * D * t),
    dv: (_x, t) => D / (S0 * S0 + 2 * D * t),
  },
}

const H = 0.01
const PER_FRAME = 5
const FRAMES = 41
const T_GRID = toFlat(linspace(0, (FRAMES - 1) * PER_FRAME * H, FRAMES))
const LO = -4
const HI = 4
const X_GRID = toFlat(linspace(LO, HI, 97))
const U = (() => {
  const rs = stream(12)
  const normal = () => drawNormal(rs)
  return Float64Array.from({ length: 1500 }, () => normal())
})()
const Q = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95].map((v: number) => normalQuantile(v))

/**
 * Push a set of starting points and their log-densities through dx/dt = v, d log p/dt = −∂v/∂x with RK4, recording
 * every frame. Returns positions[frame][i] and logDensity[frame][i].
 */
function push(field: Field, c: number, x0: ArrayLike<number>, lp0?: ArrayLike<number>) {
  const { v, dv } = FIELDS[field]
  const n = x0.length
  const x = Float64Array.from(x0)
  const lp = lp0 ? Float64Array.from(lp0) : new Float64Array(n)
  const xs = [Float64Array.from(x)]
  const lps = [Float64Array.from(lp)]
  for (let f = 1; f < FRAMES; f++) {
    for (let k = 0; k < PER_FRAME; k++) {
      const t = ((f - 1) * PER_FRAME + k) * H
      for (let i = 0; i < n; i++) {
        const z = x[i]
        const k1 = v(z, t, c)
        const k2 = v(z + (H / 2) * k1, t + H / 2, c)
        const k3 = v(z + (H / 2) * k2, t + H / 2, c)
        const k4 = v(z + H * k3, t + H, c)
        const l1 = dv(z, t)
        const l2 = dv(z + (H / 2) * k1, t + H / 2)
        const l3 = dv(z + (H / 2) * k2, t + H / 2)
        const l4 = dv(z + H * k3, t + H)
        x[i] = z + (H / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        lp[i] -= (H / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
      }
    }
    xs.push(Float64Array.from(x))
    lps.push(Float64Array.from(lp))
  }
  return { xs, lps }
}

/** Linear interpolation of a density known at increasing points xs onto the grid; zero outside. */
function onGrid(xs: Float64Array, ps: Float64Array, grid: number[]) {
  let j = 0
  return grid.map((g) => {
    if (g < xs[0] || g > xs[xs.length - 1]) return 0
    while (j < xs.length - 2 && xs[j + 1] < g) j++
    const w = (g - xs[j]) / (xs[j + 1] - xs[j] || 1)
    return (1 - w) * ps[j] + w * ps[j + 1]
  })
}

export function DensityTransport() {
  const state = useFigureState({
    field: choice<Field>(
      (Object.keys(FIELDS) as Field[]).map((k) => ({ value: k, label: FIELDS[k].label })),
      'split',
      { label: 'velocity field' },
    ),
    frame: slider(0, FRAMES - 1, 20, { step: 1, label: 'time t', format: (fr) => formatNumber(T_GRID[fr]) }),
    c: float(0.2, { min: -2, max: 2, step: 0.05, label: 'start centre c' }),
  })

  const sim = useMemo(() => {
    // Dense starting grid over ±5 standard deviations, so the moved grid still covers the mass.
    const start = toFlat(linspace(state.c - 5 * S0, state.c + 5 * S0, 401))
    const lp0 = start.map((x) => Math.log(normalPdf((x - state.c) / S0) / S0))
    const curve = push(state.field, state.c, start, lp0)
    const particles = push(
      state.field,
      state.c,
      Float64Array.from(U, (u) => state.c + S0 * u),
    ).xs
    const tracks = push(
      state.field,
      state.c,
      Q.map((q) => state.c + S0 * q),
    ).xs
    const density = curve.xs.map((xs, f) => onGrid(xs, curve.lps[f].map(Math.exp), X_GRID))
    return { curve, particles, tracks, density }
  }, [state.field, state.c])

  const f = state.frame
  const xsF = sim.curve.xs[f]
  const psF = sim.curve.lps[f].map(Math.exp)
  let mass = 0
  for (let i = 1; i < xsF.length; i++) mass += 0.5 * (psF[i] + psF[i - 1]) * (xsF[i] - xsF[i - 1])

  const series = useMemo(() => {
    const hist = histogramDensity(sim.particles[f], LO, HI, 64)
    return [
      { name: 'particles (histogram)', x: hist.x, y: hist.y, muted: true },
      {
        name: 'initial density',
        x: Array.from(sim.curve.xs[0]),
        y: Array.from(sim.curve.lps[0], Math.exp),
        slot: 1,
        dashed: true,
      },
      { name: 'density from characteristics', x: Array.from(xsF), y: Array.from(psF), slot: 0 },
    ] as const
  }, [sim, f, xsF, psF])

  const overlay = useMemo<SeriesSpec[]>(
    () =>
      Q.map((_, k) => ({
        name: 'characteristics',
        type: 'line' as const,
        x: T_GRID.map((_, fr) => sim.tracks[fr][k]),
        y: T_GRID,
        slot: 1,
      })),
    [sim],
  )
  const handles = useMemo<Handle[]>(
    () => [{ kind: 'x', at: state.c, onDrag: (v: number) => state.set('c', v) }],
    [state.bind('c')],
  )

  const xAxis = useAxis({ label: 'x', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'x' })
  const yAxis2 = useAxis({ label: 't' })
  return (
    <Figure
      title="A density carried by a velocity field"
      state={state}
      caption="Particles start from N(c, 0.5²) and each follows dx/dt = v(x, t). Left: their histogram at time t, and the density predicted by the continuity equation, computed by moving each point along its characteristic while its log-density falls at rate ∂v/∂x. Right: the density over x and time, with seven characteristics. Heat flow is the deterministic field whose density matches the heat equation: particles spread as if diffusing, but without noise. Drag the vertical line to move the start."

      readouts={
        <>
          <Readout label="field" value={FIELDS[state.field].formula} />
          <Readout label="total mass" value={formatNumber(mass)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Bars {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Raster x={X_GRID} y={T_GRID} z={sim.density} scale={'sequential'} range={[0, 1.5]} valueLabel={'p'} />
          {seriesLayers(overlay, { live: true })}
        </Plot>
      </div>
    </Figure>
  )
}
