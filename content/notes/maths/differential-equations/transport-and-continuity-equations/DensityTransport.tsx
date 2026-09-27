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
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { normalPdf, normalQuantile } from '@/lib/math/special'
import { histogramDensity } from '../_shared/ode'

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
const T_GRID = linspace(0, (FRAMES - 1) * PER_FRAME * H, FRAMES)
const LO = -4
const HI = 4
const X_GRID = linspace(LO, HI, 97)
const U = (() => {
  const { normal } = rng(12)
  return Float64Array.from({ length: 1500 }, () => normal())
})()
const Q = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95].map(normalQuantile)

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
  const [field, setField] = useState<Field>('split')
  const frame = useParam(20, { min: 0, max: FRAMES - 1, step: 1 })
  const c = useParam(0.2, { min: -2, max: 2, step: 0.05 })

  const sim = useMemo(() => {
    // Dense starting grid over ±5 standard deviations, so the moved grid still covers the mass.
    const start = linspace(c.value - 5 * S0, c.value + 5 * S0, 401)
    const lp0 = start.map((x) => Math.log(normalPdf((x - c.value) / S0) / S0))
    const curve = push(field, c.value, start, lp0)
    const particles = push(
      field,
      c.value,
      Float64Array.from(U, (u) => c.value + S0 * u),
    ).xs
    const tracks = push(
      field,
      c.value,
      Q.map((q) => c.value + S0 * q),
    ).xs
    const density = curve.xs.map((xs, f) => onGrid(xs, curve.lps[f].map(Math.exp), X_GRID))
    return { curve, particles, tracks, density }
  }, [field, c.value])

  const f = frame.value
  const xsF = sim.curve.xs[f]
  const psF = sim.curve.lps[f].map(Math.exp)
  let mass = 0
  for (let i = 1; i < xsF.length; i++) mass += 0.5 * (psF[i] + psF[i - 1]) * (xsF[i] - xsF[i - 1])

  const series = useMemo<XYSeries[]>(() => {
    const hist = histogramDensity(sim.particles[f], LO, HI, 64)
    return [
      { name: 'particles (histogram)', type: 'bar', x: hist.x, y: hist.y, muted: true },
      {
        name: 'initial density',
        type: 'line',
        x: Array.from(sim.curve.xs[0]),
        y: Array.from(sim.curve.lps[0], Math.exp),
        slot: 1,
        dashed: true,
      },
      { name: 'density from characteristics', type: 'line', x: Array.from(xsF), y: Array.from(psF), slot: 0 },
    ]
  }, [sim, f, xsF, psF])

  const overlay = useMemo<HeatmapOverlay[]>(
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
  const handles = useMemo<Handle[]>(() => [{ kind: 'x', at: c.value, onDrag: c.set }], [c])

  return (
    <Interactive
      title="A density carried by a velocity field"
      caption="Particles start from N(c, 0.5²) and each follows dx/dt = v(x, t). Left: their histogram at time t, and the density predicted by the continuity equation, computed by moving each point along its characteristic while its log-density falls at rate ∂v/∂x. Right: the density over x and time, with seven characteristics. Heat flow is the deterministic field whose density matches the heat equation: particles spread as if diffusing, but without noise. Drag the vertical line to move the start."
      controls={
        <>
          <ParamChoice
            label="velocity field"
            value={field}
            onChange={setField}
            options={(Object.keys(FIELDS) as Field[]).map((k) => ({ value: k, label: FIELDS[k].label }))}
          />
          <ParamSlider label="time t" param={frame} format={(fr) => formatNumber(T_GRID[fr])} withArrows />
          <ParamSlider label="start centre c" param={c} />
        </>
      }
      readout={
        <>
          <Readout label="field" value={FIELDS[field].formula} />
          <Readout label="total mass" value={formatNumber(mass)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="x"
          yLabel="density"
          series={series}
          handles={handles}
          xRange={[LO, HI]}
          yRange={[0, undefined]}
        />
        <Heatmap
          x={X_GRID}
          y={T_GRID}
          z={sim.density}
          xLabel="x"
          yLabel="t"
          valueLabel="p"
          scale="sequential"
          range={[0, 1.5]}
          overlay={overlay}
          height={320}
        />
      </div>
    </Interactive>
  )
}
