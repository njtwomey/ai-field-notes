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
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { gaussPdf, histogram } from '../_shared/sde'

type Model = 'heat' | 'ou' | 'well'
const DRIFTS: Record<Model, { f: (x: number) => number; label: string; potential: ((x: number) => number) | null }> = {
  heat: { f: () => 0, label: 'none', potential: null },
  ou: { f: (x) => -x, label: 'spring −x', potential: (x) => (x * x) / 2 },
  well: { f: (x) => x - x ** 3, label: 'double well x − x³', potential: (x) => x ** 4 / 4 - (x * x) / 2 },
}

const L = 4
const N = 201
const GRID = Array.from({ length: N }, (_, i) => -L + (2 * L * i) / (N - 1))
const DX = GRID[1] - GRID[0]
const T_MAX = 3
const FRAME = 0.05
const FRAMES = Math.round(T_MAX / FRAME)
const PARTICLES = 3000
const START_SD = 0.15
const DENSITY_RANGE: [number, number] = [0, 1.2]
const TIMES = Array.from({ length: FRAMES + 1 }, (_, k) => k * FRAME)
/** Every second grid point, for the density map under the trajectories. */
const MAP_ROWS = GRID.map((_, i) => i).filter((i) => i % 2 === 0)
const MAP_Y = MAP_ROWS.map((i) => GRID[i])

/**
 * Solve ∂p/∂t = −∂(f p)/∂x + ½σ² ∂²p/∂x² on [−4, 4] with no-flux walls: a conservative finite-volume scheme with
 * central or upwind drift flux, explicit in time, with the step chosen for stability. Returns one density per frame.
 */
function solveFokkerPlanck(f: (x: number) => number, sigma: number, x0: number): Float64Array[] {
  const D = (sigma * sigma) / 2
  const mids = GRID.slice(1).map((x) => x - DX / 2)
  const fm = mids.map(f)
  const fmax = Math.max(...fm.map(Math.abs), 1e-9)
  const central = fm.map((v) => Math.abs(v) * DX < 1.5 * D)
  const dtMax = Math.min(D > 0 ? (0.4 * DX * DX) / (2 * D) : Infinity, (0.5 * DX) / fmax)
  const perFrame = Math.max(1, Math.ceil(FRAME / dtMax))
  const dt = FRAME / perFrame
  let p = Float64Array.from(GRID, (x) => gaussPdf(x, x0, START_SD ** 2))
  const out = [p]
  const J = new Float64Array(N + 1)
  for (let frame = 0; frame < FRAMES; frame++) {
    for (let s = 0; s < perFrame; s++) {
      for (let i = 0; i < N - 1; i++) {
        // Central differences where diffusion dominates the cell (accurate); upwind where drift does (stable).
        const drift = central[i] ? (fm[i] * (p[i] + p[i + 1])) / 2 : fm[i] > 0 ? fm[i] * p[i] : fm[i] * p[i + 1]
        J[i + 1] = drift - (D * (p[i + 1] - p[i])) / DX
      }
      const next = new Float64Array(N)
      for (let i = 0; i < N; i++) next[i] = p[i] - (dt * (J[i + 1] - J[i])) / DX
      p = next
    }
    out.push(p)
  }
  return out
}

/** Euler–Maruyama particles with reflection at ±4, recorded at every frame. */
function simulateParticles(f: (x: number) => number, sigma: number, x0: number, seed: number): Float64Array[] {
  const { normal } = rng(seed)
  const sub = 10
  const h = FRAME / sub
  const sq = sigma * Math.sqrt(h)
  let x = Float64Array.from({ length: PARTICLES }, () => x0 + START_SD * normal())
  const out = [x]
  for (let frame = 0; frame < FRAMES; frame++) {
    const next = Float64Array.from(x)
    for (let i = 0; i < PARTICLES; i++) {
      let xi = next[i]
      for (let s = 0; s < sub; s++) {
        xi += f(xi) * h + sq * normal()
        if (xi > L) xi = 2 * L - xi
        if (xi < -L) xi = -2 * L - xi
      }
      next[i] = xi
    }
    out.push(next)
    x = next
  }
  return out
}

function moments(xs: ArrayLike<number>, weights?: ArrayLike<number>) {
  let s0 = 0
  let s1 = 0
  let s2 = 0
  for (let i = 0; i < xs.length; i++) {
    const w = weights ? weights[i] : 1
    s0 += w
    s1 += w * xs[i]
    s2 += w * xs[i] * xs[i]
  }
  const m = s1 / s0
  return { mean: m, variance: s2 / s0 - m * m }
}

export function ParticlesAndDensity() {
  const [model, setModel] = useState<Model>('ou')
  const sigma = useParam(1, { min: 0.4, max: 1.5, step: 0.05 })
  const x0 = useParam(2.5, { min: -3, max: 3, step: 0.05 })
  const t = useParam(0.5, { min: 0, max: T_MAX, step: FRAME })
  const count = useParam(20, { min: 1, max: 50, step: 1 })

  const { f, potential } = DRIFTS[model]
  const pde = useMemo(() => solveFokkerPlanck(f, sigma.value, x0.value), [f, sigma.value, x0.value])
  const particles = useMemo(() => simulateParticles(f, sigma.value, x0.value, 12), [f, sigma.value, x0.value])

  const frame = Math.round(t.value / FRAME)
  const series = useMemo<XYSeries[]>(() => {
    const hist = histogram(particles[frame], -L, L, 64)
    const out: XYSeries[] = [
      { name: '3,000 particles', type: 'bar', x: hist.x, y: hist.y, slot: 0 },
      { name: 'Fokker–Planck density', type: 'line', x: GRID, y: Array.from(pde[frame]), emphasis: true },
    ]
    if (potential) {
      // Stationary density ∝ exp(−2U/σ²).
      const un = GRID.map((x) => Math.exp((-2 * potential(x)) / sigma.value ** 2))
      const z = un.reduce((a, b) => a + b, 0) * DX
      out.push({ name: 'stationary density', type: 'line', x: GRID, y: un.map((u) => u / z), dashed: true, slot: 1 })
    }
    return out
  }, [particles, pde, frame, potential, sigma.value])

  // The Fokker–Planck density over (t, x), with the first particles' trajectories drawn over it. The particles share one
  // random stream, so raising the count adds trajectories without changing the ones already drawn.
  const densityMap = useMemo(() => MAP_ROWS.map((i) => pde.map((p) => p[i])), [pde])
  const trajectories = useMemo<HeatmapOverlay[]>(
    () =>
      Array.from({ length: count.value }, (_, i) => ({
        name: 'particle trajectories',
        type: 'line',
        x: TIMES,
        y: particles.map((frameStates) => frameStates[i]),
        slot: 2,
        thin: count.value > 1,
      })),
    [particles, count.value],
  )

  const pm = moments(particles[frame])
  const dm = moments(GRID, pde[frame])
  return (
    <Interactive
      title="Particles and the density they follow"
      caption="Bars: a histogram of 3,000 particles simulated from the SDE dX = f(X) dt + σ dW, all released near x₀. Line: the density obtained by solving the Fokker–Planck equation numerically from the same start, with no simulation at all. The two agree at every time. With no drift the density spreads as the heat equation dictates; the spring pulls it to a fixed Gaussian; the double well splits it between two wells, and the dashed stationary density exp(−2U/σ²) is reached slowly when σ is small. Below, the numerical density over time and x, with the trajectories of some of the particles drawn over it as light lines; the paths slider sets how many. Each trajectory is rough, but together they fill the density. Drag the vertical line at x₀ to release the particles elsewhere, or the vertical line in the lower panel to move in time."
      controls={
        <>
          <ParamChoice
            label="drift f(x)"
            value={model}
            onChange={setModel}
            options={(Object.keys(DRIFTS) as Model[]).map((m) => ({ value: m, label: DRIFTS[m].label }))}
          />
          <ParamSlider label="time t" param={t} withArrows />
          <ParamSlider label="noise σ" param={sigma} />
          <ParamSlider label="start x₀" param={x0} />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="particle mean" value={formatNumber(pm.mean)} />
          <Readout label="PDE mean" value={formatNumber(dm.mean)} />
          <Readout label="particle variance" value={formatNumber(pm.variance)} />
          <Readout label="PDE variance" value={formatNumber(dm.variance)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="x"
        yLabel="density p_t(x)"
        series={series}
        xRange={[-L, L]}
        yRange={[0, 1.2]}
        handles={[{ kind: 'x', at: x0.value, label: 'x₀', onDrag: (x) => x0.set(x) }]}
      />
      <Heatmap
        x={TIMES}
        y={MAP_Y}
        z={densityMap}
        xLabel="t"
        yLabel="x"
        range={DENSITY_RANGE}
        overlay={trajectories}
        handles={[{ kind: 'x', at: t.value, label: 't', onDrag: (v) => t.set(v) }]}
        valueLabel="density p_t(x)"
        height={260}
      />
    </Interactive>
  )
}
