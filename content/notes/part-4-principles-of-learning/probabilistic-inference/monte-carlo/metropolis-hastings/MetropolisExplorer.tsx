import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { effectiveSampleSize, splitRhat } from '../../_shared/mcmc'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

type Target = 'banana' | 'bimodal'
type Starts = 'same' | 'spread'

const GX = toFlat(linspace(-5, 5, 81))
const GY = toFlat(linspace(-3, 6, 73))
const SHOWN = 1500
const PATH = 150
/** Trace points drawn per chain; longer chains are thinned for drawing only. */
const TRACE_POINTS = 1000

/** Unnormalised log densities. Banana: x ~ N(0, 4), y | x ~ N(x²/4 − 1, 0.25). Bimodal: two round Gaussians. */
const logDensity: Record<Target, (x: number, y: number) => number> = {
  banana: (x, y) => -(x * x) / 8 - (y - (x * x) / 4 + 1) ** 2 / 0.5,
  bimodal: (x, y) => {
    const a = -((x + 2.5) ** 2 + y ** 2) / (2 * 0.64)
    const b = -((x - 2.5) ** 2 + (y - 3) ** 2) / (2 * 0.64)
    const top = Math.max(a, b)
    return top + Math.log(Math.exp(a - top) + Math.exp(b - top))
  },
}

const TRUE_MEAN: Record<Target, [number, number]> = { banana: [0, 0], bimodal: [0, 1.5] }

function runChain(target: Target, start: [number, number], scale: number, n: number, seed: number) {
  const lp = logDensity[target]
  const g = stream(seed)
  let [x, y] = start
  let cur = lp(x, y)
  const xs: number[] = []
  const ys: number[] = []
  let accepted = 0
  for (let t = 0; t < n; t++) {
    const px = x + scale * normal(g)
    const py = y + scale * normal(g)
    const prop = lp(px, py)
    // Symmetric proposal, so the Hastings ratio is the ratio of target densities.
    if (Math.log(Math.max(uniform(g), 1e-300)) < prop - cur) {
      x = px
      y = py
      cur = prop
      accepted++
    }
    xs.push(x)
    ys.push(y)
  }
  return { xs, ys, rate: accepted / n }
}

/** Spread-out starts: seeded points across the plotted box, the same for every seed of the chains themselves. */
function spreadStarts(n: number): [number, number][] {
  const g = stream(97)
  return Array.from({ length: n }, () => [-4.5 + 9 * uniform(g), -2.5 + 8 * uniform(g)])
}

export function MetropolisExplorer() {
  const state = useFigureState({
    target: choice<Target>(
      [
        { value: 'banana', label: 'banana' },
        { value: 'bimodal', label: 'two modes' },
      ],
      'banana',
      { label: 'target' },
    ),
    scale: float(1, { min: 0.05, max: 5, step: 0.05, label: 'proposal scale σ' }),
    chains: int(4, { min: 1, max: 20, step: 1, label: 'chains', format: (v) => String(v) }),
    starts: choice<Starts>(
      [
        { value: 'same', label: 'one shared start' },
        { value: 'spread', label: 'spread across the plot' },
      ],
      'same',
      { label: 'starting points' },
    ),
    iterations: int(3000, { min: 500, max: 6000, step: 500, label: 'iterations', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
    sx: slider(-5, 5, 3.5, { step: 0.05, onChart: true }),
    sy: slider(-3, 6, -2, { step: 0.05, onChart: true }),
  })

  const z = useMemo(() => {
    const lp = logDensity[state.target]
    const raw = GY.map((y) => GX.map((x) => lp(x, y)))
    const top = Math.max(...raw.flat())
    return raw.map((row) => row.map((v) => Math.exp(v - top)))
  }, [state.target])

  // Each chain has its own seed; with spread starts each also has its own starting point.
  const runs = useMemo(() => {
    const origins: [number, number][] =
      state.starts === 'same'
        ? Array.from({ length: state.chains }, () => [state.sx, state.sy])
        : spreadStarts(state.chains)
    return origins.map((o, k) => ({
      start: o,
      ...runChain(state.target, o, state.scale, state.iterations, state.seed * 1000 + k),
    }))
  }, [state.target, state.starts, state.chains, state.sx, state.sy, state.scale, state.iterations, state.seed])

  const stats = useMemo(() => {
    // Diagnostics on the second half of every chain, after a burn-in of half the run.
    const half = Math.floor(state.iterations / 2)
    const bx = runs.map((r) => r.xs.slice(half))
    const by = runs.map((r) => r.ys.slice(half))
    const pooled = (v: number[][]) => v.flat().reduce((a, b) => a + b, 0) / (v.length * v[0].length)
    const ess = (v: number[][]) => v.reduce((a, c) => a + effectiveSampleSize(c), 0)
    return {
      essX: ess(bx),
      essY: ess(by),
      mx: pooled(bx),
      my: pooled(by),
      kept: bx.length * bx[0].length,
      rhat: Math.max(splitRhat(bx), splitRhat(by)),
      rate: runs.reduce((a, r) => a + r.rate, 0) / runs.length,
    }
  }, [runs, state.iterations])

  const many = runs.length > 1
  const overlay: SeriesSpec[] = useMemo(() => {
    const perChain = Math.max(1, Math.floor(SHOWN / runs.length))
    const stride = Math.max(1, Math.ceil(state.iterations / perChain))
    const px: number[] = []
    const py: number[] = []
    for (const r of runs)
      for (let i = 0; i < r.xs.length; i += stride) {
        px.push(r.xs[i])
        py.push(r.ys[i])
      }
    return [
      { name: 'samples', type: 'scatter', x: px, y: py, slot: 1 },
      ...runs.map((r): SeriesSpec => ({
        name: many ? `first ${PATH} steps of each chain` : `first ${PATH} steps`,
        type: 'line',
        x: [r.start[0], ...r.xs.slice(0, PATH)],
        y: [r.start[1], ...r.ys.slice(0, PATH)],
        slot: 3,
        thin: many,
      })),
    ]
  }, [runs, state.iterations, many])

  const trace: SeriesSpec[] = useMemo(() => {
    const stride = Math.max(1, Math.ceil(state.iterations / TRACE_POINTS))
    return runs.map((r) => {
      const idx = r.xs.map((_, i) => i).filter((i) => i % stride === 0)
      return {
        name: many ? 'x₁ of each chain' : 'x₁',
        type: 'line',
        x: idx.map((i) => i + 1),
        y: idx.map((i) => r.xs[i]),
        slot: 0,
        thin: many,
      }
    })
  }, [runs, state.iterations, many])

  const handles: Handle[] | undefined =
    state.starts === 'spread'
      ? undefined
      : [
          {
            kind: 'point',
            at: [state.sx, state.sy],
            label: 'start',
            onDrag: ([x, y]) => {
              state.set('sx', x)
              state.set('sy', y)
            },
          },
        ]
  const [tx, ty] = TRUE_MEAN[state.target]

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const xAxis2 = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis2 = useAxis({ label: 'x₁', range: [-5, 5] })
  return (
    <Figure
      title="Random-walk Metropolis on a two-dimensional target"
      state={state}
      caption={`Each step proposes a Gaussian move of standard deviation σ in both coordinates and accepts it with probability min(1, π(proposal)/π(current)). Several chains run at once, each with its own random stream, drawn as light lines: their first ${PATH} steps on the density, and their traces of x₁ below. Drag the common start point, or spread the starts across the plot. With a small σ almost every move is accepted but the chains crawl; with a large σ most proposals are rejected and the traces are flat for long stretches. Both give a small effective sample size (summed over chains, on the second half of each run). Split R̂ compares the chains: near 1 they agree, above about 1.01 they have not mixed. On the bimodal target with a small σ, chains from one start all stay in one mode and R̂ looks fine; spread starts reveal the second mode and R̂ climbs. True means: (${tx}, ${ty}).`}

      readouts={
        <>
          <Readout label="acceptance rate" value={formatNumber(stats.rate)} />
          <Readout label="split R̂ (worse coordinate)" value={formatNumber(stats.rhat)} />
          <Readout label="ESS of x₁" value={formatNumber(stats.essX)} />
          <Readout label="ESS of x₂" value={formatNumber(stats.essY)} />
          <Readout label="kept draws" value={stats.kept} />
          <Readout label="mean of x₁" value={formatNumber(stats.mx)} />
          <Readout label="mean of x₂" value={formatNumber(stats.my)} />
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <Plot x={xAxis} y={yAxis} height={380}>
          <Raster x={GX} y={GY} z={z} valueLabel={'density (peak = 1)'} />
          {seriesLayers(overlay, { live: true })}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={180}>
          {seriesLayers(trace)}
        </Plot>
      </div>
    </Figure>
  )
}
