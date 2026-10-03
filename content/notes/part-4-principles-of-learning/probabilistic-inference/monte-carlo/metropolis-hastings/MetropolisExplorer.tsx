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
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { effectiveSampleSize, splitRhat } from '../../_shared/mcmc'

type Target = 'banana' | 'bimodal'
type Starts = 'same' | 'spread'

const GX = linspace(-5, 5, 81)
const GY = linspace(-3, 6, 73)
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
  const g = rng(seed)
  let [x, y] = start
  let cur = lp(x, y)
  const xs: number[] = []
  const ys: number[] = []
  let accepted = 0
  for (let t = 0; t < n; t++) {
    const px = x + scale * g.normal()
    const py = y + scale * g.normal()
    const prop = lp(px, py)
    // Symmetric proposal, so the Hastings ratio is the ratio of target densities.
    if (Math.log(Math.max(g.uniform(), 1e-300)) < prop - cur) {
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
  const g = rng(97)
  return Array.from({ length: n }, () => [-4.5 + 9 * g.uniform(), -2.5 + 8 * g.uniform()])
}

export function MetropolisExplorer() {
  const [target, setTarget] = useState<Target>('banana')
  const [starts, setStarts] = useState<Starts>('same')
  const chains = useParam(4, { min: 1, max: 20, step: 1 })
  const scale = useParam(1, { min: 0.05, max: 5, step: 0.05 })
  const iterations = useParam(3000, { min: 500, max: 6000, step: 500 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const sx = useParam(3.5, { min: -5, max: 5, step: 0.05 })
  const sy = useParam(-2, { min: -3, max: 6, step: 0.05 })

  const z = useMemo(() => {
    const lp = logDensity[target]
    const raw = GY.map((y) => GX.map((x) => lp(x, y)))
    const top = Math.max(...raw.flat())
    return raw.map((row) => row.map((v) => Math.exp(v - top)))
  }, [target])

  // Each chain has its own seed; with spread starts each also has its own starting point.
  const runs = useMemo(() => {
    const origins: [number, number][] =
      starts === 'same' ? Array.from({ length: chains.value }, () => [sx.value, sy.value]) : spreadStarts(chains.value)
    return origins.map((o, k) => ({
      start: o,
      ...runChain(target, o, scale.value, iterations.value, seed.value * 1000 + k),
    }))
  }, [target, starts, chains.value, sx.value, sy.value, scale.value, iterations.value, seed.value])

  const stats = useMemo(() => {
    // Diagnostics on the second half of every chain, after a burn-in of half the run.
    const half = Math.floor(iterations.value / 2)
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
  }, [runs, iterations.value])

  const many = runs.length > 1
  const overlay: HeatmapOverlay[] = useMemo(() => {
    const perChain = Math.max(1, Math.floor(SHOWN / runs.length))
    const stride = Math.max(1, Math.ceil(iterations.value / perChain))
    const px: number[] = []
    const py: number[] = []
    for (const r of runs)
      for (let i = 0; i < r.xs.length; i += stride) {
        px.push(r.xs[i])
        py.push(r.ys[i])
      }
    return [
      { name: 'samples', type: 'scatter', x: px, y: py, slot: 1 },
      ...runs.map((r): HeatmapOverlay => ({
        name: many ? `first ${PATH} steps of each chain` : `first ${PATH} steps`,
        type: 'line',
        x: [r.start[0], ...r.xs.slice(0, PATH)],
        y: [r.start[1], ...r.ys.slice(0, PATH)],
        slot: 3,
        thin: many,
      })),
    ]
  }, [runs, iterations.value, many])

  const trace: XYSeries[] = useMemo(() => {
    const stride = Math.max(1, Math.ceil(iterations.value / TRACE_POINTS))
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
  }, [runs, iterations.value, many])

  const handles: Handle[] | undefined =
    starts === 'spread'
      ? undefined
      : [
          {
            kind: 'point',
            at: [sx.value, sy.value],
            label: 'start',
            onDrag: ([x, y]) => {
              sx.set(x)
              sy.set(y)
            },
          },
        ]
  const [tx, ty] = TRUE_MEAN[target]

  return (
    <Interactive
      title="Random-walk Metropolis on a two-dimensional target"
      caption={`Each step proposes a Gaussian move of standard deviation σ in both coordinates and accepts it with probability min(1, π(proposal)/π(current)). Several chains run at once, each with its own random stream, drawn as light lines: their first ${PATH} steps on the density, and their traces of x₁ below. Drag the common start point, or spread the starts across the plot. With a small σ almost every move is accepted but the chains crawl; with a large σ most proposals are rejected and the traces are flat for long stretches. Both give a small effective sample size (summed over chains, on the second half of each run). Split R̂ compares the chains: near 1 they agree, above about 1.01 they have not mixed. On the bimodal target with a small σ, chains from one start all stay in one mode and R̂ looks fine; spread starts reveal the second mode and R̂ climbs. True means: (${tx}, ${ty}).`}
      controls={
        <>
          <ParamChoice
            label="target"
            value={target}
            onChange={setTarget}
            options={[
              { value: 'banana', label: 'banana' },
              { value: 'bimodal', label: 'two modes' },
            ]}
          />
          <ParamSlider label="proposal scale σ" param={scale} />
          <ParamSlider label="chains" param={chains} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="starting points"
            value={starts}
            onChange={setStarts}
            options={[
              { value: 'same', label: 'one shared start' },
              { value: 'spread', label: 'spread across the plot' },
            ]}
          />
          <ParamSlider label="iterations" param={iterations} format={(v) => String(v)} withArrows />
          <ParamSlider label="random seed" param={seed} withArrows />
        </>
      }
      readout={
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
        <Heatmap
          x={GX}
          y={GY}
          z={z}
          xLabel="x₁"
          yLabel="x₂"
          overlay={overlay}
          handles={handles}
          valueLabel="density (peak = 1)"
          height={380}
        />
        <XYChart series={trace} xLabel="iteration" yLabel="x₁" yRange={[-5, 5]} height={180} />
      </div>
    </Interactive>
  )
}
