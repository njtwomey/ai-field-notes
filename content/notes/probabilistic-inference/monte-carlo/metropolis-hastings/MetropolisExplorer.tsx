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
import { effectiveSampleSize } from '../../_shared/mcmc'

type Target = 'banana' | 'bimodal'

const GX = linspace(-5, 5, 81)
const GY = linspace(-3, 6, 73)
const SHOWN = 1500
const PATH = 150

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

export function MetropolisExplorer() {
  const [target, setTarget] = useState<Target>('banana')
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

  const chain = useMemo(
    () => runChain(target, [sx.value, sy.value], scale.value, iterations.value, seed.value),
    [target, sx.value, sy.value, scale.value, iterations.value, seed.value],
  )

  const stats = useMemo(() => {
    // Diagnostics on the second half, after a burn-in of half the run.
    const half = Math.floor(chain.xs.length / 2)
    const bx = chain.xs.slice(half)
    const by = chain.ys.slice(half)
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
    return { essX: effectiveSampleSize(bx), essY: effectiveSampleSize(by), mx: mean(bx), my: mean(by), kept: bx.length }
  }, [chain])

  const overlay: HeatmapOverlay[] = useMemo(() => {
    const stride = Math.max(1, Math.ceil(chain.xs.length / SHOWN))
    const idx = chain.xs.map((_, i) => i).filter((i) => i % stride === 0)
    return [
      { name: 'samples', type: 'scatter', x: idx.map((i) => chain.xs[i]), y: idx.map((i) => chain.ys[i]), slot: 1 },
      {
        name: `first ${PATH} steps`,
        type: 'line',
        x: [sx.value, ...chain.xs.slice(0, PATH)],
        y: [sy.value, ...chain.ys.slice(0, PATH)],
        slot: 3,
      },
    ]
  }, [chain, sx.value, sy.value])

  const trace: XYSeries[] = useMemo(
    () => [{ name: 'x₁', type: 'line', x: chain.xs.map((_, i) => i + 1), y: chain.xs, slot: 0 }],
    [chain],
  )

  const handles: Handle[] = [
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
      caption={`Each step proposes a Gaussian move of standard deviation σ in both coordinates and accepts it with probability min(1, π(proposal)/π(current)). Drag the start point on the density. With a small σ almost every move is accepted but the chain crawls, and its trace wanders slowly. With a large σ most proposals land in low density and are rejected, and the trace is flat for long stretches. Both give a small effective sample size (computed on the second half of the run). On the bimodal target a small σ never finds the second mode, and the chain looks converged anyway. True means: (${tx}, ${ty}).`}
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
          <ParamSlider label="iterations" param={iterations} format={(v) => String(v)} withArrows />
          <ParamSlider label="random seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="acceptance rate" value={formatNumber(chain.rate)} />
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
