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

/** Target: an equal-width mixture of three isotropic Gaussians in the plane. */
const MEANS: [number, number][] = [
  [-1.4, -0.8],
  [1.4, -0.8],
  [0, 1.2],
]
const WEIGHTS = [0.35, 0.35, 0.3]
const SD = 0.6
const R = 3.5
const GRID = linspace(-R, R, 71)
const BINS = 28
/** Steps drawn per chain; the histogram uses every step. */
const SHOWN = 120
const BIN_EDGES = linspace(-R, R, BINS + 1)
const BIN_CENTRES = BIN_EDGES.slice(0, -1).map((e, i) => (e + BIN_EDGES[i + 1]) / 2)
const MARGINAL_X = linspace(-R, R, 141)

// Marginal of x₁ and its variance, for the histogram and the readout.
const marginal = (x: number) =>
  MEANS.reduce(
    (s, [m], k) => s + (WEIGHTS[k] * Math.exp(-((x - m) ** 2) / (2 * SD * SD))) / (SD * Math.sqrt(2 * Math.PI)),
    0,
  )
const MEAN1 = MEANS.reduce((s, [m], k) => s + WEIGHTS[k] * m, 0)
const VAR1 = MEANS.reduce((s, [m], k) => s + WEIGHTS[k] * (SD * SD + m * m), 0) - MEAN1 * MEAN1

/** log p(x) up to a constant, and its gradient, by log-sum-exp over the components. */
function logDensity(x: number, y: number): { lp: number; gx: number; gy: number } {
  const logs = MEANS.map(([mx, my], k) => Math.log(WEIGHTS[k]) - ((x - mx) ** 2 + (y - my) ** 2) / (2 * SD * SD))
  const top = Math.max(...logs)
  const r = logs.map((l) => Math.exp(l - top))
  const total = r[0] + r[1] + r[2]
  let gx = 0
  let gy = 0
  MEANS.forEach(([mx, my], k) => {
    gx -= (r[k] / total) * (x - mx)
    gy -= (r[k] / total) * (y - my)
  })
  return { lp: top + Math.log(total), gx: gx / (SD * SD), gy: gy / (SD * SD) }
}

const DENSITY = GRID.map((y) => GRID.map((x) => Math.exp(logDensity(x, y).lp)))

type Method = 'ula' | 'mala'

/**
 * Proposal y = x + h∇log p(x) + s√(2h) ξ. ULA always moves; MALA accepts with the Metropolis–Hastings ratio for the
 * target p and this proposal, whose covariance is 2hs²I.
 */
function simulate(method: Method, h: number, s: number, steps: number, starts: [number, number][], seed: number) {
  const g = rng(seed)
  const sd = s * Math.sqrt(2 * h)
  const paths: { x: number[]; y: number[] }[] = []
  let proposed = 0
  let accepted = 0
  for (const start of starts) {
    let [x, y] = start
    let cur = logDensity(x, y)
    const path = { x: [x], y: [y] }
    for (let t = 0; t < steps; t++) {
      const px = x + h * cur.gx + sd * g.normal()
      const py = y + h * cur.gy + sd * g.normal()
      if (method === 'ula') {
        x = px
        y = py
        cur = logDensity(x, y)
      } else {
        const prop = logDensity(px, py)
        const fwd = (px - x - h * cur.gx) ** 2 + (py - y - h * cur.gy) ** 2
        const bwd = (x - px - h * prop.gx) ** 2 + (y - py - h * prop.gy) ** 2
        const logRatio = sd > 0 ? prop.lp - cur.lp - (bwd - fwd) / (2 * sd * sd) : -Infinity
        proposed++
        if (Math.log(g.uniform()) < logRatio) {
          accepted++
          x = px
          y = py
          cur = prop
        }
      }
      // A chain that leaves the plot by far has diverged; stop it rather than overflow.
      if (!Number.isFinite(x) || Math.abs(x) > 1e3 || Math.abs(y) > 1e3) break
      path.x.push(x)
      path.y.push(y)
    }
    paths.push(path)
  }
  return { paths, acceptance: proposed ? accepted / proposed : null }
}

const START_SPEC = { min: -R, max: R, step: 0.05 }

export function LangevinSampler() {
  const [method, setMethod] = useState<Method>('ula')
  const h = useParam(0.1, { min: 0.01, max: 0.8, step: 0.01 })
  const s = useParam(1, { min: 0, max: 1, step: 0.05 })
  const steps = useParam(800, { min: 10, max: 3000, step: 10 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const starts = [
    [useParam(-3, START_SPEC), useParam(3, START_SPEC)],
    [useParam(3, START_SPEC), useParam(3, START_SPEC)],
    [useParam(-3, START_SPEC), useParam(-3, START_SPEC)],
    [useParam(3, START_SPEC), useParam(-3, START_SPEC)],
  ]
  const startKey = starts.map(([a, b]) => `${a.value},${b.value}`).join(';')

  const run = useMemo(() => {
    const pts = startKey.split(';').map((p) => p.split(',').map(Number) as [number, number])
    return simulate(method, h.value, s.value, steps.value, pts, seed.value)
  }, [method, h.value, s.value, steps.value, seed.value, startKey])

  const { overlay, histogram, sampleVar } = useMemo(() => {
    const overlay: HeatmapOverlay[] = run.paths.map((p, i) => ({
      name: `chain ${i + 1}`,
      type: 'line',
      x: p.x.slice(0, SHOWN + 1),
      y: p.y.slice(0, SHOWN + 1),
      slot: i + 1,
    }))
    // Pool every chain after discarding its first 10 % as burn-in.
    const pooled = run.paths.flatMap((p) => p.x.slice(Math.floor(p.x.length / 10)))
    const counts = new Array<number>(BINS).fill(0)
    const width = BIN_EDGES[1] - BIN_EDGES[0]
    for (const v of pooled) {
      const b = Math.floor((v + R) / width)
      if (b >= 0 && b < BINS) counts[b]++
    }
    const n = pooled.length || 1
    const m = pooled.reduce((a, b) => a + b, 0) / n
    const sampleVar = pooled.reduce((a, b) => a + (b - m) ** 2, 0) / n
    const histogram: XYSeries[] = [
      { name: 'samples of x₁', type: 'bar', x: BIN_CENTRES, y: counts.map((c) => c / (n * width)), slot: 0 },
      { name: 'true marginal p(x₁)', type: 'line', x: MARGINAL_X, y: MARGINAL_X.map(marginal), emphasis: true },
    ]
    return { overlay, histogram, sampleVar }
  }, [run])

  const handles: Handle[] = starts.map(([a, b], i) => ({
    kind: 'point',
    at: [a.value, b.value],
    label: `start ${i + 1}`,
    onDrag: ([x, y]) => {
      a.set(x)
      b.set(y)
    },
  }))

  return (
    <Interactive
      title="Langevin chains on a three-mode density"
      caption="Four chains follow the proposal x′ = x + h∇log p(x) + s√(2h)ξ on a mixture of three Gaussians (shaded). The lines show each chain's first 120 steps; the histogram pools every step after a 10% burn-in. Drag the start points. With noise scale s = 1 and a small step h, ULA's histogram of x₁ matches the true marginal. As h grows ULA overdisperses: its sample variance exceeds the target's. MALA with the same h accepts fewer moves but keeps the correct variance. At s = 0 ULA is gradient ascent on log p and each chain stops at a mode; intermediate s is the reduced-noise heuristic of energy-based training, which samples a sharpened density. MALA with reduced noise still targets p, so it rejects almost every move instead."
      controls={
        <>
          <ParamChoice
            label="sampler"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'ula', label: 'ULA' },
              { value: 'mala', label: 'MALA' },
            ]}
          />
          <ParamSlider label="step size h" param={h} />
          <ParamSlider label="noise scale s" param={s} />
          <ParamSlider label="steps per chain" param={steps} format={(v) => String(v)} withArrows />
          <ParamSlider label="noise seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout
            label="acceptance rate"
            value={run.acceptance === null ? 'none (ULA)' : formatNumber(run.acceptance)}
          />
          <Readout label="sample variance of x₁" value={formatNumber(sampleVar)} />
          <Readout label="target variance of x₁" value={formatNumber(VAR1)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Heatmap
          x={GRID}
          y={GRID}
          z={DENSITY}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="p(x)"
          overlay={overlay}
          handles={handles}
          height={380}
        />
        <XYChart series={histogram} xLabel="x₁" yLabel="density" xRange={[-R, R]} yRange={[0, 0.5]} height={380} />
      </div>
    </Interactive>
  )
}
