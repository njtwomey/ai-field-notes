import { useMemo, useState, type ReactNode } from 'react'
import {
  Heatmap,
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
import { cholesky, posterior } from './blr'

/** Prior precision α and noise precision β, as in Bishop's straight-line example. */
const ALPHA = 2
const BETA = 25
const MAX_POINTS = 20
const GRID = linspace(-1, 1, 41)
const LINE_X = [-1, 1]
const SAMPLES = 6

/**
 * Sequential Bayesian learning of a straight line y = w₀ + w₁x. The posterior over (w₀, w₁) after n points is the
 * prior for point n + 1; lines drawn from it close in on the data.
 */
export function SequentialLearning() {
  const n = useParam(0, { min: 0, max: MAX_POINTS, step: 1 })
  // Seed 2 starts with typical points; seed 1 opens with a 3.4σ outlier, a misleading first step.
  const [seed, setSeed] = useState(2)
  const [truth, setTruth] = useState<[number, number]>([-0.3, 0.5])

  // Inputs and unit noise are fixed per seed, so dragging the true line moves the data without reshuffling it.
  const draws = useMemo(() => {
    const g = rng(seed)
    return Array.from({ length: MAX_POINTS }, () => ({ x: 2 * g.uniform() - 1, e: g.normal() }))
  }, [seed])
  const data = useMemo(
    () => draws.slice(0, n.value).map((d) => ({ x: d.x, y: truth[0] + truth[1] * d.x + d.e / Math.sqrt(BETA) })),
    [draws, n.value, truth],
  )

  const r = useMemo(() => {
    const fit = (points: typeof data) =>
      posterior(
        points.map((d) => [1, d.x]),
        points.map((d) => d.y),
        ALPHA,
        BETA,
        2,
      )
    // Unnormalised Gaussian density exp(−½ dᵀ A d) on the grid, which peaks at 1 at the mean.
    const density = (post: ReturnType<typeof fit>) => {
      const [[a, b], [, c]] = post.precision
      return GRID.map((w1) =>
        GRID.map((w0) => {
          const d0 = w0 - post.mean[0]
          const d1 = w1 - post.mean[1]
          return Math.exp(-0.5 * (a * d0 * d0 + 2 * b * d0 * d1 + c * d1 * d1))
        }),
      )
    }
    const post = fit(data)
    // The prior for the newest point is the posterior after all the earlier ones.
    const prior = fit(data.slice(0, -1))
    // Likelihood of the newest point as a function of the weights: exp(−β/2 (y − w₀ − w₁x)²), a ridge along the lines
    // through that point. Scaled to peak at 1 like the other panels.
    const last = data.at(-1)
    const likelihood = last
      ? GRID.map((w1) => GRID.map((w0) => Math.exp(-0.5 * BETA * (last.y - w0 - w1 * last.x) ** 2)))
      : undefined
    const [[a, b], [, c]] = post.precision
    // Covariance S = A⁻¹ in closed form for 2 × 2, then samples m + L z with S = L Lᵀ.
    const det = a * c - b * b
    const cov = [
      [c / det, -b / det],
      [-b / det, a / det],
    ]
    const l = cholesky(cov)
    const g = rng(1000 + seed)
    const lines = Array.from({ length: SAMPLES }, () => {
      const [z0, z1] = [g.normal(), g.normal()]
      return [post.mean[0] + l[0][0] * z0, post.mean[1] + l[1][0] * z0 + l[1][1] * z1]
    })
    return { post, z: density(post), prior: density(prior), likelihood, cov, lines, last }
  }, [data, seed])

  const series: XYSeries[] = [
    // One shared name, so the legend shows a single entry that toggles every sample.
    ...r.lines.map((w): XYSeries => ({
      name: 'samples from the posterior',
      type: 'line',
      x: LINE_X,
      y: LINE_X.map((x) => w[0] + w[1] * x),
      muted: true,
    })),
    {
      name: 'true line',
      type: 'line',
      x: LINE_X,
      y: LINE_X.map((x) => truth[0] + truth[1] * x),
      dashed: true,
      slot: 2,
    },
    { name: 'data', type: 'scatter', x: data.map((d) => d.x), y: data.map((d) => d.y), slot: 1 },
    ...(r.last
      ? [{ name: 'newest point', type: 'scatter' as const, x: [r.last.x], y: [r.last.y], emphasis: true }]
      : []),
  ]

  const handles: Handle[] = [
    {
      kind: 'point',
      at: truth,
      label: 'true weights',
      onDrag: ([w0, w1]) => setTruth([Math.max(-1, Math.min(1, w0)), Math.max(-1, Math.min(1, w1))]),
    },
  ]
  const sd = [Math.sqrt(r.cov[0][0]), Math.sqrt(r.cov[1][1])]
  const weightMap = (z: number[][], dragHandles?: Handle[]) => (
    <Heatmap
      x={GRID}
      y={GRID}
      z={z}
      range={[0, 1]}
      xLabel="w₀ (intercept)"
      yLabel="w₁ (slope)"
      valueLabel="relative density"
      handles={dragHandles}
      height={300}
    />
  )
  const panel = (title: string, body: ReactNode) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      {body}
    </div>
  )

  return (
    <Interactive
      title="Learning a line one point at a time"
      caption="Bayes' theorem one point at a time, over the intercept w₀ and slope w₁ (dark = probable), with prior N(0, 0.5 I) and noise precision β = 25. Top left: the prior, which is the posterior from the earlier points. Top right: the likelihood of the newest point, a band of lines passing near it. Bottom left: their product, the new posterior. Bottom right: lines drawn from the posterior, with the newest point marked. Step through the points with the arrows; each posterior becomes the next prior. Drag the true weights on the posterior to move the line that generates the data."
      controls={
        <>
          <ParamSlider label="points observed" param={n} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={20} step={1} />
        </>
      }
      readout={
        <>
          <Readout
            label="posterior mean (w₀, w₁)"
            value={`(${formatNumber(r.post.mean[0])}, ${formatNumber(r.post.mean[1])})`}
          />
          <Readout label="posterior sd" value={`(${formatNumber(sd[0])}, ${formatNumber(sd[1])})`} />
          <Readout label="correlation" value={formatNumber(r.cov[0][1] / (sd[0] * sd[1]))} />
        </>
      }
    >
      <div className="grid gap-x-4 gap-y-6 md:grid-cols-2">
        {panel(
          n.value === 0 ? 'prior' : `prior: the posterior after ${n.value - 1} point${n.value === 2 ? '' : 's'}`,
          weightMap(r.prior),
        )}
        {panel(
          'likelihood of the newest point',
          r.likelihood ? (
            weightMap(r.likelihood)
          ) : (
            <div className="flex h-[300px] items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground">
              No points observed yet
            </div>
          ),
        )}
        {panel(
          `posterior after ${n.value} point${n.value === 1 ? '' : 's'} = prior × likelihood, normalised`,
          weightMap(r.z, handles),
        )}
        {panel(
          'lines drawn from the posterior',
          <XYChart series={series} xLabel="x" yLabel="y" xRange={[-1, 1]} yRange={[-1.5, 1.5]} height={300} />,
        )}
      </div>
    </Interactive>
  )
}
