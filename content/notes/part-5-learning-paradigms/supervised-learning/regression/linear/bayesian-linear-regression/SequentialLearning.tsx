import { useMemo, useState, type ReactNode } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  useAxis,
  useFigureState,
  type SeriesSpec,
} from 'aifn-render'
import { cholesky, posterior } from './blr'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

/** Prior precision α and noise precision β, as in Bishop's straight-line example. */
const ALPHA = 2
const BETA = 25
const MAX_POINTS = 20
const GRID = toFlat(linspace(-1, 1, 41))
const LINE_X = [-1, 1]
const MAX_SAMPLES = 50

/**
 * Sequential Bayesian learning of a straight line y = w₀ + w₁x. The posterior over (w₀, w₁) after n points is the
 * prior for point n + 1; lines drawn from it close in on the data.
 */
export function SequentialLearning() {
  const state = useFigureState({
    n: int(0, { min: 0, max: MAX_POINTS, step: 1, label: 'points observed', format: (v) => String(v) }),
    count: int(6, { min: 1, max: MAX_SAMPLES, step: 1, label: 'draws', format: (v) => String(v) }),
    seed: int(2, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  // Seed 2 starts with typical points; seed 1 opens with a 3.4σ outlier, a misleading first step.
  const [truth, setTruth] = useState<[number, number]>([-0.3, 0.5])

  // Inputs and unit noise are fixed per seed, so dragging the true line moves the data without reshuffling it.
  const draws = useMemo(() => {
    const g = stream(state.seed)
    return Array.from({ length: MAX_POINTS }, () => ({ x: 2 * uniform(g) - 1, e: normal(g) }))
  }, [state.seed])
  const data = useMemo(
    () => draws.slice(0, state.n).map((d) => ({ x: d.x, y: truth[0] + truth[1] * d.x + d.e / Math.sqrt(BETA) })),
    [draws, state.n, truth],
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
    return { post, z: density(post), prior: density(prior), likelihood, cov, last }
  }, [data])

  // Sample k uses its own stream, so raising the count adds lines without moving the earlier ones.
  const lines = useMemo(() => {
    const l = cholesky(r.cov)
    return Array.from({ length: state.count }, (_, k) => {
      const g = stream((1000 + state.seed) * 1000 + k)
      const [z0, z1] = [normal(g), normal(g)]
      return [r.post.mean[0] + l[0][0] * z0, r.post.mean[1] + l[1][0] * z0 + l[1][1] * z1]
    })
  }, [r, state.seed, state.count])
  const many = lines.length > 1

  const series: SeriesSpec[] = [
    // One shared name, so the legend shows a single entry that toggles every sample.
    ...lines.map((w): SeriesSpec => ({
      name: 'samples from the posterior',
      type: 'line',
      x: LINE_X,
      y: LINE_X.map((x) => w[0] + w[1] * x),
      slot: 0,
      thin: many,
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
  const wAxis = useAxis({ label: 'w₀ (intercept)', range: [-1, 1] })
  const w1Axis = useAxis({ label: 'w₁ (slope)', range: [-1, 1] })
  const sd = [Math.sqrt(r.cov[0][0]), Math.sqrt(r.cov[1][1])]
  // The three weight-space maps share one pair of axes.
  const weightMap = (z: number[][], dragHandles: Handle[] = []) => (
    <Plot x={wAxis} y={w1Axis} height={300}>
      <Raster x={GRID} y={GRID} z={z} range={[0, 1]} valueLabel="relative density" />
      {dragHandles.map((h, i) => (
        <Handle key={i} {...h} />
      ))}
    </Plot>
  )
  const panel = (title: string, body: ReactNode) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      {body}
    </div>
  )

  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-1.5, 1.5] })
  return (
    <Figure
      title="Learning a line one point at a time"
      state={state}
      caption="Bayes' theorem one point at a time, over the intercept w₀ and slope w₁ (dark = probable), with prior N(0, 0.5 I) and noise precision β = 25. Top left: the prior, which is the posterior from the earlier points. Top right: the likelihood of the newest point, a band of lines passing near it. Bottom left: their product, the new posterior. Bottom right: lines drawn from the posterior (light; the draws slider sets how many), with the newest point marked. Step through the points with the arrows; each posterior becomes the next prior. Drag the true weights on the posterior to move the line that generates the data."
      readouts={
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
          state.n === 0 ? 'prior' : `prior: the posterior after ${state.n - 1} point${state.n === 2 ? '' : 's'}`,
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
          `posterior after ${state.n} point${state.n === 1 ? '' : 's'} = prior × likelihood, normalised`,
          weightMap(r.z, handles),
        )}
        {panel(
          'lines drawn from the posterior',
          <Plot x={xAxis} y={yAxis} height={300}>
            {seriesLayers(series)}
          </Plot>,
        )}
      </div>
    </Figure>
  )
}
