import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  KERNEL_OPTIONS,
  addDiagonal,
  cholesky,
  makeKernel,
  posterior,
  samplesFromFactor,
  type KernelName,
} from '../_shared/gp'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'

const GRID = toFlat(linspace(-5, 5, 101))
const X_RANGE: [number, number] = [-5, 5]
const Y_RANGE: [number | undefined, number | undefined] = [-3, 3]
const INITIAL: [number, number][] = [
  [-4, -1.2],
  [-2.5, 0.6],
  [-1, 1.1],
  [0.5, 0.2],
  [2, -0.8],
  [3.2, -0.3],
  [4.2, 0.9],
  [-3.2, -0.2],
]
const MAX_SAMPLES = 30
/**
 * Fixed standard normals, so posterior samples deform smoothly instead of jumping when a parameter moves. Draw k has
 * its own stream, so raising the count adds draws without changing the earlier ones.
 */
const NORMALS = Array.from({ length: MAX_SAMPLES }, (_, k) => {
  const g = stream(11 * 1000 + k)
  return GRID.map(() => normal(g))
})

/** GP regression on draggable training points: posterior mean, a band of ±2 posterior sd of f, and samples. */
export function GpRegression() {
  const [points, setPoints] = useState<[number, number][]>(INITIAL)
  const state = useFigureState({
    name: choice<KernelName>(KERNEL_OPTIONS, 'se', { label: 'kernel' }),
    sf: float(1, { min: 0.2, max: 2, step: 0.05, label: 'signal sd σ_f' }),
    sn: float(0.1, { min: 0.01, max: 1, step: 0.01, label: 'noise sd σ_n' }),
    showSamples: setting(true, 'posterior samples'),
    n: int(5, { min: 1, max: INITIAL.length, label: 'training points N' }),
    ell: float(1, { min: 0.06, max: 10, scale: 'log10', suggestions: [0.1, 0.3, 1, 3, 10], label: 'length-scale ℓ' }),
    count: int(MAX_SAMPLES, {
      min: 1,
      max: MAX_SAMPLES,
      suggestions: [1, 5, 10, MAX_SAMPLES],
      label: 'draws',
      when: (v) => v.showSamples === true,
    }),
  })
  const ell = state.ell
  const data = useMemo(() => points.slice(0, state.n), [points, state.n])

  const r = useMemo(() => {
    const k = makeKernel(state.name, { ell, sf: state.sf, period: 3 })
    const x = data.map((p) => p[0])
    const y = data.map((p) => p[1])
    const post = posterior(k, x, y, state.sn ** 2, GRID, true)
    // The Cholesky factor is computed once per posterior; changing the number of draws reuses it.
    const factor = cholesky(addDiagonal(post.covariance!, 1e-6))
    const sd = post.variance.map(Math.sqrt)
    const meanSd = sd.reduce((s, v) => s + v, 0) / sd.length
    return { post, factor, sd, meanSd }
  }, [data, state.name, ell, state.sf, state.sn])
  const draws = useMemo(
    () => (state.showSamples ? samplesFromFactor(r.post.mean, r.factor, NORMALS.slice(0, state.count)) : []),
    [r, state.count, state.showSamples],
  )
  const many = draws.length > 1

  const series: SeriesSpec[] = [
    ...draws.map((d): SeriesSpec => ({
      name: many ? 'posterior samples' : 'posterior sample',
      type: 'line',
      x: GRID,
      y: d,
      slot: 2,
      thin: many,
    })),
    {
      name: 'mean + 2 sd',
      type: 'line',
      x: GRID,
      y: r.post.mean.map((m, i) => m + 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'mean − 2 sd',
      type: 'line',
      x: GRID,
      y: r.post.mean.map((m, i) => m - 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'posterior mean', type: 'line', x: GRID, y: r.post.mean, slot: 0 },
    { name: 'training data', type: 'scatter', x: data.map((p) => p[0]), y: data.map((p) => p[1]), slot: 1 },
  ]
  const handles: Handle[] = data.map((p, i) => ({
    kind: 'point',
    at: p,
    label: `point ${i + 1}`,
    onDrag: ([x, y]) =>
      setPoints((prev) =>
        prev.map((q, j): [number, number] =>
          j === i ? [Math.min(Math.max(x, -4.9), 4.9), Math.min(Math.max(y, -2.9), 2.9)] : q,
        ),
      ),
  }))

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'f(x)', range: Y_RANGE })
  return (
    <Figure
      title="Gaussian process regression"
      state={state}
      caption="Drag the training points. The solid line is the posterior mean, the dashed lines are two posterior standard deviations of f either side of it, and the light curves are functions drawn from the posterior; the draws field sets how many. Near the data the band narrows to about the noise level; far from it the band returns to the prior's ±2σ_f and the mean returns to zero. A short length-scale lets the function turn quickly and forget the data within a short distance. Matérn 1/2 gives rough, continuous but nowhere-differentiable samples; the periodic kernel (period 3) repeats the data."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setPoints(INITIAL)}>
            Reset points
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="mean posterior sd over the plot" value={formatNumber(r.meanSd)} />
          <Readout label="prior sd σ_f" value={formatNumber(state.sf)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        {seriesLayers(series)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
