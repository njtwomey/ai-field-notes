import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { KERNEL_OPTIONS, makeKernel, posterior, samples, type KernelName } from '../_shared/gp'

const GRID = linspace(-5, 5, 101)
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
const SAMPLES = 3
/** Fixed standard normals, so posterior samples deform smoothly instead of jumping when a parameter moves. */
const NORMALS = (() => {
  const g = rng(11)
  return Array.from({ length: SAMPLES }, () => GRID.map(() => g.normal()))
})()

/** GP regression on draggable training points: posterior mean, a band of ±2 posterior sd of f, and samples. */
export function GpRegression() {
  const [points, setPoints] = useState<[number, number][]>(INITIAL)
  const [name, setName] = useState<KernelName>('se')
  const n = useParam(5, { min: 0, max: INITIAL.length, step: 1 })
  const logEll = useParam(0, { min: -1.2, max: 1, step: 0.02 })
  const sf = useParam(1, { min: 0.2, max: 2, step: 0.05 })
  const sn = useParam(0.1, { min: 0.01, max: 1, step: 0.01 })
  const [showSamples, setShowSamples] = useState(true)
  const ell = 10 ** logEll.value
  const data = points.slice(0, n.value)

  const r = useMemo(() => {
    const k = makeKernel(name, { ell, sf: sf.value, period: 3 })
    const x = data.map((p) => p[0])
    const y = data.map((p) => p[1])
    const post = posterior(k, x, y, sn.value ** 2, GRID, true)
    const draws = samples(post.mean, post.covariance!, NORMALS)
    const sd = post.variance.map(Math.sqrt)
    const meanSd = sd.reduce((s, v) => s + v, 0) / sd.length
    return { post, draws, sd, meanSd }
  }, [data, name, ell, sf.value, sn.value])

  const series: XYSeries[] = [
    ...(showSamples
      ? r.draws.map((d, i): XYSeries => ({
          name: `posterior sample ${i + 1}`,
          type: 'line',
          x: GRID,
          y: d,
          muted: true,
        }))
      : []),
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

  return (
    <Interactive
      title="Gaussian process regression"
      caption="Drag the training points. The solid line is the posterior mean, the dashed lines are two posterior standard deviations of f either side of it, and the grey curves are samples from the posterior. Near the data the band narrows to about the noise level; far from it the band returns to the prior's ±2σ_f and the mean returns to zero. A short length-scale lets the function turn quickly and forget the data within a short distance. Matérn 1/2 gives rough, continuous but nowhere-differentiable samples; the periodic kernel (period 3) repeats the data."
      controls={
        <>
          <ParamChoice label="kernel" value={name} onChange={setName} options={KERNEL_OPTIONS} />
          <ParamSlider label="training points N" param={n} format={(v) => String(v)} />
          <ParamSlider label="length-scale ℓ" param={logEll} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="signal sd σ_f" param={sf} />
          <ParamSlider label="noise sd σ_n" param={sn} />
          <ParamSwitch label="posterior samples" checked={showSamples} onChange={setShowSamples} />
          <ParamButton onClick={() => setPoints(INITIAL)}>Reset points</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="mean posterior sd over the plot" value={formatNumber(r.meanSd)} />
          <Readout label="prior sd σ_f" value={formatNumber(sf.value)} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xLabel="x"
        yLabel="f(x)"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        height={380}
      />
    </Interactive>
  )
}
