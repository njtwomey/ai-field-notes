import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { evaluate, logDet, makeBasis, penalise, smooth } from '../../regression/nonlinear-regression/_shared/splines'

const N = 100
const KNOTS = Array.from({ length: 16 }, (_, i) => (i + 1) / 17)
const LOG_LAMBDAS = linspace(-8, 1, 37)
const GRID = linspace(0, 1, 101)
const truth = (x: number) => Math.sin(2 * Math.PI * x)

/** Rescale a criterion to [0, 1] over the grid so that GCV and REML share one axis. */
const unit = (v: number[]) => {
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  return v.map((x) => (x - lo) / (hi - lo || 1))
}

export function SmoothingSelection() {
  const [noise, setNoise] = useState(0.3)
  const [seed, setSeed] = useState(5)
  const logLambda = useParam(-3, { min: -8, max: 1, step: 0.05 })

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    const eps = Array.from({ length: N }, () => r.normal())
    return { x, y: x.map((xi, i) => truth(xi) + noise * eps[i]) }
  }, [seed, noise])
  const basis = useMemo(() => makeBasis(data.x, KNOTS, 0, 1), [data])
  const p = basis.BtB.length

  // Both criteria over the λ grid. The penalty's null space (straight lines) has dimension 2.
  const curves = useMemo(() => {
    const gcv: number[] = []
    const reml: number[] = []
    for (const lg of LOG_LAMBDAS) {
      const lambda = 10 ** lg
      const s = penalise(basis, lambda)
      const f = smooth(s, data.y)
      const rss = data.y.reduce((acc, y, i) => acc + (y - f.fitted[i]) ** 2, 0)
      const pen =
        lambda * f.coef.reduce((acc, c, j) => acc + c * f.coef.reduce((q, d, k) => q + basis.omega[j][k] * d, 0), 0)
      gcv.push((N * rss) / (N - s.edf) ** 2)
      reml.push((N - 2) * Math.log(rss + pen) + logDet(s.A) - (p - 2) * Math.log(lambda))
    }
    const best = (v: number[]) => LOG_LAMBDAS[v.indexOf(Math.min(...v))]
    return { gcv: unit(gcv), reml: unit(reml), gcvBest: best(gcv), remlBest: best(reml) }
  }, [basis, data, p])

  const smoother = useMemo(() => penalise(basis, 10 ** logLambda.value), [basis, logLambda.value])
  const fit = useMemo(() => smooth(smoother, data.y), [smoother, data])

  const criteria: XYSeries[] = [
    { name: 'GCV', type: 'line', x: LOG_LAMBDAS, y: curves.gcv, slot: 0 },
    { name: 'REML', type: 'line', x: LOG_LAMBDAS, y: curves.reml, slot: 1 },
  ]
  const handles: Handle[] = [{ kind: 'x', at: logLambda.value, label: 'λ', onDrag: logLambda.set }]
  const fitSeries: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: 'true curve', type: 'line', x: GRID, y: GRID.map(truth), slot: 2, dashed: true },
    { name: 'fit at chosen λ', type: 'line', x: GRID, y: evaluate(smoother, fit.coef, GRID), slot: 3 },
  ]

  return (
    <Interactive
      title="Choosing λ by GCV and REML"
      caption={
        <>
          A penalised cubic spline with 20 basis functions fitted to 100 noisy points. The left panel plots both
          criteria against log₁₀ λ, each rescaled to run from 0 at its minimum to 1. Drag the vertical line, or use the
          slider, to fit at any λ. Draw new samples: the GCV choice jumps around more than the REML choice, and now and
          then lands on a much smaller λ that follows the noise.
        </>
      }
      controls={
        <>
          <ParamSlider label="log₁₀ λ" param={logLambda} format={(v) => v.toFixed(2)} />
          <ParamSlider label="noise σ" value={noise} onChange={setNoise} min={0.05} max={1.5} step={0.05} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="effective df" value={formatNumber(smoother.edf)} />
          <Readout label="GCV best log₁₀ λ" value={curves.gcvBest.toFixed(2)} />
          <Readout label="REML best log₁₀ λ" value={curves.remlBest.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        <XYChart
          series={criteria}
          xRange={[-8, 1]}
          yRange={[0, 1]}
          xLabel="log₁₀ λ"
          yLabel="criterion (rescaled)"
          handles={handles}
          height={260}
        />
        <XYChart series={fitSeries} xRange={[0, 1]} yRange={[-3, 3]} xLabel="x" yLabel="y" height={260} />
      </div>
    </Interactive>
  )
}
