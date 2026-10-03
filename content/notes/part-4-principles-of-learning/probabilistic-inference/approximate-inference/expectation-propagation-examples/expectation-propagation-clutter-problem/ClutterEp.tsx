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
import { clutterEp, clutterExact, clutterFactorAndSite, clutterLaplace, clutterVb } from '../_shared/clutter'
import { grid, normalPdf, toMoments, type Moments } from '../_shared/ep'

const X_RANGE: [number, number] = [-6, 8]
const PLOT = grid(-6, 8, 281)
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
const INITIAL = [-2.5, 0.5, 1.5, 2, 3]
const SWEEPS = 10

/** Density on the plot grid; an improper Gaussian (EP can pass through one) is drawn as zero. */
const density = (m: Moments) => PLOT.map((t) => (m.variance > 0 ? normalPdf(t, m.mean, m.variance) : 0))

/**
 * EP on the clutter problem, one site update at a time, against the exact posterior, Laplace and mean-field VB. The lower
 * chart shows the update in progress: the exact factor, the cavity and the Gaussian site that replaces the factor.
 */
export function ClutterEp() {
  const [points, setPoints] = useState(INITIAL)
  const w = useParam(0.2, { min: 0.05, max: 0.9, step: 0.05 })
  const damping = useParam(1, { min: 0.1, max: 1, step: 0.05 })
  const step = useParam(INITIAL.length, { min: 0, max: INITIAL.length * SWEEPS, step: 1 })

  const r = useMemo(() => {
    const exact = clutterExact(points, w.value)
    // Plot the exact density on the plot grid by interpolating the wide grid, which is 0.04 apart.
    const h = exact.thetas[1] - exact.thetas[0]
    const exactPlot = PLOT.map((t) => exact.density[Math.round((t - exact.thetas[0]) / h)])
    return {
      exact,
      exactPlot,
      steps: clutterEp(points, w.value, SWEEPS, damping.value),
      laplace: clutterLaplace(points, w.value),
      vb: clutterVb(points, w.value),
    }
  }, [points, w.value, damping.value])

  const current = step.value === 0 ? null : r.steps[step.value - 1]
  const q: Moments = current ? toMoments(current.q) : { mean: 0, variance: 100 }
  const upper: XYSeries[] = [
    { name: 'exact', type: 'line', x: PLOT, y: r.exactPlot, emphasis: true },
    { name: 'EP', type: 'line', x: PLOT, y: density(q), slot: 0 },
    ...(Number.isFinite(r.laplace.variance)
      ? [{ name: 'Laplace', type: 'line' as const, x: PLOT, y: density(r.laplace), slot: 1, dashed: true }]
      : []),
    { name: 'VB', type: 'line', x: PLOT, y: density(r.vb), slot: 2, dashed: true },
  ]

  let lower: XYSeries[] = []
  let site = '—'
  if (current?.ok && current.q.tau > 0) {
    const x = points[current.site]
    // The stored site is q_after / cavity; with damping this differs from the undamped projection of the tilted density.
    const next = toMoments(current.q)
    const fs = clutterFactorAndSite(x, w.value, PLOT, current.cavity, next, current.tilted.logZ)
    const top = Math.max(...fs.factor)
    const cav = PLOT.map((t) => normalPdf(t, current.cavity.mean, current.cavity.variance))
    const cavTop = Math.max(...cav)
    const tau = current.sites[current.site].tau
    site = `x = ${formatNumber(x)}, precision ${formatNumber(tau)}${tau < 0 ? ' (negative)' : ''}`
    lower = [
      { name: 'cavity (scaled)', type: 'line', x: PLOT, y: cav.map((c) => (c / cavTop) * top), muted: true },
      { name: 'exact factor', type: 'line', x: PLOT, y: fs.factor, emphasis: true },
      { name: 'site', type: 'line', x: PLOT, y: fs.site.map((s) => Math.min(s, 3 * top)), slot: 0, dashed: true },
    ]
  } else if (current) site = 'skipped: cavity precision ≤ 0'

  const negative = current ? current.sites.filter((s) => s.tau < 0).length : 0
  const handles: Handle[] = points.map((x, i) => ({
    kind: 'point',
    at: [x, 0],
    label: `x${i + 1}`,
    onDrag: ([nx]) =>
      setPoints((ps) => ps.map((p, j) => (j === i ? Math.round(Math.min(7.5, Math.max(-5.5, nx)) * 20) / 20 : p))),
  }))
  const where = current ? `sweep ${current.sweep + 1}, site ${current.site + 1}` : 'prior'

  return (
    <Interactive
      title="EP on the clutter problem, one site at a time"
      caption="Top: the exact posterior of θ (ink), EP after the chosen number of site updates, and the Laplace and variational (VB) Gaussians. The five data points sit on the axis; drag them. Bottom: the update just made, with the exact factor for that point, the cavity (scaled) and the Gaussian site that replaces the factor. An outlier's site has negative precision and curves upward. The first sweep is assumed density filtering."
      controls={
        <>
          <ParamSlider label="site updates" param={step} withArrows format={(v) => String(v)} />
          <ParamSlider label="clutter weight w" param={w} />
          <ParamSlider label="damping (1 = none)" param={damping} />
          <ParamButton onClick={() => setPoints(INITIAL)}>Reset points</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="position" value={where} />
          <Readout label="EP mean, var" value={`${formatNumber(q.mean)}, ${formatNumber(q.variance)}`} />
          <Readout label="exact mean, var" value={`${formatNumber(r.exact.mean)}, ${formatNumber(r.exact.variance)}`} />
          <Readout label="Laplace" value={`${formatNumber(r.laplace.mean)}, ${formatNumber(r.laplace.variance)}`} />
          <Readout label="VB" value={`${formatNumber(r.vb.mean)}, ${formatNumber(r.vb.variance)}`} />
          <Readout label="updated site" value={site} />
          <Readout label="negative sites" value={negative} />
        </>
      }
    >
      <XYChart series={upper} xLabel="θ" yLabel="density" xRange={X_RANGE} yRange={Y_RANGE} handles={handles} />
      {lower.length > 0 && (
        <XYChart series={lower} xLabel="θ" yLabel="factor value" xRange={X_RANGE} yRange={Y_RANGE} height={240} />
      )}
    </Interactive>
  )
}
