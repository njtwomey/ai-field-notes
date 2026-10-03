import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { clutterExact, clutterLaplace, clutterVb, CLUTTER_VAR } from '../_shared/clutter'
import { grid, normalPdf, powerEp1d, toMoments, toNat, type Moments } from '../_shared/ep'

const DATA = [-1.5, -1, 4, 4.5]
const W = 0.5
const SWEEPS = 40
const PLOT = grid(-12, 16, 281)
const X_RANGE: [number, number] = [-12, 16]
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
const MEAN_RANGE: [number | undefined, number | undefined] = [-8, 8]
const UPDATES = Array.from({ length: DATA.length * SWEEPS }, (_, k) => (k + 1) / DATA.length)
const logFactor = (i: number, t: number) =>
  Math.log((1 - W) * normalPdf(DATA[i], t, 1) + W * normalPdf(DATA[i], 0, CLUTTER_VAR))
const run = (damping: number, power: number) =>
  powerEp1d(toNat({ mean: 0, variance: 100 }), DATA.length, logFactor, { sweeps: SWEEPS, damping, power })
const means = (steps: ReturnType<typeof run>) => steps.map((s) => (s.q.tau > 0 ? s.q.nu / s.q.tau : NaN))
const density = (m: Moments) => PLOT.map((t) => (m.variance > 0 ? normalPdf(t, m.mean, m.variance) : 0))

// Fixed references: the undamped run, and the exact, Laplace and VB answers, do not depend on the controls.
const PLAIN = means(run(1, 1))
const EXACT = clutterExact(DATA, W)
const EXACT_PLOT = PLOT.map(
  (t) => EXACT.density[Math.round((t - EXACT.thetas[0]) / (EXACT.thetas[1] - EXACT.thetas[0]))],
)
const LAPLACE = clutterLaplace(DATA, W)
const VB = clutterVb(DATA, W)

/**
 * Clutter data in two groups make the posterior bimodal. Plain EP oscillates; damping or power EP settle it. Left: the
 * EP posterior mean after every site update. Right: the final Gaussian against the exact posterior, Laplace and VB.
 */
export function DampingPower() {
  const damping = useParam(0.5, { min: 0.1, max: 1, step: 0.05 })
  const power = useParam(1, { min: 0.1, max: 1, step: 0.05 })

  const r = useMemo(() => {
    const steps = run(damping.value, power.value)
    const last = steps[steps.length - 1]
    const prev = steps[steps.length - 1 - DATA.length]
    const q = toMoments(last.q)
    const moved = Math.abs(q.mean - toMoments(prev.q).mean)
    return { trace: means(steps), q, moved, skipped: steps.filter((s) => !s.ok).length }
  }, [damping.value, power.value])

  const trace: XYSeries[] = [
    { name: 'plain EP', type: 'line', x: UPDATES, y: PLAIN, muted: true },
    { name: 'chosen settings', type: 'line', x: UPDATES, y: r.trace, slot: 0 },
  ]
  const fits: XYSeries[] = [
    { name: 'exact', type: 'line', x: PLOT, y: EXACT_PLOT, emphasis: true },
    { name: 'EP (chosen settings)', type: 'line', x: PLOT, y: density(r.q), slot: 0 },
    { name: 'Laplace', type: 'line', x: PLOT, y: density(LAPLACE), slot: 1, dashed: true },
    { name: 'VB', type: 'line', x: PLOT, y: density(VB), slot: 2, dashed: true },
  ]

  return (
    <Interactive
      title="Oscillation, damping and power EP"
      caption="Clutter problem with w = 0.5 and data −1.5, −1, 4 and 4.5: two groups, so the exact posterior has two modes. Left: EP's posterior mean after each site update, for 40 sweeps. Plain EP (grey) swings between the modes and never settles. Damping mixes each new site with the old one; power EP updates only a fraction of each site. Either makes the sweeps converge. Right: the final Gaussian covers both modes, where Laplace and VB sit on one."
      controls={
        <>
          <ParamSlider label="damping (1 = none)" param={damping} />
          <ParamSlider label="power α (1 = ordinary EP)" param={power} />
        </>
      }
      readout={
        <>
          <Readout label="final EP mean, var" value={`${formatNumber(r.q.mean)}, ${formatNumber(r.q.variance)}`} />
          <Readout label="change over last sweep" value={formatNumber(r.moved)} />
          <Readout label="skipped updates" value={r.skipped} />
          <Readout label="exact mean, var" value={`${formatNumber(EXACT.mean)}, ${formatNumber(EXACT.variance)}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart series={trace} xLabel="sweep" yLabel="EP mean of θ" yRange={MEAN_RANGE} height={280} />
        <XYChart series={fits} xLabel="θ" yLabel="density" xRange={X_RANGE} yRange={Y_RANGE} height={280} />
      </div>
    </Interactive>
  )
}
