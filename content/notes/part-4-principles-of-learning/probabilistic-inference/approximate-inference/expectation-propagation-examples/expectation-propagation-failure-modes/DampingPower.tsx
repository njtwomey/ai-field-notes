import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
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
  const state = useFigureState({
    damping: float(0.5, { min: 0.1, max: 1, step: 0.05, label: 'damping (1 = none)' }),
    power: float(1, { min: 0.1, max: 1, step: 0.05, label: 'power α (1 = ordinary EP)' }),
  })

  const r = useMemo(() => {
    const steps = run(state.damping, state.power)
    const last = steps[steps.length - 1]
    const prev = steps[steps.length - 1 - DATA.length]
    const q = toMoments(last.q)
    const moved = Math.abs(q.mean - toMoments(prev.q).mean)
    return { trace: means(steps), q, moved, skipped: steps.filter((s) => !s.ok).length }
  }, [state.damping, state.power])

  const trace = [
    { name: 'plain EP', x: UPDATES, y: PLAIN, muted: true },
    { name: 'chosen settings', x: UPDATES, y: r.trace, slot: 0 },
  ] as const
  const fits = [
    { name: 'exact', x: PLOT, y: EXACT_PLOT, emphasis: true },
    { name: 'EP (chosen settings)', x: PLOT, y: density(r.q), slot: 0 },
    { name: 'Laplace', x: PLOT, y: density(LAPLACE), slot: 1, dashed: true },
    { name: 'VB', x: PLOT, y: density(VB), slot: 2, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'sweep', hold: 'union' })
  const yAxis = useAxis({ label: 'EP mean of θ', range: MEAN_RANGE })
  const xAxis2 = useAxis({ label: 'θ', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'density', range: Y_RANGE })
  return (
    <Figure
      title="Oscillation, damping and power EP"
      state={state}
      caption="Clutter problem with w = 0.5 and data −1.5, −1, 4 and 4.5: two groups, so the exact posterior has two modes. Left: EP's posterior mean after each site update, for 40 sweeps. Plain EP (grey) swings between the modes and never settles. Damping mixes each new site with the old one; power EP updates only a fraction of each site. Either makes the sweeps converge. Right: the final Gaussian covers both modes, where Laplace and VB sit on one."

      readouts={
        <>
          <Readout label="final EP mean, var" value={`${formatNumber(r.q.mean)}, ${formatNumber(r.q.variance)}`} />
          <Readout label="change over last sweep" value={formatNumber(r.moved)} />
          <Readout label="skipped updates" value={r.skipped} />
          <Readout label="exact mean, var" value={`${formatNumber(EXACT.mean)}, ${formatNumber(EXACT.variance)}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...trace[0]} />
          <Curve {...trace[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve {...fits[0]} />
          <Curve {...fits[1]} />
          <Curve {...fits[2]} />
          <Curve {...fits[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
